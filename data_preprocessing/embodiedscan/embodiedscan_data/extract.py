import json
import logging
import os
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool
from typing import Optional

from tqdm import tqdm

from embodiedscan_data.datasets import get_dataset_config, ALL_DATASETS
from embodiedscan_data.datasets.base import DatasetConfig

logger = logging.getLogger(__name__)

# Global reference for worker processes
_worker_explorer = None
_worker_config = None
_worker_data_root = None


def _init_worker(config_name: str, data_root: str):
    """Initialize Explorer in each worker process."""
    global _worker_explorer, _worker_config, _worker_data_root
    from embodiedscan.explorer import EmbodiedScanExplorer

    _worker_config = get_dataset_config(config_name)
    _worker_data_root = data_root
    explorer_kwargs = _worker_config.get_explorer_kwargs(data_root)
    _worker_explorer = EmbodiedScanExplorer(**explorer_kwargs)


def _process_single(args):
    """Process a single (scene, camera) pair in a worker.

    Returns:
        tuple: (status, payload, scene, camera)
          status: "ok" | "explorer_none" | "intrinsic_err" | "exception"
          payload: info dict when ok, else short reason string
    """
    scene, camera = args
    try:
        save_path = _worker_config.get_save_path(_worker_data_root, scene)
        os.makedirs(save_path, exist_ok=True)
        project_root = os.path.dirname(os.path.abspath(_worker_data_root))
        info = _worker_explorer.get_info(
            scene, camera, render_box=True,
            save_path=save_path,
            root_path=project_root,
        )
        if info is None:
            return ("explorer_none",
                    "EmbodiedScanExplorer.get_info returned None "
                    "(scene or camera not found in annotation, or "
                    "scene filtered out because its dir does not exist)",
                    scene, camera)

        info["dataset"] = _worker_config.name
        info["scene_id"] = _worker_config.get_scene_id(scene)
        info["depth_scale"] = _worker_config.depth_scale

        try:
            info["intrinsic"] = _worker_config.get_intrinsic(
                _worker_data_root, scene, camera)
        except Exception as e:
            return ("intrinsic_err",
                    f"{type(e).__name__}: {e}",
                    scene, camera)

        depth_map = _worker_config.get_depth_map(
            _worker_data_root, scene, camera)
        if depth_map is not None:
            info["depth_map"] = depth_map

        info = _worker_config.post_process(
            info, _worker_data_root, scene, camera)

        for field in ("image", "depth_map", "intrinsic", "pose",
                      "axis_align_matrix"):
            val = info.get(field)
            if val and isinstance(val, str) and os.path.isabs(val):
                info[field] = os.path.relpath(val, _worker_data_root)

        return ("ok", info, scene, camera)
    except Exception:
        return ("exception", traceback.format_exc(), scene, camera)


def _diagnose_explorer(dataset_name: str, data_root: str,
                       collected_scenes, max_print: int = 5):
    """Print a quick diagnostic: how many scenes Explorer actually recognises
    vs. how many list_scenes produced. This surfaces sample_idx/dir mismatches
    (e.g. pkl uses 'arkitscenes/Training/<id>' but disk has 'arkitscenes/<id>').
    """
    try:
        from embodiedscan.explorer import EmbodiedScanExplorer
    except Exception as e:
        logger.warning("Cannot import EmbodiedScanExplorer for diagnosis: %s", e)
        return

    cfg = get_dataset_config(dataset_name)
    try:
        explorer = EmbodiedScanExplorer(
            **cfg.get_explorer_kwargs(data_root))
    except Exception as e:
        logger.warning("Diagnosis: failed to build Explorer: %s", e)
        return

    try:
        explorer_scenes = [s for s in explorer.list_scenes()
                           if s.split('/')[0] == dataset_name]
    except Exception as e:
        logger.warning("Diagnosis: explorer.list_scenes() failed: %s", e)
        return

    print(f"\n[diagnose] dataset={dataset_name}")
    print(f"[diagnose]   list_scenes (from disk) : {len(collected_scenes)}")
    print(f"[diagnose]   explorer scenes (from pkl, dir-verified): "
          f"{len(explorer_scenes)}")
    if collected_scenes:
        print(f"[diagnose]   disk sample     : {collected_scenes[:max_print]}")
    if explorer_scenes:
        print(f"[diagnose]   explorer sample : {explorer_scenes[:max_print]}")

    disk_set = set(collected_scenes)
    exp_set = set(explorer_scenes)
    overlap = disk_set & exp_set
    print(f"[diagnose]   overlap (scene ids present in BOTH): {len(overlap)}")
    if len(overlap) == 0:
        print("[diagnose]   !!! zero overlap -> every task will fail with "
              "'explorer_none'. Most likely the scene-id format from "
              "list_scenes does not match the pkl's sample_idx, or the "
              "dataset dir layout is missing an intermediate folder "
              "(e.g. arkitscenes/Training/).")
    print()


def extract_dataset(
    dataset_name: str,
    data_root: str,
    output_dir: str,
    workers: int = 24,
    max_scenes: Optional[int] = None,
    max_tasks: Optional[int] = None,
) -> str:
    """Extract per-image info for a dataset.

    Args:
        dataset_name: One of "scannet", "3rscan", "matterport3d", "arkitscenes"
        data_root: Root data directory
        output_dir: Output directory for JSONL
        workers: Number of parallel workers
        max_scenes: Limit number of scenes (for smoke testing)
        max_tasks:  Hard-cap on total (scene, camera) tasks (for smoke testing).
                    Applied AFTER task collection; useful when you just want to
                    sanity-check the pipeline end-to-end with a handful of
                    frames without waiting for full-dataset extraction.

    Returns:
        Path to output JSONL file
    """
    config = get_dataset_config(dataset_name)
    output_path = os.path.join(output_dir, f"{config.name}.jsonl")
    failures_path = os.path.join(output_dir, f"{config.name}.failures.log")
    os.makedirs(output_dir, exist_ok=True)

    # Resume: load existing IDs
    existing_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line.strip()).get("id"))
                except (json.JSONDecodeError, AttributeError):
                    pass
        logger.info("Resume: found %d existing records", len(existing_ids))

    # Collect tasks
    logger.info("Collecting scenes for %s...", dataset_name)
    scenes = config.list_scenes(data_root)
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    # Pre-flight diagnosis BEFORE spinning up a 12M-task pool
    _diagnose_explorer(dataset_name, data_root, scenes)

    # Task collection. On networked filesystems (e.g. CephFS) listdir on 2000+
    # scene dirs can take many minutes. When max_tasks is set we early-stop
    # task collection as soon as we hit the budget, so smoke-tests start
    # processing in seconds instead of minutes.
    tasks = []
    collect_pbar = tqdm(scenes, desc=f"Collecting tasks ({dataset_name})")
    for scene in collect_pbar:
        if config.skip_scene(data_root, scene):
            continue
        cameras = config.list_cameras(data_root, scene)
        for camera in cameras:
            if config.skip_camera(data_root, scene, camera):
                continue
            tasks.append((scene, camera))
        if max_tasks is not None and max_tasks > 0 and len(tasks) >= max_tasks:
            collect_pbar.set_description(
                f"Collecting tasks ({dataset_name}) [early-stop]")
            break
    collect_pbar.close()

    logger.info("Found %d tasks across %d scenes", len(tasks), len(scenes))

    if max_tasks is not None and max_tasks > 0 and len(tasks) > max_tasks:
        logger.warning("Smoke-test mode: truncating tasks %d -> %d",
                       len(tasks), max_tasks)
        tasks = tasks[:max_tasks]

    if not tasks:
        logger.warning("No tasks found for %s", dataset_name)
        return output_path

    # Process in parallel
    results = []
    failed_counts = defaultdict(int)
    failed_samples = defaultdict(list)   # status -> list of (scene, cam, reason)
    SAMPLE_CAP = 10
    start = time.time()

    failures_fh = open(failures_path, "w", encoding="utf-8")
    failures_fh.write(f"# failures for {dataset_name}\n")

    with Pool(processes=workers,
              initializer=_init_worker,
              initargs=(dataset_name, data_root)) as pool:
        pbar = tqdm(total=len(tasks), desc=f"Extracting {dataset_name}")
        try:
            for status, payload, scene, camera in pool.imap_unordered(
                    _process_single, tasks):
                if status == "ok":
                    info = payload
                    if info.get("id") not in existing_ids:
                        results.append(info)
                else:
                    failed_counts[status] += 1
                    if len(failed_samples[status]) < SAMPLE_CAP:
                        failed_samples[status].append((scene, camera, payload))
                    failures_fh.write(
                        f"[{status}] {scene} / {camera}\n{payload}\n\n")
                pbar.update(1)
        except KeyboardInterrupt:
            logger.info("Interrupted, saving partial results...")
            pool.terminate()
            pool.join()
        finally:
            pbar.close()
            failures_fh.close()

    elapsed = time.time() - start

    # Append results
    if results:
        with open(output_path, "a", encoding="utf-8") as f:
            for info in results:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")

    total_failed = sum(failed_counts.values())

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  New results: {len(results)}")
    print(f"  Skipped (existing): {len(existing_ids)}")
    print(f"  Failed: {total_failed}")
    for status, cnt in sorted(failed_counts.items(),
                              key=lambda x: -x[1]):
        print(f"    - {status:<16}: {cnt}")
    if total_failed > 0:
        print(f"  Sample failures (up to {SAMPLE_CAP} per category):")
        for status, samples in failed_samples.items():
            print(f"    [{status}]")
            for scene, cam, reason in samples:
                # Keep the last line (most informative for exceptions) and
                # use a generous cap so file paths aren't truncated mid-name.
                short_reason = str(reason).splitlines()[-1][:500]
                print(f"      {scene} / {cam}  ->  {short_reason}")
        print(f"  Full details: {failures_path}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}\n")

    return output_path
