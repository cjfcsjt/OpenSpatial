import argparse
import copy
import multiprocessing
import os
import traceback
from types import SimpleNamespace

import yaml


class DuplicateKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that preserves duplicate mapping keys and their order."""


def _construct_mapping_keep_duplicates(loader, node, deep=False):
    """Build mapping node while keeping duplicate keys as ordered list entries."""
    pairs = []
    result = {}
    has_duplicate_key = False

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        value = loader.construct_object(value_node, deep=deep)
        pairs.append((key, value))
        if key in result:
            has_duplicate_key = True
        result[key] = value

    if has_duplicate_key:
        # Keep duplicates as list entries in original order.
        return [{k: v} for k, v in pairs]
    return result


DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_keep_duplicates,
)


def dict_to_namespace(data):
    """Recursively convert dict/list config into dot-access namespaces."""
    if isinstance(data, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    return data


def _iter_stage_items(stages_cfg):
    """Yield (stage_name, items) from dict-style or list-style stage config."""
    if isinstance(stages_cfg, dict):
        for stage_name, items in stages_cfg.items():
            yield stage_name, items
        return

    if isinstance(stages_cfg, list):
        for idx, stage_def in enumerate(stages_cfg):
            if not isinstance(stage_def, dict):
                raise ValueError(f"Stage entry at index {idx} should be a dict.")
            if len(stage_def) != 1:
                raise ValueError(f"Stage entry at index {idx} should contain exactly one stage key.")
            stage_name, items = next(iter(stage_def.items()))
            yield stage_name, items
        return

    raise ValueError("'pipeline.stages' should be a dict or a list of single-key dicts.")


def validate_config(config_dict):
    """Validate minimal required config schema for pipeline execution."""
    if "pipeline" not in config_dict:
        raise ValueError("YAML config must contain a 'pipeline' field.")
    if "stages" not in config_dict["pipeline"]:
        raise ValueError("YAML config must contain 'pipeline.stages'.")

    for stage, items in _iter_stage_items(config_dict["pipeline"]["stages"]):
        if not isinstance(items, list):
            raise ValueError(f"Stage '{stage}' should be a list.")

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item {idx} in stage '{stage}' should be a dict.")
            if "method" not in item or "output_dir" not in item:
                raise ValueError(
                    f"Item {idx} in stage '{stage}' must contain 'method' and 'output_dir'."
                )


def _load_yaml_config(config_path):
    """Load YAML config with duplicate-key preserving loader."""
    with open(config_path, "r") as file_obj:
        return yaml.load(file_obj, Loader=DuplicateKeySafeLoader)


def _normalize_output_dir(output_dir):
    """Normalize output directory to absolute path."""
    if os.path.isabs(output_dir):
        return output_dir
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(base_dir, output_dir))


def _build_run_output_dir(output_root, pipeline_file_name, config_path):
    """Create and return output directory for current config run."""
    config_stem = os.path.basename(config_path).replace(".yaml", "")
    run_output_dir = os.path.join(output_root, f"{pipeline_file_name}_{config_stem}")
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def get_config():
    """Parse CLI args, load and validate config, and attach runtime output_dir."""
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--parallel_workers", type=int, default=1,
                        help="Number of parallel workers for processing multiple parquet files. "
                             "Default 1 (sequential). Set >1 to enable multiprocessing.Pool.")
    args = parser.parse_args()

    config_dict = _load_yaml_config(args.config)
    validate_config(config_dict)
    config = dict_to_namespace(config_dict)

    args.output_dir = _normalize_output_dir(args.output_dir)
    config.output_dir = _build_run_output_dir(args.output_dir, config.pipeline.file_name, args.config)
    return args, config


def _check_parquet_file(data_dirs):
    """Validate parquet path list: no duplicates and all files exist."""
    seen = set()
    duplicates = []
    missing_files = []

    for data_path in data_dirs:
        if data_path in seen:
            duplicates.append(data_path)
        seen.add(data_path)
        if not os.path.exists(data_path):
            missing_files.append(data_path)

    if duplicates:
        raise ValueError(f"Duplicate file paths: {duplicates}")
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")

    print(">>>>>>All files exist and no duplicates were found. Ready to proceed.")


def _create_pipeline_instance(config):
    """Create pipeline instance from config."""
    from utils.common import get_pipeline

    return get_pipeline(config)


def _run_single_pipeline(config):
    """Run a single pipeline config and return True/False for success."""
    pipeline = _create_pipeline_instance(config)
    if pipeline is None:
        return False
    pipeline.run()
    return True


def _run_single_pipeline_worker(task_args):
    """Worker function for multiprocessing.Pool.

    Args:
        task_args: tuple of (index, total, data_dir, config)

    Returns:
        (index, success, error_msg)
    """
    idx, total, data_dir, config = task_args
    config_copy = copy.deepcopy(config)
    config_copy.dataset.data_dir = data_dir
    config_copy.output_dir = os.path.join(config.output_dir, f"part_{idx + 1}")
    print(f">>>>>>Running datafile [{idx + 1}/{total}]: {data_dir}")
    try:
        success = _run_single_pipeline(config_copy)
        if success:
            print(f">>>>>>Finished datafile [{idx + 1}/{total}]: {data_dir}")
        else:
            print(f">>>>>>Failed datafile [{idx + 1}/{total}]: {data_dir} (pipeline returned None)")
        return (idx, success, None)
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f">>>>>>Error datafile [{idx + 1}/{total}]: {data_dir}\n{error_msg}")
        return (idx, False, error_msg)


def main(args, config):
    """Entry point for executing pipeline on single or multiple input parquet files."""
    if not isinstance(config.dataset.data_dir, list):
        _run_single_pipeline(config)
        return

    data_dirs = config.dataset.data_dir
    _check_parquet_file(data_dirs)

    parallel_workers = getattr(args, "parallel_workers", 1)
    total = len(data_dirs)

    if parallel_workers <= 1:
        # Sequential execution (original behavior)
        for i, data_dir in enumerate(data_dirs):
            config_copy = copy.deepcopy(config)
            config_copy.dataset.data_dir = data_dir
            config_copy.output_dir = os.path.join(config.output_dir, f"part_{i+1}")
            print(f">>>>>>Running datafile [{i+1}/{total}]: {data_dir}")
            if not _run_single_pipeline(config_copy):
                return
    else:
        # Parallel execution with multiprocessing.Pool
        num_workers = min(parallel_workers, total)
        print(f">>>>>>Parallel mode: {num_workers} workers for {total} parquet files")

        task_args_list = [
            (i, total, data_dir, config)
            for i, data_dir in enumerate(data_dirs)
        ]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_run_single_pipeline_worker, task_args_list)

        # Report results
        failed = [(idx, err) for idx, success, err in results if not success]
        succeeded = sum(1 for _, success, _ in results if success)
        print(f">>>>>>Parallel execution complete: {succeeded}/{total} succeeded")
        if failed:
            print(f">>>>>>Failed parquet files:")
            for idx, err in failed:
                print(f"  part_{idx + 1}: {data_dirs[idx]}")
                if err:
                    print(f"    Error: {err.splitlines()[-1]}")


if __name__ == "__main__":
    parsed_args, parsed_config = get_config()
    main(parsed_args, parsed_config)
