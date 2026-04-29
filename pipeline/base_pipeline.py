from utils.common import get_task_instance
from dataset import build_dataset
from dataset.blink_writer import BlinkWriter
import copy
import os
from collections import Counter


class BasePipeline:
    """Pipeline executor with stage-based task management and dependency resolution."""

    @staticmethod
    def _get_duplicate_suffix(occurrence):
        """Convert occurrence count to duplicate suffix: 2->dup1, 3->dup2."""
        return f"dup{occurrence - 1}" if occurrence > 1 else ""

    @staticmethod
    def _iter_stages(stages_cfg):
        """Yield (stage_name, tasks) from config in order."""
        stage_groups = stages_cfg if isinstance(stages_cfg, (list, tuple)) else [stages_cfg]

        for group in stage_groups:
            if hasattr(group, "items"):
                items = group.items()
            elif hasattr(group, "__dict__"):
                items = group.__dict__.items()
            else:
                raise ValueError("stages must be a mapping or list of mappings")

            for stage_name, tasks in items:
                yield stage_name, tasks

    @staticmethod
    def _format_task_ref(stage_name, task_name, occurrence):
        """Format task reference as stage/task#N."""
        return f"{stage_name}/{task_name}#{occurrence}"

    def _resolve_dependency_path(self, depends_on, current_task_idx=None):
        """Resolve depends_on to concrete parquet path.

        Priority:
        1. Absolute/relative file path
        2. Explicit task ref with #N
        3. Bare stage/task (if unique before current task)
        4. Legacy output_root/depends_on/data.parquet
        """
        if os.path.isfile(depends_on):
            return depends_on

        abs_path = os.path.join(self.output_root, depends_on)
        if os.path.isfile(abs_path):
            return abs_path

        # Explicit #N reference
        if depends_on in self.task_ref_to_rel_dir:
            return os.path.join(self.output_root, self.task_ref_to_rel_dir[depends_on], "data.parquet")

        # Bare "stage/task" - resolve from tasks before current
        if depends_on in self.base_ref_to_tasks:
            candidates = self.base_ref_to_tasks[depends_on]
            if current_task_idx is not None:
                candidates = [t for t in candidates if t["queue_idx"] < current_task_idx]

            if len(candidates) == 1:
                return os.path.join(self.output_root, candidates[0]["rel_output_dir"], "data.parquet")

            if len(candidates) > 1:
                refs = [self._format_task_ref(t["stage_name"], t["task_name"], t["occurrence"])
                        for t in candidates]
                raise ValueError(f"Ambiguous depends_on '{depends_on}'. Use one of: {refs}")

        # Legacy fallback
        legacy_path = os.path.join(self.output_root, depends_on, "data.parquet")
        if os.path.isfile(legacy_path):
            return legacy_path

        raise ValueError(f"Cannot resolve depends_on: {depends_on}")

    def __init__(self, cfg):
        self.cfg = cfg 
        self.data_cfg = self.cfg.dataset 
        self.output_root = self.cfg.output_dir

        # Build task queue with occurrence tracking for duplicates
        self.task_queue = []
        pair_counter = Counter()
        for stage_name, tasks in self._iter_stages(cfg.pipeline.stages):
            for task_cfg in tasks:
                task_name = task_cfg.file_name
                pair_counter[(stage_name, task_name)] += 1
                occurrence = pair_counter[(stage_name, task_name)]
                suffix = self._get_duplicate_suffix(occurrence)
                unique_dir = f"{task_name}_{suffix}" if suffix else task_name
                self.task_queue.append({
                    "stage_name": stage_name,
                    "task_cfg": task_cfg,
                    "task_name": task_name,
                    "task": get_task_instance(stage_name, task_cfg, self.cfg),
                    "queue_idx": len(self.task_queue),
                    "occurrence": occurrence,
                    "base_ref": f"{stage_name}/{task_name}",
                    "full_ref": self._format_task_ref(stage_name, task_name, occurrence),
                    "rel_output_dir": os.path.join(stage_name, unique_dir),
                })

        # Build lookup tables for dependency resolution
        self.base_ref_to_tasks = {}
        self.task_ref_to_rel_dir = {}
        for task in self.task_queue:
            self.base_ref_to_tasks.setdefault(task["base_ref"], []).append(task)
            self.task_ref_to_rel_dir[task["full_ref"]] = task["rel_output_dir"]

        # Reuse models across tasks in same stage
        first_task_by_stage = {}
        for task in self.task_queue:
            first_task_by_stage.setdefault(task["stage_name"], task)

        for task in self.task_queue:
            reuse_stage = getattr(task["task"], "reuse_model", None)
            if reuse_stage and reuse_stage in first_task_by_stage:
                task["task"].model = first_task_by_stage[reuse_stage]["task"].model
                print(f">>> Reusing model from {reuse_stage} for {task['task_name']}")

        # Build dataset
        self.dataset = build_dataset(self.data_cfg, self.data_cfg.dataset_name)

        # Initialize dataset from first task's depends_on if present
        if not self.task_queue:
            raise ValueError("No tasks found in pipeline stages")
        first_task_cfg = self.task_queue[0]["task_cfg"]
        if hasattr(first_task_cfg, "depends_on"):
            resolved_path = self._resolve_dependency_path(first_task_cfg.depends_on, current_task_idx=0)
            print(f">>> Overriding dataset with {resolved_path}...")
            self.dataset.override_data(resolved_path)

    def _resolve_output_path(self, stage_name, task_name, task_cfg, default_rel_dir=None):
        """Resolve output directory for task results."""
        output_dir = task_cfg.output_dir
        if output_dir:
            return output_dir if os.path.exists(output_dir) else os.path.join(self.output_root, output_dir)

        rel_dir = default_rel_dir or os.path.join(stage_name, task_name)
        return os.path.join(self.output_root, rel_dir)

    def save_task_data(self, stage_name, task_name, task_cfg, processed_data, default_rel_dir=None):
        """Save processed task data.

        Annotation-stage outputs default to BLINK format (per-task jsonl +
        images/ folder) so no on-the-fly parquet <-> jsonl round-trip is
        needed downstream.  Parquet can still be produced in parallel via
        the per-task ``output.format`` config, or as a fallback format.
        """
        output_path = self._resolve_output_path(stage_name, task_name, task_cfg, default_rel_dir)
        os.makedirs(output_path, exist_ok=True)

        print(f"  [save] stage={stage_name} task={task_name}")
        print(f"  [save] output_root={self.output_root}")
        print(f"  [save] output_path={output_path}")

        if stage_name != "annotation_stage":
            self.dataset.save_data(os.path.join(output_path, "data.parquet"), processed_data)
            return

        formats = self._resolve_annotation_formats(task_cfg)
        batch_size = getattr(task_cfg, "save_batch_size", 1000)
        keep_cols = getattr(task_cfg, "keep_data_columns",
                           ["messages", "QA_images", "question_tags", "question_types",
                            "cognitive_maps"])

        # DataFrame overview -- highly useful when converted=0/0.
        try:
            nrows = 0 if processed_data is None else len(processed_data)
        except Exception:
            nrows = -1
        print(f"  [save] formats={formats}  batch_size={batch_size}  rows={nrows}")
        if nrows > 0 and hasattr(processed_data, "columns"):
            cols = list(processed_data.columns)
            print(f"  [save] columns={cols}")
            row0 = processed_data.iloc[0]
            for key in ("messages", "QA_images", "question_tags", "question_types",
                        "cognitive_maps"):
                if key not in cols:
                    print(f"  [save]   {key!s:>16}: <missing column>")
                    continue
                val = row0.get(key)
                tp = type(val).__name__
                try:
                    n = len(val) if hasattr(val, "__len__") else "-"
                except Exception:
                    n = "?"
                print(f"  [save]   {key!s:>16}: type={tp} len={n}")

        if "parquet" in formats:
            self.dataset.save_data(os.path.join(output_path, "data.parquet"), processed_data,
                                  annotation_flag=True, batch_size=batch_size,
                                  keep_data_columns=keep_cols)

        if "blink" in formats:
            blink_root = self._resolve_blink_root(task_cfg)
            data_source = self._resolve_blink_data_source(task_cfg)
            print(f"  [save] blink_root={blink_root}  data_source={data_source}")
            writer = BlinkWriter(blink_root=blink_root, data_source=data_source)
            writer.write(task_name, processed_data)

    # ------------------------------------------------------------------
    # Annotation output-format resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_namespace(obj):
        """Yield (key, value) from a dict or SimpleNamespace uniformly."""
        if isinstance(obj, dict):
            return obj.items()
        if hasattr(obj, "__dict__"):
            return obj.__dict__.items()
        return []

    def _get_output_cfg(self, task_cfg):
        """Return the ``output`` sub-config (dict or SimpleNamespace) or None."""
        output_cfg = getattr(task_cfg, "output", None)
        if output_cfg is None:
            return None
        # yaml may render as plain dict or SimpleNamespace; both are fine.
        return output_cfg

    def _resolve_annotation_formats(self, task_cfg):
        """Return normalized list of output formats for annotation stage.

        Defaults to ``["blink"]`` when nothing is configured.  Accepts:
            output:
              format: blink
              format: [blink, parquet]
        """
        output_cfg = self._get_output_cfg(task_cfg)
        fmt = None
        if output_cfg is not None:
            if isinstance(output_cfg, dict):
                fmt = output_cfg.get("format")
            else:
                fmt = getattr(output_cfg, "format", None)
        if fmt is None:
            return ["blink"]
        if isinstance(fmt, str):
            fmt = [fmt]
        normalized = []
        for name in fmt:
            name = str(name).strip().lower()
            if name in ("blink", "parquet"):
                normalized.append(name)
        return normalized or ["blink"]

    def _resolve_blink_root(self, task_cfg):
        """Resolve the root directory for BLINK outputs.

        Priority:
            1. task_cfg.output.blink_dir (absolute or relative to output_root's parent)
            2. $BLINK_OUT_DIR env var (absolute path injected by the shell driver)
            3. <parent_of_output_root>/05_blink   (pipeline-wide default)
            4. <output_root>/05_blink             (fallback)
        """
        output_cfg = self._get_output_cfg(task_cfg)
        blink_dir = None
        if output_cfg is not None:
            if isinstance(output_cfg, dict):
                blink_dir = output_cfg.get("blink_dir")
            else:
                blink_dir = getattr(output_cfg, "blink_dir", None)

        if blink_dir:
            if os.path.isabs(blink_dir):
                return blink_dir
            # Treat as relative to parent of output_root so sibling of 04_annotation.
            return os.path.join(os.path.dirname(self.output_root), blink_dir)

        env_dir = os.environ.get("BLINK_OUT_DIR", "").strip()
        if env_dir:
            return env_dir

        # Pipeline-wide default: sibling of 04_annotation named 05_blink.
        parent = os.path.dirname(self.output_root)
        if parent:
            return os.path.join(parent, "05_blink")
        return os.path.join(self.output_root, "05_blink")

    def _resolve_blink_data_source(self, task_cfg):
        output_cfg = self._get_output_cfg(task_cfg)
        if output_cfg is not None:
            if isinstance(output_cfg, dict):
                ds = output_cfg.get("data_source")
            else:
                ds = getattr(output_cfg, "data_source", None)
            if ds:
                return str(ds)
        return "OpenSpatial"

    def load_task_data(self, task_name, task_cfg, current_task_idx):
        """Load input data from depends_on path."""
        depends_on = getattr(task_cfg, "depends_on", None)
        if not depends_on:
            raise ValueError(f"Task {task_name} missing depends_on field")

        resolved_path = self._resolve_dependency_path(depends_on, current_task_idx=current_task_idx)
        self.dataset.override_data(resolved_path)

    def run(self):
        """Execute all tasks in pipeline sequentially."""
        print(">>> Running Pipeline...")
        for i, task_info in enumerate(self.task_queue):
            stage_name = task_info["stage_name"]
            task_cfg = task_info["task_cfg"]
            task_name = task_info["task_name"]
            task = task_info["task"]

            if i > 0:
                print(f">>> Loading data from: {self.task_queue[i-1]['full_ref']}...")
                self.load_task_data(task_name, task_cfg, current_task_idx=i)

            print(f">>> Running Task [{i+1}/{len(self.task_queue)}]: {task_name}...")
            processed_data = task.run(copy.deepcopy(self.dataset.data))
            self.save_task_data(stage_name, task_name, task_cfg, processed_data, task_info["rel_output_dir"])
            # Cognitive-map rendering failure-rate warning (best-effort).
            total = getattr(task, "_cog_total_count", 0)
            fail = getattr(task, "_cog_fail_count", 0)
            if total > 0:
                rate = fail / float(total)
                print(f">>> Cognitive-map render: {total - fail}/{total} "
                      f"succeeded ({rate * 100:.1f}% failure).")
                if rate > 0.1:
                    print(">>> [WARN] Cognitive-map render failure rate > 10%. "
                          "Check matplotlib backend / font availability.")

        print(">>> Pipeline Finished.")

