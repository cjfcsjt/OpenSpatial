"""Base class for all OpenSpatial task stages."""

import os
import tqdm
import pandas as pd


# Fields that may contain file paths (relative to data_root) in each example.
# When data_root is configured, these fields will be auto-resolved to absolute paths.
_PATH_FIELDS = ("image", "depth_map", "intrinsic", "pose", "axis_align_matrix")


class BaseTask:
    """
    Root base class for all tasks.

    Provides:
        - run(dataset) — standard DataFrame iteration + optional multi-threading
        - _run_multi_processing(dataset) — ThreadPoolExecutor parallel execution
        - resolve_path(path) — join relative path with data_root
        - _resolve_example_paths(example) — auto-resolve path fields in an example

    Subclasses must override:
        - apply_transform(self, example, idx) -> (example, bool)
    """

    def __init__(self, args):
        self.args = args
        self.use_multi_processing = args.get("use_multi_processing", False)
        # data_root is injected by the pipeline from cfg.dataset.data_root.
        # When set, relative path fields in examples are auto-resolved before apply_transform.
        self.data_root = args.get("dataset_data_root", None)

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def resolve_path(self, path):
        """Join a relative path with data_root; return absolute paths unchanged."""
        if not isinstance(path, str) or not path:
            return path
        if os.path.isabs(path) or not self.data_root:
            return path
        return os.path.join(self.data_root, path)

    def _resolve_example_paths(self, example):
        """Auto-resolve known path fields in an example to absolute paths."""
        if not self.data_root:
            return example
        for key in _PATH_FIELDS:
            if key in example:
                val = example[key]
                if isinstance(val, str):
                    example[key] = self.resolve_path(val)
                elif isinstance(val, list):
                    example[key] = [self.resolve_path(v) for v in val]
        return example

    def apply_transform(self, example, idx):
        raise NotImplementedError

    def run(self, dataset):
        if self.use_multi_processing:
            return self._run_multi_processing(dataset)

        processed = []
        for idx in tqdm.tqdm(range(len(dataset)), total=len(dataset),
                             desc="Processing examples"):
            example = dataset.iloc[idx].to_dict()
            example = self._resolve_example_paths(example)
            result, flag = self.apply_transform(example, idx)
            if flag:
                processed.append(result)

        return pd.DataFrame(processed).reset_index(drop=True)

    def _run_multi_processing(self, dataset):
        from concurrent.futures import ThreadPoolExecutor

        num_workers = self.args.get('num_workers', 8)
        examples = list(enumerate(dataset.to_dict('records')))

        def _process(item):
            idx, example = item
            example = self._resolve_example_paths(example)
            return self.apply_transform(example, idx)

        processed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_process, examples)
            for result, flag in tqdm.tqdm(results, total=len(examples),
                                          desc="Processing examples"):
                if flag:
                    processed.append(result)

        return pd.DataFrame(processed).reset_index(drop=True)
