import importlib
import os


def get_pipeline(config):
    """Create pipeline instance from config."""
    module_path = f"pipeline.{config.pipeline.file_name}"
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, config.pipeline.class_name)
        return cls(config)
    except Exception as e:
        print(f"Failed to instantiate pipeline: {e}")
        return None


def get_task_instance(stage_name, task_cfg, cfg):
    """Create task instance from stage config."""
    module_path = f"task.{stage_name[:-6]}.{task_cfg.file_name}"
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, task_cfg.method)
        task_cfg_dict = task_cfg.__dict__.copy()
        if not task_cfg_dict.get("output_dir"):
            task_cfg_dict["output_dir"] = os.path.join(cfg.output_dir, stage_name)
        # Inject dataset.data_root so BaseTask can auto-resolve relative path
        # fields (image/depth_map/intrinsic/pose/axis_align_matrix) stored in parquet.
        dataset_cfg = getattr(cfg, "dataset", None)
        data_root = getattr(dataset_cfg, "data_root", None) if dataset_cfg is not None else None
        if data_root:
            task_cfg_dict.setdefault("dataset_data_root", data_root)
        return cls(task_cfg_dict)
    except Exception as e:
        raise ImportError(f"Failed to import {module_path}.{task_cfg.method}: {e}")
