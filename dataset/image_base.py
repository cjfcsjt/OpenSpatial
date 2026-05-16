import io
import math
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image as PILImage

from utils.data_utils import flatten_annotations

HF_REPO_PATTERN = re.compile(r'^[\w-]+/[\w-]+$')

# Matches arkitscenes per-image path:
#   arkitscenes/<split>/<scene_id>/<scene_id>_frames/...
# and inserts an extra '<scene_id>/' level after <scene_id>/ to reflect the
# actual on-disk layout produced by the raw unzipper:
#   arkitscenes/<split>/<scene_id>/<scene_id>/<scene_id>_frames/...
_ARKIT_PATH_RE = re.compile(
    r"(arkitscenes/[^/]+/)(?P<sid>[^/]+)/(?P=sid)_frames/"
)


def _rewrite_arkit_path(path):
    """Insert the missing `<scene_id>/` level for arkitscenes paths.

    Only rewrites if the target path does not already contain the duplicated
    scene_id segment (i.e. the function is idempotent).
    """
    if not isinstance(path, str) or "arkitscenes" not in path:
        return path
    m = _ARKIT_PATH_RE.search(path)
    if not m:
        return path
    sid = m.group("sid")
    # Already rewritten?  e.g. '.../<sid>/<sid>/<sid>_frames/...'
    prefix = path[: m.start()] + m.group(1)  # up to and including 'arkitscenes/<split>/'
    remainder = path[m.start(1) + len(m.group(1)) :]  # '<sid>/<sid>_frames/...'
    if remainder.startswith(f"{sid}/{sid}/{sid}_frames/"):
        return path
    return f"{prefix}{sid}/{remainder}"


class ImageBaseDataset:
    """Base image dataset backed by parquet or HuggingFace Hub."""

    MODALITY = "image"

    # Columns that may contain arkitscenes file paths requiring rewrite.
    _ARKIT_PATH_COLUMNS = ("image", "depth_map")

    def __init__(self, cfg):
        if not cfg.data_dir:
            raise ValueError("cfg.data_dir is required")
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.data = self._load()

    # ------------------------------------------------------------------
    # Load / Override
    # ------------------------------------------------------------------

    def _load(self):
        """Load data from HuggingFace Hub or local parquet."""
        if HF_REPO_PATTERN.match(self.data_dir):
            df = pd.DataFrame(load_dataset(self.data_dir, split="train"))
        else:
            df = pd.read_parquet(self.data_dir, engine="pyarrow", dtype_backend="pyarrow")
        return self._fix_arkitscenes_paths(df)

    def override_data(self, data_path):
        """Replace in-memory data with another parquet file."""
        try:
            df = pd.read_parquet(data_path, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as exc:
            raise ValueError(f"Failed to load parquet: {data_path}") from exc
        self.data = self._fix_arkitscenes_paths(df)

    # ------------------------------------------------------------------
    # ARKitScenes path rewrite
    # ------------------------------------------------------------------

    @classmethod
    def _fix_arkitscenes_paths(cls, df):
        """Patch arkitscenes `image`/`depth_map` paths to match on-disk layout.

        The parquet produced by `embodiedscan_data` stores paths as
        `arkitscenes/<split>/<scene_id>/<scene_id>_frames/...`, but the raw
        unzipper leaves an extra `<scene_id>/` directory on disk
        (`.../<scene_id>/<scene_id>/<scene_id>_frames/...`). This method
        rewrites the in-memory DataFrame so downstream tasks can open the
        files directly without changing the stored parquet content.

        Only rows with `dataset == "arkitscenes"` are touched; other
        datasets are left untouched.
        """
        if df is None or len(df) == 0 or "dataset" not in df.columns:
            return df

        mask = df["dataset"].astype(str) == "arkitscenes"
        if not mask.any():
            return df

        for col in cls._ARKIT_PATH_COLUMNS:
            if col not in df.columns:
                continue
            df.loc[mask, col] = df.loc[mask, col].map(_rewrite_arkit_path)
        return df

    # ------------------------------------------------------------------
    # Image format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_dict_to_pil(img_dict):
        """Convert {"bytes": ...} dict to PIL Image."""
        if isinstance(img_dict, dict) and img_dict.get("bytes"):
            try:
                return PILImage.open(io.BytesIO(img_dict["bytes"]))
            except Exception:
                return img_dict
        return img_dict

    @staticmethod
    def _pil_to_bytes_dict(image):
        """Convert PIL Image to {"bytes": ...} dict."""
        if isinstance(image, PILImage.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return {"bytes": buf.getvalue()}
        return image

    def convert_image_column_to_pil(self, df, col="image"):
        """Convert bytes dicts in a column to PIL objects (in-place)."""
        def _convert(item):
            if item is None:
                return None
            if isinstance(item, dict):
                return self._bytes_dict_to_pil(item)
            if isinstance(item, (list, tuple, np.ndarray)):
                seq = list(item)
                if seq and all(isinstance(x, dict) and "bytes" in x for x in seq):
                    return [self._bytes_dict_to_pil(x) for x in seq]
                return seq
            return item

        df[col] = [_convert(item) for item in df[col]]
        return df

    def pil_convert_to_bytes(self, df):
        """Convert PIL images in all DataFrame columns to bytes dicts."""
        def _is_pil(x):
            return isinstance(x, PILImage.Image) or (
                isinstance(x, list) and all(isinstance(i, PILImage.Image) for i in x))

        for col in df.columns:
            if df[col].apply(_is_pil).any():
                df[col] = df[col].apply(
                    lambda x: [self._pil_to_bytes_dict(i) for i in x]
                    if isinstance(x, list) else self._pil_to_bytes_dict(x))
        return df

    def pil_convert_to_np(self, data):
        """Convert image column from PIL to nested Python lists."""
        images = data["image"]
        if not len(images):
            return data

        if isinstance(images.iloc[0], list):
            data["image"] = [[np.array(img).tolist() for img in row] for row in images]
        else:
            data["image"] = [np.array(img).tolist() for img in images]
        return data

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_data(self, data_path, data=None, annotation_flag=False,
                  batch_size=1000, keep_data_columns=None):
        """Save DataFrame to parquet with optional annotation flattening."""
        if data is None:
            raise ValueError("Data to save is None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only pandas DataFrame is supported")

        if annotation_flag:
            keep_data_columns = keep_data_columns or [
                "messages", "QA_images", "question_tags", "question_types"]
            data = flatten_annotations(data, keep_keys=keep_data_columns)
            if len(data) > batch_size:
                self._save_batches(data_path, data, batch_size)
                return

        data.to_parquet(data_path, engine="pyarrow")

    @staticmethod
    def _save_batches(data_path, data, batch_size):
        """Save DataFrame into multiple parquet parts."""
        base = os.path.splitext(data_path)[0]
        for i in range(math.ceil(len(data) / batch_size)):
            batch = data.iloc[i * batch_size:(i + 1) * batch_size]
            batch.to_parquet(f"{base}_part_{i}.parquet", engine="pyarrow")

    def convert_to_hf_dataset(self, data):
        """Convert pandas DataFrame to HuggingFace Dataset."""
        return Dataset.from_dict(data.to_dict(orient="list"))
