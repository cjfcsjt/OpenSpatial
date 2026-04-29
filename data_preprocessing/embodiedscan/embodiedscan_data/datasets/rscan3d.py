import logging
import os
import pickle
from typing import List, Optional

import numpy as np

from embodiedscan_data.datasets import register
from embodiedscan_data.datasets.base import DatasetConfig

logger = logging.getLogger(__name__)


@register
class RScan3DConfig(DatasetConfig):
    name = "3rscan"
    dataset_key = "3rscan"
    depth_scale = 1000
    ann_files = [
        "data/embodiedscan_infos_train.pkl",
        "data/embodiedscan_infos_val.pkl",
        "data/embodiedscan_infos_test.pkl",
    ]

    # Cache parsed scene entries / per-scene cameras so we only read pkl once.
    _cached_scene_entries: Optional[List[str]] = None
    _cached_cameras: Optional[dict] = None  # sample_idx -> sorted[str]

    def _load_scene_entries(self, data_root: str) -> List[str]:
        """Parse pkl annotations and return list of sample_idx strings.

        sample_idx follows pkl format: '3rscan/<scene_hash>'.
        Camera names are derived as ``basename.split('.')[0]`` of each
        ``images[i].img_path`` (e.g. ``frame-XXXXXX.color.jpg`` ->
        ``frame-XXXXXX``), matching the legacy disk-listing rule.
        """
        if RScan3DConfig._cached_scene_entries is not None:
            return RScan3DConfig._cached_scene_entries

        project_root = os.path.dirname(os.path.abspath(data_root))
        entries: List[str] = []
        cameras_by_scene: dict = {}
        seen = set()
        for rel in self.ann_files:
            pkl = os.path.join(project_root, rel)
            if not os.path.isfile(pkl):
                continue
            try:
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue
            for item in data.get("data_list", []):
                sample_idx = item.get("sample_idx", "")
                if not sample_idx.startswith("3rscan/"):
                    continue
                if sample_idx in seen:
                    continue
                seen.add(sample_idx)
                entries.append(sample_idx)
                cams = []
                for img in item.get("images", []):
                    img_path = img.get("img_path", "")
                    if not img_path:
                        continue
                    base = os.path.basename(img_path)
                    if not (base.endswith(".jpg") or base.endswith(".png")):
                        continue
                    # Legacy rule: camera name is the part before the first
                    # dot, e.g. 'frame-XXXXXX.color.jpg' -> 'frame-XXXXXX'.
                    cam = base.split(".")[0]
                    cams.append(cam)
                cameras_by_scene[sample_idx] = sorted(cams)

        RScan3DConfig._cached_scene_entries = entries
        RScan3DConfig._cached_cameras = cameras_by_scene
        return entries

    def list_scenes(self, data_root: str) -> List[str]:
        rscan_dir = os.path.join(data_root, "3rscan")
        if not os.path.isdir(rscan_dir):
            return []

        entries = self._load_scene_entries(data_root)
        if entries:
            # Keep only scenes whose on-disk sequence dir actually exists.
            scenes = [
                s for s in entries
                if os.path.isdir(os.path.join(
                    rscan_dir, s.split("/")[-1], "sequence"))
            ]
            scenes.sort()
            return scenes

        # Legacy fallback: disk listing (when pkl is absent / unreadable).
        return sorted(
            f"3rscan/{d}" for d in os.listdir(rscan_dir)
            if os.path.isdir(os.path.join(rscan_dir, d))
        )

    def list_cameras(self, data_root: str, scene: str) -> List[str]:
        # Prefer pkl-derived camera list to (a) avoid slow listdir on
        # networked filesystems and (b) guarantee every camera we enqueue is
        # actually present in the pkl (so get_info won't return None).
        self._load_scene_entries(data_root)
        if RScan3DConfig._cached_cameras is not None:
            cams = RScan3DConfig._cached_cameras.get(scene)
            if cams is not None:
                return cams
        # Fallback to disk listing (legacy path).
        scene_name = scene.split("/")[-1]
        seq_dir = os.path.join(data_root, "3rscan", scene_name, "sequence")
        if not os.path.isdir(seq_dir):
            return []
        return sorted(f.split(".")[0] for f in os.listdir(seq_dir) if f.endswith(".jpg"))

    def get_scene_id(self, scene: str) -> str:
        return scene.split("/")[-1]

    def get_intrinsic(self, data_root: str, scene: str, camera: str) -> str:
        scene_name = scene.split("/")[-1]
        info_path = os.path.join(data_root, "3rscan", scene_name, "sequence", "_info.txt")
        output_path = os.path.join(data_root, "3rscan", scene_name, "sequence", "_depth_intrinsic.txt")
        if not os.path.exists(output_path):
            self._extract_intrinsic(info_path, output_path)
        return os.path.relpath(output_path, data_root)

    def _extract_intrinsic(self, info_path: str, output_path: str) -> None:
        with open(info_path, "r") as f:
            content = f.read()
        for line in content.split("\n"):
            if line.startswith("m_calibrationDepthIntrinsic"):
                values_str = line.split("=", 1)[1].strip()
                values = [float(x) for x in values_str.split()]
                matrix = np.array(values).reshape(4, 4)
                with open(output_path, "w") as f:
                    for row in matrix:
                        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                return
        logger.warning("No m_calibrationDepthIntrinsic found in %s", info_path)

    def skip_scene(self, data_root: str, scene: str) -> bool:
        scene_name = scene.split("/")[-1]
        seq_dir = os.path.join(data_root, "3rscan", scene_name, "sequence")
        if not os.path.isdir(seq_dir):
            return True
        info_path = os.path.join(seq_dir, "_info.txt")
        if not os.path.exists(info_path):
            return True
        with open(info_path, "r") as f:
            return "m_calibrationDepthIntrinsic" not in f.read()
