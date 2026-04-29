import os
import pickle
from typing import List, Optional

from embodiedscan_data.datasets import register
from embodiedscan_data.datasets.base import DatasetConfig


@register
class ScanNetConfig(DatasetConfig):
    name = "scannet"
    dataset_key = "scannet"
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

        sample_idx follows pkl format: 'scannet/<scene_id>'.
        Also populates the cameras cache keyed by sample_idx. Camera names are
        derived as ``basename.rsplit('.', 1)[0]`` of each
        ``images[i].img_path`` (matching the legacy disk-listing rule since
        ScanNet basenames have no extra dots, e.g. ``00000.jpg``).
        """
        if ScanNetConfig._cached_scene_entries is not None:
            return ScanNetConfig._cached_scene_entries

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
                if not sample_idx.startswith("scannet/"):
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
                    cam = base.rsplit(".", 1)[0]
                    cams.append(cam)
                cameras_by_scene[sample_idx] = sorted(cams)

        ScanNetConfig._cached_scene_entries = entries
        ScanNetConfig._cached_cameras = cameras_by_scene
        return entries

    def list_scenes(self, data_root: str) -> List[str]:
        scannet_dir = os.path.join(data_root, "scannet")
        if not os.path.isdir(scannet_dir):
            return []

        entries = self._load_scene_entries(data_root)
        if entries:
            # Keep only scenes whose posed_images dir actually exists, so we
            # don't hand the Explorer a scene it can't read.
            posed_root = os.path.join(scannet_dir, "posed_images")
            scenes = [
                s for s in entries
                if os.path.isdir(os.path.join(posed_root, s.split("/")[-1]))
            ]
            scenes.sort()
            return scenes

        # Legacy fallback: disk listing (when pkl is absent / unreadable).
        posed_dir = os.path.join(scannet_dir, "posed_images")
        if not os.path.isdir(posed_dir):
            return []
        return sorted(
            f"scannet/{d}" for d in os.listdir(posed_dir)
            if os.path.isdir(os.path.join(posed_dir, d))
        )

    def list_cameras(self, data_root: str, scene: str) -> List[str]:
        # Prefer pkl-derived camera list to (a) avoid slow listdir on
        # networked filesystems and (b) guarantee every camera we enqueue is
        # actually present in the pkl (so get_info won't return None).
        self._load_scene_entries(data_root)
        if ScanNetConfig._cached_cameras is not None:
            cams = ScanNetConfig._cached_cameras.get(scene)
            if cams is not None:
                return cams
        # Fallback to disk listing (legacy path).
        scene_name = scene.split("/")[-1]
        scene_dir = os.path.join(data_root, "scannet", "posed_images", scene_name)
        if not os.path.isdir(scene_dir):
            return []
        return sorted(f.split(".")[0] for f in os.listdir(scene_dir) if f.endswith(".jpg"))

    def get_scene_id(self, scene: str) -> str:
        return scene.split("/")[-1]

    def get_intrinsic(self, data_root: str, scene: str, camera: str) -> str:
        scene_name = scene.split("/")[-1]
        return os.path.join("scannet", "scans", scene_name, "intrinsic", "intrinsic_depth.txt")

    def post_process(self, info: dict, data_root: str, scene: str, camera: str) -> dict:
        from PIL import Image
        image_path = info.get("image")
        depth_path = info.get("depth_map")
        if not image_path or not depth_path:
            return info
        abs_image = os.path.join(data_root, image_path) if not os.path.isabs(image_path) else image_path
        abs_depth = os.path.join(data_root, depth_path) if not os.path.isabs(depth_path) else depth_path
        if os.path.exists(abs_image) and os.path.exists(abs_depth):
            img = Image.open(abs_image)
            depth = Image.open(abs_depth)
            if img.size != depth.size:
                img = img.resize(depth.size)
                resized_path = abs_image.replace(".jpg", "_resized.jpg")
                img.save(resized_path)
                info["image"] = os.path.relpath(resized_path, data_root)
        return info

    def skip_scene(self, data_root: str, scene: str) -> bool:
        scene_name = scene.split("/")[-1]
        intrinsic_path = os.path.join(data_root, "scannet", "scans", scene_name, "intrinsic", "intrinsic_depth.txt")
        return not os.path.exists(intrinsic_path)

    def get_save_path(self, data_root: str, scene: str) -> str:
        scene_name = scene.split("/")[-1]
        return os.path.join(data_root, "scannet", "posed_images", scene_name)
