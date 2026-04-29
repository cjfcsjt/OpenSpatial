import os
import pickle
from typing import List, Optional

import numpy as np

from embodiedscan_data.datasets import register
from embodiedscan_data.datasets.base import DatasetConfig


@register
class ARKitScenesConfig(DatasetConfig):
    name = "arkitscenes"
    dataset_key = "arkitscenes"
    depth_scale = 1000
    ann_files = [
        "embodiedscan-v2/embodiedscan_infos_train.pkl",
        "embodiedscan-v2/embodiedscan_infos_val.pkl",
        "embodiedscan-v2/embodiedscan_infos_test.pkl",
    ]

    # Cache parsed scene entries / per-scene cameras / per-frame intrinsics so
    # we only read pkl once.
    _cached_scene_entries: Optional[List[tuple]] = None
    _cached_cameras: Optional[dict] = None     # sample_idx -> sorted[str]
    _cached_intrinsics: Optional[dict] = None  # (sample_idx, camera) -> 4x4 list

    def _scene_disk_rel(self, scene: str) -> str:
        """Return '<split>/<scene_id>' from a pkl sample_idx.

        Examples:
            'arkitscenes/Training/40776203' -> 'Training/40776203'
            'arkitscenes/40776203'          -> '40776203'   (legacy flat)
        """
        parts = scene.split("/", 1)
        # parts[0] is always 'arkitscenes'; the rest is split/scene_id.
        return parts[1] if len(parts) == 2 else scene

    def _disk_scene_dir(self, data_root: str, scene: str) -> str:
        """Absolute scene directory on disk, including the split layer."""
        return os.path.join(data_root, "arkitscenes", self._scene_disk_rel(scene))

    def _disk_frames_dir(self, data_root: str, scene: str) -> str:
        """Absolute path to <scene>/<scene_id>_frames on disk."""
        scene_id = scene.split("/")[-1]
        return os.path.join(self._disk_scene_dir(data_root, scene),
                            f"{scene_id}_frames")

    def _load_scene_entries(self, data_root: str) -> List[tuple]:
        """Parse pkl annotations and return list of (sample_idx, scene_id).

        sample_idx follows pkl format: 'arkitscenes/Training/<id>'.
        Also populates the cameras cache keyed by sample_idx.
        """
        if ARKitScenesConfig._cached_scene_entries is not None:
            return ARKitScenesConfig._cached_scene_entries

        project_root = os.path.dirname(os.path.abspath(data_root))
        entries = []
        cameras_by_scene: dict = {}
        intrinsics: dict = {}
        # Scene-level cam2img fallback (if pkl ever puts it at scene level).
        scene_cam2img: dict = {}
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
                if not sample_idx.startswith("arkitscenes/"):
                    continue
                if sample_idx in seen:
                    continue
                seen.add(sample_idx)
                scene_id = sample_idx.split("/")[-1]
                entries.append((sample_idx, scene_id))
                if "cam2img" in item:
                    scene_cam2img[sample_idx] = item["cam2img"]
                cams = []
                for img in item.get("images", []):
                    img_path = img.get("img_path", "")
                    base = os.path.basename(img_path)
                    # arkitscenes: '<scene>_<timestamp>.png' -> strip ext
                    if not (base.endswith(".png") or base.endswith(".jpg")):
                        continue
                    cam = base[:-4]
                    cams.append(cam)
                    # Prefer per-frame cam2img; fall back to scene-level.
                    k = (sample_idx, cam)
                    if "cam2img" in img:
                        intrinsics[k] = img["cam2img"]
                    elif sample_idx in scene_cam2img:
                        intrinsics[k] = scene_cam2img[sample_idx]
                cameras_by_scene[sample_idx] = sorted(cams)

        ARKitScenesConfig._cached_scene_entries = entries
        ARKitScenesConfig._cached_cameras = cameras_by_scene
        ARKitScenesConfig._cached_intrinsics = intrinsics
        return entries

    def list_scenes(self, data_root: str) -> List[str]:
        arkit_dir = os.path.join(data_root, "arkitscenes")
        if not os.path.isdir(arkit_dir):
            return []

        entries = self._load_scene_entries(data_root)
        scenes = []
        for sample_idx, _scene_id in entries:
            # Disk layout: data/arkitscenes/<split>/<scene_id>/<scene_id>_frames/
            # matches pkl sample_idx 'arkitscenes/<split>/<scene_id>' directly.
            if not os.path.isdir(self._disk_scene_dir(data_root, sample_idx)):
                continue
            scenes.append(sample_idx)
        scenes.sort()
        return scenes

    def list_cameras(self, data_root: str, scene: str) -> List[str]:
        # Use pkl-derived camera list to (a) avoid slow listdir on networked
        # filesystems and (b) guarantee every camera we enqueue is actually
        # present in the pkl (so get_info won't return None).
        self._load_scene_entries(data_root)
        if ARKitScenesConfig._cached_cameras is not None:
            cams = ARKitScenesConfig._cached_cameras.get(scene)
            if cams is not None:
                return cams
        # Fallback to disk listing (legacy path)
        frames_dir = os.path.join(
            self._disk_frames_dir(data_root, scene), "lowres_wide"
        )
        if not os.path.isdir(frames_dir):
            return []
        cameras = sorted(
            ".".join(f.split(".")[:-1])
            for f in os.listdir(frames_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        )
        return cameras

    def get_scene_id(self, scene: str) -> str:
        return scene.split("/")[-1]

    def get_intrinsic(self, data_root: str, scene: str, camera: str) -> str:
        """Write intrinsic as a 4x4 matrix txt and return its relative path.

        Primary source: pkl's per-frame ``cam2img`` (guaranteed to match the
        frames EmbodiedScan annotates). Fallback: parse the raw ARKitScenes
        ``.pincam`` file on disk (legacy path, may be missing due to
        timestamp drift between RGB and intrinsic sampling rates).
        """
        frames_dir = self._disk_frames_dir(data_root, scene)
        # We keep intrinsics under a separate folder so we don't need the
        # original .pincam file to exist and we never race with other writers.
        out_dir = os.path.join(frames_dir, "lowres_wide_intrinsics_from_pkl")
        output_path = os.path.join(out_dir, f"{camera}_matrix.txt")
        if not os.path.exists(output_path):
            matrix = self._lookup_pkl_intrinsic(data_root, scene, camera)
            if matrix is not None:
                os.makedirs(out_dir, exist_ok=True)
                self._write_matrix(matrix, output_path)
            else:
                # Legacy fallback: parse original .pincam on disk.
                pincam_path = os.path.join(
                    frames_dir, "lowres_wide_intrinsics", f"{camera}.pincam",
                )
                legacy_output = pincam_path.replace(".pincam", "_matrix.txt")
                if os.path.exists(legacy_output):
                    return os.path.relpath(legacy_output, data_root)
                self._parse_pincam(pincam_path, legacy_output)
                return os.path.relpath(legacy_output, data_root)
        return os.path.relpath(output_path, data_root)

    def _lookup_pkl_intrinsic(self, data_root: str, scene: str,
                              camera: str) -> Optional[np.ndarray]:
        """Return the 4x4 intrinsic matrix from pkl cache, or None."""
        self._load_scene_entries(data_root)
        if ARKitScenesConfig._cached_intrinsics is None:
            return None
        m = ARKitScenesConfig._cached_intrinsics.get((scene, camera))
        if m is None:
            return None
        arr = np.asarray(m, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.shape == (3, 3):
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = arr
            return out
        if arr.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :4] = arr
            return out
        return None

    @staticmethod
    def _write_matrix(matrix: np.ndarray, output_path: str) -> None:
        with open(output_path, "w") as f:
            for row in matrix:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    def _parse_pincam(self, pincam_path: str, output_path: str) -> None:
        with open(pincam_path, "r") as f:
            content = f.read()
        values = [float(x) for x in content.split()]
        fx, fy, cx, cy = values[2], values[3], values[4], values[5]
        matrix = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._write_matrix(matrix, output_path)

    def get_depth_map(self, data_root: str, scene: str, camera: str) -> Optional[str]:
        scene_id = scene.split("/")[-1]
        return os.path.join(
            "arkitscenes", self._scene_disk_rel(scene),
            f"{scene_id}_frames", "lowres_depth", f"{camera}.png"
        )

    def skip_scene(self, data_root: str, scene: str) -> bool:
        # Scene entries come from pkl, and list_scenes already verified the
        # disk directory exists. Skipping the per-scene os.path.isdir check
        # here avoids redundant networked-FS stats on 2000+ scenes.
        return False

    def get_save_path(self, data_root: str, scene: str) -> str:
        return os.path.join(self._disk_frames_dir(data_root, scene), "lowres_wide")
