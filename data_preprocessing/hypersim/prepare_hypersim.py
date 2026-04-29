"""Hypersim dataset preprocessing pipeline.

Sequentially executes all preprocessing steps:
  1. Extract intrinsics from camera parameters CSV → per-scene JSON
  2. Extract extrinsics from HDF5 orientation/position data → per-frame JSON
  3. Tonemap raw HDR renders (HDF5) → preview JPG images
  4. Convert Euclidean distance maps (HDF5) → planar depth PNG (16-bit)
  5. Aggregate all data + instance/semantic annotations → Parquet files
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re

import cv2
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: Intrinsics
# ---------------------------------------------------------------------------

def extract_intrinsics(camera_params_csv: str, scene_name: str, output_json: str) -> None:
    """Extract 3x3 intrinsic matrix from the Hypersim camera-parameters CSV."""
    df = pd.read_csv(camera_params_csv)
    try:
        row = df[df["scene_name"] == scene_name].iloc[0]
    except IndexError:
        print(f"[intrinsics] Scene '{scene_name}' not found in CSV, skipping.")
        return

    width = float(row["settings_output_img_width"])
    height = float(row["settings_output_img_height"])
    m00 = float(row["M_proj_00"])
    m11 = float(row["M_proj_11"])

    fx = m00 * (width - 1) * 0.5
    fy = m11 * (height - 1) * 0.5
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    intrinsic_matrix = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(intrinsic_matrix, f, indent=4)

    # Save 4x4 intrinsic as txt
    intrinsic_4x4 = np.eye(4)
    intrinsic_4x4[:3, :3] = np.array(intrinsic_matrix)
    output_txt = output_json.replace(".json", ".txt")
    np.savetxt(output_txt, intrinsic_4x4)


def run_intrinsics(
    input_root: str,
    camera_params_csv: str,
    max_scenes: int | None = None,
) -> None:
    """Step 1: extract intrinsics for every scene."""
    print("=" * 60)
    print("[Step 1/5] Extracting intrinsics ...")
    scene_names = sorted(os.listdir(input_root))
    if max_scenes is not None and max_scenes > 0:
        scene_names = scene_names[:max_scenes]
        print(f"  (limited to first {len(scene_names)} scene(s) via --max-scenes)")
    for scene_name in tqdm(scene_names, desc="intrinsics"):
        output_json = os.path.join(input_root, scene_name, "_detail", "intrinsics.json")
        extract_intrinsics(camera_params_csv, scene_name, output_json)
    print("[Step 1/5] Done.\n")


# ---------------------------------------------------------------------------
# Step 2: Extrinsics
# ---------------------------------------------------------------------------

def save_extrinsics_per_frame(
    orientation_path: str,
    position_path: str,
    scale_csv: str,
    output_dir: str,
) -> None:
    """Convert per-camera HDF5 orientation/position to per-frame 4x4 extrinsic JSON."""
    os.makedirs(output_dir, exist_ok=True)

    scale_data = pd.read_csv(scale_csv)
    scale_factor = scale_data.loc[0, "parameter_value"]

    with h5py.File(orientation_path, "r") as f_ori, h5py.File(position_path, "r") as f_pos:
        orientations = f_ori[list(f_ori.keys())[0]][:]  # (N, 3, 3)
        positions = f_pos[list(f_pos.keys())[0]][:]      # (N, 3)

        num_frames = orientations.shape[0]
        for i in range(num_frames):
            R = orientations[i]
            t = positions[i] * scale_factor

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t

            # Flip y/z axes to match Open3D coordinate system
            flip = np.array([
                [1, -1, -1, 1],
                [1, -1, -1, 1],
                [1, -1, -1, 1],
                [1,  1,  1, 1],
            ])
            extrinsic = flip * extrinsic

            file_name = f"frame.{i:04d}.extrinsic.json"
            with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
                json.dump(extrinsic.tolist(), f, indent=4)

            # Save 4x4 extrinsic as txt
            txt_name = f"frame.{i:04d}.extrinsic.txt"
            np.savetxt(os.path.join(output_dir, txt_name), extrinsic)


def run_extrinsics(input_root: str, max_scenes: int | None = None) -> None:
    """Step 2: extract extrinsics for every scene / camera."""
    print("=" * 60)
    print("[Step 2/5] Extracting extrinsics ...")
    scenes = sorted(os.listdir(input_root))
    if max_scenes is not None and max_scenes > 0:
        scenes = scenes[:max_scenes]
        print(f"  (limited to first {len(scenes)} scene(s) via --max-scenes)")
    for scene_name in tqdm(scenes, desc="extrinsics"):
        scene_dir = os.path.join(input_root, scene_name, "_detail")
        if not os.path.isdir(scene_dir):
            continue
        cam_dirs = sorted([d for d in os.listdir(scene_dir) if d.startswith("cam_")])
        for cam_dir in cam_dirs:
            orientation_path = os.path.join(scene_dir, cam_dir, "camera_keyframe_orientations.hdf5")
            position_path = os.path.join(scene_dir, cam_dir, "camera_keyframe_positions.hdf5")
            scale_csv = os.path.join(scene_dir, "metadata_scene.csv")
            output_dir = os.path.join(scene_dir, cam_dir, "extrinsics")
            if not os.path.exists(orientation_path) or not os.path.exists(position_path):
                continue
            save_extrinsics_per_frame(orientation_path, position_path, scale_csv, output_dir)
    print("[Step 2/5] Done.\n")


# ---------------------------------------------------------------------------
# Step 3: RGB tonemap
# ---------------------------------------------------------------------------

def tonemap_single_frame(
    in_rgb_hdf5: str, in_render_entity_id_hdf5: str, out_jpg: str
) -> None:
    """Tonemap a single HDR frame to an 8-bit JPG preview image."""
    try:
        with h5py.File(in_rgb_hdf5, "r") as f:
            rgb_color = f["dataset"][:].astype(np.float32)
    except Exception:
        print(f"[tonemap] WARNING: could not load color image: {in_rgb_hdf5}")
        return

    try:
        with h5py.File(in_render_entity_id_hdf5, "r") as f:
            render_entity_id = f["dataset"][:].astype(np.int32)
    except Exception:
        print(f"[tonemap] WARNING: could not load render entity id: {in_render_entity_id_hdf5}")
        return

    assert np.all(render_entity_id != 0)

    gamma = 1.0 / 2.2
    inv_gamma = 1.0 / gamma
    percentile = 90
    brightness_nth_percentile_desired = 0.8

    valid_mask = render_entity_id != -1

    if np.count_nonzero(valid_mask) == 0:
        scale = 1.0
    else:
        brightness = (
            0.3 * rgb_color[:, :, 0]
            + 0.59 * rgb_color[:, :, 1]
            + 0.11 * rgb_color[:, :, 2]
        )
        brightness_valid = brightness[valid_mask]
        eps = 0.0001
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            scale = (
                np.power(brightness_nth_percentile_desired, inv_gamma)
                / brightness_nth_percentile_current
            )

    rgb_color_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
    rgb_color_tm = np.clip(rgb_color_tm, 0, 1)

    # Save as uint8 JPG via cv2 (BGR order)
    img_uint8 = np.clip(rgb_color_tm * 255 + 0.5, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_jpg, img_bgr)


def run_rgb_tonemap(
    input_root: str,
    max_workers: int = 16,
    max_scenes: int | None = None,
) -> None:
    """Step 3: tonemap HDR renders to preview JPGs."""
    print("=" * 60)
    print("[Step 3/5] Tonemapping RGB images ...")
    scene_dirs = sorted(os.listdir(input_root))
    if max_scenes is not None and max_scenes > 0:
        scene_dirs = scene_dirs[:max_scenes]
        print(f"  (limited to first {len(scene_dirs)} scene(s) via --max-scenes)")

    for scene_dir in tqdm(scene_dirs, desc="tonemap"):
        images_dir = os.path.join(input_root, scene_dir, "images")
        if not os.path.isdir(images_dir):
            continue

        hdf5_dir_pattern = os.path.join(images_dir, "scene_*_final_hdf5")
        matching_dirs = glob.glob(hdf5_dir_pattern)

        camera_ids = []
        for d in matching_dirs:
            match = re.search(r"scene_(.*)_final_hdf5", os.path.basename(d))
            if match:
                camera_ids.append(match.group(1))

        for camera_id in camera_ids:
            in_rgb_hdf5_dir = os.path.join(images_dir, f"scene_{camera_id}_final_hdf5")
            in_geometry_dir = os.path.join(images_dir, f"scene_{camera_id}_geometry_hdf5")
            out_preview_dir = os.path.join(images_dir, f"scene_{camera_id}_final_preview")

            if not os.path.isdir(in_rgb_hdf5_dir) or not os.path.isdir(in_geometry_dir):
                continue
            os.makedirs(out_preview_dir, exist_ok=True)

            in_files = sorted(glob.glob(os.path.join(in_rgb_hdf5_dir, "frame.*.color.hdf5")))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for in_file in in_files:
                    file_root = os.path.basename(in_file).replace(".color.hdf5", "")
                    entity_id_file = os.path.join(in_geometry_dir, f"{file_root}.render_entity_id.hdf5")
                    out_jpg = os.path.join(out_preview_dir, f"{file_root}.tonemap.jpg")
                    executor.submit(tonemap_single_frame, in_file, entity_id_file, out_jpg)

    print("[Step 3/5] Done.\n")


# ---------------------------------------------------------------------------
# Step 4: Depth conversion
# ---------------------------------------------------------------------------

def convert_depth_single_frame(
    file_path: str, output_dir: str, scaling_factor: np.ndarray
) -> None:
    """Convert a single Euclidean-distance HDF5 to planar-depth 16-bit PNG."""
    base_name = os.path.basename(file_path).replace(".depth_meters.hdf5", "")
    with h5py.File(file_path, "r") as f:
        key = list(f.keys())[0]
        distance = f[key][:].astype(np.float32)

    depth = distance * scaling_factor
    depth[np.isnan(depth)] = 0
    depth_mm = (depth * 1000).astype(np.uint16)

    output_path = os.path.join(output_dir, f"{base_name}.planar_depth.png")
    cv2.imwrite(output_path, depth_mm)


def convert_distance_to_planar_depth(
    input_pattern: str,
    output_dir: str,
    width: int,
    height: int,
    focal: float,
    max_workers: int = 32,
) -> None:
    """Convert all Euclidean distance HDF5 files matching *input_pattern* to planar depth."""
    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute per-pixel cos(theta) scaling factor
    x = np.linspace(-0.5 * width + 0.5, 0.5 * width - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    y = np.linspace(-0.5 * height + 0.5, 0.5 * height - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    z = np.full([height, width, 1], focal, np.float32)
    imageplane = np.concatenate([x, y, z], axis=2)
    scaling_factor = focal / np.linalg.norm(imageplane, ord=2, axis=2)

    files = sorted(glob.glob(input_pattern))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(convert_depth_single_frame, fp, output_dir, scaling_factor)
            for fp in files
        ]
        concurrent.futures.wait(futures)


def load_intrinsic_json(json_path: str) -> np.ndarray:
    """Load a 3x3 intrinsic matrix from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return np.array(json.load(f))


def run_depth(input_root: str, max_scenes: int | None = None) -> None:
    """Step 4: convert Euclidean distance to planar depth for every scene / camera."""
    print("=" * 60)
    print("[Step 4/5] Converting depth maps ...")
    scene_names = sorted(os.listdir(input_root))
    if max_scenes is not None and max_scenes > 0:
        scene_names = scene_names[:max_scenes]
        print(f"  (limited to first {len(scene_names)} scene(s) via --max-scenes)")

    for scene_name in tqdm(scene_names, desc="depth"):
        intrinsic_path = os.path.join(input_root, scene_name, "_detail", "intrinsics.json")
        if not os.path.exists(intrinsic_path):
            continue
        intrinsic = load_intrinsic_json(intrinsic_path)

        cam_pattern = os.path.join(input_root, scene_name, "images", "scene_*_geometry_hdf5")
        cam_dirs = sorted(glob.glob(cam_pattern))

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir).replace("geometry_hdf5", "depth")
            output_dir = os.path.join(input_root, scene_name, "images", cam_name)
            convert_distance_to_planar_depth(
                input_pattern=os.path.join(cam_dir, "frame.*.depth_meters.hdf5"),
                output_dir=output_dir,
                width=int(intrinsic[0, 2] * 2 + 1),
                height=int(intrinsic[1, 2] * 2 + 1),
                focal=intrinsic[1, 1],
            )
    print("[Step 4/5] Done.\n")


# ---------------------------------------------------------------------------
# Step 5: Parquet generation
# ---------------------------------------------------------------------------

def read_hdf5(filename: str) -> np.ndarray:
    """Read the first dataset from an HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f[list(f.keys())[0]][()]


def extract_and_clean_recursive(text: str) -> str:
    """Extract a human-readable object name from Hypersim mesh object names."""
    text = "0_" + text + "_0"
    text = text.replace("obj", "").replace("Obj", "")
    pattern = r"[\d_]*\d[\d_]*"
    matches = list(re.finditer(pattern, text))

    if len(matches) < 2:
        return ""

    for i in range(len(matches) - 1, 0, -1):
        start_pos = matches[i - 1].end()
        end_pos = matches[i].start()
        content = text[start_pos:end_pos]

        while content and (content[-1].isupper() or content[-1] == "_"):
            content = content[:-1]

        if content:
            return content.lower().replace("_", " ")

    return ""


def process_frame_instances(
    instance_hdf5_path: str,
    semantic_hdf5_path: str,
    id_to_name: dict,
    mask_save_dir: str,
) -> tuple:
    """Extract instance IDs, 2D bboxes, semantic labels, and mask paths for a single frame."""
    os.makedirs(mask_save_dir, exist_ok=True)

    with h5py.File(instance_hdf5_path, "r") as f:
        instance_map = f[list(f.keys())[0]][()]
    with h5py.File(semantic_hdf5_path, "r") as f:
        semantic_map = f[list(f.keys())[0]][()]

    unique_ids = [idx for idx in np.unique(instance_map) if idx >= 0]

    res_instance_ids = []
    res_bboxes = []
    res_label_names = []
    res_mask_paths = []

    for inst_id in unique_ids:
        rows, cols = np.where(instance_map == inst_id)
        if len(rows) == 0:
            continue

        bbox = [int(np.min(cols)), int(np.min(rows)), int(np.max(cols)), int(np.max(rows))]

        nyu40_id = semantic_map[rows[0], cols[0]]
        label_name = id_to_name.get(nyu40_id, "Unknown")
        if label_name in ("blinds", "door", "otherfurniture", "otherprop", "otherstructure", "wall"):
            label_name = "Unknown"

        mask_filename = f"mask_inst_{inst_id}.png"
        mask_path = os.path.join(mask_save_dir, mask_filename)

        res_instance_ids.append(inst_id)
        res_bboxes.append(bbox)
        res_label_names.append(label_name)
        res_mask_paths.append(mask_path)

    return res_instance_ids, res_bboxes, res_label_names, res_mask_paths


def process_single_scene_parquet(
    input_root_dir: str,
    scene_id: str,
    camera_id: str,
    frame_id: str,
    id_to_name: dict,
    name_filter: dict | None,
    mesh_root_dir: str | None = None,
) -> dict | None:
    """Process a single (scene, camera, frame) tuple and return a record dict.

    ``mesh_root_dir`` optionally points to a *separate* root that contains the
    per-scene ``_detail/mesh/`` sub-tree (Hypersim distributes mesh assets as an
    independent archive). When ``None``, falls back to ``input_root_dir``.
    """
    try:
        mesh_root = mesh_root_dir if mesh_root_dir else input_root_dir

        def _mesh_path(*parts: str) -> str:
            """Resolve a ``_detail/mesh/...`` file under ``mesh_root`` with
            per-file fallback to ``input_root_dir``.

            When users keep Hypersim's mesh archive on a separate disk, some
            scenes/files may still live under the raw root (e.g. partial
            downloads). We therefore try ``mesh_root`` first and transparently
            fall back to ``input_root_dir`` whenever the mesh-root copy is
            missing, so the worker degrades gracefully instead of skipping the
            whole frame.
            """
            primary = os.path.abspath(os.path.join(mesh_root, *parts))
            if os.path.exists(primary) or mesh_root == input_root_dir:
                return primary
            fallback = os.path.abspath(os.path.join(input_root_dir, *parts))
            return fallback if os.path.exists(fallback) else primary

        record_id = f"{scene_id}-{camera_id}-{frame_id}"
        image = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "images", f"scene_{camera_id}_final_preview", f"frame.{frame_id}.tonemap.jpg"))
        pose = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "_detail", camera_id, "extrinsics", f"frame.{frame_id}.extrinsic.txt"))
        intrinsic = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "_detail", "intrinsics.txt"))
        depth_map = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "images", f"scene_{camera_id}_depth", f"frame.{frame_id}.planar_depth.png"))

        instance_hdf5 = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "images", f"scene_{camera_id}_geometry_hdf5", f"frame.{frame_id}.semantic_instance.hdf5"))
        semantic_hdf5 = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "images", f"scene_{camera_id}_geometry_hdf5", f"frame.{frame_id}.semantic.hdf5"))
        # Mesh-side assets live under ``mesh_root`` so users can keep them on a
        # separate disk / archive from the rendered RGB/depth data. We fall
        # back to ``input_root_dir`` on a per-file basis via ``_mesh_path``.
        object_label_csv = _mesh_path(
            scene_id, "_detail", "mesh", "metadata_objects.csv")
        sii_hdf5 = _mesh_path(
            scene_id, "_detail", "mesh", "mesh_objects_sii.hdf5")

        extents_path = _mesh_path(
            scene_id, "_detail", "mesh",
            "metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5")
        orientations_path = _mesh_path(
            scene_id, "_detail", "mesh",
            "metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5")
        positions_path = _mesh_path(
            scene_id, "_detail", "mesh",
            "metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5")
        scale_csv = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "_detail", "metadata_scene.csv"))

        required_files = [
            instance_hdf5, semantic_hdf5, object_label_csv, sii_hdf5,
            extents_path, orientations_path, positions_path, scale_csv,
            image, pose, intrinsic, depth_map,
        ]
        for fp in required_files:
            if not os.path.exists(fp):
                print(f"[parquet] Missing file: {fp}, skipping {record_id}.")
                return None

        # Instance / semantic annotations
        mask_save_dir = os.path.abspath(os.path.join(
            input_root_dir, scene_id, "masks", camera_id, frame_id))
        instance_ids, bboxes_2d, label_names, mask_paths = process_frame_instances(
            instance_hdf5, semantic_hdf5, id_to_name, mask_save_dir
        )

        # Instance ID → object label mapping
        with h5py.File(sii_hdf5, "r") as f:
            sii_data = f[list(f.keys())[0]][()]

        df_objects = pd.read_csv(object_label_csv)

        sii_to_name = {}
        for obj_id, inst_id in enumerate(sii_data):
            inst_id = int(inst_id[0]) if isinstance(inst_id, np.ndarray) else int(inst_id)
            if inst_id < 0:
                continue
            if str(inst_id) not in sii_to_name:
                if obj_id < len(df_objects):
                    clean_name = extract_and_clean_recursive(df_objects.iloc[obj_id]["object_name"])
                    sii_to_name[str(inst_id)] = clean_name if clean_name else "Unknown"
                else:
                    sii_to_name[str(inst_id)] = "Unknown"

        obj_names = [sii_to_name.get(str(int(iid)), "Unknown") for iid in instance_ids]

        if name_filter is not None:
            for i in range(len(obj_names)):
                if obj_names[i] not in name_filter or name_filter[obj_names[i]] != "True":
                    obj_names[i] = "Unknown"

        obj_tags_gathered = []
        for i in range(len(label_names)):
            if label_names[i] != "Unknown":
                obj_tags_gathered.append(label_names[i])
            elif obj_names[i] != "Unknown":
                obj_tags_gathered.append(obj_names[i])
            else:
                obj_tags_gathered.append("Unknown")

        # 3D OBB
        scale_data = pd.read_csv(scale_csv)
        scale_factor = scale_data.loc[0, "parameter_value"]
        extents_data = read_hdf5(extents_path)
        orientations_data = read_hdf5(orientations_path)
        positions_data = read_hdf5(positions_path)

        obj_obbs_3d = []
        for inst_id in instance_ids:
            if inst_id >= len(extents_data):
                print(f"[parquet] Instance ID {inst_id} out of bounds for OBB in scene {scene_id}")
                return None
            extent = extents_data[inst_id] * scale_factor
            orientation = orientations_data[inst_id]
            position = positions_data[inst_id] * scale_factor

            if orientation.size == 9:
                orientation = orientation.reshape(3, 3)

            position *= np.array([1, 1, 1])
            flip = np.array([
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
            ])
            orientation = orientation * flip

            euler_zxy = Rotation.from_matrix(orientation).as_euler("zxy", degrees=False)
            obb = np.concatenate([position, extent, euler_zxy])
            obj_obbs_3d.append(obb.tolist())

        return {
            "scene_id": scene_id,
            "id": record_id,
            "image": image,
            "pose": pose,
            "intrinsic": intrinsic,
            "obj_tags": obj_tags_gathered,
            "depth_map": depth_map,
            "bboxes_3d_world_coords": obj_obbs_3d,
            "depth_scale": 1000,
            "is_metric_depth": True,
            "masks": mask_paths,
            "bboxes_2d": bboxes_2d,
            "nyu_tags": label_names,
            "obj_names_detailed": obj_names,
            "instance_ids": instance_ids,
            # Hypersim is already expressed in a metric right-handed camera
            # frame, so no extra axis-alignment rotation is needed. We still
            # emit the field (as None) to match the schema expected by
            # task/group/group.py's default ``group_col_list``.
            "axis_align_matrix": None,
        }
    except Exception as e:
        print(f"[parquet] Error processing {scene_id}: {e}")
        return None


def run_parquet(
    input_root: str,
    output_dir: str,
    labels_tsv: str,
    name_filter_json: str | None,
    chunk_size: int = 1000,
    max_workers: int = 32,
    max_scenes: int | None = None,
    max_tasks: int | None = None,
    mesh_root: str | None = None,
) -> None:
    """Step 5: generate Parquet files from all processed data."""
    print("=" * 60)
    print("[Step 5/5] Generating Parquet files ...")
    os.makedirs(output_dir, exist_ok=True)

    # Enumerate all (scene, camera, frame) tuples
    scene_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    if max_scenes is not None and max_scenes > 0:
        scene_dirs = scene_dirs[:max_scenes]
        print(f"  (limited to first {len(scene_dirs)} scene(s) via --max-scenes)")

    scene_ids, camera_ids, frame_ids = [], [], []
    reached_task_cap = False
    for scene_dir in tqdm(scene_dirs, desc="enumerating frames"):
        if reached_task_cap:
            break
        detail_dir = os.path.join(input_root, scene_dir, "_detail")
        if not os.path.isdir(detail_dir):
            continue
        cam_dirs = [d for d in os.listdir(detail_dir) if d.startswith("cam_")]
        for cam_dir in cam_dirs:
            if reached_task_cap:
                break
            frame_files = glob.glob(os.path.join(
                input_root, scene_dir, "images",
                f"scene_{cam_dir}_final_preview", "frame.*.tonemap.jpg"))
            for frame_file in frame_files:
                frame_idx_str = os.path.basename(frame_file).split(".")[1]
                scene_ids.append(scene_dir)
                camera_ids.append(cam_dir)
                frame_ids.append(frame_idx_str)
                if max_tasks is not None and max_tasks > 0 and len(scene_ids) >= max_tasks:
                    reached_task_cap = True
                    break

    if max_tasks is not None and max_tasks > 0 and len(scene_ids) > max_tasks:
        print(f"  (truncating {len(scene_ids)} → {max_tasks} tasks via --max-tasks)")
        scene_ids = scene_ids[:max_tasks]
        camera_ids = camera_ids[:max_tasks]
        frame_ids = frame_ids[:max_tasks]

    print(f"Found {len(scene_ids)} frames in total.")

    # Load label mapping
    df_labels = pd.read_csv(labels_tsv, sep="\t")
    id_to_name = pd.Series(df_labels.nyu40class.values, index=df_labels.nyu40id).to_dict()

    name_filter = None
    if name_filter_json is not None:
        with open(name_filter_json, "r") as f:
            name_filter = json.load(f)

    # Parallel processing — collect all results first
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_scene_parquet,
                input_root, scene_ids[i], camera_ids[i], frame_ids[i],
                id_to_name, name_filter, mesh_root,
            ): i
            for i in range(len(scene_ids))
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="parquet"):
            result = future.result()
            if result:
                all_results.append(result)

    # Group by scene_id so the same scene never spans two files
    from collections import defaultdict
    scene_groups = defaultdict(list)
    for r in all_results:
        scene_groups[r["scene_id"]].append(r)

    chunk, chunk_count = [], 0
    for scene_id in sorted(scene_groups):
        chunk.extend(scene_groups[scene_id])
        if len(chunk) >= chunk_size:
            save_path = os.path.join(output_dir, f"batch_{chunk_count}.parquet")
            pd.DataFrame(chunk).to_parquet(save_path, engine="pyarrow")
            chunk = []
            chunk_count += 1

    if chunk:
        save_path = os.path.join(output_dir, f"batch_{chunk_count}.parquet")
        pd.DataFrame(chunk).to_parquet(save_path, engine="pyarrow")
        chunk_count += 1

    print(f"[Step 5/5] Done. Generated {chunk_count} Parquet file(s).\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hypersim dataset preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_root", type=str, required=True,
        help="Root directory of raw Hypersim data (e.g. /data/Hypersim).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for Parquet files.",
    )
    parser.add_argument(
        "--camera_params_csv", type=str, required=True,
        help="Path to metadata_camera_parameters.csv.",
    )
    parser.add_argument(
        "--labels_tsv", type=str, required=True,
        help="Path to scannet-labels.combined.tsv (NYU40 label mapping).",
    )
    parser.add_argument(
        "--name_filter_json", type=str, default=None,
        help="Path to Hypersim_name_filter_results.json. If not provided, no tag filtering is applied.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000,
        help="Number of records per Parquet file (default: 1000).",
    )
    parser.add_argument(
        "--max_workers", type=int, default=32,
        help="Max parallel workers for depth conversion and parquet generation (default: 32).",
    )
    parser.add_argument(
        "--tonemap_workers", type=int, default=16,
        help="Max parallel workers for RGB tonemapping (default: 16).",
    )
    parser.add_argument(
        "--max_scenes", type=int, default=None,
        help="Limit number of scenes processed in every step (for smoke testing).",
    )
    parser.add_argument(
        "--max_tasks", type=int, default=None,
        help="Hard-cap on total (scene, camera, frame) tasks in the Parquet step (for smoke testing).",
    )
    parser.add_argument(
        "--mesh_root", type=str, default=None,
        help=(
            "Optional root dir that hosts the per-scene `_detail/mesh/` sub-tree "
            "(Hypersim distributes mesh assets as an independent archive). "
            "Must follow the same `<scene_id>/_detail/mesh/...` layout as "
            "--input_root. If omitted, falls back to --input_root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_intrinsics(args.input_root, args.camera_params_csv, max_scenes=args.max_scenes)
    run_extrinsics(args.input_root, max_scenes=args.max_scenes)
    run_rgb_tonemap(
        args.input_root,
        max_workers=args.tonemap_workers,
        max_scenes=args.max_scenes,
    )
    run_depth(args.input_root, max_scenes=args.max_scenes)
    run_parquet(
        args.input_root,
        args.output_dir,
        args.labels_tsv,
        args.name_filter_json,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        max_scenes=args.max_scenes,
        max_tasks=args.max_tasks,
        mesh_root=args.mesh_root,
    )

    print("=" * 60)
    print("All steps completed successfully.")


if __name__ == "__main__":
    main()
