"""Data processing utilities: tag filtering, mask/box merging, annotation flattening."""

import numpy as np
import pandas as pd
import tqdm


def filter_color_tags(tags):
    """Filter out color-named tags from a bilingual (en, cn) tag list.

    Args:
        tags: [en_tags_list, cn_tags_list]

    Returns:
        Filtered list of English tags (excluding color names).
    """
    color_en = ['red', 'green', 'blue', 'yellow', 'black', 'white',
                'orange', 'purple', 'brown', 'gray', 'pink', 'gold', 'silver']
    new_tags = []
    for i, (en, cn) in enumerate(zip(tags[0], tags[1])):
        if not (cn.endswith('色') or any(color in en.lower() for color in color_en)):
            new_tags.append(tags[0][i])
    return new_tags


def merge_overlapping_masks(masks, obj_tags, boxes=None, overlap_threshold=0.8):
    """Merge same-tag masks that overlap significantly, keeping the larger one.

    Args:
        masks: np.ndarray of binary masks.
        obj_tags: list of tag strings.
        boxes: optional np.ndarray of bounding boxes.
        overlap_threshold: IoU threshold for merging.

    Returns:
        (filtered_masks, filtered_tags) or (filtered_masks, filtered_tags, filtered_boxes).
    """
    if len(masks) == 0:
        return masks

    tag_to_indices = {}
    for idx, tag in enumerate(obj_tags):
        tag_to_indices.setdefault(tag, []).append(idx)
    keep_indices = set()
    for tag, indices in tag_to_indices.items():
        if len(indices) == 1:
            keep_indices.add(indices[0])
        else:
            mask_areas = [masks[i].sum() for i in indices]
            to_remove = set()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    m1 = masks[indices[i]].astype(bool)
                    m2 = masks[indices[j]].astype(bool)
                    intersection = np.logical_and(m1, m2).sum()
                    area_i = mask_areas[i]
                    area_j = mask_areas[j]
                    if (intersection / (area_i + 1e-6) > 0.8) or \
                       (intersection / (area_j + 1e-6) > overlap_threshold):
                        if area_i >= area_j:
                            to_remove.add(indices[j])
                        else:
                            to_remove.add(indices[i])
            for idx in indices:
                if idx not in to_remove:
                    keep_indices.add(idx)

    keep_indices = sorted(list(keep_indices))
    masks = masks[keep_indices]
    outputs = [masks]
    obj_tags = [obj_tags[i] for i in keep_indices]
    outputs.append(obj_tags)
    if boxes is not None:
        boxes = boxes[keep_indices]
        outputs.append(boxes)
    return outputs


def merge_overlapping_boxes(obj_tags, boxes, overlap_threshold=0.8):
    """Merge same-tag 2D boxes that overlap significantly, keeping the larger one.

    Args:
        obj_tags: list of tag strings.
        boxes: np.ndarray of [x1, y1, x2, y2] boxes.
        overlap_threshold: IoU threshold for merging.

    Returns:
        (filtered_boxes, filtered_tags).
    """
    tag_to_indices = {}
    for idx, tag in enumerate(obj_tags):
        tag_to_indices.setdefault(tag, []).append(idx)
    keep_indices = set()
    for tag, indices in tag_to_indices.items():
        if len(indices) == 1:
            keep_indices.add(indices[0])
        else:
            box_areas = [
                (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                for i in indices
            ]
            to_remove = set()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    box1 = boxes[indices[i]]
                    box2 = boxes[indices[j]]
                    xA = max(box1[0], box2[0])
                    yA = max(box1[1], box2[1])
                    xB = min(box1[2], box2[2])
                    yB = min(box1[3], box2[3])
                    inter_w = max(0, xB - xA)
                    inter_h = max(0, yB - yA)
                    inter_area = inter_w * inter_h
                    area_i = box_areas[i]
                    area_j = box_areas[j]
                    min_area = min(area_i, area_j)
                    if min_area > 0 and inter_area / (min_area + 1e-6) > overlap_threshold:
                        if area_i >= area_j:
                            to_remove.add(indices[j])
                        else:
                            to_remove.add(indices[i])
            for idx in indices:
                if idx not in to_remove:
                    keep_indices.add(idx)
    keep_indices = sorted(list(keep_indices))
    boxes = boxes[keep_indices]
    obj_tags = [obj_tags[i] for i in keep_indices]
    return boxes, obj_tags


def flatten_annotations(data, keep_keys):
    """Flatten multi-QA annotation rows into one row per QA pair.

    Args:
        data: pd.DataFrame with list-valued columns.
        keep_keys: list of column names to flatten.

    Returns:
        pd.DataFrame with one row per QA.
    """
    new_data = []
    for idx, _ in tqdm.tqdm(data.iterrows(), total=len(data), desc="Flatten annotation"):
        example = data.iloc[idx].copy()
        # Separate keys that exist (and are list-valued) from those that are
        # missing or not iterable.  Missing keys are filled with None later.
        present_keys = []
        absent_keys = []
        for key in keep_keys:
            if key in example and isinstance(example[key], (list, tuple, np.ndarray)) and len(example[key]) > 0:
                present_keys.append(key)
            else:
                absent_keys.append(key)

        if not present_keys:
            continue  # nothing to flatten for this row

        lengths = [len(example[key]) for key in present_keys]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                f"Inconsistent lengths found in keep_keys (row {idx}): "
                + ", ".join(f"{k}={l}" for k, l in zip(present_keys, lengths))
            )

        for i in range(lengths[0]):
            new_example = {}
            for key in present_keys:
                new_example[key] = example[key][i]
            for key in absent_keys:
                new_example[key] = None
            new_example["image"] = example["image"]
            new_data.append(new_example)
    return pd.DataFrame(new_data)
