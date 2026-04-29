"""Inline BLINK-format writer for annotation tasks.

Writes a per-task ``<task>.jsonl`` plus an ``images/<task>/`` folder directly
from an in-memory annotation DataFrame, bypassing parquet entirely.  This is
the default output format for annotation tasks; parquet can still be produced
in parallel when explicitly requested via ``output.format: [parquet, blink]``.

Output layout
-------------

    <blink_root>/
        <task>.jsonl
        images/<task>/<row_idx>_q<qa_idx>_view<k>.png

Record schema (one line per QA, same keys as ``convert_to_blink.py``)
---------------------------------------------------------------------

    {
      "id": "<data_source>_<task>_<idx:06d>",
      "image": ["images/<task>/000000_q0_view0.png", ...],
      "video": [],
      "conversations": [{"from": "human", "value": "..."}, ...],
      "task": "<task>",
      "input_type": "image",
      "output_type": "MCQ" | "open",
      "data_source": "<data_source>",
      "sub_task": "singleview" | "multiview",
      "others": {"question_tags": [...], "question_types": "...",
                 "cognitive_map": {...}}
    }
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
from collections import Counter
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image as PILImage


# ─── Helpers ────────────────────────────────────────────────────────────

class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder tolerant to numpy scalars / arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _to_pil(img) -> Optional[PILImage.Image]:
    """Best-effort conversion of a heterogeneous image field to PIL."""
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        return img
    if isinstance(img, dict):
        data = img.get("bytes")
        if data:
            try:
                return PILImage.open(io.BytesIO(data))
            except Exception:
                return None
        # Some codepaths store {"path": "..."} without bytes.
        path = img.get("path")
        if path and isinstance(path, str) and os.path.exists(path):
            try:
                return PILImage.open(path)
            except Exception:
                return None
        return None
    if isinstance(img, (bytes, bytearray)):
        try:
            return PILImage.open(io.BytesIO(bytes(img)))
        except Exception:
            return None
    if isinstance(img, np.ndarray):
        # Typical HxW or HxWxC uint8 array.
        try:
            return PILImage.fromarray(img)
        except Exception:
            return None
    if isinstance(img, str) and os.path.exists(img):
        try:
            return PILImage.open(img)
        except Exception:
            return None
    return None


def _normalize_qa_images(qa_images_field, num_prompts: int):
    """Normalize a row-level QA_images field into ``list[list[PIL.Image]]``.

    ``num_prompts`` is the number of QA prompts produced for that row.  The
    return is always aligned so that ``out[i]`` is the image list for QA ``i``.
    Missing entries become empty lists.
    """
    if qa_images_field is None:
        return [[] for _ in range(num_prompts)]

    # Unwrap numpy arrays to plain python lists so we can iterate uniformly.
    if isinstance(qa_images_field, np.ndarray):
        qa_images_field = qa_images_field.tolist()

    # Case 1: single image object -> applies to the single QA.
    single = _to_pil(qa_images_field)
    if single is not None:
        return [[single]]

    if not isinstance(qa_images_field, (list, tuple)):
        # Unknown type; give up silently.
        return [[] for _ in range(num_prompts)]

    out: List[List[PILImage.Image]] = []
    for entry in qa_images_field:
        if isinstance(entry, np.ndarray) and entry.dtype == object:
            entry = entry.tolist()

        if isinstance(entry, (list, tuple)):
            # multiview: each QA is itself a list of view images.
            pil_views = [p for p in (_to_pil(v) for v in entry) if p is not None]
            out.append(pil_views)
        else:
            pil = _to_pil(entry)
            out.append([pil] if pil is not None else [])

    # Pad / truncate to num_prompts so we stay aligned with messages.
    if len(out) < num_prompts:
        out.extend([[] for _ in range(num_prompts - len(out))])
    elif len(out) > num_prompts:
        out = out[:num_prompts]
    return out


def _split_messages_per_qa(messages):
    """Split a flat messages list into per-QA 2-message chunks.

    Annotation tasks already produce one (human, gpt) pair per QA, so this
    just pairs consecutive entries.  Extra unpaired trailing messages are
    dropped (should not happen in practice).
    """
    if messages is None:
        return []
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if not isinstance(messages, (list, tuple)) or len(messages) == 0:
        return []

    # Case A (new in-memory format from apply_transform):
    # ``messages`` is ``list[list[dict]]`` — one inner list per QA, each
    # already holding a [human, gpt, (human, gpt)...] chat.  We keep each
    # inner list as-is (it may be multi-turn).
    first = messages[0]
    if isinstance(first, np.ndarray):
        first = first.tolist()
    if isinstance(first, (list, tuple)):
        pairs = []
        for qa in messages:
            if isinstance(qa, np.ndarray):
                qa = qa.tolist()
            if isinstance(qa, (list, tuple)) and len(qa) >= 2:
                pairs.append(list(qa))
        return pairs

    # Case B (legacy flat format, e.g. read back from parquet):
    # ``messages`` is ``[human, gpt, human, gpt, ...]`` — pair consecutive
    # entries into QAs.
    pairs = []
    i = 0
    while i + 1 < len(messages):
        pairs.append([messages[i], messages[i + 1]])
        i += 2
    return pairs


def _strip_image_tags(messages):
    """Produce BLINK-style conversations: drop ``<image>`` markers from text."""
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("from", "")
        value = msg.get("value", "")
        if role == "human":
            value = re.sub(r"<image>\s*", "", value).strip()
        cleaned.append({"from": role, "value": value})
    return cleaned


_MCQ_RE = re.compile(r"\b[A-D]\.\s|\([A-D]\)")


def _infer_output_type(conversations):
    text = ""
    for msg in conversations:
        if msg.get("from") == "human":
            text += msg.get("value", "")
    return "MCQ" if _MCQ_RE.search(text) else "open"


def _classify_task(task_name: str) -> str:
    for p in ("multiview_", "mmsi_"):
        if task_name.startswith(p):
            return "multiview"
    return "singleview"


# ─── Main writer ────────────────────────────────────────────────────────

class BlinkWriter:
    """Serialize annotation output directly to BLINK jsonl + images folder.

    One instance is safe to reuse across tasks; a per-file lock guards the
    jsonl append so parallel task runs don't interleave lines.
    """

    def __init__(self, blink_root: str, data_source: str = "OpenSpatial"):
        self.blink_root = os.path.abspath(blink_root)
        self.data_source = data_source
        self._locks = {}
        self._locks_guard = threading.Lock()
        os.makedirs(self.blink_root, exist_ok=True)

    # Public API ---------------------------------------------------------

    def write(self, task_name: str, data: pd.DataFrame) -> dict:
        """Flatten ``data`` into per-QA records and dump them to disk.

        Returns a small stats dict for logging.
        """
        if data is None or len(data) == 0:
            print(f"  [blink] {task_name}: empty DataFrame, nothing to write")
            return {"total": 0, "converted": 0, "image_errors": 0}

        images_dir = os.path.join(self.blink_root, "images", task_name)
        jsonl_path = os.path.join(self.blink_root, f"{task_name}.jsonl")
        os.makedirs(images_dir, exist_ok=True)

        # Truncate the target jsonl so repeated runs don't accumulate.
        open(jsonl_path, "w", encoding="utf-8").close()

        # ── Diagnostics: DataFrame overview ────────────────────────────
        print(f"  [blink] {task_name}: rows={len(data)}  blink_root={self.blink_root}")
        cols = list(data.columns)
        print(f"  [blink] {task_name}: columns={cols}")
        if len(data) > 0:
            first = data.iloc[0]
            qa_field = first.get("QA_images") if "QA_images" in cols else None
            msg_field = first.get("messages") if "messages" in cols else None
            qa_type = type(qa_field).__name__
            qa_len = len(qa_field) if hasattr(qa_field, "__len__") else "-"
            msg_type = type(msg_field).__name__
            msg_len = len(msg_field) if hasattr(msg_field, "__len__") else "-"
            print(f"  [blink] {task_name}: row0 QA_images type={qa_type} "
                  f"len={qa_len}  messages type={msg_type} len={msg_len}")
            # Peek the innermost type of QA_images for troubleshooting.
            if isinstance(qa_field, (list, tuple)) and len(qa_field) > 0:
                inner = qa_field[0]
                inner_type = type(inner).__name__
                inner_len = len(inner) if hasattr(inner, "__len__") else "-"
                print(f"  [blink] {task_name}: row0 QA_images[0] "
                      f"type={inner_type} len={inner_len}")

        stats = {"total": 0, "converted": 0, "image_errors": 0}
        # Why did pil_views end up empty?  Track reasons for the log.
        empty_reasons = Counter()
        task_type = _classify_task(task_name)
        lock = self._get_lock(jsonl_path)

        flat_idx = 0
        # Track per-dataset rotation stats for a concise summary log.
        rotated_count = 0
        # One-shot debug: print the 'dataset' value as seen on the first row so
        # we can tell why rotate-cw90 did or didn't fire for ARKitScenes.
        debug_printed = False
        for row_idx in range(len(data)):
            row = data.iloc[row_idx].to_dict()
            qa_pairs = _split_messages_per_qa(row.get("messages"))
            if not qa_pairs:
                empty_reasons["no_qa_pairs"] += 1
                continue

            raw_qa_images = row.get("QA_images")
            qa_images = _normalize_qa_images(raw_qa_images,
                                             num_prompts=len(qa_pairs))
            tags_field = row.get("question_tags")
            types_field = row.get("question_types")
            cogs_field = row.get("cognitive_maps")

            tags_list = self._as_list(tags_field, len(qa_pairs))
            types_list = self._as_list(types_field, len(qa_pairs))
            cogs_list = self._as_list(cogs_field, len(qa_pairs))

            # ── Dataset-specific image orientation fix ─────────────────
            # ARKitScenes' lowres_wide frames are stored sideways on disk
            # (camera is held in portrait but frames are saved landscape).
            # We ONLY rotate here, at the moment we materialize the BLINK
            # QA images — the upstream DataFrame (and any other writer)
            # keeps the raw, unrotated bytes intact.
            needs_rotate_cw90 = self._row_is_arkitscenes(row)

            if not debug_printed:
                ds_raw = row.get("dataset")
                print(f"  [blink] {task_name}: row0 dataset value="
                      f"{ds_raw!r} type={type(ds_raw).__name__}  "
                      f"-> rotate_cw90={needs_rotate_cw90}", flush=True)
                debug_printed = True

            for qa_idx, msg_pair in enumerate(qa_pairs):
                stats["total"] += 1

                # Dump images for this QA.
                pil_views = qa_images[qa_idx]
                rel_paths = []
                if not pil_views:
                    stats["image_errors"] += 1
                    # Classify why we got nothing.
                    if raw_qa_images is None:
                        empty_reasons["qa_images_field_is_None"] += 1
                    elif isinstance(raw_qa_images, (list, tuple, np.ndarray)) \
                            and len(raw_qa_images) == 0:
                        empty_reasons["qa_images_field_empty"] += 1
                    else:
                        empty_reasons["decode_failed_or_unknown_type"] += 1
                for view_idx, pil in enumerate(pil_views):
                    filename = f"{flat_idx:06d}_view{view_idx}.png"
                    abs_path = os.path.join(images_dir, filename)
                    pil_to_save = pil
                    if needs_rotate_cw90:
                        try:
                            # PIL.ROTATE_270 == rotate 270° counter-clockwise
                            # == rotate 90° CLOCKWISE (which is what we want
                            # for ARKitScenes lowres_wide frames).
                            pil_to_save = pil.transpose(PILImage.ROTATE_270)
                            rotated_count += 1
                        except Exception as exc:
                            print(f"  [blink] {task_name} row={row_idx} "
                                  f"qa={qa_idx} view={view_idx}: rotate "
                                  f"failed ({exc}); saving original")
                            pil_to_save = pil
                    try:
                        pil_to_save.save(abs_path, format="PNG")
                    except Exception as exc:
                        print(f"  [blink] {task_name} row={row_idx} qa={qa_idx} "
                              f"view={view_idx}: image save failed ({exc})")
                        stats["image_errors"] += 1
                        continue
                    rel_paths.append(
                        os.path.join("images", task_name, filename))

                conversations = _strip_image_tags(msg_pair)
                output_type = _infer_output_type(conversations)

                others = {
                    "question_tags": tags_list[qa_idx] if qa_idx < len(tags_list) else [],
                    "question_types": types_list[qa_idx] if qa_idx < len(types_list) else "",
                }
                cog = cogs_list[qa_idx] if qa_idx < len(cogs_list) else None
                if cog is not None:
                    others["cognitive_map"] = cog

                record = {
                    "id": f"{self.data_source}_{task_name}_{flat_idx:06d}",
                    "image": rel_paths,
                    "video": [],
                    "conversations": conversations,
                    "task": task_name,
                    "input_type": "image",
                    "output_type": output_type,
                    "data_source": self.data_source,
                    "sub_task": task_type,
                    "others": others,
                }

                with lock, open(jsonl_path, "a", encoding="utf-8") as fp:
                    json.dump(record, fp, ensure_ascii=False,
                              cls=_NumpyJSONEncoder)
                    fp.write("\n")
                stats["converted"] += 1
                flat_idx += 1

        # Count real image files on disk for a sanity-check.
        try:
            n_files = sum(1 for _ in os.scandir(images_dir)
                          if _.name.endswith(".png"))
        except FileNotFoundError:
            n_files = 0

        print(f"  [blink] {task_name}: converted={stats['converted']}/"
              f"{stats['total']} image_errors={stats['image_errors']} "
              f"png_on_disk={n_files}  → {jsonl_path}")
        if rotated_count > 0:
            print(f"  [blink] {task_name}: arkitscenes rotate_cw90 applied "
                  f"to {rotated_count} image(s)")
        if empty_reasons:
            reasons_str = ", ".join(f"{k}={v}" for k, v in empty_reasons.most_common())
            print(f"  [blink] {task_name}: empty-image breakdown: {reasons_str}")
        return stats

    # Internals ----------------------------------------------------------

    def _get_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @staticmethod
    def _row_is_arkitscenes(row: dict) -> bool:
        """Return True if this annotation row comes from ARKitScenes.

        We check the ``dataset`` column (expected per-row string). For
        safety we also handle list/array values (take the first entry)
        and do a case-insensitive match on the ``arkitscenes`` prefix.
        """
        ds = row.get("dataset")
        if ds is None:
            return False
        # Sometimes per-row fields are stored as a numpy array / list;
        # ARKitScenes scenes are a single dataset per row but be defensive.
        if isinstance(ds, np.ndarray):
            ds = ds.tolist()
        if isinstance(ds, (list, tuple)):
            ds = ds[0] if len(ds) > 0 else None
        if not isinstance(ds, str):
            return False
        return ds.strip().lower().startswith("arkitscenes")

    @staticmethod
    def _as_list(field, n: int):
        """Coerce a row-level field into a per-QA list of length ``n``."""
        if field is None:
            return [None] * n
        if isinstance(field, np.ndarray):
            field = field.tolist()
        if isinstance(field, (list, tuple)):
            field = list(field)
            if len(field) < n:
                field = field + [None] * (n - len(field))
            return field
        # Scalar shared across QAs.
        return [field] * n
