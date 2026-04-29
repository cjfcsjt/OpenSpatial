#!/usr/bin/env python3
"""Export RGB + SAM2-refined-mask overlays as PNG for visual sanity check.

Pick N random rows from a stage parquet (default: after sam2_refiner) and
render each row as a side-by-side panel:

    [ RGB | RGB + mask overlay + 2D box + tag label ]

Output goes to ``<out-dir>/<basename>_<idx>.png``. All rendering is done with
matplotlib / PIL / numpy only — no GUI, no web server.

Usage:
    # Visualize 20 random frames from SAM2 output:
    python tools/inspect/vis_masks.py \
        --parquet  output/embodiedscan_run/03_pipeline/base_pipeline_preprocessing_embodiedscan/part_1/localization_stage/sam2_refiner/data.parquet \
        --out-dir  output/embodiedscan_run/_vis/sam2_part1 \
        --num 20 \
        --data-root /path/to/EmbodiedScan/data

    # Or visualize after 3dbox_filter (coarse masks, before SAM2):
    python tools/inspect/vis_masks.py \
        --parquet .../filter_stage/3dbox_filter/data.parquet \
        --out-dir .../_vis/filter \
        --num 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ImportError as exc:  # noqa: BLE001
    print(f"[FAIL] matplotlib is required: {exc}", file=sys.stderr)
    sys.exit(1)


# Deterministic tab20 palette (20 distinct colors, reused if more objects).
_PALETTE = plt.get_cmap("tab20").colors


# ----------------------------------------------------------------------------
# Path resolution (parquet stores either absolute paths, or paths relative to
# dataset.data_root). Resolve against --data-root when given.
# ----------------------------------------------------------------------------
def _resolve(path_like, data_root: Optional[Path]) -> Path:
    if isinstance(path_like, dict) and "path" in path_like:
        path_like = path_like["path"]
    p = Path(str(path_like))
    if p.is_absolute() or data_root is None:
        return p
    return data_root / p


# ----------------------------------------------------------------------------
# Mask loader: accepts file-path or {"bytes": ...} dict (same as pipeline code)
# ----------------------------------------------------------------------------
def _load_mask(item, data_root: Optional[Path]) -> np.ndarray:
    import io

    if isinstance(item, dict) and "bytes" in item:
        return np.array(Image.open(io.BytesIO(item["bytes"])))
    return np.array(Image.open(_resolve(item, data_root)))


# ----------------------------------------------------------------------------
# Draw one sample
# ----------------------------------------------------------------------------
def render_sample(
    row: pd.Series,
    out_path: Path,
    data_root: Optional[Path],
    max_objs: int = 20,
) -> None:
    """Render a single flat (per-image) row as RGB | RGB+overlay panel."""
    img_path = _resolve(row["image"], data_root)
    if not img_path.exists():
        print(f"[skip] image not found: {img_path}")
        return

    rgb = np.array(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]

    tags: List[str] = list(row.get("obj_tags") or [])
    masks_raw = list(row.get("masks") or [])
    bboxes = list(row.get("bboxes_2d") or [])

    if len(tags) == 0 or len(masks_raw) == 0:
        print(f"[skip] empty tags/masks for {img_path}")
        return

    n = min(len(tags), len(masks_raw), max_objs)

    # Build overlay: RGBA image, one color per object
    overlay = rgb.astype(np.float32).copy()
    for i in range(n):
        try:
            m = _load_mask(masks_raw[i], data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cannot load mask {i} for {img_path}: {exc}")
            continue
        if m.shape[:2] != (H, W):
            m = np.array(Image.fromarray(m).resize((W, H), resample=Image.NEAREST))
        m_bool = m > 0
        if not m_bool.any():
            continue
        color = np.array(_PALETTE[i % len(_PALETTE)]) * 255.0
        overlay[m_bool] = 0.5 * overlay[m_bool] + 0.5 * color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB  ({img_path.name})")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"masks + boxes  (n={n})")
    axes[1].axis("off")

    # Draw 2D boxes + tag labels on the right panel
    for i in range(n):
        if i >= len(bboxes):
            break
        try:
            x1, y1, x2, y2 = [float(v) for v in bboxes[i]]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        color = _PALETTE[i % len(_PALETTE)]
        axes[1].add_patch(
            mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=1.5)
        )
        tag = tags[i] if i < len(tags) else "?"
        axes[1].text(
            x1, max(y1 - 4, 8), tag,
            fontsize=8, color="black",
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def pick_rows(df: pd.DataFrame, num: int, seed: int) -> pd.DataFrame:
    if num >= len(df):
        return df
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), num)
    indices.sort()
    return df.iloc[indices].reset_index(drop=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", type=Path, required=True,
                   help="Path to a stage data.parquet (flat / per-image rows).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for overlay PNGs.")
    p.add_argument("--num", type=int, default=20,
                   help="Number of random rows to render (default: 20).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Optional data root to resolve relative image/mask paths "
                        "(e.g. EMBODIEDSCAN_DATA).")
    p.add_argument("--max-objs", type=int, default=20,
                   help="Max objects per frame (default: 20).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if not args.parquet.is_file():
        print(f"[FAIL] parquet not found: {args.parquet}", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.parquet)
    if len(df) == 0:
        print(f"[FAIL] parquet is empty: {args.parquet}", file=sys.stderr)
        return 1

    # Reject grouped (per-scene) parquets: 'image' column must be a scalar path.
    first_img = df["image"].iloc[0] if "image" in df.columns else None
    if isinstance(first_img, (list, np.ndarray)):
        print(
            "[FAIL] this parquet looks grouped (image column is a list). "
            "vis_masks.py only supports per-image (flat) parquets.",
            file=sys.stderr,
        )
        return 1

    picked = pick_rows(df, args.num, args.seed)
    print(f"Rendering {len(picked)} sample(s) from {args.parquet}")
    print(f"  → {args.out_dir}")

    basename = args.parquet.parent.parent.name  # stage/method/data.parquet
    for i, (_, row) in enumerate(picked.iterrows()):
        out_path = args.out_dir / f"{basename}_{i:03d}.png"
        try:
            render_sample(row, out_path, args.data_root, args.max_objs)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] render failed for row {i}: {exc}")

    print(f"[OK] done. {len(list(args.out_dir.glob('*.png')))} PNG(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
