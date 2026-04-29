#!/usr/bin/env python3
"""Sanity check the OpenSpatial preprocessing pipeline output.

Walk through ``<PIPELINE_OUT_DIR>/<run_name>/part_*`` directories produced
by ``run.py``, read every ``data.parquet`` under each stage, and print:

    1. Funnel table: # samples surviving after each stage, per part.
    2. Per-stage schema check: required columns must be present.
    3. obj_tags top-K distribution: helps verify filter_tags actually removed
       wall/floor/ceiling/object before SAM2 refinement.
    4. Warnings for empty parquets, missing stages, schema anomalies, or
       suspicious filter ratios.

Usage:
    python tools/inspect/check_pipeline.py \
        --pipeline-out output/embodiedscan_run/03_pipeline \
        --run-name     base_pipeline_preprocessing_embodiedscan

    # Or auto-detect the single run directory inside pipeline-out:
    python tools/inspect/check_pipeline.py \
        --pipeline-out output/embodiedscan_run/03_pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------------------------------------------------------
# Stage specification: ordered list describing the expected funnel.
# Each entry: (stage_dir, method_dir, required_columns)
# The script will auto-skip stages that don't exist in a given part.
# ----------------------------------------------------------------------------
STAGE_SPEC: List[Tuple[str, str, List[str]]] = [
    ("flatten_stage",      "flatten",                ["image", "obj_tags", "bboxes_3d_world_coords"]),
    ("filter_stage",       "3dbox_filter",           ["image", "obj_tags", "masks"]),
    ("localization_stage", "sam2_refiner",           ["image", "obj_tags", "masks", "bboxes_2d"]),
    ("scene_fusion_stage", "depth_back_projection",  ["image", "obj_tags", "masks", "pointclouds"]),
    ("group_stage",        "group",                  ["image", "obj_tags"]),
]

TAG_TOPK = 15
FILTER_RATIO_WARN = 0.02   # warn if <2 % samples survive a single stage
FILTER_RATIO_STRONG_WARN = 0.005  # error-level warn if <0.5 % survive


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def _autodetect_run_name(pipeline_out: Path) -> str:
    """If run-name is not given, pick the unique subdirectory."""
    candidates = [p for p in pipeline_out.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {pipeline_out}")
    if len(candidates) > 1:
        names = ", ".join(sorted(p.name for p in candidates))
        raise ValueError(
            f"Multiple run directories under {pipeline_out}: {names}. "
            f"Please specify --run-name."
        )
    return candidates[0].name


def _list_parts(run_dir: Path) -> List[Path]:
    """List part_* subdirectories, sorted by numeric suffix."""
    parts = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("part_")]
    if not parts:
        # Single-parquet run: pipeline writes stages directly under run_dir.
        if any((run_dir / s).exists() for s, _, _ in STAGE_SPEC):
            return [run_dir]
        raise FileNotFoundError(f"No part_* directories (and no stages) under {run_dir}")
    parts.sort(key=lambda p: int(p.name.split("_")[-1]))
    return parts


def _safe_read_parquet(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return (df, err). df is None on error."""
    try:
        return pd.read_parquet(path), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _count_tags(df: pd.DataFrame) -> Counter:
    """Flatten obj_tags column (list-per-row) into a Counter."""
    counter: Counter = Counter()
    if "obj_tags" not in df.columns:
        return counter
    for tags in df["obj_tags"].dropna():
        # obj_tags may be np.ndarray, list, or a nested list (grouped rows).
        try:
            if len(tags) == 0:
                continue
            if hasattr(tags[0], "__iter__") and not isinstance(tags[0], str):
                # Grouped: list of lists of tags → flatten one level.
                for sub in tags:
                    counter.update(sub)
            else:
                counter.update(tags)
        except TypeError:
            continue
    return counter


def _fmt_int(n: Optional[int]) -> str:
    if n is None:
        return "   —"
    return f"{n:>6d}"


# ----------------------------------------------------------------------------
# Per-part inspection
# ----------------------------------------------------------------------------
def inspect_part(part_dir: Path) -> Dict:
    """Collect stage-wise statistics for one part directory."""
    result: Dict = {
        "part": part_dir.name,
        "path": str(part_dir),
        "stages": [],
        "warnings": [],
    }

    prev_n: Optional[int] = None
    for stage, method, required_cols in STAGE_SPEC:
        parquet_path = part_dir / stage / method / "data.parquet"
        stage_info: Dict = {
            "stage": stage,
            "method": method,
            "parquet": str(parquet_path),
            "exists": parquet_path.exists(),
            "n_rows": None,
            "columns": None,
            "missing_cols": None,
            "tag_top": None,
            "survival": None,
            "error": None,
        }

        if not parquet_path.exists():
            # Missing stage directory: mark as skipped, don't warn (could be
            # intentional, e.g. embodiedscan has no flatten / group stage).
            result["stages"].append(stage_info)
            continue

        df, err = _safe_read_parquet(parquet_path)
        if err is not None:
            stage_info["error"] = err
            result["warnings"].append(f"[{stage}/{method}] read error: {err}")
            result["stages"].append(stage_info)
            continue

        n_rows = len(df)
        stage_info["n_rows"] = n_rows
        stage_info["columns"] = list(df.columns)
        stage_info["missing_cols"] = [c for c in required_cols if c not in df.columns]

        if n_rows == 0:
            result["warnings"].append(f"[{stage}/{method}] parquet is EMPTY (0 rows)")
        if stage_info["missing_cols"]:
            result["warnings"].append(
                f"[{stage}/{method}] missing columns: {stage_info['missing_cols']}"
            )

        # Survival ratio vs previous stage
        if prev_n is not None and prev_n > 0:
            ratio = n_rows / prev_n
            stage_info["survival"] = ratio
            if ratio < FILTER_RATIO_STRONG_WARN:
                result["warnings"].append(
                    f"[{stage}/{method}] survival {ratio:.2%} (<0.5 %) vs previous stage"
                )
            elif ratio < FILTER_RATIO_WARN:
                result["warnings"].append(
                    f"[{stage}/{method}] survival {ratio:.2%} (<2 %) vs previous stage"
                )

        # obj_tags distribution (only after filter, to see what survives)
        if stage in ("filter_stage", "localization_stage", "scene_fusion_stage", "group_stage"):
            counter = _count_tags(df)
            if counter:
                stage_info["tag_top"] = counter.most_common(TAG_TOPK)
                bad_tags = {"wall", "floor", "ceiling", "object"}
                leaked = {t: c for t, c in counter.items() if t.lower() in bad_tags}
                if leaked and stage != "flatten_stage":
                    result["warnings"].append(
                        f"[{stage}/{method}] filter_tags leaked through: {leaked}"
                    )

        prev_n = n_rows
        result["stages"].append(stage_info)

    return result


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def print_funnel_table(parts_report: List[Dict]) -> None:
    """Print a compact ASCII funnel: rows = parts, cols = stages."""
    stage_names = [f"{s}/{m}" for s, m, _ in STAGE_SPEC]
    header = f"{'part':<10}" + "".join(f"{name:>26}" for name in stage_names)
    print(header)
    print("-" * len(header))
    for rep in parts_report:
        row = f"{rep['part']:<10}"
        for stage_info in rep["stages"]:
            if not stage_info["exists"]:
                cell = "   skipped"
            elif stage_info["error"]:
                cell = "   ERROR"
            else:
                n = stage_info["n_rows"]
                surv = stage_info["survival"]
                if surv is None:
                    cell = _fmt_int(n).strip()
                else:
                    cell = f"{n} ({surv:.1%})"
            row += f"{cell:>26}"
        print(row)
    print()


def print_tag_distribution(parts_report: List[Dict], stage_name: str) -> None:
    """Aggregate tag distribution across parts for a given stage."""
    total: Counter = Counter()
    for rep in parts_report:
        for stage_info in rep["stages"]:
            if stage_info.get("stage") != stage_name:
                continue
            if stage_info.get("tag_top"):
                for tag, cnt in stage_info["tag_top"]:
                    total[tag] += cnt
    if not total:
        return
    print(f"Top-{TAG_TOPK} obj_tags after <{stage_name}>:")
    for tag, cnt in total.most_common(TAG_TOPK):
        print(f"  {cnt:>8d}  {tag}")
    print()


def print_warnings(parts_report: List[Dict]) -> int:
    n_warn = 0
    for rep in parts_report:
        if not rep["warnings"]:
            continue
        print(f"[WARN] {rep['part']}:")
        for w in rep["warnings"]:
            print(f"  - {w}")
            n_warn += 1
    if n_warn == 0:
        print("[OK] No warnings.")
    else:
        print(f"[WARN] {n_warn} warning(s) total.")
    print()
    return n_warn


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline-out", type=Path, required=True,
                   help="Preprocessing output root, e.g. output/embodiedscan_run/03_pipeline")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run subdir name (default: auto-detect if unique).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional: dump per-part stats as JSON.")
    p.add_argument("--fail-on-warn", action="store_true",
                   help="Exit with code 2 if any warnings were raised.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    pipeline_out: Path = args.pipeline_out.resolve()
    if not pipeline_out.is_dir():
        print(f"[FAIL] pipeline-out not found: {pipeline_out}", file=sys.stderr)
        return 1

    run_name = args.run_name or _autodetect_run_name(pipeline_out)
    run_dir = pipeline_out / run_name
    if not run_dir.is_dir():
        print(f"[FAIL] run dir not found: {run_dir}", file=sys.stderr)
        return 1

    parts = _list_parts(run_dir)
    print(f"Scanning {len(parts)} part(s) under {run_dir}\n")

    parts_report = [inspect_part(p) for p in parts]

    # 1) funnel table
    print("=" * 80)
    print("Funnel table (rows = parts, cols = stage/method, cell = n_rows (survival%))")
    print("=" * 80)
    print_funnel_table(parts_report)

    # 2) tag distribution after filter and after back-projection
    print("=" * 80)
    print("Tag distribution")
    print("=" * 80)
    print_tag_distribution(parts_report, "filter_stage")
    print_tag_distribution(parts_report, "scene_fusion_stage")

    # 3) warnings
    print("=" * 80)
    print("Warnings")
    print("=" * 80)
    n_warn = print_warnings(parts_report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(parts_report, f, indent=2, default=str)
        print(f"[JSON] wrote {args.json_out}")

    if args.fail_on_warn and n_warn > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
