#!/usr/bin/env bash
# =============================================================================
# unzip_arkitscenes.sh
#
# Unzip ARKitScenes raw data into the directory layout expected by OpenSpatial:
#   <OUTPUT_ROOT>/arkitscenes/<scene_id>/<scene_id>_frames/lowres_wide/
#   <OUTPUT_ROOT>/arkitscenes/<scene_id>/<scene_id>_frames/lowres_wide_intrinsics/
#   <OUTPUT_ROOT>/arkitscenes/<scene_id>/<scene_id>_frames/lowres_depth/
#
# Only the three zips required by the preprocessing pipeline are extracted;
# highres_depth / ultrawide / vga_wide / confidence are intentionally skipped.
#
# Usage:
#   bash unzip_arkitscenes.sh [RAW_ROOT] [OUTPUT_ROOT] [WORKERS]
#
#   RAW_ROOT    – directory containing Training/ and Validation/ sub-folders
#                 default: /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/ar_raw_all/raw
#   OUTPUT_ROOT – where the organised arkitscenes/ tree will be written
#                 default: /apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/organised
#   WORKERS     – parallel jobs (default: 8)
# =============================================================================

set -euo pipefail

RAW_ROOT="${1:-/apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/ARKitScenes/ar_raw_all/raw}"
OUTPUT_ROOT="${2:-/apdcephfs_303747097/share_303747097/jingfanchen/data/arkitscenes/organised}"
WORKERS="${3:-8}"

# The three zip names we actually need
NEEDED_ZIPS=("lowres_wide.zip" "lowres_wide_intrinsics.zip" "lowres_depth.zip")

echo "============================================================"
echo "  ARKitScenes unzip script"
echo "  RAW_ROOT    : ${RAW_ROOT}"
echo "  OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo "  WORKERS     : ${WORKERS}"
echo "============================================================"

# Collect all scene directories (Training/* and Validation/*)
mapfile -t SCENE_DIRS < <(find "${RAW_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)

TOTAL=${#SCENE_DIRS[@]}
echo "Found ${TOTAL} scene directories."

if [[ ${TOTAL} -eq 0 ]]; then
    echo "ERROR: No scene directories found under ${RAW_ROOT}. Check the path."
    exit 1
fi

# ── worker function ──────────────────────────────────────────────────────────
process_scene() {
    local scene_dir="$1"
    local output_root="$2"
    local scene_id
    scene_id=$(basename "${scene_dir}")

    local frames_dir="${output_root}/arkitscenes/${scene_id}/${scene_id}_frames"

    for zip_name in "lowres_wide.zip" "lowres_wide_intrinsics.zip" "lowres_depth.zip"; do
        local zip_path="${scene_dir}/${zip_name}"
        local sub_dir="${zip_name%.zip}"          # e.g. lowres_wide
        local target_dir="${frames_dir}/${sub_dir}"

        if [[ ! -f "${zip_path}" ]]; then
            echo "[SKIP]  ${scene_id}/${zip_name} – zip not found"
            continue
        fi

        if [[ -d "${target_dir}" ]] && [[ -n "$(ls -A "${target_dir}" 2>/dev/null)" ]]; then
            echo "[EXIST] ${scene_id}/${zip_name} – already extracted, skipping"
            continue
        fi

        mkdir -p "${target_dir}"

        # ARKitScenes zips typically contain a single top-level folder matching
        # the zip stem; use -j to junk directory paths so files land directly
        # in target_dir.
        if unzip -q -j "${zip_path}" -d "${target_dir}" 2>/dev/null; then
            echo "[OK]    ${scene_id}/${zip_name}"
        else
            echo "[ERROR] ${scene_id}/${zip_name} – unzip failed (exit $?)"
        fi
    done
}

export -f process_scene

# ── parallel execution ───────────────────────────────────────────────────────
echo ""
echo "Starting extraction with ${WORKERS} parallel workers..."
echo ""

printf '%s\n' "${SCENE_DIRS[@]}" | xargs -P "${WORKERS}" -I{} bash -c 'process_scene "$@"' _ {} "${OUTPUT_ROOT}"

echo ""
echo "============================================================"
echo "  Extraction complete."
echo "  Output: ${OUTPUT_ROOT}/arkitscenes/"
echo "============================================================"

# ── quick sanity check ───────────────────────────────────────────────────────
echo ""
echo "Sanity check – counting extracted scenes:"
EXTRACTED=$(find "${OUTPUT_ROOT}/arkitscenes" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "  Scenes with output directories : ${EXTRACTED} / ${TOTAL}"

echo ""
echo "Sample directory listing (first scene):"
FIRST_SCENE=$(find "${OUTPUT_ROOT}/arkitscenes" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -1)
if [[ -n "${FIRST_SCENE}" ]]; then
    find "${FIRST_SCENE}" -maxdepth 3 | head -20
fi
