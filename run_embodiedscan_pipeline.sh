#!/usr/bin/env bash
# =============================================================================
#  OpenSpatial × EmbodiedScan end-to-end pipeline
# -----------------------------------------------------------------------------
#  Stages:
#    [1] Raw 3D data → per-image JSONL          (embodiedscan_data extract)
#    [2] Per-image JSONL → per-scene JSONL       (embodiedscan_data merge)
#    [3] JSONL → Parquet                         (embodiedscan_data export)
#    [4] Parquet → preprocessed Parquet          (run.py preprocessing)
#    [5] Preprocessed Parquet → QA annotation    (run.py annotation)
#    [6] Result summary
#
#  Usage:
#    ./run_embodiedscan_pipeline.sh                                    # all sources, all steps
#    ./run_embodiedscan_pipeline.sh --source scannet                   # only ScanNet
#    ./run_embodiedscan_pipeline.sh --source 3rscan,arkitscenes        # multiple sources
#    ./run_embodiedscan_pipeline.sh --start-step 4                     # skip extraction
#    ./run_embodiedscan_pipeline.sh --start-step 5 --multiview-only    # annotation only
#    ./run_embodiedscan_pipeline.sh --tasks demo_mmsi_camera_camera    # specific tasks

# EmbodiedScan 官方代码（必须）
# OpenSpatial 的 extract.py 在 worker 初始化时直接 from embodiedscan.explorer import EmbodiedScanExplorer，这个类来自 EmbodiedScan 官方仓库，不是 OpenSpatial 自带的。

# git clone https://github.com/OpenRobotLab/EmbodiedScan.git
# cd EmbodiedScan
# pip install -e . 
# 安装后 Python 环境里就有了 embodiedscan 包，EmbodiedScanExplorer 才能被导入。

# 2️⃣ 原始 3D 数据集 + EmbodiedScan 标注文件（必须）
# 你需要按照 EmbodiedScan 官方要求的目录结构准备数据：

# /path/to/EmbodiedScan/                    ← project_root（data-root 的父目录）
# ├── data/                                  ← --data-root 指向这里
# │   ├── scannet/
# │   ├── 3rscan/
# │   ├── matterport3d/
# │   ├── arkitscenes/
# │   │   └── <scene_id>/
# │   │       └── <scene_id>_frames/
# │   │           ├── lowres_wide/*.jpg
# │   │           ├── lowres_wide_intrinsics/*.pincam
# │   │           └── lowres_depth/*.png
# │   ├── embodiedscan_infos_train.pkl       ← v1 标注（ScanNet/3RScan/Matterport3D 用）
# │   ├── embodiedscan_infos_val.pkl
# │   └── embodiedscan_infos_test.pkl
# │
# └── embodiedscan-v2/                       ← ⭐ ARKitScenes v2 标注放这里
#     ├── embodiedscan_infos_train.pkl
#     ├── embodiedscan_infos_val.pkl
#     └── embodiedscan_infos_test.pkl 
# =============================================================================
set -Eeuo pipefail

# --------- helpers ----------------------------------------------------------
log()  { printf '\033[1;36m[%(%F %T)T]\033[0m %s\n' -1 "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; }
trap 'err "line $LINENO exited with status $?"; exit 1' ERR

need_file() { [[ -f "$1" ]] || { err "missing file: $1"; exit 1; }; }
need_dir()  { [[ -d "$1" ]] || { err "missing dir : $1"; exit 1; }; }

# ==================== CLI argument parsing ==================================
START_STEP=1
MULTIVIEW_ONLY=false
SINGLEVIEW_ONLY=false
TASK_FILTER=""
SOURCE="all"                  # scannet | 3rscan | matterport3d | arkitscenes | all
EXTRACT_WORKERS=24
MAX_SCENES=""                 # empty = no limit

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --start-step N        Start from step N (1-6, default: 1)
  --source SRC          3D dataset source(s), comma-separated
                        Choices: scannet, 3rscan, matterport3d, arkitscenes, all
                        (default: all)
  --multiview-only      Only run multiview annotation tasks (step 5)
  --singleview-only     Only run singleview annotation tasks (step 5)
  --tasks TASK1,TASK2   Only run specified tasks (comma-separated, no .yaml)
  --workers N           Number of parallel workers for extraction (default: 24)
  --max-scenes N        Limit number of scenes per source (for testing)
  -h, --help            Show this help message

Examples:
  $0                                                  # full pipeline, all sources
  $0 --source scannet                                 # only ScanNet source
  $0 --source 3rscan,arkitscenes                      # two sources
  $0 --start-step 4 --source scannet                  # skip extraction, preprocess
  $0 --start-step 5 --multiview-only                  # annotation only
  $0 --tasks demo_mmsi_camera_camera                  # one specific task
  $0 --source scannet --max-scenes 10                 # smoke test with 10 scenes
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-step)
            START_STEP="$2"; shift 2
            if [[ ! "$START_STEP" =~ ^[1-6]$ ]]; then
                err "--start-step must be 1-6"; exit 1
            fi
            ;;
        --source)
            SOURCE="$2"; shift 2
            ;;
        --multiview-only)
            MULTIVIEW_ONLY=true; shift
            ;;
        --singleview-only)
            SINGLEVIEW_ONLY=true; shift
            ;;
        --tasks)
            TASK_FILTER="$2"; shift 2
            ;;
        --workers)
            EXTRACT_WORKERS="$2"; shift 2
            ;;
        --max-scenes)
            MAX_SCENES="$2"; shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            err "Unknown option: $1"; usage
            ;;
    esac
done

if [[ "$MULTIVIEW_ONLY" == true && "$SINGLEVIEW_ONLY" == true ]]; then
    err "--multiview-only and --singleview-only are mutually exclusive"; exit 1
fi

# Validate source names
ALL_SOURCES=("scannet" "3rscan" "matterport3d" "arkitscenes")
if [[ "$SOURCE" == "all" ]]; then
    SOURCES=("${ALL_SOURCES[@]}")
else
    IFS=',' read -ra SOURCES <<< "$SOURCE"
    for s in "${SOURCES[@]}"; do
        valid=false
        for a in "${ALL_SOURCES[@]}"; do
            [[ "$s" == "$a" ]] && valid=true
        done
        if [[ "$valid" == false ]]; then
            err "Unknown source: $s. Valid: ${ALL_SOURCES[*]}"; exit 1
        fi
    done
fi

log "Config: start-step=${START_STEP}  source=${SOURCE}  multiview-only=${MULTIVIEW_ONLY}  singleview-only=${SINGLEVIEW_ONLY}  tasks=${TASK_FILTER:-all}"

# ==================== [0] paths ==============================================
export PROJECT_ROOT="/apdcephfs_303747097/share_303747097/jingfanchen/code/OpenSpatial"
export EMBODIEDSCAN_ROOT="/path/to/EmbodiedScan"       # <-- EmbodiedScan project root
export EMBODIEDSCAN_DATA="${EMBODIEDSCAN_ROOT}/data"    # <-- contains scannet/, 3rscan/, etc.
export RUN_ROOT="${PROJECT_ROOT}/output/embodiedscan_run"
export EXTRACT_OUT_DIR="${RUN_ROOT}/01_extract"
export PARQUET_DIR="${RUN_ROOT}/02_parquet"
export PIPELINE_OUT_DIR="${RUN_ROOT}/03_pipeline"
export ANNOT_OUT_DIR="${RUN_ROOT}/04_annotation"
export TMP_CFG_DIR="${RUN_ROOT}/_tmp_configs"
export LOG_DIR="${RUN_ROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export EMBODIEDSCAN_PREPROC="${PROJECT_ROOT}/data_preprocessing/embodiedscan"

mkdir -p "${EXTRACT_OUT_DIR}" "${PARQUET_DIR}" "${PIPELINE_OUT_DIR}" "${ANNOT_OUT_DIR}" "${TMP_CFG_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

# ==================== task definitions =======================================
SINGLEVIEW_TASKS=(
    demo_distance
    demo_depth
    demo_size
    demo_position
    demo_counting
    demo_3d_grounding
)

MULTIVIEW_TASKS=(
    demo_multiview_distance
    demo_multiview_size
    demo_multiview_correspondence
    demo_multiview_distance_obj_cam
    demo_multiview_object_position
    demo_multiview_clockwise
    demo_multiview_camera_movement
    demo_multiview_relative_distance
    demo_multiview_bev_pose_estimation
    demo_multiview_manipulation_view
    demo_mmsi_camera_camera
    demo_mmsi_camera_motion
    demo_mmsi_camera_object
    demo_mmsi_object_object
)

# ==================== task filter helper =====================================
should_run_task() {
    local task_name="$1"
    if [[ -z "$TASK_FILTER" ]]; then
        return 0
    fi
    IFS=',' read -ra FILTER_LIST <<< "$TASK_FILTER"
    for t in "${FILTER_LIST[@]}"; do
        [[ "$t" == "$task_name" ]] && return 0
    done
    return 1
}

# ==================== sanity check ==========================================
if [[ "$START_STEP" -le 1 ]]; then
    log "Sanity check ..."
    need_dir "${PROJECT_ROOT}"
    need_dir "${EMBODIEDSCAN_DATA}"
    need_file "${PROJECT_ROOT}/run.py"
    python - <<'PY' || { err "env broken"; exit 1; }
import importlib, torch
for mod in ["yaml","pandas","open3d","trimesh","spacy","sam2"]:
    importlib.import_module(mod)
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
PY
    ok "env looks good"
fi

# ==================== [1] Extract: Raw 3D → per-image JSONL =================
if [[ "$START_STEP" -le 1 ]]; then
    log "[Step 1] EmbodiedScan extract: raw 3D data → per-image JSONL"
    for src in "${SOURCES[@]}"; do
        STEP1_DONE="${EXTRACT_OUT_DIR}/.done_${src}"
        if [[ -f "${STEP1_DONE}" ]]; then
            log "  ↳ ${src} skip (already done, remove ${STEP1_DONE} to rerun)"
            continue
        fi
        log "  ↳ Extracting ${src} ..."

        MAX_SCENES_ARG=""
        if [[ -n "$MAX_SCENES" ]]; then
            MAX_SCENES_ARG="--max-scenes ${MAX_SCENES}"
        fi

        PYTHONPATH="${EMBODIEDSCAN_PREPROC}:${PYTHONPATH}" \
        python -m embodiedscan_data extract \
            --dataset "${src}" \
            --data-root "${EMBODIEDSCAN_DATA}" \
            --output "${EXTRACT_OUT_DIR}" \
            --workers "${EXTRACT_WORKERS}" \
            ${MAX_SCENES_ARG} \
            2>&1 | tee "${LOG_DIR}/step1_extract_${src}.log"

        touch "${STEP1_DONE}"
        ok "  ${src} extraction done"
    done
    ok "[Step 1] Extraction complete"
else
    log "[Step 1] skipped (--start-step ${START_STEP})"
fi

# ==================== [2] Merge: per-image JSONL → per-scene JSONL ==========
if [[ "$START_STEP" -le 2 ]]; then
    STEP2_DONE="${EXTRACT_OUT_DIR}/.done_merge"
    if [[ -f "${STEP2_DONE}" ]]; then
        log "[Step 2] skip (already done)"
    else
        log "[Step 2] Merging per-image JSONL → per-scene JSONL"
        PYTHONPATH="${EMBODIEDSCAN_PREPROC}:${PYTHONPATH}" \
        python -m embodiedscan_data merge \
            --input "${EXTRACT_OUT_DIR}" \
            2>&1 | tee "${LOG_DIR}/step2_merge.log"
        touch "${STEP2_DONE}"
    fi
    ok "[Step 2] Merge complete"
else
    log "[Step 2] skipped (--start-step ${START_STEP})"
fi

# ==================== [3] Export: JSONL → Parquet ============================
if [[ "$START_STEP" -le 3 ]]; then
    STEP3_DONE="${PARQUET_DIR}/.done"
    if [[ -f "${STEP3_DONE}" ]]; then
        log "[Step 3] skip (already done)"
    else
        log "[Step 3] Exporting JSONL → Parquet"
        PYTHONPATH="${EMBODIEDSCAN_PREPROC}:${PYTHONPATH}" \
        python -m embodiedscan_data export \
            --input "${EXTRACT_OUT_DIR}" \
            --format both \
            --batch-size 3000 \
            2>&1 | tee "${LOG_DIR}/step3_export.log"

        # Move generated parquet files to PARQUET_DIR
        if [[ -d "${EXTRACT_OUT_DIR}/per_image" ]]; then
            cp -r "${EXTRACT_OUT_DIR}/per_image" "${PARQUET_DIR}/per_image"
        fi
        if [[ -d "${EXTRACT_OUT_DIR}/per_scene" ]]; then
            cp -r "${EXTRACT_OUT_DIR}/per_scene" "${PARQUET_DIR}/per_scene"
        fi

        touch "${STEP3_DONE}"
    fi
    ok "[Step 3] Export complete"
else
    log "[Step 3] skipped (--start-step ${START_STEP})"
fi

# Locate parquet files from step 3
PER_IMAGE_PARQUETS=( $(ls "${PARQUET_DIR}"/per_image/*.parquet 2>/dev/null || true) )
PER_SCENE_PARQUETS=( $(ls "${PARQUET_DIR}"/per_scene/*.parquet 2>/dev/null || true) )

if [[ "$START_STEP" -le 4 ]]; then
    [[ ${#PER_IMAGE_PARQUETS[@]} -gt 0 ]] || { err "no per-image parquet found in ${PARQUET_DIR}/per_image"; exit 1; }
    ok "per-image parquets: ${#PER_IMAGE_PARQUETS[@]} file(s)"
    ok "per-scene parquets: ${#PER_SCENE_PARQUETS[@]} file(s)"
fi

# ==================== yaml render tool ======================================
render_cfg() {
    python - "$1" "$2" "$3" <<'PY'
import sys, yaml, pathlib, ast
src, dst, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(pathlib.Path(src).read_text())
try:
    parsed = ast.literal_eval(data_dir)
    cfg["dataset"]["data_dir"] = parsed
except Exception:
    cfg["dataset"]["data_dir"] = data_dir
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

# ==================== [4] Preprocessing pipeline ============================
if [[ "$START_STEP" -le 4 ]]; then
    STEP4_CFG_SRC="${PROJECT_ROOT}/config/preprocessing/demo_preprocessing_embodiedscan.yaml"
    STEP4_CFG="${TMP_CFG_DIR}/preprocessing_embodiedscan.yaml"
    STEP4_DONE="${PIPELINE_OUT_DIR}/.done"

    # Build list of per-image parquets as data_dir
    RAW_LIST_REPR="[$(printf '"%s",' "${PER_IMAGE_PARQUETS[@]}")]"
    render_cfg "${STEP4_CFG_SRC}" "${STEP4_CFG}" "${RAW_LIST_REPR}"
    ok "rendered ${STEP4_CFG}"

    if [[ -f "${STEP4_DONE}" ]]; then
        log "[Step 4] skip (already done)"
    else
        log "[Step 4] filter → SAM2 refine → back-projection → group"
        python run.py \
            --config     "${STEP4_CFG}" \
            --output_dir "${PIPELINE_OUT_DIR}" \
            2>&1 | tee "${LOG_DIR}/step4_preprocessing.log"
        touch "${STEP4_DONE}"
    fi
else
    log "[Step 4] skipped (--start-step ${START_STEP})"
fi

# Locate output parquets from step 4
PIPE_RUN_DIR="${PIPELINE_OUT_DIR}/base_pipeline_preprocessing_embodiedscan"
if [[ -d "${PIPE_RUN_DIR}/part_1" ]]; then
    PIPE_RUN_DIR="${PIPE_RUN_DIR}/part_1"
    warn "multiple parts detected; using part_1."
fi
SINGLEVIEW_PARQUET="${PIPE_RUN_DIR}/scene_fusion_stage/depth_back_projection/data.parquet"
MULTIVIEW_PARQUET="${PIPE_RUN_DIR}/group_stage/group/data.parquet"

if [[ "$START_STEP" -le 5 ]]; then
    [[ "$SINGLEVIEW_ONLY" == true ]] || need_file "${MULTIVIEW_PARQUET}"
    [[ "$MULTIVIEW_ONLY" == true ]]  || need_file "${SINGLEVIEW_PARQUET}"
    ok "singleview input : ${SINGLEVIEW_PARQUET}"
    ok "multiview  input : ${MULTIVIEW_PARQUET}"
fi

# ==================== [5] Annotation tasks ==================================
run_annot() {
    local cfg_name="$1"
    local data_dir="$2"
    local src="${PROJECT_ROOT}/config/annotation/${cfg_name}.yaml"
    local dst="${TMP_CFG_DIR}/${cfg_name}.yaml"
    need_file "${src}"
    render_cfg "${src}" "${dst}" "${data_dir}"
    log "  ↳ ${cfg_name}"
    python run.py --config "${dst}" --output_dir "${ANNOT_OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/step5_${cfg_name}.log"
    ok "  ${cfg_name} done"
}

if [[ "$START_STEP" -le 5 ]]; then
    # --- Singleview tasks ---
    if [[ "$MULTIVIEW_ONLY" == false ]]; then
        log "[Step 5a] Singleview annotation tasks"
        for cfg in "${SINGLEVIEW_TASKS[@]}"; do
            should_run_task "$cfg" && run_annot "$cfg" "${SINGLEVIEW_PARQUET}" || \
                log "  ↳ ${cfg} (skipped by --tasks filter)"
        done
    else
        log "[Step 5a] Singleview tasks skipped (--multiview-only)"
    fi

    # --- Multiview tasks ---
    if [[ "$SINGLEVIEW_ONLY" == false ]]; then
        log "[Step 5b] Multiview annotation tasks"
        for cfg in "${MULTIVIEW_TASKS[@]}"; do
            should_run_task "$cfg" && run_annot "$cfg" "${MULTIVIEW_PARQUET}" || \
                log "  ↳ ${cfg} (skipped by --tasks filter)"
        done
    else
        log "[Step 5b] Multiview tasks skipped (--singleview-only)"
    fi
else
    log "[Step 5] skipped (--start-step ${START_STEP})"
fi

# ==================== [6] Result summary ====================================
if [[ "$START_STEP" -le 6 ]]; then
    log "[Step 6] Result summary"
    python - "${ANNOT_OUT_DIR}" <<'PY'
import pathlib, sys, pandas as pd
root = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(root.rglob("data.parquet")):
    try:
        n = len(pd.read_parquet(p, columns=["messages"])) if "messages" in pd.read_parquet(p, columns=[]).columns else len(pd.read_parquet(p))
    except Exception:
        try: n = len(pd.read_parquet(p))
        except Exception as e: n = f"err:{e}"
    rows.append((p.relative_to(root), n))
for rel, n in rows:
    print(f"  {n:>8}  {rel}")
print(f"\nTotal parquet files: {len(rows)}")
PY
fi

ok "ALL DONE. Outputs under ${RUN_ROOT}"
log "Next:  python visualize_server.py --data_dir ${ANNOT_OUT_DIR} --port 8888"
