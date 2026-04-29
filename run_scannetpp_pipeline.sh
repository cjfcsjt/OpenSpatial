#!/usr/bin/env bash
# =============================================================================
#  OpenSpatial × ScanNet++ end-to-end pipeline
# -----------------------------------------------------------------------------
#  Stages:
#    [1] Raw ScanNet++  →  raw Parquet            (data_preprocessing script)
#    [2] Raw Parquet    →  per-image + per-scene Parquet (run.py preprocessing)
#    [3] per-image Parquet → singleview QA        (run.py annotation)
#        per-scene Parquet → multiview  QA        (run.py annotation)
#    [4] Result summary
#
#  Usage:
#    ./run_scannetpp_pipeline.sh                          # run all steps, all tasks
#    ./run_scannetpp_pipeline.sh --start-step 3           # skip step 1 & 2
#    ./run_scannetpp_pipeline.sh --multiview-only         # only multiview tasks
#    ./run_scannetpp_pipeline.sh --singleview-only        # only singleview tasks
#    ./run_scannetpp_pipeline.sh --tasks demo_multiview_clockwise,demo_mmsi_camera_camera
#    ./run_scannetpp_pipeline.sh --start-step 3 --multiview-only
# # 1. 完整跑全部流程（和以前一样）
# ./run_scannetpp_pipeline.sh

# # 2. 只跑多视图任务（跳过 step 1/2，假设 parquet 已就绪）
# ./run_scannetpp_pipeline.sh --start-step 3 --multiview-only

# # 3. 只跑指定的几个任务
# ./run_scannetpp_pipeline.sh --start-step 3 \
#     --tasks demo_multiview_clockwise,demo_mmsi_camera_camera,demo_mmsi_camera_motion

# # 4. 只跑单视图任务
# ./run_scannetpp_pipeline.sh --start-step 3 --singleview-only

# # 5. 从 step 2 开始（跳过原始数据转 parquet）
# ./run_scannetpp_pipeline.sh --start-step 2
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
TASK_FILTER=""       # comma-separated task names, empty = all

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --start-step N        Start from step N (1-4, default: 1)
  --multiview-only      Only run multiview annotation tasks (step 3)
  --singleview-only     Only run singleview annotation tasks (step 3)
  --tasks TASK1,TASK2   Only run specified tasks (comma-separated, no .yaml)
                        e.g. --tasks demo_multiview_clockwise,demo_mmsi_camera_camera
  -h, --help            Show this help message

Examples:
  $0                                          # full pipeline
  $0 --start-step 3                           # skip preprocessing, run annotation
  $0 --start-step 3 --multiview-only          # only multiview QA from step 3
  $0 --tasks demo_mmsi_camera_camera          # only one specific task
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-step)
            START_STEP="$2"; shift 2
            if [[ ! "$START_STEP" =~ ^[1-4]$ ]]; then
                err "--start-step must be 1, 2, 3, or 4"; exit 1
            fi
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

log "Config: start-step=${START_STEP}  multiview-only=${MULTIVIEW_ONLY}  singleview-only=${SINGLEVIEW_ONLY}  tasks=${TASK_FILTER:-all}"

# ==================== [0] paths ==============================================
export PROJECT_ROOT="/apdcephfs_303747097/share_303747097/jingfanchen/code/OpenSpatial"
export SCANNETPP_RAW="/path/to/scannetpp/data"          # <-- ScanNet++ raw data root
export RUN_ROOT="${PROJECT_ROOT}/output/scannetpp_run"
export PARQUET_RAW_DIR="${RUN_ROOT}/01_parquet_raw"
export PIPELINE_OUT_DIR="${RUN_ROOT}/02_pipeline"
export ANNOT_OUT_DIR="${RUN_ROOT}/03_annotation"
export BLINK_OUT_DIR="${RUN_ROOT}/04_blink"
export TMP_CFG_DIR="${RUN_ROOT}/_tmp_configs"
export LOG_DIR="${RUN_ROOT}/logs"
export TAGS_TSV="${PROJECT_ROOT}/data_preprocessing/scannetpp/scannet-labels.combined.tsv"

# export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PARQUET_RAW_DIR}" "${PIPELINE_OUT_DIR}" "${ANNOT_OUT_DIR}" "${BLINK_OUT_DIR}" "${TMP_CFG_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

# ==================== task definitions =======================================
# Singleview tasks (use per-frame parquet)
SINGLEVIEW_TASKS=(
    demo_distance
    demo_depth
    demo_size
    demo_position
    demo_counting
    demo_3d_grounding
)

# Multiview tasks (use per-scene grouped parquet)
MULTIVIEW_TASKS=(
    # --- original tasks ---
    demo_multiview_distance
    demo_multiview_size
    demo_multiview_correspondence
    demo_multiview_distance_obj_cam
    demo_multiview_object_position
    # --- new benchmark tasks ---
    demo_multiview_clockwise
    demo_multiview_camera_movement
    demo_multiview_relative_distance
    demo_multiview_bev_pose_estimation
    demo_multiview_manipulation_view
    # --- MMSI-bench tasks ---
    demo_mmsi_camera_camera
    demo_mmsi_camera_motion
    demo_mmsi_camera_object
    demo_mmsi_object_object
)

# ==================== task filter helper =====================================
should_run_task() {
    local task_name="$1"
    if [[ -z "$TASK_FILTER" ]]; then
        return 0  # no filter, run all
    fi
    # check if task_name is in the comma-separated TASK_FILTER
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
    need_dir "${SCANNETPP_RAW}"
    need_file "${TAGS_TSV}"
    need_file "${PROJECT_ROOT}/run.py"
    python - <<'PY' || { err "env broken"; exit 1; }
import importlib, torch
for mod in ["yaml","pandas","open3d","trimesh","spacy","sam2"]:
    importlib.import_module(mod)
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
PY
    ok "env looks good"
fi

# ==================== [1] Raw ScanNet++ → raw Parquet =======================
if [[ "$START_STEP" -le 1 ]]; then
    STEP1_DONE="${PARQUET_RAW_DIR}/.done"
    if [[ -f "${STEP1_DONE}" ]]; then
        log "[Step 1] skip (already done, remove ${STEP1_DONE} to rerun)"
    else
        log "[Step 1] ScanNet++ raw data → raw parquet"
        python data_preprocessing/scannetpp/prepare_scannetpp.py \
            --input_root         "${SCANNETPP_RAW}" \
            --output_dir         "${PARQUET_RAW_DIR}" \
            --selected_tags_file "${TAGS_TSV}" \
            --chunk_size 100 \
            --max_workers 32 \
            2>&1 | tee "${LOG_DIR}/step1_prepare_scannetpp.log"
        touch "${STEP1_DONE}"
        ok   "[Step 1] produced: $(ls "${PARQUET_RAW_DIR}" | grep -c '\.parquet$') parquet file(s)"
    fi
else
    log "[Step 1] skipped (--start-step ${START_STEP})"
fi

# collect raw parquets (needed by step 2+)
RAW_PARQUETS=( $(ls "${PARQUET_RAW_DIR}"/batch_*.parquet 2>/dev/null || true) )
if [[ "$START_STEP" -le 2 ]]; then
    [[ ${#RAW_PARQUETS[@]} -gt 0 ]] || { err "no raw parquet found in ${PARQUET_RAW_DIR}"; exit 1; }
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

# ==================== [2] Preprocessing pipeline ============================
if [[ "$START_STEP" -le 2 ]]; then
    STEP2_CFG_SRC="${PROJECT_ROOT}/config/preprocessing/demo_preprocessing_scannetpp.yaml"
    STEP2_CFG="${TMP_CFG_DIR}/preprocessing_scannetpp.yaml"
    STEP2_DONE="${PIPELINE_OUT_DIR}/.done"

    RAW_LIST_REPR="[$(printf '\"%s\",' "${RAW_PARQUETS[@]}")]"
    render_cfg "${STEP2_CFG_SRC}" "${STEP2_CFG}" "${RAW_LIST_REPR}"
    ok "rendered ${STEP2_CFG}"

    if [[ -f "${STEP2_DONE}" ]]; then
        log "[Step 2] skip (already done)"
    else
        log "[Step 2] flatten → filter → SAM2 refine → back-projection → group"
        python run.py \
            --config     "${STEP2_CFG}" \
            --output_dir "${PIPELINE_OUT_DIR}" \
            2>&1 | tee "${LOG_DIR}/step2_preprocessing.log"
        touch "${STEP2_DONE}"
    fi
else
    log "[Step 2] skipped (--start-step ${START_STEP})"
fi

# locate output parquets from step 2
PIPE_RUN_DIR="${PIPELINE_OUT_DIR}/base_pipeline_preprocessing_scannetpp"
if [[ -d "${PIPE_RUN_DIR}/part_1" ]]; then
    PIPE_RUN_DIR="${PIPE_RUN_DIR}/part_1"
    warn "multiple parts detected; using part_1. See logs to merge others if needed."
fi
SINGLEVIEW_PARQUET="${PIPE_RUN_DIR}/scene_fusion_stage/depth_back_projection/data.parquet"
MULTIVIEW_PARQUET="${PIPE_RUN_DIR}/group_stage/group/data.parquet"

if [[ "$START_STEP" -le 3 ]]; then
    [[ "$SINGLEVIEW_ONLY" == true ]] || need_file "${MULTIVIEW_PARQUET}"
    [[ "$MULTIVIEW_ONLY" == true ]]  || need_file "${SINGLEVIEW_PARQUET}"
    ok "singleview input : ${SINGLEVIEW_PARQUET}"
    ok "multiview  input : ${MULTIVIEW_PARQUET}"
fi

# ==================== [3] Annotation tasks ==================================
run_annot() {
    local cfg_name="$1"
    local data_dir="$2"
    local src="${PROJECT_ROOT}/config/annotation/${cfg_name}.yaml"
    local dst="${TMP_CFG_DIR}/${cfg_name}.yaml"
    need_file "${src}"
    render_cfg "${src}" "${dst}" "${data_dir}"
    log "  ↳ ${cfg_name}"
    python run.py --config "${dst}" --output_dir "${ANNOT_OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/step3_${cfg_name}.log"
    ok "  ${cfg_name} done"
}

if [[ "$START_STEP" -le 3 ]]; then
    # --- Singleview tasks ---
    if [[ "$MULTIVIEW_ONLY" == false ]]; then
        log "[Step 3a] Singleview annotation tasks"
        for cfg in "${SINGLEVIEW_TASKS[@]}"; do
            should_run_task "$cfg" && run_annot "$cfg" "${SINGLEVIEW_PARQUET}" || \
                log "  ↳ ${cfg} (skipped by --tasks filter)"
        done
    else
        log "[Step 3a] Singleview tasks skipped (--multiview-only)"
    fi

    # --- Multiview tasks ---
    if [[ "$SINGLEVIEW_ONLY" == false ]]; then
        log "[Step 3b] Multiview annotation tasks"
        for cfg in "${MULTIVIEW_TASKS[@]}"; do
            should_run_task "$cfg" && run_annot "$cfg" "${MULTIVIEW_PARQUET}" || \
                log "  ↳ ${cfg} (skipped by --tasks filter)"
        done
    else
        log "[Step 3b] Multiview tasks skipped (--singleview-only)"
    fi
else
    log "[Step 3] skipped (--start-step ${START_STEP})"
fi

# ==================== [3c] Export QA parquet → JSONL + images folder ========
# See the embodiedscan pipeline for rationale — we keep parquet for the
# visualize_server / downstream reuse, but also emit a BLINK / MindCube-style
# JSONL plus a standalone ``images/`` folder for training-time consumption.
if [[ "$START_STEP" -le 3 ]]; then
    log "[Step 3c] Export QA parquet → JSONL + images folder (${BLINK_OUT_DIR})"
    python "${PROJECT_ROOT}/convert_to_blink.py" \
        --input_dir   "${ANNOT_OUT_DIR}" \
        --output_dir  "${BLINK_OUT_DIR}" \
        --data_source "OpenSpatial_ScanNetPP" \
        2>&1 | tee "${LOG_DIR}/step3c_convert_to_blink.log"
    ok "[Step 3c] BLINK export done: ${BLINK_OUT_DIR}"
fi

# ==================== [4] Result summary ====================================
if [[ "$START_STEP" -le 4 ]]; then
    log "[Step 4] Result summary"
    python - "${ANNOT_OUT_DIR}" "${BLINK_OUT_DIR}" <<'PY'
import pathlib, sys, pandas as pd
annot_root, blink_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = []
for p in sorted(annot_root.rglob("data.parquet")):
    try:
        n = len(pd.read_parquet(p, columns=["messages"])) if "messages" in pd.read_parquet(p, columns=[]).columns else len(pd.read_parquet(p))
    except Exception:
        try: n = len(pd.read_parquet(p))
        except Exception as e: n = f"err:{e}"
    rows.append((p.relative_to(annot_root), n))
print("== parquet (annotation) ==")
for rel, n in rows:
    print(f"  {n:>8}  {rel}")
print(f"  Total parquet files: {len(rows)}")

print("\n== jsonl (BLINK-format) ==")
jsonl_rows = []
for p in sorted(blink_root.glob("*.jsonl")):
    try:
        with p.open() as f:
            n = sum(1 for _ in f)
    except Exception as e:
        n = f"err:{e}"
    jsonl_rows.append((p.name, n))
for name, n in jsonl_rows:
    print(f"  {n:>8}  {name}")
print(f"  Total jsonl files:   {len(jsonl_rows)}")
img_root = blink_root / "images"
if img_root.is_dir():
    n_imgs = sum(1 for _ in img_root.rglob("*.png"))
    print(f"  Total images (png):  {n_imgs}  ({img_root})")
PY
fi

ok "ALL DONE. Outputs under ${RUN_ROOT}"
log "Next:  python visualize_server.py --data_dir ${ANNOT_OUT_DIR} --port 8888"