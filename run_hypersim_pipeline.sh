#!/usr/bin/env bash
# =============================================================================
#  OpenSpatial × Hypersim end-to-end pipeline
# -----------------------------------------------------------------------------
#  Stages:
#    [1] Raw Hypersim → intrinsics + extrinsics + tonemap + depth + Parquet
#        (data_preprocessing/hypersim/prepare_hypersim.py)
#    [2] Raw Parquet → preprocessed Parquet (run.py preprocessing)
#    [3] Preprocessed Parquet → QA annotation (run.py annotation)
#    [4] Result summary
#
#  Usage:
#    ./run_hypersim_pipeline.sh                            # run all steps
#    ./run_hypersim_pipeline.sh --start-step 2             # skip raw preprocessing
#    ./run_hypersim_pipeline.sh --start-step 3 --multiview-only
#    ./run_hypersim_pipeline.sh --tasks demo_mmsi_camera_camera,demo_depth
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
MAX_WORKERS=32
TONEMAP_WORKERS=16
CHUNK_SIZE=1000
MAX_SCENES=""                 # empty = no limit (smoke testing)
MAX_TASKS=""                  # empty = no limit (hard-cap on total (scene,cam,frame) tasks in Parquet step)

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --start-step N        Start from step N (1-4, default: 1)
  --multiview-only      Only run multiview annotation tasks (step 3)
  --singleview-only     Only run singleview annotation tasks (step 3)
  --tasks TASK1,TASK2   Only run specified tasks (comma-separated, no .yaml)
  --max-workers N       Max parallel workers for preprocessing (default: 32)
  --tonemap-workers N   Max parallel workers for tonemapping (default: 16)
  --chunk-size N        Records per Parquet file (default: 1000)
  --max-scenes N        Limit number of scenes per step (for smoke testing)
  --max-tasks N         Hard-cap on total (scene, camera, frame) tasks in Parquet step (for smoke testing)
  -h, --help            Show this help message

Examples:
  $0                                          # full pipeline
  $0 --start-step 2                           # skip raw preprocessing
  $0 --start-step 3 --multiview-only          # only multiview QA
  $0 --tasks demo_depth,demo_distance         # specific tasks
  $0 --max-scenes 2 --max-tasks 20            # smoke test
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
        --max-workers)
            MAX_WORKERS="$2"; shift 2
            ;;
        --tonemap-workers)
            TONEMAP_WORKERS="$2"; shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"; shift 2
            ;;
        --max-scenes)
            MAX_SCENES="$2"; shift 2
            ;;
        --max-tasks)
            MAX_TASKS="$2"; shift 2
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
export HYPERSIM_RAW="/path/to/Hypersim"                 # <-- Hypersim raw data root (RGB/depth/HDF5)
export HYPERSIM_MESH_ROOT=""                             # <-- Hypersim mesh archive root (contains <scene>/_detail/mesh/). Leave empty to reuse HYPERSIM_RAW.
export CAMERA_PARAMS_CSV="${PROJECT_ROOT}/data_preprocessing/hypersim/metadata_camera_parameters.csv"
export LABELS_TSV="${PROJECT_ROOT}/data_preprocessing/scannetpp/scannet-labels.combined.tsv"
export NAME_FILTER_JSON=""                               # <-- optional: path to Hypersim_name_filter_results.json

export RUN_ROOT="${PROJECT_ROOT}/output/hypersim_run"
export PARQUET_RAW_DIR="${RUN_ROOT}/01_parquet_raw"
export PIPELINE_OUT_DIR="${RUN_ROOT}/02_pipeline"
export ANNOT_OUT_DIR="${RUN_ROOT}/03_annotation"
export TMP_CFG_DIR="${RUN_ROOT}/_tmp_configs"
export LOG_DIR="${RUN_ROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PARQUET_RAW_DIR}" "${PIPELINE_OUT_DIR}" "${ANNOT_OUT_DIR}" "${TMP_CFG_DIR}" "${LOG_DIR}"
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
    need_dir "${HYPERSIM_RAW}"
    need_file "${CAMERA_PARAMS_CSV}"
    need_file "${LABELS_TSV}"
    need_file "${PROJECT_ROOT}/run.py"
    python - <<'PY' || { err "env broken"; exit 1; }
import importlib, torch
for mod in ["yaml","pandas","open3d","trimesh","spacy","sam2","h5py","cv2"]:
    importlib.import_module(mod)
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
PY
    ok "env looks good"
fi

# ==================== [1] Raw Hypersim → Parquet ============================
if [[ "$START_STEP" -le 1 ]]; then
    STEP1_DONE="${PARQUET_RAW_DIR}/.done"
    if [[ -f "${STEP1_DONE}" ]]; then
        log "[Step 1] skip (already done, remove ${STEP1_DONE} to rerun)"
    else
        log "[Step 1] Hypersim raw data → intrinsics + extrinsics + tonemap + depth + Parquet"

        NAME_FILTER_ARG=""
        if [[ -n "${NAME_FILTER_JSON}" && -f "${NAME_FILTER_JSON}" ]]; then
            NAME_FILTER_ARG="--name_filter_json ${NAME_FILTER_JSON}"
        fi

        EXTRA_ARGS=""
        if [[ -n "${MAX_SCENES}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max_scenes ${MAX_SCENES}"
        fi
        if [[ -n "${MAX_TASKS}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max_tasks ${MAX_TASKS}"
        fi
        if [[ -n "${HYPERSIM_MESH_ROOT}" ]]; then
            need_dir "${HYPERSIM_MESH_ROOT}"
            EXTRA_ARGS="${EXTRA_ARGS} --mesh_root ${HYPERSIM_MESH_ROOT}"
        fi

        python data_preprocessing/hypersim/prepare_hypersim.py \
            --input_root       "${HYPERSIM_RAW}" \
            --output_dir       "${PARQUET_RAW_DIR}" \
            --camera_params_csv "${CAMERA_PARAMS_CSV}" \
            --labels_tsv       "${LABELS_TSV}" \
            --chunk_size       "${CHUNK_SIZE}" \
            --max_workers      "${MAX_WORKERS}" \
            --tonemap_workers  "${TONEMAP_WORKERS}" \
            ${NAME_FILTER_ARG} \
            ${EXTRA_ARGS} \
            2>&1 | tee "${LOG_DIR}/step1_prepare_hypersim.log"

        touch "${STEP1_DONE}"
        ok "[Step 1] produced: $(ls "${PARQUET_RAW_DIR}" | grep -c '\.parquet$') parquet file(s)"
    fi
else
    log "[Step 1] skipped (--start-step ${START_STEP})"
fi

# Collect raw parquets (needed by step 2+)
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
    STEP2_CFG_SRC="${PROJECT_ROOT}/config/preprocessing/demo_preprocessing_hypersim.yaml"
    STEP2_CFG="${TMP_CFG_DIR}/preprocessing_hypersim.yaml"
    STEP2_DONE="${PIPELINE_OUT_DIR}/.done"

    RAW_LIST_REPR="[$(printf '"%s",' "${RAW_PARQUETS[@]}")]"
    render_cfg "${STEP2_CFG_SRC}" "${STEP2_CFG}" "${RAW_LIST_REPR}"
    ok "rendered ${STEP2_CFG}"

    if [[ -f "${STEP2_DONE}" ]]; then
        log "[Step 2] skip (already done)"
    else
        log "[Step 2] filter → SAM2 refine → back-projection → group"
        python run.py \
            --config     "${STEP2_CFG}" \
            --output_dir "${PIPELINE_OUT_DIR}" \
            2>&1 | tee "${LOG_DIR}/step2_preprocessing.log"
        touch "${STEP2_DONE}"
    fi
else
    log "[Step 2] skipped (--start-step ${START_STEP})"
fi

# Locate output parquets from step 2
PIPE_RUN_DIR="${PIPELINE_OUT_DIR}/base_pipeline_preprocessing_hypersim"
if [[ -d "${PIPE_RUN_DIR}/part_1" ]]; then
    PIPE_RUN_DIR="${PIPE_RUN_DIR}/part_1"
    warn "multiple parts detected; using part_1."
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

# ==================== [4] Result summary ====================================
if [[ "$START_STEP" -le 4 ]]; then
    log "[Step 4] Result summary"
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
