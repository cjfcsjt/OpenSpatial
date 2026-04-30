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
#  Usage (single source is REQUIRED):
#    ./run_embodiedscan_pipeline.sh --source scannet                   # ScanNet end-to-end
#    ./run_embodiedscan_pipeline.sh --source 3rscan --start-step 4     # skip extraction
#    ./run_embodiedscan_pipeline.sh --source arkitscenes --multiview-only
#    ./run_embodiedscan_pipeline.sh --source scannet --tasks demo_mmsi_camera_camera
#
#  All outputs are bucketed under  ${RUN_ROOT}/<source>/...  so different
#  sources never collide on disk. Each invocation processes exactly one source.

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
SOURCE=""                     # REQUIRED: scannet | 3rscan | matterport3d | arkitscenes
EXTRACT_WORKERS=24
MAX_SCENES=""                 # empty = no limit
MAX_TASKS=""                  # empty = no limit (hard-cap on total tasks for smoke test)

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --start-step N        Start from step N (1-6, default: 1)
  --source SRC          3D dataset source (REQUIRED, exactly one)
                        Choices: scannet, 3rscan, matterport3d, arkitscenes
  --multiview-only      Only run multiview annotation tasks (step 5)
  --singleview-only     Only run singleview annotation tasks (step 5)
  --tasks TASK1,TASK2   Only run specified tasks (comma-separated, no .yaml)
  --workers N           Number of parallel workers for extraction (default: 24)
  --max-scenes N        Limit number of scenes (for testing)
  --max-tasks N         Hard-cap on total (scene, camera) tasks (for smoke testing)
  -h, --help            Show this help message

Outputs are written to ${RUN_ROOT}/<source>/... so different sources are
fully isolated on disk. Each invocation processes exactly one source.

Examples:
  $0 --source scannet                                 # ScanNet end-to-end
  $0 --source 3rscan --start-step 4                   # skip extraction
  $0 --source arkitscenes --multiview-only            # annotation only
  $0 --source scannet --tasks demo_mmsi_camera_camera # one specific task
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

# Validate source name (exactly one required; no "all", no comma-list)
ALL_SOURCES=("scannet" "3rscan" "matterport3d" "arkitscenes")
if [[ -z "$SOURCE" ]]; then
    err "--source is required. Choose exactly one of: ${ALL_SOURCES[*]}"; exit 1
fi
if [[ "$SOURCE" == *,* ]]; then
    err "--source only accepts a single dataset (got '${SOURCE}'). Run the script once per source."; exit 1
fi
if [[ "$SOURCE" == "all" ]]; then
    err "--source=all is no longer supported. Run the script once per source."; exit 1
fi
valid=false
for a in "${ALL_SOURCES[@]}"; do
    [[ "$SOURCE" == "$a" ]] && valid=true
done
if [[ "$valid" == false ]]; then
    err "Unknown source: ${SOURCE}. Valid: ${ALL_SOURCES[*]}"; exit 1
fi
SOURCES=("${SOURCE}")   # kept as array for downstream loops (always length 1)

log "Config: start-step=${START_STEP}  source=${SOURCE}  multiview-only=${MULTIVIEW_ONLY}  singleview-only=${SINGLEVIEW_ONLY}  tasks=${TASK_FILTER:-all}"

# ==================== [0] paths ==============================================
export PROJECT_ROOT="/apdcephfs_303747097/share_303747097/jingfanchen/code/OpenSpatial"
export EMBODIEDSCAN_ROOT="/apdcephfs_303747097/share_303747097/jingfanchen/data/EmbodiedScan"       # <-- EmbodiedScan project root
export EMBODIEDSCAN_DATA="${EMBODIEDSCAN_ROOT}/data"    # <-- contains scannet/, 3rscan/, etc.
export EMBODIEDSCAN_PROJECT="/apdcephfs_303747097/share_303747097/jingfanchen/code/EmbodiedScan"
export RUN_ROOT="${PROJECT_ROOT}/output/embodiedscan_run"
# All artefacts are bucketed under  ${RUN_ROOT}/<source>/...  so different
# sources never collide on disk.
export SOURCE_ROOT="${RUN_ROOT}/${SOURCE}"
export EXTRACT_OUT_DIR="${SOURCE_ROOT}/01_extract"
export PARQUET_DIR="${SOURCE_ROOT}/02_parquet"
export PIPELINE_OUT_DIR="${SOURCE_ROOT}/03_pipeline"
export ANNOT_OUT_DIR="${SOURCE_ROOT}/04_annotation"
export BLINK_OUT_DIR="${SOURCE_ROOT}/05_blink"
export TMP_CFG_DIR="${SOURCE_ROOT}/_tmp_configs"
export LOG_DIR="${SOURCE_ROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}:${EMBODIEDSCAN_PROJECT}:${PYTHONPATH:-}"
export EMBODIEDSCAN_PREPROC="${PROJECT_ROOT}/data_preprocessing/embodiedscan"

mkdir -p "${SOURCE_ROOT}" "${EXTRACT_OUT_DIR}" "${PARQUET_DIR}" "${PIPELINE_OUT_DIR}" "${ANNOT_OUT_DIR}" "${BLINK_OUT_DIR}" "${TMP_CFG_DIR}" "${LOG_DIR}"
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

# ┌─────────────────────────────────────────┬──────────┬────────┬──────────────────────────────────┐
# │ 任务                                     │ QA 类型   │ 视角数  │ 核心问题                          │
# ├─────────────────────────────────────────┼──────────┼────────┼──────────────────────────────────┤
# │ 1.  multiview_distance                  │ OE       │ 2~6    │ 两物体3D距离 / N物体最近最远        │
# │ 2.  multiview_size                      │ OE       │ 2~6    │ 两物体大小比较 / N物体最大最小       │
# │ 3.  multiview_correspondence            │ OE/MCQ   │ 2      │ 跨视角点对应                       │
# │ 4.  multiview_distance_obj_cam          │ MCQ(3)   │ 2      │ 同一物体在哪个视角离相机更近         │
# │ 5.  multiview_object_position           │ OE/MCQ   │ 2      │ 跨视角物体相对方位(8方向)           │
# │ 6.  multiview_clockwise                 │ MCQ(2)   │ 2      │ 相机绕物体顺/逆时针               │
# │ 7.  multiview_camera_movement           │ MCQ(4)   │ 2      │ 相机主运动方向(前后左右)            │
# │ 8.  multiview_relative_distance         │ MCQ(3)   │ 1~2    │ A离B近还是离C近                   │
# │ 9.  multiview_bev_pose_estimation       │ MCQ(4)   │ 3+4BEV │ 哪张BEV图正确描述3相机布局          │
# │ 10. multiview_manipulation_view         │ MCQ(4)   │ 2      │ 视角变化类型(旋转/平移/倾斜/不变)    │
# │ 11. mmsi_camera_camera                  │ MCQ(4)   │ 2      │ Camera B在Camera A的哪个方向       │
# │ 12. mmsi_camera_motion                  │ MCQ(4)   │ 2      │ 复合运动分类(平移+旋转组合)         │
# │ 13. mmsi_camera_object                  │ MCQ(4)   │ 2      │ 物体在Camera A坐标系的哪个方向      │
# │ 14. mmsi_object_object                  │ MCQ(3)   │ 2      │ 物体A在物体B的左/右/同线           │
# └─────────────────────────────────────────┴──────────┴────────┴──────────────────────────────────┘

# OE = Open-Ended    MCQ = Multiple Choice Question    MCQ(N) = N个选项的选择题

# 帧采样分为两层：

# SceneGraph 构建层（scene_graph.py 的 from_multiview_example）：如果场景帧数超过 max_num_views（默认 400），先用 np.linspace 均匀下采样到 400 帧。

# 任务采样层（各任务的 _find_* 函数）：在 SceneGraph 的候选帧池中，按任务需求选取具体帧。

# ┌──────────────────────────────────┬──────┬──────────────────────────────────────────────────────┬──────────────────────┐
# │ 任务                              │ 帧数  │ 采样策略                                              │ Pose Diversity       │
# ├──────────────────────────────────┼──────┼──────────────────────────────────────────────────────┼──────────────────────┤
# │ singleview (6个)                  │ 1    │ 无采样，输入即单帧                                      │ N/A                  │
# ├──────────────────────────────────┼──────┼──────────────────────────────────────────────────────┼──────────────────────┤
# │ multiview_distance (pair)        │ 2    │ 链式游走 _find_view_chain                             │ ✅ rot≥15° or trans≥0 │
# │ multiview_distance (multi)       │ 3~6  │ 链式游走 _find_view_chain                             │ ✅ 同上               │
# │ multiview_size (pair)            │ 2    │ 链式游走 _find_view_chain                             │ ✅ 同上               │
# │ multiview_size (multi)           │ 3~6  │ 链式游走 _find_view_chain                             │ ✅ 同上               │
# │ multiview_correspondence         │ 2    │ 重叠视角 _find_overlapping_views + 深度重叠检查          │ ✅ 同上               │
# │ multiview_distance_obj_cam       │ 2    │ 重叠视角 _find_overlapping_views (同一物体)             │ ✅ 同上               │
# │ multiview_object_position        │ 2    │ 重叠视角 _find_overlapping_views (pose_diversity=False) │ ❌ 不要求             │
# │ multiview_clockwise              │ 2    │ 重叠视角 _find_overlapping_views + 角度≥20°            │ ✅ 同上               │
# │ multiview_camera_movement        │ 2    │ 按 view_index 排序取首尾帧                             │ ❌ 确定性选取          │
# │ multiview_relative_distance      │ 1~2  │ 物体驱动：先选3物体，再从可见视角池随机选≤2帧             │ ❌ 不要求             │
# │ multiview_bev_pose_estimation    │ 3+4  │ 随机选3帧，要求两两 pose-diverse                       │ ✅ 两两检查           │
# │ multiview_manipulation_view      │ 2    │ 重叠视角 _find_overlapping_views                      │ ✅ 同上               │
# │ mmsi_camera_camera               │ 2    │ 纯相机随机 _find_diverse_pair                         │ ✅ 同上               │
# │ mmsi_camera_motion               │ 2    │ 完全随机 random.sample (无任何过滤)                     │ ❌ 故意不要求          │
# │ mmsi_camera_object               │ 2    │ 随机+物体约束 (物体不在View A可见)                      │ ✅ 同上               │
# │ mmsi_object_object               │ 2    │ 随机两视角+随机两物体                                   │ ✅ 同上               │
# └──────────────────────────────────┴──────┴──────────────────────────────────────────────────────┴──────────────────────┘

MULTIVIEW_TASKS=(
    demo_multiview_distance
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

        EXTRA_ARGS=""
        if [[ -n "$MAX_SCENES" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max-scenes ${MAX_SCENES}"
        fi
        if [[ -n "$MAX_TASKS" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max-tasks ${MAX_TASKS}"
        fi

        PYTHONPATH="${EMBODIEDSCAN_PREPROC}:${PYTHONPATH}" \
        python -m embodiedscan_data extract \
            --dataset "${src}" \
            --data-root "${EMBODIEDSCAN_DATA}" \
            --output "${EXTRACT_OUT_DIR}" \
            --workers "${EXTRACT_WORKERS}" \
            ${EXTRA_ARGS} \
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
    # Args: src dst data_dir [data_root]
    #   - data_dir:   dataset.data_dir value (string or python-literal list)
    #   - data_root:  optional dataset.data_root (absolute path to raw dataset root)
    local data_root_arg="${4:-}"
    python - "$1" "$2" "$3" "${data_root_arg}" <<'PY'
import sys, yaml, pathlib, ast
src, dst, data_dir, data_root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = yaml.safe_load(pathlib.Path(src).read_text())
try:
    parsed = ast.literal_eval(data_dir)
    cfg["dataset"]["data_dir"] = parsed
except Exception:
    cfg["dataset"]["data_dir"] = data_dir
if data_root:
    cfg["dataset"]["data_root"] = data_root
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
    render_cfg "${STEP4_CFG_SRC}" "${STEP4_CFG}" "${RAW_LIST_REPR}" "${EMBODIEDSCAN_DATA}"
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
# per_image.parquet (Step 3)
#   ├── image, depth_map, pose, intrinsic
#   ├── obj_tags           (整个场景的全部物体标签)
#   └── bboxes_3d_world    (整个场景的全部 3D bbox)
#         │
#         ▼  filter_stage/3dbox_filter
#   + masks (粗), bboxes_2d (粗)
#   - 丢掉不可见 / wall-floor-ceiling-object
#         │
#         ▼  localization_stage/sam2_refiner
#   * masks / bboxes_2d 被 SAM2 精修
#         │
#         ▼  scene_fusion_stage/depth_back_projection
#   + pointclouds (每个实例的点云)
#   + depth_scale
#         │
#         ▼  group_stage/group
#   scene_id 聚合 → 每场景一行（multiview 用）
 
run_annot() {
    local cfg_name="$1"
    local data_dir="$2"
    local src="${PROJECT_ROOT}/config/annotation/${cfg_name}.yaml"
    local dst="${TMP_CFG_DIR}/${cfg_name}.yaml"
    need_file "${src}"
    render_cfg "${src}" "${dst}" "${data_dir}" "${EMBODIEDSCAN_DATA}"
    log "  ↳ ${cfg_name}  (BLINK_OUT_DIR=${BLINK_OUT_DIR})"
    BLINK_OUT_DIR="${BLINK_OUT_DIR}" \
    python run.py --config "${dst}" --output_dir "${ANNOT_OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/step5_${cfg_name}.log"

    # Post-task summary: show what actually landed on disk.
    local task_stem="${cfg_name#demo_}"
    local jsonl_file="${BLINK_OUT_DIR}/${task_stem}.jsonl"
    local img_dir="${BLINK_OUT_DIR}/images/${task_stem}"
    if [[ -f "${jsonl_file}" ]]; then
        local nline
        nline=$(wc -l < "${jsonl_file}" 2>/dev/null || echo 0)
        log "     jsonl=${jsonl_file}  lines=${nline}"
    else
        warn "     jsonl missing: ${jsonl_file}"
    fi
    if [[ -d "${img_dir}" ]]; then
        local nimg
        nimg=$(find "${img_dir}" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
        log "     images=${img_dir}  png_count=${nimg}"
    else
        warn "     images dir missing: ${img_dir}"
    fi
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

# ==================== [5c] Export QA parquet → JSONL + images folder ========
# The pipeline natively writes each QA task to parquet (good for resuming /
# piping into the visualize_server). We additionally materialize a BLINK /
# MindCube-style JSONL per task together with a standalone ``images/`` folder
# so downstream training / evaluation can consume the data without pyarrow.
#
# Layout produced here:
#   ${BLINK_OUT_DIR}/
#     <task_name>.jsonl                  (one JSONL per annotation task)
#     images/<task_name>/XXXXXX_view*.png
# ==================== [5c] Legacy: parquet → JSONL + images (opt-in) =========
# Since Step 5 now writes BLINK-format (<task>.jsonl + images/<task>/) inline
# via dataset/blink_writer.py, this post-hoc conversion is no longer required.
# It is kept as an opt-in escape hatch for re-exporting from existing parquet
# outputs:
#     LEGACY_BLINK_EXPORT=1 ./run_embodiedscan_pipeline.sh ...
if [[ "$START_STEP" -le 5 && "${LEGACY_BLINK_EXPORT:-0}" == "1" ]]; then
    log "[Step 5c] (legacy) Export QA parquet → JSONL + images folder (${BLINK_OUT_DIR})"
    python "${PROJECT_ROOT}/convert_to_blink.py" \
        --input_dir   "${ANNOT_OUT_DIR}" \
        --output_dir  "${BLINK_OUT_DIR}" \
        --data_source "OpenSpatial_EmbodiedScan" \
        2>&1 | tee "${LOG_DIR}/step5c_convert_to_blink.log"
    ok "[Step 5c] BLINK export done: ${BLINK_OUT_DIR}"
else
    log "[Step 5c] skipped (inline BLINK writer is the default; set LEGACY_BLINK_EXPORT=1 to force post-hoc conversion)"
fi

# ==================== [6] Result summary ====================================
if [[ "$START_STEP" -le 6 ]]; then
    log "[Step 6] Result summary"
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

ok "ALL DONE. Outputs under ${SOURCE_ROOT}  (source=${SOURCE})"
log "Next:  python visualize_server.py --data_dir ${BLINK_OUT_DIR} --port 8888"
