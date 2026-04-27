#!/usr/bin/env bash
# =============================================================================
#  OpenSpatial × ScanNet++ end-to-end pipeline
# -----------------------------------------------------------------------------
#  Stages:
#    [1] Raw ScanNet++  →  raw Parquet            (data_preprocessing script)
#    [2] Raw Parquet    →  per-image + per-scene Parquet (run.py preprocessing)
#    [3] per-image Parquet → singleview QA        (run.py annotation)
#        per-scene Parquet → multiview  QA        (run.py annotation)
#    [4] Visualize results with built-in server
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

# ==================== [0] 用户需要修改的路径 =================================
export PROJECT_ROOT="/apdcephfs_303747097/share_303747097/jingfanchen/code/OpenSpatial"
export SCANNETPP_RAW="/path/to/scannetpp/data"          # <-- ScanNet++ 原始数据根
export RUN_ROOT="${PROJECT_ROOT}/output/scannetpp_run"  # 所有产物都放到这里
export PARQUET_RAW_DIR="${RUN_ROOT}/01_parquet_raw"     # Step 1 产物
export PIPELINE_OUT_DIR="${RUN_ROOT}/02_pipeline"       # Step 2 产物根
export ANNOT_OUT_DIR="${RUN_ROOT}/03_annotation"        # Step 3 产物根
export TMP_CFG_DIR="${RUN_ROOT}/_tmp_configs"           # 临时 yaml 存放目录
export LOG_DIR="${RUN_ROOT}/logs"                       # 日志
export TAGS_TSV="${PROJECT_ROOT}/data_preprocessing/scannetpp/scannet-labels.combined.tsv"

# 可选：HF 镜像（无法直连 HuggingFace 时打开）
# export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PARQUET_RAW_DIR}" "${PIPELINE_OUT_DIR}" "${ANNOT_OUT_DIR}" "${TMP_CFG_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

# ==================== sanity check ==========================================
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

# ==================== [1] 原始 ScanNet++ → raw Parquet ======================
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

# 把所有 batch_*.parquet 收集成 YAML list 字符串
RAW_PARQUETS=( $(ls "${PARQUET_RAW_DIR}"/batch_*.parquet 2>/dev/null || true) )
[[ ${#RAW_PARQUETS[@]} -gt 0 ]] || { err "no raw parquet found in ${PARQUET_RAW_DIR}"; exit 1; }

# ==================== yaml 生成工具 =========================================
# 用 python 动态改写 demo yaml 的 dataset.data_dir，避免污染原文件
render_cfg() {
    # $1 原 yaml,  $2 新 yaml,  $3 新的 data_dir (string or yaml-list)
    python - "$1" "$2" "$3" <<'PY'
import sys, yaml, pathlib, ast
src, dst, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(pathlib.Path(src).read_text())
# data_dir 既可能是单字符串，也可能是 json list
try:
    parsed = ast.literal_eval(data_dir)
    cfg["dataset"]["data_dir"] = parsed
except Exception:
    cfg["dataset"]["data_dir"] = data_dir
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

# ==================== [2] Preprocessing pipeline ============================
STEP2_CFG_SRC="${PROJECT_ROOT}/config/preprocessing/demo_preprocessing_scannetpp.yaml"
STEP2_CFG="${TMP_CFG_DIR}/preprocessing_scannetpp.yaml"
STEP2_DONE="${PIPELINE_OUT_DIR}/.done"

# 把 list 传进 render_cfg —— 写成 Python list 字面量
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

# run.py 的输出规则：${output_dir}/base_pipeline_<config_stem>/<stage>/<task>/
PIPE_RUN_DIR="${PIPELINE_OUT_DIR}/base_pipeline_preprocessing_scannetpp"
# 如果输入是 list，还会分 part_1/part_2/...
if [[ -d "${PIPE_RUN_DIR}/part_1" ]]; then
    PIPE_RUN_DIR="${PIPE_RUN_DIR}/part_1"     # demo 示意：实际多 part 需合并
    warn "multiple parts detected; using part_1. See logs to merge others if needed."
fi
SINGLEVIEW_PARQUET="${PIPE_RUN_DIR}/scene_fusion_stage/depth_back_projection/data.parquet"
MULTIVIEW_PARQUET="${PIPE_RUN_DIR}/group_stage/group/data.parquet"
need_file "${SINGLEVIEW_PARQUET}"
need_file "${MULTIVIEW_PARQUET}"
ok "singleview input : ${SINGLEVIEW_PARQUET}"
ok "multiview  input : ${MULTIVIEW_PARQUET}"

# ==================== [3] Annotation tasks ==================================
run_annot() {
    local cfg_name="$1"   # 不带 .yaml
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

log "[Step 3a] Singleview annotation tasks"
for cfg in demo_distance demo_depth demo_size demo_position demo_counting demo_3d_grounding; do
    run_annot "${cfg}" "${SINGLEVIEW_PARQUET}"
done

log "[Step 3b] Multiview annotation tasks"
for cfg in demo_multiview_distance \
           demo_multiview_size \
           demo_multiview_correspondence \
           demo_multiview_distance_obj_cam \
           demo_multiview_object_position; do
    run_annot "${cfg}" "${MULTIVIEW_PARQUET}"
done

# ==================== [4] 结果概览 ==========================================
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

ok "ALL DONE. Outputs under ${RUN_ROOT}"
log "Next:  python visualize_server.py --data_dir ${ANNOT_OUT_DIR} --port 8888"