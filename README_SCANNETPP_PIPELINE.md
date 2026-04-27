# OpenSpatial × ScanNet++ 端到端管线使用指南

## 📋 概述

`run_scannetpp_pipeline.sh` 是一个完整的端到端脚本，用于从原始 ScanNet++ 数据生成 OpenSpatial 多视角空间理解 QA 数据。

## 🚀 快速开始

### 1. 环境准备

确保已安装所有依赖：
```bash
pip install -r requirements.txt
# SAM2 包
pip install 'git+https://github.com/facebookresearch/sam2.git'
# spacy 英语模型
python -m spacy download en_core_web_sm
# FFmpeg (用于video mkv格式转换)
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

### 2. 数据格式转换 (重要！)

**如果您的ScanNet++数据是video mkv格式**，需要先转换为帧格式：

```bash
# 转换video mkv格式为帧格式
python convert_scannetpp_video_to_frames.py \
    --input_dir /path/to/raw/scannetpp/data \
    --output_dir /path/to/converted/scannetpp/frames

# 然后使用转换后的数据路径
SCANNETPP_RAW="/path/to/converted/scannetpp/frames"
```

### 3. 配置路径

编辑 `run_scannetpp_pipeline.sh` 顶部的路径变量：

```bash
# 必须修改的路径
export SCANNETPP_RAW="/path/to/scannetpp/data"          # ScanNet++ 数据根目录

# 可选修改的路径  
export RUN_ROOT="${PROJECT_ROOT}/output/scannetpp_run"  # 输出目录根
# export HF_ENDPOINT="https://hf-mirror.com"           # 无法直连 HuggingFace 时打开
```

### 4. 运行完整管线

```bash
cd /apdcephfs_303747097/share_303747097/jingfanchen/code/OpenSpatial
./run_scannetpp_pipeline.sh
```

## 🔄 数据格式说明

### ScanNet++ 数据格式差异

ScanNet++ 有两种主要数据格式：

#### 1. Video MKV 格式 (原始格式)
```
<scene_id>/
├── video.mkv                  # RGB视频文件
├── depth.mkv                  # 深度视频文件
└── metadata/                  # 元数据文件
    ├── pose_*.json           # 位姿数据
    └── intrinsic_*.json      # 内参数据
```

#### 2. 帧格式 (OpenSpatial所需格式)
```
<scene_id>/
├── iphone/
│   ├── rgb/*.jpg             # 已解压的RGB帧
│   ├── depth/*.png           # 已解压的深度图
│   ├── aligned_pose/*.json    # 位姿文件
│   └── intrinsic/*.json       # 内参文件
└── scans/
    ├── mesh_aligned_0.05.ply  # 3D mesh
    └── segments_anno.json     # 物体标注
```

### 格式转换工具

使用 `convert_scannetpp_video_to_frames.py` 进行自动转换：

```bash
# 基本用法
python convert_scannetpp_video_to_frames.py \
    --input_dir /path/to/raw/scannetpp \
    --output_dir /path/to/converted/scannetpp

# 并行处理（加速）
python convert_scannetpp_video_to_frames.py \
    --input_dir /path/to/raw/scannetpp \
    --output_dir /path/to/converted/scannetpp \
    --num_workers 8
```

## 📊 管线流程

### Stage 1: 数据预处理
- **输入**: 原始 ScanNet++ 数据
- **输出**: `01_parquet_raw/batch_*.parquet`
- **功能**: 读取 RGB、depth、pose、intrinsic，从 mesh 和 annotation 提取 3D OBB

### Stage 2: 预处理管线
- **输入**: Stage 1 的 parquet 文件
- **输出**: `02_pipeline/base_pipeline_*/.../data.parquet`
- **流程**:
  - `flatten`: 展开 per-scene 为 per-image
  - `3dbox_filter`: 2D 投影 + 点云校验 3D box
  - `sam2_refiner`: SAM2 重分割得到高质量 mask + 2D box
  - `depth_back_projection`: depth + mask → per-object 点云
  - `group`: 按 scene_id 聚回 per-scene

### Stage 3: 标注生成
- **Singleview QA** (6个任务):
  - `demo_distance`: 物体间距离
  - `demo_depth`: 物体深度
  - `demo_size`: 物体尺寸
  - `demo_position`: 物体位置
  - `demo_counting`: 物体计数
  - `demo_3d_grounding`: 3D 定位

- **Multiview QA** (5个任务):
  - `demo_multiview_distance`: 跨视角距离
  - `demo_multiview_size`: 跨视角尺寸
  - `demo_multiview_correspondence`: 跨视角对应关系
  - `demo_multiview_distance_obj_cam`: 物体-相机距离
  - `demo_multiview_object_position`: 跨视角物体位置

## 🔧 配置说明

### ScanNet++ 数据要求

**转换后的数据目录结构**：
```
<SCANNETPP_ROOT>/
├── <scene_id_1>/
│   ├── iphone/
│   │   ├── rgb/*0.jpg              # 关键帧（文件名以 0 结尾）
│   │   ├── depth/*.png             # 深度图，depth_scale=1000
│   │   ├── aligned_pose/*.json     # 4x4 外参矩阵
│   │   └── intrinsic/*.json        # 3x3/4x4 内参矩阵
│   └── scans/
│       ├── mesh_aligned_0.05.ply    # 重力对齐的 mesh
│       └── segments_anno.json      # 物体标注（含 OBB）
└── <scene_id_2>/
    └── ...
```

### 断点续跑

脚本支持断点续跑，每个阶段完成后会创建 `.done` 文件：

- **重跑 Stage 1**: `rm output/scannetpp_run/01_parquet_raw/.done`
- **重跑 Stage 2**: `rm output/scannetpp_run/02_pipeline/.done`
- **重跑 Stage 3**: 默认会重跑所有任务，如需跳过已完成的可手动删除对应日志

## 🐛 常见问题

### Q: ScanNet++数据是video mkv格式，如何处理？
**A**: 使用 `convert_scannetpp_video_to_frames.py` 脚本进行转换。该脚本会自动：
- 从video.mkv提取RGB帧
- 从depth.mkv提取深度图
- 生成位姿和内参文件
- 复制mesh和annotation文件

### Q: SAM2 模型下载失败
**A**: 打开脚本中的 `HF_ENDPOINT="https://hf-mirror.com"`，或手动下载：
```bash
huggingface-cli download facebook/sam2-hiera-small
```

### Q: 显存不足
**A**: 在 `config/preprocessing/demo_preprocessing_scannetpp.yaml` 中修改：
```yaml
sam2_refiner:
  segmenter_model: "facebook/sam2-hiera-tiny"  # 使用更小的模型
```

### Q: 某个 scene 被跳过
**A**: 检查日志中的 `Missing data for scene ...`，确认该 scene 是否缺少必要文件

### Q: 多 part 输出
**A**: 当输入多个 parquet 文件时，run.py 会生成 `part_1/part_2/...`，脚本默认使用 `part_1`

## 📈 验证结果

### 1. 命令行验证
```bash
# 检查各阶段输出
ls output/scannetpp_run/01_parquet_raw/batch_*.parquet
ls output/scannetpp_run/02_pipeline/base_pipeline_*/*/data.parquet
ls output/scannetpp_run/03_annotation/base_pipeline_*/*/data.parquet

# 查看 QA 数据示例
python -c "import pandas as pd; df=pd.read_parquet('output/scannetpp_run/03_annotation/base_pipeline_demo_distance/annotation_stage/distance/data.parquet'); print(df.iloc[0]['messages'])"
```

### 2. 可视化验证
```bash
python visualize_server.py --data_dir output/scannetpp_run/03_annotation --port 8888
# 浏览器访问 http://<host>:8888
```

## 🧪 冒烟测试

首次运行建议使用单 scene 测试：

```bash
# 创建单 scene 测试目录
SMOKE_ROOT=/tmp/scannetpp_smoke
mkdir -p ${SMOKE_ROOT}
ln -sfn ${SCANNETPP_RAW}/<your_scene_id> ${SMOKE_ROOT}/<your_scene_id}

# 临时修改脚本路径
SCANNETPP_RAW=${SMOKE_ROOT} ./run_scannetpp_pipeline.sh
```

## 📁 输出结构

```
output/scannetpp_run/
├── 01_parquet_raw/           # Stage 1: raw parquet
├── 02_pipeline/              # Stage 2: preprocessing pipeline
├── 03_annotation/            # Stage 3: QA annotation
├── logs/                     # 各阶段日志
└── _tmp_configs/             # 临时配置文件
```

## 📞 技术支持

遇到问题时请检查：
1. 日志文件 `logs/step*.log`
2. 确认所有路径配置正确
3. 确认依赖包版本兼容
4. 检查显存和磁盘空间

---

**注意**: 首次运行全量 ScanNet++ 数据可能需要较长时间（数小时到数天），建议先进行冒烟测试。