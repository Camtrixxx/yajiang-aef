# yajiang-aef

雅江区域 AlphaEarth-style 多源遥感表征学习项目。

模型输入 Sentinel-2 / Sentinel-1 / Landsat 多时相遥感影像，训练一个雅江地区通用多模态 embedding。预训练阶段用 DEM、WorldCover、JRC Water 三个目标做重建约束；评测阶段重点看冻结 embedding 后接简单任务头的 few-shot / linear probe 表现。

当前主线已从华为 Ascend/NPU 训练迁移到 NVIDIA A800 CUDA 环境，默认使用 8 卡 DDP。

## 当前主线

| 项目 | 当前设置 |
| --- | --- |
| 环境 | `conda activate hyh-dl` |
| 硬件 | NVIDIA A800，默认 8 卡 |
| 主配置 | `configs/yajiang_v1_2.yaml` |
| 输入 | S2 (6ch) / S1 (2ch) / Landsat (6ch) |
| 目标 | DEM / WorldCover / JRC Water |
| 训练样本 | `data/full_npy/train.jsonl` |
| 训练输出 | `outputs/aef_hyh_yajiang_v1_2*` |
| 评测输出 | `outputs/model_eval/v1_2*` |

## 快速开始

```bash
cd /data/heyuhang/yajiang-aef
conda activate hyh-dl
```

训练 v1.2 50 epoch：

```bash
bash scripts/run_v1_2.sh
```

从 50 epoch 续训到 100 epoch：

```bash
bash scripts/run_v1_2_continue_100.sh
```

从 100 epoch 续训到 200 epoch：

```bash
bash scripts/run_v1_2_continue_200.sh
```

默认使用 8 张卡：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_v1_2.sh
```

如果只想指定部分 GPU：

```bash
GPU_IDS=0,1,2,3 bash scripts/run_v1_2.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_v1_2.sh
```

## 评测

评测默认用单张 CUDA GPU，跑 512 个 patch，输出重建指标、few-shot linear probe 指标和 demo panel。

```bash
bash scripts/run_eval_suite_v1_2.sh
bash scripts/run_eval_suite_v1_2_continue_100.sh
bash scripts/run_eval_suite_v1_2_continue_200.sh
```

可调参数：

```bash
MAX_PATCHES=1024 BATCH_SIZE=4 MAX_PIXELS_PER_PATCH=256 \
bash scripts/run_eval_suite_v1_2_continue_200.sh
```

输出文件：

```text
outputs/model_eval/<version>/
├── metrics.json
├── report.md
├── report.html
├── fewshot_curves.png
└── demo_panels/*.png
```

## 应该关注哪些指标

本项目的目标是训练通用多模态 embedding，再接简单任务头完成不同下游任务。因此选模型时应优先关注下游 probe 指标，而不是只看重建 loss。

主指标：

| 指标 | 含义 |
| --- | --- |
| `aef_linear` | 冻结 AEF embedding 后训练简单线性头的效果 |
| `composite_linear` | 直接用原始多模态 composite 特征训练线性头的 baseline |
| `delta = aef_linear - composite_linear` | AEF embedding 是否带来下游增益 |
| 1/5/10/50-shot 曲线 | embedding 在少标注场景下是否稳定 |

重建指标仍然有用，但主要作为辅助诊断：

| 目标 | 指标 |
| --- | --- |
| DEM | MAE / R2 |
| WorldCover | macro F1 / macro IoU |
| JRC Water | binary F1 / IoU / boundary F1 |

当前实验显示：训练到 200 epoch 会显著提升重建指标，但 few-shot embedding 表现反而下降。因此如果目标是“接几个简单任务头就有不错效果”，优先用 downstream / few-shot 指标选择 checkpoint。

## 已完成实验

| 版本 | 训练 | 重建表现 | few-shot embedding 表现 | 适用判断 |
| --- | --- | --- | --- | --- |
| v1.2 | 50 epoch | DEM R2 0.9750 / WorldCover F1 0.5736 | 当前最好 | 更适合通用 embedding |
| v1.2_continue_100 | 100 epoch | DEM R2 0.9845 / WorldCover F1 0.6917 | 低于 50 epoch | 折中版本 |
| v1.2_continue_200 | 200 epoch | DEM R2 0.9889 / WorldCover F1 0.7837 | 继续下降 | 更适合重建/解码任务 |

详细记录见 `docs/experiments/v1.2_a800.md`。

## 数据准备

主训练数据位于：

```text
data/full_npy/
├── train.jsonl
├── s2/
├── s1/
├── landsat/
├── dem/
├── worldcover/
└── jrc_water/
```

重建 manifest：

```bash
python scripts/build_full_manifest.py
```

如需重新准备 Landsat / JRC Water 等 `.npy` 数据，需要当前环境具备对应 GIS 依赖，例如 `rasterio`、`opencv-python`。建议显式传入源目录和目标目录：

```bash
python scripts/prepare_landsat_npy.py \
  --src-root /path/to/raw/yajiang/landsat \
  --dst-root data/full_npy \
  --skip-existing

python scripts/prepare_jrc_water_npy.py \
  --src-root /path/to/raw/yajiang/jrc_water \
  --dst-root data/full_npy \
  --skip-existing
```

部分历史数据准备脚本仍保留 `/workspace/...` 默认路径，直接运行前需要确认路径和依赖。

## 模型结构

```text
S2 / S1 / Landsat 多时相输入
        ↓
SensorEncoderBank（每源独立编码，1x1 adapter + stride-2 stem）
        ↓
STPBlock x4（Space-Time-Precision 三路注意力）
        ↓
vMF Bottleneck（训练跳 L2 norm + 高斯噪声，推理球面投影）
        ↓
AEF Embedding Map [B, 128, H/2, W/2]
        ↓
ContinuousDecoder（DEM）/ CategoricalDecoder（WorldCover, JRC Water）
```

核心代码：

```text
src/models/
src/data/
src/training/
src/eval/
```

## v1.2 关键参数

| 参数 | 值 |
| --- | --- |
| image_size | 128 |
| max_frames | 16 |
| batch_size | 4 / GPU |
| precision_dim | 256 |
| embedding_dim | 128 |
| num_blocks | 4 |
| num_heads | 4 |
| vmf_kappa | 2000 |
| AMP | bf16 |
| optimizer | AdamW |
| lr | 1e-4，续训 5e-5 |
| TF32 | enabled |

训练加速相关配置：

```yaml
training:
  amp: true
  amp_dtype: bf16
  allow_tf32: true
  matmul_precision: high
  cudnn_benchmark: true
  find_unused_parameters: false

data:
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
  drop_last: true
```

## 训练输出

```text
outputs/aef_hyh_yajiang_v1_2/
├── checkpoints/
│   ├── best.pt
│   ├── final.pt
│   └── latest.pt
├── exports/
│   └── aef_hyh_yajiang_v1_2_deploy.pt
└── logs/
    ├── console.log
    └── train.log
```

续训版本对应：

```text
outputs/aef_hyh_yajiang_v1_2_continue_100/
outputs/aef_hyh_yajiang_v1_2_continue_200/
```

## 历史版本

| 版本 | 环境 | 说明 |
| --- | --- | --- |
| v0.2 | 早期单卡/小数据 | 真实数据链路联调 |
| v0.3a | Ascend/NPU | 全量数据早期实验 |
| v0.3c | Ascend/NPU 4 卡 | 第一版完整多卡训练 |
| v1.1 | CUDA 迁移前后 | 加入 Landsat 的过渡版本 |
| v1.2 | A800/CUDA 8 卡 | 当前主线 |

历史实验文档保留原始机器路径和 NPU 说明，用作追溯记录。

## 文档

| 文档 | 说明 |
| --- | --- |
| `CLAUDE.md` | 代码架构与开发参考 |
| `docs/roadmap.md` | 当前阶段和后续计划 |
| `docs/experiments/v1.2_a800.md` | A800 v1.2 实验记录 |
| `docs/experiments/evaluation_protocol.md` | 评测体系说明 |
| `docs/data/dataset_protocol.md` | 数据协议 |
| `docs/reference/2507.22291v2_analysis.md` | AlphaEarth 论文分析 |
