# yajiang-aef

雅江区域 AlphaEarth-style 多源遥感表征学习。

输入 Sentinel-2 / Sentinel-1 / Landsat 多时相遥感影像，训练一个通用遥感 embedding，支持 DEM、WorldCover、JRC Water 三目标重建，embedding 可用于下游 few-shot 任务。

**硬件**：华为 Ascend NPU（4 卡 DDP），CUDA 兼容路径备选。

## 快速开始

```bash
# 进入容器
docker exec -it heyuhang-dl bash
cd /workspace/hyh/yajiang-aef

# v0.3c 训练（S2 + S1）
bash scripts/run_v0_3_c.sh

# v1.1 训练（S2 + S1 + Landsat）
bash scripts/run_v1_1.sh

# 指定 NPU 卡
NPU_IDS=4,5,6,7 NPROC_PER_NODE=4 bash scripts/run_v0_3_c.sh

# 评测
bash scripts/run_eval_suite_v0_3_c.sh

# Web 演示
bash scripts/run_demo_server.sh          # 访问 http://localhost:7860
pkill -f scripts/serve_demo.py           # 停止
```

## 数据

| 项目 | 说明 |
| --- | --- |
| 数据目录 | `data/full_npy` |
| 训练样本 | 1708 patches |
| 输入 | S2 (6ch) / S1 (2ch) / Landsat (6ch) |
| 目标 | DEM (连续) / WorldCover (9类) / JRC Water (101类, ignore_index=255) |
| Manifest | JSONL，一行一个 patch |

数据准备：

```bash
python scripts/prepare_landsat_npy.py --skip-existing
python scripts/build_full_manifest.py
```

## 模型结构

```
S2 / S1 / Landsat 多时相输入
        ↓
SensorEncoderBank（每源独立编码，1×1 adapter + stride-2 stem）
        ↓
STPBlock ×4（Space-Time-Precision 三路注意力）
        ↓
vMF Bottleneck（训练跳 L2 norm + 高斯噪声，推理球面投影）
        ↓
AEF Embedding Map [B, 128, H/2, W/2]
        ↓
ContinuousDecoder（DEM）/ CategoricalDecoder（WorldCover, JRC Water）
```

核心代码：`src/models/`、`src/training/`、`src/data/`、`src/eval/`

### 关键参数

| 参数 | 值 |
| --- | --- |
| image_size | 128 |
| max_frames | 16 |
| precision_dim | 256 |
| embedding_dim | 128 |
| num_blocks | 4 |
| num_heads | 4 |
| vmf_kappa | 2000 |
| batch_size | 2 / GPU |
| optimizer | AdamW, lr=1e-4, wd=0.01 |

## 训练输出

```
outputs/aef_hyh_yajiang_v0_3_c/
├── checkpoints/         best.pt / final.pt / latest.pt
├── exports/             aef_hyh_yajiang_v0_3_c_deploy.pt
└── logs/                train.log
```

单卡调试：

```bash
PYTHONPATH=. ASCEND_RT_VISIBLE_DEVICES=4 \
python scripts/train_with_manifest.py \
  --config configs/yajiang_v0_3_c.yaml \
  --manifest data/full_npy/train.jsonl
```

## 评测

两个核心问题：**重建能力**和**下游任务增益**。

| 目标 | 指标 |
| --- | --- |
| DEM | MAE / R² |
| WorldCover | macro F1 / macro IoU |
| JRC Water | F1 / IoU / boundary F1 |

下游任务：few-shot linear probe，比较 AEF embedding vs composite baseline（多时相平均合成特征），涵盖 worldcover / jrc_binary / dem，1~50 shot。

输出：`outputs/model_eval/v0_3_c/` 下的 `metrics.json`、`report.md`、`report.html`、`fewshot_curves.png`。

## v0.3c 结果

| 目标 | 指标 | 结果 |
| --- | --- | --- |
| DEM | R² | 0.93 |
| JRC Water | F1 | 0.99 |
| WorldCover | macro F1 | 0.19 |

- JRC Water 下游任务 AEF 明显优于 composite
- WorldCover / DEM 下游增益不明显
- WorldCover 类别不均衡仍是主要瓶颈

## 项目版本

| 版本 | 输入 | 状态 |
| --- | --- | --- |
| v0.3c | S2 + S1 | 主线，已训练 |
| v1.1 | S2 + S1 + Landsat | 实验 |

## 文档

| 文档 | 说明 |
| --- | --- |
| `CLAUDE.md` | 代码架构与开发参考 |
| `docs/roadmap.md` | 版本规划 |
| `docs/experiments/v0.3c.md` | v0.3c 实验记录 |
| `docs/experiments/evaluation_protocol.md` | 评测体系说明 |
| `docs/data/dataset_protocol.md` | 数据协议 |
| `docs/reference/2507.22291v2_analysis.md` | AlphaEarth 论文分析 |
