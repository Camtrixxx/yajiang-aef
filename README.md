# yajiang-aef

雅江场景下的多模态时空 embedding 初版训练框架。

这个仓库当前聚焦一件事：**先把 AEF 风格的训练主链路跑通**，包括多源输入编码、时空主干、embedding bottleneck、条件解码重建，以及最小训练循环。

## 当前状态

目前已经完成：

- 配置读取
- manifest 构建与读取
- dataset 组 batch
- AEF 初版模型骨架
- reconstruction + regularization loss
- 最小 trainer
- dummy 数据训练链路打通
- 真实 manifest 训练入口预留

换句话说，这个仓库已经不是单纯的模型代码草稿，而是一个**可以开始做小规模真实数据联调**的初版训练工程。

## 当前支持的数据栈

### 输入源
- Sentinel-2 L2A
- Sentinel-1 GRD
- HLS

### 目标源
- DEM
- ESA WorldCover
- JRC Global Surface Water

当前这套组合主要用于先跑通雅江场景下的多模态训练闭环，覆盖：

- 光学信息
- SAR 信息
- 地形信息
- 地表覆盖信息
- 水体信息

## 项目结构

```text
yajiang-aef/
├── README.md
├── pyproject.toml
├── configs/
│   ├── yajiang_v1.yaml
│   └── yajiang_v1_example.yaml
├── scripts/
│   ├── build_manifest.py
│   ├── prepare_data.py
│   ├── train.py
│   └── train_with_manifest.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── manifest.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── bottleneck.py
│   │   ├── decoders.py
│   │   ├── model.py
│   │   ├── sensor_encoders.py
│   │   └── time_encoding.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── geo.py
│       └── io.py
└── tests/
当前模型包含的核心模块
多源传感器编码器

时间编码 / 窗口编码 / 相对时间编码

STP 主干

vMF bottleneck

条件解码器

reconstruction + anti-collapse regularization

当前版本的目标是先训练一个可用的时空 embedding backbone，暂时不追求复杂下游任务头。

快速开始
1. 安装
Bash

cd /workspace/hyh/yajiang-aef
pip install -e .
2. 无真实数据时先跑 dummy 训练
Bash

PYTHONPATH=/workspace/hyh/yajiang-aef CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/yajiang_v1.yaml
这条命令会使用 DummyYajiangDataset 跑通：

配置读取

模型实例化

forward

loss

backward

epoch 训练循环

适合作为 smoke test。

3. 有 manifest 后跑真实数据训练
Bash

PYTHONPATH=/workspace/hyh/yajiang-aef CUDA_VISIBLE_DEVICES=6 python scripts/train_with_manifest.py --config configs/yajiang_v1.yaml --manifest /path/to/train.jsonl
数据组织
推荐将代码仓库与数据目录分开：


/workspace/hyh/yajiang-aef
/workspace/raw/yajiang_open/
推荐的 patch 组织方式：


data_root/
  patch_0001/
    inputs/
      s2/
      s1/
      hls/
    targets/
      dem.npy
      worldcover.npy
      jrc_water.npy
当前 dataset.py 默认支持：

.npy

.npz

.pt

manifest
manifest 使用 jsonl 格式，每行一个样本。

每个样本至少包含：

sample_id

valid_start_ms

valid_end_ms

inputs

targets

仓库里已经提供：

src/data/manifest.py

scripts/build_manifest.py

用于构建和读取 manifest。

当前配置文件
主配置文件：


configs/yajiang_v1.yaml
重点字段包括：

数据相关
data.input_sources

data.target_sources

data.image_size

data.max_frames

data.batch_size

模型相关
model.source_channels

model.stem_dim

model.precision_dim

model.embedding_dim

model.num_blocks

model.num_heads

model.vmf_kappa

训练相关
training.epochs

training.lr

training.weight_decay

training.reconstruction_weight

training.uniformity_weight

training.variance_weight

training.decorrelation_weight

training.orthogonality_weight

当前已验证的能力
当前仓库已经完成的验证重点是：

dummy 数据训练链路可运行

loss 可以正常计算

训练循环可以稳定执行多个 epoch

代码结构已经具备接真实 manifest 的基础

当前限制
当前版本仍然是第一阶段工程，主要限制包括：

真实数据训练和评估还在继续联调

checkpoint / resume / logging 仍可继续补强

多卡训练还没有作为主路径稳定验证

下游任务头和专题任务还未展开

下一步计划
用小规模真实 patch + manifest 跑通训练

做小样本 overfit 检查

增加 checkpoint 保存与 resume

增加更完整的日志与评估

再考虑 DDP / 加速 / 更大规模训练

说明
这个仓库当前更适合被看作：

“雅江场景下 AEF 风格 backbone 的初版训练工程”