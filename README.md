# yajiang-aef

雅江场景下的多模态时空 embedding 初版训练框架。

当前阶段只做一件事：**先把训练主链路跑通**。  
项目聚焦于开源遥感数据的接入与训练，先完成一个最小可运行版本（MVP），打通：

- 配置读取
- manifest 构建与读取
- dataset 输出 batch
- model forward
- loss 计算
- train step

后续再逐步扩展到冰川、滑坡、水体变化、工程安全等专题任务。

---

## 1. 当前目标

当前版本的目标很明确：

1. 基于雅江 AOI 的开源多模态数据构建训练样本
2. 训练一个多模态时空 embedding 主干模型
3. 为后续扩展更多专题任务打基础

当前不追求：

- 完整工程系统
- 复杂下游任务
- 全量数字底座
- 一次性覆盖所有模态和标签

一句话概括：

**先训起来。**

---

## 2. 当前支持的数据栈（MVP）

### 输入源
- Sentinel-2 L2A
- Sentinel-1 GRD
- HLS

### 目标/辅助源
- DEM
- ESA WorldCover
- JRC Global Surface Water

这套组合用于先搭建雅江初版训练链路，已经可以覆盖：

- 光学信息
- SAR 信息
- 地形信息
- 地表覆盖信息
- 水体信息

---

## 3. 项目结构

```text
yajiang-aef/
├── README.md
├── .gitignore
├── pyproject.toml
├── configs/
│   └── yajiang_v1.yaml
├── scripts/
│   ├── build_manifest.py
│   ├── prepare_data.py
│   └── train.py
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
│       ├── io.py
│       └── geo.py
├── tests/
│   └── __init__.py
└── outputs/
````

---

## 4. 目录说明

### `configs/`

训练配置文件目录。

### `scripts/`

脚本入口：

* `build_manifest.py`：根据 patch 数据生成 `train.jsonl`
* `prepare_data.py`：后续用于数据预处理、patch 构建、统计量生成
* `train.py`：最小训练入口

### `src/config.py`

读取 yaml 配置并转换为训练代码可用的配置对象。

### `src/data/`

数据读取逻辑：

* `manifest.py`：读取和校验 manifest
* `dataset.py`：根据 manifest 组装训练样本

### `src/models/`

模型主干：

* `blocks.py`：STP block
* `bottleneck.py`：bottleneck
* `decoders.py`：条件解码器
* `sensor_encoders.py`：多源传感器编码器
* `time_encoding.py`：时间/窗口/相对时间编码
* `model.py`：主模型

### `src/training/`

训练损失与训练逻辑。

### `src/utils/`

通用工具函数，后续扩展使用。

### `outputs/`

训练输出目录，例如：

* checkpoints
* logs
* embeddings
* 临时结果

---

## 5. 当前最小可运行目标

当前版本的“最小可运行”定义如下：

1. 可以读取配置文件
2. 可以读取 manifest
3. 可以从 patch 数据生成训练样本
4. 可以实例化模型
5. 可以跑通一次 `forward()`
6. 可以计算一次 loss 并完成 `backward()`

也就是说，当前优先保证主链路跑通，而不是一次性覆盖所有功能。

---

## 6. 数据目录组织建议

推荐将代码仓库与数据目录分开管理。

### 代码仓库目录

```text
/workspace/hyh/yajiang-aef
```

### 原始与中间数据目录

```text
/workspace/raw/yajiang_open/
├── aoi/
├── manifests/
│   └── train.jsonl
├── patches/
│   ├── s2/
│   ├── s1/
│   ├── hls/
│   └── targets/
│       ├── dem/
│       ├── worldcover/
│       └── jrc_water/
└── ...
```

### 统计量目录

```text
/workspace/statistics/yajiang_open/
```

---

## 7. 数据准备流程

当前阶段的数据准备主线如下：

### Step 1：确定 AOI

先确定雅江第一版训练区域。

推荐目录：

```text
/workspace/raw/yajiang_open/aoi/yajiang_aoi.geojson
```

### Step 2：准备原始数据

#### 输入源

* Sentinel-2 L2A
* Sentinel-1 GRD
* HLS

#### 目标/辅助源

* DEM
* WorldCover
* JRC GSW

### Step 3：统一预处理

对不同来源数据进行：

* 裁剪到 AOI
* 统一坐标系
* 统一分辨率
* 统一 patch 切分规则
* 统一命名方式

### Step 4：生成 patch

将大图裁成固定大小 patch，并保存为 `.npy` 文件。

当前约定：

* 输入 patch：`[C, H, W]`
* 连续目标：`[C, H, W]` 或 `[H, W]`
* 类别目标：`[H, W]`

### Step 5：生成 manifest

通过 `scripts/build_manifest.py` 生成训练样本清单。

---

## 8. patch 命名建议

### 输入 patch

推荐命名为：

```text
tile_0001_patch_0003_20240115.npy
```

包含：

* tile 或区域编号
* patch 编号
* 日期

### 目标 patch

推荐命名为：

```text
tile_0001_patch_0003.npy
```

因为静态目标不一定需要额外时间戳。

---

## 9. manifest 格式约定

manifest 使用 `jsonl` 格式，每行一个样本。

每个样本包含：

* `sample_id`
* `valid_start_ms`
* `valid_end_ms`
* `sources`

  * `s2`
  * `s1`
  * `hls`
* `targets`

  * `dem`
  * `worldcover`
  * `jrc_water`

### 单条 manifest 示例

```json
{
  "sample_id": "tile_0001_patch_0003",
  "valid_start_ms": 1704067200000,
  "valid_end_ms": 1735603200000,
  "sources": {
    "s2": {
      "type_id": 0,
      "frames": [
        {
          "path": "/workspace/raw/yajiang_open/patches/s2/tile_0001_patch_0003_20240115.npy",
          "timestamp_ms": 1705276800000,
          "valid": true
        }
      ]
    },
    "s1": {
      "type_id": 1,
      "frames": [
        {
          "path": "/workspace/raw/yajiang_open/patches/s1/tile_0001_patch_0003_20240118.npy",
          "timestamp_ms": 1705536000000,
          "valid": true
        }
      ]
    },
    "hls": {
      "type_id": 2,
      "frames": [
        {
          "path": "/workspace/raw/yajiang_open/patches/hls/tile_0001_patch_0003_20240117.npy",
          "timestamp_ms": 1705449600000,
          "valid": true
        }
      ]
    }
  },
  "targets": {
    "dem": {
      "path": "/workspace/raw/yajiang_open/patches/targets/dem/tile_0001_patch_0003.npy",
      "relative_time": 0.0,
      "metadata": [0.0, 0.0, 0.0, 0.0]
    },
    "worldcover": {
      "path": "/workspace/raw/yajiang_open/patches/targets/worldcover/tile_0001_patch_0003.npy",
      "relative_time": 0.0,
      "metadata": [0.0, 0.0, 0.0, 0.0]
    },
    "jrc_water": {
      "path": "/workspace/raw/yajiang_open/patches/targets/jrc_water/tile_0001_patch_0003.npy",
      "relative_time": 0.0,
      "metadata": [0.0, 0.0, 0.0, 0.0]
    }
  }
}
```

---

## 10. 当前模型设计

当前版本只保留 backbone 训练所需核心模块：

* 多源传感器编码器
* 时间编码
* STP 主干
* bottleneck
* 条件解码器

第一阶段主要关注：

* 多模态时序特征编码
* embedding 学习
* 基础重建与抗坍缩训练

当前不做：

* 变化检测头
* few-shot 任务头
* 复杂下游任务
* 大量专题扩展

---

## 11. 当前配置文件说明

主配置文件为：

```text
configs/yajiang_v1.yaml
```

其中最关键的几项包括：

### 输入源定义

* `data.input_sources`
* `model.source_channels`

### 目标源定义

* `data.target_sources`

### 模型主干参数

* `model.stem_dim`
* `model.precision_dim`
* `model.embedding_dim`
* `model.num_blocks`
* `model.num_heads`
* `model.vmf_kappa`

### 训练参数

* `training.epochs`
* `training.lr`
* `training.reconstruction_weight`
* `training.uniformity_weight`
* `training.variance_weight`
* `training.decorrelation_weight`
* `training.orthogonality_weight`

---

## 12. 安装

在当前环境中执行：

```bash
cd /workspace/hyh/yajiang-aef
pip install -e .
```

---

## 13. 构建 manifest

假设 patch 数据目录是：

```text
/workspace/raw/yajiang_open/patches
```

则可执行：

```bash
python scripts/build_manifest.py \
  --patch-root /workspace/raw/yajiang_open/patches \
  --output /workspace/raw/yajiang_open/manifests/train.jsonl
```

---

## 14. 训练（最小版本）

当前训练目标是先跑通最小版本：

```bash
python scripts/train.py --config configs/yajiang_v1.yaml
```

后续再扩展为：

* 单卡完整训练
* 多卡训练
* 更完整的 trainer 和 logging
* 验证与可视化
* embedding 导出

---

## 15. GPU 使用建议

当前可用 GPU：

* GPU 6
* GPU 7

因此第一阶段建议：

### 单卡冒烟测试

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/yajiang_v1.yaml
```

### 双卡扩展（后续）

等单卡跑稳后，再考虑双卡：

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 scripts/train.py --config configs/yajiang_v1.yaml
```

第一阶段不要急着上复杂 DDP，先保证：

* dataset 没问题
* batch 没问题
* forward 没问题
* loss 没问题

---

## 16. 当前开发顺序建议

推荐按下面顺序推进：

### Step 1

完成 `config.py`

### Step 2

完成 `manifest.py` 和 `dataset.py`

### Step 3

完成最小 `losses.py`

### Step 4

完成 `train.py`，先跑通一个 batch

### Step 5

补充数据预处理和 patch 构建流程

### Step 6

扩展更复杂的损失、评估和专题任务

---

## 17. 后续扩展方向

等 MVP 跑通后，可以逐步加入：

* 更完整的 HLS 时序支持
* Dynamic World
* 冰川边界
* 滑坡事件数据
* 更复杂的 temporal window augmentation
* 变化检测和专题下游任务
* 多 GPU 分布式训练
* embedding 检索与可视化

---

## 18. 当前状态总结

这是一个面向雅江场景的多模态时空 embedding 最小训练框架。

它当前只做一件事：

**先把训练主链路跑通。**

```
```
