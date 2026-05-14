# yajiang-aef

雅江区域的 AlphaEarth-style 多源遥感表征学习项目。

本项目基于 AlphaEarth Foundations 的思想，训练一个区域级 AEF embedding 模型：输入 Sentinel-2 / Sentinel-1 多时相遥感数据，学习一个可用于重建和下游任务的通用遥感 embedding。

当前主线版本是 `v0.3c`，已经完成：

- 华为 Ascend NPU 环境适配；
- S2 + S1 多源输入训练；
- DEM / WorldCover / JRC Water 三目标重建；
- 4 卡 DDP 正式训练；
- deploy 模型导出；
- 简化测评体系；
- 在线 Web 可视化演示。

当前新增的 `v1.1` 分支配置在 `v0.3c` 基础上接入了 Landsat 作为第三个输入源：

```text
S2 + S1 + Landsat -> DEM / WorldCover / JRC Water
```

## 1. 环境与路径

推荐在 Docker 容器中运行：

```bash
docker exec -it heyuhang-dl bash
cd /workspace/hyh/yajiang-aef
```

路径对应关系：

```text
宿主机: /data/heyuhang/hyh/yajiang-aef
容器内: /workspace/hyh/yajiang-aef
```

当前服务器为华为 Ascend 环境，默认使用后四张卡中的物理 `4` 号卡进行评测和演示：

```bash
ASCEND_RT_VISIBLE_DEVICES=4
```

正式训练脚本默认使用：

```text
NPU_IDS=4,5,6,7
```

## 2. 当前数据

当前训练数据目录：

```text
data/full_npy
```

manifest：

```text
data/full_npy/train.jsonl
```

样本规模：

```text
1708 patches
```

输入源：

```text
s2: Sentinel-2, 6 channels
s1: Sentinel-1, 2 channels
landsat: Landsat, 6 channels
```

监督目标：

```text
dem: continuous, 1 channel
worldcover: categorical, 9 classes
jrc_water: categorical, 101 classes, 255 as ignore index
```

当前训练仍使用全量 patch，不强行拆分预训练验证集/测试集。

## 3. 模型结构

当前 AEF 模型主流程：

```text
S2 / S1 多时相输入
        ↓
SensorEncoderBank
        ↓
STP blocks
        ↓
vMF bottleneck
        ↓
AEF embedding map
        ↓
DEM / WorldCover / JRC Water decoders
```

核心代码：

```text
src/models/model.py
src/models/sensor_encoders.py
src/models/blocks.py
src/models/bottleneck.py
src/models/decoders.py
src/training/losses.py
```

v0.3c 关键配置：

```text
configs/yajiang_v0_3_c.yaml
```

v1.1 Landsat 配置：

```text
configs/yajiang_v1_1.yaml
```

主要参数：

```text
image_size: 128
max_frames: 16
precision_dim: 256
embedding_dim: 128
num_blocks: 4
num_heads: 4
vmf_kappa: 2000.0
```

## 4. v0.3c 训练

启动 4 卡训练：

```bash
cd /workspace/hyh/yajiang-aef
bash scripts/run_v0_3_c.sh
```

显式指定后四张卡：

```bash
NPU_IDS=4,5,6,7 NPROC_PER_NODE=4 bash scripts/run_v0_3_c.sh
```

训练输出：

```text
outputs/aef_hyh_yajiang_v0_3_c
```

主要模型文件：

```text
checkpoints/best.pt
checkpoints/final.pt
checkpoints/latest.pt
exports/aef_hyh_yajiang_v0_3_c_deploy.pt
```

部署或演示优先使用：

```text
outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt
```

## 4.1 v1.1 Landsat 接入

Landsat 原始数据目录：

```text
/workspace/raw/yajiang/landsat
```

转换后的数据位置：

```text
data/full_npy/patch_xxxxxx/inputs/landsat/YYYYQn.npy
```

转换命令：

```bash
cd /workspace/hyh/yajiang-aef
python scripts/prepare_landsat_npy.py --skip-existing
python scripts/build_full_manifest.py
```

当前已转换：

```text
1708 patches
22204 Landsat frames
13 frames per patch
```

启动 v1.1 训练脚本：

```bash
bash scripts/run_v1_1.sh
```

显式指定后四张卡：

```bash
NPU_IDS=4,5,6,7 NPROC_PER_NODE=4 bash scripts/run_v1_1.sh
```

## 5. 测评体系

当前评测体系已经简化为两个核心问题：

```text
1. 模型重建能力怎么样？
2. AEF embedding 对下游任务有没有帮助？
```

运行评测：

```bash
cd /workspace/hyh/yajiang-aef
bash scripts/run_eval_suite_v0_3_c.sh
```

默认配置：

```text
MAX_PATCHES=128
MAX_PIXELS_PER_PATCH=128
ASCEND_RT_VISIBLE_DEVICES=4
```

评测输出：

```text
outputs/model_eval/v0_3_c
```

默认只保留清晰核心文件：

```text
metrics.json
report.md
report.html
fewshot_curves.png
demo_panels/*.png
```

### 5.1 重建能力

| 目标 | 指标 | 说明 |
| --- | --- | --- |
| DEM | MAE / R2 | 高程连续值重建 |
| WorldCover | macro F1 / macro IoU | 地表覆盖分类重建 |
| JRC Water | F1 / IoU / boundary F1 | 水体二分类与轮廓质量 |

### 5.2 下游任务能力

下游评测比较：

```text
composite linear
AEF linear
```

其中：

- `composite` 是 S2/S1 多时相平均合成特征；
- `AEF` 是模型输出的 embedding map；
- 两者使用同样的少量标签和同样的 linear probe。

当前下游任务：

```text
worldcover: macro F1
jrc_binary: macro F1
dem: R2
```

few-shot 设置：

```text
1-shot
5-shot
10-shot
50-shot
```

当前默认下游评测范围是 `train.jsonl` 的前 128 个 patch，即：

```text
patch_000000 ~ patch_000127
```

每个 patch 最多抽 128 个像元。

更多说明见：

```text
docs/experiments/evaluation_protocol.md
```

## 6. 在线 Web 演示

启动服务：

```bash
docker exec -it heyuhang-dl bash
cd /workspace/hyh/yajiang-aef
bash scripts/run_demo_server.sh
```

默认端口：

```text
7860
```

本地浏览器通过 VS Code 端口转发访问：

```text
http://localhost:7860
```

Web 页面支持：

- 手动选择 patch index；
- 随机选择 patch；
- 展示 S2 RGB 输入；
- 展示 AEF embedding PCA；
- 展示 DEM target / reconstruction / error；
- 展示 WorldCover target / reconstruction；
- 展示 JRC Water target + predicted water contour；
- 展示单样本指标；
- 展示整体下游任务曲线。

演示说明：

```text
docs/experiments/demo_server.md
```

停止服务：

```bash
pkill -f scripts/serve_demo.py
```

## 7. 重要脚本

### 数据与 manifest

```text
scripts/prepare_full_npy.py
scripts/prepare_jrc_water_npy.py
scripts/prepare_landsat_npy.py
scripts/build_full_manifest.py
scripts/build_debug_small_manifest.py
```

### 训练

```text
scripts/train.py
scripts/train_with_manifest.py
scripts/run_v0_3_c.sh
scripts/run_v1_1.sh
```

### 测评与展示

```text
scripts/evaluate_model_suite.py
scripts/run_eval_suite_v0_3_c.sh
scripts/demo_visualize.py
scripts/serve_demo.py
scripts/run_demo_server.sh
```

## 8. 文档

```text
docs/reference/2507.22291v2_analysis.md
docs/experiments/v0.3c.md
docs/experiments/evaluation_protocol.md
docs/experiments/demo_server.md
docs/data/dataset_protocol.md
docs/roadmap.md
```

## 9. 当前结果摘要

v0.3c 当前结论：

- DEM 重建较好，R2 约 `0.93`；
- JRC 水体二分类重建较好，F1 约 `0.99`；
- WorldCover macro F1 较低，小类识别仍弱；
- 下游水体任务中，AEF embedding 相比 composite 有明显增益；
- WorldCover 下游基本持平；
- DEM 下游回归暂时不稳定。

适合当前阶段展示的主线是：

```text
S2/S1 多源输入
  → AEF embedding
  → DEM / WorldCover / JRC Water 重建
  → 水体相关下游任务有增益
```

## 10. 后续方向

优先级较高的下一步：

1. 构建独立 eval manifest  
   使用新区域、新时间段或人工标注数据，作为固定评测集。

2. 加强 WorldCover 类别学习  
   处理类别不均衡、小类样本不足和空间边界问题。

3. 优化下游任务协议  
   固定评测 patch 范围，增加版本对比表。

4. 导出 embedding field  
   将每个 patch 的 `embedding_map` 输出为 `.npy` 或 GeoTIFF，供 GIS 和下游模型使用。

5. 增加缺源一致性训练  
   提升缺 S2、缺 S1、云污染场景下的稳定性。
