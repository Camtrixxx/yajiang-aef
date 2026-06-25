# yajiang-aef Roadmap

## 当前阶段

项目已经完成从早期真实数据联调、Ascend/NPU 多卡训练，到 NVIDIA A800 CUDA 8 卡训练的迁移。

当前主线：

```text
v1.2 / v1.3 A800/CUDA 8-card DDP
S2 + S1 + Landsat -> DEM + WorldCover + JRC Water
目标：雅江地区通用多模态 embedding
```

最新实验记录：

```text
docs/experiments/v1.2_a800.md
```

## 已完成阶段

### v0.1 Mock 数据冒烟测试

目标：打通最小训练主链路。

状态：已完成。

完成内容：

- 配置读取
- mock dataset
- forward / loss / backward
- 基础训练循环

### v0.2 小规模真实数据联调

目标：从 mock 数据切换到真实遥感数据，验证数据协议和 target 格式。

状态：已完成。

完成内容：

- debug manifest
- 小规模 `.npy` 数据
- 单卡训练联调
- DEM / WorldCover / JRC Water target shape 与 ignore mask 验证

### v0.3 Ascend/NPU 多卡 baseline

目标：在华为 Ascend 环境完成第一版完整多卡训练。

状态：已完成，作为历史 baseline 保留。

代表版本：

```text
v0.3a
v0.3c
```

相关文档：

```text
docs/experiments/v0.3a.md
docs/experiments/v0.3c.md
```

### v1.1 Landsat 输入与迁移准备

目标：加入 Landsat，准备从旧环境迁移到新 GPU 环境。

状态：已完成，作为过渡版本保留。

### v1.2 A800 主线

目标：适配 NVIDIA A800 8 卡训练，建立当前主线实验。

状态：已完成。

完成内容：

- CUDA/A800 训练脚本
- bf16 AMP / TF32 / cudnn benchmark
- DataLoader persistent workers / prefetch
- DDP resume
- source preprocessing
- target loss weights / class weights
- 50 / 100 / 200 epoch 训练与评测

## 当前结论

当前 50 / 100 / 200 epoch 对比显示：

| 版本 | 重建能力 | few-shot embedding |
| --- | --- | --- |
| 50 epoch | 较好 | 当前最好 |
| 100 epoch | 更好 | 下降 |
| 200 epoch | 最好 | 继续下降 |

这说明模型继续训练后更偏向 decoder 重建目标，但冻结 embedding 接简单任务头的能力下降。

因此，如果目标是通用 embedding，下一阶段的核心不是继续单纯拉长 epoch，而是让训练目标更直接服务于 embedding。

## 当前优化：v1.3 embedding-first

### 目标

训练一个更适合冻结后接简单任务头的雅江多模态 embedding。

状态：代码已实现，待正式 8 卡训练和评测。

v1.3 新增内容：

- teacher-student consistency；
- student 输入随机缺源、缺时间帧、缺前半段或后半段时序；
- AlphaEarth-style batch orthogonal uniformity；
- 每 10 epoch 保存中间 checkpoint，方便按 downstream probe 选模型；
- checkpoint 到 deploy 格式的导出工具。

### 完成标准

- few-shot AEF 指标稳定超过 composite baseline；
- WorldCover / DEM / JRC binary 至少两个任务有正向 delta；
- 1/5/10/50-shot 曲线稳定，不只在高 shot 下有效；
- 重建指标不严重退化；
- 有固定 eval manifest 或固定 benchmark 支持 checkpoint selection。

### 优先事项

#### 1. 建立 embedding-first checkpoint selection

训练过程中定期评测中间 checkpoint，不只保存训练 loss 最低的 `best.pt`。

建议指标：

```text
mean downstream delta
WorldCover few-shot macro F1
DEM few-shot R2
cross-shot stability
```

#### 2. 真实下游任务头 benchmark

当前 probe 使用 WorldCover / DEM / JRC 作为诊断任务。后续应补充更接近应用场景的轻量任务头，例如：

```text
河谷/水体识别
裸地/建设用地区分
高程分层
植被覆盖等级
滑坡或地灾相关样本
```

#### 3. 调整训练目标

已实现第一版：

```text
teacher 完整输入
student 扰动输入
loss = reconstruction + consistency + batch_uniformity + existing regularizers
```

后续候选方向：

- 降低 decoder 重建 loss 权重；
- 加入 masked source reconstruction；
- 使用 projection head，把 decoder-specific 信息和 embedding 表征解耦；
- 阶段性冻结 decoder 或 encoder，观察 downstream probe 变化。

#### 4. 扩展 baseline

当前主要比较 AEF vs composite。后续增加：

```text
S2-only
S1-only
Landsat-only
S2+S1
S2+S1+Landsat composite
```

这样能判断 AEF 的增益来自模型表征，而不是简单多源拼接。

#### 5. 独立 eval manifest

当前评测是 diagnostic on supplied manifest。下一步应准备独立评测集：

```text
新区域
新时间段
人工标注样本
任务特定样本
```

## 建议开发顺序

1. 固定当前 v1.2 三个 checkpoint 和评测结果；
2. 写一个 metrics 汇总脚本，自动对比多个 `outputs/model_eval/*/metrics.json`；
3. 增加 checkpoint probe 流程；
4. 设计 v1.3 训练目标；
5. 跑 50 epoch 对照实验；
6. 如果 downstream probe 提升，再考虑 100/200 epoch；
7. 建立独立 eval manifest。

## 当前可用命令

训练：

```bash
cd /data/heyuhang/yajiang-aef
conda activate hyh-dl

bash scripts/run_v1_2.sh
bash scripts/run_v1_2_continue_100.sh
bash scripts/run_v1_2_continue_200.sh
bash scripts/run_v1_3.sh
```

评测：

```bash
bash scripts/run_eval_suite_v1_2.sh
bash scripts/run_eval_suite_v1_2_continue_100.sh
bash scripts/run_eval_suite_v1_2_continue_200.sh
bash scripts/run_eval_suite_v1_3.sh
```

导出中间 checkpoint：

```bash
python scripts/export_checkpoint_to_deploy.py \
  --config configs/yajiang_v1_3.yaml \
  --checkpoint outputs/aef_hyh_yajiang_v1_3/checkpoints/epoch_010.pt \
  --output outputs/aef_hyh_yajiang_v1_3/exports/aef_hyh_yajiang_v1_3_epoch_010_deploy.pt
```

## 备注

历史文档中仍有 `/workspace/...`、Ascend/NPU、v0.2/v0.3 相关路径。这些内容表示当时实验环境，不代表当前主线。当前主线以 README、CLAUDE.md、`docs/experiments/v1.2_a800.md` 和本 roadmap 为准。
