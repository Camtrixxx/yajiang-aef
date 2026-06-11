# Yajiang AEF 评测体系

## 1. 评测目的

AEF 预训练使用 DEM、WorldCover、JRC Water 三个目标做重建约束，但项目最终目标是获得一个雅江地区通用多模态 embedding，后续冻结 encoder 并接简单任务头完成下游任务。

因此评测分成两类：

```text
1. 重建能力：decoder 能不能从 embedding 恢复训练目标
2. 下游能力：冻结 embedding 后，简单任务头是否优于传统 composite 特征
```

模型选型时优先看下游能力，重建能力作为辅助诊断。

## 2. 当前默认评测命令

进入环境：

```bash
cd /data/heyuhang/yajiang-aef
conda activate hyh-dl
```

评测 v1.2 50 epoch：

```bash
bash scripts/run_eval_suite_v1_2.sh
```

评测 100 epoch：

```bash
bash scripts/run_eval_suite_v1_2_continue_100.sh
```

评测 200 epoch：

```bash
bash scripts/run_eval_suite_v1_2_continue_200.sh
```

默认设置：

```text
CUDA_VISIBLE_DEVICES=0
MAX_PATCHES=512
BATCH_SIZE=4
MAX_PIXELS_PER_PATCH=256
```

更大规模评测：

```bash
MAX_PATCHES=1024 BATCH_SIZE=4 MAX_PIXELS_PER_PATCH=256 \
bash scripts/run_eval_suite_v1_2_continue_200.sh
```

## 3. 输出文件

每个评测脚本输出到：

```text
outputs/model_eval/<version>/
```

默认保留：

```text
metrics.json
report.md
report.html
fewshot_curves.png
demo_panels/*.png
```

说明：

| 文件 | 说明 |
| --- | --- |
| `metrics.json` | 结构化核心指标，适合后续自动汇总 |
| `report.md` | 适合实验记录和汇报摘录 |
| `report.html` | 适合浏览器查看 |
| `fewshot_curves.png` | 下游任务 few-shot 曲线 |
| `demo_panels/*.png` | 少量 patch 的直观可视化 |

## 4. 重建指标

重建指标回答：

```text
模型是否学到了能支撑 decoder 恢复目标的空间和语义信息？
```

| 目标 | 指标 | 解释 |
| --- | --- | --- |
| DEM | MAE / R2 | 高程连续值重建，MAE 越低越好，R2 越高越好 |
| WorldCover | macro F1 / macro IoU | 地表覆盖分类重建，macro 指标能减少主类占比影响 |
| JRC Water | binary F1 / binary IoU / boundary F1 | 水体二分类和边界质量 |

不要只看 accuracy。WorldCover 和 JRC Water 类别不均衡，accuracy 容易虚高。

## 5. 下游 probe 指标

下游 probe 指标回答：

```text
冻结 AEF embedding 后，训练一个简单头是否能比传统 composite 特征更好？
```

当前比较：

```text
composite_linear
aef_linear
delta = aef_linear - composite_linear
```

当前任务：

| 任务 | 指标 |
| --- | --- |
| WorldCover | macro F1 |
| JRC binary water | macro F1 |
| DEM | R2 |

少样本设置：

```text
1-shot
5-shot
10-shot
50-shot
```

`delta` 是关键列：

```text
delta > 0: AEF embedding 优于 composite
delta = 0: 两者接近
delta < 0: AEF embedding 暂时没有超过 composite
```

## 6. 当前 v1.2 对比

评测设置统一为 512 patches。

### 重建

| 版本 | DEM MAE | DEM R2 | WorldCover macro F1 | WorldCover macro IoU |
| --- | ---: | ---: | ---: | ---: |
| v1.2 50 epoch | 0.1086 | 0.9750 | 0.5736 | 0.4387 |
| v1.2 continue 100 | 0.0829 | 0.9845 | 0.6917 | 0.5533 |
| v1.2 continue 200 | 0.0712 | 0.9889 | 0.7837 | 0.6596 |

结论：200 epoch 的重建能力最好。

### WorldCover few-shot AEF macro F1

| 版本 | 1-shot | 5-shot | 10-shot | 50-shot |
| --- | ---: | ---: | ---: | ---: |
| v1.2 50 epoch | 0.2031 | 0.3216 | 0.3345 | 0.3267 |
| v1.2 continue 100 | 0.1535 | 0.2284 | 0.2711 | 0.3065 |
| v1.2 continue 200 | 0.1146 | 0.1494 | 0.2191 | 0.2700 |

### DEM few-shot AEF R2

| 版本 | 1-shot | 5-shot | 10-shot | 50-shot |
| --- | ---: | ---: | ---: | ---: |
| v1.2 50 epoch | 0.4182 | 0.3551 | 0.3370 | 0.6592 |
| v1.2 continue 100 | 0.1956 | 0.2524 | 0.1906 | 0.6337 |
| v1.2 continue 200 | 0.1118 | 0.2314 | 0.0561 | 0.5013 |

结论：当前更长训练提升了 decoder 重建，但削弱了 few-shot embedding。若目标是通用 embedding，当前优先选择 v1.2 50 epoch。

## 7. 如何用评测结果选模型

如果应用是：

```text
多模态 embedding + 简单任务头
```

优先排序：

```text
1. 下游 few-shot aef_linear
2. AEF 相对 composite 的 delta
3. 不同 shot 下的稳定性
4. embedding diagnostics
5. 重建指标
```

如果应用是：

```text
重建 / 补全 / 目标解码
```

优先排序：

```text
1. 对应目标的重建指标
2. demo panel 视觉质量
3. 训练 loss 稳定性
4. 下游 probe
```

## 8. 后续建议

评测体系下一步应补强：

1. 独立 eval manifest
   当前评测仍在 supplied manifest 上做诊断，后续应增加新区域、新时间段或人工标注样本。

2. 真实任务头 benchmark
   增加雅江实际应用任务，例如滑坡/裸地/建设用地/河谷水体/高程分层等轻量任务头。

3. checkpoint selection
   训练时周期性导出中间 checkpoint，并用 fixed probe 自动选 embedding 最佳 epoch。

4. baseline 扩展
   除 composite 外，增加单模态 S2、S1、Landsat baseline。
