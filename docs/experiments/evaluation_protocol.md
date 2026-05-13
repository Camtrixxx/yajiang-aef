# Yajiang AEF 简化测评体系

## 1. 原则

当前预训练模型不强行划分 validation/test。模型训练可以使用全量 patch；评测作为训练后的统一诊断流程，用于比较不同模型版本、判断模型是否值得继续迭代。

默认评测只看两个问题：

```text
1. 模型重建能力怎么样？
2. AEF embedding 对下游任务有没有帮助？
```

不再把 embedding 诊断、混淆矩阵、大量中间图作为默认报告内容，避免历史文件堆积和解释负担。

## 2. 运行命令

```bash
docker exec -it heyuhang-dl bash
cd /workspace/hyh/yajiang-aef
bash scripts/run_eval_suite_v0_3_c.sh
```

默认使用：

```text
ASCEND_RT_VISIBLE_DEVICES=4
MAX_PATCHES=128
MAX_PIXELS_PER_PATCH=128
```

如果要改评测 patch 数：

```bash
MAX_PATCHES=256 bash scripts/run_eval_suite_v0_3_c.sh
```

脚本默认会清理并重新生成输出目录：

```text
outputs/model_eval/v0_3_c
```

## 3. 输出文件

默认只保留：

```text
metrics.json
report.md
report.html
fewshot_curves.png
demo_panels/*.png
```

说明：

- `metrics.json`：结构化核心指标；
- `report.md`：适合实验记录和汇报摘录；
- `report.html`：适合浏览器打开；
- `fewshot_curves.png`：下游任务能力曲线；
- `demo_panels/*.png`：少量直观样本展示。

## 4. 指标一：重建能力

重建能力评估 decoder 能不能从 AEF embedding 恢复训练目标。

默认保留三个目标：

| 目标 | 指标 | 解释 |
| --- | --- | --- |
| DEM | MAE / R2 | 高程连续值重建，MAE 越低越好，R2 越高越好 |
| WorldCover | macro F1 / macro IoU | 地表覆盖分类重建，macro 指标能减少主类占比影响 |
| JRC Water | F1 / IoU / boundary F1 | 水体二分类和轮廓质量 |

不默认展示 accuracy，因为 WorldCover 和 JRC Water 类别极不均衡，accuracy 容易虚高。

## 5. 指标二：下游任务能力

下游能力评估 AEF embedding 是否比传统 S2/S1 composite 特征更适合少标签任务。

当前比较：

```text
composite linear
AEF linear
```

当前任务：

```text
worldcover: macro F1
jrc_binary: macro F1
dem: R2
```

当前少样本设置：

```text
1-shot
5-shot
10-shot
50-shot
```

报告中的 `AEF - composite` 是关键列：

```text
大于 0：AEF embedding 有增益
接近 0：两者差不多
小于 0：AEF embedding 暂时没有超过传统 composite
```

## 6. 当前 v0.3c 结果解读

当前 `outputs/model_eval/v0_3_c/report.md` 中的核心结论是：

- DEM 重建较好：R2 约 `0.93`；
- 水体二分类重建较好：F1 约 `0.99`，boundary F1 约 `0.96`；
- WorldCover 重建较弱：macro F1 约 `0.19`，说明类别不均衡和小类识别仍是问题；
- 下游水体任务上 AEF embedding 明显优于 composite；
- 下游 WorldCover 基本持平或略弱；
- 下游 DEM 回归暂时没有体现出稳定优势。

因此当前模型更适合展示：

```text
S2/S1 输入 → AEF embedding → 水体相关任务
```

而不是声称它已经全面优于传统特征。

## 7. 在线演示

在线演示服务：

```bash
cd /workspace/hyh/yajiang-aef
bash scripts/run_demo_server.sh
```

浏览器访问：

```text
http://localhost:7860
```

页面只保留：

- 当前 patch 可视化；
- 单样本重建指标；
- 下游 few-shot 曲线。

## 8. 后续建议

后续如果要让评测更严谨，优先做两件事：

1. 构建独立 eval manifest  
   可以来自新区域、新时间段或人工标注样本，但不影响预训练仍使用全量数据。

2. 做版本对比表  
   把 `v0.3c / v0.4 / v0.5` 的 `metrics.json` 汇总，重点比较这两组指标：

```text
重建能力
下游任务能力
```
