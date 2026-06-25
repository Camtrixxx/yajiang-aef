# Yajiang AEF Inference Service

独立 AEF 推理服务，目标是给 `agent` 模块提供稳定的模型侧能力。当前版本先以
`sample_indices` 作为输入，后续再接入 “地区 + 时间 -> patch/sample” 的选择服务。

## 启动

```bash
conda activate hyh-dl
scripts/run_aef_inference_server.sh
```

默认地址：

```text
http://localhost:7861
```

可用环境变量覆盖：

```bash
HOST=0.0.0.0 \
PORT=7861 \
AEF_CONFIG=configs/yajiang_v1_2.yaml \
AEF_MANIFEST=data/full_npy/train.jsonl \
AEF_DEPLOY_MODEL=outputs/aef_hyh_yajiang_v1_2/exports/aef_hyh_yajiang_v1_2_deploy.pt \
AEF_CACHE_DIR=outputs/aef_inference_service \
AEF_DEVICE=auto \
scripts/run_aef_inference_server.sh
```

## 接口

### `GET /api/health`

返回服务状态、设备和数据集大小。

### `GET /api/meta`

返回配置路径、模型路径、输入源、目标源等元信息。

### `GET /api/patch-rgb`

直接返回原始输入 patch 渲染后的 RGB PNG。示例：

```text
/api/patch-rgb?sample_index=4&source=s2&period=2025Q3
```

默认 `source=s2`，`period=latest`。当前可用于展示的输入源包括：

- `s2`: Sentinel-2，优先推荐，当前 patch 为 `128 x 128`。
- `landsat`: Landsat，当前 patch 为 `43 x 43`。
- `s1`: Sentinel-1，按双极化组合成伪 RGB。

### `GET /api/patch-rgb-info`

返回 RGB PNG 的元信息，包括原始 `.npy` 路径、实际季度、时间戳和产物 URL。

### `POST /api/infer`

请求示例：

```json
{
  "sample_indices": [4],
  "task": "water",
  "use_cache": true,
  "water_threshold": 0.5,
  "rgb_source": "s2",
  "rgb_period": "2025Q3"
}
```

返回内容包括：

- `summary`: 面向 agent 报告生成的聚合指标。
- `items[].metrics`: 单个 patch 的任务指标。
- `items[].artifacts`: 可直接展示的任务 PNG 图片路径。

图片通过 `/artifacts/...` 访问，缓存写入 `outputs/aef_inference_service/`。

当前支持任务：

- `water`: 水体分类，输出原始 RGB patch、水体概率图、阈值 mask、真值图、正确/错误图、四联对比图、叠加图和水体等级图。
- `landcover`: 地物分类，输出原始 RGB patch、9 类预测图、真值图、正确/错误图、四联对比图、top-1 置信度图、叠加图和类别占比。
- `dem`: DEM 重建，输出原始 RGB patch、用户版地形分析图、真值 DEM、预测 DEM、绝对误差图、四联对比图和回归指标。
- `all`: 同时输出上述任务产物。

水体分类不直接使用 `argmax > 0`，而是对 `jrc_water` 的 101 类 logits 做
softmax 后计算 0-100 水体等级期望，再按 `water_threshold` 生成分类 mask。
`water_target_png` 来自当前 patch 的 `targets/jrc_water.npy`，其中 `> 0` 视为水体，
`255` 为忽略区域。`water_correctness_png` 中绿色表示预测正确，红色表示预测错误。
指标包含 `accuracy`、`precision`、`recall`、`f1` 和 `iou`。

地物分类使用 `worldcover` 的 9 类 logits，经过 softmax 后取 top-1 类别作为分类结果，
同时输出 top-1 置信度。类别编号来自 `data/full_npy/preprocess_meta.json` 中的
WorldCover 重映射：

| class_id | ESA WorldCover code | label |
| --- | --- | --- |
| 0 | 10 | tree_cover |
| 1 | 30 | grassland |
| 2 | 40 | cropland |
| 3 | 50 | built_up |
| 4 | 60 | bare_sparse_vegetation |
| 5 | 70 | snow_and_ice |
| 6 | 80 | permanent_water_bodies |
| 7 | 100 | moss_and_lichen |
| 8 | 20 | shrubland |

当前模型 decoder 输出网格为 `image_size // 2`，默认是 `64 x 64`。服务返回的
`grid_shape` 和 `grid_pixels` 即为报告统计口径。

`landcover_target_png` 来自当前 patch 的 `targets/worldcover.npy`，经过与模型输出一致的
nearest resize 后作为评估真值。`landcover_correctness_png` 中绿色表示预测正确，红色表示预测错误。

DEM 重建使用 `targets/dem.npy` 作为真值。为了让高程任务更适合报告展示，服务会同时输出两组图：

- 用户展示图：`dem_terrain_overview_png`、`dem_hillshade_png`、`dem_contours_png`、
  `dem_elevation_zones_png`、`dem_slope_png`、`dem_profile_png`。这些图强调地形起伏、
  等高线、高程分区、坡度强度和中心线剖面，比直接看 DEM 色带更直观。
- 模型验证图：`dem_target_png`、`dem_reconstruction_png`、`dem_abs_error_png`、
  `dem_compare_png`。这些图用于判断模型重建和真值之间的偏差。

如果 `data/full_npy/preprocess_meta.json` 中存在 DEM 的 z-score 参数，服务会将指标反归一化为米，
指标包括 `mae`、`rmse`、`bias`、`r2`、`pearson_r`、`terrain_relief`、`slope_mean`、
`slope_p95` 和 `profile_relief`；同时保留 `normalized_metrics` 便于模型调试。用户展示图会对预测
DEM 做轻微平滑以降低栅格噪声。当前坡度指标是 `smoothed_elevation_change_per_output_grid_cell`，
用于 patch 内相对陡缓比较，不直接等价于真实坡度角。

## Agent 集成边界

推荐保持三层职责：

1. Agent 负责理解用户意图，得到标准化字段：任务、地区、月份、补充约束。
2. Patch 选择服务负责把地区和月份映射成一个或多个 `sample_indices`。
3. AEF 推理服务负责加载模型、执行推理、返回指标和图像产物。

这样 agent 不需要直接持有模型，也不会把显存生命周期和对话生命周期绑在一起。
