# CLAUDE.md

This file provides guidance to Claude Code and other coding agents when working in this repository.

## Project overview

Yajiang-AEF is an AlphaEarth-style multimodal spatiotemporal remote sensing representation learning project for the Yajiang region. The model takes multi-temporal Sentinel-2, Sentinel-1, and Landsat imagery as input, produces a vMF-bottlenecked AEF embedding map, and learns to reconstruct DEM, WorldCover, and JRC Water targets.

The intended application is not only reconstruction. The main goal is to learn a regional multimodal embedding that can be frozen and reused with simple downstream task heads. For model selection, few-shot / linear-probe downstream metrics are more important than reconstruction loss alone.

Current primary training environment:

```text
Path: /data/heyuhang/yajiang-aef
Conda env: hyh-dl
Hardware: NVIDIA A800, normally 8 CUDA GPUs
Primary config: configs/yajiang_v1_2.yaml
```

Ascend/NPU scripts and historical docs remain in the repository for reproducibility of older v0.2/v0.3 experiments, but CUDA/A800 is the current mainline.

## Commands

### Environment

```bash
cd /data/heyuhang/yajiang-aef
conda activate hyh-dl
```

### Training

```bash
# v1.2, 50 epochs, 8-card CUDA DDP by default
bash scripts/run_v1_2.sh

# Continue v1.2 from 50 to 100 epochs
bash scripts/run_v1_2_continue_100.sh

# Continue v1.2 from 100 to 200 epochs
bash scripts/run_v1_2_continue_200.sh
```

GPU selection:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_v1_2.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_v1_2.sh
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_v1_2.sh
```

Resume manually:

```bash
torchrun --nproc_per_node=8 --master_port=29614 scripts/train_with_manifest.py \
  --config configs/yajiang_v1_2_continue_200.yaml \
  --manifest data/full_npy/train.jsonl \
  --resume outputs/aef_hyh_yajiang_v1_2_continue_100/checkpoints/best.pt \
  --device auto
```

Training outputs go to:

```text
outputs/aef_hyh_yajiang_v1_2/
outputs/aef_hyh_yajiang_v1_2_continue_100/
outputs/aef_hyh_yajiang_v1_2_continue_200/
```

Each contains `checkpoints/{best,final,latest}.pt`, `exports/*_deploy.pt`, and logs.

### Evaluation

```bash
bash scripts/run_eval_suite_v1_2.sh
bash scripts/run_eval_suite_v1_2_continue_100.sh
bash scripts/run_eval_suite_v1_2_continue_200.sh
```

Override evaluation scale:

```bash
MAX_PATCHES=1024 BATCH_SIZE=4 MAX_PIXELS_PER_PATCH=256 \
bash scripts/run_eval_suite_v1_2_continue_200.sh
```

Evaluation outputs:

```text
outputs/model_eval/v1_2/
outputs/model_eval/v1_2_continue_100/
outputs/model_eval/v1_2_continue_200/
```

Important outputs are `metrics.json`, `report.md`, `report.html`, `fewshot_curves.png`, and `demo_panels/*.png`.

### Data preparation

```bash
python scripts/build_full_manifest.py
python scripts/prepare_landsat_npy.py \
  --src-root /path/to/raw/yajiang/landsat \
  --dst-root data/full_npy \
  --skip-existing
python scripts/prepare_jrc_water_npy.py \
  --src-root /path/to/raw/yajiang/jrc_water \
  --dst-root data/full_npy \
  --skip-existing
```

Some older preparation scripts still have `/workspace/...` defaults from previous machines and require GIS dependencies such as `rasterio` or `opencv-python`. Prefer explicit source and destination arguments when re-running data conversion on a new host.

### Quick checks

```bash
python -m py_compile \
  scripts/train_with_manifest.py \
  scripts/build_full_manifest.py \
  src/data/dataset.py \
  src/models/model.py \
  src/models/sensor_encoders.py \
  src/training/losses.py \
  src/training/trainer.py
```

## Architecture

### Data flow

```text
S2 / S1 / Landsat .npy frames
  -> YajiangAEFDataset
  -> source_frames [B, S, T, C, H, W] with masks, timestamps, type_ids
  -> AEFModel.forward()
  -> AEFOutput(embedding_map, embedding, reconstructions)
  -> compute_total_loss()
```

### Model internals (`src/models/`)

1. `SensorEncoderBank` (`sensor_encoders.py`): one per-source adapter and stride-2 stem. Source types: `s2=0`, `s1=1`, `hls=2`, `landsat=3`. Output spatial size is H/2 x W/2.
2. `STPBlock` (`blocks.py`): Space-Time-Precision block with precision, time, and space paths fused through residual blocks.
3. Time/window encoding (`time_encoding.py`): absolute time, valid window, and relative time encoders. In v1.2, absolute `time_encoder` is frozen unless `model.use_time_codes: true`.
4. `VMFBottleneck` (`bottleneck.py`): 1x1 projection to `embedding_dim`. Training can skip L2 normalization; inference produces normalized embedding maps.
5. Decoders (`decoders.py`): continuous decoder for DEM and categorical decoders for WorldCover / JRC Water.

### Data system (`src/data/`)

The manifest is JSONL with one record per patch. Each record contains:

```text
sample_id
valid_start_ms / valid_end_ms
inputs[source].frames[path, timestamp_ms]
targets[name].path
split
```

`YajiangAEFDataset` loads `.npy/.npz/.pt`, normalizes sources according to `data.source_preprocessing`, pads temporal frames to `max_frames`, and resizes targets to decoder output resolution.

### Training (`src/training/`)

`Trainer.fit()` supports:

- CUDA bf16 autocast through `training.amp` and `training.amp_dtype`;
- resume via `--resume`;
- optimizer/scaler/state loading from checkpoint;
- saving `best.pt`, `latest.pt`, `final.pt`;
- exporting deploy model after training.

`scripts/train_with_manifest.py` configures CUDA speedups:

```text
TF32 matmul/cudnn
cudnn.benchmark
DataLoader persistent_workers / prefetch_factor / drop_last
DDP find_unused_parameters from config
```

### Losses (`src/training/losses.py`)

`compute_total_loss()` combines:

- reconstruction loss;
- per-target weights via `training.target_loss_weights`;
- categorical class weights via `training.class_weights`;
- regularizers: uniformity, variance, decorrelation, orthogonality.

Categorical all-ignore patches return graph-preserving zero loss.

### Evaluation (`src/eval/`, `scripts/evaluate_model_suite.py`)

The standard suite evaluates:

1. Reconstruction quality: DEM MAE/R2, WorldCover macro F1/IoU, JRC Water binary F1/IoU/boundary F1.
2. Downstream probe quality: frozen AEF embedding vs composite baseline, with linear heads over 1/5/10/50 shot settings.

For the project goal, prioritize:

```text
aef_linear
composite_linear
delta = aef_linear - composite_linear
cross-shot stability
```

## Current experiment summary

Same evaluation protocol, 512 patches:

| Version | DEM R2 | WorldCover macro F1 | Embedding takeaway |
| --- | ---: | ---: | --- |
| v1.2 50 epoch | 0.9750 | 0.5736 | best downstream few-shot among current runs |
| v1.2 continue 100 | 0.9845 | 0.6917 | better reconstruction, weaker embedding |
| v1.2 continue 200 | 0.9889 | 0.7837 | best reconstruction, weakest downstream probe |

Interpretation: decoder reconstruction keeps improving with longer training, but the embedding becomes less useful for simple task heads. Future model selection should use downstream probe metrics as the primary criterion.

## Key conventions

- Timestamps are in milliseconds.
- Target spatial resolution is H/2 of input because the sensor stem has stride 2.
- Categorical targets use `ignore_index=255`.
- Channels in `source_frames` are padded to `max_input_channels`; only the first `source_channels[src]` channels are valid.
- Deploy model format is a dict with `model`, `config`, `epoch`, `global_step`, and `format: aef_deploy_v1`.
- Current CUDA scripts use `CUDA_VISIBLE_DEVICES`; old Ascend scripts use `ASCEND_RT_VISIBLE_DEVICES`.
