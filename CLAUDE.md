# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Yajiang-AEF is an AlphaEarth-style multimodal spatiotemporal remote sensing representation learning project for the Yajiang (雅江) region. The model takes multi-temporal Sentinel-2, Sentinel-1, and Landsat imagery as input, produces a vMF-bottlenecked AEF embedding map, and learns to reconstruct DEM, WorldCover, and JRC Water targets. The embedding is evaluated via few-shot linear probe on downstream tasks.

**Hardware**: Huawei Ascend NPU environment (primary), with CUDA fallback. Production training uses NPU IDs 4,5,6,7 in a Docker container at `/workspace/hyh/yajiang-aef`.

## Commands

### Training
```bash
# v0.3c (S2 + S1 → DEM / WorldCover / JRC Water), 4-card DDP
bash scripts/run_v0_3_c.sh
# v1.1 (S2 + S1 + Landsat → same targets)
bash scripts/run_v1_1.sh
# Explicit NPU selection
NPU_IDS=4,5,6,7 NPROC_PER_NODE=4 bash scripts/run_v0_3_c.sh
```

Training output goes to `outputs/aef_hyh_yajiang_v0_3_c/` with checkpoints (`best.pt`, `final.pt`, `latest.pt`) and deploy export.

### Data preparation
```bash
python scripts/prepare_landsat_npy.py --skip-existing   # Convert Landsat raw → .npy
python scripts/build_full_manifest.py                     # Rebuild train.jsonl manifest
```

### Evaluation
```bash
bash scripts/run_eval_suite_v0_3_c.sh                     # Runs reconstruction + downstream eval
MAX_PATCHES=256 bash scripts/run_eval_suite_v0_3_c.sh     # With more patches
```

### Demo server
```bash
bash scripts/run_demo_server.sh                           # Starts Gradio on port 7860
pkill -f scripts/serve_demo.py                            # Stop it
```

### Single-process training (debug / single-card)
```bash
PYTHONPATH=/workspace/hyh/yajiang-aef ASCEND_RT_VISIBLE_DEVICES=4 \
python scripts/train_with_manifest.py \
  --config configs/yajiang_v0_3_c.yaml \
  --manifest data/full_npy/train.jsonl
```

## Architecture

### Data flow
```
S2 / S1 / Landsat .npy frames
  → YajiangAEFDataset (reads JSONL manifest, loads per-patch .npy files)
  → source_frames [B, S, T, C, H, W] with masks, timestamps, type_ids
  → AEFModel.forward()
  → AEFOutput (embedding_map, embedding, reconstructions dict)
  → compute_total_loss() (reconstruction + regularity losses)
```

### Model internals (`src/models/`)

1. **SensorEncoderBank** (`sensor_encoders.py`): One per-source adapter (1x1 conv if channel mismatch) → stem conv (stride 2) → projection. Source types: s2=0, s1=1, hls=2, landsat=3. Output spatial size is H/2 × W/2.

2. **STPBlock** (`blocks.py`): Space-Time-Precision block. Three parallel paths — precision (2D conv), time (downsample → multihead attention over time → upsample), space (downsample → multihead attention over space → upsample) — fused with residual. Frame mask controls which temporal positions attend.

3. **Time encoding** (`time_encoding.py`): `TimeCodeEncoder` (sinusoidal encoding of absolute timestamps), `WindowCodeEncoder` (encodes valid_start/end range), `RelativeTimeCodeEncoder` (sinusoidal encoding of [0,1] relative position).

4. **VMFBottleneck** (`bottleneck.py`): 1×1 conv to embedding_dim. Training: adds Gaussian noise scaled by 1/√kappa, skips L2 normalization (to preserve magnitude information). Inference: L2-normalizes + vMF sampling. Outputs both per-pixel `embedding_map` [B, D, H, W] and pooled `embedding` vector [B, D].

5. **Decoders** (`decoders.py`): `ContinuousDecoder` for DEM, `CategoricalDecoder` for WorldCover/JRC Water. Each uses `ConditionInjector` to gate window_code + relative_time + metadata into the embedding_map, then a small conv head.

6. **Losses** (`training/losses.py`): `compute_total_loss()` — L1 for continuous targets, cross-entropy (ignore_index=255) for categorical. Plus optional regularity losses: uniformity (Wang & Isola), batch variance, decorrelation, orthogonality. Weighted by `reconstruction_weight`, `uniformity_weight`, etc. from config.

### Data system (`src/data/`)
- **Manifest** (`manifest.py`): JSONL with one record per patch. Each record has `sample_id`, `valid_start_ms`, `valid_end_ms`, `inputs` (per-source frame paths + timestamps), `targets` (paths + relative_time + metadata), `split`.
- **Dataset** (`dataset.py`): `YajiangAEFDataset` loads .npy/.npz/.pt files, ensures CHW layout, center-crops/pads to `image_size`, builds padded tensors for `max_frames` per source. Targets are loaded at H/2 resolution (decoder output size).

### Training (`src/training/trainer.py`)
`Trainer.fit()` loops epochs, calls `train_one_epoch()` which: autocasts (bf16 on NPU, fp16 + GradScaler on CUDA), computes loss via `compute_total_loss()`, clips gradients, and steps optimizer. Saves `best.pt`, `final.pt`, `latest.pt`, and exports a deploy model (weights + config as dict).

### Evaluation (`src/eval/`)
- `features.py`: `extract_aef_embedding_map()` runs the deployed model to get embedding maps; `extract_composite_map()` builds a per-source temporal-average baseline.
- `metrics.py`: Reconstruction metrics (MAE, R2, macro F1, IoU, boundary F1) and few-shot linear probe.
- `sampling.py`: Stratified pixel sampling for downstream evaluation.
- `baselines.py`: Composite baseline feature extraction.

### Distributed training (`src/utils/distributed.py`)
`DistributedState` dataclass reads WORLD_SIZE/RANK/LOCAL_RANK from env. `init_distributed()` selects backend: hccl for NPU, nccl for CUDA, gloo for CPU.

### Device resolution (`src/utils/device.py`)
`resolve_device("auto")` tries NPU first (via `torch_npu`), then CUDA, then CPU. `build_grad_scaler()` only returns a GradScaler for CUDA+fp16.

### Config system (`src/config.py`)
`load_config()` reads YAML, converts to `SimpleNamespace` recursively. Config has sections: `experiment` (name, seed, output_dir), `data` (batch_size, input_sources, target_sources, manifest), `model` (dims, channels, vMF kappa, source_channels), `training` (epochs, lr, loss weights), `evaluation`.

## Key conventions
- Timestamps in the manifest are in **milliseconds**.
- Target spatial resolution is H/2 of input (due to one stride-2 stem conv in the sensor encoder).
- Categorical targets use `ignore_index=255` for nodata regions (especially JRC Water).
- Channels in `source_frames` tensor are padded to `max_input_channels`; only the first `source_channels[src]` channels carry valid data.
- Deploy model format: dict with keys `model` (state_dict), `config` (serialized cfg), `epoch`, `global_step`, `format: "aef_deploy_v1"`.
- The NPU environment uses `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
