# 训练加速方案汇报

> PPT 素材稿。每个 `## 第 N 页` 对应一页幻灯片，页内的「讲述要点」是口头补充，
> 不必进正文。所有数字来自 `docs/experiments/v1.2_training_acceleration.md`，
> 可复现脚本在 `scripts/`。

---

## 第 1 页｜背景

Yajiang-AEF 是一个 AlphaEarth 风格的多源遥感表征学习模型，输入 Sentinel-2、
Sentinel-1、Landsat 三个数据源的多时相影像，输出一张通用的地表 embedding 图。训练
阶段用 DEM、WorldCover、JRC Water 三个目标做重建约束，但真正的目标不是重建本身，
而是让这张 embedding 冻结之后，接几个简单的任务头就能完成不同的下游任务。目前的
主线在雅江区域，8 卡 A800、DDP 数据并行，数据量 1708 个 patch、12.64 GB，一次
200 epoch 的完整训练用最朴素的写法约需 8.3 小时。

做这件事的动因是**下一步要把训练扩到全国范围**。在雅江这个规模上，几小时是可以
忍受的；但数据量放大若干个数量级之后，任何一处低效都会被同比放大，原本只是「有点
慢」的环节会直接变成不可接受的墙。更关键的是，全国尺度下数据不再能整体装进内存，
帧数也会随地区变化，一些在雅江成立的假设会失效——这些问题必须在小规模上先暴露、
先定位清楚，而不是等扩上去之后再从一堆混在一起的现象里倒查。所以这次工作有两个
产出：一是雅江主线的实测加速，二是把每一项改动**是否可迁移**标注清楚，为扩规模
留出判断依据。

### 讲述要点

- 一次 step 的链路是：磁盘读取 → CPU 预处理 → 拷贝到 GPU → GPU 前向/反向 → 多卡
  梯度同步。前三段是输入侧，后两段是计算侧和通信侧；三段流水线并行，**总时间由最慢
  的那一段决定**，优化非瓶颈段不会有任何收益。
- 加速比不能相乘：五项改动独立倍率相乘约 3.4×，端到端实测只有 1.88×。后面给出的
  总倍率都是端到端实测值。
- 「95% 计算受限」后面会反复出现，含义是输入侧已经不再是瓶颈。
- 8.3 小时是**真正的裸跑**（fp32、无 TF32、无 AMP、无 compile），不是我们的历史起点。
  我们的历史起点是 2.05 小时，因为 AMP 和 TF32 早就开着。这个区别在第 3 页交代。

---

## 第 2 页｜技术方案总览

### 两个板块，解决的不是同一个问题

| 板块 | 面向的问题 | 收益 | 适用对象 |
| --- | --- | --- | --- |
| **一、数据管线** | 数据格式选错，读取成为瓶颈 | 加载层 **5×** | tif 直读的训练流程 |
| **二、训练加速** | 计算侧未充分利用 GPU | 端到端 **1.88×**，显存 **−53%** | 我们的 v1.2 主线 |

### 两者不叠加，因为面对的瓶颈不同

我们的主线在优化之后已经是**约 95% 计算受限**（epoch 20.1 秒里，纯计算 19.18 秒，
DataLoader 等待仅 0.75 秒）。对这条主线而言，数据管线已经不是瓶颈，再快也榨不出东西。

数据管线那 5× 的价值在另一个场景：**直接用 .tif 训练的流程**。那里输入侧仍是瓶颈，
所以同一个改动在不同起点上收益完全不同。

### 讲述要点

- 如果有人问「那 5× 和 1.88× 能不能相乘变成 9.4×」——不能。5× 是我们已经做完的事情
  （数据早已是 npy），1.88× 是在此基础上做的计算侧优化。

---

## 第 3 页｜从零开始：裸跑 DDP 要多久

### 用最朴素的写法拉起 8 卡，逐层加优化

四臂各一次全新 torchrun（compile 状态和 cudnn 启发式会跨运行泄漏），8 卡 A800：

| 臂 | 配置 | Step | 抖动 | 峰值显存 | Epoch | 200 epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **naive** | fp32，什么都不开 | 2718.9 ms | 14.6 | 52.08 GB | 148.9 s | **8.27 h** |
| +tf32 | 加 3 行 backend 开关 | 814.6 ms | 219.0 | 52.08 GB | 48.0 s | 2.67 h |
| +amp | 再加 bf16 autocast | 607.1 ms | 4.4 | 44.57 GB | 37.0 s | 2.05 h |
| **current** | 本次五项改动 | **350.6 ms** | 10.0 | **20.86 GB** | **20.4 s** | **1.14 h** |

**裸跑 8.3 小时 → 当前 1.14 小时，整体 7.3×。**

### 但这 7.3× 必须拆开讲，否则是在虚报

| 阶段 | 倍率 | 改动量 |
| --- | ---: | --- |
| naive → +tf32 | **3.34×** | 3 行 `torch.backends` 设置 |
| +tf32 → +amp | 1.34× | 2 行 bf16 autocast |
| +amp → current | **1.73×** | ← **本次工作的五项改动** |
| 合计 | 7.31× | |

最大的一块（3.34×）来自 TF32，而它只是三行 backend 开关，且**在本次加速工作开始之前
就已经启用**。所以我们真实的历史起点是 2.05 小时那一行，不是 8.27 小时。

**汇报口径**：如果被问「不优化会怎样」，答 8.3 小时 → 1.14 小时；但要立刻补一句，其中
4.5× 来自 AMP + TF32 这类标准配置，**1.73× 才是本次专门做的工作**。把 7.3× 整体算作
本次成果站不住。

### 两个交叉验证

- `+amp` 臂测出 607.1 ms / 37.0 s，与独立测得的 606.4 ms / 37.7 s 吻合（差异在抖动内）。
- naive 与 +tf32 峰值显存都是 52.08 GB，符合预期——TF32 只改计算精度，不改存储格式。

### 讲述要点

- `+tf32` 臂抖动 219 ms（其余三臂 4.4–14.6 ms）：这是 `cudnn.benchmark` 在 fp32 下
  反复试探算法的特征，加上 AMP 后 bf16 的 kernel 选择空间小得多，抖动降到 4.4 ms。
  不影响结论，但有人盯着抖动列问就是这个原因。
- fp32 裸跑没有 OOM（52.08 GB / 81.92 GB），但已经用掉 64% 显存；这也解释了为什么
  显存从 52.08 降到 20.86 GB 是有意义的额度，而不只是「省」。

---

## 第 4 页｜板块一：数据管线（tif vs npy）

### 问题

同一份雅江数据有两种存在形式：原始 GeoTIFF（`dataset/raw/yajiang/`，11 GB，
28.6 万个文件）和转换后的 npy（`data/full_npy/`，12.64 GB）。直接用 tif 训练是否
可行？转换成 npy 值多少？

### 实测：npy 比 tif 快约 5×

同一个 quarkfs 文件系统、400 个真实文件、暖 page cache、3 次取中位数：

| 读取方式 | ms/文件 | 相对 npy_f32 |
| --- | ---: | ---: |
| 真实 tif（deflate + float64） | 6.774 | 5.04× 慢 |
| npy（float32） | **1.345** | **1.00×** |

折算到一个 epoch 的 66612 个输入帧（单进程，未计 num_workers 并行）：
**tif 456 秒 → npy 92 秒**。

### 关键发现：慢的原因不是压缩

这是本节最反直觉的一点。把 DEFLATE 压缩去掉，**反而更慢**：

| 变体 | ms/文件 | 说明 |
| --- | ---: | --- |
| tif deflate float64 | 6.774 | 真实数据 |
| tif 无压缩 float64 | 7.991 | 去掉压缩后**变慢 18%** |
| tif 无压缩 float32 | 6.942 | 再改 float32，仍是 npy 的 5.16× |
| npy float32 | 1.345 | — |

去掉压缩后文件从 74 MB 涨到 300 MB，多出来的磁盘 I/O 比省下的解压 CPU 更贵。

时间的真实去向：

- **`rasterio.open()` 的固定开销占 37%**——解析文件头、CRS、GeoTIFF 标签、块索引。
  这是**每文件固定成本**，与数据量无关，所以在我们这种 128×128 的小 patch 上占比极高。
- **剩下约 60% 是 GDAL 的数据装配路径**：RasterIO 调用 + numpy 包装 + 条带/分块布局
  处理，比不过 `np.load()` 的 memmap 近零拷贝。
- **dtype float64→float32** 在 npy 侧值 36%，但在 tif 侧被 rasterio 的开销吞没了。

**结论：这不是「换个压缩参数就能修好」的问题，格式本身决定了下限。**

### 一次测量失误，以及它的教训

第一版 benchmark 把 tif 镜像写到 `/tmp`、npy 留在 `/data`，得出「tif 比 npy 快
2.08×」的结论。`/tmp` 是本地 overlay 盘，`/data` 是 quarkfs FUSE 挂载——比的是文件
系统，不是格式。

发现它靠的不是复查代码，而是一个**物理上不可能的数字**：npy 的暖 cache 读取（3.686 s）
比冷 cache（2.514 s）还慢。暖 cache 不可能慢于冷 cache，所以测量一定有问题。

修正后两侧都放在同一个 quarkfs 上，才得到自洽的 5×。这条写进了脚本注释，避免后人重踩。

### 讲述要点

- 如果被问「那我同事该不该转」：取决于他那边 IO 是否真是瓶颈。判断方法是看训练时
  `nvidia-smi` 的 GPU 利用率——长期 80% 以上说明计算受限、转格式收益有限；长期
  20–40% 说明输入饿着了，转换能省约 80% 的加载时间。
- 另外单进程 5× 经过 `num_workers=8` 并行会摊薄到墙上时钟约 1.4×，别直接报 5×。

---

## 第 5 页｜板块二：训练加速 — 结果

### 8 卡 A800，`configs/yajiang_v1_2.yaml`

本页的「优化前」是**我们的实际历史起点**（已开 AMP + TF32，即上一页的 `+amp` 臂），
不是裸跑。这是本次五项改动的净收益。

| 指标 | 优化前（+amp） | 优化后 | 倍率 |
| --- | ---: | ---: | ---: |
| Step 时间 | 606.4 ms | 350.2 ms | **1.73×** |
| Epoch 时间 | 37.7 s | 20.1 s | **1.88×** |
| 峰值显存 | 44.57 GB | 20.86 GB | **−53%** |

按 epoch 时间折算，200 epoch 训练从约 **2.05 小时降到约 1.14 小时**。

若以裸跑为起点则是 8.27 → 1.14 小时（7.3×），但如上一页所述，那 7.3× 里有 4.5× 属于
AMP/TF32 标准配置，不应记在本次工作账上。

### 已做端到端验证，不只是 benchmark 数字

通过真实训练入口 `scripts/train_with_manifest.py` 跑完 2 epoch：

- loss 4.11 → 3.36 正常下降
- `best/latest/final.pt` 与 deploy 模型均正常写出
- 每个 checkpoint 240 个 key、0 个污染 key、`strict=True` 可加载

### 讲述要点

- 显存降一半的意义不只是「省」：它是后续加大 batch、加大 image_size、或者上更大模型
  的额度。

---

## 第 6 页｜板块二：五项改动

### 五项独立收益，以及它们为什么不能相乘

| # | 改动 | 独立收益 | 性质 |
| --- | --- | --- | --- |
| 1 | `training.compile: true` | 8 卡 1.32× | 纯性能，通用 |
| 2 | `model.fast_attention: true` | 显存 −13.5%，速度 1.038× | 纯性能，通用 |
| 3 | `model.stem_norm: group` | 速度中性 | ⚠️ 破坏 checkpoint 兼容 |
| 4 | `data.max_frames: 13` | 8 卡 1.230× | ⚠️ 数据集特定，依赖 #3 |
| 5 | `data.num_workers: 2` | epoch 1.34× | ⚠️ 环境特定（暖 cache） |

相乘会得到约 3.4×，实测 1.88×。原因：#1 和 #2 优化的开销有重叠（compile 已经吸收了
约 56% 的 attention 胶水开销）；#5 修的是 #1–#4 加速计算之后**才暴露出来**的新瓶颈。

### 1. torch.compile — 最大的单项收益

```yaml
training:
  compile: true
  compile_mode: max-autotune-no-cudagraphs
```

| 模式 | Step | 首步编译耗时 |
| --- | ---: | ---: |
| eager | 468.8 ms | — |
| `default` | 409.1 ms | ~85 s |
| `max-autotune-no-cudagraphs` | **355.3 ms** | ~590 s |

用 500 秒的一次性编译换 1.151× 的长期收益。**改代码时用 `default`，正式跑用
`max-autotune-no-cudagraphs`。**

两个踩过的坑：

- **不要用不带后缀的 `max-autotune`**：它会启用 CUDA graphs，在本项目的 DDP 下捕获
  阶段直接失败（`cudaErrorStreamCaptureInvalidated`）。
- **compile 必须包在 DDP 外面**，让 DDPOptimizer 在梯度 bucket 边界切图，保住
  通信/计算 overlap。包装层次是 `OptimizedModule → DDP → AEFModel`。

### 2. fast_attention — 买的是显存，不是速度

用 `F.scaled_dot_product_attention` 手写实现替换 `nn.MultiheadAttention`。单卡仅前向：

```text
time_attn  (4096 seqs x len  48 x dim 256): 3.78 -> 1.47 ms  (2.56x)
space_attn ( 192 seqs x len 256 x dim 256): 1.83 -> 0.36 ms  (5.13x)
```

但端到端只有 1.038×——attention 只占 7.7% 参数量，且 compile 已吸收大部分胶水开销。
**真正的理由是显存**：SDPA 从不物化注意力矩阵，峰值 28.38 → 24.56 GB（−13.5%），
这是 compile 拿不到的。

等价性：fp32 下 max|diff| 5.1e-07，参数名与 `nn.MultiheadAttention` 保持一致，
旧 checkpoint `strict=True` 可加载。

### 3 + 4. GroupNorm + max_frames — 一个正确性问题伪装成性能问题

这一对是本方案里唯一需要讲清「为什么」的改动。

**观察**：全量 manifest 核对发现，1708 条记录在 s2/s1/landsat 三个源上**都恰好 13 帧**，
所以配置里的 `max_frames: 16` 意味着槽位 14–16 恒为空——**18.75% 的时间轴计算跑在保证
为零的数据上**。

**但直接改成 13 会动 loss**：|dloss| 8.3e-02。这说明有东西在偷偷依赖那些空槽。

**根因**：`SensorEncoderBank.forward` 把全部 T 个帧槽折进 batch 维，包括全零的 padding
槽，所以 **BatchNorm 的批统计量是在真实帧和 padding 帧的混合上算的**。把 BN 换成
GroupNorm 之后，同一改动的 loss 变化只剩 9.5e-07——足以把原因定在 BN 而非帧数本身。

**为什么这不只是噪声**：全国尺度下 padding 比例随地区变化（云量、重访周期、传感器
可用性），所以这是归一化统计量里一个**与地理相关的偏差**，不是随机扰动。

**代价（必须在汇报中说明）**：GroupNorm 丢弃 18 个 BatchNorm running-stat key，
`outputs/aef_hyh_yajiang_v1_2/` 下的旧权重**无法**加载进当前配置，从这里开始的训练
形成新的谱系。回退方式：`stem_norm: batch` + `max_frames: 16`。

### 5. num_workers 8 → 2 — 少即是多

原来每 rank 8 个 worker，8 卡就是 **64 个 worker 跑在 64 个物理核上**。

| num_workers | loader 等待 | epoch |
| ---: | ---: | ---: |
| 8 | 7.71 s | 27.0 s |
| 4 | 1.70 s | 21.0 s |
| **2** | **0.75 s** | **20.1 s** |
| 1 | 2.58 s | 22.4 s |

nw=1 比 nw=2 更差，说明存在**真实最优值**，不是「越少越好」。

竞争是**超线性**的：只跑 loader 的一遍，8 rank × 8 worker 用了 34.52 s，而单 rank 做
1/8 的工作只要 3.68 s——8 倍工作量花了 9.37 倍时间，总吞吐**下降**而非持平。

一个没预料到的额外收益：64 个 worker 同时在**饿死那个负责下发 CUDA kernel 的主进程**。
仅仅减少 worker 数，epoch 内计算时间就从 23.14 s 降到 19.18 s（1.21×）。

经验法则：总 worker 数约 16，即 `num_workers ≈ 16 / nproc_per_node`。

**⚠️ 暖 cache 警告**：12.64 GB 数据集完全装进 903 GB page cache，0.904 ms/文件是内存
速度不是存储速度。冷读时 worker 还要隐藏 IO 延迟，最优值会上移——全国尺度请重新测量。

### 讲述要点

- 第 3+4 项是这次工作里最有价值的部分：它不是调参调出来的，是**先发现 loss 异常，再
  定位到 BatchNorm 污染**。汇报时值得多花 30 秒。

---

## 第 7 页｜当前瓶颈与后续方向

### 现状：约 95% 计算受限

epoch 20.1 s 中，epoch 内计算 19.18 s。输入侧和 H2D 已经没有有意义的空间了
（loader 等待 0.75 s，H2D 0.25 s）。**下一步必须攻计算侧。**

### 已测但未采纳（负面结果）

| 尝试 | 结果 | 结论 |
| --- | --- | --- |
| 加大 batch_size | 8 卡吞吐 bs=4/8/12 为 78.5/80.4/80.9 patches/s，3 倍区间只涨 3.1%；epoch 时间反而变差 28.1/31.1/35.3 s；bs=16 OOM | GPU 在 bs=4 已饱和 |
| 全模型 channels_last | 0.80×，更慢 | 放弃 |
| `gradient_as_bucket_view=True` | 消不掉 DDP stride 警告 | 常见误传 |

### 尚未攻击的方向

- **数据搬运**：28.6% 的时间花在零 FLOP 的数据搬运上（NCHW↔NHWC 占 9.4%），MFU 仅 24.4%。
  需要重组 `STPBlock` 的 `reshape`/`permute`，而不是简单套 `channels_last`。**这是目前
  最大的一块。**
- **prefetch_factor**：nw=2 时 step 抖动 292.4 ms（nw=8 时仅 2.6 ms），中位数不受影响，
  说明偶有供给停顿，提高 prefetch 也许能补上。

### 换硬件/换规模时必须重测的项

| 项 | 是否需要重测 |
| --- | --- |
| `compile` / `fast_attention` | 否，通用 |
| `stem_norm: group` | 否，全国尺度下更重要 |
| `max_frames` | **是**，由数据决定；全国尺度需改为变长打包 |
| `num_workers` | **是**，与核数、内存带宽、存储速度都相关 |
| `batch_size` | 换 A100/H100 后可重试 |

### 讲述要点

- MFU 24.4% 是个可以主动提的数字：它说明还有 3 倍以上的理论空间，这次拿到的 1.88×
  远不是天花板。

---

## 附：所有数字的来源

```bash
bash scripts/run_bench_naive_ddp.sh     # 裸跑 -> 当前，四臂逐层加优化（第 3 页）
bash scripts/run_bench_ddp8.sh          # 8 卡分臂对比
bash scripts/run_bench_modes.sh         # compile 模式对比
bash scripts/run_bench_grad_stride.sh   # DDP stride 修复四臂验证

# tif vs npy
PYTHONPATH=. python scripts/bench_tif_vs_npy_real.py --patches 64
PYTHONPATH=. python scripts/bench_tif_breakdown.py --files 400
```

测量约定（写新 benchmark 时沿用）：

- 显式指定解释器 `/home/heyuhang/miniconda3/envs/hyh-dl/bin/python`
- 一臂一个全新 torchrun，避免 compile 状态泄漏
- 停表前先 `torch.cuda.synchronize()`，取多次中位数，并**同时报告抖动**——小于抖动的
  差异不携带信息
- **必须有对照臂**；benchmark 与真实数据必须在同一文件系统上

完整实验记录：`docs/experiments/v1.2_training_acceleration.md`
