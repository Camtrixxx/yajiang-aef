from __future__ import annotations

import argparse
import html
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from demo_visualize import plot_demo
from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.eval.features import batch_to_device, load_deploy_model
from src.utils.device import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an interactive Yajiang AEF visual demo.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", type=str, default="configs/yajiang_v0_3_c.yaml")
    parser.add_argument("--manifest", type=str, default="data/full_npy/train.jsonl")
    parser.add_argument(
        "--deploy-model",
        type=str,
        default="outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=str, default="outputs/demo_server")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resize_continuous(pred: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(pred.detach().cpu(), size=size, mode="bilinear", align_corners=False)


def resize_categorical(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    pred = logits.detach().cpu().argmax(dim=1, keepdim=True).float()
    return F.interpolate(pred, size=size, mode="nearest").long()


def finite_float(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


class DemoState:
    def __init__(self, args: argparse.Namespace) -> None:
        set_seed(args.seed)
        self.root = Path.cwd()
        self.args = args
        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = load_config(args.config)
        self.device = resolve_device(args.device)
        self.model, self.deploy_cfg = load_deploy_model(args.deploy_model, device=self.device)
        self.dataset = YajiangAEFDataset(cfg=self.cfg, manifest_path=args.manifest, split="train")
        self.lock = threading.Lock()
        self.report_dir = Path("outputs/model_eval/v0_3_c")
        self.quick_indices = [4, 1425, 0, 32, 128]

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    def render(self, sample_index: int) -> dict:
        if sample_index < 0 or sample_index >= len(self.dataset):
            raise ValueError(f"sample_index must be in [0, {len(self.dataset) - 1}]")

        out_path = self.cache_dir / f"patch_{sample_index:06d}_panel.png"
        metrics_path = self.cache_dir / f"patch_{sample_index:06d}_metrics.json"
        if out_path.exists() and metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))

        ds = YajiangAEFDataset(cfg=self.cfg, manifest_path=self.args.manifest, split="train")
        ds.records = [ds.records[sample_index]]
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=aef_collate_fn)
        batch = next(iter(loader))

        with self.lock:
            with torch.no_grad():
                device_batch = batch_to_device(batch, self.device)
                output = self.model(
                    source_frames=device_batch["source_frames"],
                    source_timestamps_ms=device_batch["source_timestamps_ms"],
                    source_frame_mask=device_batch["source_frame_mask"],
                    source_input_mask=device_batch["source_input_mask"],
                    source_type_ids=device_batch["source_type_ids"],
                    valid_start_ms=device_batch["valid_start_ms"],
                    valid_end_ms=device_batch["valid_end_ms"],
                    target_relative_time=device_batch["target_relative_time"],
                    target_metadata=device_batch["target_metadata"],
                )

        plot_demo(batch, output, out_path)
        sample_metrics = self._sample_metrics(batch, output)
        payload = {
            "sample_index": sample_index,
            "sample_id": batch["sample_id"][0],
            "image": f"/cache/{out_path.name}",
            "metrics": sample_metrics,
            "embedding": {
                "dim": int(output.embedding.shape[-1]),
                "map_shape": list(output.embedding_map.shape[-2:]),
                "norm": float(output.embedding.detach().cpu().norm(dim=1)[0]),
            },
        }
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _sample_metrics(self, batch: dict, output) -> dict:
        dem_true = batch["targets"]["dem"][:, 0].float()
        dem_pred = resize_continuous(output.reconstructions["dem"], dem_true.shape[-2:])[:, 0]
        dem_valid = torch.isfinite(dem_true)
        dem_y = dem_true[dem_valid].numpy()
        dem_p = dem_pred[dem_valid].numpy()

        wc_true = batch["targets"]["worldcover"][:, 0].long()
        wc_pred = resize_categorical(output.reconstructions["worldcover"], wc_true.shape[-2:])[:, 0]
        wc_valid = wc_true != 255
        wc_y = wc_true[wc_valid].numpy()
        wc_p = wc_pred[wc_valid].numpy()

        jrc_true = batch["targets"]["jrc_water"][:, 0].long()
        jrc_pred = resize_categorical(output.reconstructions["jrc_water"], jrc_true.shape[-2:])[:, 0]
        jrc_valid = jrc_true != 255
        water_y = (jrc_true[jrc_valid].numpy() > 0).astype(np.int64)
        water_p = (jrc_pred[jrc_valid].numpy() > 0).astype(np.int64)

        wc_classes, wc_counts = np.unique(wc_y, return_counts=True)
        jrc_values = jrc_true[jrc_valid].numpy()
        water_ratio = float((jrc_values > 0).mean()) if len(jrc_values) else 0.0

        return {
            "dem_mae": finite_float(mean_absolute_error(dem_y, dem_p)),
            "dem_r2": finite_float(r2_score(dem_y, dem_p)),
            "worldcover_macro_f1": finite_float(f1_score(wc_y, wc_p, average="macro", zero_division=0)),
            "jrc_binary_f1": finite_float(f1_score(water_y, water_p, average="binary", zero_division=0)),
            "water_pixel_ratio": water_ratio,
            "worldcover_classes": [
                {"class": int(k), "pixels": int(v)}
                for k, v in zip(wc_classes.tolist(), wc_counts.tolist(), strict=False)
            ],
        }


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def file_response(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(state: DemoState):
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._page()
                return
            if parsed.path == "/api/meta":
                self._meta()
                return
            if parsed.path == "/api/render":
                self._render(parsed.query)
                return
            if parsed.path.startswith("/cache/"):
                name = Path(parsed.path).name
                file_response(self, state.cache_dir / name, "image/png")
                return
            if parsed.path.startswith("/artifact/"):
                name = Path(parsed.path).name
                content_type = "image/png" if name.endswith(".png") else "text/plain; charset=utf-8"
                file_response(self, state.report_dir / name, content_type)
                return
            self.send_error(404)

        def _meta(self) -> None:
            payload = {
                "dataset_size": state.dataset_size,
                "device": str(state.device),
                "deploy_model": state.args.deploy_model,
                "quick_indices": [idx for idx in state.quick_indices if idx < state.dataset_size],
                "artifacts": {
                    "downstream": "/artifact/fewshot_curves.png",
                },
            }
            json_response(self, payload)

        def _render(self, query: str) -> None:
            params = parse_qs(query)
            raw_index = params.get("sample_index", ["4"])[0]
            try:
                sample_index = int(raw_index)
                payload = state.render(sample_index)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=400)
                return
            json_response(self, payload)

        def _page(self) -> None:
            data = APP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DemoHandler


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yajiang AEF Model Explorer</title>
  <style>
    :root {
      --bg: #f6f7f3;
      --panel: #ffffff;
      --ink: #1d2327;
      --muted: #697177;
      --line: #d9dfd3;
      --green: #2d6a4f;
      --blue: #246a8f;
      --red: #b94b43;
      --shadow: 0 12px 28px rgba(29, 35, 39, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(280px, 340px) 1fr;
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 20px;
      background: #fbfcf8;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    main {
      padding: 22px;
      overflow: hidden;
    }
    h1 {
      font-size: 24px;
      line-height: 1.2;
      margin: 0 0 6px;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 14px 0 6px;
    }
    input {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 15px;
      background: white;
      color: var(--ink);
    }
    button {
      height: 38px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 6px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      width: 100%;
      margin-top: 12px;
      background: var(--green);
      color: white;
      border-color: var(--green);
    }
    button:disabled { opacity: .6; cursor: wait; }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .chip {
      min-width: 48px;
      background: #eef3ea;
    }
    .status {
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric {
      padding: 14px;
      min-height: 118px;
    }
    .metric .k {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .metric .v {
      font-size: 24px;
      font-weight: 750;
    }
    .metric .hint {
      margin-top: 8px;
      font-size: 12px;
      line-height: 1.35;
      color: var(--muted);
    }
    .metric .v.good { color: var(--green); }
    .metric .v.warn { color: var(--red); }
    .viewer {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(280px, .55fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      padding: 14px;
    }
    .panel h2 {
      font-size: 16px;
      margin: 0 0 12px;
    }
    .panel p {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      margin: 0 0 12px;
    }
    .panel img {
      display: block;
      width: 100%;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #f1f3ef;
    }
    .kv {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid #edf0ea;
      font-size: 13px;
    }
    .kv span:first-child { color: var(--muted); }
    .classes {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    .classrow {
      background: #f4f6f1;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font-size: 12px;
    }
    .artifacts {
      display: block;
      margin-top: 16px;
    }
    .downstream-panel {
      padding: 18px;
    }
    .downstream-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
      gap: 16px;
      align-items: start;
    }
    .explain-list {
      display: grid;
      gap: 10px;
    }
    .explain {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f7f9f4;
      padding: 10px;
      font-size: 13px;
      line-height: 1.45;
    }
    .explain b {
      display: block;
      margin-bottom: 4px;
    }
    .footer {
      margin-top: 16px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { position: relative; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      .viewer, .summary, .downstream-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Yajiang AEF Model Explorer</h1>
      <div class="sub">选择一个 patch，在线查看 S2 输入、AEF embedding、DEM / WorldCover / JRC Water 重建效果。</div>
      <label for="sampleIndex">Patch index</label>
      <input id="sampleIndex" type="number" min="0" value="1425">
      <button id="runBtn" class="primary">生成可视化</button>
      <label>常用样本</label>
      <div id="quick" class="chips"></div>
      <button id="randomBtn" style="width:100%; margin-top:12px;">随机选择</button>
      <div id="status" class="status">正在读取模型信息...</div>
      <div class="footer">
        当前页面是模型诊断和演示系统，不代表独立测试集评估。后续接入独立 eval manifest 后，展示逻辑不需要改。
      </div>
    </aside>
    <main>
      <section class="summary">
        <div class="metric">
          <div class="k">DEM MAE</div><div id="demMae" class="v">--</div>
          <div class="hint">高程重建误差，越小越好。</div>
        </div>
        <div class="metric">
          <div class="k">WorldCover macro F1</div><div id="wcF1" class="v">--</div>
          <div class="hint">地表覆盖分类能力，越接近 1 越好。</div>
        </div>
        <div class="metric">
          <div class="k">Water F1</div><div id="waterF1" class="v good">--</div>
          <div class="hint">水体/非水体识别能力，越接近 1 越好。</div>
        </div>
        <div class="metric">
          <div class="k">Water pixel ratio</div><div id="waterRatio" class="v">--</div>
          <div class="hint">当前 patch 中真实水体像元占比。</div>
        </div>
      </section>
      <section class="viewer">
        <div class="panel">
          <h2 id="panelTitle">Patch visualization</h2>
          <p>这张图展示同一个 patch 的输入、模型中间特征和重建结果。第一行看输入和 AEF embedding；第二行看 DEM 高程重建；第三行看 WorldCover 和 JRC 水体轮廓。</p>
          <img id="mainImage" alt="AEF visualization panel">
        </div>
        <div class="panel">
          <h2>样本信息</h2>
          <p>AEF embedding 是模型学到的通用遥感特征。后续下游任务不是直接用原始图像，而是可以用这个 embedding 来训练轻量模型。</p>
          <div class="kv"><span>sample id</span><b id="sampleId">--</b></div>
          <div class="kv"><span>embedding dim</span><b id="embDim">--</b></div>
          <div class="kv"><span>embedding map</span><b id="embMap">--</b></div>
          <div class="kv"><span>embedding norm</span><b id="embNorm">--</b></div>
          <h2 style="margin-top:18px;">WorldCover 类别</h2>
          <div id="classes" class="classes"></div>
        </div>
      </section>
      <section class="artifacts">
        <div class="panel downstream-panel">
          <h2>下游任务能力</h2>
          <p>这部分回答的是：AEF embedding 对真实任务有没有帮助。曲线把传统 S2/S1 composite 特征和 AEF embedding 放在同样的少标签条件下比较。</p>
          <div class="downstream-grid">
            <img id="fewshot" src="/artifact/fewshot_curves.png" alt="Few-shot curves">
            <div class="explain-list">
              <div class="explain"><b>横轴：shots</b>表示每个类别或任务可用的少量训练样本。1-shot、5-shot 越靠左，代表标签越稀缺。</div>
              <div class="explain"><b>纵轴：任务指标</b>WorldCover 和水体任务看 macro F1，DEM 看 R2。越高说明下游任务效果越好。</div>
              <div class="explain"><b>AEF vs Composite</b>AEF 曲线高于 Composite，说明预训练得到的 embedding 比直接使用原始合成特征更有用。</div>
              <div class="explain"><b>当前结论</b>v0.3c 在水体下游任务上更有优势；WorldCover 基本持平；DEM 下游回归还需要继续优化。</div>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    let meta = { dataset_size: 0, quick_indices: [] };
    const fmt = (x, n = 3) => x === null || x === undefined ? "--" : Number(x).toFixed(n);

    async function loadMeta() {
      const res = await fetch("/api/meta");
      meta = await res.json();
      $("sampleIndex").max = Math.max(0, meta.dataset_size - 1);
      $("status").textContent = `设备 ${meta.device}，共 ${meta.dataset_size} 个 patch`;
      const quick = $("quick");
      quick.innerHTML = "";
      meta.quick_indices.forEach((idx) => {
        const btn = document.createElement("button");
        btn.className = "chip";
        btn.textContent = idx;
        btn.onclick = () => { $("sampleIndex").value = idx; render(); };
        quick.appendChild(btn);
      });
    }

    async function render() {
      const idx = Number($("sampleIndex").value || 0);
      $("runBtn").disabled = true;
      $("status").textContent = `正在生成 patch ${idx}...`;
      try {
        const res = await fetch(`/api/render?sample_index=${idx}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "render failed");
        $("panelTitle").textContent = `${data.sample_id} 可视化`;
        $("mainImage").src = `${data.image}?t=${Date.now()}`;
        $("sampleId").textContent = data.sample_id;
        $("embDim").textContent = data.embedding.dim;
        $("embMap").textContent = data.embedding.map_shape.join(" x ");
        $("embNorm").textContent = fmt(data.embedding.norm, 4);
        $("demMae").textContent = fmt(data.metrics.dem_mae, 4);
        $("wcF1").textContent = fmt(data.metrics.worldcover_macro_f1, 3);
        $("waterF1").textContent = fmt(data.metrics.jrc_binary_f1, 3);
        $("waterRatio").textContent = `${fmt(data.metrics.water_pixel_ratio * 100, 2)}%`;
        const classes = $("classes");
        classes.innerHTML = "";
        data.metrics.worldcover_classes.forEach((row) => {
          const div = document.createElement("div");
          div.className = "classrow";
          div.textContent = `class ${row.class}: ${row.pixels} px`;
          classes.appendChild(div);
        });
        $("status").textContent = `完成：patch ${idx}`;
      } catch (err) {
        $("status").textContent = err.message;
      } finally {
        $("runBtn").disabled = false;
      }
    }

    $("runBtn").onclick = render;
    $("randomBtn").onclick = () => {
      const max = Math.max(1, meta.dataset_size);
      $("sampleIndex").value = Math.floor(Math.random() * max);
      render();
    };
    loadMeta().then(render);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    state = DemoState(args)
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Yajiang AEF demo running on http://{args.host}:{args.port}")
    print(f"Device: {state.device}")
    print(f"Dataset size: {state.dataset_size}")
    server.serve_forever()


if __name__ == "__main__":
    main()
