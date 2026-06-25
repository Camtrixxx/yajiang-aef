from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aef_inference.runner import AEFInferenceRunner, AEFRunnerConfig


class InferRequest(BaseModel):
    sample_indices: list[int] = Field(default_factory=list)
    task: str = "all"
    use_cache: bool = True
    water_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    rgb_source: str = "s2"
    rgb_period: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Yajiang AEF model inference for agent workflows.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    parser.add_argument("--manifest", default="data/full_npy/train.jsonl")
    parser.add_argument(
        "--deploy-model",
        default="outputs/aef_hyh_yajiang_v1_2/exports/aef_hyh_yajiang_v1_2_deploy.pt",
    )
    parser.add_argument("--cache-dir", default="outputs/aef_inference_service")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_app(runner: AEFInferenceRunner) -> FastAPI:
    app = FastAPI(title="Yajiang AEF Inference Service", version="0.1.0")
    app.mount("/artifacts", StaticFiles(directory=str(runner.artifact_dir)), name="artifacts")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        meta = runner.meta()
        return {
            "status": "ok",
            "service": meta["service"],
            "device": meta["device"],
            "dataset_size": meta["dataset_size"],
        }

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return runner.meta()

    @app.post("/api/infer")
    def infer(request: InferRequest) -> JSONResponse:
        try:
            payload = runner.infer(
                sample_indices=request.sample_indices,
                task=request.task,
                use_cache=request.use_cache,
                water_threshold=request.water_threshold,
                rgb_source=request.rgb_source,
                rgb_period=request.rgb_period,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/patch-rgb")
    def patch_rgb(
        sample_index: int = Query(..., ge=0),
        source: str = "s2",
        period: str | None = None,
        use_cache: bool = True,
    ) -> FileResponse:
        try:
            payload = runner.render_patch_rgb(
                sample_index=sample_index,
                source=source,
                period=period,
                use_cache=use_cache,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(payload["artifact_path"], media_type="image/png")

    @app.get("/api/patch-rgb-info")
    def patch_rgb_info(
        sample_index: int = Query(..., ge=0),
        source: str = "s2",
        period: str | None = None,
        use_cache: bool = True,
    ) -> JSONResponse:
        try:
            payload = runner.render_patch_rgb(
                sample_index=sample_index,
                source=source,
                period=period,
                use_cache=use_cache,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(payload)

    return app


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yajiang AEF Inference Service</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f6f7f9; color: #1f2937; }
    main { max-width: 900px; margin: 0 auto; padding: 32px 20px; }
    section { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-top: 14px; }
    h1 { margin: 0 0 8px; }
    p, li { line-height: 1.7; color: #4b5563; }
    code, pre { background: #0f172a; color: #e5e7eb; border-radius: 8px; }
    code { padding: 2px 5px; }
    pre { padding: 14px; overflow: auto; }
  </style>
</head>
<body>
  <main>
    <h1>Yajiang AEF Inference Service</h1>
    <p>独立 AEF 模型推理服务，面向后续 Agent 调用。当前接口以 sample index 为输入，后续由 Agent 负责 region/time 到 patch 的选择。</p>
    <section>
      <h2>接口</h2>
      <ul>
        <li><code>GET /api/health</code></li>
        <li><code>GET /api/meta</code></li>
        <li><code>GET /api/patch-rgb?sample_index=4&amp;source=s2&amp;period=2025Q3</code></li>
        <li><code>GET /api/patch-rgb-info?sample_index=4&amp;source=s2&amp;period=2025Q3</code></li>
        <li><code>POST /api/infer</code></li>
      </ul>
      <p><code>task=landcover</code> 返回地物分类预测、真值、置信度和类别占比；<code>task=water</code> 返回水体概率、阈值 mask、真值和精度指标；<code>task=dem</code> 返回地形阴影、等高线、高程分区、坡度、剖面曲线，以及真值/预测/误差验证图。</p>
    </section>
    <section>
      <h2>示例请求</h2>
      <pre>{
  "sample_indices": [400],
  "task": "dem",
  "use_cache": true,
  "water_threshold": 0.5,
  "rgb_source": "s2",
  "rgb_period": "2025Q3"
}</pre>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    runner = AEFInferenceRunner(
        AEFRunnerConfig(
            config_path=Path(args.config),
            manifest_path=Path(args.manifest),
            deploy_model_path=Path(args.deploy_model),
            cache_dir=Path(args.cache_dir),
            device=args.device,
            seed=args.seed,
        )
    )
    app = create_app(runner)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
