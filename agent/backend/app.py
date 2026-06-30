from __future__ import annotations

import argparse
import html
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from agent.config import load_config
from agent.graph.report_agent import ReportAgent
from agent.schemas.report import ReportRequest, to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_PATH = PROJECT_ROOT / "agent" / "ui" / "agent_dashboard_mock.html"
API_DOC_PATH = PROJECT_ROOT / "agent" / "API.md"
REPORT_DIR = PROJECT_ROOT / "agent" / "reports"

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    FastAPI = None
    HTTPException = None
    CORSMiddleware = None
    FileResponse = None
    HTMLResponse = None
    JSONResponse = None
    StaticFiles = None


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Serve the Yajiang report agent.")
    parser.add_argument("--host", default=config.server.host)
    parser.add_argument("--port", type=int, default=config.server.port)
    parser.add_argument("--legacy-http", action="store_true", help="Use the built-in http.server fallback.")
    return parser.parse_args()


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


def _safe_report_path(path: str) -> Path:
    rel = unquote(path.removeprefix("/reports/"))
    target = (REPORT_DIR / rel).resolve()
    root = REPORT_DIR.resolve()
    try:
        if not target.is_relative_to(root):
            raise ValueError("report path is outside report directory")
    except AttributeError:
        if not str(target).startswith(str(root)):
            raise ValueError("report path is outside report directory")
    return target


def _report_content_type(path: Path) -> str:
    if path.suffix == ".png":
        return "image/png"
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    return "text/html; charset=utf-8"


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _render_markdown_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    chunks: list[str] = []
    paragraph: list[str] = []
    bullets: list[str] = []
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            chunks.append(f"<p>{_inline_markdown(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            items = "".join(f"<li>{_inline_markdown(item)}</li>" for item in bullets)
            chunks.append(f"<ul>{items}</ul>")
            bullets = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in table_lines
            if line.strip().startswith("|")
        ]
        table_lines = []
        if len(rows) < 2:
            return
        header = rows[0]
        body = rows[2:] if all(set(cell) <= {"-", ":", " "} for cell in rows[1]) else rows[1:]
        head_html = "".join(f"<th>{_inline_markdown(cell)}</th>" for cell in header)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{_inline_markdown(cell)}</td>" for cell in row) + "</tr>"
            for row in body
        )
        chunks.append(f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>")

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_bullets()
            flush_table()
            if in_code:
                chunks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.strip().startswith("|"):
            flush_paragraph()
            flush_bullets()
            table_lines.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            continue
        if line.startswith("#"):
            flush_paragraph()
            flush_bullets()
            level = min(len(line) - len(line.lstrip("#")), 3)
            title = line[level:].strip()
            anchor = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", title).strip("-").lower()
            chunks.append(f'<h{level} id="{html.escape(anchor)}">{_inline_markdown(title)}</h{level}>')
            continue
        if line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:].strip())
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    flush_bullets()
    flush_table()
    if in_code:
        chunks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    return "\n".join(chunks)


def _api_docs_page() -> str:
    markdown_text = API_DOC_PATH.read_text(encoding="utf-8")
    body = _render_markdown_html(markdown_text)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yajiang Report Agent API</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #1f2937; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 34px 20px 64px; }}
    .top {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 18px; }}
    .top a {{ color: #2563eb; text-decoration: none; font-weight: 700; }}
    article {{ background: #fff; border: 1px solid #dbe3ea; border-radius: 10px; padding: 28px; box-shadow: 0 16px 42px rgba(15, 23, 42, 0.07); }}
    h1 {{ margin-top: 0; font-size: 34px; }}
    h2 {{ margin-top: 34px; padding-top: 10px; border-top: 1px solid #e5e7eb; }}
    h3 {{ margin-top: 28px; }}
    p, li {{ line-height: 1.78; }}
    code {{ background: #eef2ff; color: #1d4ed8; border-radius: 5px; padding: 2px 5px; font-size: 0.92em; }}
    pre {{ background: #0f172a; color: #e5e7eb; border-radius: 8px; padding: 16px; overflow: auto; line-height: 1.55; }}
    pre code {{ background: transparent; color: inherit; padding: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 20px; font-size: 14px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; }}
    ul {{ padding-left: 22px; }}
    @media (max-width: 760px) {{ article {{ padding: 18px; }} h1 {{ font-size: 28px; }} }}
  </style>
</head>
<body>
  <main>
    <div class="top">
      <strong>Yajiang Report Agent</strong>
      <nav><a href="/docs">Swagger</a> · <a href="/api-docs.md">Markdown</a> · <a href="/api/health">Health</a></nav>
    </div>
    <article>{body}</article>
  </main>
</body>
</html>
"""


def create_app(agent: ReportAgent | None = None):
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Use --legacy-http or install fastapi uvicorn.")

    report_agent = agent or ReportAgent()
    app = FastAPI(title="Yajiang Report Agent", version="0.2.0")
    config = load_config()
    if CORSMiddleware is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if StaticFiles is not None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")

    @app.get("/", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    @app.get("/ui", response_class=HTMLResponse)
    def ui_alias() -> HTMLResponse:
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    @app.get("/workflow", response_class=HTMLResponse)
    def workflow() -> HTMLResponse:
        return HTMLResponse(WORKFLOW_HTML)

    @app.get("/api-docs", response_class=HTMLResponse)
    def api_docs() -> HTMLResponse:
        return HTMLResponse(_api_docs_page())

    @app.get("/api-docs.md")
    def api_docs_markdown() -> FileResponse:
        return FileResponse(API_DOC_PATH, media_type="text/markdown; charset=utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "yajiang-report-agent",
            "backend": "fastapi",
        }

    @app.get("/api/sessions")
    def sessions(limit: int = 30) -> dict:
        return {"status": "ok", "sessions": report_agent.list_sessions(limit=limit)}

    @app.get("/api/session/{session_id}")
    def session_detail(session_id: str) -> dict:
        session = report_agent.load_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "status": "ok",
            "session": session,
            "memory": report_agent.memory_service.snapshot(session_id),
        }

    @app.post("/api/report")
    def report(payload: dict) -> JSONResponse:
        try:
            request = ReportRequest.from_dict(payload)
            response = report_agent.run(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(to_dict(response))

    @app.post("/api/session/reset")
    def reset_session(payload: dict) -> dict:
        session_id = str(payload.get("session_id") or "default")
        report_agent.memory_service.reset(session_id)
        return {"status": "ok", "session_id": session_id}

    return app


def make_handler(agent: ReportAgent):
    class AgentHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/ui"}:
                file_response(self, UI_PATH, "text/html; charset=utf-8")
                return
            if parsed.path == "/workflow":
                self._workflow_page()
                return
            if parsed.path == "/api/health":
                json_response(self, {"status": "ok", "service": "yajiang-report-agent", "backend": "http.server"})
                return
            if parsed.path == "/api/sessions":
                params = parse_qs(parsed.query)
                raw_limit = params.get("limit", ["30"])[0]
                try:
                    limit = int(raw_limit)
                except ValueError:
                    limit = 30
                json_response(self, {"status": "ok", "sessions": agent.list_sessions(limit=limit)})
                return
            if parsed.path.startswith("/api/session/"):
                session_id = unquote(parsed.path.removeprefix("/api/session/"))
                session = agent.load_session(session_id)
                if not session:
                    self.send_error(404)
                    return
                json_response(self, {"status": "ok", "session": session, "memory": agent.memory_service.snapshot(session_id)})
                return
            if parsed.path.startswith("/reports/"):
                self._serve_report(parsed.path)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/session/reset":
                self._reset_session()
                return
            if parsed.path != "/api/report":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                request = ReportRequest.from_dict(payload)
                response = agent.run(request)
            except Exception as exc:
                json_response(self, {"status": "error", "error": str(exc)}, status=400)
                return
            json_response(self, to_dict(response))

        def _reset_session(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                session_id = str(payload.get("session_id") or "default")
                agent.memory_service.reset(session_id)
            except Exception as exc:
                json_response(self, {"status": "error", "error": str(exc)}, status=400)
                return
            json_response(self, {"status": "ok", "session_id": session_id})

        def _serve_report(self, path: str) -> None:
            try:
                target = _safe_report_path(path)
            except ValueError:
                self.send_error(403)
                return
            file_response(self, target, _report_content_type(target))

        def _workflow_page(self) -> None:
            data = WORKFLOW_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return AgentHandler


WORKFLOW_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>遥感报告 Agent 节点编排</title>
  <style>
    body { margin: 0; background: #eef2f5; color: #1f2937; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }
    main { max-width: 1120px; margin: 0 auto; padding: 34px 20px 58px; }
    h1 { margin: 0 0 10px; font-size: 32px; }
    .lead { color: #4b5563; line-height: 1.8; margin-bottom: 20px; }
    .flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; align-items: stretch; }
    .node { background: #fff; border: 1px solid #dbe3ea; border-radius: 8px; padding: 16px; position: relative; min-height: 186px; }
    .node h2 { margin: 0 0 8px; font-size: 17px; }
    .node p, li { line-height: 1.7; color: #4b5563; font-size: 14px; }
    .node strong { color: #111827; }
    .badge { display: inline-block; margin-bottom: 10px; color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 700; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }
    section { background: #fff; border: 1px solid #dbe3ea; border-radius: 8px; padding: 18px; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; padding: 14px; border-radius: 8px; overflow: auto; }
    @media (max-width: 980px) { .flow, .split { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>遥感报告 Agent 节点编排</h1>
    <p class="lead">当前流程支持多轮会话、SQLite 持久记忆、规则优先意图解析、LLM 兜底、真实 AEF 推理服务调用、报告复用与运行产物治理。时间月份仍是报告生成必填字段；若历史月份存在，Agent 会先请求用户确认，不会静默复用。</p>
    <div class="flow">
      <article class="node"><span class="badge">Node 1</span><h2>load_memory</h2><p><strong>输入：</strong>session_id、用户消息、前端任务/地区标签。</p><p><strong>职责：</strong>读取 SQLite 会话状态，追加用户消息。</p></article>
      <article class="node"><span class="badge">Node 2</span><h2>parse_intent</h2><p><strong>服务：</strong>规则优先 + DeepSeek 兜底。</p><p><strong>分类：</strong>report_request / slot_fill / free_chat / change_context / confirmation。</p></article>
      <article class="node"><span class="badge">Node 3</span><h2>merge_memory</h2><p><strong>职责：</strong>合并新槽位和历史槽位。</p><p><strong>策略：</strong>历史月份存在但用户未指定时先确认。</p></article>
      <article class="node"><span class="badge">Node 4</span><h2>route</h2><p><strong>分支：</strong>ask_clarification / ask_confirmation / chat_response / run_analysis。</p><p><strong>规则：</strong>缺月份先追问，聊天不生成报告。</p></article>
      <article class="node"><span class="badge">Node 5</span><h2>ask/chat</h2><p><strong>追问：</strong>补月份或确认沿用历史月份。</p><p><strong>聊天：</strong>自然语言回答，不触发报告。</p></article>
      <article class="node"><span class="badge">Node 6</span><h2>run_analysis</h2><p><strong>输入：</strong>标准化 AEF 调用字段。</p><p><strong>输出：</strong>真实 AEF 指标、图像产物、风险、局限性和专题解读。</p></article>
      <article class="node"><span class="badge">Node 7</span><h2>generate_report</h2><p><strong>服务：</strong>ReportService + DeepSeek。</p><p><strong>输出：</strong>HTML、Markdown、复用标记和报告记录。</p></article>
      <article class="node"><span class="badge">Node 8</span><h2>write_memory</h2><p><strong>职责：</strong>写回槽位、状态、摘要、消息和报告索引。</p><p><strong>输出：</strong>下一轮可继续补槽、改任务或聊天。</p></article>
    </div>
    <div class="split">
      <section>
        <h2>标准化字段</h2>
        <pre>{
  "task": "地物分类",
  "region": "雅江区域",
  "time_range": "2025-10",
  "aoi": {"name": "雅江区域"},
  "sample_indices": [300],
  "selector": "temporary_deterministic_patch_selector",
  "outputs": ["metrics", "artifacts", "report_assets"]
}</pre>
      </section>
      <section>
        <h2>产品原则</h2>
        <ul>
          <li>报告生成前必须补齐或确认关键字段。</li>
          <li>LLM 负责理解和表达，结构化指标由分析服务提供。</li>
          <li>当前已调用真实 AEF 推理服务，区域到 patch 的映射后续替换为正式 AOI 检索。</li>
          <li>运行产物进入 agent/reports 和 agent/runtime，不进入 git。</li>
        </ul>
      </section>
    </div>
  </main>
</body>
</html>
"""


def run_legacy_server(host: str, port: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), make_handler(ReportAgent()))
    print(f"Yajiang report agent listening on http://{host}:{port}")
    print(f"Open the UI at http://{host}:{port}/")
    print("Health check: /api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    if args.legacy_http or FastAPI is None:
        run_legacy_server(args.host, args.port)
        return
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


app = create_app() if FastAPI is not None else None


if __name__ == "__main__":
    main()
