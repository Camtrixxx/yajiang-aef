from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from agent.graph.report_agent import ReportAgent
from agent.schemas.report import ReportRequest, to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_PATH = PROJECT_ROOT / "agent" / "ui" / "agent_dashboard_mock.html"
REPORT_DIR = PROJECT_ROOT / "agent" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Yajiang report agent.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7870)
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
                json_response(self, {"status": "ok", "service": "yajiang-report-agent"})
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
            rel = unquote(path.removeprefix("/reports/"))
            target = (REPORT_DIR / rel).resolve()
            if not str(target).startswith(str(REPORT_DIR.resolve())):
                self.send_error(403)
                return
            if target.suffix == ".png":
                content_type = "image/png"
            elif target.suffix == ".md":
                content_type = "text/markdown; charset=utf-8"
            else:
                content_type = "text/html; charset=utf-8"
            file_response(self, target, content_type)

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
    .node { background: #fff; border: 1px solid #dbe3ea; border-radius: 8px; padding: 16px; position: relative; min-height: 180px; }
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
    <p class="lead">当前流程支持多轮会话：系统会读取会话记忆，识别用户是在请求报告、补充字段、修改上下文还是普通聊天。时间月份是报告生成的必填字段；若无法提取，Agent 会先要求用户补充，不会生成不完整报告。</p>
    <div class="flow">
      <article class="node">
        <span class="badge">Node 1</span>
        <h2>load_memory</h2>
        <p><strong>输入：</strong>session_id、用户消息、前端任务/地区标签。</p>
        <p><strong>职责：</strong>读取会话状态，追加用户消息。</p>
      </article>
      <article class="node">
        <span class="badge">Node 2</span>
        <h2>parse_intent</h2>
        <p><strong>服务：</strong>IntentService + DeepSeek。</p>
        <p><strong>分类：</strong>report_request / slot_fill / free_chat / change_context。</p>
      </article>
      <article class="node">
        <span class="badge">Node 3</span>
        <h2>merge_memory</h2>
        <p><strong>职责：</strong>把新槽位和历史槽位合并。</p>
        <p><strong>例子：</strong>上一轮缺月份，用户说“去年十月份”后自动补齐。</p>
      </article>
      <article class="node">
        <span class="badge">Node 4</span>
        <h2>route</h2>
        <p><strong>分支：</strong>ask_clarification / chat_response / run_analysis。</p>
        <p><strong>规则：</strong>缺月份先追问，普通聊天不生成报告。</p>
      </article>
      <article class="node">
        <span class="badge">Node 5</span>
        <h2>ask/chat</h2>
        <p><strong>追问：</strong>提示补充月份等关键字段。</p>
        <p><strong>聊天：</strong>自然语言回答，不触发报告。</p>
      </article>
      <article class="node">
        <span class="badge">Node 6</span>
        <h2>run_analysis</h2>
        <p><strong>输入：</strong>标准化 AEF 调用字段。</p>
        <p><strong>输出：</strong>指标卡、图表、专题解读数据。</p>
      </article>
      <article class="node">
        <span class="badge">Node 7</span>
        <h2>generate_report</h2>
        <p><strong>服务：</strong>ReportService + DeepSeek。</p>
        <p><strong>输出：</strong>HTML、Markdown、报告卡片。</p>
      </article>
      <article class="node">
        <span class="badge">Node 8</span>
        <h2>write_memory</h2>
        <p><strong>职责：</strong>写回最新槽位、状态、摘要和最近消息。</p>
        <p><strong>输出：</strong>下一轮可继续补槽、改任务或聊天。</p>
      </article>
    </div>
    <div class="split">
      <section>
        <h2>标准化字段</h2>
        <pre>{
  "task": "地物分类",
  "region": "雅江区域",
  "time_range": "2025-10",
  "aoi": {"name": "雅江区域"},
  "outputs": ["embedding_map", "landcover_distribution", "confidence_summary"]
}</pre>
      </section>
      <section>
        <h2>产品原则</h2>
        <ul>
          <li>报告先服务决策阅读，再服务模型展示。</li>
          <li>缺少关键字段时先澄清，不生成错误报告。</li>
          <li>正文隐藏工程占位信息，技术状态放在调试字段或附录。</li>
          <li>图表必须可读，中文字体缺失时使用英文图内标签。</li>
        </ul>
      </section>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ReportAgent()))
    print(f"Yajiang report agent listening on http://{args.host}:{args.port}")
    print(f"Open the UI at http://{args.host}:{args.port}/")
    print("Health check: /api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
