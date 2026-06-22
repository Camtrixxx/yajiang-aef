from __future__ import annotations

import html
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from agent.schemas.report import AnalysisResult, ReportArtifact, ReportRequest
from agent.services.llm_provider import DeepSeekProvider, LLMProvider


class ReportService:
    def __init__(
        self,
        report_dir: str | Path = "agent/reports",
        llm: LLMProvider | None = None,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm or DeepSeekProvider()

    def build(self, request: ReportRequest, analysis: AnalysisResult) -> ReportArtifact:
        title = f"{analysis.headline}报告"
        generated = self._write_report_content(request, analysis)
        abstract = generated["abstract"]
        llm_status = getattr(self.llm, "last_status", "template")
        llm_provider = "deepseek" if llm_status == "ok" else f"template:{llm_status}"
        sections = [
            {
                "heading": "一、分析概览",
                "body": abstract,
            },
            {
                "heading": "二、主要发现",
                "items": generated["findings"],
            },
            {
                "heading": "三、建议与后续工作",
                "items": generated["recommendations"],
            },
            {
                "heading": "四、方法说明",
                "items": generated["method_notes"],
            },
        ]

        slug = self._slug(f"{request.region}-{request.task}-{request.time_range}")
        html_path = self.report_dir / f"{slug}.html"
        md_path = self.report_dir / f"{slug}.md"
        html_path.write_text(self._render_html(title, abstract, sections, analysis), encoding="utf-8")
        md_path.write_text(self._render_markdown(title, abstract, sections, analysis), encoding="utf-8")

        return ReportArtifact(
            title=title,
            abstract=abstract,
            sections=sections,
            metrics=analysis.metrics,
            charts=analysis.charts,
            html_url=f"/reports/{html_path.name}",
            markdown_url=f"/reports/{md_path.name}",
            llm_provider=llm_provider,
        )

    def _write_report_content(self, request: ReportRequest, analysis: AnalysisResult) -> dict[str, list[str] | str]:
        system_prompt = (
            "你是遥感分析报告助手。请基于给定结构化分析结果生成中文图文报告内容。"
            "必须忠于输入数据，不得编造未提供的指标。只输出 JSON，不要输出 Markdown。"
        )
        user_prompt = json.dumps(
            {
                "用户需求": request.prompt,
                "区域": request.region,
                "任务": request.task,
                "时间": request.time_range,
                "分析摘要": analysis.summary,
                "指标": [asdict(m) for m in analysis.metrics],
                "发现": analysis.findings,
                "建议": analysis.recommendations,
                "专题解读": analysis.narrative_blocks,
                "AEF标准化调用字段": analysis.aef_payload,
                "输出格式": {
                    "abstract": "一段 260-380 字的专业执行摘要，说明区域、时间、任务、关键结论和应用价值",
                    "findings": "5-7 条主要发现，每条 50-110 字，覆盖空间格局、主导类型、异常/关注点、指标解释",
                    "recommendations": "4-6 条建议，每条 45-100 字，偏业务行动建议",
                    "method_notes": "3-5 条方法说明，写数据源、时相、AOI、模型/指标解释，不要出现 mock 字样",
                },
                "禁止": ["不要在报告正文出现 mock、占位、模拟、原型等字样", "不要编造输入中没有的具体面积、坐标或真实灾害事件"],
            },
            ensure_ascii=False,
            indent=2,
        )
        llm_text = self.llm.complete(system_prompt, user_prompt)
        if llm_text:
            parsed = self._extract_json(llm_text)
            if parsed:
                return {
                    "abstract": str(parsed.get("abstract") or self._fallback_abstract(request, analysis)),
                    "findings": self._list_or_default(parsed.get("findings"), analysis.findings),
                    "recommendations": self._list_or_default(parsed.get("recommendations"), analysis.recommendations),
                    "method_notes": self._list_or_default(
                        parsed.get("method_notes"),
                        [
                            "分析以用户选择的区域、任务标签和自然语言时间范围为约束，形成标准化输入字段。",
                            "指标解释以地表类型占比、有效像元、置信度、时相覆盖和稳定性为核心。",
                        ],
                    ),
                }
        return self._fallback_content(request, analysis)

    def _fallback_content(self, request: ReportRequest, analysis: AnalysisResult) -> dict[str, list[str] | str]:
        return {
            "abstract": self._fallback_abstract(request, analysis),
            "findings": analysis.findings,
            "recommendations": analysis.recommendations,
            "method_notes": [
                "分析以用户选择的区域、任务标签和自然语言时间范围为约束，形成标准化输入字段。",
                "指标解释以地表类型占比、有效像元、置信度、时相覆盖和稳定性为核心。",
            ],
        }

    def _fallback_abstract(self, request: ReportRequest, analysis: AnalysisResult) -> str:
        return (
            f"本报告围绕{request.region}在{request.time_range}期间的{request.task}需求展开，"
            f"综合区域统计、专题指标与图表信息形成分析结论。{analysis.summary}"
            "报告重点关注主导地表类型、空间格局特征、结果可信度和后续行动建议。"
        )

    def _extract_json(self, text: str) -> dict | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def _list_or_default(self, value, default: list[str]) -> list[str]:
        if not isinstance(value, list):
            return default
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or default

    def _render_html(
        self,
        title: str,
        abstract: str,
        sections: list[dict],
        analysis: AnalysisResult,
    ) -> str:
        metric_html = "\n".join(
            f"""<div class="metric"><span>{html.escape(m.label)}</span><strong>{html.escape(m.value)}</strong><small>{html.escape(m.description)}</small></div>"""
            for m in analysis.metrics
        )
        chart_html = "\n".join(
            f"""<figure><img src="{html.escape(c.url)}" alt="{html.escape(c.title)}"><figcaption>{html.escape(c.caption)}</figcaption></figure>"""
            for c in analysis.charts
        )
        narrative_html = "\n".join(
            f"""<article class="narrative"><h3>{html.escape(block.get("title", ""))}</h3><p>{html.escape(block.get("text", ""))}</p></article>"""
            for block in analysis.narrative_blocks
        )
        payload_rows = "\n".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(json.dumps(v, ensure_ascii=False))}</td></tr>"
            for k, v in analysis.aef_payload.items()
        )
        section_html = []
        for section in sections:
            if "items" in section:
                items = "".join(f"<li>{html.escape(item)}</li>" for item in section["items"])
                body = f"<ul>{items}</ul>"
            else:
                body = f"<p>{html.escape(section['body'])}</p>"
            section_html.append(f"<section><h2>{html.escape(section['heading'])}</h2>{body}</section>")

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #eef2f5; color: #1f2937; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
    main {{ max-width: 1060px; margin: 0 auto; padding: 34px 20px 58px; }}
    header {{ background: #ffffff; border: 1px solid #dbe3ea; border-radius: 8px; padding: 28px; margin-bottom: 16px; }}
    .eyebrow {{ color: #2563eb; font-weight: 700; font-size: 13px; margin-bottom: 10px; }}
    h1 {{ margin: 0 0 12px; font-size: 34px; letter-spacing: 0; }}
    .lead {{ margin: 0; color: #374151; line-height: 1.9; font-size: 16px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .meta span {{ border: 1px solid #dbe3ea; border-radius: 999px; padding: 6px 11px; color: #4b5563; background: #f8fafc; font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
    .metric {{ background: #fff; border: 1px solid #dbe3ea; border-radius: 8px; padding: 14px; min-height: 98px; }}
    .metric span, .metric small {{ display: block; color: #6b7280; font-size: 12px; }}
    .metric strong {{ display: block; margin: 7px 0; font-size: 19px; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 14px; align-items: start; }}
    section, figure, .narrative, .payload {{ background: #fff; border: 1px solid #dbe3ea; border-radius: 8px; padding: 18px; margin: 14px 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p, li {{ line-height: 1.8; }}
    ul {{ padding-left: 20px; }}
    img {{ width: 100%; max-height: 460px; object-fit: contain; }}
    figcaption {{ margin-top: 8px; color: #6b7280; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-top: 1px solid #e5e7eb; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ width: 130px; color: #4b5563; }}
    .note {{ color: #6b7280; font-size: 12px; margin-top: 10px; }}
    @media (max-width: 860px) {{ .metrics {{ grid-template-columns: repeat(2, 1fr); }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">遥感专题分析报告</div>
      <h1>{html.escape(title)}</h1>
      <p class="lead">{html.escape(abstract)}</p>
      <div class="meta">
        <span>区域：{html.escape(analysis.region)}</span>
        <span>任务：{html.escape(analysis.task)}</span>
        <span>时间：{html.escape(analysis.time_range)}</span>
        <span>输出：HTML / Markdown</span>
      </div>
    </header>
    <div class="metrics">{metric_html}</div>
    <div class="grid">
      <div>{chart_html}</div>
      <div>{narrative_html}</div>
    </div>
    {''.join(section_html)}
    <section class="payload">
      <h2>五、标准化 AEF 调用字段</h2>
      <table>{payload_rows}</table>
      <p class="note">说明：该字段表用于后续接入真实 AEF 分析服务，报告正文不依赖用户手工整理参数。</p>
    </section>
  </main>
</body>
</html>
"""

    def _render_markdown(
        self,
        title: str,
        abstract: str,
        sections: list[dict],
        analysis: AnalysisResult,
    ) -> str:
        lines = [f"# {title}", "", abstract, "", "## 关键指标", ""]
        for metric in analysis.metrics:
            lines.append(f"- **{metric.label}**：{metric.value}。{metric.description}")
        lines.append("")
        for chart in analysis.charts:
            lines.extend([f"![{chart.title}]({chart.url})", "", chart.caption, ""])
        lines.extend(["## 专题解读", ""])
        for block in analysis.narrative_blocks:
            lines.extend([f"### {block.get('title', '')}", "", str(block.get("text", "")), ""])
        for section in sections:
            lines.extend([f"## {section['heading']}", ""])
            if "items" in section:
                lines.extend(f"- {item}" for item in section["items"])
            else:
                lines.append(section["body"])
            lines.append("")
        lines.extend(["## 五、标准化 AEF 调用字段", ""])
        lines.append("```json")
        lines.append(json.dumps(analysis.aef_payload, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    def _slug(self, text: str) -> str:
        digest = re.sub(r"[^0-9a-zA-Z_-]+", "-", text).strip("-")
        suffix = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{digest}-{suffix}" if digest else f"report-{suffix}"
