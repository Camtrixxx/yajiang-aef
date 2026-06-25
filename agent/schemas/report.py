from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import re
from typing import Any


class AgentStatus:
    IDLE = "idle"
    PARSED = "parsed"
    OK = "ok"
    NEEDS_INPUT = "needs_input"
    NEEDS_CONFIRMATION = "needs_confirmation"
    CHAT = "chat"
    ERROR = "error"


class AgentRoute:
    ASK_CLARIFICATION = "ask_clarification"
    ASK_CONFIRMATION = "ask_confirmation"
    CHAT_RESPONSE = "chat_response"
    RUN_ANALYSIS = "run_analysis"


class MessageType:
    REPORT_REQUEST = "report_request"
    SLOT_FILL = "slot_fill"
    FREE_CHAT = "free_chat"
    CHANGE_CONTEXT = "change_context"
    CONFIRMATION = "confirmation"


@dataclass(slots=True)
class ReportRequest:
    task: str
    region: str
    prompt: str
    time_range: str = ""
    session_id: str = "default"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReportRequest":
        prompt = str(payload.get("prompt") or "生成一份遥感分析报告")
        raw_time_range = str(payload.get("time_range") or "").strip()
        return cls(
            task=str(payload.get("task") or "地物分类"),
            region=str(payload.get("region") or "雅江区域"),
            prompt=prompt,
            time_range=raw_time_range,
            session_id=str(payload.get("session_id") or "default"),
        )


@dataclass(slots=True)
class AgentIntent:
    message_type: str
    task: str
    region: str
    time_range: str
    user_prompt: str
    missing_fields: list[str] = field(default_factory=list)
    confirmation_fields: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "rule"
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields) == 0 and len(self.confirmation_fields) == 0


@dataclass(slots=True)
class MetricCard:
    label: str
    value: str
    description: str = ""


@dataclass(slots=True)
class ChartAsset:
    title: str
    kind: str
    url: str
    caption: str


@dataclass(slots=True)
class AnalysisResult:
    task: str
    region: str
    time_range: str
    headline: str
    summary: str
    metrics: list[MetricCard]
    findings: list[str]
    recommendations: list[str]
    narrative_blocks: list[dict[str, Any]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    method_notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    confidence_notes: list[str] = field(default_factory=list)
    data_source: str = "prototype"
    generated_at: str = ""
    aef_payload: dict[str, Any] = field(default_factory=dict)
    charts: list[ChartAsset] = field(default_factory=list)


@dataclass(slots=True)
class ReportArtifact:
    title: str
    abstract: str
    sections: list[dict[str, Any]]
    metrics: list[MetricCard]
    charts: list[ChartAsset]
    html_url: str
    markdown_url: str
    llm_provider: str = "template"
    reused: bool = False
    generated_at: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResponse:
    status: str
    request: ReportRequest
    intent: AgentIntent | None = None
    message: str = ""
    session_id: str = "default"
    memory: dict[str, Any] = field(default_factory=dict)
    analysis: AnalysisResult | None = None
    report: ReportArtifact | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def infer_time_range(prompt: str, today: date | None = None) -> str:
    text = prompt.strip()
    current = today or date.today()
    month_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
    }

    year = current.year - 1 if "去年" in text else current.year
    if "前年" in text:
        year = current.year - 2
    if "明年" in text:
        year = current.year + 1
    year_match = re.search(r"(20\d{2})\s*年", text)
    if year_match:
        year = int(year_match.group(1))

    if "上个月" in text or "上月" in text:
        month = current.month - 1
        if month == 0:
            return f"{current.year - 1}-12"
        return f"{current.year}-{month:02d}"

    if "这个月" in text or "本月" in text or "当月" in text:
        return f"{current.year}-{current.month:02d}"

    numeric_month = re.search(r"(?<!\d)(1[0-2]|0?[1-9])\s*月", text)
    if numeric_month:
        return f"{year}-{int(numeric_month.group(1)):02d}"

    for zh_month in sorted(month_map, key=len, reverse=True):
        if f"{zh_month}月" in text or f"{zh_month}月份" in text:
            return f"{year}-{month_map[zh_month]:02d}"

    return ""
