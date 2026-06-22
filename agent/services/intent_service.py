from __future__ import annotations

import json
import re

from agent.schemas.report import AgentIntent, ReportRequest, infer_time_range
from agent.services.llm_provider import DeepSeekProvider, LLMProvider


SUPPORTED_TASKS = ["地物分类", "水体分布", "高程地形"]
SUPPORTED_REGIONS = ["雅江区域", "哈尔滨区域", "北京市海淀区"]


class IntentService:
    """Extract normalized report parameters before AEF execution."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm or DeepSeekProvider()

    def parse(self, request: ReportRequest) -> AgentIntent:
        llm_intent = self._parse_with_llm(request)
        if llm_intent is not None:
            return self._validate(llm_intent)
        return self._validate(self._parse_with_rules(request))

    def _parse_with_llm(self, request: ReportRequest) -> AgentIntent | None:
        system_prompt = (
            "你是遥感分析任务的意图解析器。你的任务是把用户输入和前端选择项转换为标准 JSON，"
            "供后续 AEF 遥感模型调用。只输出 JSON，不要输出解释文字。"
        )
        user_prompt = json.dumps(
            {
                "前端选择任务": request.task,
                "前端选择地区": request.region,
                "用户自然语言": request.prompt,
                "支持任务": SUPPORTED_TASKS,
                "支持地区": SUPPORTED_REGIONS,
                "当前日期": "2026-06-15",
                "要求": {
                    "message_type": "report_request / slot_fill / free_chat / change_context 之一",
                    "task": "必须是支持任务之一，优先使用前端选择，除非用户文本明确改写",
                    "region": "必须是支持地区之一，优先使用前端选择，除非用户文本明确改写",
                    "time_range": "YYYY-MM 格式；如果用户没有明确月份，返回空字符串",
                    "missing_fields": "缺少的字段名列表；缺月份时包含 time_range",
                    "confidence": "0 到 1 的数字",
                },
                "输出 JSON 示例": {
                    "message_type": "report_request",
                    "task": "地物分类",
                    "region": "雅江区域",
                    "time_range": "2025-10",
                    "missing_fields": [],
                    "confidence": 0.92,
                },
            },
            ensure_ascii=False,
        )
        text = self.llm.complete(system_prompt, user_prompt)
        if not text:
            return None
        payload = self._extract_json(text)
        if payload is None:
            return None
        return AgentIntent(
            message_type=str(payload.get("message_type") or "report_request"),
            task=str(payload.get("task") or request.task),
            region=str(payload.get("region") or request.region),
            time_range=str(payload.get("time_range") or ""),
            user_prompt=request.prompt,
            missing_fields=list(payload.get("missing_fields") or []),
            confidence=float(payload.get("confidence") or 0.0),
            source="deepseek",
        )

    def _parse_with_rules(self, request: ReportRequest) -> AgentIntent:
        message_type = "report_request"
        prompt = request.prompt.strip()
        if any(key in prompt for key in ["你是谁", "你能做什么", "你好", "介绍一下", "闲聊"]):
            message_type = "free_chat"
        if any(key in prompt for key in ["改成", "换成", "切换到", "地区改", "任务改"]):
            message_type = "change_context"
        if request.time_range and re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", request.time_range):
            message_type = "slot_fill"

        task = request.task if request.task in SUPPORTED_TASKS else "地物分类"
        region = request.region if request.region in SUPPORTED_REGIONS else "雅江区域"
        for candidate in SUPPORTED_TASKS:
            if candidate in prompt:
                task = candidate
        for candidate in SUPPORTED_REGIONS:
            if candidate in prompt:
                region = candidate
        time_range = request.time_range or infer_time_range(prompt)
        return AgentIntent(
            message_type=message_type,
            task=task,
            region=region,
            time_range=time_range,
            user_prompt=request.prompt,
            confidence=0.62 if message_type != "free_chat" else 0.45,
            source="rule",
        )

    def _validate(self, intent: AgentIntent) -> AgentIntent:
        missing = set(intent.missing_fields)
        if intent.task not in SUPPORTED_TASKS:
            intent.task = "地物分类"
        if intent.region not in SUPPORTED_REGIONS:
            intent.region = "雅江区域"
        if intent.message_type == "free_chat":
            intent.missing_fields = []
            return intent
        if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", intent.time_range):
            intent.time_range = ""
            missing.add("time_range")
        else:
            missing.discard("time_range")
        if intent.message_type == "change_context" and intent.time_range:
            missing.discard("time_range")
        intent.missing_fields = sorted(missing)
        return intent

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
