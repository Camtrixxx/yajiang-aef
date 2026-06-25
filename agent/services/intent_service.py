from __future__ import annotations

import json
import re
import time
from datetime import date

from agent.config import IntentConfig
from agent.schemas.report import AgentIntent, MessageType, ReportRequest, infer_time_range
from agent.services.llm_provider import DeepSeekProvider, LLMProvider


SUPPORTED_TASKS = ["地物分类", "水体分布", "高程地形"]
SUPPORTED_REGIONS = ["雅江区域", "哈尔滨区域", "北京市海淀区"]


class IntentService:
    """Extract normalized report parameters before AEF execution."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        config: IntentConfig | None = None,
        today: date | None = None,
    ) -> None:
        self.llm = llm or DeepSeekProvider()
        self.config = config or IntentConfig()
        self.today = today

    def parse(self, request: ReportRequest) -> AgentIntent:
        started = time.perf_counter()
        rule_intent = self._validate(self._parse_with_rules(request))
        rule_intent.debug["rule_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        if self.config.rules_first and rule_intent.confidence >= self.config.rule_confidence_threshold:
            rule_intent.debug["llm_skipped"] = True
            return rule_intent

        llm_started = time.perf_counter()
        llm_intent = self._parse_with_llm(request)
        if llm_intent is not None:
            llm_intent.debug["llm_elapsed_ms"] = int((time.perf_counter() - llm_started) * 1000)
            llm_intent.debug["rule_candidate"] = {
                "message_type": rule_intent.message_type,
                "task": rule_intent.task,
                "region": rule_intent.region,
                "time_range": rule_intent.time_range,
                "confidence": rule_intent.confidence,
            }
            return self._validate(llm_intent)

        rule_intent.debug["llm_status"] = getattr(self.llm, "last_status", "not_called")
        rule_intent.debug["llm_elapsed_ms"] = int((time.perf_counter() - llm_started) * 1000)
        return rule_intent

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
                "当前日期": (self.today or date.today()).isoformat(),
                "要求": {
                    "message_type": "report_request / slot_fill / free_chat / change_context / confirmation 之一",
                    "task": "必须是支持任务之一，优先使用前端选择，除非用户文本明确改写",
                    "region": "必须是支持地区之一，优先使用前端选择，除非用户文本明确改写",
                    "time_range": "YYYY-MM 格式；如果用户没有明确月份，返回空字符串",
                    "missing_fields": "缺少的字段名列表；缺月份时包含 time_range",
                    "confirmation_fields": "需要用户确认的字段名列表；通常为空",
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
            confirmation_fields=list(payload.get("confirmation_fields") or []),
            confidence=float(payload.get("confidence") or 0.0),
            source="deepseek",
            debug={"llm_status": getattr(self.llm, "last_status", "ok")},
        )

    def _parse_with_rules(self, request: ReportRequest) -> AgentIntent:
        message_type = MessageType.REPORT_REQUEST
        prompt = request.prompt.strip()
        confirmation_words = ["确认", "沿用", "可以", "好的", "没问题", "继续", "用上次"]
        negative_words = ["不要", "不是", "重新", "换一个", "不沿用"]
        capability_questions = [
            "你是谁",
            "你是什么",
            "你是什么助手",
            "你能做什么",
            "你可以做什么",
            "你会做什么",
            "你会干什么",
            "你是干什么",
            "你有什么功能",
            "介绍一下你",
            "介绍你自己",
        ]
        greeting_or_chat = ["你好", "您好", "闲聊", "聊聊天"]
        report_signals = [
            "报告",
            "分析",
            "生成",
            "出一份",
            "看一下",
            "看看",
            "地物",
            "水体",
            "高程",
            "地形",
            "遥感",
            "分类",
            "分布",
            "重建",
            "去年",
            "今年",
            "明年",
            "上月",
            "本月",
            "月份",
            "月",
        ]
        has_report_signal = any(key in prompt for key in report_signals)
        is_short_user_question = "你" in prompt and len(prompt) <= 16 and not has_report_signal
        if (
            any(key in prompt for key in capability_questions)
            or any(key in prompt for key in greeting_or_chat)
            or is_short_user_question
        ):
            message_type = MessageType.FREE_CHAT
        if any(key in prompt for key in ["改成", "换成", "切换到", "地区改", "任务改"]):
            message_type = MessageType.CHANGE_CONTEXT
        if any(key in prompt for key in confirmation_words) and not any(key in prompt for key in negative_words):
            message_type = MessageType.CONFIRMATION
        if request.time_range and re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", request.time_range):
            message_type = MessageType.SLOT_FILL

        task = request.task if request.task in SUPPORTED_TASKS else "地物分类"
        region = request.region if request.region in SUPPORTED_REGIONS else "雅江区域"
        for candidate in SUPPORTED_TASKS:
            if candidate in prompt:
                task = candidate
        for candidate in SUPPORTED_REGIONS:
            if candidate in prompt:
                region = candidate
        time_range = request.time_range or infer_time_range(prompt, today=self.today)
        if time_range and message_type == MessageType.REPORT_REQUEST and len(prompt) <= 12:
            message_type = MessageType.SLOT_FILL
        if time_range and any(key in prompt for key in ["去年", "今年", "明年", "月", "上月", "本月"]):
            confidence = 0.84
        elif message_type in {MessageType.FREE_CHAT, MessageType.CHANGE_CONTEXT, MessageType.CONFIRMATION}:
            confidence = 0.82
        else:
            confidence = 0.58
        return AgentIntent(
            message_type=message_type,
            task=task,
            region=region,
            time_range=time_range,
            user_prompt=request.prompt,
            confidence=confidence,
            source="rule",
        )

    def _validate(self, intent: AgentIntent) -> AgentIntent:
        missing = set(intent.missing_fields)
        confirmation = set(intent.confirmation_fields)
        if intent.task not in SUPPORTED_TASKS:
            intent.task = "地物分类"
        if intent.region not in SUPPORTED_REGIONS:
            intent.region = "雅江区域"
        if intent.message_type not in {
            MessageType.REPORT_REQUEST,
            MessageType.SLOT_FILL,
            MessageType.FREE_CHAT,
            MessageType.CHANGE_CONTEXT,
            MessageType.CONFIRMATION,
        }:
            intent.message_type = MessageType.REPORT_REQUEST
        if intent.message_type == MessageType.FREE_CHAT:
            intent.missing_fields = []
            intent.confirmation_fields = []
            return intent
        if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", intent.time_range):
            intent.time_range = ""
            if intent.message_type != MessageType.CONFIRMATION:
                missing.add("time_range")
        else:
            missing.discard("time_range")
            confirmation.discard("time_range")
        if intent.message_type == MessageType.CHANGE_CONTEXT and intent.time_range:
            missing.discard("time_range")
        if intent.message_type == MessageType.CONFIRMATION:
            missing.discard("time_range")
        intent.missing_fields = sorted(missing)
        intent.confirmation_fields = sorted(confirmation)
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
