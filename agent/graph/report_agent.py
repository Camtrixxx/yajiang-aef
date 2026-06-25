from __future__ import annotations

from typing import Any, TypedDict

from agent.schemas.report import AgentResponse, AgentRoute, AgentStatus, MessageType, ReportRequest, to_dict
from agent.services.aef_analysis_service import AEFAnalysisService
from agent.services.analysis_service import MockAnalysisService
from agent.services.intent_service import IntentService
from agent.services.llm_provider import DeepSeekProvider, LLMProvider
from agent.services.memory_service import MemoryService
from agent.services.report_service import ReportService

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # Keep the agent runnable before optional deps are installed.
    END = None
    StateGraph = None


class ReportAgentState(TypedDict, total=False):
    request: ReportRequest
    intent: dict[str, Any]
    memory: dict[str, Any]
    status: str
    message: str
    analysis: Any
    report: Any
    debug: dict[str, Any]


class ReportAgent:
    """Report orchestration.

    Uses LangGraph when available, while preserving a small fallback path so the
    local prototype can still run before optional agent packages are installed.
    """

    def __init__(
        self,
        intent_service: IntentService | None = None,
        memory_service: MemoryService | None = None,
        chat_llm: LLMProvider | None = None,
        analysis_service: AEFAnalysisService | MockAnalysisService | None = None,
        report_service: ReportService | None = None,
    ) -> None:
        self.intent_service = intent_service or IntentService()
        self.memory_service = memory_service or MemoryService()
        self.chat_llm = chat_llm or DeepSeekProvider()
        self.analysis_service = analysis_service or AEFAnalysisService()
        self.report_service = report_service or ReportService()
        self.graph = self._build_graph() if StateGraph is not None else None

    def run(self, request: ReportRequest) -> AgentResponse:
        if self.graph is None:
            state = self._merge_memory(self._parse_intent(self._load_memory({"request": request})))
            route = self._route_after_merge(state)
            if route == AgentRoute.CHAT_RESPONSE:
                state = self._chat_response(state)
            elif route == AgentRoute.ASK_CLARIFICATION:
                state = self._ask_clarification(state)
            elif route == AgentRoute.ASK_CONFIRMATION:
                state = self._ask_confirmation(state)
            else:
                state = self._generate_report(self._run_analysis(state))
            state = self._write_memory(state)
            if state.get("status") in {AgentStatus.NEEDS_INPUT, AgentStatus.NEEDS_CONFIRMATION, AgentStatus.CHAT}:
                return self._response_from_state(request, state)
        else:
            state = self.graph.invoke({"request": request})

        return self._response_from_state(request, state)

    def load_session(self, session_id: str) -> dict[str, Any]:
        return self.memory_service.get_session(session_id)

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.memory_service.list_sessions(limit=limit)

    def _response_from_state(self, request: ReportRequest, state: ReportAgentState) -> AgentResponse:
        return AgentResponse(
            status=str(state.get("status") or "ok"),
            request=request,
            intent=state.get("intent"),
            message=str(state.get("message") or ""),
            session_id=request.session_id,
            memory=self.memory_service.snapshot(request.session_id),
            analysis=state.get("analysis"),
            report=state.get("report"),
            debug=state.get("debug") or {},
        )

    def _build_graph(self):
        graph = StateGraph(ReportAgentState)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("parse_intent", self._parse_intent)
        graph.add_node("merge_memory", self._merge_memory)
        graph.add_node("ask_clarification", self._ask_clarification)
        graph.add_node("ask_confirmation", self._ask_confirmation)
        graph.add_node("chat_response", self._chat_response)
        graph.add_node("run_analysis", self._run_analysis)
        graph.add_node("generate_report", self._generate_report)
        graph.add_node("write_memory", self._write_memory)
        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "parse_intent")
        graph.add_edge("parse_intent", "merge_memory")
        graph.add_conditional_edges(
            "merge_memory",
            self._route_after_merge,
            {
                AgentRoute.CHAT_RESPONSE: "chat_response",
                AgentRoute.ASK_CLARIFICATION: "ask_clarification",
                AgentRoute.ASK_CONFIRMATION: "ask_confirmation",
                AgentRoute.RUN_ANALYSIS: "run_analysis",
            },
        )
        graph.add_edge("ask_clarification", "write_memory")
        graph.add_edge("ask_confirmation", "write_memory")
        graph.add_edge("chat_response", "write_memory")
        graph.add_edge("run_analysis", "generate_report")
        graph.add_edge("generate_report", "write_memory")
        graph.add_edge("write_memory", END)
        return graph.compile()

    def _load_memory(self, state: ReportAgentState) -> ReportAgentState:
        request = state["request"]
        self.memory_service.append_user(request.session_id, request.prompt)
        state["memory"] = self.memory_service.snapshot(request.session_id)
        state["debug"] = {"session_id": request.session_id}
        return state

    def _parse_intent(self, state: ReportAgentState) -> ReportAgentState:
        request = state["request"]
        intent = self.intent_service.parse(request)
        state["intent"] = intent
        state["status"] = AgentStatus.PARSED
        state["message"] = "已完成意图分类。"
        state.setdefault("debug", {})["intent"] = intent.debug
        return state

    def _merge_memory(self, state: ReportAgentState) -> ReportAgentState:
        intent = state["intent"]
        memory = state.get("memory") or {}
        previous = memory.get("current_intent") or {}
        pending = memory.get("pending_slots") or []

        if intent.message_type == MessageType.FREE_CHAT:
            state["status"] = AgentStatus.CHAT
            state["message"] = ""
            return state

        if intent.message_type == MessageType.CONFIRMATION and "time_range" in pending and previous.get("time_range"):
            intent.task = previous.get("task") or intent.task or "地物分类"
            intent.region = previous.get("region") or intent.region or "雅江区域"
            intent.time_range = previous.get("time_range") or ""
            intent.missing_fields = []
            intent.confirmation_fields = []

        if intent.message_type == MessageType.SLOT_FILL or (pending and intent.time_range):
            intent.message_type = MessageType.SLOT_FILL
            intent.task = previous.get("task") or intent.task or "地物分类"
            intent.region = previous.get("region") or intent.region or "雅江区域"
            if not intent.time_range:
                intent.time_range = previous.get("time_range") or ""
            intent.missing_fields = [slot for slot in pending if not getattr(intent, slot, "")]

        if intent.message_type == MessageType.CHANGE_CONTEXT:
            intent.task = intent.task or previous.get("task") or "地物分类"
            intent.region = intent.region or previous.get("region") or "雅江区域"
            intent.time_range = intent.time_range or previous.get("time_range") or ""

        if intent.time_range:
            intent.missing_fields = [slot for slot in intent.missing_fields if slot != "time_range"]

        if intent.message_type == MessageType.REPORT_REQUEST and previous and not intent.time_range:
            previous_time = previous.get("time_range") or ""
            if previous_time:
                intent.time_range = previous_time
                intent.missing_fields = [slot for slot in intent.missing_fields if slot != "time_range"]
                intent.confirmation_fields = ["time_range"]
                state["status"] = AgentStatus.NEEDS_CONFIRMATION
                state["message"] = f"检测到你上次使用的月份是 {previous_time}，是否沿用这个月份生成报告？也可以直接输入新的月份。"
                return state

        if not intent.is_complete:
            state["status"] = AgentStatus.NEEDS_INPUT
            state["message"] = "请在需求里补充要分析的月份，例如：去年十月份、2025年9月。"
            return state
        state["status"] = AgentStatus.OK
        state["message"] = "已完成意图解析，准备执行遥感分析。"
        return state

    def _route_after_merge(self, state: ReportAgentState) -> str:
        if state.get("status") == AgentStatus.CHAT:
            return AgentRoute.CHAT_RESPONSE
        if state.get("status") == AgentStatus.NEEDS_INPUT:
            return AgentRoute.ASK_CLARIFICATION
        if state.get("status") == AgentStatus.NEEDS_CONFIRMATION:
            return AgentRoute.ASK_CONFIRMATION
        return AgentRoute.RUN_ANALYSIS

    def _ask_clarification(self, state: ReportAgentState) -> ReportAgentState:
        state["status"] = AgentStatus.NEEDS_INPUT
        state["message"] = "请在需求里补充要分析的月份，例如：去年十月份、2025年9月。"
        return state

    def _ask_confirmation(self, state: ReportAgentState) -> ReportAgentState:
        state["status"] = AgentStatus.NEEDS_CONFIRMATION
        if not state.get("message"):
            intent = state["intent"]
            state["message"] = f"是否沿用 {intent.time_range} 作为本次分析月份？你也可以直接输入新的月份。"
        return state

    def _chat_response(self, state: ReportAgentState) -> ReportAgentState:
        request = state["request"]
        memory = state.get("memory") or {}
        system_prompt = "你是遥感报告助手。请用简洁中文回答用户，不要生成报告，除非用户明确要求。"
        user_prompt = (
            f"当前会话记忆：{memory}\n"
            f"用户问题：{request.prompt}\n"
            "请回答用户，并可简要说明你可以帮助生成地物分类、水体分布、高程地形报告。"
        )
        text = self.chat_llm.complete(system_prompt, user_prompt)
        state["status"] = AgentStatus.CHAT
        state["message"] = text.strip() if text else self._fallback_chat_response(request.prompt)
        return state

    def _fallback_chat_response(self, prompt: str) -> str:
        text = prompt.strip()
        if any(key in text for key in ["你是谁", "你是什么", "你是什么助手", "你是干什么"]):
            return (
                "我是雅江遥感报告助手，主要帮你把自然语言需求整理成标准化遥感任务，"
                "调用 AEF 模型完成地物分类、水体分类或高程地形分析，然后生成带图表的报告。"
            )
        if any(key in text for key in ["你能做什么", "你可以做什么", "你会做什么", "功能"]):
            return (
                "我可以帮你生成地物分类、水体分类和高程地形报告。你只要告诉我地区、任务和月份，"
                "比如“给我一份去年九月份的水体分类报告”，我会自动补齐流程并生成报告。"
            )
        return "我在。你可以直接和我聊天，也可以让我生成地物分类、水体分类或高程地形分析报告。"

    def _run_analysis(self, state: ReportAgentState) -> ReportAgentState:
        intent = state["intent"]
        normalized_request = ReportRequest(
            task=intent.task,
            region=intent.region,
            prompt=intent.user_prompt,
            time_range=intent.time_range,
            session_id=state["request"].session_id,
        )
        state["request"] = normalized_request
        state["analysis"] = self.analysis_service.analyze(normalized_request)
        return state

    def _generate_report(self, state: ReportAgentState) -> ReportAgentState:
        state["report"] = self.report_service.build(state["request"], state["analysis"])
        state["message"] = "报告已生成。"
        state["status"] = AgentStatus.OK
        return state

    def _write_memory(self, state: ReportAgentState) -> ReportAgentState:
        request = state["request"]
        intent = state.get("intent")
        current_intent = {}
        pending_slots = []
        mode = str(state.get("status") or "idle")
        if intent is not None:
            current_intent = {
                "message_type": intent.message_type,
                "task": intent.task,
                "region": intent.region,
                "time_range": intent.time_range,
                "missing_fields": intent.missing_fields,
                "confirmation_fields": intent.confirmation_fields,
                "confidence": intent.confidence,
                "source": intent.source,
            }
            pending_slots = intent.missing_fields or intent.confirmation_fields
        if state.get("status") == AgentStatus.CHAT:
            previous = self.memory_service.snapshot(request.session_id)
            current_intent = previous.get("current_intent") or current_intent
            pending_slots = previous.get("pending_slots") or pending_slots
        summary = self._summarize_state(state)
        self.memory_service.update(
            request.session_id,
            current_intent=current_intent,
            pending_slots=pending_slots,
            summary=summary,
            mode=mode,
        )
        self.memory_service.append_agent(request.session_id, str(state.get("message") or ""))
        report = state.get("report")
        if report is not None:
            self.memory_service.record_report(
                request.session_id,
                title=report.title,
                html_url=report.html_url,
                markdown_url=report.markdown_url,
                request=to_dict(request),
            )
        state["memory"] = self.memory_service.snapshot(request.session_id)
        return state

    def _summarize_state(self, state: ReportAgentState) -> str:
        intent = state.get("intent")
        if intent is None:
            return ""
        if state.get("status") == AgentStatus.NEEDS_INPUT:
            return f"用户正在准备{intent.region}{intent.task}报告，缺少字段：{','.join(intent.missing_fields)}。"
        if state.get("status") == AgentStatus.NEEDS_CONFIRMATION:
            return f"用户正在准备{intent.region}{intent.task}报告，待确认字段：{','.join(intent.confirmation_fields)}。"
        if state.get("status") == AgentStatus.CHAT:
            return "用户正在与遥感报告助手进行自然语言对话。"
        return f"最近一次报告任务：{intent.region}，{intent.task}，{intent.time_range}。"
