from __future__ import annotations

from typing import Any, TypedDict

from agent.schemas.report import AgentResponse, ReportRequest
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
        analysis_service: MockAnalysisService | None = None,
        report_service: ReportService | None = None,
    ) -> None:
        self.intent_service = intent_service or IntentService()
        self.memory_service = memory_service or MemoryService()
        self.chat_llm = chat_llm or DeepSeekProvider()
        self.analysis_service = analysis_service or MockAnalysisService()
        self.report_service = report_service or ReportService()
        self.graph = self._build_graph() if StateGraph is not None else None

    def run(self, request: ReportRequest) -> AgentResponse:
        if self.graph is None:
            state = self._merge_memory(self._parse_intent(self._load_memory({"request": request})))
            route = self._route_after_merge(state)
            if route == "chat_response":
                state = self._chat_response(state)
            elif route == "needs_input":
                state = self._ask_clarification(state)
            else:
                state = self._generate_report(self._run_analysis(state))
            state = self._write_memory(state)
            if state.get("status") in {"needs_input", "chat"}:
                return self._response_from_state(request, state)
        else:
            state = self.graph.invoke({"request": request})

        return self._response_from_state(request, state)

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
        )

    def _build_graph(self):
        graph = StateGraph(ReportAgentState)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("parse_intent", self._parse_intent)
        graph.add_node("merge_memory", self._merge_memory)
        graph.add_node("ask_clarification", self._ask_clarification)
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
                "chat_response": "chat_response",
                "needs_input": "ask_clarification",
                "run_analysis": "run_analysis",
            },
        )
        graph.add_edge("ask_clarification", "write_memory")
        graph.add_edge("chat_response", "write_memory")
        graph.add_edge("run_analysis", "generate_report")
        graph.add_edge("generate_report", "write_memory")
        graph.add_edge("write_memory", END)
        return graph.compile()

    def _load_memory(self, state: ReportAgentState) -> ReportAgentState:
        request = state["request"]
        self.memory_service.append_user(request.session_id, request.prompt)
        state["memory"] = self.memory_service.snapshot(request.session_id)
        return state

    def _parse_intent(self, state: ReportAgentState) -> ReportAgentState:
        request = state["request"]
        intent = self.intent_service.parse(request)
        state["intent"] = intent
        state["status"] = "parsed"
        state["message"] = "已完成意图分类。"
        return state

    def _merge_memory(self, state: ReportAgentState) -> ReportAgentState:
        intent = state["intent"]
        memory = state.get("memory") or {}
        previous = memory.get("current_intent") or {}
        pending = memory.get("pending_slots") or []

        if intent.message_type == "free_chat":
            state["status"] = "chat"
            state["message"] = ""
            return state

        if intent.message_type == "slot_fill" or (pending and intent.time_range):
            intent.message_type = "slot_fill"
            intent.task = previous.get("task") or intent.task or "地物分类"
            intent.region = previous.get("region") or intent.region or "雅江区域"
            if not intent.time_range:
                intent.time_range = previous.get("time_range") or ""
            intent.missing_fields = [slot for slot in pending if not getattr(intent, slot, "")]

        if intent.message_type == "change_context":
            intent.task = intent.task or previous.get("task") or "地物分类"
            intent.region = intent.region or previous.get("region") or "雅江区域"
            intent.time_range = intent.time_range or previous.get("time_range") or ""

        if intent.time_range:
            intent.missing_fields = [slot for slot in intent.missing_fields if slot != "time_range"]

        if intent.message_type == "report_request" and previous and not intent.time_range:
            # Keep selected task/region from the UI, but allow a pending month to remain pending.
            intent.time_range = previous.get("time_range") or intent.time_range
            if intent.time_range:
                intent.missing_fields = [slot for slot in intent.missing_fields if slot != "time_range"]

        if not intent.is_complete:
            state["status"] = "needs_input"
            state["message"] = "请在需求里补充要分析的月份，例如：去年十月份、2025年9月。"
            return state
        state["status"] = "ok"
        state["message"] = "已完成意图解析，准备执行遥感分析。"
        return state

    def _route_after_merge(self, state: ReportAgentState) -> str:
        if state.get("status") == "chat":
            return "chat_response"
        if state.get("status") == "needs_input":
            return "needs_input"
        return "run_analysis"

    def _ask_clarification(self, state: ReportAgentState) -> ReportAgentState:
        state["status"] = "needs_input"
        state["message"] = "请在需求里补充要分析的月份，例如：去年十月份、2025年9月。"
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
        state["status"] = "chat"
        state["message"] = text.strip() if text else "我可以帮你生成遥感分析报告，也可以先帮你梳理任务、地区和月份。"
        return state

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
                "confidence": intent.confidence,
                "source": intent.source,
            }
            pending_slots = intent.missing_fields
        if state.get("status") == "chat":
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
        state["memory"] = self.memory_service.snapshot(request.session_id)
        return state

    def _summarize_state(self, state: ReportAgentState) -> str:
        intent = state.get("intent")
        if intent is None:
            return ""
        if state.get("status") == "needs_input":
            return f"用户正在准备{intent.region}{intent.task}报告，缺少字段：{','.join(intent.missing_fields)}。"
        if state.get("status") == "chat":
            return "用户正在与遥感报告助手进行自然语言对话。"
        return f"最近一次报告任务：{intent.region}，{intent.task}，{intent.time_range}。"
