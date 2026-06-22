from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationMemory:
    session_id: str
    current_intent: dict[str, Any] = field(default_factory=dict)
    last_user_message: str = ""
    last_agent_message: str = ""
    summary: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    pending_slots: list[str] = field(default_factory=list)
    mode: str = "idle"
    turn_count: int = 0


class MemoryService:
    def __init__(self) -> None:
        self._store: dict[str, ConversationMemory] = {}

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._store:
            self._store[session_id] = ConversationMemory(session_id=session_id)
        return self._store[session_id]

    def append_user(self, session_id: str, content: str) -> ConversationMemory:
        memory = self.get(session_id)
        memory.last_user_message = content
        memory.messages.append({"role": "user", "content": content})
        memory.turn_count += 1
        self._trim(memory)
        return memory

    def append_agent(self, session_id: str, content: str) -> ConversationMemory:
        memory = self.get(session_id)
        memory.last_agent_message = content
        memory.messages.append({"role": "assistant", "content": content})
        self._trim(memory)
        return memory

    def update(
        self,
        session_id: str,
        *,
        current_intent: dict[str, Any] | None = None,
        pending_slots: list[str] | None = None,
        summary: str | None = None,
        mode: str | None = None,
    ) -> ConversationMemory:
        memory = self.get(session_id)
        if current_intent is not None:
            memory.current_intent = current_intent
        if pending_slots is not None:
            memory.pending_slots = pending_slots
        if summary is not None:
            memory.summary = summary
        if mode is not None:
            memory.mode = mode
        return memory

    def reset(self, session_id: str) -> ConversationMemory:
        self._store[session_id] = ConversationMemory(session_id=session_id)
        return self._store[session_id]

    def snapshot(self, session_id: str) -> dict[str, Any]:
        memory = self.get(session_id)
        return {
            "session_id": memory.session_id,
            "mode": memory.mode,
            "current_intent": memory.current_intent,
            "pending_slots": memory.pending_slots,
            "summary": memory.summary,
            "turn_count": memory.turn_count,
            "recent_messages": memory.messages[-6:],
        }

    def _trim(self, memory: ConversationMemory, limit: int = 12) -> None:
        if len(memory.messages) > limit:
            memory.messages = memory.messages[-limit:]
