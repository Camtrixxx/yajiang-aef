from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import MemoryConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    updated_at: str = ""
    created_at: str = ""


class MemoryService:
    """SQLite-backed conversation memory.

    The agent keeps structured slots durable across service restarts while still
    exposing the small snapshot shape expected by the UI.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def get(self, session_id: str) -> ConversationMemory:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT current_intent, pending_slots, summary, mode, turn_count,
                       last_user_message, last_agent_message, created_at, updated_at
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                self._create_session(conn, session_id)
                return ConversationMemory(session_id=session_id)
            messages = self._recent_messages(conn, session_id)
            return ConversationMemory(
                session_id=session_id,
                current_intent=self._loads(row["current_intent"], {}),
                pending_slots=self._loads(row["pending_slots"], []),
                summary=row["summary"] or "",
                mode=row["mode"] or "idle",
                turn_count=int(row["turn_count"] or 0),
                last_user_message=row["last_user_message"] or "",
                last_agent_message=row["last_agent_message"] or "",
                messages=messages,
                created_at=row["created_at"] or "",
                updated_at=row["updated_at"] or "",
            )

    def append_user(self, session_id: str, content: str) -> ConversationMemory:
        return self._append_message(session_id, "user", content, increment_turn=True)

    def append_agent(self, session_id: str, content: str) -> ConversationMemory:
        return self._append_message(session_id, "assistant", content, increment_turn=False)

    def update(
        self,
        session_id: str,
        *,
        current_intent: dict[str, Any] | None = None,
        pending_slots: list[str] | None = None,
        summary: str | None = None,
        mode: str | None = None,
    ) -> ConversationMemory:
        with self._lock, self._connect() as conn:
            self._ensure_session(conn, session_id)
            existing = self.get(session_id)
            conn.execute(
                """
                UPDATE sessions
                SET current_intent = ?, pending_slots = ?, summary = ?, mode = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    json.dumps(current_intent if current_intent is not None else existing.current_intent, ensure_ascii=False),
                    json.dumps(pending_slots if pending_slots is not None else existing.pending_slots, ensure_ascii=False),
                    summary if summary is not None else existing.summary,
                    mode if mode is not None else existing.mode,
                    _now(),
                    session_id,
                ),
            )
        return self.get(session_id)

    def reset(self, session_id: str) -> ConversationMemory:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM reports WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._create_session(conn, session_id)
        return self.get(session_id)

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
            "reports": self.recent_reports(session_id, limit=5),
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, current_intent, summary, mode, turn_count,
                       last_user_message, last_agent_message, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._session_row_to_dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, current_intent, summary, mode, turn_count,
                       last_user_message, last_agent_message, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return {}
        return self._session_row_to_dict(row)

    def record_report(
        self,
        session_id: str,
        *,
        title: str,
        html_url: str,
        markdown_url: str,
        request: dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                """
                INSERT INTO reports(session_id, title, html_url, markdown_url, request_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, title, html_url, markdown_url, json.dumps(request, ensure_ascii=False), _now()),
            )

    def recent_reports(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT title, html_url, markdown_url, request_json, created_at
                FROM reports WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "title": row["title"],
                "html_url": row["html_url"],
                "markdown_url": row["markdown_url"],
                "request": self._loads(row["request_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _append_message(self, session_id: str, role: str, content: str, *, increment_turn: bool) -> ConversationMemory:
        with self._lock, self._connect() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, _now()),
            )
            if role == "user":
                conn.execute(
                    """
                    UPDATE sessions
                    SET last_user_message = ?, turn_count = turn_count + ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (content, 1 if increment_turn else 0, _now(), session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET last_agent_message = ?, updated_at = ? WHERE session_id = ?",
                    (content, _now(), session_id),
                )
        return self.get(session_id)

    def _recent_messages(self, conn: sqlite3.Connection, session_id: str) -> list[dict[str, str]]:
        rows = conn.execute(
            """
            SELECT role, content, created_at FROM messages
            WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (session_id, self.config.recent_message_limit),
        ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    current_intent TEXT NOT NULL DEFAULT '{}',
                    pending_slots TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT 'idle',
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    last_user_message TEXT NOT NULL DEFAULT '',
                    last_agent_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    html_url TEXT NOT NULL,
                    markdown_url TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reports_session_id ON reports(session_id, id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_session(self, conn: sqlite3.Connection, session_id: str) -> None:
        row = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            self._create_session(conn, session_id)

    def _create_session(self, conn: sqlite3.Connection, session_id: str) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (session_id, _now(), _now()),
        )

    def _loads(self, raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def _session_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        current_intent = self._loads(row["current_intent"], {})
        task = current_intent.get("task") or self._guess_task(row["summary"] or "", row["last_user_message"] or "")
        region = current_intent.get("region") or self._guess_region(row["summary"] or "", row["last_user_message"] or "")
        time_range = current_intent.get("time_range") or self._guess_time(row["summary"] or "", row["last_user_message"] or "")
        title = self._build_title(task, region, time_range, row["mode"] or "idle", row["summary"] or "", row["last_user_message"] or "")
        return {
            "session_id": row["session_id"],
            "title": title,
            "summary": row["summary"] or "",
            "mode": row["mode"] or "idle",
            "turn_count": int(row["turn_count"] or 0),
            "last_user_message": row["last_user_message"] or "",
            "last_agent_message": row["last_agent_message"] or "",
            "current_intent": current_intent,
            "task": task,
            "region": region,
            "time_range": time_range,
            "created_at": row["created_at"] or "",
            "updated_at": row["updated_at"] or "",
        }

    def _guess_task(self, summary: str, text: str) -> str:
        for candidate in ["地物分类", "水体分布", "高程地形"]:
            if candidate in summary or candidate in text:
                return candidate
        return "对话"

    def _guess_region(self, summary: str, text: str) -> str:
        for candidate in ["雅江区域", "哈尔滨区域", "北京市海淀区"]:
            if candidate in summary or candidate in text:
                return candidate
        return "未指定地区"

    def _guess_time(self, summary: str, text: str) -> str:
        for source in (summary, text):
            match = re.search(r"20\d{2}-(0[1-9]|1[0-2])", source)
            if match:
                return match.group(0)
        return ""

    def _build_title(self, task: str, region: str, time_range: str, mode: str, summary: str, text: str) -> str:
        pieces = []
        if region and region != "未指定地区":
            pieces.append(region)
        if time_range:
            pieces.append(time_range)
        if task and task != "对话":
            pieces.append(task)
        if not pieces:
            pieces = ["新会话"]
        title = " ".join(pieces)
        if mode == "needs_input":
            title += " 待补月份"
        elif mode == "needs_confirmation":
            title += " 待确认"
        elif mode == "chat":
            title += " 聊天"
        if len(title) > 24:
            title = title[:24].rstrip()
        return title
