from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = PROJECT_ROOT / "agent"


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class LLMConfig:
    """DeepSeek / LLM provider configuration (env-driven, no hardcoded key)."""

    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    endpoint: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_ENDPOINT", "https://api.deepseek.com/chat/completions"
        )
    )
    timeout: int = field(default_factory=lambda: _get_int("DEEPSEEK_TIMEOUT", 15))
    temperature: float = 0.35


@dataclass(slots=True)
class IntentConfig:
    """Controls rules-first parsing vs. always calling the LLM."""

    # When the rule parser is confident enough, skip the LLM entirely.
    rules_first: bool = True
    rule_confidence_threshold: float = 0.6


@dataclass(slots=True)
class ReportConfig:
    """Report output directory and retention policy."""

    report_dir: Path = field(default_factory=lambda: AGENT_ROOT / "reports")
    asset_dir: Path = field(default_factory=lambda: AGENT_ROOT / "reports" / "assets")
    # Reuse an existing report when the same region/task/time was already produced.
    reuse_existing: bool = True
    # Keep at most this many report html/md pairs; older ones are pruned. 0 = unlimited.
    max_reports: int = field(default_factory=lambda: _get_int("AGENT_MAX_REPORTS", 50))


@dataclass(slots=True)
class MemoryConfig:
    """Conversation memory persistence."""

    db_path: Path = field(default_factory=lambda: AGENT_ROOT / "runtime" / "agent_memory.sqlite3")
    recent_message_limit: int = field(default_factory=lambda: _get_int("AGENT_RECENT_MESSAGES", 12))


@dataclass(slots=True)
class ServerConfig:
    host: str = field(default_factory=lambda: os.getenv("AGENT_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_int("AGENT_PORT", 7870))


@dataclass(slots=True)
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config() -> AgentConfig:
    return AgentConfig()
