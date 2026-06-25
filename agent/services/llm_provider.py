from __future__ import annotations

import json
import urllib.error
import urllib.request

from agent.config import LLMConfig


class LLMProvider:
    def complete(self, system_prompt: str, user_prompt: str) -> str | None:
        raise NotImplementedError


class DeepSeekProvider(LLMProvider):
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.api_key = self.config.api_key
        self.model = self.config.model
        self.endpoint = self.config.endpoint
        self.timeout = self.config.timeout
        self.last_status = "not_called"

    def complete(self, system_prompt: str, user_prompt: str) -> str | None:
        if not self.api_key:
            self.last_status = "missing_api_key"
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "stream": False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self.last_status = f"http_error_{exc.code}"
            return None
        except urllib.error.URLError as exc:
            self.last_status = f"url_error_{exc.reason}"
            return None
        except (TimeoutError, json.JSONDecodeError) as exc:
            self.last_status = exc.__class__.__name__
            return None

        choices = body.get("choices") or []
        if not choices:
            self.last_status = "empty_choices"
            return None
        self.last_status = "ok"
        return choices[0].get("message", {}).get("content")
