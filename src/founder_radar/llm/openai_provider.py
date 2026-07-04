"""OpenAI-compatible HTTP provider.

Uses `httpx` directly rather than the `openai` SDK. Why?

  1. **Compatibility**: we want to point at any server that speaks the
     `/chat/completions` shape (LM Studio, Ollama, vLLM, Together, Groq,
     OpenRouter, OpenAI itself). The official SDK is fine for OpenAI but
     has historically lagged on the rest.
  2. **Dependency weight**: `httpx` is already pulled in transitively by
     Typer; we don't need to add another heavy client.
  3. **Control**: when we want streaming, retries, or custom timeouts,
     raw HTTP is easier to reason about than SDK abstractions.

This module is not used in Phase 1 — it exists so Phase 3 can call LLMs
without restructuring the codebase.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

if TYPE_CHECKING:
    from founder_radar.config.settings import Settings

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    """Talk to any server that speaks the OpenAI Chat Completions API."""

    name: str = "openai-compatible"

    def __init__(
        self,
        settings: "Settings",
        *,
        timeout: float = 60.0,
    ) -> None:
        self._settings = settings
        self._timeout = timeout

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """POST the messages to `<base_url>/chat/completions` and return text.

        Raises `RuntimeError` with a descriptive message on any failure so
        the CLI can present a clean error without leaking HTTP details.
        """
        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            # Authorization is optional for local servers. Send it only if
            # configured so we don't generate `Bearer ` headers on unauthed
            # servers (some reject that with 400).
        }
        if self._settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            # The server returned 4xx/5xx. Try to extract a useful detail.
            detail = exc.response.text[:500] if exc.response is not None else ""
            raise RuntimeError(
                f"LLM provider returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"LLM provider request failed ({self._settings.llm_base_url}): "
                f"{exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM provider returned non-JSON response: {exc}"
            ) from exc

        # Standard OpenAI shape:
        #   {"choices": [{"message": {"role": "assistant", "content": "..."}}],
        #    "model": "...", "usage": {...}}
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected LLM response shape: {json.dumps(data)[:500]}"
            ) from exc

        return LLMResponse(
            content=content,
            model=data.get("model", self._settings.llm_model),
            usage=data.get("usage"),
        )