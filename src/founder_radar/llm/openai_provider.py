"""OpenAI-compatible HTTP provider.

Uses `httpx` directly rather than the `openai` SDK. Why?
  1. **Compatibility**: we want to point at any server that speaks the
     `/chat/completions` shape (LM Studio, Ollama, vLLM, Together, Groq,
     OpenRouter, OpenAI itself, MiniMax-M3). The official SDK is fine
     for OpenAI but has historically lagged on the rest.
  2. **Dependency weight**: `httpx` is already pulled in transitively by
     Typer; we don't need to add another heavy client.
  3. **Control**: when we want streaming, retries, or custom timeouts,
     raw HTTP is easier to reason about than SDK abstractions.

V2.1 (MiniMax-M3 reasoning support): this provider now reads three
extra knobs from `Settings` and surfaces them via the OpenAI Chat
Completions body in a provider-agnostic way:

  - `response_format` -> top-level `response_format` field (defaults to
    `{"type": "json_object"}`). Standard OpenAI.
  - `extra_body` -> merged into the JSON body under the `extra_body` key
    (the OpenAI-compatible convention used by MiniMax-M3, DeepSeek,
    Qwen, and Anthropic-via-OpenAI-shim).
  - `thinking_mode` -> emitted under `extra_body.thinking` as
    `{"type": "<mode>"}` (Anthropic convention) and also under
    `extra_body.reasoning` as `{"effort": "<mode>"}` for providers that
    expect `reasoning.effort`. The provider picks the one it understands.

We never hardcode MiniMax-M3. The provider is intentionally
provider-agnostic and works with any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

if TYPE_CHECKING:
    from founder_radar.config.settings import Settings

logger = logging.getLogger(__name__)

# Allowed values for the response_format setting. Anything else falls
# back to the safe default of `json_object` (or `none` if explicitly
# requested).
_VALID_RESPONSE_FORMATS = {"json_object", "none"}

# Allowed values for the thinking-mode setting. The "empty" sentinel
# means "send the field with an empty value" (MiniMax-M3 convention) to
# disable reasoning explicitly.
_VALID_THINKING_MODES = {"disabled", "adaptive", "enabled", "empty"}


class OpenAICompatibleProvider(BaseLLMProvider):
    """Talk to any server that speaks the OpenAI Chat Completions API."""

    name: str = "openai-compatible"

    def __init__(
        self,
        settings: "Settings",
        *,
        timeout: float | None = None,
    ) -> None:
        self._settings = settings
        # Honor `settings.llm_timeout_seconds` when present; fall back to
        # the explicit kwarg, then to the historical 60s default for tests
        # that don't construct via Settings.
        if timeout is None:
            timeout = float(
                getattr(settings, "llm_timeout_seconds", 60.0) or 60.0
            )
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
        url = self._url()
        payload = self.build_request_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
        # Some providers (MiniMax-M3 with reasoning_split=true) put the
        # reasoning in a separate field and leave content="". We try
        # the standard field first, then fall back to the reasoning
        # field. message.content is still the "JSON answer" channel.
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected LLM response shape: {json.dumps(data)[:500]}"
            ) from exc

        content = msg.get("content") or ""
        # If the provider sent reasoning in a separate field, copy it
        # back into content so downstream parsers can strip the
        # <think>...</think> tags. This is a safety net for the case
        # where reasoning_split=true was NOT set (or wasn't honored) and
        # the provider still inlined the trace into content.
        if not content:
            for key in ("reasoning_content", "reasoning_text", "reasoning"):
                if msg.get(key):
                    content = msg[key]
                    logger.debug(
                        "Recovered reasoning from message.%s (content was empty).",
                        key,
                    )
                    break

        # Strip thinking blocks from content when configured. This is
        # the safety net for reasoning models that ignore reasoning_split
        # and inline the trace into message.content. Imported lazily to
        # avoid a circular import (analysis.llm_json imports llm.base,
        # which the llm package re-exports).
        if getattr(self._settings, "llm_strip_thinking_always", True) and content:
            from founder_radar.analysis.llm_json import strip_thinking_blocks
            content = strip_thinking_blocks(content)

        return LLMResponse(
            content=content,
            model=data.get("model", self._settings.llm_model),
            usage=data.get("usage"),
        )

    # ------------------------------------------------------------------
    # URL / payload construction
    # ------------------------------------------------------------------
    def _url(self) -> str:
        return f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"

    def build_request_payload(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Build the JSON request body for one chat completion call.

        Does NOT perform any HTTP I/O. Used by `llm-smoke-test --debug-request`
        to print a sanitized preview of the payload, and by `complete()`
        to share the same shape with the live request.

        Reasoning-model fields are emitted at the TOP LEVEL of the payload,
        NOT nested under an `extra_body` key — that's the shape the
        MiniMax-M3 / DeepSeek OpenAI-compatible APIs actually consume.
        """
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 1. response_format: standard OpenAI field. Default
        #    `json_object` keeps strict-JSON contracts intact.
        rf = (self._settings.llm_response_format or "json_object").strip()
        if rf in _VALID_RESPONSE_FORMATS and rf != "none":
            payload["response_format"] = {"type": rf}

        # 2. Reasoning-model fields at TOP LEVEL (not under extra_body).
        #    This is the convention MiniMax-M3 / DeepSeek R1 actually
        #    consume — they look for `reasoning_split`, `thinking`, and
        #    `reasoning` as direct keys in the request body.
        payload.update(self._build_reasoning_fields())

        return payload

    def _build_reasoning_fields(self) -> dict[str, Any]:
        """Assemble provider-specific reasoning fields at top level.

        Returns an empty dict when no knobs are configured. The caller
        is expected to merge the result into the top-level payload —
        NOT to nest it under `extra_body`.
        """
        s = self._settings
        fields: dict[str, Any] = {}

        # 2a. reasoning_split: MiniMax-M3 / DeepSeek / etc. expect this
        #     flag at the top level to split reasoning out of content.
        if getattr(s, "llm_reasoning_split", False):
            fields["reasoning_split"] = True

        # 2b. thinking_mode: Anthropic uses `thinking: {type: ...}`,
        #     OpenAI-compatible reasoning endpoints use
        #     `reasoning.effort: ...`. We emit BOTH shapes so any
        #     provider picks the one it understands. The mode is
        #     one of {disabled, adaptive, enabled, empty}.
        mode = (getattr(s, "llm_thinking_mode", "empty") or "empty").strip()
        if mode not in _VALID_THINKING_MODES:
            logger.warning(
                "Unknown llm_thinking_mode %r; falling back to 'empty'.", mode,
            )
            mode = "empty"
        if mode != "empty":
            fields["thinking"] = {"type": mode}
            fields["reasoning"] = {"effort": mode}
        else:
            # "empty" -> emit empty value to disable reasoning
            # explicitly (MiniMax-M3 convention).
            fields["thinking"] = {"type": ""}
            fields["reasoning"] = {"effort": ""}

        return fields
