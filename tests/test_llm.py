"""Tests for the LLM provider abstraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from founder_radar.llm.base import LLMMessage, LLMResponse
from founder_radar.llm.openai_provider import OpenAICompatibleProvider


def test_complete_sends_messages_and_returns_content(tmp_settings) -> None:
    """Happy path: server returns a valid OpenAI shape and we extract content."""
    tmp_settings.llm_api_key = "sk-test"

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "model": "gpt-test",
        "usage": {"total_tokens": 5},
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response) as mock_post:
        provider = OpenAICompatibleProvider(tmp_settings)
        response = provider.complete(
            [LLMMessage(role="user", content="hi")],
            temperature=0.5,
            max_tokens=64,
        )

    assert response.content == "hello"
    assert response.model == "gpt-test"
    assert response.usage == {"total_tokens": 5}

    # Verify the request was shaped correctly.
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["json"]["temperature"] == 0.5
    assert kwargs["json"]["max_tokens"] == 64
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_complete_omits_auth_when_key_empty(tmp_settings) -> None:
    tmp_settings.llm_api_key = ""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "model": "x",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response) as mock_post:
        OpenAICompatibleProvider(tmp_settings).complete(
            [LLMMessage(role="user", content="hi")]
        )

    assert "Authorization" not in mock_post.call_args.kwargs["headers"]


def test_complete_wraps_http_status_error(tmp_settings) -> None:
    response = httpx.Response(500, content=b"server boom")
    err = httpx.HTTPStatusError("500", request=MagicMock(), response=response)

    with patch("httpx.Client.post", side_effect=err):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            OpenAICompatibleProvider(tmp_settings).complete(
                [LLMMessage(role="user", content="hi")]
            )


def test_complete_wraps_network_error(tmp_settings) -> None:
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("nope")):
        with pytest.raises(RuntimeError, match="LLM provider request failed"):
            OpenAICompatibleProvider(tmp_settings).complete(
                [LLMMessage(role="user", content="hi")]
            )


def test_complete_handles_unexpected_shape(tmp_settings) -> None:
    fake_response = MagicMock()
    fake_response.json.return_value = {"weird": "shape"}
    fake_response.raise_for_status = MagicMock()
    with patch("httpx.Client.post", return_value=fake_response):
        with pytest.raises(RuntimeError, match="Unexpected LLM response shape"):
            OpenAICompatibleProvider(tmp_settings).complete(
                [LLMMessage(role="user", content="hi")]
            )


def test_llm_response_and_message_are_dataclasses() -> None:
    """Sanity: base types remain dataclasses so future providers can subclass."""
    msg = LLMMessage(role="user", content="x")
    resp = LLMResponse(content="y", model="m")
    assert msg.role == "user" and msg.content == "x"
    assert resp.content == "y" and resp.model == "m"