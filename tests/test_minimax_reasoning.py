"""Tests for the MiniMax-M3 reasoning-model support (V2.1).

Brief requirements (paraphrased):
  - content starting with <think>...</think>{"verdict":"reject"} parses
  - fenced JSON parses
  - malformed MiniMax-style reasoning output triggers retry
  - review invalid JSON still fails safely
  - extract --method heuristic does not call LLM even when LLM_API_KEY exists
  - llm-smoke-test exits cleanly with fake provider

Covered here:
  - llm_json.extract_json / try_extract_json / strip_thinking_blocks
  - llm_json.parse_with_repair (retry contract)
  - llm_json.make_repair_callback
  - OpenAICompatibleProvider extra_body construction
  - OpenAICompatibleProvider response_format handling
  - OpenAICompatibleProvider recovery from empty content + reasoning field
  - LLMBasedExtractor retry-repair behaviour (simulated MiniMax output)
  - LLMBasedExtractor still falls back to heuristic on total failure
  - extract --method heuristic does NOT build the LLM extractor
  - llm-smoke-test command behaviour
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from founder_radar.analysis.llm_json import (
    ERROR_PREVIEW_CHARS,
    LLMJsonError,
    extract_json,
    make_repair_callback,
    parse_with_repair,
    strip_markdown_fences,
    strip_thinking_blocks,
    try_extract_json,
)
from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from founder_radar.llm.openai_provider import OpenAICompatibleProvider
from founder_radar.main import app
from founder_radar.config.settings import Settings


# -------------------------------------------------------------------------
# Shared test helpers
# -------------------------------------------------------------------------
def _post(title: str, body: str = "") -> object:
    from founder_radar.database.models import Post
    return Post(
        source="github",
        external_id=title,
        source_category="test",
        title=title,
        body=body,
        author="op",
        url=None,
        score=1,
        num_comments=1,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


class _FakeLLMProvider(BaseLLMProvider):
    """Records calls; returns a queued series of responses."""
    name: str = "fake-llm"

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = ""
        return LLMResponse(content=content, model="fake-model")


# -------------------------------------------------------------------------
# llm_json: strip + extract
# -------------------------------------------------------------------------
def test_strip_thinking_blocks_removes_think_with_answer() -> None:
    """The MiniMax-M3 failure mode: <think>...</think> followed by JSON."""
    raw = '<think>\nLet me analyze this cluster carefully...\n</think>\n{"verdict": "reject"}'
    assert strip_thinking_blocks(raw) == '{"verdict": "reject"}'


def test_strip_thinking_blocks_handles_unterminated_think() -> None:
    """An unterminated <think> block (no closing tag) is also stripped."""
    raw = '<think>still going...\n{"x": 1}'
    stripped = strip_thinking_blocks(raw)
    # The opening tag and everything up to "</think>" (or EOF) is dropped.
    # Without a closing tag we keep the JSON part.
    assert "<think>" not in stripped
    assert "{" in stripped


def test_strip_thinking_blocks_case_insensitive() -> None:
    raw = '<THINK>noise</THINK>{"a": 1}'
    assert strip_thinking_blocks(raw) == '{"a": 1}'


def test_strip_thinking_blocks_returns_text_unchanged_when_no_think() -> None:
    raw = '{"a": 1}'
    assert strip_thinking_blocks(raw) == '{"a": 1}'


def test_strip_markdown_fences_removes_json_fence() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    assert strip_markdown_fences(raw) == '{"a": 1}'


def test_extract_json_think_then_json() -> None:
    raw = '<think>\nLet me analyze this cluster carefully...\n</think>\n{"verdict": "reject"}'
    assert extract_json(raw) == {"verdict": "reject"}


def test_extract_json_fenced() -> None:
    raw = "```json\n" + json.dumps({"a": 1}) + "\n```"
    assert extract_json(raw) == {"a": 1}


def test_extract_json_combined_think_and_fence() -> None:
    raw = (
        "<think>reasoning here</think>\n"
        "```json\n" + json.dumps({"a": 1}) + "\n```"
    )
    assert extract_json(raw) == {"a": 1}


def test_extract_json_direct() -> None:
    assert extract_json(json.dumps({"a": 1})) == {"a": 1}


def test_extract_json_subspan_inside_prose() -> None:
    raw = "Some leading text. " + json.dumps({"a": 1}) + " trailing noise."
    assert extract_json(raw) == {"a": 1}


def test_extract_json_raises_with_preview_on_garbage() -> None:
    with pytest.raises(LLMJsonError) as exc_info:
        extract_json("not json at all")
    assert exc_info.value.preview == "not json at all"
    assert "Could not extract" in str(exc_info.value)


def test_extract_json_raises_on_empty_content() -> None:
    with pytest.raises(LLMJsonError) as exc_info:
        extract_json("")
    assert "empty" in str(exc_info.value).lower()


def test_try_extract_json_returns_none_on_failure() -> None:
    assert try_extract_json("not json at all") is None
    assert try_extract_json("<think>hello</think> no json here") is None


def test_try_extract_json_returns_dict_on_success() -> None:
    assert try_extract_json('{"a": 1}') == {"a": 1}


# -------------------------------------------------------------------------
# llm_json: retry-repair contract
# -------------------------------------------------------------------------
def test_parse_with_repair_succeeds_on_first_try() -> None:
    result = parse_with_repair('{"a": 1}', repair=lambda s: None)
    assert result.value == {"a": 1}
    assert result.attempts == 1
    assert result.last_error is None


def test_parse_with_repair_succeeds_on_repair() -> None:
    """The repair callable converts garbage into valid JSON."""
    result = parse_with_repair(
        "garbage",
        repair=lambda s: '{"fixed": true}',
    )
    assert result.value == {"fixed": True}
    assert result.attempts == 2
    assert result.last_error is None


def test_parse_with_repair_returns_none_when_repair_returns_none() -> None:
    """If the repair callable gives up (returns None), the loop aborts."""
    result = parse_with_repair(
        "garbage",
        repair=lambda s: None,
    )
    assert result.value is None
    assert result.attempts == 1  # no repair attempt was made
    assert result.last_error is not None
    assert "garbage" in result.last_content


def test_parse_with_repair_returns_none_when_repair_still_invalid() -> None:
    """If both the initial parse AND the repair fail, return None."""
    result = parse_with_repair(
        "garbage",
        repair=lambda s: "still garbage",
    )
    assert result.value is None
    assert result.attempts == 2
    assert result.last_content == "still garbage"


def test_parse_with_repair_preview_includes_first_300_chars() -> None:
    """The repair callback receives a preview capped at ERROR_PREVIEW_CHARS."""
    big = "x" * 1000
    captured: list[str] = []
    def capture_and_fix(preview: str) -> str:
        captured.append(preview)
        return '{"fixed": true}'
    parse_with_repair(big, repair=capture_and_fix)
    assert len(captured) == 1
    assert len(captured[0]) == ERROR_PREVIEW_CHARS
    assert captured[0] == "x" * ERROR_PREVIEW_CHARS


def test_make_repair_callback_constructs_messages() -> None:
    """make_repair_callback wraps an llm_complete function into a repair callable."""
    captured: list[list[LLMMessage]] = []

    def fake_complete(messages: list[LLMMessage]) -> str:
        captured.append(list(messages))
        return '{"fixed": true}'

    repair = make_repair_callback(
        fake_complete,
        schema_hint='{"x": int}',
    )
    out = repair("preview of failed output")
    assert out == '{"fixed": true}'
    assert len(captured) == 1
    msgs = captured[0]
    assert msgs[0].role == "system"
    assert "JSON repair assistant" in msgs[0].content
    assert msgs[1].role == "user"
    assert "preview of failed output" in msgs[1].content
    assert '{"x": int}' in msgs[1].content


def test_make_repair_callback_returns_none_when_complete_returns_none() -> None:
    repair = make_repair_callback(
        lambda messages: None,
        schema_hint="",
    )
    assert repair("anything") is None


def test_make_repair_callback_swallows_exceptions_and_returns_none() -> None:
    def boom(messages):
        raise RuntimeError("provider down")
    repair = make_repair_callback(boom, schema_hint="")
    # Must not raise; must signal give-up via None.
    assert repair("anything") is None


# -------------------------------------------------------------------------
# OpenAICompatibleProvider: extra_body + response_format
# -------------------------------------------------------------------------
def test_provider_sends_response_format_json_object(tmp_settings) -> None:
    """Default config sets response_format to {"type": "json_object"}."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_response_format = "json_object"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_provider_skips_response_format_when_set_to_none(tmp_settings) -> None:
    """llm_response_format='none' omits the field entirely."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_response_format = "none"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    assert "response_format" not in captured["json"]


def test_provider_sends_reasoning_split_extra_body(tmp_settings) -> None:
    """llm_reasoning_split=True adds reasoning_split=true at the TOP LEVEL.

    MiniMax-M3 / DeepSeek OpenAI-compatible APIs read reasoning_split
    as a direct key in the request body, NOT nested under extra_body.
    """
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_reasoning_split = True
    tmp_settings.llm_thinking_mode = "empty"  # don't emit thinking block

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    assert captured["json"]["reasoning_split"] is True
    # Must NOT be nested under extra_body.
    assert "extra_body" not in captured["json"]


def test_provider_disabled_thinking_mode(tmp_settings) -> None:
    """llm_thinking_mode='disabled' emits thinking={type:disabled} at top level."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_reasoning_split = False
    tmp_settings.llm_thinking_mode = "disabled"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    assert captured["json"]["thinking"]["type"] == "disabled"


def test_provider_adaptive_thinking_mode(tmp_settings) -> None:
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_thinking_mode = "adaptive"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    assert captured["json"]["thinking"]["type"] == "adaptive"
    # Also emits Anthropic-style reasoning.effort for cross-provider support.
    assert captured["json"]["reasoning"]["effort"] == "adaptive"


def test_provider_empty_thinking_mode_emits_empty_strings(tmp_settings) -> None:
    """llm_thinking_mode='empty' sends thinking.type='' (MiniMax-M3 convention)."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_thinking_mode = "empty"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    # "empty" -> empty string to disable reasoning explicitly.
    assert captured["json"]["thinking"]["type"] == ""


def test_provider_recovers_empty_content_from_reasoning_field(tmp_settings) -> None:
    """When reasoning_split=true, providers may set content='' and a
    parallel reasoning_content field. We fall back to reasoning_content."""
    tmp_settings.llm_api_key = "sk-test"

    r = MagicMock()
    r.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": '{"verdict": "reject"}',
            }
        }],
        "model": "MiniMax-M3",
    }
    r.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=r):
        response = OpenAICompatibleProvider(tmp_settings).complete(
            [LLMMessage(role="user", content="hi")],
        )
    assert response.content == '{"verdict": "reject"}'


def test_provider_unknown_thinking_mode_falls_back_to_empty(tmp_settings) -> None:
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_thinking_mode = "totally-bogus"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        # Should not raise; falls back to "empty".
        OpenAICompatibleProvider(tmp_settings).complete([LLMMessage(role="user", content="hi")])
    assert captured["json"]["thinking"]["type"] == ""


# -------------------------------------------------------------------------
# LLMBasedExtractor: retry-repair behaviour
# -------------------------------------------------------------------------
def test_llm_extractor_retry_repair_fixes_think_block_output(
    tmp_settings,
) -> None:
    """Initial call returns MiniMax-M3 style output (<think> + JSON).
    extract_json strips the think block and parses the JSON.

    The retry-repair pass is only invoked when the first parse fails.
    With the new llm_json helper, think-block stripping happens BEFORE
    parse, so the first attempt succeeds.
    """
    from founder_radar.analysis.opportunity import LLMBasedExtractor

    minimax_output = (
        "<think>\nLet me analyze this cluster carefully...\n</think>\n"
        + json.dumps({
            "title": "Test opportunity",
            "problem_summary": "A problem",
            "target_audience": "Devs",
            "saas_ideas": ["Idea 1"],
            "competitors": [],
            "scores": {
                "market_size": 0.5,
                "ease_of_implementation": 0.5,
                "recurring_revenue": 0.5,
                "technical_feasibility": 0.5,
            },
        })
    )

    llm = _FakeLLMProvider([minimax_output])
    extractor = LLMBasedExtractor(llm=llm)
    # Create a few posts so the cluster isn't empty.
    from founder_radar.database.models import Post
    posts = [_post("p1", "body1"), _post("p2", "body2")]
    data = extractor.extract(cluster_id=0, posts=posts)
    assert data["extraction_method"] == "llm"
    assert data["title"] == "Test opportunity"
    # Only one LLM call (the repair pass wasn't needed).
    assert len(llm.calls) == 1


def test_llm_extractor_retries_when_first_output_garbage(tmp_settings) -> None:
    """When the first output is un-parseable garbage, the repair
    callback is invoked exactly once. If it returns valid JSON, the
    extractor uses that."""
    from founder_radar.analysis.opportunity import LLMBasedExtractor

    valid_json = json.dumps({
        "title": "Recovered",
        "problem_summary": "Recovered",
        "target_audience": "Devs",
        "saas_ideas": [],
        "competitors": [],
        "scores": {
            "market_size": 0.5, "ease_of_implementation": 0.5,
            "recurring_revenue": 0.5, "technical_feasibility": 0.5,
        },
    })
    llm = _FakeLLMProvider(["this is total garbage", valid_json])
    extractor = LLMBasedExtractor(llm=llm)
    posts = [_post("p1"), _post("p2")]
    data = extractor.extract(cluster_id=0, posts=posts)
    assert data["title"] == "Recovered"
    # 1 initial call + 1 repair call = 2 calls.
    assert len(llm.calls) == 2


def test_llm_extractor_falls_back_to_heuristic_when_repair_also_fails(
    tmp_settings,
) -> None:
    """When both the initial call AND the repair call fail, the
    extractor falls back to the heuristic extractor (existing behavior)."""
    from founder_radar.analysis.opportunity import (
        HeuristicExtractor, LLMBasedExtractor,
    )

    llm = _FakeLLMProvider(["garbage 1", "garbage 2"])
    extractor = LLMBasedExtractor(llm=llm, fallback=HeuristicExtractor())
    posts = [_post("p1", "body"), _post("p2", "body")]
    data = extractor.extract(cluster_id=0, posts=posts)
    # Heuristic fallback was used.
    assert data["extraction_method"] == "heuristic"
    assert len(llm.calls) == 2  # initial + one repair attempt


# -------------------------------------------------------------------------
# CLI: extract --method heuristic
# -------------------------------------------------------------------------
def test_extract_method_heuristic_does_not_require_llm_key(
    configured_db, tmp_path, monkeypatch
) -> None:
    """--method heuristic runs even when no LLM_API_KEY is set."""
    # Force settings to have no LLM key. configured_db fixture already
    # sets up a clean engine; we override the env to be sure.
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", str(configured_db.database_url))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_API_BASE", "")  # n/a
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()

    # Seed posts + a cluster so extract has something to chew on.
    from founder_radar.database.connection import get_session
    from founder_radar.database.repository import PostRepository
    from founder_radar.config.settings import get_settings
    settings = get_settings()
    settings.llm_api_key = ""
    with get_session() as session:
        post_repo = PostRepository(session)
        for i in range(3):
            post_repo.add(_post(f"p{i}", body="body"))
        # Force-cluster them: id=0 with 3 posts.
        session.execute(
            __import__("sqlalchemy").text(
                "UPDATE posts SET cluster_id = 0 WHERE id IN (1, 2, 3)"
            )
        )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["extract", "--method", "heuristic", "--min-cluster-size", "1"],
    )
    # Heuristic path must succeed without an LLM key.
    assert result.exit_code == 0, result.stdout
    assert "Extractor: heuristic" in result.output


def test_extract_method_llm_errors_when_no_llm_key(
    configured_db, tmp_path, monkeypatch
) -> None:
    """--method llm without LLM_API_KEY exits with code 2."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", str(configured_db.database_url))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()

    runner = CliRunner()
    result = runner.invoke(app, ["extract", "--method", "llm"])
    assert result.exit_code == 2
    assert "LLM_API_KEY" in result.output


# -------------------------------------------------------------------------
# CLI: llm-smoke-test
# -------------------------------------------------------------------------
def test_llm_smoke_test_errors_when_no_llm_key(
    tmp_path, monkeypatch
) -> None:
    """llm-smoke-test exits with code 2 when LLM_API_KEY is unset."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()
    runner = CliRunner()
    result = runner.invoke(app, ["llm-smoke-test"])
    assert result.exit_code == 2
    assert "LLM_API_KEY" in result.output


def test_llm_smoke_test_calls_provider_and_reports(
    tmp_path, monkeypatch
) -> None:
    """llm-smoke-test makes exactly one provider call and reports findings."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    get_settings.cache_clear()

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": '{"ok": true, "model_acknowledged": "fake"}',
            }
        }],
        "model": "fake-model",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        runner = CliRunner()
        result = runner.invoke(app, ["llm-smoke-test"])

    assert result.exit_code == 0, result.stdout
    assert "Provider:" in result.output
    assert "Endpoint:" in result.output
    assert "Model:" in result.output
    assert "OK: LLM smoke test passed" in result.output


def test_llm_smoke_test_reports_think_block_failure(
    tmp_path, monkeypatch
) -> None:
    """When the response contains <think>...</think> blocks AND the
    un-stripped content cannot be parsed as JSON, the smoke test
    flags it as FAIL. When the strip-think provider is on (the
    default), the provider already strips blocks before the smoke
    test sees them — so the content here is the JSON-only fallback.
    We force strip_thinking_always=False so we observe the raw
    response from the model.
    """
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_STRIP_THINKING_ALWAYS", "false")
    get_settings.cache_clear()

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    # The un-stripped response has think blocks AND
                    # cannot be parsed (no JSON inside).
                    "<think>\nreasoning here that never resolves to JSON"
                ),
            }
        }],
        "model": "fake-model",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        runner = CliRunner()
        result = runner.invoke(app, ["llm-smoke-test"])

    assert result.exit_code == 2, result.stdout
    assert "FAIL" in result.output


def test_llm_smoke_test_warns_when_think_with_parseable_json(
    tmp_path, monkeypatch
) -> None:
    """When the response contains <think>...</think> blocks but the
    JSON still parses after stripping, the smoke test reports WARN
    (NOT FAIL) — the model is leaking its reasoning trace, but the
    contract still holds. Force strip_thinking_always=False so the
    raw response reaches the parser."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_STRIP_THINKING_ALWAYS", "false")
    get_settings.cache_clear()

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    "<think>\nreasoning here</think>\n"
                    '{"ok": true}'
                ),
            }
        }],
        "model": "fake-model",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        runner = CliRunner()
        result = runner.invoke(app, ["llm-smoke-test"])

    # WARN, not FAIL: think-block leakage is undesirable but the JSON
    # parses. Exit code 0 with a "WARN" finding in the output.
    assert result.exit_code == 0, result.stdout
    assert "WARN" in result.output
    assert "think" in result.output.lower()


def test_llm_smoke_test_reports_repair_pass(
    tmp_path, monkeypatch
) -> None:
    """When the first parse fails but the retry-repair pass returns
    valid JSON, the smoke test reports PASS for the repair path."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_STRIP_THINKING_ALWAYS", "false")
    get_settings.cache_clear()

    # First call: garbage. Second call (repair): clean JSON.
    call_count = {"n": 0}

    def make_response():
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] == 1:
            r.json.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "not even close to JSON",
                    }
                }],
                "model": "fake-model",
            }
        else:
            r.json.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": '{"ok": true, "model_acknowledged": "fake"}',
                    }
                }],
                "model": "fake-model",
            }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=lambda *a, **kw: make_response()):
        runner = CliRunner()
        result = runner.invoke(app, ["llm-smoke-test"])

    assert result.exit_code == 0, result.stdout
    assert "repair path recovered" in result.output
    assert call_count["n"] == 2  # initial + one repair


# -------------------------------------------------------------------------
# OpenAICompatibleProvider: top-level reasoning fields (V2.2)
# -------------------------------------------------------------------------
def test_provider_sends_reasoning_fields_at_top_level(tmp_settings) -> None:
    """V2.2: reasoning_split, thinking, reasoning are sent at the TOP
    LEVEL of the payload (not nested under extra_body). MiniMax-M3 /
    DeepSeek OpenAI-compatible APIs read them as direct body keys.
    """
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_reasoning_split = True
    tmp_settings.llm_thinking_mode = "disabled"

    captured: dict = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        r = MagicMock()
        r.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "model": "gpt-4o-mini",
        }
        r.raise_for_status = MagicMock()
        return r

    with patch("httpx.Client.post", side_effect=fake_post):
        OpenAICompatibleProvider(tmp_settings).complete(
            [LLMMessage(role="user", content="hi")]
        )
    payload = captured["json"]
    # All three live at the top level.
    assert payload["reasoning_split"] is True
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["reasoning"] == {"effort": "disabled"}
    # None of them are nested under extra_body.
    assert "extra_body" not in payload
    # Common fields are still present.
    assert payload["model"] == tmp_settings.llm_model
    assert payload["response_format"] == {"type": "json_object"}


def test_build_request_payload_does_not_perform_io(tmp_settings) -> None:
    """`build_request_payload()` returns the dict WITHOUT making an HTTP
    call. The smoke-test --debug-request path uses this to print a
    sanitized payload preview."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_reasoning_split = True
    tmp_settings.llm_thinking_mode = "disabled"

    provider = OpenAICompatibleProvider(tmp_settings)
    payload = provider.build_request_payload(
        [LLMMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=200,
    )
    assert payload["model"] == tmp_settings.llm_model
    assert payload["reasoning_split"] is True
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 200
    assert payload["temperature"] == 0.0


def test_build_request_payload_never_includes_api_key(tmp_settings) -> None:
    """`build_request_payload()` must never surface the API key. The
    smoke-test --debug-request view uses this; an accidental leak
    would print credentials into terminal scrollback."""
    tmp_settings.llm_api_key = "sk-supersecret-do-not-print"
    provider = OpenAICompatibleProvider(tmp_settings)
    payload = provider.build_request_payload(
        [LLMMessage(role="user", content="hi")],
    )
    serialized = json.dumps(payload)
    assert "sk-supersecret" not in serialized
    assert "sk-" not in serialized


def test_provider_uses_settings_timeout(tmp_settings) -> None:
    """The provider reads `llm_timeout_seconds` from settings when the
    explicit `timeout=` kwarg is not given."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_timeout_seconds = 42.0
    provider = OpenAICompatibleProvider(tmp_settings)
    assert provider._timeout == 42.0


def test_provider_explicit_timeout_overrides_settings(tmp_settings) -> None:
    """An explicit `timeout=` kwarg beats the settings field. Tests
    that want a tight timeout can pass one without mutating settings."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_timeout_seconds = 200.0
    provider = OpenAICompatibleProvider(tmp_settings, timeout=5.0)
    assert provider._timeout == 5.0


def test_strip_thinking_always_removes_think_blocks_in_provider(
    tmp_settings,
) -> None:
    """When `llm_strip_thinking_always=True` (default), the provider
    strips `<think>...</think>` blocks from content BEFORE returning
    the LLMResponse. Downstream extractors see clean content even
    when the model leaks its reasoning trace."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_strip_thinking_always = True

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    "<think>\nlet me reason about this...</think>\n"
                    '{"verdict": "reject"}'
                ),
            }
        }],
        "model": "fake-model",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        response = OpenAICompatibleProvider(tmp_settings).complete(
            [LLMMessage(role="user", content="hi")],
        )
    assert "<think>" not in response.content
    assert response.content == '{"verdict": "reject"}'


def test_strip_thinking_always_false_preserves_raw_content(
    tmp_settings,
) -> None:
    """When `llm_strip_thinking_always=False`, the provider returns
    the model's content verbatim. Useful for `--debug-request` /
    diagnostic workflows where you want to see the raw model output."""
    tmp_settings.llm_api_key = "sk-test"
    tmp_settings.llm_strip_thinking_always = False

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    "<think>reasoning trace</think>\n"
                    '{"verdict": "reject"}'
                ),
            }
        }],
        "model": "fake-model",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        response = OpenAICompatibleProvider(tmp_settings).complete(
            [LLMMessage(role="user", content="hi")],
        )
    assert "<think>" in response.content


def test_strip_thinking_always_default_is_true(tmp_settings) -> None:
    """The Settings default for llm_strip_thinking_always is True
    so reasoning-model responses are always cleaned unless the user
    explicitly opts out."""
    # Use a fresh Settings to bypass any test mutations.
    fresh = Settings()
    assert fresh.llm_strip_thinking_always is True


# -------------------------------------------------------------------------
# CLI: llm-smoke-test --debug-request
# -------------------------------------------------------------------------
def test_llm_smoke_test_debug_request_prints_sanitized_payload(
    tmp_path, monkeypatch
) -> None:
    """`--debug-request` prints the sanitized payload WITHOUT making
    an HTTP call. Works without LLM_API_KEY (because no HTTP)."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "")  # empty -> still works for debug
    monkeypatch.setenv("LLM_REASONING_SPLIT", "true")
    monkeypatch.setenv("LLM_THINKING_MODE", "disabled")
    get_settings.cache_clear()

    # If a network call WERE attempted, this test would hang or fail.
    # We deliberately do NOT patch httpx.Client.post here.
    runner = CliRunner()
    result = runner.invoke(app, ["llm-smoke-test", "--debug-request"])

    assert result.exit_code == 0, result.stdout
    assert "sanitized request payload" in result.output
    assert "reasoning_split" in result.output
    assert "thinking" in result.output
    assert "timeout_seconds" in result.output
    # The api key MUST NOT appear, even as a prefix.
    assert "sk-" not in result.output


def test_llm_smoke_test_debug_request_never_prints_api_key(
    tmp_path, monkeypatch
) -> None:
    """Regression test: the --debug-request output must never include
    the configured API key value, even if it's set."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "sk-MiniMax-supersecret-xyz-987")
    get_settings.cache_clear()

    runner = CliRunner()
    result = runner.invoke(app, ["llm-smoke-test", "--debug-request"])

    assert result.exit_code == 0, result.stdout
    assert "sk-MiniMax-supersecret" not in result.output
    assert "never printed" in result.output  # we explicitly say so


def test_llm_smoke_test_reports_unparseable_response(
    tmp_path, monkeypatch
) -> None:
    """A completely unparseable response is reported as FAIL with code 2."""
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    get_settings.cache_clear()

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "not json at all, sorry",
            }
        }],
        "model": "fake-model",
    }
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=fake_response):
        runner = CliRunner()
        result = runner.invoke(app, ["llm-smoke-test"])

    assert result.exit_code == 2, result.stdout
    assert "FAIL" in result.output
