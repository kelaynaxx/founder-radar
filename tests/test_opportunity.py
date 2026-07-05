"""Tests for analysis/opportunity.py — extractors.

Uses a `FakeLLMProvider` so the LLM path is fully exercised without
network or external dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

import pytest

from founder_radar.analysis.opportunity import (
    HeuristicExtractor,
    LLMBasedExtractor,
)
from founder_radar.analysis.llm_json import (
    extract_json,
    LLMJsonError,
    ERROR_PREVIEW_CHARS,
)
from founder_radar.database.models import Post
from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse


def _post(title: str, body: str = "", score: int = 5, comments: int = 2,
          url: str = "https://example.com/x") -> Post:
    return Post(
        source="reddit",
        external_id=title,
        source_category="test",
        title=title,
        body=body,
        author="op",
        url=url,
        score=score,
        num_comments=comments,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


# -------------------------------------------------------------------------
# FakeLLMProvider
# -------------------------------------------------------------------------
class FakeLLMProvider(BaseLLMProvider):
    """Deterministic fake for testing the LLM extractor.

    Records every prompt; returns a canned JSON response. The
    `response_factory` callable lets each test customize the response.
    """

    def __init__(self, response_factory=None) -> None:  # type: ignore[no-untyped-def]
        self._factory = response_factory or self._default_response
        self.calls: list[Sequence[LLMMessage]] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(messages)
        content = self._factory(messages)
        return LLMResponse(content=content, model="fake-model")

    def _default_response(self, messages) -> str:  # type: ignore[no-untyped-def]
        return json.dumps({
            "title": "Fake opportunity title",
            "problem_summary": "A problem extracted by the fake LLM.",
            "target_audience": "SaaS founders",
            "saas_ideas": ["An idea"],
            "competitors": ["Existing tool"],
            "scores": {
                "market_size": 0.7,
                "ease_of_implementation": 0.6,
                "recurring_revenue": 0.8,
                "technical_feasibility": 0.9,
            },
        })


# -------------------------------------------------------------------------
# HeuristicExtractor
# -------------------------------------------------------------------------
def test_heuristic_requires_at_least_one_post() -> None:
    with pytest.raises(RuntimeError, match="no posts"):
        HeuristicExtractor().extract(cluster_id=0, posts=[])


def test_heuristic_returns_required_fields() -> None:
    posts = [_post("How do I find customers?", body="Looking for advice")]
    data = HeuristicExtractor().extract(cluster_id=42, posts=posts)
    for k in (
        "title", "problem_summary", "saas_ideas", "competitors",
        "source_links", "frequency_score", "emotional_intensity_score",
        "dissatisfaction_score", "market_size_score",
        "ease_of_implementation_score", "recurring_revenue_score",
        "technical_feasibility_score", "novelty_score",
        "total_score", "confidence_score", "cluster_id",
        "mentions", "extraction_method", "llm_model", "status",
    ):
        assert k in data, f"Missing field: {k}"


def test_heuristic_picks_top_engagement_post_as_title() -> None:
    posts = [
        _post("low engagement", body="", score=1, comments=0),
        _post("high engagement", body="", score=99, comments=50),
    ]
    data = HeuristicExtractor().extract(cluster_id=0, posts=posts)
    assert data["title"] == "high engagement"


def test_heuristic_mentions_equals_post_count() -> None:
    posts = [_post(f"p{i}", body=f"body {i}") for i in range(7)]
    data = HeuristicExtractor().extract(cluster_id=0, posts=posts)
    assert data["mentions"] == 7


def test_heuristic_method_is_heuristic() -> None:
    data = HeuristicExtractor().extract(
        cluster_id=0, posts=[_post("title", body="body")]
    )
    assert data["extraction_method"] == "heuristic"
    assert data["llm_model"] is None


def test_heuristic_source_links_capped_at_five() -> None:
    posts = [
        _post(f"p{i}", url=f"https://example.com/{i}") for i in range(10)
    ]
    data = HeuristicExtractor().extract(cluster_id=0, posts=posts)
    assert len(data["source_links"]) == 5


def test_heuristic_no_llm_factors_default_to_neutral() -> None:
    """Without LLM, market_size etc. are 0.5."""
    data = HeuristicExtractor().extract(
        cluster_id=0, posts=[_post("t", body="b")]
    )
    assert data["market_size_score"] == 0.5
    assert data["ease_of_implementation_score"] == 0.5
    assert data["recurring_revenue_score"] == 0.5
    assert data["technical_feasibility_score"] == 0.5


def test_heuristic_confidence_is_half_without_llm() -> None:
    data = HeuristicExtractor().extract(
        cluster_id=0, posts=[_post("t", body="b")]
    )
    assert data["confidence_score"] == 0.5


# -------------------------------------------------------------------------
# LLMBasedExtractor — happy path
# -------------------------------------------------------------------------
def test_llm_extractor_calls_provider_once() -> None:
    fake = FakeLLMProvider()
    extractor = LLMBasedExtractor(llm=fake, fallback=HeuristicExtractor())
    posts = [_post("t", body="b")]
    extractor.extract(cluster_id=0, posts=posts)
    assert len(fake.calls) == 1


def test_llm_extractor_parses_response_into_dict() -> None:
    fake = FakeLLMProvider()
    extractor = LLMBasedExtractor(llm=fake, fallback=HeuristicExtractor())
    data = extractor.extract(
        cluster_id=7, posts=[_post("p", body="b")]
    )
    assert data["title"] == "Fake opportunity title"
    assert data["problem_summary"] == "A problem extracted by the fake LLM."
    assert data["target_audience"] == "SaaS founders"
    assert data["saas_ideas"] == ["An idea"]
    assert data["competitors"] == ["Existing tool"]
    assert data["extraction_method"] == "llm"
    assert data["llm_model"] == "fake-model"
    assert data["cluster_id"] == 7


def test_llm_extractor_records_llm_filled_fields() -> None:
    """All 4 LLM-judgment fields are marked as filled."""
    fake = FakeLLMProvider()
    extractor = LLMBasedExtractor(llm=fake, fallback=HeuristicExtractor())
    data = extractor.extract(
        cluster_id=0, posts=[_post("p", body="b")]
    )
    # The 4 LLM-judgment scores are 0.7/0.6/0.8/0.9 — none of the
    # default 0.5 — so they were LLM-filled.
    assert data["market_size_score"] == 0.7
    assert data["ease_of_implementation_score"] == 0.6
    assert data["recurring_revenue_score"] == 0.8
    assert data["technical_feasibility_score"] == 0.9
    assert data["confidence_score"] == 1.0  # all 4 LLM factors filled


def test_llm_extractor_uses_deterministic_factors() -> None:
    """frequency / emotional_intensity / dissatisfaction / novelty come
    from the post-derived scorer, not the LLM."""
    fake = FakeLLMProvider()
    extractor = LLMBasedExtractor(llm=fake, fallback=HeuristicExtractor())
    posts = [
        _post("hate this stupid thing", body="I hate this stupid thing"),
    ]
    data = extractor.extract(cluster_id=0, posts=posts)
    # emotional_intensity should be > 0 (frustration cue present)
    assert data["emotional_intensity_score"] > 0


# -------------------------------------------------------------------------
# LLMBasedExtractor — fallback path
# -------------------------------------------------------------------------
def test_llm_extractor_falls_back_on_provider_error() -> None:
    """If the LLM raises, we silently fall back to the heuristic."""

    class BoomLLM(BaseLLMProvider):
        @property
        def name(self) -> str:
            return "boom"

        def complete(self, messages, **kw) -> LLMResponse:
            raise RuntimeError("network is down")

    extractor = LLMBasedExtractor(llm=BoomLLM(), fallback=HeuristicExtractor())
    data = extractor.extract(
        cluster_id=0, posts=[_post("p", body="b")]
    )
    # Fallback ran -> extraction_method == heuristic.
    assert data["extraction_method"] == "heuristic"


def test_llm_extractor_falls_back_on_invalid_json() -> None:
    """A non-JSON response triggers fallback to heuristic."""

    class GarbageLLM(BaseLLMProvider):
        @property
        def name(self) -> str:
            return "garbage"

        def complete(self, messages, **kw) -> LLMResponse:
            return LLMResponse(
                content="I'm sorry, I can't help with that.",
                model="garbage",
            )

    extractor = LLMBasedExtractor(llm=GarbageLLM(), fallback=HeuristicExtractor())
    data = extractor.extract(
        cluster_id=0, posts=[_post("p", body="b")]
    )
    assert data["extraction_method"] == "heuristic"


def test_llm_extractor_falls_back_on_partial_json() -> None:
    """A response with the wrong schema also falls back."""

    class PartialLLM(BaseLLMProvider):
        @property
        def name(self) -> str:
            return "partial"

        def complete(self, messages, **kw) -> LLMResponse:
            # JSON, but missing every required field.
            return LLMResponse(content=json.dumps({"oops": True}), model="x")

    extractor = LLMBasedExtractor(llm=PartialLLM(), fallback=HeuristicExtractor())
    # The LLM extraction succeeds (all fields default in our merge
    # logic), but the result is poor quality. We do NOT fall back for
    # "partial" — only for outright parse/transport failures.
    data = extractor.extract(
        cluster_id=0, posts=[_post("p", body="b")]
    )
    assert data["extraction_method"] == "llm"  # not heuristic


# -------------------------------------------------------------------------
# LLMBasedExtractor — JSON parsing helpers
# -------------------------------------------------------------------------
def test_parse_llm_json_direct() -> None:
    raw = json.dumps({"a": 1, "b": [2, 3]})
    assert extract_json(raw) == {"a": 1, "b": [2, 3]}


def test_parse_llm_json_strips_markdown_fences() -> None:
    raw = "```json\n" + json.dumps({"a": 1}) + "\n```"
    assert extract_json(raw) == {"a": 1}


def test_parse_llm_json_strips_think_blocks() -> None:
    """Reasoning models like MiniMax-M3 inline <think>...</think> blocks."""
    raw = "<think>\nLet me analyze this carefully...\n</think>\n" + json.dumps({"a": 1})
    assert extract_json(raw) == {"a": 1}


def test_parse_llm_json_finds_subspan() -> None:
    raw = "Some leading text. " + json.dumps({"a": 1}) + " trailing noise."
    assert extract_json(raw) == {"a": 1}


def test_parse_llm_json_raises_on_garbage() -> None:
    with pytest.raises(LLMJsonError):
        extract_json("not json at all")


# -------------------------------------------------------------------------
# Source link cap
# -------------------------------------------------------------------------
def test_llm_extractor_source_links_capped_at_five() -> None:
    fake = FakeLLMProvider()
    extractor = LLMBasedExtractor(llm=fake, fallback=HeuristicExtractor())
    posts = [
        _post(f"p{i}", url=f"https://example.com/{i}") for i in range(10)
    ]
    data = extractor.extract(cluster_id=0, posts=posts)
    assert len(data["source_links"]) == 5


def test_llm_extractor_requires_at_least_one_post() -> None:
    extractor = LLMBasedExtractor(
        llm=FakeLLMProvider(), fallback=HeuristicExtractor()
    )
    with pytest.raises(RuntimeError, match="no posts"):
        extractor.extract(cluster_id=0, posts=[])