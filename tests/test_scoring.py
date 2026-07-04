"""Tests for analysis/scoring.py — the 8-factor opportunity scorer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from founder_radar.analysis.scoring import (
    ScoreFactors,
    compute_deterministic_scores,
)
from founder_radar.database.models import Post


def _post(title: str, body: str = "", score: int = 5, comments: int = 2) -> Post:
    return Post(
        source="reddit",
        external_id=title,
        source_category="test",
        title=title,
        body=body,
        author="op",
        url=None,
        score=score,
        num_comments=comments,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


# -------------------------------------------------------------------------
# Defaults and basic accessors
# -------------------------------------------------------------------------
def test_score_factors_default_to_neutral() -> None:
    s = ScoreFactors()
    assert s.frequency == 0.0
    assert s.emotional_intensity == 0.0
    assert s.dissatisfaction == 0.0
    assert s.market_size == 0.5
    assert s.ease_of_implementation == 0.5
    assert s.recurring_revenue == 0.5
    assert s.technical_feasibility == 0.5
    assert s.novelty == 0.0


def test_total_score_is_unweighted_mean() -> None:
    s = ScoreFactors(
        frequency=0.5,
        emotional_intensity=0.6,
        dissatisfaction=0.7,
        market_size=0.8,
        ease_of_implementation=0.9,
        recurring_revenue=1.0,
        technical_feasibility=0.4,
        novelty=0.2,
    )
    assert abs(s.total - (0.5 + 0.6 + 0.7 + 0.8 + 0.9 + 1.0 + 0.4 + 0.2) / 8) < 1e-9


def test_confidence_no_llm_is_half() -> None:
    s = ScoreFactors()
    assert s.confidence == 0.5


def test_confidence_with_llm_is_higher() -> None:
    s = ScoreFactors()
    s.llm_filled = {"market_size"}
    assert s.confidence > 0.5
    s.llm_filled = {"market_size", "ease_of_implementation",
                    "recurring_revenue", "technical_feasibility"}
    assert s.confidence == 1.0


# -------------------------------------------------------------------------
# Deterministic scoring
# -------------------------------------------------------------------------
def test_compute_empty_returns_safe_defaults() -> None:
    s = compute_deterministic_scores([])
    assert s.frequency == 0.0
    assert s.emotional_intensity == 0.0
    assert s.dissatisfaction == 0.0
    assert s.novelty == 0.5
    assert s.confidence == 0.5


def test_compute_frequency_grows_with_post_count() -> None:
    one = compute_deterministic_scores([_post("p1")])
    ten = compute_deterministic_scores([_post(f"p{i}") for i in range(10)])
    hundred = compute_deterministic_scores([_post(f"p{i}") for i in range(100)])
    assert one.frequency < ten.frequency < hundred.frequency
    assert 0.0 <= one.frequency <= 1.0
    assert hundred.frequency <= 1.0


def test_compute_emotional_intensity_picks_up_frustration() -> None:
    neutral = [_post("q1", "How do I do X?")]
    frustrated = [_post("hate this", "I hate this stupid thing")]
    s_neutral = compute_deterministic_scores(neutral)
    s_frustrated = compute_deterministic_scores(frustrated)
    assert s_frustrated.emotional_intensity > s_neutral.emotional_intensity


def test_compute_dissatisfaction_picks_up_switching_cues() -> None:
    posts_just_asking = [_post("How do I find customers?")]
    posts_switched = [_post("I switched from Mailchimp", "Fed up with it")]
    s_asking = compute_deterministic_scores(posts_just_asking)
    s_switched = compute_deterministic_scores(posts_switched)
    assert s_switched.dissatisfaction > s_asking.dissatisfaction


def test_compute_novelty_higher_with_fewer_competitor_mentions() -> None:
    fresh = [_post("nobody solves this")]
    crowded = [_post(
        "vs. Mailchimp vs. ConvertKit vs. Sendgrid",
        "Better alternative to Mailchimp",
    )]
    s_fresh = compute_deterministic_scores(fresh)
    s_crowded = compute_deterministic_scores(crowded)
    assert s_fresh.novelty > s_crowded.novelty


def test_compute_returns_score_factors_with_no_llm() -> None:
    s = compute_deterministic_scores([_post("p1", "Body text")])
    assert s.llm_filled == set()
    assert s.confidence == 0.5


def test_compute_does_not_mutate_post_objects() -> None:
    p = _post("p1", "body")
    compute_deterministic_scores([p])
    assert not hasattr(p, "frequency_score")


def test_log_scale_monotonic() -> None:
    from founder_radar.analysis.scoring import _log_scale
    scores = [_log_scale(n) for n in [0, 1, 5, 10, 50, 100, 1000]]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1]
    assert scores[-1] == 1.0
    assert scores[0] == 0.0


# -------------------------------------------------------------------------
# Phase 3+ weighted scoring (pain-dominated)
# -------------------------------------------------------------------------
def test_pain_subscore_weights_dissatisfaction_and_emotion_highest() -> None:
    """Pain = dissatisfaction * 0.4 + emotional_intensity * 0.4 + frequency * 0.2."""
    s = ScoreFactors(
        dissatisfaction=0.5, emotional_intensity=0.5, frequency=1.0,
    )
    # pain = 0.4*0.5 + 0.4*0.5 + 0.2*1.0 = 0.6
    assert abs(s.pain - 0.6) < 1e-9


def test_monetization_subscore_weights_recurring_revenue_highest() -> None:
    s = ScoreFactors(
        recurring_revenue=1.0, market_size=0.0, ease_of_implementation=0.0,
    )
    assert abs(s.monetization - 0.4) < 1e-9


def test_weighted_is_pain_dominated() -> None:
    s = ScoreFactors(
        dissatisfaction=1.0, emotional_intensity=1.0, frequency=1.0,
        market_size=0.0, ease_of_implementation=0.0,
        recurring_revenue=0.0, novelty=0.0,
    )
    # weighted = pain * 0.5 + monetization * 0.4 + novelty * 0.1
    #         = 1.0 * 0.5 + 0.0 + 0.0 = 0.5
    assert abs(s.weighted - 0.5) < 1e-9


def test_weighted_ignores_novelty_for_high_pain() -> None:
    """A high-pain saturated market should still rank high despite low novelty."""
    s_pain_saturated = ScoreFactors(
        dissatisfaction=1.0, emotional_intensity=1.0, frequency=1.0,
        novelty=0.0,
    )
    s_novelty_lite = ScoreFactors(
        dissatisfaction=0.3, emotional_intensity=0.3, frequency=0.3,
        novelty=1.0,
    )
    assert s_pain_saturated.weighted > s_novelty_lite.weighted


def test_weighted_is_in_unit_interval() -> None:
    s = ScoreFactors(
        frequency=1.0, emotional_intensity=1.0, dissatisfaction=1.0,
        market_size=1.0, ease_of_implementation=1.0,
        recurring_revenue=1.0, technical_feasibility=1.0, novelty=1.0,
    )
    assert 0.0 <= s.weighted <= 1.0


def test_weighted_zero_when_all_factors_zero() -> None:
    """When *all* factors are zero, weighted is zero.

    Default ScoreFactors() has the 4 LLM-judgment factors at 0.5
    (neutral). To get a true zero weighted, we override them.
    """
    s = ScoreFactors(
        market_size=0.0,
        ease_of_implementation=0.0,
        recurring_revenue=0.0,
        technical_feasibility=0.0,
    )
    assert s.weighted == 0.0


def test_pain_uses_deterministic_factors_only() -> None:
    """Pain must not depend on LLM-filled factors.

    Per the brief: pain intensity, frequency, dissatisfaction are the
    core. LLM factors (market_size, ease, recurring_revenue, technical)
    do NOT affect pain.
    """
    s = ScoreFactors(
        dissatisfaction=0.5, emotional_intensity=0.5, frequency=0.5,
        market_size=0.0,
    )
    # pain = 0.5*0.4 + 0.5*0.4 + 0.5*0.2 = 0.5
    assert abs(s.pain - 0.5) < 1e-9