"""Tests for the Reality Check Layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from founder_radar.analysis.reality_check import (
    KNOWN_COMPETITORS,
    RealityCheck,
    run_reality_check,
)
from founder_radar.database.models import Post


def _post(title: str, body: str = "") -> Post:
    return Post(
        source="reddit",
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


# -------------------------------------------------------------------------
# Empty / degenerate inputs
# -------------------------------------------------------------------------
def test_empty_posts_returns_empty_check() -> None:
    rc = run_reality_check([])
    assert rc.competitors == []
    assert rc.distinct_competitor_count == 0
    assert rc.competitor_mention_count == 0
    assert rc.saturation_score == 0.0
    assert rc.has_real_competition is False
    assert rc.is_saturated is False


# -------------------------------------------------------------------------
# Lexicon-based detection
# -------------------------------------------------------------------------
def test_detects_known_competitor_from_lexicon() -> None:
    """Mentioning a known SaaS name surfaces it as a competitor."""
    rc = run_reality_check([
        _post("Switched from Mailchimp", "Mailchimp was too expensive."),
    ])
    assert "Mailchimp" in rc.competitors
    assert rc.distinct_competitor_count >= 1
    assert rc.competitor_mention_count >= 1


def test_case_insensitive_lexicon_match() -> None:
    rc = run_reality_check([
        _post("I love notion for notes"),
    ])
    assert any("notion" in c.lower() for c in rc.competitors)


def test_word_boundary_prevents_partial_match() -> None:
    """'noted' must NOT match 'notion'."""
    rc = run_reality_check([_post("I noted this down")])
    assert rc.distinct_competitor_count == 0


def test_lexicon_deduplicates_case_insensitive() -> None:
    """Mailchimp and mailchimp and MAILCHIMP all collapse to one entry."""
    rc = run_reality_check([
        _post("Mailchimp is fine", "mailchimp works for me"),
    ])
    # Should have one "Mailchimp" entry, not three.
    mailchimp_count = sum(1 for c in rc.competitors if c.lower() == "mailchimp")
    assert mailchimp_count == 1


# -------------------------------------------------------------------------
# Regex-based detection
# -------------------------------------------------------------------------
def test_detects_competitor_from_alternative_to_phrase() -> None:
    rc = run_reality_check([
        _post("Looking for an alternative to SuperSpecialApp"),
    ])
    # "SuperSpecialApp" should appear (capitalized noun phrase after "alternative to").
    assert any("SuperSpecialApp" in c for c in rc.competitors)


def test_detects_competitor_from_switched_from_phrase() -> None:
    rc = run_reality_check([
        _post("I switched from OldThing to NewThing"),
    ])
    # We capture the noun after "switched from" — that's OldThing.
    assert any("OldThing" in c for c in rc.competitors)


def test_skips_very_short_candidates() -> None:
    """One-letter matches shouldn't appear."""
    rc = run_reality_check([
        _post("alternative to X"),  # single letter, len < 2
    ])
    # "X" is one character; we filter it.
    assert rc.distinct_competitor_count == 0


def test_skips_lowercase_candidates_from_regex() -> None:
    """Regex candidates must start with a capital letter (proper nouns)."""
    rc = run_reality_check([
        _post("alternative to lowercaseThing"),
    ])
    # "lowercaseThing" doesn't start with capital — we skip it.
    # The regex returns "lowercaseThing" but the filter rejects it.
    assert rc.distinct_competitor_count == 0


# -------------------------------------------------------------------------
# Saturation scoring
# -------------------------------------------------------------------------
def test_zero_competitors_is_zero_saturation() -> None:
    rc = run_reality_check([_post("no competitors here")])
    assert rc.saturation_score == 0.0


def test_one_competitor_is_low_saturation() -> None:
    rc = run_reality_check([
        _post("I use Mailchimp for emails"),
    ])
    # 1 distinct, ~1 mention in 1 post -> low base + density bonus.
    assert 0.0 < rc.saturation_score < 0.5


def test_many_competitors_is_high_saturation() -> None:
    posts = [
        _post("Use Mailchimp or ConvertKit or ActiveCampaign or Sendinblue"),
    ]
    rc = run_reality_check(posts)
    assert rc.distinct_competitor_count >= 4
    assert rc.saturation_score >= 0.5


def test_saturation_capped_at_one() -> None:
    """Even with a million competitors, score stays in [0, 1]."""
    text = " ".join(
        f"alternative to Name{i}" for i in range(100)
    )
    rc = run_reality_check([_post(text)])
    assert 0.0 <= rc.saturation_score <= 1.0


def test_has_real_competition_requires_two() -> None:
    rc1 = run_reality_check([_post("Use Mailchimp")])
    rc2 = run_reality_check([
        _post("Mailchimp and ConvertKit are both fine"),
    ])
    assert rc1.has_real_competition is False
    assert rc2.has_real_competition is True


def test_is_saturated_threshold() -> None:
    """`is_saturated` returns True only above 0.7."""
    rc = run_reality_check([
        _post("Use Mailchimp"),
    ])
    assert rc.is_saturated is False


# -------------------------------------------------------------------------
# Mention count
# -------------------------------------------------------------------------
def test_mention_count_is_total_occurrences() -> None:
    """Two posts mentioning Mailchimp -> competitor_mention_count >= 2."""
    rc = run_reality_check([
        _post("p1", "Mailchimp is fine"),
        _post("p2", "Mailchimp costs too much"),
    ])
    assert rc.competitor_mention_count >= 2
    assert rc.distinct_competitor_count == 1


def test_lexicon_size_is_sane() -> None:
    """Sanity: we have a meaningful competitor list (not too small, not absurdly large)."""
    assert 50 <= len(KNOWN_COMPETITORS) <= 500


# -------------------------------------------------------------------------
# Dataclass shape
# -------------------------------------------------------------------------
def test_reality_check_dataclass_fields() -> None:
    """The dataclass exposes the documented fields."""
    rc = RealityCheck(
        competitors=["a", "b"],
        distinct_competitor_count=2,
        competitor_mention_count=3,
        saturation_score=0.5,
    )
    assert rc.competitors == ["a", "b"]
    assert rc.distinct_competitor_count == 2
    assert rc.competitor_mention_count == 3
    assert rc.saturation_score == 0.5
    assert rc.has_real_competition is True


def test_reality_check_defaults() -> None:
    rc = RealityCheck()
    assert rc.competitors == []
    assert rc.distinct_competitor_count == 0
    assert rc.competitor_mention_count == 0
    assert rc.saturation_score == 0.0