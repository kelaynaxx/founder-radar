"""Tests for the Phase 3.5 Reality Validation Layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from founder_radar.analysis.reality_validator import (
    ALL_STATUSES,
    COMPETITIVE_STRENGTH_LOWER,
    DISSATISFACTION_HITS_THRESHOLD,
    PAIN_DENSITY_THRESHOLD,
    RealityAssessment,
    SATURATION_COUNT_THRESHOLD,
    SATURATION_STRENGTH_THRESHOLD,
    STATUS_COMPETITIVE,
    STATUS_SATURATED,
    STATUS_UNDERSERVED,
    STATUS_UNKNOWN,
    UNDERSERVED_STRENGTH_CEILING,
    assess_reality,
    _competitor_strength,
    _count_dissatisfaction_cues,
    _count_pain_cues,
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
def test_empty_posts_returns_unknown() -> None:
    rc = assess_reality([])
    assert rc.status == STATUS_UNKNOWN
    assert rc.saturation_confidence < 0.5
    assert rc.competitor_strength_estimate == 0.0
    assert rc.evidence  # at least one explanation line


def test_no_competitors_no_pain_returns_unknown() -> None:
    """Generic praise without competitors or complaints -> unknown."""
    posts = [_post("I love using this app", "It's great for my workflow.")]
    rc = assess_reality(posts)
    assert rc.status == STATUS_UNKNOWN
    # Should mention "no competitors" and "no frustration cues" in evidence.
    assert any("competitor" in e.lower() for e in rc.evidence)


def test_assess_reality_accepts_precomputed_competitor_info() -> None:
    """Caller can supply competitor info to skip the regex pass."""
    posts = [_post("I hate this stupid thing")]
    rc = assess_reality(
        posts,
        competitors=["BigCorp", "OtherCorp"],
        distinct_competitor_count=2,
        competitor_mention_count=2,
    )
    # `RealityAssessment` doesn't expose distinct_competitor_count as a field
    # (the caller already had it). Verify it influenced the strength.
    assert rc.competitor_strength_estimate > 0
    # Should still detect dissatisfaction cues.
    assert any("dissatisfaction" in e.lower() or "switched" in e.lower()
               or "cancelled" in e.lower() or "fed up" in e.lower()
               for e in rc.evidence)


# -------------------------------------------------------------------------
# Saturated detection
# -------------------------------------------------------------------------
def test_detects_saturated_market() -> None:
    """Many well-known competitors + no complaints = saturated."""
    posts = [
        _post("p1", "I use Mailchimp for emails"),
        _post("p2", "HubSpot is great for our CRM"),
        _post("p3", "Salesforce is the standard"),
        _post("p4", "ConvertKit is a good option"),
    ]
    rc = assess_reality(posts)
    assert rc.status == STATUS_SATURATED
    assert rc.saturation_confidence >= 0.5
    assert rc.competitor_strength_estimate >= 0.5


def test_saturated_requires_minimum_distinct_count() -> None:
    """< 3 competitors + high strength -> NOT saturated (need count too)."""
    # Only 1 competitor but lots of mentions -> competitor_strength may
    # still be high, but distinct_count < 3 blocks "saturated".
    posts = [_post("p", "Mailchimp Mailchimp Mailchimp Mailchimp")]
    rc = assess_reality(posts)
    assert rc.status != STATUS_SATURATED


# -------------------------------------------------------------------------
# Competitive detection
# -------------------------------------------------------------------------
def test_detects_competitive_market() -> None:
    """Some competitors + explicit complaints about them = competitive."""
    posts = [
        _post("p1", "I switched from Mailchimp — it was too expensive."),
        _post("p2", "I cancelled my HubSpot subscription; too bloated."),
        _post("p3", "Mailchimp's automation is broken; tired of it."),
    ]
    rc = assess_reality(posts)
    assert rc.status == STATUS_COMPETITIVE
    assert rc.saturation_confidence > 0.0


def test_competitive_requires_dissatisfaction_signals() -> None:
    """Some competitors + happy users = NOT competitive (just saturated-ish)."""
    posts = [
        _post("p1", "Mailchimp works fine for our emails"),
        _post("p2", "I love HubSpot, it's so easy"),
        _post("p3", "Mailchimp is reliable"),
    ]
    rc = assess_reality(posts)
    # Has competitors but no dissatisfaction — NOT "competitive".
    # It might be "saturated" or "unknown" depending on distinct count.
    assert rc.status != STATUS_COMPETITIVE


# -------------------------------------------------------------------------
# Underserved detection
# -------------------------------------------------------------------------
def test_detects_underserved_market() -> None:
    """Few/no competitors + strong pain signals = underserved."""
    posts = [
        _post("p1", "I hate this stupid thing. frustrated, broken."),
        _post("p2", "terrible experience, hate it. annoyed."),
        _post("p3", "this is so painful, hate. frustrating."),
    ]
    rc = assess_reality(posts)
    assert rc.status == STATUS_UNDERSERVED
    assert rc.saturation_confidence > 0.0


def test_underserved_requires_low_competitor_strength() -> None:
    """Pain signals + many competitors = NOT underserved."""
    posts = [
        _post("p1", "Mailchimp sucks and HubSpot sucks too"),
        _post("p2", "I hate Salesforce, terrible experience"),
        _post("p3", "frustrated with Mailchimp and ConvertKit"),
    ]
    rc = assess_reality(posts)
    assert rc.status != STATUS_UNDERSERVED


# -------------------------------------------------------------------------
# Unknown classification
# -------------------------------------------------------------------------
def test_no_clear_signal_returns_unknown() -> None:
    """Generic posts without strong pain or competitor mentions."""
    posts = [
        _post("just wondering about something"),
        _post("another generic post"),
    ]
    rc = assess_reality(posts)
    assert rc.status == STATUS_UNKNOWN


def test_unknown_confidence_is_low() -> None:
    posts = [_post("a", "b")]
    rc = assess_reality(posts)
    if rc.status == STATUS_UNKNOWN:
        assert rc.saturation_confidence <= 0.3


# -------------------------------------------------------------------------
# RealityAssessment dataclass
# -------------------------------------------------------------------------
def test_reality_assessment_defaults() -> None:
    rc = RealityAssessment()
    assert rc.status == STATUS_UNKNOWN
    assert rc.saturation_confidence == 0.0
    assert rc.evidence == []
    assert rc.competitor_strength_estimate == 0.0
    assert rc.is_viable is False


def test_reality_assessment_is_viable() -> None:
    """Only 'competitive' and 'underserved' are 'viable' (worth building)."""
    for status, expected in [
        (STATUS_SATURATED, False),
        (STATUS_COMPETITIVE, True),
        (STATUS_UNDERSERVED, True),
        (STATUS_UNKNOWN, False),
    ]:
        rc = RealityAssessment(status=status)
        assert rc.is_viable is expected, f"failed for {status}"


def test_all_statuses_constant_is_complete() -> None:
    assert set(ALL_STATUSES) == {
        STATUS_SATURATED,
        STATUS_COMPETITIVE,
        STATUS_UNDERSERVED,
        STATUS_UNKNOWN,
}


# -------------------------------------------------------------------------
# Internal signal helpers
# -------------------------------------------------------------------------
def test_pain_cue_counter_finds_keywords() -> None:
    posts = [
        _post("p1", "I hate this stupid thing"),
        _post("p2", "frustrated broken terrible"),
    ]
    assert _count_pain_cues(posts) >= 5  # hate, stupid, frustrated, broken, terrible


def test_dissatisfaction_cue_counter_finds_keywords() -> None:
    posts = [
        _post("p1", "I switched from Mailchimp; gave up on it"),
        _post("p2", "cancelled my subscription, fed up"),
    ]
    # "switched from" + "gave up on" + "cancelled" = 3 distinct cue phrases.
    assert _count_dissatisfaction_cues(posts) >= 3


def test_cue_counters_handle_no_text() -> None:
    posts = [
        Post(source="r", external_id="x", source_category="t",
             title=None, body=None, url=None, score=0, num_comments=0,
             created_at=None, collected_at=None),
    ]
    # Should not crash.
    assert _count_pain_cues(posts) == 0
    assert _count_dissatisfaction_cues(posts) == 0


def test_competitor_strength_zero_with_no_competitors() -> None:
    s = _competitor_strength(
        distinct_competitor_count=0,
        competitor_mention_count=0,
        n_posts=5,
        competitors=[],
    )
    assert s == 0.0


def test_competitor_strength_grows_with_count_and_density() -> None:
    s1 = _competitor_strength(
        distinct_competitor_count=1, competitor_mention_count=1,
        n_posts=5, competitors=["A"],
    )
    s6 = _competitor_strength(
        distinct_competitor_count=6, competitor_mention_count=10,
        n_posts=5, competitors=["A", "B", "C", "D", "E", "F"],
    )
    assert s6 > s1
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s6 <= 1.0


def test_competitor_strength_boosts_for_known_lexicon() -> None:
    """Same count but lexicon matches should score higher."""
    s_unknown = _competitor_strength(
        distinct_competitor_count=3, competitor_mention_count=3,
        n_posts=5, competitors=["NewApp1", "NewApp2", "NewApp3"],
    )
    s_known = _competitor_strength(
        distinct_competitor_count=3, competitor_mention_count=3,
        n_posts=5, competitors=["Mailchimp", "HubSpot", "Salesforce"],
    )
    assert s_known > s_unknown


def test_competitor_strength_capped_at_one() -> None:
    s = _competitor_strength(
        distinct_competitor_count=100, competitor_mention_count=1000,
        n_posts=1, competitors=["A"] * 100,
    )
    assert s <= 1.0


# -------------------------------------------------------------------------
# Evidence list is human-readable and useful
# -------------------------------------------------------------------------
def test_evidence_is_non_empty_for_classified_results() -> None:
    """Whatever the status, evidence should explain why."""
    posts = [
        _post("hate this", "frustrated and broken"),
    ]
    rc = assess_reality(posts)
    assert rc.evidence
    assert all(isinstance(line, str) for line in rc.evidence)


def test_evidence_includes_competitor_names_when_found() -> None:
    posts = [
        _post("p", "I use Mailchimp and HubSpot and Salesforce"),
    ]
    rc = assess_reality(posts)
    # The competitor names should appear in the evidence list.
    blob = " ".join(rc.evidence)
    assert "Mailchimp" in blob or "mailchimp" in blob.lower()# =============================================================================
# Calibration pass: invariant tests (locked-in classification rules)
# =============================================================================
# These tests directly assert the documented invariants in the module
# docstring. They exist to prevent silent regressions if anyone tweaks
# _classify without re-running the audit.


def test_invariant_underserved_requires_pain_above_threshold() -> None:
    """Status 'underserved' MUST mean pain_density >= PAIN_DENSITY_THRESHOLD.

    Concretely: posts with low frustration density (below threshold)
    cannot be classified as underserved, even if competitors are absent.
    """
    posts = [
        _post("p1", "I wonder if anyone has tried approach X"),
        _post("p2", "Looking for recommendations on Y"),
        _post("p3", "Anyone using Z? Curious about your experience"),
    ]
    rc = assess_reality(posts)
    if rc.status == STATUS_UNDERSERVED:
        pytest.fail(
            f"Invariant violated: 'underserved' assigned despite "
            f"pain_density={rc.pain_density:.3f} "
            f"< {PAIN_DENSITY_THRESHOLD}"
        )


def test_invariant_no_competitors_no_pain_is_unknown_not_underserved() -> None:
    """When there are no competitors AND no pain cues, status MUST be 'unknown'.

    This is the explicit "borderline" case the user wants guarded.
    """
    posts = [
        _post("p1", "I'm researching topic X, anyone have thoughts?"),
        _post("p2", "Looking for advice on Y"),
        _post("p3", "Curious about Z"),
    ]
    rc = assess_reality(posts)
    assert rc.status != STATUS_UNDERSERVED, (
        f"Invariant violated: 'underserved' with no competitors AND "
        f"pain_density={rc.pain_density:.3f}"
    )
    assert rc.status == STATUS_UNKNOWN, (
        f"Expected 'unknown' for empty signals, got {rc.status!r}"
    )
    assert rc.pain_density == 0.0
    assert rc.distinct_competitor_count == 0
    assert rc.dissatisfaction_hits == 0


def test_invariant_saturated_requires_both_thresholds() -> None:
    """Status 'saturated' requires BOTH strength AND count thresholds.

    Specifically: high competitor_strength but fewer than 3 distinct
    competitors is NOT saturated.
    """
    posts = [
        _post("p", "Mailchimp Mailchimp Mailchimp Mailchimp Mailchimp"),
    ]
    rc = assess_reality(posts)
    assert rc.status != STATUS_SATURATED, (
        f"Invariant violated: 'saturated' with only 1 distinct competitor "
        f"(threshold={SATURATION_COUNT_THRESHOLD})"
    )


def test_invariant_saturated_requires_strength_threshold() -> None:
    """Niche competitors + no mention density -> NOT saturated.

    With 3 well-known SaaS names (Mailchimp, HubSpot, Salesforce),
    count_score + known_ratio alone push competitor_strength above
    the saturation threshold — that's the formula working as intended.

    To verify the strength threshold actually matters, we use NICHE
    (non-lexicon) competitor names. With low known_ratio and no
    mention density, the strength stays below threshold and status
    cannot be "saturated".
    """
    # Pass non-lexicon competitor names so known_ratio = 0.
    # Each post mentions one of the names once, so distinct=3,
    # mentions=3, density low.
    posts = [
        _post("p1", "I tried NicheToolA, it was okay"),
        _post("p2", "NicheToolB has some good features"),
        _post("p3", "NicheToolC might work too"),
    ]
    rc = assess_reality(
        posts,
        competitors=["NicheToolA", "NicheToolB", "NicheToolC"],
    )
    # The strength should be low because known_ratio=0 and density low.
    assert rc.competitor_strength_estimate < SATURATION_STRENGTH_THRESHOLD, (
        f"Test setup wrong: expected low competitor_strength, "
        f"got {rc.competitor_strength_estimate:.3f}"
    )
    assert rc.status != STATUS_SATURATED, (
        f"Invariant violated: 'saturated' with competitor_strength="
        f"{rc.competitor_strength_estimate:.3f} "
        f"< {SATURATION_STRENGTH_THRESHOLD}"
    )


def test_invariant_competitive_requires_dissatisfaction_cues() -> None:
    """Status 'competitive' requires dissatisfaction_hits >= threshold."""
    posts = [
        _post("p1", "Mailchimp works great for us"),
        _post("p2", "We love HubSpot, perfect fit"),
        _post("p3", "Mailchimp and HubSpot both reliable"),
    ]
    rc = assess_reality(posts, competitors=["Mailchimp", "HubSpot"])
    assert rc.status != STATUS_COMPETITIVE, (
        f"Invariant violated: 'competitive' with "
        f"dissatisfaction_hits={rc.dissatisfaction_hits} "
        f"< {DISSATISFACTION_HITS_THRESHOLD}"
    )


def test_invariant_underserved_documented_in_evidence() -> None:
    """When classified underserved, the evidence MUST show pain cues."""
    posts = [
        _post("p1", "I hate this stupid thing. frustrated, broken."),
        _post("p2", "terrible experience, hate it. annoyed."),
        _post("p3", "this is so painful, hate. frustrating."),
    ]
    rc = assess_reality(posts)
    assert rc.status == STATUS_UNDERSERVED
    blob = " ".join(rc.evidence).lower()
    assert "frustration" in blob, "Evidence should mention frustration cues"
    assert rc.pain_density >= PAIN_DENSITY_THRESHOLD, (
        "underserved requires pain_density >= PAIN_DENSITY_THRESHOLD"
    )


def test_invariant_reason_string_explains_status() -> None:
    """The reason string MUST mention relevant thresholds and signals."""
    posts = [
        _post("p1", "hate this, broken, terrible"),
        _post("p2", "frustrated, awful, useless"),
        _post("p3", "painful, hate, garbage"),
    ]
    rc = assess_reality(posts)
    assert rc.reason, "Reason string must be populated"
    # The reason starts with the status name (e.g. "Underserved: ...");
    # compare case-insensitively so we don't break when we tweak casing.
    assert rc.reason.lower().startswith(rc.status), (
        f"Reason {rc.reason!r} doesn't start with status {rc.status!r}"
    )
    if rc.status == STATUS_UNDERSERVED:
        assert "pain_density" in rc.reason
        assert str(PAIN_DENSITY_THRESHOLD) in rc.reason


def test_invariant_saturated_reason_mentions_thresholds() -> None:
    """For 'saturated', reason should reference both thresholds."""
    posts = [
        _post("p1", "Mailchimp HubSpot Salesforce ActiveCampaign ConvertKit"),
        _post("p2", "Mailchimp is standard HubSpot for CRM Salesforce"),
        _post("p3", "we use Mailchimp HubSpot Salesforce daily"),
    ]
    rc = assess_reality(posts)
    if rc.status == STATUS_SATURATED:
        assert "competitor_strength" in rc.reason
        assert "distinct_competitors" in rc.reason
        assert str(SATURATION_STRENGTH_THRESHOLD) in rc.reason
        assert str(SATURATION_COUNT_THRESHOLD) in rc.reason


def test_invariant_unknown_when_competitors_but_no_dissatisfaction() -> None:
    """Competitors exist + no complaints = unknown (not competitive, not saturated)."""
    posts = [
        _post("p1", "Mailchimp is fine"),
        _post("p2", "HubSpot works for us"),
        _post("p3", "Mailchimp is reliable, no complaints"),
    ]
    rc = assess_reality(posts, competitors=["Mailchimp", "HubSpot"])
    assert rc.status != STATUS_COMPETITIVE
    assert rc.status != STATUS_SATURATED


def test_invariant_unknown_when_pain_but_no_competitors_below_threshold() -> None:
    """Pain signals present but below density threshold -> unknown."""
    posts = [
        _post("p1", "this thing is mildly frustrating"),
        _post("p2", "no opinion really"),
        _post("p3", "looking for alternatives"),
        _post("p4", "curious about pricing"),
    ]
    rc = assess_reality(posts)
    assert rc.pain_density < PAIN_DENSITY_THRESHOLD
    assert rc.status != STATUS_UNDERSERVED


# =============================================================================
# Calibration pass: raw signal fields exposed on the dataclass
# =============================================================================


def test_dataclass_exposes_pain_density_field() -> None:
    """The audit pass added pain_density so the CLI can show it."""
    ra = RealityAssessment()
    assert hasattr(ra, "pain_density")
    assert ra.pain_density == 0.0


def test_dataclass_exposes_dissatisfaction_hits_field() -> None:
    ra = RealityAssessment()
    assert hasattr(ra, "dissatisfaction_hits")
    assert ra.dissatisfaction_hits == 0


def test_dataclass_exposes_distinct_competitor_count_field() -> None:
    ra = RealityAssessment()
    assert hasattr(ra, "distinct_competitor_count")
    assert ra.distinct_competitor_count == 0


def test_dataclass_exposes_reason_field() -> None:
    ra = RealityAssessment()
    assert hasattr(ra, "reason")
    assert ra.reason == ""


def test_assess_reality_populates_all_raw_signal_fields() -> None:
    """End-to-end: assess_reality must fill in all the audit fields."""
    posts = [
        _post("p1", "Mailchimp sucks, I hate it. frustrated."),
        _post("p2", "HubSpot is broken, terrible."),
        _post("p3", "I switched from Mailchimp, gave up on it."),
    ]
    rc = assess_reality(posts)
    assert rc.pain_density > 0
    assert rc.dissatisfaction_hits > 0
    assert rc.distinct_competitor_count >= 2
    assert rc.competitor_strength_estimate > 0
    assert rc.reason


def test_assess_reality_populates_raw_signals_for_empty_input() -> None:
    """Empty input: all raw signals are 0, reason is non-empty."""
    rc = assess_reality([])
    assert rc.pain_density == 0.0
    assert rc.dissatisfaction_hits == 0
    assert rc.distinct_competitor_count == 0
    assert rc.competitor_strength_estimate == 0.0
    assert rc.reason
    assert rc.status == STATUS_UNKNOWN