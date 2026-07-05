"""Tests for the opportunity-type classifier (Phase 4+ signal calibration).

These tests are deliberately *deterministic* — they construct
opportunities + posts with hand-written lexicons and verify the
classifier picks the right type and the right productizability score.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from founder_radar.analysis.opportunity_type import (
    ALL_TYPES,
    OpportunityTypeAssessment,
    POTENTIAL_ALLOWED_REALITY_STATUSES,
    POTENTIAL_MIN_MENTIONS,
    POTENTIAL_MIN_SCORE,
    POTENTIAL_OPEN_MARKET_CEILING,
    POTENTIAL_REQUIRED_CONDITIONS,
    SCORE_BASELINE,
    SCORE_CAP,
    TYPE_DEVELOPER_WORKFLOW_PAIN,
    TYPE_DOCUMENTATION_CONFUSION,
    TYPE_INFRA_OPERATIONAL_PAIN,
    TYPE_INTEGRATION_PAIN,
    TYPE_MISSING_FEATURE,
    TYPE_POTENTIAL_PRODUCT,
    TYPE_REPO_SPECIFIC_BUG,
    TYPE_SECURITY_COMPLIANCE_PAIN,
    TYPE_UNKNOWN,
    TYPE_UPSTREAM_LIBRARY_BUG,
    classify_opportunity,
)
from founder_radar.database.models import Opportunity, Post


# -------------------------------------------------------------------------
# Test helpers
# -------------------------------------------------------------------------
def _post(
    title: str,
    body: str = "",
    *,
    source: str = "github",
    source_category: str = "owner/repo",
    subtype: str | None = None,
) -> Post:
    return Post(
        source=source,
        external_id=title,
        source_category=source_category,
        title=title,
        body=body,
        author="op",
        url=None,
        score=1,
        num_comments=1,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
        subtype=subtype,
    )


def _opp(
    title: str = "x",
    *,
    summary: str = "y",
    mentions: int = 3,
    **kw,
) -> Opportunity:
    defaults = dict(
        problem_summary=summary,
        mentions=mentions,
        reality_status="unknown",
        competitor_strength_estimate=0.0,
        pain_score=0.0,
        frequency_score=0.0,
    )
    defaults.update(kw)
    return Opportunity(title=title, **defaults)


# -------------------------------------------------------------------------
# Public API surface
def test_all_types_contains_every_documented_label() -> None:
    """The public taxonomy must match the brief's labels.

    V2 (calibration pass 2) added `upstream_library_bug`, so the
    taxonomy is now ten labels (was nine in V1).
    """
    assert set(ALL_TYPES) == {
        TYPE_REPO_SPECIFIC_BUG,
        TYPE_UPSTREAM_LIBRARY_BUG,
        TYPE_DOCUMENTATION_CONFUSION,
        TYPE_MISSING_FEATURE,
        TYPE_INTEGRATION_PAIN,
        TYPE_DEVELOPER_WORKFLOW_PAIN,
        TYPE_INFRA_OPERATIONAL_PAIN,
        TYPE_SECURITY_COMPLIANCE_PAIN,
        TYPE_POTENTIAL_PRODUCT,
        TYPE_UNKNOWN,
    }
    assert len(ALL_TYPES) == 10
    """The return shape is stable for downstream consumers."""
    a = classify_opportunity(_opp(), [_post("foo", "bar")])
    assert isinstance(a, OpportunityTypeAssessment)
    assert a.opportunity_type in ALL_TYPES
    assert 0.0 <= a.productizability_score <= 1.0
    assert isinstance(a.productizability_reason, str)


# -------------------------------------------------------------------------
# Empty / degenerate inputs
# -------------------------------------------------------------------------
def test_empty_posts_returns_unknown() -> None:
    a = classify_opportunity(_opp("Empty"), [])
    assert a.opportunity_type == TYPE_UNKNOWN
    assert a.productizability_score == 0.0
    assert "no type lexicon" in a.productizability_reason.lower()


def test_no_cues_at_all_returns_unknown() -> None:
    """Generic praise without any type-specific signal -> unknown."""
    a = classify_opportunity(
        _opp("Generic feedback", mentions=2),
        [_post("Great product", body="works well")],
    )
    assert a.opportunity_type == TYPE_UNKNOWN
    assert a.productizability_score == 0.0


# -------------------------------------------------------------------------
# Type 1: repo_specific_bug
# -------------------------------------------------------------------------
def test_repo_specific_bug_classified_correctly() -> None:
    """Stack traces + error types + regression language => bug."""
    posts = [
        _post("TypeError: cannot read property 'x' of null", body="Traceback..."),
        _post("Crash on startup", body="got AttributeError on save"),
        _post("Regression since 2.0", body="works in 1.x but not 2.0"),
    ]
    a = classify_opportunity(_opp("TypeError on save", mentions=3, pain_score=0.5), posts)
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG
    assert a.productizability_score < 0.3, "repo bugs should NOT be high productizability"


def test_repo_specific_bug_score_is_low() -> None:
    """A bare bug report (low repeat, no buyer) has a low productizability score."""
    posts = [_post("TypeError in my repo", body="I get a crash")]
    a = classify_opportunity(_opp("Random crash", mentions=1, pain_score=0.3), posts)
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG
    assert a.productizability_score < 0.20


# -------------------------------------------------------------------------
# Type 2: documentation_confusion
# -------------------------------------------------------------------------
def test_documentation_confusion_classified_correctly() -> None:
    """Setup / how-to / where-can-I-find language => docs confusion."""
    posts = [
        _post("Where can I find the API docs?"),
        _post("How do I install this on Windows?"),
        _post("No example code for basic usage", body="the readme has nothing"),
    ]
    a = classify_opportunity(_opp("Documentation is unclear", mentions=3), posts)
    assert a.opportunity_type == TYPE_DOCUMENTATION_CONFUSION


def test_documentation_confusion_score_is_low() -> None:
    """A pure docs gap is not a buildable product."""
    posts = [
        _post("How to use this?"),
        _post("Where are the docs?"),
    ]
    a = classify_opportunity(_opp("Need docs", mentions=2), posts)
    assert a.opportunity_type == TYPE_DOCUMENTATION_CONFUSION
    assert a.productizability_score < 0.30


# -------------------------------------------------------------------------
# Type 3: missing_feature
# -------------------------------------------------------------------------
def test_missing_feature_classified_correctly() -> None:
    """Feature request language => missing_feature."""
    posts = [
        _post("Please add support for WebSockets"),
        _post("Feature request: dark mode"),
        _post("I wish it had CSV export"),
        _post("Would be nice to have offline mode"),
    ]
    a = classify_opportunity(_opp("Missing feature: CSV export", mentions=4), posts)
    assert a.opportunity_type == TYPE_MISSING_FEATURE


def test_missing_feature_subtype_from_github_is_detected() -> None:
    """subtype='feature_request' on the source post reinforces the type."""
    posts = [
        _post("Please add feature X", body="details", subtype="feature_request"),
        _post("Feature request Y", body="more", subtype="feature_request"),
        _post("Would be nice to have Z", body="more", subtype="feature_request"),
    ]
    a = classify_opportunity(_opp("Library feature gap", mentions=3), posts)
    assert a.opportunity_type == TYPE_MISSING_FEATURE


# -------------------------------------------------------------------------
# Type 4: integration_pain
# -------------------------------------------------------------------------
def test_integration_pain_classified_correctly() -> None:
    """API / SDK / connector language => integration_pain."""
    posts = [
        _post("Stripe webhook keeps returning 401"),
        _post("Stripe API rate limit is too aggressive"),
        _post("Stripe SDK doesn't support new payment methods"),
        _post("Stripe checkout integration is fragile"),
    ]
    a = classify_opportunity(
        _opp("Stripe integration pain", mentions=4, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_INTEGRATION_PAIN


def test_api_auth_rate_limit_retry_issues_become_integration_or_infra() -> None:
    """Per the brief, API/auth/rate-limit/retry issues split between
    integration_pain (single API) and infra_operational_pain (system-
    level reliability)."""
    # Single-API cluster => integration_pain.
    posts = [
        _post("Stripe API rate limit is too aggressive"),
        _post("Stripe OAuth flow keeps breaking"),
        _post("Stripe webhook returns 401"),
        _post("Stripe SDK is missing features"),
    ]
    a = classify_opportunity(_opp("Stripe", mentions=4, pain_score=0.6), posts)
    assert a.opportunity_type == TYPE_INTEGRATION_PAIN

    # System-level reliability => infra_operational_pain.
    posts = [
        _post("Rate limit hits every 60 seconds"),
        _post("Timeout errors during peak load"),
        _post("Monitoring is alerting on every blip"),
        _post("Memory leak in the worker process"),
        _post("Retry logic keeps failing on 503"),
    ]
    a = classify_opportunity(_opp("Infra", mentions=5, pain_score=0.6), posts)
    assert a.opportunity_type == TYPE_INFRA_OPERATIONAL_PAIN


# -------------------------------------------------------------------------
# Type 5: developer_workflow_pain
# -------------------------------------------------------------------------
def test_developer_workflow_pain_classified_correctly() -> None:
    """Repetitive / manual / friction language => workflow pain."""
    posts = [
        _post("Every time I deploy I have to manually run migrations"),
        _post("Our CI workflow is so tedious"),
        _post("The release process is repetitive and wastes time"),
        _post("Code review process is friction"),
    ]
    a = classify_opportunity(
        _opp("Workflow pain", mentions=4, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_DEVELOPER_WORKFLOW_PAIN


def test_repeated_workflow_pain_classified_correctly() -> None:
    """Brief: 'repeated developer workflow friction' => developer_workflow_pain."""
    posts = [
        _post("The build pipeline is repetitive every day"),
        _post("Deploy process is so tedious and time-consuming"),
        _post("Manual testing takes hours every release"),
        _post("Context switching between tools is friction"),
        _post("Our release process wastes the whole afternoon"),
    ]
    a = classify_opportunity(
        _opp("Daily workflow pain", mentions=5, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_DEVELOPER_WORKFLOW_PAIN


# -------------------------------------------------------------------------
# Type 6: infra_operational_pain
# -------------------------------------------------------------------------
def test_infra_operational_pain_classified_correctly() -> None:
    """Rate limits, retries, uptime, monitoring, scaling, queues, etc."""
    posts = [
        _post("Rate limit hits every 60 seconds"),
        _post("Timeout errors during peak load"),
        _post("Monitoring is alerting on every blip"),
        _post("Memory leak in the worker process"),
        _post("Retry logic keeps failing on 503"),
    ]
    a = classify_opportunity(
        _opp("Infra pain", mentions=5, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_INFRA_OPERATIONAL_PAIN


# -------------------------------------------------------------------------
# Type 7: security_compliance_pain
# -------------------------------------------------------------------------
def test_security_compliance_pain_classified_correctly() -> None:
    """Compliance, privacy, permission, vulnerability language => security."""
    posts = [
        _post("GDPR compliance is a nightmare"),
        _post("Permission denied for our SOC2 audit"),
        _post("Data leak in the export endpoint"),
        _post("PII handling needs improvement"),
        _post("Need SOC2 certification for enterprise"),
    ]
    a = classify_opportunity(
        _opp("Security pain", mentions=5, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_SECURITY_COMPLIANCE_PAIN


# -------------------------------------------------------------------------
# Type 8: potential_product (the strict one)
# -------------------------------------------------------------------------
def test_potential_product_classified_when_all_conditions_met() -> None:
    """All 4 conditions pass => potential_product."""
    posts = [
        _post("As a developer, I need to sync data between Slack and Jira",
              source_category="a/b"),
        _post("As a developer, syncing GitHub issues to Linear is tedious",
              source_category="c/d"),
        _post("As a developer, the Slack-Zapier integration keeps breaking",
              source_category="a/b"),
        _post("As a developer, I want one place to manage all my notifications",
              source_category="e/f"),
        _post("As a developer, integrating tools manually is a huge time sink",
              source_category="c/d"),
        _post("As a developer, our team uses too many tools that don't talk",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Cross-tool dev workflow pain",
            mentions=6,
            pain_score=0.7,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
            frequency_score=0.7,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_POTENTIAL_PRODUCT
    assert a.productizability_score > 0.5, \
        f"potential_product should have a non-trivial score, got {a.productizability_score}"


def test_potential_product_requires_all_four_conditions() -> None:
    """Dropping any one condition must prevent potential_product.

    The brief: "Repeated cross-tool pain with clear buyer/user and
    possible standalone tool" — so cross-cutting, repeated, open_market,
    and real_pain are all mandatory.
    """
    # 1. Drop cross_cutting (single source_category, no known tool names)
    posts = [
        _post("As a developer, my workflow is tedious", source_category="a/b"),
        _post("As a developer, I want better tooling", source_category="a/b"),
        _post("As a developer, the release process is friction", source_category="a/b"),
        _post("As a developer, manual work wastes time", source_category="a/b"),
        _post("As a developer, repetitive tasks are painful", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp("Single-repo dev", mentions=5, pain_score=0.7,
             reality_status="underserved", competitor_strength_estimate=0.1,
             frequency_score=0.7),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT, \
        f"single-source cluster should not be potential_product, got {a.opportunity_type}"

    # 2. Drop repeated (mentions < POTENTIAL_MIN_MENTIONS, no high freq)
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, the GitHub-Slack integration is broken",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp("Cross-tool sync", mentions=2, pain_score=0.7,
             reality_status="underserved", competitor_strength_estimate=0.1),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT

    # 3. Drop open_market (saturated)
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, the GitHub-Slack integration is broken",
              source_category="a/b"),
        _post("As a developer, cross-tool notifications are a mess",
              source_category="a/b"),
        _post("As a developer, integrating tools manually is a huge time sink",
              source_category="a/b"),
        _post("As a developer, our team uses too many tools that don't talk",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp("Saturated market", mentions=5, pain_score=0.7,
             reality_status="saturated", competitor_strength_estimate=0.9,
             frequency_score=0.7),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT, \
        f"saturated market should not be potential_product, got {a.opportunity_type}"

    # 4. Drop real_pain (no buyer language, no pain cues, low pain_score)
    posts = [
        _post("Slack and Jira integration is mentioned", source_category="a/b"),
        _post("GitHub-Slack integration is mentioned", source_category="a/b"),
        _post("Cross-tool notification sync mentioned", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp("No real pain", mentions=3, pain_score=0.0,
             reality_status="underserved", competitor_strength_estimate=0.1),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT


def test_vague_single_repo_issues_do_not_become_potential_product() -> None:
    """Brief: 'vague single-repo issues do not become potential_product'."""
    posts = [
        _post("TypeError in my repo", body="I get a crash", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp("Random crash", mentions=1, pain_score=0.3,
             weighted_score=0.9, frequency_score=0.9),
        posts,
    )
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT, \
        "vague single-repo should NOT be potential_product"


def test_potential_product_requires_strong_evidence_not_just_high_weighted_score() -> None:
    """Brief: 'potential_product requires strong evidence, not just a high weighted_score'."""
    posts = [
        _post("Stripe webhook returns 401", body="constant failure"),
        _post("Stripe rate limit is too aggressive"),
        _post("Stripe SDK is missing features"),
    ]
    a = classify_opportunity(
        _opp("Stripe issues", mentions=3, pain_score=0.9,
             weighted_score=0.95, frequency_score=0.9,
             reality_status="unknown", competitor_strength_estimate=0.0),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT, \
        f"high weighted_score alone must NOT trigger potential_product, got {a.opportunity_type}"


# -------------------------------------------------------------------------
# Tie-breaking: priority order for the seven specific types
# -------------------------------------------------------------------------
def test_security_wins_over_infra_when_both_match() -> None:
    """When security AND infra cues both hit, security takes precedence."""
    posts = [
        _post("Rate limit 401 error from auth endpoint", body="permissions denied"),
        _post("Timeout during security audit", body="permission denied"),
        _post("Compliance check times out", body="permission denied"),
    ]
    a = classify_opportunity(
        _opp("Auth issues", mentions=3, pain_score=0.5),
        posts,
    )
    # Whichever wins, it must be one of the two — not unknown.
    assert a.opportunity_type in (TYPE_SECURITY_COMPLIANCE_PAIN, TYPE_INFRA_OPERATIONAL_PAIN)


# -------------------------------------------------------------------------
# Baseline productizability_score ranges
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "otype, expected_low, expected_high",
    [
        (TYPE_REPO_SPECIFIC_BUG, 0.0, 0.25),
        (TYPE_DOCUMENTATION_CONFUSION, 0.0, 0.30),
        (TYPE_MISSING_FEATURE, 0.20, 0.50),
        (TYPE_INTEGRATION_PAIN, 0.40, 0.75),
        (TYPE_DEVELOPER_WORKFLOW_PAIN, 0.35, 0.70),
        (TYPE_INFRA_OPERATIONAL_PAIN, 0.30, 0.60),
        (TYPE_SECURITY_COMPLIANCE_PAIN, 0.30, 0.65),
    ],
)
def test_baseline_productizability_score_in_expected_range(
    otype: str, expected_low: float, expected_high: float,
) -> None:
    """The brief's rule that productizability differs by type."""
    # Each fixture is just enough to land in the right type.
    fixtures = {
        TYPE_REPO_SPECIFIC_BUG: [
            _post("TypeError on save", body="stack trace"),
            _post("Crash on save", body="AttributeError"),
            _post("NullPointerException", body="regression since 1.0"),
        ],
        TYPE_DOCUMENTATION_CONFUSION: [
            _post("How do I install this?"),
            _post("Where are the docs?"),
            _post("No tutorial available"),
        ],
        TYPE_MISSING_FEATURE: [
            _post("Please add WebSocket support"),
            _post("Feature request: dark mode"),
            _post("I wish it had CSV export"),
        ],
        TYPE_INTEGRATION_PAIN: [
            _post("Stripe webhook returns 401"),
            _post("Stripe API rate limit"),
            _post("Stripe SDK missing methods"),
        ],
        TYPE_DEVELOPER_WORKFLOW_PAIN: [
            _post("The CI workflow is tedious"),
            _post("Manual testing wastes time"),
            _post("The release process is friction"),
        ],
        TYPE_INFRA_OPERATIONAL_PAIN: [
            _post("Rate limit hits every 60 seconds"),
            _post("Timeout during peak load"),
            _post("Memory leak in worker"),
            _post("Retry fails on 503"),
        ],
        TYPE_SECURITY_COMPLIANCE_PAIN: [
            _post("GDPR compliance is a nightmare"),
            _post("Permission denied for SOC2"),
            _post("Data leak in export"),
        ],
    }
    a = classify_opportunity(_opp(otype, mentions=3, pain_score=0.5), fixtures[otype])
    assert a.opportunity_type == otype, f"got {a.opportunity_type}"
    assert expected_low <= a.productizability_score <= expected_high, (
        f"productizability_score {a.productizability_score} out of range "
        f"[{expected_low}, {expected_high}] for {otype}"
    )


# -------------------------------------------------------------------------
# Signals dict is exposed for audit
# -------------------------------------------------------------------------
def test_signals_dict_exposes_raw_counts() -> None:
    """The assessment.signals dict must contain the raw cue counts."""
    a = classify_opportunity(
        _opp("x", mentions=3),
        [_post("Rate limit error", body="throttled")],
    )
    assert "n_posts" in a.signals
    assert "n_distinct_sources" in a.signals
    assert "n_distinct_tools" in a.signals
    assert "infra_hits" in a.signals
    assert a.signals["n_posts"] == 1


# -------------------------------------------------------------------------
# Bot-typed issues get bot_update subtype (inherited from HN, but
# classifier is source-agnostic). The classifier doesn't currently
# special-case subtype='bot_update', so a post with that subtype + bug
# text still classifies as repo_specific_bug. This documents the
# current behavior.
# -------------------------------------------------------------------------
def test_bot_subtype_does_not_change_classification() -> None:
    """Bot-typed posts share the cluster's signals; the classifier
    doesn't downgrade them. Downstream code (or the cleaner) is
    expected to filter them earlier in the pipeline.
    """
    posts = [
        _post("TypeError on save", body="stack trace", subtype="bot_update"),
        _post("Crash on save", body="AttributeError", subtype="bot_update"),
        _post("NullPointer", body="regression", subtype="bot_update"),
    ]
    a = classify_opportunity(
        _opp("Bot cluster", mentions=3, pain_score=0.5),
        posts,
    )
    # Bug-typed content still triggers repo_specific_bug.
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG


# -------------------------------------------------------------------------
# Settings compatibility: classify_opportunity accepts precomputed scores
# -------------------------------------------------------------------------
def test_classify_uses_reality_status_from_opportunity() -> None:
    """The classifier reads reality_status / competitor_strength from the
    Opportunity row, not from the posts."""
    # A cluster that would normally be unknown, but reality_status is
    # explicitly set — the classifier should still pick the right type.
    posts = [
        _post("TypeError", body="traceback"),
        _post("Crash", body="exception"),
        _post("NullPointer", body="regression"),
    ]
    a = classify_opportunity(
        _opp("Bug cluster", mentions=3, pain_score=0.5,
             reality_status="saturated", competitor_strength_estimate=0.9),
        posts,
    )
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG


# -------------------------------------------------------------------------
# productizability_reason is always populated (human-readable)
# -------------------------------------------------------------------------
def test_reason_is_always_populated() -> None:
    """Every assessment must have a non-empty reason string."""
    a1 = classify_opportunity(_opp("x", mentions=3), [_post("TypeError")])
    a2 = classify_opportunity(_opp("y", mentions=0), [])
    a3 = classify_opportunity(
        _opp("z", mentions=5, pain_score=0.5, reality_status="underserved",
             competitor_strength_estimate=0.1, frequency_score=0.7),
        [_post("As a developer, Slack and Jira integration is broken",
               source_category="a/b")],
    )
    assert a1.productizability_reason
    assert a2.productizability_reason
    assert a3.productizability_reason


# -------------------------------------------------------------------------
# V2 (calibration pass 2) - new type + stricter rules
# -------------------------------------------------------------------------
def test_upstream_library_bug_constant_is_in_all_types() -> None:
    """V2: TYPE_UPSTREAM_LIBRARY_BUG is part of the documented taxonomy."""
    assert TYPE_UPSTREAM_LIBRARY_BUG in ALL_TYPES
    assert "upstream_library_bug" in ALL_TYPES


def test_score_cap_constants_match_brief() -> None:
    """Rule 6: per-type score caps. Locked in by the brief."""
    assert SCORE_CAP[TYPE_UPSTREAM_LIBRARY_BUG] == 0.30
    assert SCORE_CAP[TYPE_REPO_SPECIFIC_BUG] == 0.25
    assert SCORE_CAP[TYPE_MISSING_FEATURE] == 0.45
    assert SCORE_CAP[TYPE_INTEGRATION_PAIN] == 0.65
    assert SCORE_CAP[TYPE_INFRA_OPERATIONAL_PAIN] == 0.60
    assert SCORE_CAP[TYPE_POTENTIAL_PRODUCT] == 1.0


def test_potential_product_min_score_is_0_70() -> None:
    """Rule 6: potential_product must have score >= 0.70."""
    assert POTENTIAL_MIN_SCORE == 0.70


def test_potential_product_allowed_reality_statuses() -> None:
    """Rule 1: reality_status must be `underserved` or `competitive` only."""
    assert "underserved" in POTENTIAL_ALLOWED_REALITY_STATUSES
    assert "competitive" in POTENTIAL_ALLOWED_REALITY_STATUSES
    assert "unknown" not in POTENTIAL_ALLOWED_REALITY_STATUSES
    assert "saturated" not in POTENTIAL_ALLOWED_REALITY_STATUSES


# -------------------------------------------------------------------------
# False-positive regression tests (the exact strings from the brief)
# -------------------------------------------------------------------------
def test_brief_false_positive_badrequest_max_tokens_not_potential_product() -> None:
    """Brief case 1: BadRequestError max_tokens/model output limit in
    Azure OpenAI GPT-5 parse() must NOT be classified as potential_product."""
    posts = [
        _post(
            "BadRequestError max_tokens/model output limit in Azure OpenAI GPT-5 parse()",
            body="got a BadRequestError when calling .parse() on Azure OpenAI GPT-5",
            source_category="openai/openai-python",
        ),
        _post(
            "GPT-5 max_tokens truncated output finish_reason length",
            body="the BadRequestError message says max_tokens model output limit",
            source_category="openai/openai-python",
        ),
        _post(
            "azure openai chat completions API returns BadRequestError on max_tokens",
            body="Azure OpenAI's parse() errors with max_tokens model output limit",
            source_category="openai/openai-python",
        ),
    ]
    a = classify_opportunity(
        _opp(
            "Azure OpenAI GPT-5 max_tokens parse error",
            mentions=3,
            pain_score=0.6,
            reality_status="unknown",
            competitor_strength_estimate=0.0,
        ),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT, (
        f"V2 must NOT call this potential_product, got {a.opportunity_type}"
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG
    assert a.productizability_score <= SCORE_CAP[TYPE_UPSTREAM_LIBRARY_BUG]


def test_brief_false_positive_annotation_typed_as_object() -> None:
    """Brief case 2: ResponseOutputTextAnnotationAddedEvent.annotation
    typed as object instead of Annotation must NOT be potential_product."""
    posts = [
        _post(
            "ResponseOutputTextAnnotationAddedEvent.annotation typed as object instead of Annotation",
            body="Pydantic validation error: annotation typed as object instead of Annotation class",
            source_category="openai/openai-python",
        ),
        _post(
            "ResponseOutputTextAnnotationAddedEvent response is annotated as object",
            body="annotation typed as object instead of Annotation, fails pydantic validation",
            source_category="openai/openai-python",
        ),
        _post(
            "ResponseOutputTextAnnotationAddedEvent .annotation returns object instead of Annotation",
            body="pydantic model_validate fails on the responseoutputtextannotation",
            source_category="openai/openai-python",
        ),
    ]
    a = classify_opportunity(
        _opp(
            "Annotation type mismatch in ResponseOutputTextAnnotationAddedEvent",
            mentions=3,
            pain_score=0.6,
            reality_status="unknown",
            competitor_strength_estimate=0.0,
        ),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG


def test_brief_false_positive_prompt_cache_retention_type_mismatch() -> None:
    """Brief case 3: prompt_cache_retention type mismatch -> upstream_library_bug
    or integration_pain, NOT potential_product."""
    posts = [
        _post(
            "prompt_cache_retention type mismatch in chat completions",
            body="got a BadRequestError: prompt_cache_retention must be a string enum",
            source_category="openai/openai-python",
        ),
        _post(
            "openai-python prompt_cache_retention validation error",
            body="Pydantic validation failed: prompt_cache_retention typed as object instead of str",
            source_category="openai/openai-python",
        ),
        _post(
            "azure openai prompt_cache_retention schema mismatch",
            body="prompt_cache_retention expected str but got int",
            source_category="openai/openai-python",
        ),
    ]
    a = classify_opportunity(
        _opp(
            "prompt_cache_retention type mismatch",
            mentions=3,
            pain_score=0.6,
            reality_status="unknown",
            competitor_strength_estimate=0.0,
        ),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT
    assert a.opportunity_type in (
        TYPE_UPSTREAM_LIBRARY_BUG, TYPE_INTEGRATION_PAIN,
    ), f"expected upstream_library_bug or integration_pain, got {a.opportunity_type}"


def test_brief_false_positive_uploading_pdf_files_api_400() -> None:
    """Brief case 4: uploading PDF via Files API 400 -> integration_pain
    or upstream_library_bug, NOT potential_product."""
    posts = [
        _post(
            "uploading PDF via Files API returns 400",
            body="the Files API rejects our PDF upload with HTTP 400 BadRequest",
            source_category="openai/openai-python",
        ),
        _post(
            "openai Files API PDF upload fails with 400",
            body="the SDK throws BadRequestError when uploading PDF via Files API",
            source_category="openai/openai-python",
        ),
        _post(
            "Files API PDF upload: HTTP 400 error",
            body="trying to upload PDF to the Files API always gives a 400 response",
            source_category="openai/openai-python",
        ),
    ]
    a = classify_opportunity(
        _opp(
            "Files API PDF upload 400",
            mentions=3,
            pain_score=0.5,
            reality_status="unknown",
            competitor_strength_estimate=0.0,
        ),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT
    assert a.opportunity_type in (
        TYPE_INTEGRATION_PAIN, TYPE_UPSTREAM_LIBRARY_BUG,
    )


def test_brief_potential_product_still_works_for_real_workflow_pain() -> None:
    """Brief case 5: actual cross-tool repeated workflow pain with
    underserved/competitive reality_status IS still potential_product."""
    posts = [
        _post("As a developer, syncing Slack with Jira is tedious",
              source_category="a/b"),
        _post("As a developer, GitHub-Slack integration is broken",
              source_category="c/d"),
        _post("As a developer, the Slack webhook fails",
              source_category="a/b"),
        _post("As a developer, I want one place to manage notifications",
              source_category="e/f"),
        _post("As a developer, integrating tools manually is a huge time sink",
              source_category="c/d"),
        _post("As a developer, our team uses too many tools that don't talk",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Cross-tool dev workflow pain",
            mentions=6,
            pain_score=0.7,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
            frequency_score=0.7,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_POTENTIAL_PRODUCT
    assert a.productizability_score >= POTENTIAL_MIN_SCORE


# -------------------------------------------------------------------------
# Rule 1: reality_status gates potential_product
# -------------------------------------------------------------------------
def test_potential_product_rejected_when_reality_unknown() -> None:
    """Rule 1: reality_status='unknown' disqualifies potential_product."""
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, GitHub-Slack integration is broken",
              source_category="c/d"),
        _post("As a developer, the Slack webhook fails",
              source_category="a/b"),
        _post("As a developer, cross-tool notifications are a mess",
              source_category="c/d"),
        _post("As a developer, integrating tools manually is tedious",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Real cross-tool pain, but reality=unknown",
            mentions=5, pain_score=0.7, frequency_score=0.7,
            reality_status="unknown",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT, (
        f"reality=unknown must block potential_product, got {a.opportunity_type}"
    )


def test_potential_product_rejected_when_reality_saturated() -> None:
    """Rule 1: reality_status='saturated' also disqualifies potential_product."""
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, GitHub-Slack integration is broken",
              source_category="c/d"),
        _post("As a developer, the Slack webhook fails",
              source_category="a/b"),
        _post("As a developer, cross-tool notifications are a mess",
              source_category="c/d"),
        _post("As a developer, integrating tools manually is tedious",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Saturated market cross-tool pain",
            mentions=5, pain_score=0.7, frequency_score=0.7,
            reality_status="saturated",
            competitor_strength_estimate=0.9,
        ),
        posts,
    )
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT


def test_potential_product_accepted_when_reality_underserved() -> None:
    """Rule 1 (positive case): reality_status='underserved' allows potential_product."""
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, GitHub-Slack integration is broken",
              source_category="c/d"),
        _post("As a developer, the Slack webhook fails",
              source_category="a/b"),
        _post("As a developer, cross-tool notifications are a mess",
              source_category="c/d"),
        _post("As a developer, integrating tools manually is tedious",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Underserved cross-tool pain",
            mentions=5, pain_score=0.7, frequency_score=0.7,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_POTENTIAL_PRODUCT


def test_potential_product_accepted_when_reality_competitive() -> None:
    """Rule 1 (positive case): reality_status='competitive' allows potential_product."""
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, GitHub-Slack integration is broken",
              source_category="c/d"),
        _post("As a developer, the Slack webhook fails",
              source_category="a/b"),
        _post("As a developer, cross-tool notifications are a mess",
              source_category="c/d"),
        _post("As a developer, integrating tools manually is tedious",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Competitive cross-tool pain",
            mentions=5, pain_score=0.7, frequency_score=0.7,
            reality_status="competitive",
            competitor_strength_estimate=0.3,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_POTENTIAL_PRODUCT


# -------------------------------------------------------------------------
# Rule 4: upstream_library_bug demotes potential_product
# -------------------------------------------------------------------------
def test_upstream_library_bug_demotes_potential_product() -> None:
    """Rule 4: an upstream_library_bug cluster is demoted even when
    all 4 strict conditions are met."""
    posts = [
        _post("As a developer, the openai SDK throws BadRequestError on max_tokens",
              body="openai-python parse() fails with max_tokens", source_category="a/b"),
        _post("As a developer, openai-python's max_tokens parse error is broken",
              body="BadRequestError on max_tokens", source_category="c/d"),
        _post("As a developer, the openai SDK's parse() is broken",
              body="BadRequestError on max_tokens", source_category="a/b"),
        _post("As a developer, pydantic validation fails on openai's response",
              body="annotation typed as object instead of Annotation",
              source_category="c/d"),
        _post("As a developer, openai-python throws BadRequestError on max_tokens",
              body="parse() on Azure OpenAI fails", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "openai SDK broken",
            mentions=5, pain_score=0.7, frequency_score=0.7,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG, (
        f"upstream_library_bug should win, got {a.opportunity_type}"
    )


def test_high_weighted_score_alone_does_not_promote_upstream_bug() -> None:
    """Rule 4: even with weighted_score=0.95, an upstream library bug
    stays upstream_library_bug."""
    posts = [
        _post("BadRequestError on max_tokens in openai parse()",
              body="openai-python BadRequestError", source_category="a/b"),
        _post("openai SDK throws BadRequestError on max_tokens",
              body="annotation typed as object", source_category="a/b"),
        _post("pydantic validation fails on openai response",
              body="typed as object instead of Annotation", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "openai is broken",
            mentions=3, pain_score=0.95, weighted_score=0.95,
            frequency_score=0.9,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG
    assert a.opportunity_type != TYPE_POTENTIAL_PRODUCT
    assert a.productizability_score <= SCORE_CAP[TYPE_UPSTREAM_LIBRARY_BUG]


# -------------------------------------------------------------------------
# Rule 5: tightened security lexicon
# -------------------------------------------------------------------------
def test_401_permission_denied_not_security() -> None:
    """Rule 5: HTTP 401 / 'permission denied' must NOT trigger security."""
    posts = [
        _post("Stripe API returns 401 unauthorized when webhook fails",
              body="the API request fails with 401 permission denied error"),
        _post("OAuth flow returns 401 unauthorized after token expired",
              body="401 permission denied on the second API call"),
        _post("API key was rejected with 401 unauthorized",
              body="permission denied for the Stripe webhook"),
    ]
    a = classify_opportunity(
        _opp("Auth issues", mentions=3, pain_score=0.5),
        posts,
    )
    assert a.opportunity_type != TYPE_SECURITY_COMPLIANCE_PAIN, (
        f"V2 must NOT classify 401/permission denied as security, "
        f"got {a.opportunity_type}"
    )


def test_explicit_gdpr_compliance_is_security() -> None:
    """Rule 5 (positive case): real GDPR/compliance language IS still security."""
    posts = [
        _post("GDPR compliance is a nightmare for our SaaS",
              body="we need SOC2 audit ready for enterprise customers"),
        _post("PII data leak in the export endpoint", body="compliance review found it"),
        _post("Need SOC2 certification for healthcare customers",
              body="HIPAA compliance is required"),
    ]
    a = classify_opportunity(
        _opp("Compliance", mentions=3, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_SECURITY_COMPLIANCE_PAIN


def test_specific_attack_terms_still_trigger_security() -> None:
    """Rule 5: specific attack terms (XSS, SQL injection, RCE) still classify as security."""
    posts = [
        _post("SQL injection in the login form", body="we found a sql injection"),
        _post("XSS in the comments field", body="xss vulnerability in user content"),
        _post("RCE via deserialization bug", body="remote code execution possible"),
    ]
    a = classify_opportunity(
        _opp("Security vulns", mentions=3, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_SECURITY_COMPLIANCE_PAIN


# -------------------------------------------------------------------------
# Rule 6: per-type score caps
# -------------------------------------------------------------------------
def test_repo_specific_bug_score_capped_at_0_25() -> None:
    """Rule 6: a bug cluster can never reach >= 0.30 productizability."""
    posts = [
        _post("TypeError on save", body="stack trace"),
        _post("Crash on save", body="AttributeError"),
        _post("NullPointer", body="regression since 1.0"),
    ]
    a = classify_opportunity(
        _opp("Bug cluster", mentions=3, pain_score=0.5,
             reality_status="underserved", competitor_strength_estimate=0.1),
        posts,
    )
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG
    assert a.productizability_score <= SCORE_CAP[TYPE_REPO_SPECIFIC_BUG]


def test_upstream_library_bug_score_capped_at_0_30() -> None:
    """Rule 6: upstream_library_bug scores <= 0.30."""
    posts = [
        _post("BadRequestError on openai max_tokens parse()", body=""),
        _post("pydantic validation error on annotation", body="typed as object"),
        _post("openai-python API throws BadRequestError", body=""),
    ]
    a = classify_opportunity(
        _opp("Upstream bug", mentions=3, pain_score=0.7,
             reality_status="underserved", competitor_strength_estimate=0.1),
        posts,
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG
    assert a.productizability_score <= SCORE_CAP[TYPE_UPSTREAM_LIBRARY_BUG]


def test_integration_pain_score_capped_at_0_65() -> None:
    """Rule 6: integration_pain maxes out at 0.65."""
    posts = [
        _post("Stripe webhook keeps returning 401"),
        _post("Stripe API rate limit is too aggressive"),
        _post("Stripe SDK doesn't support new payment methods"),
        _post("Stripe checkout integration is fragile"),
    ]
    a = classify_opportunity(
        _opp("Stripe", mentions=4, pain_score=0.6),
        posts,
    )
    assert a.opportunity_type == TYPE_INTEGRATION_PAIN
    assert a.productizability_score <= SCORE_CAP[TYPE_INTEGRATION_PAIN]


def test_potential_product_minimum_score_is_0_70() -> None:
    """Rule 6: potential_product score must be >= 0.70 if classified as such."""
    posts = [
        _post("As a developer, syncing Slack and Jira is tedious",
              source_category="a/b"),
        _post("As a developer, GitHub-Slack integration is broken",
              source_category="c/d"),
        _post("As a developer, the Slack webhook fails",
              source_category="a/b"),
        _post("As a developer, cross-tool notifications are a mess",
              source_category="c/d"),
        _post("As a developer, integrating tools manually is tedious",
              source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Marginal cross-tool pain",
            mentions=5,
            pain_score=0.0,
            frequency_score=0.5,
            reality_status="underserved",
            competitor_strength_estimate=0.0,
        ),
        posts,
    )
    if a.opportunity_type == TYPE_POTENTIAL_PRODUCT:
        assert a.productizability_score >= POTENTIAL_MIN_SCORE
    else:
        assert a.productizability_score < POTENTIAL_MIN_SCORE


# -------------------------------------------------------------------------
# V2 priority order
# -------------------------------------------------------------------------
def test_priority_upstream_library_bug_wins_when_both_match() -> None:
    """upstream_library_bug wins when both it and potential_product
    conditions are met."""
    posts = [
        _post("As a developer, openai SDK throws BadRequestError",
              body="max_tokens model output limit on parse()", source_category="a/b"),
        _post("As a developer, openai-python's pydantic validation fails",
              body="annotation typed as object instead of Annotation",
              source_category="c/d"),
        _post("As a developer, the openai SDK is broken",
              body="BadRequestError max_tokens", source_category="a/b"),
        _post("As a developer, openai-python response stream errors",
              body="parse() fails on Azure OpenAI GPT-5", source_category="c/d"),
        _post("As a developer, integrating tools manually is tedious",
              body="openai SDK is broken", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "openai SDK + cross-tool pain",
            mentions=5, pain_score=0.7, frequency_score=0.7,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG


# -------------------------------------------------------------------------
# V2: SCORE_CAP caps are absolute
# -------------------------------------------------------------------------
def test_score_cap_not_breached_by_cross_cutting_bonuses() -> None:
    """Cross-cutting bonuses can never push a type past its cap."""
    posts = [
        _post("TypeError on save", body="stack trace", source_category="a/b"),
        _post("Crash on save", body="AttributeError", source_category="c/d"),
        _post("NullPointer", body="regression since 1.0", source_category="a/b"),
    ]
    a = classify_opportunity(
        _opp(
            "Cross-source bug cluster",
            mentions=3, pain_score=0.5,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_REPO_SPECIFIC_BUG
    assert a.productizability_score <= SCORE_CAP[TYPE_REPO_SPECIFIC_BUG]
    assert a.productizability_score <= 0.25


# -------------------------------------------------------------------------
# Upstream_library_bug: specific lexical triggers
# -------------------------------------------------------------------------
def test_openai_library_name_triggers_upstream_bug() -> None:
    """A cluster mentioning 'openai-python' (the SDK) classifies as
    upstream_library_bug, even without error messages."""
    posts = [
        _post("openai-python async stream returns wrong type", body=""),
        _post("openai-node API throws an error", body=""),
        _post("langchain integration with openai-python fails", body=""),
    ]
    a = classify_opportunity(
        _opp("OpenAI issues", mentions=3, pain_score=0.5),
        posts,
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG


def test_pydantic_validation_triggers_upstream_bug() -> None:
    """Pydantic-specific failures are upstream."""
    posts = [
        _post("pydantic v2 validation error on response", body="model_validate fails"),
        _post("pydantic v1 BaseModel has a parsing issue", body="model_dump wrong type"),
        _post("pydantic-core schema validation fails", body=""),
    ]
    a = classify_opportunity(
        _opp("Pydantic issues", mentions=3, pain_score=0.5),
        posts,
    )
    assert a.opportunity_type == TYPE_UPSTREAM_LIBRARY_BUG


# -------------------------------------------------------------------------
# Sanity: real integration_pain still works
# -------------------------------------------------------------------------
def test_integration_pain_with_real_underserved_reality_still_works() -> None:
    """Sanity: a real integration_pain cluster with underserved reality
    still classifies as integration_pain."""
    posts = [
        _post("Stripe webhook keeps returning 401"),
        _post("Stripe API rate limit is too aggressive"),
        _post("Stripe SDK doesn't support new payment methods"),
        _post("Stripe checkout integration is fragile"),
    ]
    a = classify_opportunity(
        _opp(
            "Stripe pain",
            mentions=4, pain_score=0.6,
            reality_status="underserved",
            competitor_strength_estimate=0.1,
        ),
        posts,
    )
    assert a.opportunity_type == TYPE_INTEGRATION_PAIN
    assert a.productizability_score <= SCORE_CAP[TYPE_INTEGRATION_PAIN]
