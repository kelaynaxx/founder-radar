"""Tests for the LLM-assisted opportunity review layer.

Brief required:
  - internal repo status report -> reject
  - upstream API/library bug -> reject
  - CI/CD chore -> reject
  - vague feature request -> reject or maybe
  - repeated workflow pain with clear buyer -> strong_candidate
  - LLM invalid JSON -> reject safely

We use a `FakeLLMProvider` that returns canned responses per scenario
(plus a `BoomLLM` that raises, an `InvalidJSONLLM` that returns
non-JSON, and a `MarkdownFenceLLM` that wraps valid JSON in fences).
The classifier's behaviour is deterministic; the LLM is the only
non-deterministic part, and the FakeLLMProvider removes that
variance from the test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

import pytest

from founder_radar.analysis.opportunity_review import (
    ALL_REVIEW_REASONS,
    ALL_REVIEW_VERDICTS,
    REVIEW_FAILED_TAG,
    REVIEW_REASON_DOCUMENTATION_ONLY,
    REVIEW_REASON_MAINTENANCE_CHORE,
    REVIEW_REASON_NOT_BUYER_PAIN,
    REVIEW_REASON_POSSIBLE_DEVTOOL,
    REVIEW_REASON_POSSIBLE_INFRA_TOOL,
    REVIEW_REASON_POSSIBLE_MICRO_SAAS,
    REVIEW_REASON_REPO_INTERNAL_TASK,
    REVIEW_REASON_STRONG_REPEATED_PAIN,
    REVIEW_REASON_TOO_REPO_SPECIFIC,
    REVIEW_REASON_TOO_VAGUE,
    REVIEW_REASON_UPSTREAM_BUG,
    REVIEW_VERDICT_MAYBE,
    REVIEW_VERDICT_REJECT,
    REVIEW_VERDICT_STRONG_CANDIDATE,
    ReviewVerdict,
    review_opportunity,
    review_opportunities_batch,
)
from founder_radar.database.models import Opportunity, Post
from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse


# -------------------------------------------------------------------------
# Test helpers
# -------------------------------------------------------------------------
def _post(
    title: str,
    body: str = "",
    *,
    source: str = "github",
    source_category: str = "a/b",
    subtype: str | None = None,
) -> Post:
    return Post(
        source=source,
        external_id=title,
        source_category=source_category,
        title=title,
        body=body,
        author="x",
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
        pain_score=0.5,
        opportunity_type="potential_product",
    )
    defaults.update(kw)
    return Opportunity(title=title, **defaults)


def _json_response(payload: dict) -> str:
    return json.dumps(payload)


class FakeLLMProvider(BaseLLMProvider):
    """Returns a canned JSON response on every `complete()` call.

    `response_factory` lets each test customize the response (e.g. to
    inspect the prompt before responding). The default returns a
    `reject` verdict with `repo_internal_task` reason — the "no LLM
    conviction, default-reject" baseline.
    """

    def __init__(self, response_factory=None) -> None:  # type: ignore[no-untyped-def]
        self._factory = response_factory or self._default_response
        self.calls: list[Sequence[LLMMessage]] = []

    @property
    def name(self) -> str:
        return "fake-review"

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> LLMResponse:
        self.calls.append(messages)
        content = self._factory(messages)
        return LLMResponse(content=content, model="fake-review-model")

    def _default_response(self, messages) -> str:  # type: ignore[no-untyped-def]
        return _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [REVIEW_REASON_REPO_INTERNAL_TASK],
            "summary": "default fake: not a product opportunity",
            "confidence": 0.4,
        })


class InvalidJSONLLM(FakeLLMProvider):
    """Returns prose (no JSON, no fences). The review layer must
    return `reject` with `review_failed` and not crash."""

    def __init__(self) -> None:
        super().__init__(
            response_factory=lambda m: (
                "I think this is a maintainer chore but I'm not "
                "going to give you a structured answer. The cluster "
                "looks like a single internal ticket."
            )
        )


class MarkdownFenceLLM(FakeLLMProvider):
    """Returns valid JSON wrapped in markdown fences. The parser
    must strip the fences and parse the contents."""

    def __init__(self, payload: dict) -> None:
        super().__init__(
            response_factory=lambda m: (
                "```json\n" + json.dumps(payload, indent=2) + "\n```"
            )
        )


class BoomLLM(FakeLLMProvider):
    """Raises on every call. The review layer must catch the
    exception and return `reject` with `review_failed`."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> LLMResponse:
        raise RuntimeError("simulated LLM provider outage")


# -------------------------------------------------------------------------
# Public API surface
# -------------------------------------------------------------------------
def test_all_verdicts_match_brief() -> None:
    """Verdict taxonomy is the brief's three labels."""
    assert set(ALL_REVIEW_VERDICTS) == {
        REVIEW_VERDICT_REJECT,
        REVIEW_VERDICT_MAYBE,
        REVIEW_VERDICT_STRONG_CANDIDATE,
    }


def test_all_reasons_match_brief() -> None:
    """Reason taxonomy matches the brief's 11 labels (plus `review_failed`
    which is our synthetic tag)."""
    assert set(ALL_REVIEW_REASONS) == {
        REVIEW_REASON_REPO_INTERNAL_TASK,
        REVIEW_REASON_UPSTREAM_BUG,
        REVIEW_REASON_MAINTENANCE_CHORE,
        REVIEW_REASON_DOCUMENTATION_ONLY,
        REVIEW_REASON_NOT_BUYER_PAIN,
        REVIEW_REASON_TOO_VAGUE,
        REVIEW_REASON_TOO_REPO_SPECIFIC,
        REVIEW_REASON_POSSIBLE_DEVTOOL,
        REVIEW_REASON_POSSIBLE_MICRO_SAAS,
        REVIEW_REASON_POSSIBLE_INFRA_TOOL,
        REVIEW_REASON_STRONG_REPEATED_PAIN,
    }
    assert len(ALL_REVIEW_REASONS) == 11


def test_review_verdict_default_is_reject() -> None:
    """A default `ReviewVerdict()` is `reject` with empty reasons.

    The brief: 'Default to reject.'
    """
    v = ReviewVerdict()
    assert v.verdict == REVIEW_VERDICT_REJECT
    assert v.reasons == []
    assert v.confidence == 0.0
    assert not v.is_acceptable


def test_review_verdict_is_acceptable_for_maybe_and_strong() -> None:
    """`is_acceptable` is True for the two non-reject verdicts."""
    assert ReviewVerdict(verdict=REVIEW_VERDICT_MAYBE).is_acceptable
    assert ReviewVerdict(verdict=REVIEW_VERDICT_STRONG_CANDIDATE).is_acceptable
    assert not ReviewVerdict(verdict=REVIEW_VERDICT_REJECT).is_acceptable


def test_review_opportunity_returns_dataclass() -> None:
    """The function returns a ReviewVerdict dataclass."""
    posts = [_post("TypeError", body="stack trace")]
    a = review_opportunity(_opp("x"), posts, FakeLLMProvider())
    assert isinstance(a, ReviewVerdict)
    assert a.verdict in ALL_REVIEW_VERDICTS


# -------------------------------------------------------------------------
# Brief case 1: internal repo status report -> reject
# -------------------------------------------------------------------------
def test_internal_repo_status_report_is_rejected() -> None:
    """A status report about a CI run or internal ticket must
    be rejected as `repo_internal_task` / `maintenance_chore`."""
    posts = [
        _post(
            "[Bot] CI build #4521 failed: timeout in test_main.py",
            body="The CI job timed out after 30 minutes. This is a "
                 "transient failure. Re-running the pipeline.",
        ),
        _post(
            "[Bot] Daily test report: 1247 passed, 3 flaky",
            body="Generated by the CI bot. See attached logs.",
        ),
        _post(
            "Update CONTRIBUTING.md with new test instructions",
            body="Internal docs update for the team.",
        ),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [
                REVIEW_REASON_REPO_INTERNAL_TASK,
                REVIEW_REASON_MAINTENANCE_CHORE,
            ],
            "summary": (
                "This is an internal CI status report and a docs "
                "update. Posts are bot-generated or internal team "
                "tickets. No external user pain."
            ),
            "confidence": 0.9,
        })
    )
    a = review_opportunity(_opp("CI failures", mentions=3, opportunity_type="repo_specific_bug"), posts, llm)
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_REPO_INTERNAL_TASK in a.reasons or \
           REVIEW_REASON_MAINTENANCE_CHORE in a.reasons


# -------------------------------------------------------------------------
# Brief case 2: upstream API/library bug -> reject
# -------------------------------------------------------------------------
def test_upstream_api_library_bug_is_rejected() -> None:
    """An upstream library bug cluster must be rejected as
    `upstream_bug`, NOT promoted to `strong_candidate`."""
    posts = [
        _post(
            "BadRequestError max_tokens/model output limit in Azure OpenAI GPT-5 parse()",
            body="openai-python throws BadRequestError on max_tokens",
        ),
        _post(
            "ResponseOutputTextAnnotationAddedEvent.annotation typed as object instead of Annotation",
            body="Pydantic validation error on openai response",
        ),
        _post(
            "pydantic v2 validation error on openai response",
            body="openai-python's pydantic model_validate fails",
        ),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [REVIEW_REASON_UPSTREAM_BUG],
            "summary": (
                "All posts describe failures in the openai SDK. "
                "Fix belongs upstream in openai-python / pydantic, "
                "not a new product."
            ),
            "confidence": 0.95,
        })
    )
    a = review_opportunity(_opp("openai broken", mentions=3, opportunity_type="upstream_library_bug"), posts, llm)
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_UPSTREAM_BUG in a.reasons


# -------------------------------------------------------------------------
# Brief case 3: CI/CD chore -> reject
# -------------------------------------------------------------------------
def test_cicd_chore_is_rejected() -> None:
    """A CI/CD chore (e.g. dependency bump, version pin) must be
    rejected as `maintenance_chore`."""
    posts = [
        _post(
            "Bump dependency version in requirements.txt",
            body="Routine maintenance. Updating to the latest patch.",
        ),
        _post(
            "Pin transitive dep to fix CI build",
            body="The build started failing after a transitive dep update.",
        ),
        _post(
            "Update GitHub Actions workflow to use Node 20",
            body="Routine upgrade of CI config.",
        ),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [REVIEW_REASON_MAINTENANCE_CHORE],
            "summary": (
                "All posts are CI / dependency maintenance tasks. "
                "No external user pain. Routine chores."
            ),
            "confidence": 0.95,
        })
    )
    a = review_opportunity(_opp("chores", mentions=3, opportunity_type="maintenance_chore"), posts, llm)
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_MAINTENANCE_CHORE in a.reasons


# -------------------------------------------------------------------------
# Brief case 4: vague feature request -> reject or maybe
# -------------------------------------------------------------------------
def test_vague_feature_request_rejected_or_maybe() -> None:
    """A vague feature request without a clear buyer can be
    rejected or `maybe`, but NEVER `strong_candidate`."""
    posts = [
        _post(
            "Please add a new feature to the platform",
            body="Would be nice to have. No concrete use case.",
        ),
        _post(
            "I wish it could do more things",
            body="The platform could be more powerful. Not specific.",
        ),
        _post(
            "Feature request: add a thing",
            body="Just an idea. We need this. Or maybe not.",
        ),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [REVIEW_REASON_TOO_VAGUE, REVIEW_REASON_NOT_BUYER_PAIN],
            "summary": (
                "Posts are vague feature requests without a clear "
                "buyer persona. No concrete use case. Not a "
                "standalone product opportunity."
            ),
            "confidence": 0.85,
        })
    )
    a = review_opportunity(_opp("vague", mentions=3, opportunity_type="missing_feature"), posts, llm)
    assert a.verdict in (REVIEW_VERDICT_REJECT, REVIEW_VERDICT_MAYBE)
    assert a.verdict != REVIEW_VERDICT_STRONG_CANDIDATE


def test_vague_feature_request_can_be_maybe_with_concrete_buyer() -> None:
    """A feature request with a clearer buyer can land at `maybe`."""
    posts = [
        _post(
            "Please add SSO support for our enterprise plan",
            body="Our security team requires SSO. We're a 200-person SaaS.",
        ),
        _post(
            "Feature request: SAML SSO",
            body="As a startup, we need SSO for our enterprise customers.",
        ),
        _post(
            "We need SCIM provisioning for SSO",
            body="Manual user provisioning is killing us.",
        ),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_MAYBE,
            "reasons": [REVIEW_REASON_POSSIBLE_DEVTOOL],
            "summary": (
                "Repeated SSO requests from startups. Could be a "
                "lightweight SSO tool, but there's no clear willingness "
                "to pay yet."
            ),
            "confidence": 0.55,
        })
    )
    a = review_opportunity(_opp("SSO", mentions=3, opportunity_type="missing_feature"), posts, llm)
    assert a.verdict == REVIEW_VERDICT_MAYBE
    assert REVIEW_REASON_POSSIBLE_DEVTOOL in a.reasons


# -------------------------------------------------------------------------
# Brief case 5: repeated workflow pain with clear buyer -> strong_candidate
# -------------------------------------------------------------------------
def test_repeated_workflow_pain_with_buyer_is_strong_candidate() -> None:
    """Repeated cross-tool developer workflow pain with a clear
    buyer and a plausible standalone tool is a `strong_candidate`."""
    posts = [
        _post(
            "As a developer, syncing Slack with Jira is tedious and manual",
            body="Every morning I copy issues from Slack into Jira by hand.",
            source_category="a/b",
        ),
        _post(
            "As a developer, GitHub-Slack integration is broken",
            body="Notifications don't propagate. We use both daily.",
            source_category="c/d",
        ),
        _post(
            "As a developer, the Slack webhook keeps failing",
            body="We have to manually re-trigger the integration weekly.",
            source_category="a/b",
        ),
        _post(
            "As a developer, I want one place to manage notifications",
            body="Too many tools, none of them talk to each other.",
            source_category="e/f",
        ),
        _post(
            "As a developer, integrating tools manually is a huge time sink",
            body="We spend ~3 hours/week on cross-tool glue work.",
            source_category="c/d",
        ),
        _post(
            "As a developer, our team uses too many tools that don't talk",
            body="Slack + Jira + GitHub + Linear. None of them sync.",
            source_category="a/b",
        ),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_STRONG_CANDIDATE,
            "reasons": [REVIEW_REASON_STRONG_REPEATED_PAIN, REVIEW_REASON_POSSIBLE_INFRA_TOOL],
            "summary": (
                "Six posts from 'as a developer' across multiple "
                "sources. Repeated manual sync between Slack, Jira, "
                "and GitHub. Clear buyer (developer teams using many "
                "tools). A standalone sync tool is plausible."
            ),
            "confidence": 0.85,
        })
    )
    a = review_opportunity(
        _opp(
            "Cross-tool dev workflow pain",
            mentions=6,
            opportunity_type="potential_product",
        ),
        posts,
        llm,
    )
    assert a.verdict == REVIEW_VERDICT_STRONG_CANDIDATE
    assert REVIEW_REASON_STRONG_REPEATED_PAIN in a.reasons
    assert a.is_acceptable


# -------------------------------------------------------------------------
# Brief case 6: LLM invalid JSON -> reject safely
# -------------------------------------------------------------------------
def test_invalid_json_response_is_reject_safely() -> None:
    """A non-JSON LLM response must NOT crash the CLI. It must
    return `reject` with `review_failed`."""
    posts = [_post("x", body="y")]
    a = review_opportunity(_opp("x"), posts, InvalidJSONLLM())
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert a.reasons == [REVIEW_FAILED_TAG]
    assert "parseable" in a.summary.lower() or "json" in a.summary.lower() or "failed" in a.summary.lower()


def test_markdown_fence_response_is_parsed() -> None:
    """Valid JSON wrapped in ```json fences is parsed correctly."""
    posts = [_post("x", body="y")]
    payload = {
        "verdict": REVIEW_VERDICT_REJECT,
        "reasons": [REVIEW_REASON_TOO_VAGUE],
        "summary": "Fenced JSON parses correctly.",
        "confidence": 0.7,
    }
    a = review_opportunity(_opp("x"), posts, MarkdownFenceLLM(payload))
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_TOO_VAGUE in a.reasons


def test_boom_llm_is_reject_safely() -> None:
    """An LLM that raises an exception must NOT crash. The result
    is `reject` with `review_failed`."""
    posts = [_post("x", body="y")]
    a = review_opportunity(_opp("x"), posts, BoomLLM())
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert a.reasons == [REVIEW_FAILED_TAG]


# -------------------------------------------------------------------------
# Safety nets
# -------------------------------------------------------------------------
def test_strong_candidate_for_non_potential_product_is_demoted() -> None:
    """If the LLM says `strong_candidate` for a non-`potential_product`
    cluster, the safety net demotes to `maybe`. The brief:
    'The classifier (deterministic) is the source of truth.'"""
    posts = [
        _post("TypeError in my repo", body="stack trace"),
        _post("Crash on save", body="AttributeError"),
    ]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_STRONG_CANDIDATE,
            "reasons": [REVIEW_REASON_POSSIBLE_DEVTOOL],
            "summary": "Bug cluster is actually a product, trust me.",
            "confidence": 0.9,
        })
    )
    # We need to inject the opportunity_type into the parsed data;
    # the safety net reads it from data.get("_opportunity_type") or
    # the opportunity directly. The review layer reads from the
    # opportunity row, so we set the type on the opportunity.
    a = review_opportunity(
        _opp("x", opportunity_type="repo_specific_bug"),
        posts, llm,
    )
    assert a.verdict == REVIEW_VERDICT_MAYBE
    assert a.verdict != REVIEW_VERDICT_STRONG_CANDIDATE


def test_maybe_with_only_reject_reasons_demotes() -> None:
    """If the LLM says `maybe` with all-reject-class reasons, the
    safety net demotes to `reject`."""
    posts = [_post("x")]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_MAYBE,
            "reasons": [REVIEW_REASON_REPO_INTERNAL_TASK],
            "summary": "Internal ticket but maybe?",
            "confidence": 0.5,
        })
    )
    a = review_opportunity(_opp("x", opportunity_type="potential_product"), posts, llm)
    assert a.verdict == REVIEW_VERDICT_REJECT


def test_unknown_verdict_coerces_to_reject() -> None:
    """An LLM that returns a verdict not in the canonical set
    is coerced to `reject` (the safe default)."""
    posts = [_post("x")]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": "great_product",  # not in ALL_REVIEW_VERDICTS
            "reasons": ["foo"],
            "summary": "ignored",
            "confidence": 0.9,
        })
    )
    a = review_opportunity(_opp("x"), posts, llm)
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert a.reasons == []  # 'foo' is not a canonical reason; dropped


def test_unknown_reason_tag_is_dropped() -> None:
    """Reason tags outside the canonical set are dropped silently."""
    posts = [_post("x")]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [
                REVIEW_REASON_REPO_INTERNAL_TASK,
                "hallucinated_tag_xyz",  # not canonical
                REVIEW_REASON_NOT_BUYER_PAIN,
            ],
            "summary": "x",
            "confidence": 0.5,
        })
    )
    a = review_opportunity(_opp("x"), posts, llm)
    # hallucinated_tag_xyz is dropped; the other two remain.
    assert "hallucinated_tag_xyz" not in a.reasons
    assert REVIEW_REASON_REPO_INTERNAL_TASK in a.reasons
    assert REVIEW_REASON_NOT_BUYER_PAIN in a.reasons


def test_reasons_capped_at_three() -> None:
    """Even if the LLM returns 5 reasons, we keep at most 3."""
    posts = [_post("x")]
    llm = FakeLLMProvider(
        response_factory=lambda m: _json_response({
            "verdict": REVIEW_VERDICT_REJECT,
            "reasons": [
                REVIEW_REASON_REPO_INTERNAL_TASK,
                REVIEW_REASON_UPSTREAM_BUG,
                REVIEW_REASON_MAINTENANCE_CHORE,
                REVIEW_REASON_DOCUMENTATION_ONLY,
                REVIEW_REASON_NOT_BUYER_PAIN,
            ],
            "summary": "x",
            "confidence": 0.5,
        })
    )
    a = review_opportunity(_opp("x"), posts, llm)
    assert len(a.reasons) <= 3


# -------------------------------------------------------------------------
# Default behavior
# -------------------------------------------------------------------------
def test_default_llm_response_is_reject() -> None:
    """With the default FakeLLMProvider (no factory), the review
    always says `reject`. The brief: 'Default to reject.'"""
    posts = [_post("x")]
    a = review_opportunity(_opp("x", opportunity_type="potential_product"), posts, FakeLLMProvider())
    assert a.verdict == REVIEW_VERDICT_REJECT


# -------------------------------------------------------------------------
# Batch review
# -------------------------------------------------------------------------
def test_review_opportunities_batch_runs_all() -> None:
    """`review_opportunities_batch` returns one verdict per opportunity."""
    opps = [_opp("a", id=1), _opp("b", id=2), _opp("c", id=3)]
    # Pretend they have ids by setting them after creation.
    for i, opp in enumerate(opps, start=1):
        opp.id = i
    posts_by_id = {1: [_post("x")], 2: [_post("y")], 3: [_post("z")]}
    results = review_opportunities_batch(
        opps, posts_by_id, FakeLLMProvider(), progress=False,
    )
    assert set(results.keys()) == {1, 2, 3}
    for v in results.values():
        assert v.verdict == REVIEW_VERDICT_REJECT


# -------------------------------------------------------------------------
# V2.2: regression for the `self is not defined` repair-callback bug
# -------------------------------------------------------------------------
class _TwoCallLLM(FakeLLMProvider):
    """Returns garbage on the first call (so the parser triggers the
    repair path), then valid JSON on the second call.

    The repair callback MUST be able to call back into the LLM to
    recover. The V2.2 bug was that the callback referenced `self`
    from inside a module-level function, raising NameError.
    """

    def __init__(self) -> None:
        super().__init__()
        self._queued = [
            "this is not parseable JSON at all, sorry",
            json.dumps({
                "verdict": REVIEW_VERDICT_REJECT,
                "reasons": [REVIEW_REASON_NOT_BUYER_PAIN],
                "summary": "Recovered by repair path.",
                "confidence": 0.6,
            }),
        ]

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        content = (
            self._queued.pop(0)
            if self._queued
            else json.dumps({
                "verdict": REVIEW_VERDICT_REJECT,
                "reasons": [REVIEW_REASON_NOT_BUYER_PAIN],
                "summary": "default after repair queue empty",
                "confidence": 0.4,
            })
        )
        return LLMResponse(content=content, model="two-call")


class _MiniMaxStyleLLM(FakeLLMProvider):
    """Returns a MiniMax-M3 style response: think block followed by
    valid JSON. The parser must strip the think block and parse."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        content = (
            "reasoning...\n"
            "Let me think about this carefully.\n"
            "```json\n"
            + json.dumps({
                "verdict": REVIEW_VERDICT_REJECT,
                "reasons": [REVIEW_REASON_REPO_INTERNAL_TASK],
                "summary": "MiniMax-style response with think + fence + JSON.",
                "confidence": 0.7,
            })
            + "\n```"
        )
        return LLMResponse(content=content, model="minimax-style")


def test_review_repair_callback_does_not_reference_self() -> None:
    """Regression for V2.2 bug: the repair callback in
    `_parse_response` used `self._llm.complete(...)` even though
    `_parse_response` is a module-level function. The result was
    `NameError: name 'self' is not defined` and the review was
    marked `review_failed` even when the LLM had recoverable output.

    This test must NEVER raise NameError. If it does, the bug is back.
    """
    posts = [_post("x", body="y")]
    # _TwoCallLLM returns garbage on call 1, valid JSON on call 2.
    # The parser should hit the repair callback and recover.
    a = review_opportunity(
        _opp("x"), posts, _TwoCallLLM(),
    )
    # Must NOT be a NameError swallowing: the recovery should give us
    # the real verdict from the repair response.
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_NOT_BUYER_PAIN in a.reasons
    # Sanity: review_failed should NOT be a reason — repair succeeded.
    assert REVIEW_FAILED_TAG not in a.reasons


def test_review_repair_callback_repairs_malformed_json() -> None:
    """The repair callback, when invoked, asks the LLM to convert
    its own garbage into valid JSON, and the parser uses that JSON."""
    posts = [_post("x")]
    llm = _TwoCallLLM()
    a = review_opportunity(_opp("x"), posts, llm)
    # Two calls: one initial, one repair.
    assert len(llm.calls) == 2
    # The second call's user message should mention the repair contract.
    repair_user_msg = llm.calls[1][1].content
    assert "previous" in repair_user_msg.lower() or "convert" in repair_user_msg.lower()
    # And we got the recovered verdict.
    assert a.verdict == REVIEW_VERDICT_REJECT


def test_review_handles_minimax_think_block_with_parseable_json() -> None:
    """A MiniMax-M3 style response (think block + markdown fence + JSON)
    parses successfully on the first attempt — no repair needed."""
    posts = [_post("x")]
    a = review_opportunity(_opp("x"), posts, _MiniMaxStyleLLM())
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_REPO_INTERNAL_TASK in a.reasons
    assert REVIEW_FAILED_TAG not in a.reasons


def test_review_think_plus_parseable_json_no_repair_call() -> None:
    """When the think block + JSON is parseable after stripping, the
    review layer does NOT invoke the repair callback (single LLM call)."""
    posts = [_post("x")]
    llm = _MiniMaxStyleLLM()
    a = review_opportunity(_opp("x"), posts, llm)
    # Only the initial call — parser succeeded without repair.
    assert len(llm.calls) == 1
    assert a.verdict == REVIEW_VERDICT_REJECT


def test_review_repair_returns_none_when_llm_raises() -> None:
    """When the LLM raises on every call (including the repair attempt),
    the review layer must gracefully fall back to `reject` + `review_failed`."""
    posts = [_post("x")]
    # BoomLLM raises on every call, so the initial parse gets nothing,
    # and the repair call also raises. The shared make_repair_callback
    # returns None on exception, which makes parse_with_repair abort.
    a = review_opportunity(_opp("x"), posts, BoomLLM())
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_FAILED_TAG in a.reasons


def test_review_strip_thinking_provider_does_not_change_contract() -> None:
    """Provider-level thinking-block stripping (LLM_STRIP_THINKING_ALWAYS)
    is orthogonal: review_opportunity takes already-clean content from
    the provider, then the shared extract_json parser handles any
    remaining noise. A response with NO think blocks still parses."""
    posts = [_post("x")]
    a = review_opportunity(
        _opp("x"),
        posts,
        FakeLLMProvider(
            response_factory=lambda m: json.dumps({
                "verdict": REVIEW_VERDICT_REJECT,
                "reasons": [REVIEW_REASON_REPO_INTERNAL_TASK],
                "summary": "clean response, no think",
                "confidence": 0.5,
            })
        ),
    )
    assert a.verdict == REVIEW_VERDICT_REJECT
    assert REVIEW_REASON_REPO_INTERNAL_TASK in a.reasons
    assert REVIEW_FAILED_TAG not in a.reasons
