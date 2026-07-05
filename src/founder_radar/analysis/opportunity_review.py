"""Opportunity Review Layer (Phase 4+ LLM-assisted triage filter).

This module is a *narrow* review layer that sits on top of the
deterministic `opportunity_type` + `productizability_score` classifier.
Its job is to act like a strict startup-analyst filter:

  - **Default to `reject`.** Most opportunity clusters that the
    deterministic classifier surfaces are NOT real product
    opportunities — they're internal implementation tickets,
    chores, CI configuration, generated status reports, upstream
    library bugs, single-repo defects, or vague feature requests.
  - **`maybe`** is allowed for developer/infrastructure pain that
    *could* become a tool but lacks proof of willingness to pay.
  - **`strong_candidate`** ONLY when there is clear repeated pain,
    a clear buyer/user, and a plausible standalone tool/service.

This is NOT a generator. It's a triage filter.
This is NOT final truth. It's a model opinion.
The deterministic classifier's verdict and `productizability_score`
are still the primary ranking signal.

The review layer is optional. The normal pipeline
(`founder-radar extract && productizable`) does NOT call the LLM.
To use the review layer, run `founder-radar review-opportunities`.

Design rules:
  - **Deterministic JSON contract.** The LLM is asked to output a
    strict JSON object with a fixed schema. We parse it; if parsing
    fails, we return `reject` with reason `review_failed`. The CLI
    never crashes on a bad LLM response.
  - **LLM is required to quote evidence.** The system prompt forbids
    generic opinions — every claim must reference concrete source
    posts. The summary field is free-form but must include at least
    one concrete reference.
  - **Tone of the system prompt: strict analyst.** It defaults to
    `reject` and is told to be conservative. Most clusters should
    be rejected.
  - **The classifier (deterministic) is the source of truth.** The
    review layer can override `potential_product` -> `reject` (the
    common case on real GitHub runs) but is not allowed to promote
    a non-`potential_product` cluster to `strong_candidate`. That
    second rule is a safety net: it stops the LLM from "discovering"
    products the deterministic layer already dismissed.
  - **Re-runnable.** The `review-opportunities` CLI is idempotent
    and can be re-run after editing posts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from founder_radar.analysis.llm_json import (
    ERROR_PREVIEW_CHARS,
    extract_json,
    make_repair_callback,
    parse_with_repair,
    try_extract_json,
)
from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

if TYPE_CHECKING:
    from founder_radar.database.models import Opportunity, Post

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Public constants: the review taxonomy
# -------------------------------------------------------------------------
# Verdicts. Strict 3-state. The CLI never shows "unreviewed" opportunities
# as a 4th bucket; the absence of a verdict means "unreviewed" and the
# CLI surfaces them as "reject" by default.
REVIEW_VERDICT_REJECT = "reject"
REVIEW_VERDICT_MAYBE = "maybe"
REVIEW_VERDICT_STRONG_CANDIDATE = "strong_candidate"

ALL_REVIEW_VERDICTS = (
    REVIEW_VERDICT_REJECT,
    REVIEW_VERDICT_MAYBE,
    REVIEW_VERDICT_STRONG_CANDIDATE,
)

# Reasons. The LLM picks 1-3 from this list. We also have one
# synthetic tag (`REVIEW_FAILED_TAG`) that we apply ourselves when
# the LLM doesn't return parseable JSON.
REVIEW_REASON_REPO_INTERNAL_TASK = "repo_internal_task"
REVIEW_REASON_UPSTREAM_BUG = "upstream_bug"
REVIEW_REASON_MAINTENANCE_CHORE = "maintenance_chore"
REVIEW_REASON_DOCUMENTATION_ONLY = "documentation_only"
REVIEW_REASON_NOT_BUYER_PAIN = "not_buyer_pain"
REVIEW_REASON_TOO_VAGUE = "too_vague"
REVIEW_REASON_TOO_REPO_SPECIFIC = "too_repo_specific"
REVIEW_REASON_POSSIBLE_DEVTOOL = "possible_devtool"
REVIEW_REASON_POSSIBLE_MICRO_SAAS = "possible_micro_saas"
REVIEW_REASON_POSSIBLE_INFRA_TOOL = "possible_infra_tool"
REVIEW_REASON_STRONG_REPEATED_PAIN = "strong_repeated_pain"
REVIEW_FAILED_TAG = "review_failed"

ALL_REVIEW_REASONS = (
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
)

# Allowed "maybe" / "strong_candidate" reasons. We use this list to
# spot a hallucinated reason tag (anything outside the canonical set
# is dropped with a warning).
_MAYBE_REASONS = frozenset({
    REVIEW_REASON_POSSIBLE_DEVTOOL,
    REVIEW_REASON_POSSIBLE_MICRO_SAAS,
    REVIEW_REASON_POSSIBLE_INFRA_TOOL,
})

_STRONG_REASONS = frozenset({
    REVIEW_REASON_STRONG_REPEATED_PAIN,
    REVIEW_REASON_POSSIBLE_MICRO_SAAS,
    REVIEW_REASON_POSSIBLE_INFRA_TOOL,
    REVIEW_REASON_POSSIBLE_DEVTOOL,
})

# Cap on how many posts we feed to the LLM in one prompt. Real
# clusters can be larger; we keep the most recent N to bound cost.
_LLM_MAX_POSTS = 20
# Max characters of post body we include per post. Long issue
# bodies waste tokens.
_LLM_MAX_BODY_CHARS = 1200
# Max characters of total post excerpt sent to the LLM. 30k chars
# is plenty for 20 posts and well within most context windows.
_LLM_MAX_CORPUS_CHARS = 30_000


# -------------------------------------------------------------------------
# System prompt
# -------------------------------------------------------------------------
_REVIEW_SYSTEM_PROMPT = """You are a strict, skeptical startup analyst \
reviewing an opportunity cluster extracted from public discussions \
(GitHub issues, Reddit posts, Hacker News threads).

Your job is to triage the cluster into ONE of three verdicts:

  - "strong_candidate"   Strong evidence of a real product opportunity
  - "maybe"               Plausible but unproven; could become a tool
  - "reject"              NOT a real product opportunity

DEFAUL TO "reject". Most opportunity clusters are NOT real product \
opportunities. They are: internal implementation tickets, chores, \
CI configuration tasks, generated status reports, upstream library \
bugs, single-repo defects, or vague feature requests. Be \
conservative.

STRONG CANDIDATE requires ALL of:
  1. Clear, REPEATED pain across multiple posts (not a one-off complaint)
  2. A clear buyer or user persona (not just "developers in general")
  3. A plausible standalone tool or service that could address the pain
  4. The pain is NOT primarily a bug in a specific upstream library
     (i.e. fixing it requires patching OpenAI, pydantic, etc., not
     building a new product)

MAYBE is allowed when:
  - Developer or infrastructure pain is real and repeated, BUT
  - There's no clear evidence of willingness to pay, OR
  - The potential product is unclear

REJECT when the cluster is:
  - An internal implementation ticket ("refactor X", "add test for Y")
  - A maintenance chore ("bump dependency version", "update CI config")
  - A generated status report (Dependabot / Renovate / GitHub Actions)
  - An upstream SDK / library bug (patching OpenAI, pydantic, etc.)
  - A single-repo defect with no cross-cutting evidence
  - A vague feature request with no concrete buyer or use-case

QUOTE OR SUMMARIZE CONCRETE EVIDENCE from the source posts. Do not \
invent reasons. If you cannot quote evidence, the verdict is "reject".

OUTPUT STRICT JSON ONLY. No commentary, no markdown fences. The \
schema is:

{
  "verdict": "reject" | "maybe" | "strong_candidate",
  "reasons": ["reason_tag_1", "reason_tag_2"],
  "summary": "1-3 sentence justification that quotes or summarizes at least one concrete post",
  "confidence": 0.0-1.0
}

Allowed reason_tag values (pick 1-3, in priority order):
  - "repo_internal_task"          internal implementation ticket
  - "upstream_bug"                SDK / library / API bug (patch upstream)
  - "maintenance_chore"           chore / dep bump / CI config
  - "documentation_only"         docs / examples / how-to request
  - "not_buyer_pain"              no clear buyer; pain is theoretical
  - "too_vague"                   not specific enough to build for
  - "too_repo_specific"           single-repo defect, no cross-cutting
  - "possible_devtool"            could be a developer tool (maybe)
  - "possible_micro_saas"         could be a small SaaS (maybe/strong)
  - "possible_infra_tool"         could be infrastructure tooling
  - "strong_repeated_pain"        strong, repeated, buyer-identified

`confidence` is YOUR self-reported confidence in the verdict \
(0.0-1.0). Be honest; if the posts are noisy, say 0.3-0.5.

Do not output anything besides the JSON object."""


# -------------------------------------------------------------------------
# Public dataclass
# -------------------------------------------------------------------------
@dataclass(slots=True)
class ReviewVerdict:
    """Output of the LLM-assisted opportunity review.

    `verdict` is one of the three `REVIEW_VERDICT_*` strings.
    `reasons` is a list of `REVIEW_REASON_*` tags (filtered to the
    canonical set; unknown tags are dropped).
    `summary` is the LLM's free-form justification, expected to
    quote or summarize at least one concrete source post.
    `confidence` is the LLM's self-reported confidence in its
    verdict, in [0, 1].
    `raw_response` is the original LLM text (debug-only).
    """

    verdict: str = REVIEW_VERDICT_REJECT
    reasons: list[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    raw_response: str = ""

    @property
    def is_acceptable(self) -> bool:
        """True for the two states we treat as 'potentially interesting'."""
        return self.verdict in (REVIEW_VERDICT_MAYBE, REVIEW_VERDICT_STRONG_CANDIDATE)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def review_opportunity(
    opportunity: "Opportunity",
    posts: Iterable["Post"],
    llm: BaseLLMProvider,
    *,
    max_posts: int = _LLM_MAX_POSTS,
) -> ReviewVerdict:
    """Run the LLM-assisted review on one opportunity cluster.

    Args:
        opportunity: The `Opportunity` row to triage.
        posts: The source posts in the cluster.
        llm: Any `BaseLLMProvider` (real or fake).
        max_posts: Cap on how many posts to include in the prompt.

    Returns:
        A `ReviewVerdict`. Never raises — on any LLM/JSON error
        we return `ReviewVerdict(verdict='reject', reasons=['review_failed'], ...)`.
    """
    messages = _build_messages(opportunity, posts, max_posts=max_posts)
    try:
        response = llm.complete(messages, temperature=0.1, max_tokens=900)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Review LLM call failed for opportunity %s: %s",
            getattr(opportunity, "id", "?"), exc,
        )
        return ReviewVerdict(
            verdict=REVIEW_VERDICT_REJECT,
            reasons=[REVIEW_FAILED_TAG],
            summary=f"review_failed: {exc}",
            confidence=0.0,
        )
    return _parse_response(
        response,
        llm=llm,
        opportunity_type=getattr(opportunity, "opportunity_type", None),
    )


def review_opportunities_batch(
    opportunities: Iterable["Opportunity"],
    posts_by_id: dict[int, list["Post"]],
    llm: BaseLLMProvider,
    *,
    progress: bool = True,
) -> dict[int, ReviewVerdict]:
    """Run the review for many opportunities. Returns {opp_id: ReviewVerdict}."""
    out: dict[int, ReviewVerdict] = {}
    for opp in opportunities:
        opp_id = getattr(opp, "id", None)
        if opp_id is None:
            continue
        posts = posts_by_id.get(opp_id, [])
        out[opp_id] = review_opportunity(opp, posts, llm)
        if progress:
            verdict = out[opp_id].verdict
            logger.info(
                "  review opp=%s -> %s (conf=%.2f)",
                opp_id, verdict, out[opp_id].confidence,
            )
    return out


# -------------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------------
def _build_messages(
    opportunity: "Opportunity",
    posts: Iterable["Post"],
    *,
    max_posts: int,
) -> list[LLMMessage]:
    """Build the (system, user) message pair for the review LLM."""
    # Take the most recent N posts (deterministic on collected_at).
    recent = sorted(
        posts, key=lambda p: getattr(p, "collected_at", None) or 0, reverse=True
    )[:max_posts]
    corpus_lines: list[str] = []
    for i, p in enumerate(recent, start=1):
        title = (getattr(p, "title", "") or "").strip()
        body = (getattr(p, "body", "") or "").strip()
        if len(body) > _LLM_MAX_BODY_CHARS:
            body = body[:_LLM_MAX_BODY_CHARS] + " ...[truncated]"
        author = getattr(p, "author", None) or "?"
        source = getattr(p, "source", "?")
        category = getattr(p, "source_category", None) or "?"
        corpus_lines.append(
            f"[{i}] source={source} category={category} author={author}\n"
            f"    title: {title}\n"
            f"    body:  {body}"
        )
    corpus = "\n\n".join(corpus_lines)
    if len(corpus) > _LLM_MAX_CORPUS_CHARS:
        corpus = corpus[:_LLM_MAX_CORPUS_CHARS] + "\n...[truncated]"

    user_prompt = (
        f"Opportunity id: {getattr(opportunity, 'id', '?')}\n"
        f"Opportunity title: {(getattr(opportunity, 'title', '') or '').strip()}\n"
        f"Opportunity problem_summary: "
        f"{(getattr(opportunity, 'problem_summary', '') or '').strip()}\n"
        f"Opportunity target_audience: "
        f"{(getattr(opportunity, 'target_audience', '') or '').strip() or '(not set)'}\n"
        f"Deterministic opportunity_type: "
        f"{getattr(opportunity, 'opportunity_type', None) or '(unset)'}\n"
        f"Deterministic productizability_score: "
        f"{getattr(opportunity, 'productizability_score', 0.0) or 0.0:.2f}\n"
        f"Number of source posts: {len(recent)}\n"
        f"Number of distinct source_category values: "
        f"{len({getattr(p, 'source_category', None) for p in recent if getattr(p, 'source_category', None)})}\n\n"
        f"Source posts (most recent first):\n{corpus}\n\n"
        "Return the JSON object as specified."
    )
    return [
        LLMMessage(role="system", content=_REVIEW_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]


def _parse_response(
    response: LLMResponse,
    *,
    llm: BaseLLMProvider,
    opportunity_type: str | None = None,
) -> ReviewVerdict:
    """Parse the LLM's reply as JSON. Tolerates markdown fences.

    Any parse failure -> `reject` with `review_failed`. The `llm`
    provider is threaded in so the retry-repair callback can re-query
    it without relying on a class-level `self` (this function lives at
    module scope — see git history for the `NameError: self` bug).
    """
    raw = response.content or ""

    # Wrap the provider's `complete` so it matches the `llm_complete`
    # callable shape that `make_repair_callback` expects: takes a
    # message list, returns a content string.
    def _llm_complete(messages: list[LLMMessage]) -> str:
        return llm.complete(messages, temperature=0.0, max_tokens=900).content

    # Schema hint drives the repair prompt — keep it in sync with the
    # strict-JSON contract documented in _REVIEW_SYSTEM_PROMPT.
    _REVIEW_SCHEMA_HINT = (
        "{\n"
        '  "verdict": "reject" | "maybe" | "strong_candidate",\n'
        '  "reasons": ["reason_tag_1", "reason_tag_2"],\n'
        '  "summary": "1-3 sentence justification",\n'
        '  "confidence": 0.0-1.0\n'
        "}"
    )
    repair = make_repair_callback(
        _llm_complete,
        schema_hint=_REVIEW_SCHEMA_HINT,
    )

    result = parse_with_repair(raw, repair=repair)
    data = result.value
    if data is None:
        # Both attempts failed. Use the shared try_extract_json
        # (which strips <think>, fences, balanced braces) so we give
        # one more chance to a partial response before giving up.
        data = try_extract_json(raw)
    if data is None:
        logger.warning(
            "Review LLM returned non-JSON after %d attempt(s) (model=%s). "
            "First %d chars: %r",
            result.attempts, response.model, ERROR_PREVIEW_CHARS, raw[:ERROR_PREVIEW_CHARS],
        )
        return ReviewVerdict(
            verdict=REVIEW_VERDICT_REJECT,
            reasons=[REVIEW_FAILED_TAG],
            summary="review_failed: LLM did not return parseable JSON.",
            confidence=0.0,
            raw_response=raw,
        )

    # Validate / sanitize fields.
    verdict = data.get("verdict", REVIEW_VERDICT_REJECT)
    if verdict not in ALL_REVIEW_VERDICTS:
        logger.warning(
            "Review LLM returned unknown verdict %r; coercing to 'reject'.",
            verdict,
        )
        verdict = REVIEW_VERDICT_REJECT
    raw_reasons = data.get("reasons") or []
    if not isinstance(raw_reasons, list):
        raw_reasons = []
    # Filter to canonical reason tags; cap at 3.
    reasons: list[str] = []
    for r in raw_reasons:
        if not isinstance(r, str):
            continue
        if r not in ALL_REVIEW_REASONS:
            logger.debug("Review LLM returned unknown reason %r; dropping.", r)
            continue
        if r in reasons:
            continue
        reasons.append(r)
        if len(reasons) >= 3:
            break

    summary = data.get("summary") or ""
    if not isinstance(summary, str):
        summary = str(summary)
    summary = summary.strip()
    if len(summary) > 1000:
        summary = summary[:1000] + "..."

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Safety net: only `potential_product` opportunities can be
    # `strong_candidate` (per the brief's "deterministic
    # classification is the source of truth" rule). If the LLM
    # promotes a non-`potential_product` cluster, demote.
    if (verdict == REVIEW_VERDICT_STRONG_CANDIDATE
            and opportunity_type
            and opportunity_type != "potential_product"):
        logger.info(
            "Review LLM said strong_candidate for non-potential_product "
            "cluster (type=%s); demoting to maybe.",
            opportunity_type,
        )
        verdict = REVIEW_VERDICT_MAYBE

    # Safety net: `maybe` reasons should be from the maybe set.
    if verdict == REVIEW_VERDICT_MAYBE and reasons and not (
        set(reasons) & _MAYBE_REASONS
    ):
        # Allow the LLM's reject-style reasons to push a maybe down to
        # reject, but only when ALL reasons are reject-class.
        if all(r not in _STRONG_REASONS for r in reasons):
            logger.info(
                "Review LLM said maybe with no maybe-class reasons; "
                "demoting to reject."
            )
            verdict = REVIEW_VERDICT_REJECT

    return ReviewVerdict(
        verdict=verdict,
        reasons=reasons,
        summary=summary,
        confidence=confidence,
        raw_response=raw,
    )


def _try_parse_json(text: str) -> dict | None:
    """Best-effort JSON parse, tolerant of <think>...</think>, fences, prose.

    Thin wrapper around the shared `try_extract_json` helper in
    `analysis.llm_json` (V2.1, brief task 2). Kept here as a
    back-compat shim for callers and tests that imported the local
    function.
    """
    return try_extract_json(text)
