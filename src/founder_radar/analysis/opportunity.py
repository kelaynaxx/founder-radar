"""Opportunity extraction.

Two concrete extractors:

  - `HeuristicExtractor` — produces a basic opportunity from a cluster's
    top posts. No LLM. Used when no LLM is configured and in tests.

  - `LLMBasedExtractor` — calls `BaseLLMProvider` with a structured
    prompt asking the model to summarize the cluster. Returns a dict
    that the OpportunityRepository stores verbatim. Falls back to
    `HeuristicExtractor` on LLM errors so the pipeline never crashes.

Phase 3+ adds:
  - Reality Check (competitor detection + saturation) — deterministic.
  - Trend analysis (emerging/recurring/etc.) — deterministic.

Phase 3.5 adds:
  - Reality Validation (saturated/competitive/underserved/unknown) —
    deterministic interpretation of competitor + pain + dissatisfaction
    evidence. Orthogonal to scoring: it does NOT affect weighted_score.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

from founder_radar.analysis.llm_json import (
    parse_with_repair,
    LLMJsonError,
    ERROR_PREVIEW_CHARS,
)
from founder_radar.analysis.reality_check import run_reality_check
from founder_radar.analysis.reality_validator import assess_reality
from founder_radar.analysis.scoring import (
ScoreFactors,
compute_deterministic_scores,
)
from founder_radar.analysis.trends import run_trend_analysis
from founder_radar.analysis.trends import run_trend_analysis
from founder_radar.llm.base import BaseLLMProvider, LLMMessage

if TYPE_CHECKING:
    from founder_radar.database.models import Post

logger = logging.getLogger(__name__)


# Cap on how many posts we feed to the LLM in one prompt.
_LLM_MAX_POSTS = 20


class BaseExtractor(ABC):
    """Abstract opportunity extractor."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier persisted in `extraction_method`."""

    @abstractmethod
    def extract(
        self,
        *,
        cluster_id: int,
        posts: Iterable["Post"],
    ) -> dict:
        """Build an Opportunity dict from a cluster."""


def _build_corpus_excerpt(posts: list["Post"], max_chars: int = 8000) -> str:
    """Build a text excerpt from a list of posts for prompting."""
    blocks: list[str] = []
    total = 0
    for i, p in enumerate(posts, start=1):
        title = (p.title or "").strip()
        body = (p.body or "").strip()
        block = f"[{i}] {title}\n{body}\n"
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


def _posts_meta(posts: list["Post"]) -> list[dict]:
    """Return small per-post metadata for embedding in prompts / dicts."""
    return [
        {
            "id": p.id,
            "title": p.title,
            "url": p.url,
            "score": p.score,
            "comments": p.num_comments,
        }
        for p in posts
    ]


def _build_opportunity_dict(
    *,
    cluster_id: int,
    posts: list["Post"],
    scores: ScoreFactors,
    competitors: list[str],
    distinct_competitor_count: int,
    competitor_mention_count: int,
    saturation_score: float,
    trend: str,
    title: str,
    problem_summary: str,
    target_audience: str | None,
    saas_ideas: list[str],
    extraction_method: str,
    llm_model: str | None,
    source_links: list[str] | None = None,
    reality_status: str = "unknown",
    reality_confidence: float = 0.0,
    competitor_strength_estimate: float = 0.0,
) -> dict:
    """Build the final Opportunity dict from all the inputs.

    Shared by both extractors so they produce identically-shaped dicts.
    """
    if source_links is None:
        source_links = [p.url for p in posts[:5] if p.url]

    return {
        "title": title,
        "problem_summary": problem_summary,
        "target_audience": target_audience,
        "saas_ideas": saas_ideas,
        "competitors": competitors,
        "source_links": source_links,
        # 8-factor scores
        "frequency_score": scores.frequency,
        "emotional_intensity_score": scores.emotional_intensity,
        "dissatisfaction_score": scores.dissatisfaction,
        "market_size_score": scores.market_size,
        "ease_of_implementation_score": scores.ease_of_implementation,
        "recurring_revenue_score": scores.recurring_revenue,
        "technical_feasibility_score": scores.technical_feasibility,
        "novelty_score": scores.novelty,
        # Aggregates (Phase 3+)
        "total_score": scores.total,
        "pain_score": scores.pain,
        "monetization_score": scores.monetization,
        "weighted_score": scores.weighted,
        "confidence_score": scores.confidence,
        # Reality Check (Phase 3+)
        "saturation_score": saturation_score,
        "distinct_competitor_count": distinct_competitor_count,
        "competitor_mention_count": competitor_mention_count,
        # Trend (Phase 3+)
        "trend": trend,
        # Phase 3.5 Reality Validation (orthogonal to scoring).
        "reality_status": reality_status,
        "reality_confidence": reality_confidence,
        "competitor_strength_estimate": competitor_strength_estimate,
        # Metadata
        "cluster_id": cluster_id,
        "mentions": len(posts),
        "extraction_method": extraction_method,
        "llm_model": llm_model,
        "status": "new",
    }


# -------------------------------------------------------------------------
# HeuristicExtractor
# -------------------------------------------------------------------------
class HeuristicExtractor(BaseExtractor):
    """Build a basic opportunity from cluster posts without an LLM."""

    name: str = "heuristic"

    def extract(self, *, cluster_id: int, posts: Iterable["Post"]) -> dict:
        posts = list(posts)
        if not posts:
            raise RuntimeError(
                f"HeuristicExtractor: cluster {cluster_id} has no posts"
            )

        ranked = sorted(
            posts,
            key=lambda p: (p.score or 0) + (p.num_comments or 0),
            reverse=True,
        )

        top = ranked[0]
        title = top.title.strip() if top.title else f"Cluster {cluster_id} opportunity"

        top_titles = [p.title for p in ranked[:3] if p.title]
        problem_summary = (
            "Multiple posts discuss: " + " | ".join(top_titles)
            if top_titles
            else f"Cluster {cluster_id} contains {len(posts)} related posts."
        )

        # Deterministic scores.
        scores = compute_deterministic_scores(posts)
        scores.market_size = 0.5
        scores.ease_of_implementation = 0.5
        scores.recurring_revenue = 0.5
        scores.technical_feasibility = 0.5
        scores.llm_filled = set()

        # Reality Check + Reality Validation + Trend.
        reality = run_reality_check(posts)
        reality_assessment = assess_reality(
            posts,
            competitors=reality.competitors,
            distinct_competitor_count=reality.distinct_competitor_count,
            competitor_mention_count=reality.competitor_mention_count,
        )
        trend_report = run_trend_analysis(posts)

        return _build_opportunity_dict(
            cluster_id=cluster_id,
            posts=posts,
            scores=scores,
            competitors=reality.competitors,
            distinct_competitor_count=reality.distinct_competitor_count,
            competitor_mention_count=reality.competitor_mention_count,
            saturation_score=reality.saturation_score,
            trend=trend_report.trend,
            title=title,
            problem_summary=problem_summary,
            target_audience=None,
            saas_ideas=[],
            extraction_method=self.name,
            llm_model=None,
            reality_status=reality_assessment.status,
            reality_confidence=reality_assessment.saturation_confidence,
            competitor_strength_estimate=(
                reality_assessment.competitor_strength_estimate
            ),
        )


# -------------------------------------------------------------------------
# LLMBasedExtractor
# -------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert SaaS opportunity analyst.
You will be given several related discussion posts. Your job is to extract
a single, well-defined software business opportunity from them.

Output strict JSON only — no commentary, no markdown fences. The JSON
schema you must follow exactly:

{
  "title": "short opportunity title (<= 12 words)",
  "problem_summary": "2-4 sentence summary of the problem people face",
  "target_audience": "1 sentence describing who suffers from this",
  "saas_ideas": ["list of 1-3 SaaS / product ideas that would address the problem"],
  "competitors": ["list of 0-5 existing products or companies that try to solve it"],
  "scores": {
    "market_size": <0.0..1.0>,
    "ease_of_implementation": <0.0..1.0>,
    "recurring_revenue": <0.0..1.0>,
    "technical_feasibility": <0.0..1.0>
  }
}

Rules:
- Scores are floats in [0, 1]. Be conservative; do not inflate.
- `market_size` reflects addressable market (0 = tiny niche, 1 = massive).
- `ease_of_implementation` reflects build cost for a small team (0 = years, 1 = weeks).
- `recurring_revenue` reflects how naturally a SaaS subscription fits.
- `technical_feasibility` reflects whether today's tech stack is enough.
- If you're unsure about a number, pick 0.5.
- Output JSON and nothing else."""


class LLMBasedExtractor(BaseExtractor):
    """Use an LLM to extract an opportunity from cluster posts."""

    name: str = "llm"

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        fallback: BaseExtractor | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or HeuristicExtractor()

    def extract(self, *, cluster_id: int, posts: Iterable["Post"]) -> dict:
        posts = list(posts)
        if not posts:
            raise RuntimeError(
                f"LLMBasedExtractor: cluster {cluster_id} has no posts"
            )

        det = compute_deterministic_scores(posts)

        try:
            llm_block, response_model = self._call_llm(cluster_id, posts)
        except Exception as exc:
            logger.warning(
                "LLM extraction failed for cluster %d (%s); falling back to heuristic.",
                cluster_id, exc,
            )
            return self._fallback.extract(cluster_id=cluster_id, posts=posts)

        llm_scores = llm_block.get("scores", {}) or {}
        scores = ScoreFactors(
            frequency=det.frequency,
            emotional_intensity=det.emotional_intensity,
            dissatisfaction=det.dissatisfaction,
            novelty=det.novelty,
            market_size=_clip01(llm_scores.get("market_size", 0.5)),
            ease_of_implementation=_clip01(
                llm_scores.get("ease_of_implementation", 0.5)
            ),
            recurring_revenue=_clip01(llm_scores.get("recurring_revenue", 0.5)),
            technical_feasibility=_clip01(
                llm_scores.get("technical_feasibility", 0.5)
            ),
            llm_filled={
                "market_size", "ease_of_implementation",
                "recurring_revenue", "technical_feasibility",
            },
        )

        # Reality Check: union LLM-named competitors with regex-detected ones.
        llm_competitors = list(llm_block.get("competitors") or [])
        reality = run_reality_check(posts)
        merged_competitors: list[str] = []
        seen: set[str] = set()
        for name in llm_competitors + reality.competitors:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                merged_competitors.append(name)

        # Phase 3.5 Reality Validation. Feed it the merged competitor list
        # so the assessment sees both LLM-suggested and regex-detected names.
        reality_assessment = assess_reality(
            posts,
            competitors=merged_competitors,
            distinct_competitor_count=len(merged_competitors),
            competitor_mention_count=reality.competitor_mention_count,
        )

        trend_report = run_trend_analysis(posts)

        return _build_opportunity_dict(
            cluster_id=cluster_id,
            posts=posts,
            scores=scores,
            competitors=merged_competitors,
            distinct_competitor_count=len(merged_competitors),
            competitor_mention_count=reality.competitor_mention_count,
            saturation_score=reality.saturation_score,
            trend=trend_report.trend,
            title=llm_block.get("title", f"Cluster {cluster_id} opportunity"),
            problem_summary=llm_block.get(
                "problem_summary",
                f"Cluster {cluster_id} contains {len(posts)} related posts.",
            ),
            target_audience=llm_block.get("target_audience"),
            saas_ideas=list(llm_block.get("saas_ideas") or []),
            extraction_method=self.name,
            llm_model=response_model or _model_name(self._llm),
            reality_status=reality_assessment.status,
            reality_confidence=reality_assessment.saturation_confidence,
            competitor_strength_estimate=(
                reality_assessment.competitor_strength_estimate
            ),
        )

    def _call_llm(self, cluster_id: int, posts: list["Post"]) -> tuple[dict, str]:
        recent = sorted(posts, key=lambda p: p.collected_at, reverse=True)[
            :_LLM_MAX_POSTS
        ]
        excerpt = _build_corpus_excerpt(recent)
        meta = _posts_meta(recent)

        user_prompt = (
            f"Cluster id: {cluster_id}\n"
            f"Number of posts in cluster: {len(posts)}\n"
            f"Sample of {len(recent)} most recent posts (with ids):\n"
            f"{json.dumps(meta, ensure_ascii=False)}\n\n"
            f"Post excerpts:\n{excerpt}\n\n"
            "Return the JSON object as specified."
        )

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Initial call. Capture BOTH the content and the model name
        # from the LLMResponse so downstream display shows the real
        # model (not a fallback). Tests use FakeLLMProvider which sets
        # `model="fake-model"`; real providers surface their actual
        # model in `LLMResponse.model`.
        try:
            initial_response = self._llm.complete(
                messages,
                temperature=0.2,
                max_tokens=800,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM provider call failed: {exc}") from exc
        initial_content = initial_response.content
        initial_model = initial_response.model

        # One retry-repair pass on parse failure (V2.1, brief task 3).
        # The repair callback asks the LLM to convert its previous
        # (failed) output back into valid JSON. If the repair call
        # itself fails, parse_with_repair returns value=None and we
        # raise — the outer extract() catches and falls back to the
        # heuristic extractor.
        def _repair_callback(failed_preview: str) -> str | None:
            repair_messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are a strict JSON repair assistant. Given "
                        "the assistant's previous response, output ONLY "
                        "valid JSON — no commentary, no markdown fences, "
                        "no <think>...</think> blocks, no prose."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"The previous response could not be parsed as "
                        f"JSON. Convert it into valid JSON matching "
                        f"the original schema.\n\n"
                        f"--- Previous response (first {ERROR_PREVIEW_CHARS} "
                        f"chars) ---\n{failed_preview}"
                    ),
                ),
            ]
            try:
                return self._llm.complete(
                    repair_messages,
                    temperature=0.0,
                    max_tokens=800,
                ).content
            except Exception as exc:
                logger.warning(
                    "LLM repair call failed for cluster %d: %s",
                    cluster_id, exc,
                )
                return None

        result = parse_with_repair(
            initial_content,
            repair=_repair_callback,
        )
        if result.value is None:
            preview = (
                result.last_content[:ERROR_PREVIEW_CHARS]
                if result.last_content else ""
            )
            raise LLMJsonError(
                f"LLM extraction failed for cluster {cluster_id} "
                f"after {result.attempts} attempt(s): "
                f"{result.last_error}",
                preview=preview,
            )
        # Use the model name from the INITIAL response (not the
        # repair response — that's a repair, not the real model).
        # Fall back to the provider's best-effort name only when the
        # provider doesn't surface a model name in its response.
        model_name = initial_model or _model_name(self._llm)
        return result.value, model_name




def _model_name(llm: BaseLLMProvider) -> str:
    """Best-effort human-readable model name from the provider."""
    for attr in ("model", "model_name"):
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if isinstance(value, str) and value:
                return value
    return getattr(llm, "name", "unknown")


def _clip01(x) -> float:  # type: ignore[no-untyped-def]
    """Clip a numeric value to `[0, 1]`. Returns 0.5 on type errors."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))


def build_extractor(settings) -> BaseExtractor:  # type: ignore[no-untyped-def]
    """Pick the extractor based on configuration."""
    if settings.llm_api_key:
        from founder_radar.llm.openai_provider import OpenAICompatibleProvider
        llm = OpenAICompatibleProvider(settings)
        return LLMBasedExtractor(llm=llm, fallback=HeuristicExtractor())
    return HeuristicExtractor()