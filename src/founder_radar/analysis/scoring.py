"""Opportunity scoring — the 8-factor score from the project brief.

The brief specifies these eight factors:

  1. frequency                       - how often the problem is mentioned
  2. emotional_intensity             - how strongly people feel about it
  3. current_dissatisfaction         - whether existing solutions fail
  4. market_size                     - estimated size of the addressable market
  5. ease_of_implementation          - how easy to build
  6. recurring_revenue_potential     - whether it lends itself to subscription
  7. technical_feasibility           - whether it's technically doable today
  8. novelty                         - how much existing competition there is

This module is intentionally **deterministic** for factors 1, 2, 3, 8 —
they can be measured from the posts themselves without an LLM. The
other four (4, 5, 6, 7) require judgment; the default values are
neutral (`0.5`) so a deterministic-only score is still valid. The LLM
extractor (in `opportunity.py`) overwrites those four with real scores.

Design:
  - All scores are on `[0, 1]`.
  - `total_score` is the unweighted mean.
  - `confidence_score` reflects how much of the score came from
    deterministic measures vs LLM. Callers pass `llm_used=True/False`.

The scorer is *pure*: it takes a list of posts (and optionally some
LLM-derived values) and returns a `ScoreFactors` dataclass. No I/O.
That makes it trivially testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from founder_radar.database.models import Post

logger = logging.getLogger(__name__)


# Heuristic dictionaries for sentiment / frustration / competition cues.
# These are intentionally simple — a real production system would use a
# proper sentiment model. For Phase 3 MVP, a small lexicon is enough to
# give reproducible, testable scores.

_FRUSTRATION_CUES = (
    "hate", "sucks", "broken", "terrible", "awful", "frustrated",
    "frustrating", "annoying", "annoyed", "useless", "doesn't work",
    "does not work", "impossible", "ridiculous", "waste", "garbage",
    "nightmare", "pain", "painful", "stupid",
)

_DIS_SATISFACTION_CUES = (
    # Phrases that signal the user has tried existing solutions and
    # walked away disappointed.
    "tried everything", "nothing works", "no good alternative",
    "no good option", "switched from", "switching from", "left",
    "cancelled", "unsubscribed", "moved away from", "fed up with",
    "tired of", "gave up on", "abandoned",
)

_COMPETITOR_CUES = (
    # Common word patterns that suggest a competitor is being mentioned.
    # We don't try to enumerate every product — we count mentions.
    "vs.", "vs ", "alternative to", "instead of", "better than",
    "compared to", "competitor", "competition", "rival",
)

_QUESTION_CUES = (
    # Posts that end in a question are usually someone *asking* for a
    # solution that doesn't exist for them.
    "?",  # cheap signal; combined with the "anyone know" / "how do"
    "anyone know", "any suggestions", "how do you", "is there a",
    "looking for", "recommend", "recommendations", "any tool",
)

_cue_pattern = lambda cues: re.compile(  # noqa: E731
    r"\b(" + "|".join(re.escape(c) for c in cues) + r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ScoreFactors:
    """Container for the 8 score factors plus meta-fields.

    All factor fields are on `[0, 1]`. `llm_filled` is a set of names
    the LLM extractor overrode; used to compute `confidence_score`.
    """

    frequency: float = 0.0
    emotional_intensity: float = 0.0
    dissatisfaction: float = 0.0
    market_size: float = 0.5
    ease_of_implementation: float = 0.5
    recurring_revenue: float = 0.5
    technical_feasibility: float = 0.5
    novelty: float = 0.0

    llm_filled: set[str] = field(default_factory=set)

    @property
    def total(self) -> float:
        """Unweighted mean of the 8 factors, on `[0, 1]`."""
        return (
            self.frequency
            + self.emotional_intensity
            + self.dissatisfaction
            + self.market_size
            + self.ease_of_implementation
            + self.recurring_revenue
            + self.technical_feasibility
            + self.novelty
        ) / 8.0

    @property
    def pain(self) -> float:
        """Pain subscore (Phase 3+ weighted ranking).

        Pain is the user-side signal that this is a *real problem* worth
        building for. We weight dissatisfaction and emotional intensity
        highest (they're the strongest signals of user pain), with
        frequency as a tie-breaker (a problem mentioned by 10 people is
        more painful than one mentioned by 1).
        """
        return (
            self.dissatisfaction * 0.4
            + self.emotional_intensity * 0.4
            + self.frequency * 0.2
        )

    @property
    def monetization(self) -> float:
        """Monetization subscore.

        Recurring revenue potential dominates (we want subscriptions,
        not one-off sales). Market size and ease of implementation are
        supporting signals.
        """
        return (
            self.recurring_revenue * 0.4
            + self.market_size * 0.3
            + self.ease_of_implementation * 0.3
        )

    @property
    def weighted(self) -> float:
        """Pain-dominated ranking score (Phase 3+).

        Per the project brief: pain > monetization > novelty. We
        translate that into 50% / 40% / 10%. The result is on `[0, 1]`
        and is the default sort key for `opportunities`.
        """
        return (
            self.pain * 0.5
            + self.monetization * 0.4
            + self.novelty * 0.1
        )

    @property
    def confidence(self) -> float:
        """Fraction of fields that came from a real LLM call.

        Four fields require LLM judgment (market_size, ease_of_impl,
        recurring_revenue, technical_feasibility). When those are LLM-
        filled, confidence is high; otherwise we report the
        deterministic-only confidence.

        We always include the 4 deterministic factors as "covered" so
        confidence never drops below 0.5. This signals "we have real
        post-level signal but no LLM judgment" — useful, not garbage.
        """
        llm_share = len(self.llm_filled) / 4.0  # 4 LLM-fillable factors
        return 0.5 + 0.5 * llm_share


def compute_deterministic_scores(posts: Iterable["Post"]) -> ScoreFactors:
    """Compute the four post-level scores for a set of posts.

    The four LLM-judgment factors (market_size, ease_of_implementation,
    recurring_revenue, technical_feasibility) are left at their neutral
    0.5 default. The caller (an extractor) can override them after
    calling this function.

    Args:
        posts: Iterable of `Post` ORM rows (we only read title + body).

    Returns:
        `ScoreFactors` with the four post-derived fields populated.
        `llm_filled` is empty.
    """
    posts = list(posts)
    if not posts:
        # No signal at all. Return neutral values rather than zeros,
        # so the opportunity still ranks in the middle instead of the
        # bottom purely because we have no posts.
        return ScoreFactors(
            frequency=0.0,
            emotional_intensity=0.0,
            dissatisfaction=0.0,
            novelty=0.5,
        )

    n = len(posts)
    corpus_parts: list[str] = []
    for p in posts:
        title = p.title or ""
        body = p.body or ""
        corpus_parts.append(f"{title}\n{body}")
    corpus = "\n".join(corpus_parts)

    # --- 1. frequency ---
    # More mentions -> higher score. We use a log scale so a cluster of
    # 100 doesn't drown out a cluster of 5. Tuned so 1 post ≈ 0.0,
    # 10 posts ≈ 0.5, 100+ posts ≈ 0.95.
    frequency = _log_scale(n, mid=10.0)

    # --- 2. emotional_intensity ---
    # Fraction of posts that contain at least one frustration cue.
    frus_re = _cue_pattern(_FRUSTRATION_CUES)
    frus_posts = sum(
        1 for p in posts if frus_re.search(f"{p.title}\n{p.body or ''}")
    )
    emotional_intensity = frus_posts / n if n else 0.0
    # Boost a bit if many cues per post.
    cue_density = len(frus_re.findall(corpus)) / max(n, 1)
    emotional_intensity = min(
        1.0, emotional_intensity + min(cue_density / 10.0, 0.2)
    )

    # --- 3. dissatisfaction ---
    # Fraction of posts that signal "I tried existing solutions and
    # they failed". Different from raw frustration: this is specifically
    # about *switching away from* something.
    diss_re = _cue_pattern(_DIS_SATISFACTION_CUES)
    diss_posts = sum(
        1 for p in posts if diss_re.search(f"{p.title}\n{p.body or ''}")
    )
    dissatisfaction = diss_posts / n if n else 0.0
    # Boost slightly for question cues — people asking questions are
    # implicitly dissatisfied with what's available.
    q_re = _cue_pattern(_QUESTION_CUES)
    q_posts = sum(
        1 for p in posts if q_re.search(f"{p.title}\n{p.body or ''}")
    )
    dissatisfaction = min(
        1.0, dissatisfaction + (q_posts / n if n else 0.0) * 0.2
    )

    # --- 8. novelty ---
    # Higher novelty = fewer competitor mentions. We invert:
    # novelty = 1 - normalized_competitor_density.
    comp_re = _cue_pattern(_COMPETITOR_CUES)
    comp_hits = len(comp_re.findall(corpus))
    # Each post contributes at most ~1 competitor mention in our
    # normalized scale.
    comp_density = comp_hits / max(n, 1)
    novelty = max(0.0, 1.0 - min(comp_density, 1.0))

    return ScoreFactors(
        frequency=frequency,
        emotional_intensity=emotional_intensity,
        dissatisfaction=dissatisfaction,
        novelty=novelty,
    )


def _log_scale(n: int, mid: float = 10.0) -> float:
    """Log-scaled value in `[0, 1]` with `mid` mapped to ~0.5.

    Uses `log1p(n) / log1p(mid * 4)` so n=mid*4 ≈ 1.0. We cap at 1.0
    so a huge cluster doesn't blow past the scale.
    """
    if n <= 0:
        return 0.0
    import math

    return min(1.0, math.log1p(n) / math.log1p(mid * 4))