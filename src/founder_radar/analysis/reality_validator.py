"""Reality Validation Layer (Phase 3.5).

Pure, deterministic classifier that maps a cluster's evidence into one
of four viability labels:

  - "saturated"      strong known competitors + many mentions -> low value
  - "competitive"    some competitors exist + people complain about them
                     -> fragmented market, medium opportunity
  - "underserved"    few/no competitors + clear pain -> high opportunity
  - "unknown"        data is too thin to classify

This module is *intentionally* separate from `reality_check.py`.
  - `reality_check` measures: how saturated is the market?
  - `reality_validator` interprets: what does that saturation mean for
    *us* as builders? (saturated / competitive / underserved)

The two views coexist:
  - ranking view (weighted_score from scoring.py) -> "what looks promising"
  - reality view (this module) -> "what is actually viable"

We never merge the two; the CLI surfaces them separately.

INVARIANTS — locked in by tests/test_reality_validator.py:

  status == "saturated"    -> competitor_strength >= 0.55
                               AND distinct_competitor_count >= 3

  status == "competitive"  -> competitor_strength >= 0.15
                               AND dissatisfaction_hits >= 2

  status == "underserved"  -> competitor_strength < 0.35
                               AND pain_density >= 0.30

  status == "unknown"      -> none of the above matched

Design rules:
  - Deterministic only. No LLM calls.
  - Pure function: takes posts + competitor info, returns RealityAssessment.
  - All evidence is preserved as human-readable strings so the CLI can
    show the reasoning, not just the label.
  - Raw signal values (pain_density, dissatisfaction_hits,
    distinct_competitor_count) are exposed on the dataclass so the
    audit pass can verify thresholds were met.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from founder_radar.analysis.reality_check import KNOWN_COMPETITORS

if TYPE_CHECKING:
    from founder_radar.database.models import Post

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Public constants
# -------------------------------------------------------------------------
STATUS_SATURATED = "saturated"
STATUS_COMPETITIVE = "competitive"
STATUS_UNDERSERVED = "underserved"
STATUS_UNKNOWN = "unknown"

ALL_STATUSES = (
    STATUS_SATURATED,
    STATUS_COMPETITIVE,
    STATUS_UNDERSERVED,
    STATUS_UNKNOWN,
)

# Pain + dissatisfaction cues. Duplicated from scoring.py to avoid
# pulling in its private members; the lexica are small and stable.
_FRUSTRATION_CUES = (
    "hate", "sucks", "broken", "terrible", "awful", "frustrated",
    "frustrating", "annoying", "annoyed", "useless", "doesn't work",
    "does not work", "impossible", "ridiculous", "waste", "garbage",
    "nightmare", "pain", "painful", "stupid",
)

_DIS_SATISFACTION_CUES = (
    "tried everything", "nothing works", "no good alternative",
    "no good option", "switched from", "switching from", "left",
    "cancelled", "unsubscribed", "moved away from", "fed up with",
    "tired of", "gave up on", "abandoned",
)

_frustration_re = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _FRUSTRATION_CUES) + r")\b",
    re.IGNORECASE,
)
_dissatisfaction_re = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _DIS_SATISFACTION_CUES) + r")\b",
    re.IGNORECASE,
)

# Thresholds. Tuned by inspection; tweak as the system gets feedback.
# These names are exposed as the basis for the invariant tests.
SATURATION_STRENGTH_THRESHOLD = 0.55    # min competitor_strength for "saturated"
SATURATION_COUNT_THRESHOLD = 3          # min distinct competitor count
COMPETITIVE_STRENGTH_LOWER = 0.15       # min competitor_strength for "competitive"
DISSATISFACTION_HITS_THRESHOLD = 2      # total dissatisfaction cue hits
UNDERSERVED_STRENGTH_CEILING = 0.35     # max competitor_strength for "underserved"
PAIN_DENSITY_THRESHOLD = 0.30           # min pain cue density for "underserved"
DENSITY_SATURATION = 1.5                 # mentions/post at which density_score = 1.0

# Backwards-compat aliases (private constants were renamed in the
# calibration pass; tests imported them with the old names).
_SATURATION_STRENGTH_THRESHOLD = SATURATION_STRENGTH_THRESHOLD
_SATURATION_COUNT_THRESHOLD = SATURATION_COUNT_THRESHOLD
_COMPETITIVE_STRENGTH_LOWER = COMPETITIVE_STRENGTH_LOWER
_DISSATISFACTION_HITS_THRESHOLD = DISSATISFACTION_HITS_THRESHOLD
_UNDERSERVED_STRENGTH_CEILING = UNDERSERVED_STRENGTH_CEILING
_PAIN_DENSITY_THRESHOLD = PAIN_DENSITY_THRESHOLD
_DENSITY_SATURATION = DENSITY_SATURATION


# -------------------------------------------------------------------------
# Public dataclass
# -------------------------------------------------------------------------
@dataclass(slots=True)
class RealityAssessment:
    """Classification of how viable an opportunity actually is.

    `status` is one of STATUS_SATURATED / STATUS_COMPETITIVE /
    STATUS_UNDERSERVED / STATUS_UNKNOWN.

    `saturation_confidence` is `[0, 1]`: how strongly the available
    evidence supports the chosen status. Higher = more confident.

    `evidence` is a list of short human-readable strings that explain
    why we picked this status. Surfaced verbatim by `founder-radar
    reality` so the user can audit the reasoning.

    `competitor_strength_estimate` is `[0, 1]`: a numeric summary of
    the competitor signal (count + lexicon match + mention density).
    Independent of `Opportunity.saturation_score` (which uses a slightly
    different formula).

    --- Phase 3.5 calibration pass additions ---

    `pain_density`, `dissatisfaction_hits`, `distinct_competitor_count`:
    raw signals surfaced so the CLI can show *what* the classifier saw.
    Without these the user has to take the status on faith.

    `reason`: single-sentence explanation of why this status was chosen.
    Includes the threshold values used, so a user can audit whether a
    borderline case missed (or passed) the cut.
    """

    status: str = STATUS_UNKNOWN
    saturation_confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    competitor_strength_estimate: float = 0.0

    # Raw signals — exposed for calibration / audit.
    pain_density: float = 0.0
    dissatisfaction_hits: int = 0
    distinct_competitor_count: int = 0

    # Human-readable explanation of the classification decision.
    reason: str = ""

    @property
    def is_viable(self) -> bool:
        """True for the two states we treat as 'real opportunities'.

        Saturated -> no (don't build into a crowded market).
        Competitive -> yes (fragmented market, room for a better solution).
        Underserved -> yes (clear pain, no good solution).
        Unknown -> no (insufficient data).
        """
        return self.status in (STATUS_COMPETITIVE, STATUS_UNDERSERVED)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def assess_reality(
    posts: Iterable["Post"],
    *,
    competitors: list[str] | None = None,
    distinct_competitor_count: int | None = None,
    competitor_mention_count: int | None = None,
) -> RealityAssessment:
    """Classify an opportunity into a viability status.

    Args:
        posts: All source posts in the cluster.
        competitors: Pre-extracted distinct competitor names (from
            RealityCheck). When None, the validator re-derives them from
            the posts so this function is callable independently.
        distinct_competitor_count: Override when known (cheaper than
            recounting). When None, derived from `competitors`.
        competitor_mention_count: Override when known. When None,
            derived from the posts.

    Returns:
        A populated `RealityAssessment`. Never raises — empty / bad
        input returns `status="unknown"` with empty evidence.
    """
    posts = list(posts)

    # Resolve competitor info. If the caller passed nothing, derive it.
    if competitors is None:
        from founder_radar.analysis.reality_check import run_reality_check
        rc = run_reality_check(posts)
        competitors = rc.competitors
        distinct_competitor_count = rc.distinct_competitor_count
        competitor_mention_count = rc.competitor_mention_count
    else:
        competitors = list(competitors)
        if distinct_competitor_count is None:
            distinct_competitor_count = len(competitors)
        if competitor_mention_count is None:
            # Fallback: count "mentions" as number of posts that contain
            # at least one competitor name. Not perfect but consistent.
            competitor_mention_count = _count_competitor_mentions(posts, competitors)

    # Compute the signals.
    pain_hits = _count_pain_cues(posts)
    dissatisfaction_hits = _count_dissatisfaction_cues(posts)
    n_posts = max(len(posts), 1)
    pain_density = pain_hits / n_posts
    dissatisfaction_density = dissatisfaction_hits / n_posts

    competitor_strength = _competitor_strength(
        distinct_competitor_count=distinct_competitor_count,
        competitor_mention_count=competitor_mention_count,
        n_posts=n_posts,
        competitors=competitors,
    )

    # Classify.
    status, confidence, reason = _classify(
        competitor_strength=competitor_strength,
        distinct_competitor_count=distinct_competitor_count,
        dissatisfaction_hits=dissatisfaction_hits,
        pain_density=pain_density,
        n_posts=n_posts,
    )

    # Build evidence.
    evidence = _build_evidence(
        competitors=competitors,
        distinct_competitor_count=distinct_competitor_count,
        competitor_mention_count=competitor_mention_count,
        pain_hits=pain_hits,
        dissatisfaction_hits=dissatisfaction_hits,
        pain_density=pain_density,
        n_posts=n_posts,
    )

    return RealityAssessment(
        status=status,
        saturation_confidence=confidence,
        evidence=evidence,
        competitor_strength_estimate=competitor_strength,
        pain_density=pain_density,
        dissatisfaction_hits=dissatisfaction_hits,
        distinct_competitor_count=distinct_competitor_count,
        reason=reason,
    )


# -------------------------------------------------------------------------
# Signal helpers (private)
# -------------------------------------------------------------------------
def _count_pain_cues(posts: list["Post"]) -> int:
    """Total frustration cue hits across all posts."""
    total = 0
    for p in posts:
        text = f"{p.title or ''}\n{p.body or ''}"
        total += len(_frustration_re.findall(text))
    return total


def _count_dissatisfaction_cues(posts: list["Post"]) -> int:
    """Total dissatisfaction cue hits across all posts."""
    total = 0
    for p in posts:
        text = f"{p.title or ''}\n{p.body or ''}"
        total += len(_dissatisfaction_re.findall(text))
    return total


def _count_competitor_mentions(
    posts: list["Post"], competitors: list[str],
) -> int:
    """Count posts that mention at least one competitor name (substring match)."""
    if not competitors:
        return 0
    lowered = [c.lower() for c in competitors]
    count = 0
    for p in posts:
        text = f"{p.title or ''}\n{p.body or ''}".lower()
        if any(name in text for name in lowered):
            count += 1
    return count


def _competitor_strength(
    *,
    distinct_competitor_count: int,
    competitor_mention_count: int,
    n_posts: int,
    competitors: list[str],
) -> float:
    """Compute `[0, 1]` strength of the competitor signal.

    Formula (weighted):
      - count_score       (50%) — log-scaled distinct competitor count
      - known_ratio       (30%) — fraction of competitors in KNOWN_COMPETITORS
      - density_score     (20%) — mentions per post, normalized

    Kept simple and readable. Tuned so 0 competitors = 0.0 and 6+
    well-known competitors with mentions in most posts approaches 1.0.
    """
    # count_score: log(1 + distinct) / log(1 + 6) — saturates around 6.
    count_score = min(1.0, math.log1p(distinct_competitor_count) / math.log1p(6))

    # known_ratio: fraction of competitors that match the lexicon.
    if competitors:
        known_count = sum(
            1 for c in competitors if c.lower() in {k.lower() for k in KNOWN_COMPETITORS}
        )
        known_ratio = known_count / max(len(competitors), 1)
    else:
        known_ratio = 0.0

    # density_score: mentions per post, saturating at DENSITY_SATURATION.
    density = competitor_mention_count / max(n_posts, 1)
    density_score = min(1.0, density / DENSITY_SATURATION)

    return (
        0.5 * count_score
        + 0.3 * known_ratio
        + 0.2 * density_score
    )


def _classify(
    *,
    competitor_strength: float,
    distinct_competitor_count: int,
    dissatisfaction_hits: int,
    pain_density: float,
    n_posts: int,
) -> tuple[str, float, str]:
    """Pick the status, confidence, and a one-line reason.

    INVARIANTS (locked in by tests/test_reality_validator.py):
      - status == "saturated"    -> competitor_strength >= SATURATION_STRENGTH_THRESHOLD
                                     AND distinct_competitor_count >= SATURATION_COUNT_THRESHOLD
      - status == "competitive"  -> competitor_strength >= COMPETITIVE_STRENGTH_LOWER
                                     AND dissatisfaction_hits >= DISSATISFACTION_HITS_THRESHOLD
      - status == "underserved"  -> competitor_strength < UNDERSERVED_STRENGTH_CEILING
                                     AND pain_density >= PAIN_DENSITY_THRESHOLD
      - status == "unknown"      -> none of the above matched
    """
    # Saturated: high competitor strength + enough distinct competitors.
    if (
        competitor_strength >= SATURATION_STRENGTH_THRESHOLD
        and distinct_competitor_count >= SATURATION_COUNT_THRESHOLD
    ):
        confidence = min(1.0, competitor_strength)
        reason = (
            f"Saturated: competitor_strength={competitor_strength:.2f} "
            f">= {SATURATION_STRENGTH_THRESHOLD} AND "
            f"distinct_competitors={distinct_competitor_count} "
            f">= {SATURATION_COUNT_THRESHOLD}"
        )
        return STATUS_SATURATED, confidence, reason

    # Competitive: some competitors + people are unhappy with them.
    if (
        competitor_strength >= COMPETITIVE_STRENGTH_LOWER
        and dissatisfaction_hits >= DISSATISFACTION_HITS_THRESHOLD
    ):
        diss_score = min(1.0, dissatisfaction_hits / 5.0)
        confidence = min(1.0, 0.6 * competitor_strength + 0.4 * diss_score)
        reason = (
            f"Competitive: competitor_strength={competitor_strength:.2f} "
            f">= {COMPETITIVE_STRENGTH_LOWER} AND "
            f"dissatisfaction_hits={dissatisfaction_hits} "
            f">= {DISSATISFACTION_HITS_THRESHOLD} (fragmented market)"
        )
        return STATUS_COMPETITIVE, confidence, reason

    # Underserved: few/no competitors + real pain signals.
    if (
        competitor_strength < UNDERSERVED_STRENGTH_CEILING
        and pain_density >= PAIN_DENSITY_THRESHOLD
    ):
        confidence = min(1.0, pain_density)
        reason = (
            f"Underserved: competitor_strength={competitor_strength:.2f} "
            f"< {UNDERSERVED_STRENGTH_CEILING} AND "
            f"pain_density={pain_density:.2f} "
            f">= {PAIN_DENSITY_THRESHOLD} (clear pain, no good solution)"
        )
        return STATUS_UNDERSERVED, confidence, reason

    # Data too thin or signals don't agree. We split into two sub-cases
    # so the reason string is informative.
    if competitor_strength >= COMPETITIVE_STRENGTH_LOWER:
        confidence = 0.3
        reason = (
            f"Unknown: competitor_strength={competitor_strength:.2f} "
            f">= {COMPETITIVE_STRENGTH_LOWER} (some competitors) but "
            f"dissatisfaction_hits={dissatisfaction_hits} "
            f"< {DISSATISFACTION_HITS_THRESHOLD} (no complaints) — "
            f"can't confirm if market is open for a better solution"
        )
    elif pain_density < PAIN_DENSITY_THRESHOLD:
        confidence = 0.2
        reason = (
            f"Unknown: competitor_strength={competitor_strength:.2f} "
            f"< {COMPETITIVE_STRENGTH_LOWER} (few/no competitors) AND "
            f"pain_density={pain_density:.2f} "
            f"< {PAIN_DENSITY_THRESHOLD} (insufficient pain signals)"
        )
    else:
        # Few competitors + enough pain — shouldn't reach here given
        # the underserved branch above. If it does, fall back to unknown.
        confidence = 0.2
        reason = (
            f"Unknown: signals mixed across {n_posts} post(s) — "
            f"competitor_strength={competitor_strength:.2f}, "
            f"pain_density={pain_density:.2f}, "
            f"dissatisfaction_hits={dissatisfaction_hits}"
        )
    return STATUS_UNKNOWN, confidence, reason


def _build_evidence(
    *,
    competitors: list[str],
    distinct_competitor_count: int,
    competitor_mention_count: int,
    pain_hits: int,
    dissatisfaction_hits: int,
    pain_density: float,
    n_posts: int,
) -> list[str]:
    """Build the human-readable evidence list for the assessment.

    Each line is short and standalone — the CLI prints them as bullets.
    Order matters: most relevant signals first.

    The pain-density line is conditional: when the density is below the
    underserved threshold, we explicitly say so — so an "underserved"
    classification always has clear pain evidence above it. This is the
    invariant the audit pass enforces.
    """
    lines: list[str] = []

    # Competitor evidence.
    if distinct_competitor_count == 0:
        lines.append("0 distinct competitors mentioned")
    else:
        named = ", ".join(competitors[:5])
        if distinct_competitor_count > 5:
            named += f", ... (+{distinct_competitor_count - 5} more)"
        lines.append(
            f"{distinct_competitor_count} distinct competitor(s): {named}"
        )
        # Highlight well-known ones separately.
        known_hits = [
            c for c in competitors
            if c.lower() in {k.lower() for k in KNOWN_COMPETITORS}
        ]
        if known_hits:
            lines.append(
                f"{len(known_hits)} of them well-known SaaS names "
                f"({', '.join(known_hits[:5])})"
            )

    if competitor_mention_count and n_posts:
        lines.append(
            f"Competitor mentions in {competitor_mention_count} of {n_posts} post(s)"
        )

    # Pain evidence. We always show the density so a user can verify
    # the underserved invariant (>= PAIN_DENSITY_THRESHOLD when
    # classified underserved).
    if pain_hits == 0:
        lines.append(
            f"No frustration cues detected "
            f"(pain_density=0.00, threshold={PAIN_DENSITY_THRESHOLD:.2f})"
        )
    else:
        threshold_note = (
            f" >= {PAIN_DENSITY_THRESHOLD:.2f}"
            if pain_density >= PAIN_DENSITY_THRESHOLD
            else f" (< {PAIN_DENSITY_THRESHOLD:.2f} — below underserved threshold)"
        )
        lines.append(
            f"Frustration cues: {pain_hits} hit(s) across {n_posts} post(s) "
            f"(pain_density={pain_density:.2f}{threshold_note})"
        )

    # Dissatisfaction evidence.
    if dissatisfaction_hits == 0:
        lines.append(
            f"No 'switched from' / 'cancelled' / 'gave up' signals "
            f"(threshold={DISSATISFACTION_HITS_THRESHOLD})"
        )
    else:
        lines.append(
            f"Dissatisfaction cues: {dissatisfaction_hits} hit(s) "
            f"(people unhappy with existing solutions)"
        )

    return lines