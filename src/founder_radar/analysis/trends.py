"""Time-based trend tracking.

Classifies a cluster as one of:
  - "emerging"   recent activity is much higher than historical
  - "stable"     recent activity matches historical
  - "declining"  recent activity is much lower than historical
  - "recurring"  posts are spread across the timeline (sustained chatter)
  - "unknown"    not enough data to classify

The classification uses simple, deterministic heuristics over the
post timestamps. We compare a "recent" window (last 7 days) against a
"historical" window (prior 30 days). For recurring, we measure how
evenly posts are spaced.

This module is *pure*: takes posts, returns a `TrendReport`. No I/O.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from founder_radar.database.models import Post

logger = logging.getLogger(__name__)


# Trend classification labels.
TREND_EMERGING = "emerging"
TREND_STABLE = "stable"
TREND_DECLINING = "declining"
TREND_RECURRING = "recurring"
TREND_UNKNOWN = "unknown"


@dataclass(slots=True)
class TrendReport:
    """Output of the Trend Analyzer for one cluster.

    `growth_rate` is `posts_last_7d / max(posts_prior_30d, 1)`.
    `span_days` is the number of days between oldest and newest post.
    """

    trend: str = TREND_UNKNOWN
    posts_last_7d: int = 0
    posts_prior_30d: int = 0
    growth_rate: float = 0.0
    span_days: int = 0
    posts_per_week: float = 0.0

    @property
    def label(self) -> str:
        """Human-readable label like 'EMERGING (3.5x growth)'."""
        if self.trend == TREND_EMERGING:
            return f"EMERGING ({self.growth_rate:.1f}x recent growth)"
        if self.trend == TREND_DECLINING:
            return f"DECLINING ({self.growth_rate:.2f}x recent activity)"
        if self.trend == TREND_RECURRING:
            return f"RECURRING (sustained over {self.span_days} days)"
        if self.trend == TREND_STABLE:
            return "STABLE"
        return "UNKNOWN"


# Tunable thresholds.
_EMERGING_GROWTH_FACTOR = 2.0       # last-7d / prior-30d
_DECLINING_GROWTH_FACTOR = 0.5      # last-7d / prior-30d
_RECURRING_MIN_SPAN_DAYS = 21       # at least 3 weeks of history
_RECURRING_MIN_WEEKS_WITH_POSTS = 2 # posts in at least 2 different weeks
_MIN_POSTS_FOR_TREND = 3            # can't classify a cluster of 1-2 posts
_MIN_BASELINE_FOR_EMERGING = 2      # need >= 2 historical posts for "emerging"


def run_trend_analysis(
    posts: Iterable["Post"],
    *,
    now: datetime | None = None,
) -> TrendReport:
    """Classify a cluster's time trend.

    Args:
        posts: Iterable of `Post` rows. We read `created_at`. Posts
            without `created_at` are skipped.
        now: Override the "current" time — useful for deterministic
            tests. Defaults to naive UTC.

    Returns:
        A `TrendReport` with classification + supporting stats.
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)

    timestamps = [
        p.created_at for p in posts
        if p.created_at is not None
    ]
    timestamps.sort()

    if len(timestamps) < _MIN_POSTS_FOR_TREND:
        return TrendReport(trend=TREND_UNKNOWN)

    oldest = timestamps[0]
    newest = timestamps[-1]
    span_days = max((newest - oldest).days, 0)

    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    posts_last_7d = sum(1 for t in timestamps if t >= seven_days_ago)
    posts_prior_30d = sum(
        1 for t in timestamps if thirty_days_ago <= t < seven_days_ago
    )

    weeks = max(span_days / 7.0, 1.0 / 7.0)
    posts_per_week = len(timestamps) / weeks

    growth_rate = posts_last_7d / max(posts_prior_30d, 1)

    trend = _classify(
        timestamps=timestamps,
        span_days=span_days,
        posts_last_7d=posts_last_7d,
        posts_prior_30d=posts_prior_30d,
        growth_rate=growth_rate,
        now=now,
    )

    return TrendReport(
        trend=trend,
        posts_last_7d=posts_last_7d,
        posts_prior_30d=posts_prior_30d,
        growth_rate=growth_rate,
        span_days=span_days,
        posts_per_week=posts_per_week,
    )


def _classify(
    *,
    timestamps: list[datetime],
    span_days: int,
    posts_last_7d: int,
    posts_prior_30d: int,
    growth_rate: float,
    now: datetime,
) -> str:
    """Pick a label. Order matters: emerging beats recurring.

    Heuristics:
      - < _MIN_POSTS_FOR_TREND -> unknown (handled by caller).
      - Emerging: recent >= 2 AND prior >= _MIN_BASELINE_FOR_EMERGING
        AND growth_rate >= _EMERGING_GROWTH_FACTOR. Baseline
        requirement avoids false "emerging" on brand-new topics
        (where growth rate is infinite but uninformative).
      - Declining: recent == 0 AND growth_rate <= _DECLINING_GROWTH_FACTOR.
      - Recurring: span >= 21 days AND >= 2 distinct weeks with posts.
      - Else: stable.
    """
    # Emerging: requires both a real recent signal AND a baseline.
    if (
        posts_last_7d >= 2
        and posts_prior_30d >= _MIN_BASELINE_FOR_EMERGING
        and growth_rate >= _EMERGING_GROWTH_FACTOR
    ):
        return TREND_EMERGING

    # Declining: recent activity dried up.
    if posts_last_7d == 0 and growth_rate <= _DECLINING_GROWTH_FACTOR:
        return TREND_DECLINING

    # Recurring: long history with steady activity.
    if span_days >= _RECURRING_MIN_SPAN_DAYS:
        weeks_with_posts = _count_distinct_post_weeks(timestamps)
        if weeks_with_posts >= _RECURRING_MIN_WEEKS_WITH_POSTS:
            return TREND_RECURRING

    return TREND_STABLE


def _count_distinct_post_weeks(timestamps: list[datetime]) -> int:
    """Count the number of distinct ISO weeks that contain at least one post."""
    weeks: set[tuple[int, int]] = set()
    for t in timestamps:
        iso_year, iso_week, _ = t.isocalendar()
        weeks.add((iso_year, iso_week))
    return len(weeks)


# -------------------------------------------------------------------------
# Per-cluster convenience (used by CLI to rank clusters by trend)
# -------------------------------------------------------------------------
def classify_trend_simple(posts: Iterable["Post"]) -> str:
    """Return just the trend label — convenience for callers that don't
    need the full report.
    """
    return run_trend_analysis(posts).trend


# Helper: distance-decayed score so a cluster's "hotness" decays
# smoothly over time. Useful if we ever want a numeric hotness column
# in addition to the categorical trend.
def recency_score(
    posts: Iterable["Post"],
    *,
    now: datetime | None = None,
    half_life_days: float = 14.0,
) -> float:
    """Compute a [0, 1] "hotness" score from exponential decay of post ages.

    Each post contributes `0.5 ** (age_days / half_life_days)`. The
    total is normalized by the number of posts so the score is the
    *average* recency across the cluster — clusters with sustained
    activity stay warm, clusters with one-off bursts cool down.
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    total = 0.0
    count = 0
    for p in posts:
        if p.created_at is None:
            continue
        age_days = max((now - p.created_at).total_seconds() / 86400.0, 0.0)
        total += math.pow(0.5, age_days / half_life_days)
        count += 1
    if count == 0:
        return 0.0
    return total / count