"""Tests for the Trend Analyzer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from founder_radar.analysis.trends import (
    TREND_DECLINING,
    TREND_EMERGING,
    TREND_RECURRING,
    TREND_STABLE,
    TREND_UNKNOWN,
    TrendReport,
    classify_trend_simple,
    recency_score,
    run_trend_analysis,
)
from founder_radar.database.models import Post


def _post_at(dt: datetime, external_id: str = "x") -> Post:
    return Post(
        source="reddit",
        external_id=external_id,
        source_category="test",
        title="title",
        body="body",
        author="op",
        url=None,
        score=1,
        num_comments=1,
        created_at=dt,
        collected_at=dt,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# -------------------------------------------------------------------------
# Degenerate inputs
# -------------------------------------------------------------------------
def test_few_posts_returns_unknown() -> None:
    report = run_trend_analysis(
        [_post_at(_utcnow() - timedelta(days=2))],
        now=_utcnow(),
    )
    assert report.trend == TREND_UNKNOWN


def test_no_timestamps_returns_unknown() -> None:
    posts = [
        Post(source="r", external_id="x", source_category="t",
             title="t", body="b", created_at=None)
    ]
    assert run_trend_analysis(posts).trend == TREND_UNKNOWN


def test_few_recent_posts_no_history_returns_stable() -> None:
    """3 recent posts, no historical baseline -> stable (not emerging).

    Without a baseline, "growth rate" is mathematically infinite but
    conceptually meaningless — the topic could be brand new, not
    necessarily emerging.
    """
    now = _utcnow()
    posts = [
        _post_at(now - timedelta(days=i), f"p{i}")
        for i in range(3)
    ]
    report = run_trend_analysis(posts, now=now)
    assert report.trend == TREND_STABLE


# -------------------------------------------------------------------------
# Emerging detection
# -------------------------------------------------------------------------
def test_detects_emerging_trend() -> None:
    """Recent activity clearly higher than historical -> emerging."""
    now = _utcnow()
    posts = []
    # 3 historical posts (>7 days ago).
    for i in range(10, 13):
        posts.append(_post_at(now - timedelta(days=i), f"h{i}"))
    # 6 recent posts.
    for i in range(0, 6):
        posts.append(_post_at(now - timedelta(days=i), f"r{i}"))
    report = run_trend_analysis(posts, now=now)
    assert report.trend == TREND_EMERGING
    assert report.posts_last_7d == 6
    assert report.posts_prior_30d >= 3


def test_emerging_with_2x_growth_factor() -> None:
    now = _utcnow()
    posts = []
    for i in range(10, 13):
        posts.append(_post_at(now - timedelta(days=i), f"h{i}"))
    for i in range(0, 6):
        posts.append(_post_at(now - timedelta(days=i), f"r{i}"))
    report = run_trend_analysis(posts, now=now)
    assert report.trend == TREND_EMERGING
    assert report.growth_rate >= 2.0


def test_single_recent_post_is_not_emerging() -> None:
    """Emerging requires >= 2 recent posts AND >= 2 historical."""
    now = _utcnow()
    posts = [_post_at(now - timedelta(days=1), "p_recent")]
    for i in range(5):
        posts.append(_post_at(now - timedelta(days=10 + i), f"h{i}"))
    report = run_trend_analysis(posts, now=now)
    assert report.trend != TREND_EMERGING


# -------------------------------------------------------------------------
# Declining detection
# -------------------------------------------------------------------------
def test_detects_declining_trend() -> None:
    now = _utcnow()
    posts = [
        _post_at(now - timedelta(days=10 + i), f"p{i}")
        for i in range(5)
    ]
    assert run_trend_analysis(posts, now=now).trend == TREND_DECLINING


# -------------------------------------------------------------------------
# Recurring detection
# -------------------------------------------------------------------------
def test_detects_recurring_long_running_cluster() -> None:
    now = _utcnow()
    posts = []
    for week in range(6):
        dt = now - timedelta(days=week * 7 + 3)
        posts.append(_post_at(dt, f"p{week}"))
    report = run_trend_analysis(posts, now=now)
    assert report.trend == TREND_RECURRING
    assert report.span_days >= 21


# -------------------------------------------------------------------------
# Recency score helper
# -------------------------------------------------------------------------
def test_recency_score_higher_for_recent_posts() -> None:
    now = _utcnow()
    recent = [_post_at(now - timedelta(hours=12))]
    old = [_post_at(now - timedelta(days=30))]
    assert recency_score(recent, now=now) > recency_score(old, now=now)


def test_recency_score_zero_for_empty() -> None:
    assert recency_score([], now=_utcnow()) == 0.0


def test_recency_score_handles_missing_timestamps() -> None:
    posts = [
        Post(source="r", external_id="x", source_category="t",
             title="t", body="b", created_at=None),
    ]
    assert recency_score(posts, now=_utcnow()) == 0.0


# -------------------------------------------------------------------------
# classify_trend_simple convenience
# -------------------------------------------------------------------------
def test_classify_trend_simple_returns_label() -> None:
    now = _utcnow()
    posts = []
    for i in range(10, 13):
        posts.append(_post_at(now - timedelta(days=i), f"h{i}"))
    for i in range(0, 6):
        posts.append(_post_at(now - timedelta(days=i), f"r{i}"))
    assert classify_trend_simple(posts) == TREND_EMERGING


# -------------------------------------------------------------------------
# TrendReport fields and labels
# -------------------------------------------------------------------------
def test_trend_report_default_values() -> None:
    r = TrendReport()
    assert r.trend == TREND_UNKNOWN
    assert r.posts_last_7d == 0
    assert r.posts_prior_30d == 0
    assert r.growth_rate == 0.0
    assert r.span_days == 0


def test_trend_report_label_for_emerging() -> None:
    r = TrendReport(trend=TREND_EMERGING, growth_rate=3.5)
    assert "EMERGING" in r.label
    assert "3.5x" in r.label


def test_trend_report_label_for_recurring() -> None:
    r = TrendReport(trend=TREND_RECURRING, span_days=42)
    assert "RECURRING" in r.label
    assert "42 days" in r.label


def test_trend_report_label_for_unknown() -> None:
    assert TrendReport(trend=TREND_UNKNOWN).label == "UNKNOWN"