"""Tests for the processor layer."""

from __future__ import annotations

from founder_radar.collectors.base import RawPost
from founder_radar.processors.cleaner import Cleaner


def _raw(external_id: str, title: str, body: str = "") -> RawPost:
    return RawPost(
        source="reddit",
        external_id=external_id,
        source_category="entrepreneur",
        title=title,
        body=body,
        author="someone",
        url=None,
        score=0,
        num_comments=0,
    )


# All test strings are designed to be longer than the cleaner's default
# `min_body_length` (20 chars combined title+body) so they only get dropped
# when explicitly tested for short-ness.


def test_cleaner_drops_duplicates() -> None:
    posts = [
        _raw(
            "a",
            "Title A is about something important",
            "Body A is also reasonably long enough to pass the cleaner.",
        ),
        _raw(
            "a",
            "Title A is about something important",
            "Body A is also reasonably long enough to pass the cleaner.",
        ),  # exact duplicate
    ]
    cleaned = Cleaner().process(posts)
    assert len(cleaned) == 1


def test_cleaner_drops_short_posts() -> None:
    posts = [
        _raw("a", "Why", "?"),  # 1+1 = 2 chars, below min
        _raw(
            "b",
            "Real question for everyone here",
            "I have a real problem and need help finding my first customer.",
        ),
    ]
    cleaned = Cleaner(min_body_length=20).process(posts)
    assert len(cleaned) == 1
    assert cleaned[0].external_id == "b"


def test_cleaner_drops_shouting() -> None:
    posts = [
        _raw(
            "a",
            "WHY IS EVERYTHING BROKEN RIGHT NOW",
            "AAAAAAAA HELP ME FIGURE THIS OUT PLEASE I AM STUCK",
        ),
    ]
    cleaned = Cleaner().process(posts)
    assert cleaned == []


def test_cleaner_drops_emoji_storms() -> None:
    posts = [
        _raw("a", "🔥🔥🔥🔥🔥 win big money now click here for free crypto"),
    ]
    cleaned = Cleaner().process(posts)
    assert cleaned == []


def test_cleaner_keeps_normal_posts() -> None:
    posts = [
        _raw(
            "a",
            "How do you find your first paying customer?",
            "I built an MVP and have been struggling to convert signups.",
        ),
    ]
    cleaned = Cleaner().process(posts)
    assert len(cleaned) == 1


def test_cleaner_does_not_mutate_input() -> None:
    posts = [
        _raw(
            "a",
            "Title A is about something important",
            "Body A is also reasonably long enough to pass the cleaner.",
        ),
        _raw(
            "b",
            "Title B is also about something important",
            "Body B is also reasonably long enough to pass the cleaner.",
        ),
    ]
    original = list(posts)
    Cleaner().process(posts)
    assert posts == original


def test_cleaner_does_not_share_state_between_calls() -> None:
    """Two consecutive runs must each dedupe independently."""
    posts = [
        _raw(
            "a",
            "Title is long enough to pass the cleaner easily",
            "Body is also long enough to pass the cleaner easily.",
        ),
    ]
    Cleaner().process(posts)
    again = Cleaner().process(posts)
    assert len(again) == 1