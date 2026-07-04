"""Tests for the Reddit collector — PRAW interactions are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from founder_radar.collectors.reddit import RedditCollector


def _submission(
    sid: str,
    title: str,
    *,
    body: str = "",
    score: int = 10,
    comments: int = 3,
    subreddit: str = "entrepreneur",
    author: str = "op_user",
) -> SimpleNamespace:
    """Build a fake PRAW submission with the attributes we read."""
    return SimpleNamespace(
        id=sid,
        title=title,
        selftext=body,
        # PRAW's `submission.author` is a `Redditor` object whose `str()` is the
        # username. In our mock we just pass the username as a plain string;
        # `str(submission.author)` then yields the username, exactly as
        # production code expects.
        author=author if author else None,
        url=f"https://reddit.com/r/{subreddit}/comments/{sid}",
        score=score,
        num_comments=comments,
        created_utc=1_700_000_000.0,
        permalink=f"/r/{subreddit}/comments/{sid}",
    )


def _fake_subreddit(submissions: list[SimpleNamespace]) -> SimpleNamespace:
    """Build a fake subreddit whose `.new()` returns the given submissions."""
    sub = SimpleNamespace()
    sub.new = lambda limit: iter(submissions[:limit])
    return sub


def _fake_reddit(subreddits: dict[str, SimpleNamespace]) -> SimpleNamespace:
    """Build a fake PRAW client whose `.subreddit(name)` returns sub mocks."""
    reddit = SimpleNamespace()
    reddit.subreddit = lambda name: subreddits[name]
    reddit.read_only = False
    return reddit


def test_collect_yields_raw_posts(tmp_settings) -> None:
    """`collect()` should call PRAW and translate submissions to RawPosts."""
    subs = {
        "entrepreneur": _fake_subreddit([
            _submission("abc", "Need help", body="body"),
            _submission("def", "Question", body="another body"),
        ]),
    }
    fake_reddit = _fake_reddit(subs)

    with patch.object(RedditCollector, "_client", return_value=fake_reddit):
        collector = RedditCollector(tmp_settings)
        posts = list(collector.collect(categories=["entrepreneur"], limit_per_category=5))

    assert len(posts) == 2
    p0 = posts[0]
    assert p0.source == "reddit"
    assert p0.external_id == "abc"
    assert p0.title == "Need help"
    assert p0.source_category == "entrepreneur"
    assert p0.score == 10
    assert p0.num_comments == 3
    assert p0.author == "op_user"
    assert p0.raw_json is not None
    assert "permalink" in p0.raw_json


def test_collect_without_categories_uses_settings(
    tmp_settings,
) -> None:
    """When no categories are passed, settings.subreddit_list is used."""
    fake_reddit = _fake_reddit({
        "test": _fake_subreddit([_submission("t1", "Title", body="body")]),
        "test2": _fake_subreddit([]),
    })
    with patch.object(RedditCollector, "_client", return_value=fake_reddit):
        collector = RedditCollector(tmp_settings)
        posts = list(collector.collect())

    assert len(posts) == 1
    assert posts[0].source_category == "test"


def test_collect_skips_invalid_submissions(tmp_settings) -> None:
    """Submissions without id/title should be skipped silently."""
    bad = SimpleNamespace(id=None, title="x", selftext="", author=None,
                          url="", score=0, num_comments=0,
                          created_utc=0.0, permalink="")
    subs = {"entrepreneur": _fake_subreddit([bad])}
    fake_reddit = _fake_reddit(subs)
    with patch.object(RedditCollector, "_client", return_value=fake_reddit):
        collector = RedditCollector(tmp_settings)
        assert list(collector.collect(categories=["entrepreneur"])) == []


def test_collect_raises_without_credentials(tmp_settings) -> None:
    """Missing credentials produce a clear, actionable error."""
    tmp_settings.reddit_client_id = ""
    tmp_settings.reddit_client_secret = ""
    collector = RedditCollector(tmp_settings)
    with pytest.raises(RuntimeError, match="Reddit credentials missing"):
        # Trigger _client() by calling collect().
        list(collector.collect(categories=["x"]))


def test_collect_continues_after_one_subreddit_fails(tmp_settings) -> None:
    """A PRAW error in one subreddit should not abort the whole run."""
    from prawcore.exceptions import PrawcoreException

    class BoomSub:
        def new(self, limit):  # noqa: ARG002
            raise PrawcoreException("rate limited")

    fake_reddit = SimpleNamespace(
        subreddit=lambda name: BoomSub() if name == "bad" else _fake_subreddit(
            [_submission("ok", "Title", body="body")]
        ),
        read_only=True,
    )
    with patch.object(RedditCollector, "_client", return_value=fake_reddit):
        collector = RedditCollector(tmp_settings)
        posts = list(
            collector.collect(categories=["bad", "good"], limit_per_category=5)
        )
    # Only the "good" subreddit yielded anything.
    assert len(posts) == 1
    assert posts[0].source_category == "good"


def test_source_name_is_reddit(tmp_settings) -> None:
    """Class attribute used by the registry must stay stable."""
    assert RedditCollector.source_name == "reddit"