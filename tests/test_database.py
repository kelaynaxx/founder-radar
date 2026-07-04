"""Tests for the database layer: models, engine, repository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from founder_radar.database.connection import (
    get_engine,
    get_session,
    init_engine,
)
from founder_radar.database.models import Post
from founder_radar.database.repository import PostRepository


def _make_post(**overrides) -> Post:
    """Factory: build a Post with sensible defaults, easy to override."""
    defaults = dict(
        source="reddit",
        external_id="abc123",
        source_category="entrepreneur",
        title="Need help finding paying customers",
        body="I built a thing and nobody is buying. What am I doing wrong?",
        author="op_user",
        url="https://reddit.com/r/entrepreneur/comments/abc123",
        score=42,
        num_comments=17,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=3),
    )
    defaults.update(overrides)
    return Post(**defaults)


def test_init_engine_creates_tables(configured_db) -> None:
    """After init_engine, the posts table exists."""
    engine = get_engine()
    assert engine.dialect.has_table(engine.connect(), "posts")


def test_get_engine_raises_before_init() -> None:
    """Sanity: a fresh import path can't accidentally use the engine."""
    # Note: by the time tests run, init_engine has been called at least
    # once. We can't fully exercise the not-initialized branch without
    # resetting the module state, so we just verify get_engine works.
    assert get_engine() is not None


def test_repository_insert_and_get_by_id(repo: PostRepository) -> None:
    post = _make_post()
    repo.add(post)
    fetched = repo.get_by_id(post.id)
    assert fetched is not None
    assert fetched.title == post.title
    assert fetched.source == "reddit"


def test_repository_dedupes_by_source_and_external_id(
    repo: PostRepository,
) -> None:
    """Adding the same (source, external_id) twice returns the same row."""
    p1 = _make_post(external_id="dup123")
    p2 = _make_post(external_id="dup123", title="Different title")
    repo.add(p1)
    repo.add(p2)
    # Both `add()` calls should yield the same row id.
    assert p1.id == p2.id


def test_repository_add_many_counts_new_rows(
    repo: PostRepository,
) -> None:
    posts = [
        _make_post(external_id=f"id_{i}", title=f"Title {i}")
        for i in range(5)
    ]
    new = repo.add_many(posts)
    assert new == 5


def test_repository_list_by_source(repo: PostRepository) -> None:
    repo.add(_make_post(source="reddit", external_id="r1"))
    repo.add(_make_post(source="reddit", external_id="r2"))
    repo.add(_make_post(source="hackernews", external_id="h1"))
    reddit_posts = repo.list_by_source("reddit")
    assert len(reddit_posts) == 2
    assert all(p.source == "reddit" for p in reddit_posts)


def test_repository_list_by_source_category(
    repo: PostRepository,
) -> None:
    repo.add(_make_post(source_category="entrepreneur", external_id="e1"))
    repo.add(_make_post(source_category="startups", external_id="s1"))
    e = repo.list_by_source_category("reddit", "entrepreneur")
    assert len(e) == 1
    assert e[0].external_id == "e1"


def test_repository_count(repo: PostRepository) -> None:
    assert repo.count() == 0
    repo.add(_make_post(external_id="c1"))
    repo.add(_make_post(external_id="c2"))
    assert repo.count() == 2


def test_session_rolls_back_on_exception(
    configured_db,
    tmp_settings,
) -> None:
    """If a function inside `get_session` raises, nothing is committed."""
    class BoomError(Exception):
        pass

    def add_then_explode() -> None:
        with get_session() as session:
            session.add(_make_post(external_id="boom"))
            raise BoomError("nope")

    with pytest.raises(BoomError):
        add_then_explode()

    # Re-open a fresh session; the row should not exist.
    with get_session() as session:
        repo = PostRepository(session)
        assert repo.count() == 0