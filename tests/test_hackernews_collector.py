"""Tests for the Hacker News collector.

All HTTP calls are mocked via httpx.MockTransport. No network access.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from founder_radar.collectors.hackernews import (
    HN_API_BASE,
    HN_STORY_TYPES,
    HackerNewsCollector,
    _first_meaningful_line,
    _safe_int,
)
from founder_radar.collectors.base import RawPost


# -------------------------------------------------------------------------
# Test fixtures
# -------------------------------------------------------------------------
def _mock_client(handler) -> httpx.Client:
    """Wrap a request handler in an httpx.Client with a mock transport."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _story_payload(
    item_id: int,
    *,
    title: str = "Sample",
    by: str = "alice",
    time: int = 1_700_000_000,
    score: int = 100,
    descendants: int = 5,
    url: str | None = "https://example.com/x",
    text: str | None = None,
    kids: list | None = None,
    deleted: bool = False,
    dead: bool = False,
    type_: str = "story",
) -> dict:
    """Build a representative HN item dict."""
    item = {
        "id": item_id,
        "type": type_,
        "by": by,
        "time": time,
        "title": title,
        "score": score,
        "descendants": descendants,
    }
    if url is not None:
        item["url"] = url
    if text is not None:
        item["text"] = text
    if kids is not None:
        item["kids"] = kids
    if deleted:
        item["deleted"] = True
    if dead:
        item["dead"] = True
    return item


def _make_handler(items: dict[int, dict], story_lists: dict[str, list] | None = None):
    """Build an httpx handler that returns canned story lists and items.

    Args:
        items: Mapping of HN item id -> item dict.
        story_lists: Mapping of story_type name -> list of item ids.
    """
    story_lists = story_lists or {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        # List endpoint: .../v0/{story_type}.json
        for st in HN_STORY_TYPES:
            if path == f"/v0/{st}.json":
                payload = story_lists.get(st, [])
                return httpx.Response(200, json=payload)
        # Item endpoint: .../v0/item/{id}.json
        if path.startswith("/v0/item/") and path.endswith(".json"):
            try:
                item_id = int(path[len("/v0/item/"):-len(".json")])
            except ValueError:
                return httpx.Response(404, json={"error": "bad id"})
            if item_id in items:
                return httpx.Response(200, json=items[item_id])
            return httpx.Response(404, json=None)
        return httpx.Response(404, json=None)

    return handler


# -------------------------------------------------------------------------
# Source name + shape
# -------------------------------------------------------------------------
def test_source_name_is_hackernews(tmp_settings) -> None:
    """The registry looks up collectors by `source_name`."""
    c = HackerNewsCollector(tmp_settings)
    assert c.source_name == "hackernews"


def test_default_story_types_match_documented_set() -> None:
    """Public constant must match the brief's requirements."""
    assert "topstories" in HN_STORY_TYPES
    assert "newstories" in HN_STORY_TYPES
    assert "askstories" in HN_STORY_TYPES
    assert "showstories" in HN_STORY_TYPES
    assert "jobstories" in HN_STORY_TYPES


# -------------------------------------------------------------------------
# Construction: no network required, no credentials required
# -------------------------------------------------------------------------
def test_construction_does_not_make_http_requests(tmp_settings) -> None:
    """Just instantiating the collector must not touch the network.

    This is the contract that lets the CLI dispatch on source=hackernews
    without Reddit credentials.
    """
    # If construction triggers an HTTP call, httpx would raise
    # ConnectError. We assert by simply instantiating and confirming
    # no client was created.
    c = HackerNewsCollector(tmp_settings)
    assert c._session is None  # type: ignore[attr-defined]


# -------------------------------------------------------------------------
# Happy path
# -------------------------------------------------------------------------
def test_collect_yields_raw_posts_for_stories(tmp_settings) -> None:
    items = {
        1: _story_payload(1, title="Story One", score=200, descendants=10),
        2: _story_payload(2, title="Story Two", score=50, descendants=3),
        3: _story_payload(3, title="Story Three", score=10, descendants=1),
    }
    handler = _make_handler(
        items,
        story_lists={"topstories": [1, 2, 3]},
    )
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=3
        ))
    assert len(posts) == 3
    assert all(isinstance(p, RawPost) for p in posts)
    assert posts[0].external_id == "1"
    assert posts[0].title == "Story One"
    assert posts[0].source == "hackernews"
    assert posts[0].source_category == "topstories"


def test_collect_maps_hn_item_into_rawpost_shape(tmp_settings) -> None:
    """Every documented HN field should land in the right RawPost slot."""
    item = _story_payload(
        42,
        title="Mapping test",
        by="bob",
        time=1_700_000_000,
        score=123,
        descendants=7,
        url="https://example.com/article",
    )
    handler = _make_handler(items={42: item}, story_lists={"topstories": [42]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=1
        ))
    p = posts[0]
    assert p.title == "Mapping test"
    assert p.author == "bob"
    assert p.url == "https://example.com/article"
    assert p.score == 123
    assert p.num_comments == 7
    # Unix 1700000000 -> 2023-11-14 22:13:20 UTC
    assert p.created_at == datetime(2023, 11, 14, 22, 13, 20)
    assert p.source == "hackernews"
    assert p.external_id == "42"
    assert p.source_category == "topstories"
    # raw_json is the original item, JSON-encoded.
    parsed = json.loads(p.raw_json)
    assert parsed["id"] == 42
    assert parsed["title"] == "Mapping test"


def test_collect_handles_askhn_post_without_url(tmp_settings) -> None:
    """Ask HN / Show HN posts often have no `url`; we should fall back
    to the HN discussion page so the row is still linkable."""
    item = _story_payload(
        99,
        title="Ask HN: how do you do X?",
        url=None,  # type: ignore[arg-type]
        text="<p>I want to do X. How?</p>",
    )
    handler = _make_handler({99: item}, story_lists={"askstories": [99]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["askstories"], limit_per_category=1
        ))
    assert posts[0].url == "https://news.ycombinator.com/item?id=99"
    # The body is preserved (we don't strip HTML — that's the report's job).
    assert "I want to do X" in posts[0].body


def test_collect_respects_limit_per_category(tmp_settings) -> None:
    """We must NOT issue more HTTP calls than `limit` per feed."""
    items = {
        i: _story_payload(i, title=f"Story {i}")
        for i in range(1, 11)
    }
    requested = []

    def handler(request):
        requested.append(request.url.path)
        return _make_handler(items, story_lists={"topstories": list(range(1, 11))})(request)

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=3
        ))
    assert len(posts) == 3
    # One list call + exactly 3 item calls = 4 requests.
    assert len(requested) == 4


def test_collect_uses_default_categories_when_none_given(tmp_settings) -> None:
    """When categories=None, the collector should use settings.hn_story_type_list."""
    tmp_settings.default_hn_story_types = "askstories,showstories"
    items = {
        1: _story_payload(1, title="Ask 1"),
        2: _story_payload(2, title="Show 1"),
    }
    handler = _make_handler(items, story_lists={
        "askstories": [1],
        "showstories": [2],
        "topstories": [],
    })
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(limit_per_category=1))
    # Both default categories should have been processed.
    titles = {p.title for p in posts}
    assert "Ask 1" in titles
    assert "Show 1" in titles


# -------------------------------------------------------------------------
# Skipping behavior
# -------------------------------------------------------------------------
def test_collect_skips_deleted_items(tmp_settings) -> None:
    """Items with deleted=true should be silently skipped."""
    items = {
        1: _story_payload(1, title="Live"),
        2: _story_payload(2, title="Dead", deleted=True),
    }
    handler = _make_handler(items, story_lists={"topstories": [1, 2]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=2
        ))
    assert [p.title for p in posts] == ["Live"]


def test_collect_skips_dead_items(tmp_settings) -> None:
    items = {
        1: _story_payload(1, title="Live"),
        2: _story_payload(2, title="Dead", dead=True),
    }
    handler = _make_handler(items, story_lists={"topstories": [1, 2]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=2
        ))
    assert [p.title for p in posts] == ["Live"]


def test_collect_skips_missing_items(tmp_settings) -> None:
    """A 404 on the item endpoint means the item was deleted server-side."""
    items = {
        1: _story_payload(1, title="Live"),
        # 2: not in the items dict -> 404 -> skipped
    }
    handler = _make_handler(items, story_lists={"topstories": [1, 2]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=2
        ))
    assert [p.title for p in posts] == ["Live"]


def test_collect_skips_items_without_title(tmp_settings) -> None:
    """An item with no `title` is not a story/job — skip it."""
    items = {
        1: _story_payload(1, title="OK"),
        2: {  # missing title
            "id": 2, "type": "story", "by": "x", "time": 1, "score": 1,
            "descendants": 0, "url": "https://e.com/2",
        },
    }
    handler = _make_handler(items, story_lists={"topstories": [1, 2]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=2
        ))
    assert [p.title for p in posts] == ["OK"]


def test_collect_skips_non_story_types(tmp_settings) -> None:
    """Polls, pollopts, etc. should not be collected as 'stories'."""
    items = {
        1: _story_payload(1, title="Story", type_="story"),
        2: _story_payload(2, title="Poll", type_="poll"),
    }
    handler = _make_handler(items, story_lists={"topstories": [1, 2]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=2
        ))
    assert [p.title for p in posts] == ["Story"]


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------
def test_collect_continues_when_story_type_returns_500(tmp_settings) -> None:
    """A failing feed should NOT abort the whole run."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v0/topstories.json":
            return httpx.Response(500, json={"error": "boom"})
        if request.url.path == "/v0/newstories.json":
            return httpx.Response(200, json=[1])
        if request.url.path == "/v0/item/1.json":
            return httpx.Response(200, json=_story_payload(1))
        return httpx.Response(404)

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories", "newstories"],
            limit_per_category=5,
        ))
    # topstories failed; newstories succeeded.
    assert len(posts) == 1
    assert posts[0].external_id == "1"


def test_collect_continues_when_single_item_errors(tmp_settings) -> None:
    """A 500 on /item/{id}.json should skip that item, not abort."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v0/topstories.json":
            return httpx.Response(200, json=[1, 2, 3])
        if request.url.path == "/v0/item/1.json":
            return httpx.Response(500, json={"error": "boom"})
        if request.url.path == "/v0/item/2.json":
            return httpx.Response(200, json=_story_payload(2, title="Two"))
        if request.url.path == "/v0/item/3.json":
            return httpx.Response(200, json=_story_payload(3, title="Three"))
        return httpx.Response(404)

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=3
        ))
    titles = [p.title for p in posts]
    assert titles == ["Two", "Three"]


def test_collect_warns_on_unknown_story_type(tmp_settings, caplog) -> None:
    """Unknown story types are logged and skipped, not raised."""
    handler = _make_handler({}, story_lists={"topstories": []})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        with caplog.at_level("WARNING", logger="founder_radar.collectors.hackernews"):
            list(collector.collect(categories=["bogus_story_type"]))
    # Warning is logged.
    assert any("Unknown HN story type" in r.message for r in caplog.records)


# -------------------------------------------------------------------------
# include-comments
# -------------------------------------------------------------------------
def test_collect_with_include_comments_yields_comments(tmp_settings) -> None:
    """When --include-comments is set, first-level comments are emitted too."""
    items = {
        1: _story_payload(1, title="Story with comments", kids=[10, 11]),
        10: {
            "id": 10, "type": "comment", "by": "c1", "time": 1_700_000_100,
            "text": "<p>This is the first comment.</p>",
        },
        11: {
            "id": 11, "type": "comment", "by": "c2", "time": 1_700_000_200,
            "text": "<p>And another one.</p>",
        },
    }
    handler = _make_handler(items, story_lists={"topstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(
            tmp_settings, include_comments=True
        )
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=1
        ))
    # 1 story + 2 comments.
    assert len(posts) == 3
    assert posts[0].title == "Story with comments"
    # Comments are emitted with a synthesized title from the first
    # non-empty line of their text.
    assert "This is the first comment" in posts[1].title
    assert "And another one" in posts[2].title
    # Their source_category is the parent story's category.
    assert all(p.source_category == "topstories" for p in posts)
    # Comments carry the comment author's name.
    assert posts[1].author == "c1"
    assert posts[2].author == "c2"


def test_collect_without_include_comments_skips_comments(tmp_settings) -> None:
    """Default behavior: stories only, no comments even if `kids` exist."""
    items = {
        1: _story_payload(1, title="Story", kids=[10, 11]),
        10: {"id": 10, "type": "comment", "by": "c1", "time": 1,
              "text": "comment 1"},
        11: {"id": 11, "type": "comment", "by": "c2", "time": 1,
              "text": "comment 2"},
    }
    handler = _make_handler(items, story_lists={"topstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        # Default: include_comments=False
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=1
        ))
    assert len(posts) == 1
    assert posts[0].title == "Story"


def test_collect_caps_comments_per_story_at_5(tmp_settings) -> None:
    """Even if a story has 20 kids, we fetch at most 5."""
    items = {
        1: _story_payload(1, title="Long thread", kids=list(range(2, 22))),
    }
    # 20 comment IDs, but we should only fetch the first 5.
    for i in range(2, 22):
        items[i] = {
            "id": i, "type": "comment", "by": f"c{i}", "time": 1,
            "text": f"comment {i}",
        }
    handler = _make_handler(items, story_lists={"topstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(
            tmp_settings, include_comments=True
        )
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=1
        ))
    # 1 story + 5 comments = 6 total. (HN_API_BASE import guard is fine.)
    assert len(posts) == 6


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def test_first_meaningful_line_strips_html_and_picks_first_nonempty() -> None:
    text = "<p>Hello world</p>\n\n<p>Second line</p>"
    assert _first_meaningful_line(text) == "Hello world"


def test_first_meaningful_line_truncates_long() -> None:
    text = "x" * 500
    out = _first_meaningful_line(text, max_len=50)
    assert len(out) <= 50
    assert out.endswith("...")


def test_first_meaningful_line_empty_returns_empty() -> None:
    assert _first_meaningful_line("") == ""
    assert _first_meaningful_line("   \n\n  ") == ""


def test_first_meaningful_line_replaces_common_html_entities() -> None:
    text = "Don&#x27;t stop &amp; go &lt;here&gt;"
    out = _first_meaningful_line(text)
    assert "Don't" in out
    assert "&" in out
    assert "<here>" in out


def test_safe_int_handles_garbage() -> None:
    assert _safe_int("42") == 42
    assert _safe_int(None) == 0  # default
    assert _safe_int("not_a_number", default=99) == 99
    assert _safe_int([1, 2, 3], default=5) == 5


# -------------------------------------------------------------------------
# Default HTTP base URL
# -------------------------------------------------------------------------
def test_default_base_url_is_official_hn_firebase() -> None:
    """The base URL must point at the official HN endpoint — no third
    party proxy. Phase 1+ collectors are explicit about this contract."""
    assert HN_API_BASE == "https://hacker-news.firebaseio.com/v0"
