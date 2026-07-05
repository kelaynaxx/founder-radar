"""Phase 4+ HN calibration tests: subtype detection + Algolia query path."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from founder_radar.collectors.hackernews import HackerNewsCollector


def _story(item_id, **kwargs):
    item = {
        "id": item_id, "type": "story", "by": "alice",
        "time": 1_700_000_000, "title": f"Story {item_id}",
        "score": 100, "descendants": 5,
        "url": f"https://example.com/{item_id}",
    }
    item.update(kwargs)
    return item


def _make_handler(items, story_lists=None):
    story_lists = story_lists or {}

    def handler(request):
        path = request.url.path
        for st in ("topstories", "newstories", "askstories",
                   "showstories", "beststories", "jobstories"):
            if path == f"/v0/{st}.json":
                return httpx.Response(200, json=story_lists.get(st, []))
        if path.startswith("/v0/item/") and path.endswith(".json"):
            try:
                item_id = int(path[len("/v0/item/"):-len(".json")])
            except ValueError:
                return httpx.Response(404, json=None)
            if item_id in items:
                return httpx.Response(200, json=items[item_id])
            return httpx.Response(404, json=None)
        return httpx.Response(404, json=None)

    return handler


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# Subtype detection (calibration tag)

def test_subtype_is_ask_hn_for_ask_hn_title(tmp_settings):
    items = {1: _story(1, title="Ask HN: what's your favorite CI tool?")}
    handler = _make_handler(items, story_lists={"askstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["askstories"], limit_per_category=1
        ))
    assert posts[0].subtype == "ask_hn"


def test_subtype_is_show_hn_for_show_hn_title(tmp_settings):
    items = {1: _story(1, title="Show HN: I built a faster grep")}
    handler = _make_handler(items, story_lists={"showstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["showstories"], limit_per_category=1
        ))
    assert posts[0].subtype == "show_hn"


def test_subtype_is_regular_story_for_normal_story(tmp_settings):
    items = {1: _story(1, title="New JS framework released")}
    handler = _make_handler(items, story_lists={"topstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["topstories"], limit_per_category=1
        ))
    assert posts[0].subtype == "regular_story"


def test_subtype_is_regular_comment_for_comments(tmp_settings):
    items = {
        1: _story(1, title="Story with comments", kids=[10]),
        10: {"id": 10, "type": "comment", "by": "c1", "time": 1,
              "text": "<p>first comment</p>"},
    }
    handler = _make_handler(items, story_lists={"topstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(
            tmp_settings, include_comments=True
        ).collect(categories=["topstories"], limit_per_category=1))
    by_id = {p.external_id: p for p in posts}
    assert by_id["10"].subtype == "regular_comment"


def test_subtype_is_job_for_job_stories(tmp_settings):
    items = {1: {"id": 1, "type": "job", "by": "acme",
              "time": 1, "title": "Founders Wanted (YC)"}}
    handler = _make_handler(items, story_lists={"jobstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["jobstories"], limit_per_category=1
        ))
    assert posts[0].subtype == "job"


def test_subtype_is_case_insensitive_for_ask_show_prefix(tmp_settings):
    items = {1: _story(1, title="ask hn: lowercase prefix")}
    handler = _make_handler(items, story_lists={"askstories": [1]})
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["askstories"], limit_per_category=1
        ))
    assert posts[0].subtype == "ask_hn"


# Algolia search path (--query)

def test_collect_with_query_uses_algolia_search(tmp_settings):
    items_called_with = []

    def handler(request):
        items_called_with.append(request.url)
        if "algolia.com" in str(request.url):
            return httpx.Response(200, json={
                "hits": [_story(1, title="Ask HN: how do you do X?")],
            })
        return httpx.Response(404, json=None)

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["askstories"], limit_per_category=1,
            query="pain problem frustrated",
        ))
    assert len(posts) == 1
    assert any("algolia.com" in str(u) for u in items_called_with)
    algolia_url = next(
        u for u in items_called_with if "algolia.com" in str(u)
    )
    # Query is URL-encoded in the Algolia request.
    assert ("query=pain" in str(algolia_url)
            or "query=pain%20problem" in str(algolia_url))
    # Story type filter is passed as a tag.
    assert "tags=ask_hn" in str(algolia_url)


def test_collect_with_query_skips_firebase_endpoints(tmp_settings):
    urls_hit = []

    def handler(request):
        urls_hit.append(str(request.url))
        if "algolia.com" in str(request.url):
            return httpx.Response(200, json={"hits": []})
        if ("firebaseio.com" in str(request.url)
                and not "/item/" in str(request.url)):
            return httpx.Response(200, json=[])
        if "/item/" in str(request.url):
            return httpx.Response(404, json=None)
        return httpx.Response(404, json=None)

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        list(HackerNewsCollector(tmp_settings).collect(
            categories=["askstories"], limit_per_category=1,
            query="anything",
        ))
    # Firebase story-list endpoints were not called.
    assert not any(
        "firebaseio.com" in u and not "/item/" in u
        for u in urls_hit
    )


def test_collect_with_query_handles_algolia_failure_gracefully(tmp_settings):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        posts = list(HackerNewsCollector(tmp_settings).collect(
            categories=["askstories"], limit_per_category=10,
            query="anything",
        ))
    # No posts (Algolia error logged, not raised), pipeline continues.
    assert posts == []


def test_algolia_tag_includes_story_base(tmp_settings):
    urls_hit = []

    def handler(request):
        urls_hit.append(str(request.url))
        if "algolia.com" in str(request.url):
            return httpx.Response(200, json={"hits": []})
        return httpx.Response(404, json=None)

    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        list(HackerNewsCollector(tmp_settings).collect(
            categories=["askstories"], limit_per_category=1,
            query="x",
        ))
    algolia_url = next(u for u in urls_hit if "algolia.com" in u)
    # Every search request includes the `story` tag, which excludes
    # comments and jobs from the result set.
    assert "tags=story" in algolia_url
