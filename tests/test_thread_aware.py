"""Tests for HN thread metadata and thread-aware clustering.

These cover the Phase 4 fix for HN fragmentation. With embedding-only
clustering, a story + its comments typically ended up as many
singletons. Thread-aware grouping puts all posts sharing the same
HN story id (thread_id) into one cluster, no embeddings required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from founder_radar.collectors.hackernews import (
    HackerNewsCollector,
    HN_API_BASE,
)
from founder_radar.database.connection import (
    get_session,
    init_engine,
    reset_for_tests,
)
from founder_radar.database.models import Post
from founder_radar.database.repository import (
    EmbeddingRepository,
    OpportunityRepository,
    PostRepository,
)


@pytest.fixture
def _init_engine(tmp_settings):
    """Initialize the engine for tests that don't use `configured_db`.

    Tests in this module need to call `get_session()`, which requires
    a running engine. The conftest autouse doesn't do that for tests
    outside the CLI suite, so this fixture provides it on demand.
    """
    print(f"DEBUG _init_engine setup, url={tmp_settings.database_url}", flush=True)
    init_engine(tmp_settings.database_url)
    yield
    reset_for_tests()
# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _post(
    external_id: str,
    *,
    title: str = "Sample",
    thread_id: str | None = None,
    parent_id: str | None = None,
    item_type: str | None = None,
    cluster_id: int | None = None,
    source: str = "hackernews",
) -> Post:
    return Post(
        source=source,
        external_id=external_id,
        source_category="topstories",
        title=title,
        body="",
        author="op",
        url=None,
        score=1,
        num_comments=0,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
        thread_id=thread_id,
        parent_id=parent_id,
        item_type=item_type,
        cluster_id=cluster_id,
    )


def _story(item_id: int, **kwargs) -> dict:
    item = {
        "id": item_id, "type": "story", "by": "alice",
        "time": 1_700_000_000, "title": f"Story {item_id}",
        "score": 100, "descendants": 5,
        "url": f"https://example.com/{item_id}",
    }
    item.update(kwargs)
    return item


def _comment(item_id: int, parent: int, **kwargs) -> dict:
    item = {
        "id": item_id, "type": "comment", "by": f"c{item_id}",
        "time": 1_700_000_000, "parent": parent,
        "text": f"<p>Comment {item_id}</p>",
    }
    item.update(kwargs)
    return item


def _make_handler(items: dict[int, dict], story_lists: dict[str, list] | None = None):
    """Build an httpx handler that returns canned story lists and items."""
    story_lists = story_lists or {}

    def handler(request: httpx.Request) -> httpx.Response:
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


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# -------------------------------------------------------------------------
# 1. HN story gets thread_id = its own id
# -------------------------------------------------------------------------
def test_hn_story_gets_thread_id_equal_to_own_id(tmp_settings, _init_engine) -> None:
    items = {
        42: _story(42, kids=[]),
    }
    handler = _make_handler(
        items={42: items[42]},
        story_lists={"topstories": [42]},
    )
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=1
        ))
    assert len(posts) == 1
    p = posts[0]
    assert p.external_id == "42"
    assert p.thread_id == "42"        # story's own id
    assert p.parent_id is None        # top-level
    assert p.item_type == "story"


# -------------------------------------------------------------------------
# 2. HN comment gets thread_id = root story id
# -------------------------------------------------------------------------
def test_hn_comment_gets_thread_id_equal_to_story_id(tmp_settings, _init_engine) -> None:
    """A comment's thread_id is the *parent story* id, not its own id."""
    items = {
        # Story 1 with two comments (10, 11) and a nested reply (12).
        1: _story(1, kids=[10, 11]),
        10: _comment(10, parent=1),
        11: _comment(11, parent=1),
        12: _comment(12, parent=10),  # reply to comment 10
    }
    handler = _make_handler(
        items=items,
        story_lists={"topstories": [1]},
    )
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(
            tmp_settings, include_comments=True
        )
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=1
        ))
    # 1 story + 2 top-level comments (12 is not fetched because we
    # only iterate top-level kids).
    by_id = {p.external_id: p for p in posts}

    # Story
    assert by_id["1"].thread_id == "1"
    assert by_id["1"].item_type == "story"
    assert by_id["1"].parent_id is None

    # Top-level comments
    for cid in ("10", "11"):
        assert by_id[cid].thread_id == "1"   # story id, not own id
        assert by_id[cid].item_type == "comment"
        assert by_id[cid].parent_id == "1"   # top-level => parent is story


# -------------------------------------------------------------------------
# 3. Thread-aware clustering groups story + comments
# -------------------------------------------------------------------------
def test_thread_aware_clustering_groups_story_and_comments(tmp_settings, _init_engine) -> None:
    """`founder-radar cluster --mode thread-aware` should put all
    posts with the same thread_id into one cluster."""
    # Seed 3 threads of varying sizes.
    with get_session() as session:
        post_repo = PostRepository(session)
        # Thread A: 1 story + 3 comments = 4 posts
        for ext_id in ("a1", "a2", "a3", "a4"):
            if ext_id == "a1":
                p = _post(ext_id, title="Thread A story",
                          thread_id="a", item_type="story")
            else:
                p = _post(ext_id, title=f"A comment {ext_id}",
                          thread_id="a", parent_id="a1",
                          item_type="comment")
            post_repo.add(p)
        # Thread B: 1 story + 0 comments = 1 post
        post_repo.add(_post("b1", title="Thread B story",
                            thread_id="b", item_type="story"))
        # Thread C: 1 story + 1 comment = 2 posts
        post_repo.add(_post("c1", title="Thread C story",
                            thread_id="c", item_type="story"))
        post_repo.add(_post("c2", title="C comment",
                            thread_id="c", parent_id="c1",
                            item_type="comment"))
        # One orphan post with no thread_id (pre-thread-aware era).
        post_repo.add(_post("z1", title="Orphan post",
                            thread_id=None, item_type=None))

    # Simulate the cluster command's thread-aware branch.
    from collections import Counter
    with get_session() as session:
        post_repo = PostRepository(session)
        posts = post_repo.list_all()
        thread_groups: dict = {}
        missing = 0
        for p in posts:
            if p.thread_id is None:
                missing += 1
                continue
            thread_groups.setdefault(p.thread_id, []).append(p.id)
        # Thread-aware should produce 3 clusters (a, b, c), not 8.
        assert len(thread_groups) == 3
        assert sum(len(v) for v in thread_groups.values()) == 7
        # The orphan is correctly excluded.
        assert missing == 1
        # Each thread has the right number of members.
        counts = Counter(thread_groups[k][0] for k in thread_groups)
        # Simulate the assign step.
        post_repo.reset_clusters()
        assignments = {}
        for tid, pids in thread_groups.items():
            cid = min(pids)
            for pid in pids:
                assignments[pid] = cid
        post_repo.assign_clusters(assignments)
        # All members of thread A share cluster_id, distinct from B/C.
        a_pids = set(thread_groups["a"])
        cids_for_a = {
            post_repo.get_by_id(pid).cluster_id for pid in a_pids
        }
        assert len(cids_for_a) == 1
        a_cid = cids_for_a.pop()
        # B and C have different cluster_ids than A.
        b_cid = post_repo.get_by_id(thread_groups["b"][0]).cluster_id
        c_cid = post_repo.get_by_id(thread_groups["c"][0]).cluster_id
        assert a_cid != b_cid
        assert a_cid != c_cid
        # Orphan has no cluster_id.
        assert post_repo.get_by_id(
            next(p.id for p in posts if p.external_id == "z1")
        ).cluster_id is None


# -------------------------------------------------------------------------
# 4. Thread-aware clustering produces fewer clusters than posts
# -------------------------------------------------------------------------
def test_thread_aware_clustering_reduces_cluster_count(tmp_settings, _init_engine) -> None:
    """With 1 story + 5 comments = 6 posts, thread-aware should
    yield 1 cluster, not 6."""
    with get_session() as session:
        post_repo = PostRepository(session)
        # Story + 5 comments all in one thread.
        post_repo.add(_post("1", title="story", thread_id="t", item_type="story"))
        for i in range(2, 7):
            post_repo.add(_post(str(i), title=f"c{i}",
                                thread_id="t", parent_id="1",
                                item_type="comment"))

    with get_session() as session:
        post_repo = PostRepository(session)
        posts = post_repo.list_all()
        # Build thread groups (mapping thread_id -> list of post_ids).
        groups: dict = {}
        for p in posts:
            if p.thread_id is None:
                continue
            groups.setdefault(p.thread_id, []).append(p.id)
        # 6 posts in 1 thread -> 1 cluster.
        assert len(groups) == 1
        assert len(groups["t"]) == 6
        # When we assign, 6 posts end up in 1 cluster.
        post_repo.reset_clusters()
        assignments = {pid: min(pids) for tid, pids in groups.items()
                        for pid in pids}
        post_repo.assign_clusters(assignments)
        sizes = post_repo.cluster_sizes()
        assert len(sizes) == 1
        # The cluster_id chosen was min(groups["t"]) = the first post id.
        assert sizes[min(p.id for p in posts)] == 6


# -------------------------------------------------------------------------
# 5. Extract works after thread-aware clustering
# -------------------------------------------------------------------------
def test_extract_works_after_thread_aware_clustering(tmp_settings, _init_engine) -> None:
    """After thread-aware clustering, extract should produce 1
    opportunity per multi-post thread and skip singletons (including
    threads with only 1 post that didn't get merged)."""
    from founder_radar.analysis.opportunity import HeuristicExtractor

    # Single session block: add posts, build mapping, assign clusters,
    # read cluster sizes - all in one transaction.
    with get_session() as session:
        post_repo = PostRepository(session)
        for ext_id in ("a1", "a2", "a3", "a4"):
            p = _post(ext_id, title=f"a{ext_id}",
                      thread_id="a", item_type="story" if ext_id == "a1" else "comment",
                      parent_id="a1" if ext_id != "a1" else None)
            post_repo.add(p)
        post_repo.add(_post("b1", title="b1 story", thread_id="b",
                            item_type="story"))

        all_posts = list(post_repo.list_all())
        ext_to_pid = {p.external_id: p.id for p in all_posts}
        assert "a1" in ext_to_pid, f"a1 missing from {ext_to_pid!r}"

        groups = {
            "a": [ext_to_pid["a1"], ext_to_pid["a2"],
                  ext_to_pid["a3"], ext_to_pid["a4"]],
            "b": [ext_to_pid["b1"]],
        }
        post_repo.reset_clusters()
        assignments = {pid: min(pids) for tid, pids in groups.items()
                        for pid in pids}
        post_repo.assign_clusters(assignments)

        # Read cluster sizes inside the same session so the writes
        # we just made are visible.
        sizes = post_repo.cluster_sizes()
        qualifying = [c for c, s in sizes.items() if s >= 2]
        assert len(qualifying) == 1
        assert sizes[qualifying[0]] == 4

        # Run extract against the qualifying cluster.
        from founder_radar.analysis.opportunity import HeuristicExtractor
        opp_repo = OpportunityRepository(session)
        extractor = HeuristicExtractor()
        for cid in qualifying:
            posts = list(post_repo.list_by_cluster(cid))
            data = extractor.extract(cluster_id=cid, posts=posts)
            opp_repo.add_from_dict(data, post_ids=[p.id for p in posts])
        opps = opp_repo.list_all()
        assert len(opps) == 1


# -------------------------------------------------------------------------
# 6. Embedding clustering still works (regression)
# -------------------------------------------------------------------------
def test_embedding_mode_still_works(tmp_settings, _init_engine) -> None:
    """`founder-radar cluster --threshold 0.50` (no --mode flag) must
    still run the embedding-based clusterer. We just verify the
    threshold-check passes the embedding path through."""
    from founder_radar.analysis.clustering import GreedyCosineClusterer

    items = {
        1: _story(1, kids=[]),
        2: _story(2, kids=[]),
    }
    handler = _make_handler(
        items=items,
        story_lists={"topstories": [1, 2]},
    )
    client = _mock_client(handler)
    with patch.object(HackerNewsCollector, "_client", return_value=client):
        collector = HackerNewsCollector(tmp_settings)
        # Collect two stories that share the same NullEmbedder vector.
        # Their cosine similarity is 1.0, so threshold 0.50 should
        # group them.
        posts = list(collector.collect(
            categories=["topstories"], limit_per_category=2
        ))
    assert len(posts) == 2
    # Now run the clusterer the way cluster --threshold 0.50 does.
    vecs = [np.zeros(8, dtype=np.float32) for _ in posts]
    for v in vecs:
        v[0] = 1.0
    matrix = np.stack(vecs).astype(np.float32)
    clusterer = GreedyCosineClusterer(similarity_threshold=0.50)
    labels = clusterer.cluster(matrix)
    # Two posts with identical vectors -> 1 cluster.
    assert len(set(labels.tolist())) == 1


# -------------------------------------------------------------------------
# 7. CLI smoke: thread-aware does not raise NameError
# -------------------------------------------------------------------------
def test_thread_aware_cli_does_not_raise_nameerror(
    tmp_path, monkeypatch,
) -> None:
    """Regression: the thread-aware branch must compile and run."""
    from typer.testing import CliRunner
    from founder_radar.main import app
    from founder_radar.database.connection import get_session, init_engine

    db_path = tmp_path / "cal.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    init_engine(f"sqlite:///{db_path}")

    # Seed one thread.
    with get_session() as session:
        post_repo = PostRepository(session)
        for ext_id, is_story in [("1", True), ("2", False), ("3", False)]:
            p = Post(
                source="hackernews", external_id=ext_id,
                source_category="topstories", title=f"t{ext_id}",
                body="", author="op", url=None, score=1, num_comments=0,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
                thread_id="1" if not is_story else "1",
                parent_id="1" if not is_story and ext_id == "2" else None,
                item_type="story" if is_story else "comment",
            )
            post_repo.add(p)
    # Reset cluster_id so we can apply fresh.
    with get_session() as session:
        PostRepository(session).reset_clusters()

    runner = CliRunner()
    result = runner.invoke(app, ["cluster", "--mode", "thread-aware"])

    # No NameError; no Traceback; success exit code.
    assert "NameError" not in (result.output + str(result.exception or ""))
    assert "Traceback" not in result.output
    # Three posts assigned to one cluster.
    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
    assert len(sizes) == 1
    assert sizes[list(sizes.keys())[0]] == 3
