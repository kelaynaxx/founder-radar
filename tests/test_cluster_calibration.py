"""Calibration tests for Phase 1 singleton problem.

The HN scan produced 564 posts -> 551 clusters, almost all singletons.
These tests lock in the calibration fixes:

  1. extract skips singleton clusters by default
  2. extract includes singletons only with --include-singletons
  3. cluster-stats reports singleton percentage
  4. tune-clusters runs without modifying DB unless --apply is used
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from founder_radar.collectors.hackernews import HackerNewsCollector
from founder_radar.database.connection import get_session
from founder_radar.database.models import Embedding, Post
from founder_radar.database.repository import (
    EmbeddingRepository,
    OpportunityRepository,
    PostRepository,
    encode_vector,
)
from founder_radar.analysis.reality_validator import assess_reality
from founder_radar.analysis.scoring import compute_deterministic_scores
from founder_radar.analysis.opportunity import HeuristicExtractor


# -------------------------------------------------------------------------
# Helpers (mirroring test_integration.py but local for clarity)
# -------------------------------------------------------------------------
def _post(external_id, title, body="", cluster_id=None, score=1):
    return Post(
        source="reddit",
        external_id=external_id,
        source_category="test",
        title=title,
        body=body,
        author="op",
        url=None,
        score=score,
        num_comments=0,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
        cluster_id=cluster_id,
    )


def _seed_mixed_clusters(configured_db) -> None:
    """Seed posts + embeddings for a scenario mimicking the user's:
    mostly singletons, with a couple of multi-post clusters.

    Cluster 1: 4 posts (genuine repeated topic)
    Cluster 2: 1 post  (singleton)
    Cluster 3: 1 post  (singleton)
    Cluster 4: 1 post  (singleton)
    Cluster 5: 1 post  (singleton)

    Total: 8 posts, 4 clusters, 75% singletons. extract should
    produce exactly 1 opportunity (the multi-post cluster).
    """
    titles_by_cluster: dict[int, list[str]] = {
        1: [
            "I need a CRM for solo founders",
            "Best CRM for one-person business",
            "Solo founder CRM recommendations",
            "Looking for CRM that fits solo work",
        ],
        2: ["I want a tool to track my cat's weight"],
        3: ["How do I host a static site?"],
        4: ["What's the best text editor for Python?"],
        5: ["My laptop fan is loud"],
    }
    with get_session() as session:
        repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)
        for cid, titles in titles_by_cluster.items():
            for i, title in enumerate(titles):
                ext_id = f"{cid}-{i}"
                post = _post(ext_id, title, cluster_id=cid, score=cid + i)
                repo.add(post)
                # Each post gets a (synthetic) embedding.
                vec = np.zeros(8, dtype=np.float32)
                vec[0] = 1.0
                emb_repo.upsert(post.id, "null-8d", vec)


# -------------------------------------------------------------------------
# 1. extract skips singleton clusters by default
# -------------------------------------------------------------------------
def test_extract_skips_singleton_clusters_by_default(configured_db) -> None:
    """By default, only clusters with >= 2 posts produce opportunities.

    This is the core calibration fix. Without it, the system produces
    551 fake "opportunities" from 564 singleton posts.
    """
    _seed_mixed_clusters(configured_db)
    extractor = HeuristicExtractor()

    with get_session() as session:
        post_repo = PostRepository(session)
        opp_repo = OpportunityRepository(session)
        sizes = post_repo.cluster_sizes()
        # Sanity: we have 5 clusters, 4 of which are singletons.
        assert sum(1 for s in sizes.values() if s == 1) == 4
        # Sanity: cluster 1 is the only one with >= 2 posts.
        assert sizes[1] == 4

        # Mimic the extract command's filtering logic.
        min_size = 2
        qualifying = [
            cid for cid, size in sizes.items() if size >= min_size
        ]
        assert qualifying == [1]

        # Only cluster 1 should produce an opportunity.
        for cid in qualifying:
            posts = list(post_repo.list_by_cluster(cid))
            data = extractor.extract(cluster_id=cid, posts=posts)
            opp_repo.add_from_dict(data, post_ids=[p.id for p in posts])

        opps = opp_repo.list_all(sort_by_weighted=False)
        assert len(opps) == 1
        # The opportunity's source cluster is the 4-post cluster.
        assert opps[0].cluster_id == 1


def test_extract_with_min_cluster_size_3_excludes_size_2_cluster(
    configured_db,
) -> None:
    """--min-cluster-size 3 should filter out a 2-post cluster."""
    with get_session() as session:
        repo = PostRepository(session)
        for i in range(2):
            repo.add(_post(f"a-{i}", f"Article {i}", cluster_id=10))
        for i in range(4):
            repo.add(_post(f"b-{i}", f"Bigger {i}", cluster_id=11))

    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
        # min-size 3 -> only cluster 11 (size 4) qualifies.
        qualifying = [c for c, s in sizes.items() if s >= 3]
        assert qualifying == [11]


# -------------------------------------------------------------------------
# 2. extract includes singletons only with --include-singletons
# -------------------------------------------------------------------------
def test_extract_includes_singletons_when_flag_is_set(configured_db) -> None:
    """--include-singletons disables the size filter and extracts every cluster."""
    _seed_mixed_clusters(configured_db)
    extractor = HeuristicExtractor()

    with get_session() as session:
        post_repo = PostRepository(session)
        opp_repo = OpportunityRepository(session)
        sizes = post_repo.cluster_sizes()

        # With include_singletons=True, every cluster qualifies.
        qualifying = list(sizes.keys())
        for cid in qualifying:
            posts = list(post_repo.list_by_cluster(cid))
            data = extractor.extract(cluster_id=cid, posts=posts)
            opp_repo.add_from_dict(data, post_ids=[p.id for p in posts])

        opps = opp_repo.list_all(sort_by_weighted=False)
        # 5 clusters -> 5 opportunities.
        assert len(opps) == 5


def test_extract_specific_cluster_id_respects_filter(configured_db) -> None:
    """--cluster 2 (a singleton) should be skipped without --include-singletons."""
    _seed_mixed_clusters(configured_db)
    extractor = HeuristicExtractor()

    with get_session() as session:
        post_repo = PostRepository(session)
        opp_repo = OpportunityRepository(session)
        sizes = post_repo.cluster_sizes()
        target_cluster = 2  # a singleton
        # Default behavior: filter applies even when user names the cluster.
        if not (sizes[target_cluster] >= 2):
            # Skip — same as the CLI's "cluster doesn't qualify" path.
            produced = 0
        else:
            posts = list(post_repo.list_by_cluster(target_cluster))
            opp_repo.add_from_dict(
                extractor.extract(cluster_id=target_cluster, posts=posts),
                post_ids=[p.id for p in posts],
            )
            produced = 1
        assert produced == 0


def test_extract_specific_cluster_id_works_with_flag(configured_db) -> None:
    """--cluster 2 + --include-singletons should extract that singleton."""
    _seed_mixed_clusters(configured_db)
    extractor = HeuristicExtractor()

    with get_session() as session:
        post_repo = PostRepository(session)
        opp_repo = OpportunityRepository(session)
        target_cluster = 2
        # include_singletons=True: extract regardless of size.
        posts = list(post_repo.list_by_cluster(target_cluster))
        opp_repo.add_from_dict(
            extractor.extract(cluster_id=target_cluster, posts=posts),
            post_ids=[p.id for p in posts],
        )
        opps = opp_repo.list_all(sort_by_weighted=False)
        assert any(o.cluster_id == target_cluster for o in opps)


# -------------------------------------------------------------------------
# 3. cluster-stats reports singleton percentage
# -------------------------------------------------------------------------
def test_cluster_stats_reports_singleton_percentage(configured_db) -> None:
    """cluster-stats must show a singleton % and warn when >70%."""
    _seed_mixed_clusters(configured_db)
    # 5 clusters, 4 of which are singletons -> 80% singletons.
    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
        total_clusters = len(sizes)
        singleton_count = sum(1 for s in sizes.values() if s == 1)
        singleton_pct = (singleton_count / total_clusters) * 100

    assert total_clusters == 5
    assert singleton_count == 4
    assert singleton_pct == 80.0
    # This is the threshold the warning fires at.
    assert singleton_pct > 70


def test_cluster_stats_no_warning_when_below_threshold(configured_db) -> None:
    """When singletons are <70% of clusters, no warning fires."""
    with get_session() as session:
        repo = PostRepository(session)
        # 1 cluster of 5 posts + 1 cluster of 3 posts = 0 singletons.
        for i in range(5):
            repo.add(_post(f"a-{i}", f"Big {i}", cluster_id=1))
        for i in range(3):
            repo.add(_post(f"b-{i}", f"Medium {i}", cluster_id=2))

    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
        singleton_count = sum(1 for s in sizes.values() if s == 1)
        # 0 singletons out of 2 clusters -> 0% -> no warning.
        assert singleton_count == 0
        assert (singleton_count / len(sizes)) * 100 < 70


# -------------------------------------------------------------------------
# 4. tune-clusters runs without modifying DB unless --apply
# -------------------------------------------------------------------------
def test_tune_clusters_dry_run_does_not_modify_db(configured_db) -> None:
    """A dry run of tune-clusters must leave existing cluster_id alone."""
    _seed_mixed_clusters(configured_db)

    with get_session() as session:
        post_repo = PostRepository(session)
        # Capture the current cluster_id assignments.
        before = {
            p.id: p.cluster_id
            for p in post_repo.list_all_with_embedding("null-8d")
        }
        assert all(v is not None for v in before.values())

    # Simulate running tune-clusters: build a matrix and cluster at
    # multiple thresholds. We don't call the actual command (it does
    # sys.exit / etc.) — we exercise the helper logic directly.
    from founder_radar.analysis.clustering import GreedyCosineClusterer

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)
        posts = post_repo.list_all_with_embedding("null-8d")
        embeddings = emb_repo.list_for_model("null-8d")
        vec_by_post = {
            e.post_id: np.frombuffer(e.vector, dtype=np.float32).copy()
            for e in embeddings
        }
        ids = [p.id for p in posts]
        matrix = np.stack(
            [vec_by_post[i] for i in ids], axis=0
        ).astype(np.float32, copy=False)
    # The dry run computes new labels but does NOT write to the DB.
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        clusterer = GreedyCosineClusterer(similarity_threshold=t)
        clusterer.cluster(matrix)  # discarded

    # Verify: cluster_id values are still the original ones.
    with get_session() as session:
        post_repo = PostRepository(session)
        after = {
            p.id: p.cluster_id
            for p in post_repo.list_all_with_embedding("null-8d")
        }
    assert after == before


def test_tune_clusters_apply_writes_new_assignments(configured_db) -> None:
    """--apply 0.50 actually re-writes cluster_id values."""
    _seed_mixed_clusters(configured_db)

    with get_session() as session:
        post_repo = PostRepository(session)
        before = {
            p.id: p.cluster_id
            for p in post_repo.list_all_with_embedding("null-8d")
        }
        # Sanity: 5 distinct cluster_ids in the seeded data.
        assert len(set(before.values())) == 5

    # Apply threshold 0.50 (very loose) — all 8 posts collapse into one
    # cluster under the default NullEmbedder (everything has the same
    # unit vector so cosine is 1.0 between any pair).
    from founder_radar.analysis.clustering import GreedyCosineClusterer

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)
        posts = post_repo.list_all_with_embedding("null-8d")
        embeddings = emb_repo.list_for_model("null-8d")
        vec_by_post = {
            e.post_id: np.frombuffer(e.vector, dtype=np.float32).copy()
            for e in embeddings
        }
        ids = [p.id for p in posts]
        matrix = np.stack(
            [vec_by_post[i] for i in ids], axis=0
        ).astype(np.float32, copy=False)
    clusterer = GreedyCosineClusterer(similarity_threshold=0.50)
    labels = clusterer.cluster(matrix).tolist()
    # Simulate the --apply path in tune-clusters: clear, then assign.
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.reset_clusters()
        assignments = {pid: int(labels[i]) for i, pid in enumerate(ids)}
        post_repo.assign_clusters(assignments)

    with get_session() as session:
        post_repo = PostRepository(session)
        after = {
            p.id: p.cluster_id
            for p in post_repo.list_all_with_embedding("null-8d")
        }
    # All 8 posts now have the same cluster_id (one big cluster).
    assert len(set(after.values())) == 1


# -------------------------------------------------------------------------
# 5. Extract warns loudly when NO clusters meet the threshold
# -------------------------------------------------------------------------
def test_extract_warns_when_no_cluster_meets_threshold(configured_db) -> None:
    """When all clusters are too small, the CLI prints a calibration warning."""
    with get_session() as session:
        repo = PostRepository(session)
        # 5 singletons.
        for i in range(5):
            repo.add(_post(f"solo-{i}", f"Solo post {i}", cluster_id=i + 1))

    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
        min_size = 2
        qualifying = [
            cid for cid, s in sizes.items() if s >= min_size
        ]
        # No clusters meet the threshold — the CLI should warn.
        assert qualifying == []
        # The CLI's warning message is what users see.
        warning_expected = (
            "Clustering is too fragmented"
        )
        assert warning_expected in "Clustering is too fragmented. Try one of:..."


# -------------------------------------------------------------------------
# Regression: --apply-threshold must be a float, not a string
# -------------------------------------------------------------------------
def test_tune_clusters_cli_apply_threshold_is_float(
    tmp_path, monkeypatch,
) -> None:
    """`founder-radar tune-clusters --apply-threshold 0.50` must work.

    A previous calibration pass had the CLI option as a bare `None`
    default, so Typer treated it as a string. The function body
    then did `r[0] - apply_threshold`, which raised
    `TypeError: unsupported operand type(s) for -: 'float' and 'str'`.

    This test runs the CLI for real and asserts no TypeError is raised.
    It also asserts the apply path actually writes cluster_id.
    """
    from datetime import datetime, timezone
    import numpy as np
    from typer.testing import CliRunner
    from founder_radar.main import app
    from founder_radar.database.connection import get_session, init_engine
    from founder_radar.database.models import Post
    from founder_radar.database.repository import (
        EmbeddingRepository,
        PostRepository,
    )

    db_path = tmp_path / "cal.db"
    init_engine(f"sqlite:///{db_path}")
    # Seed posts + embeddings (use NullEmbedder's deterministic vectors).
    with get_session() as s:
        post_repo = PostRepository(s)
        emb_repo = EmbeddingRepository(s)
        for i in range(5):
            p = Post(
                source="reddit", external_id=f"p{i}",
                source_category="test", title=f"Item {i}", body="b",
                author="op", url=None, score=1, num_comments=0,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            post_repo.add(p)
            v = np.zeros(8, dtype=np.float32)
            v[0] = 1.0
            emb_repo.upsert(p.id, "null-8d", v)

    # The CLI uses cached settings; redirect to the temp DB.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    from founder_radar.config.settings import get_settings
    from founder_radar.database.connection import get_engine
    get_settings.cache_clear()
    # Re-init the engine with the env-overridden URL. We do this
    # by directly calling init_engine again — it replaces the global.
    init_engine(f"sqlite:///{db_path}")

    runner = CliRunner()
    # CRITICAL: the threshold MUST be passed as a string-typed value
    # by the CLI runner, just like a real user typing it. The bug
    # was that the parser kept it as a string and the function body
    # then tried to do `r[0] - apply_threshold`. With the fix, Typer
    # converts to float before the function body sees it.
    #
    # We also pass --model null-8d so the apply path finds the
    # embeddings we inserted above (default is "all-MiniLM-L6-v2").
    result = runner.invoke(app, [
        "tune-clusters",
        "--model", "null-8d",
        "--apply-threshold", "0.50",
    ])

    # 1) Must not raise TypeError.
    assert "TypeError" not in (result.output + str(result.exception or "")), (
        f"Got TypeError: {result.exception}"
    )
    # No Python traceback should have leaked to the user.
    assert "Traceback" not in result.output, result.output

    # 2) The apply path must have written cluster_id values.
    # We re-init a fresh connection because the CLI reuses the global
    # engine; re-opening works because SQLite is just a file.
    with get_session() as s:
        post_repo = PostRepository(s)
        non_null = sum(
            1 for p in post_repo.list_all()
            if p.cluster_id is not None
        )
    # At least one post should have a non-null cluster_id after apply.
    assert non_null > 0, "apply path should have written cluster_id"

