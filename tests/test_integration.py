"""End-to-end integration test for the Phase 1+2+3 pipeline.

Exercises: collect (mocked) -> clean -> store -> embed (NullEmbedder) ->
cluster (greedy) -> extract (heuristic) -> opportunities list.

This is the closest thing to a real run we can do without Reddit API
access. It catches wiring bugs that the unit tests miss.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from founder_radar.analysis import (
    GreedyCosineClusterer,
    InMemoryVectorStore,
    NullEmbedder,
    compute_deterministic_scores,
)
from founder_radar.collectors import RawPost
from founder_radar.collectors.reddit import RedditCollector
from founder_radar.database.connection import get_session
from founder_radar.database.models import Post
from founder_radar.database.repository import (
    EmbeddingRepository,
    OpportunityRepository,
    PostRepository,
)


def _fake_submission(sid, title, body="", score=10, comments=3, subreddit="entrepreneur"):
    from types import SimpleNamespace
    return SimpleNamespace(
        id=sid,
        title=title,
        selftext=body,
        author="op",
        url=f"https://reddit.com/r/{subreddit}/comments/{sid}",
        score=score,
        num_comments=comments,
        created_utc=1_700_000_000.0,
        permalink=f"/r/{subreddit}/comments/{sid}",
    )


def test_full_pipeline_collect_embed_cluster_extract(configured_db, tmp_settings):
    """Walk every layer end-to-end against a real SQLite DB."""
    # ----- Phase 1: collect (PRAW mocked) -----
    fake_subs = {
        "entrepreneur": type("S", (), {
            "new": lambda self, limit: iter([
                _fake_submission(
                    "a1", "Why is finding customers so hard?",
                    body="I built an MVP and nobody is buying.",
                    score=20, comments=15,
                ),
                _fake_submission(
                    "a2", "How do I market on a zero budget?",
                    body="I have a great product but no marketing skills.",
                    score=10, comments=8,
                ),
                _fake_submission(
                    "a3", "Building a CRM for solopreneurs",
                    body="I want to build a CRM specifically for solo founders.",
                    score=5, comments=2,
                ),
            ])
        })(),
    }
    fake_reddit = type("R", (), {
        "subreddit": lambda self, name: fake_subs[name],
        "read_only": True,
    })()
    with patch.object(RedditCollector, "_client", return_value=fake_reddit):
        collector = RedditCollector(tmp_settings)
        raws = list(collector.collect(
            categories=["entrepreneur"], limit_per_category=10
        ))
    assert len(raws) == 3
    with get_session() as session:
        repo = PostRepository(session)
        repo.add_many(_raw_to_orm(r) for r in raws)
    with get_session() as session:
        assert PostRepository(session).count() == 3

    # ----- Phase 2a: embed (NullEmbedder, no model load) -----
    embedder = NullEmbedder(dim=8)
    with get_session() as session:
        repo = PostRepository(session)
        posts = [repo.get_by_id(i) for i in range(1, 4)]
        texts = [(p.title + "\n" + (p.body or "")).strip() for p in posts]
        vectors = embedder.embed_texts(texts)
        emb_repo = EmbeddingRepository(session)
        emb_repo.upsert_many(
            (p.id, embedder.model_name, vectors[i])
            for i, p in enumerate(posts)
        )
    with get_session() as session:
        assert EmbeddingRepository(session).count() == 3

    # ----- Phase 2b: cluster (greedy cosine, threshold = 0.0 so all merge) -----
    clusterer = GreedyCosineClusterer(similarity_threshold=0.0)
    with get_session() as session:
        emb_repo = EmbeddingRepository(session)
        embs = emb_repo.list_for_model(embedder.model_name)
        ids = [e.post_id for e in embs]
        matrix = np.stack(
            [np.frombuffer(e.vector, dtype=np.float32) for e in embs],
            axis=0,
        ).astype(np.float32, copy=False)
    labels = clusterer.cluster(matrix)
    with get_session() as session:
        PostRepository(session).assign_clusters(
            {pid: int(labels[i]) for i, pid in enumerate(ids)}
        )

    # ----- Phase 3: extract -----
    from founder_radar.analysis import HeuristicExtractor
    with get_session() as session:
        post_repo = PostRepository(session)
        opp_repo = OpportunityRepository(session)
        sizes = post_repo.cluster_sizes()
        for cid in sizes:
            posts = list(post_repo.list_by_cluster(cid))
            data = HeuristicExtractor().extract(cluster_id=cid, posts=posts)
            opp_repo.add_from_dict(data, post_ids=[p.id for p in posts])

    # ----- Verify the pipeline produced opportunities -----
    with get_session() as session:
        opps = OpportunityRepository(session).list_all()
    assert len(opps) >= 1
    assert opps[0].total_score >= 0.0
    assert opps[0].mentions >= 1
    assert opps[0].extraction_method == "heuristic"


def _raw_to_orm(raw: RawPost) -> Post:
    return Post(
        source=raw.source,
        external_id=raw.external_id,
        source_category=raw.source_category,
        title=raw.title,
        body=raw.body,
        author=raw.author,
        url=raw.url,
        score=raw.score,
        num_comments=raw.num_comments,
        created_at=raw.created_at,
        raw_json=raw.raw_json,
    )


def test_deterministic_scoring_picks_up_real_signals(configured_db):
    """End-to-end: a frustrated post scores higher than a neutral one."""
    from founder_radar.analysis.scoring import (
        compute_deterministic_scores,
    )
    frustrated = Post(
        source="reddit", external_id="x", source_category="test",
        title="I hate this stupid thing", body="Frustrated, broken",
    )
    neutral = Post(
        source="reddit", external_id="y", source_category="test",
        title="How do I learn X?", body="Looking for advice",
    )
    s_frustrated = compute_deterministic_scores([frustrated])
    s_neutral = compute_deterministic_scores([neutral])
    assert s_frustrated.emotional_intensity > s_neutral.emotional_intensity