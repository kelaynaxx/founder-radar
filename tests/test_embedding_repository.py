"""Tests for the EmbeddingRepository and cluster-management methods on
PostRepository. These exercise the Phase 2 schema changes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from founder_radar.database.connection import get_session
from founder_radar.database.models import Embedding, Post
from founder_radar.database.repository import (
    EmbeddingRepository,
    PostRepository,
    decode_vector,
    encode_vector,
)


def _post(**overrides) -> Post:
    defaults = dict(
        source="reddit",
        external_id="x",
        source_category="entrepreneur",
        title="Need help finding customers for our new SaaS product launch",
        body="I built a thing and nobody is buying. What am I doing wrong?",
        author="op_user",
        url="https://reddit.com/r/entrepreneur/comments/x",
        score=42,
        num_comments=17,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=3),
    )
    defaults.update(overrides)
    return Post(**defaults)


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# -------------------------------------------------------------------------
# Vector encoding helpers
# -------------------------------------------------------------------------
def test_encode_decode_roundtrip() -> None:
    v = _vec(0, dim=16)
    blob = encode_vector(v)
    out = decode_vector(blob)
    np.testing.assert_array_equal(v, out)


def test_encode_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        encode_vector(np.zeros((2, 4), dtype=np.float32))


def test_encode_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="float32"):
        encode_vector(np.zeros(4, dtype=np.float64))


def test_decode_validates_expected_dim() -> None:
    v = _vec(0, dim=8)
    blob = encode_vector(v)
    with pytest.raises(ValueError, match="dim"):
        decode_vector(blob, expected_dim=16)


# -------------------------------------------------------------------------
# EmbeddingRepository
# -------------------------------------------------------------------------
def test_upsert_inserts_new_row(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        emb_repo = EmbeddingRepository(session)
        emb_repo.upsert(1, "null-8d", _vec(0))
    with get_session() as session:
        emb_repo = EmbeddingRepository(session)
        assert emb_repo.count() == 1


def test_upsert_replaces_existing_row(configured_db) -> None:
    """Re-upserting the same (post_id, model_name) replaces, doesn't insert."""
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        emb_repo = EmbeddingRepository(session)
        emb_repo.upsert(1, "null-8d", _vec(0))
        emb_repo.upsert(1, "null-8d", _vec(1))  # Replace.
    with get_session() as session:
        emb_repo = EmbeddingRepository(session)
        assert emb_repo.count() == 1  # Still one row.
        row = emb_repo.get(1, "null-8d")
        np.testing.assert_array_equal(decode_vector(row.vector), _vec(1))


def test_upsert_many_counts_new_rows(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        for i in range(3):
            post_repo.add(_post(external_id=f"e{i}"))
        emb_repo = EmbeddingRepository(session)
        new_count = emb_repo.upsert_many(
            (i + 1, "null-8d", _vec(i)) for i in range(3)
        )
    assert new_count == 3


def test_get_returns_none_for_missing(configured_db) -> None:
    with get_session() as session:
        emb_repo = EmbeddingRepository(session)
        assert emb_repo.get(999, "null-8d") is None


def test_list_for_model_returns_only_that_model(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        for i in range(3):
            post_repo.add(_post(external_id=f"e{i}"))
        emb_repo = EmbeddingRepository(session)
        emb_repo.upsert(1, "model-a", _vec(0))
        emb_repo.upsert(2, "model-a", _vec(1))
        emb_repo.upsert(3, "model-b", _vec(2))
    with get_session() as session:
        emb_repo = EmbeddingRepository(session)
        a_rows = emb_repo.list_for_model("model-a")
        b_rows = emb_repo.list_for_model("model-b")
    assert len(a_rows) == 2
    assert len(b_rows) == 1
    assert {r.post_id for r in a_rows} == {1, 2}
    assert b_rows[0].post_id == 3


# -------------------------------------------------------------------------
# PostRepository cluster methods
# -------------------------------------------------------------------------
def test_list_unclustered_returns_only_null_cluster_id(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        post_repo.add(_post(external_id="e2"))
        post_repo.add(_post(external_id="e3"))
        post_repo.assign_clusters({1: 0, 2: 0})

    with get_session() as session:
        post_repo = PostRepository(session)
        unclustered = post_repo.list_unclustered()
    assert len(unclustered) == 1
    assert unclustered[0].external_id == "e3"


def test_list_by_cluster(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        post_repo.add(_post(external_id="e2"))
        post_repo.add(_post(external_id="e3"))
        post_repo.assign_clusters({1: 0, 2: 0, 3: 1})

    with get_session() as session:
        post_repo = PostRepository(session)
        c0 = post_repo.list_by_cluster(0)
        c1 = post_repo.list_by_cluster(1)
    assert {p.external_id for p in c0} == {"e1", "e2"}
    assert {p.external_id for p in c1} == {"e3"}


def test_cluster_sizes(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        post_repo.add(_post(external_id="e2"))
        post_repo.add(_post(external_id="e3"))
        post_repo.assign_clusters({1: 0, 2: 0, 3: 1})

    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
    assert sizes == {0: 2, 1: 1}


def test_reset_clusters_zeros_every_post(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        post_repo.add(_post(external_id="e2"))
        post_repo.assign_clusters({1: 0, 2: 1})

    with get_session() as session:
        post_repo = PostRepository(session)
        cleared = post_repo.reset_clusters()

    assert cleared == 2
    with get_session() as session:
        post_repo = PostRepository(session)
        assert post_repo.cluster_sizes() == {}


def test_assign_clusters_empty_mapping_is_noop(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        assert post_repo.assign_clusters({}) == 0


def test_list_ids_without_embeddings_returns_all_when_empty(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        post_repo.add(_post(external_id="e2"))
    with get_session() as session:
        post_repo = PostRepository(session)
        ids = post_repo.list_ids_without_embeddings("null-8d")
    assert sorted(ids) == [1, 2]


def test_list_ids_without_embeddings_skips_embedded(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        post_repo.add(_post(external_id="e1"))
        post_repo.add(_post(external_id="e2"))
        emb_repo = EmbeddingRepository(session)
        emb_repo.upsert(1, "null-8d", _vec(0))
    with get_session() as session:
        post_repo = PostRepository(session)
        ids = post_repo.list_ids_without_embeddings("null-8d")
    assert ids == [2]


def test_list_all_with_embedding_filters_correctly(configured_db) -> None:
    with get_session() as session:
        post_repo = PostRepository(session)
        for i in range(3):
            post_repo.add(_post(external_id=f"e{i}"))
        emb_repo = EmbeddingRepository(session)
        emb_repo.upsert(1, "null-8d", _vec(0))
        emb_repo.upsert(3, "null-8d", _vec(2))
    with get_session() as session:
        post_repo = PostRepository(session)
        embedded = post_repo.list_all_with_embedding("null-8d")
    assert {p.external_id for p in embedded} == {"e0", "e2"}