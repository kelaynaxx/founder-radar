"""Tests for the analysis/clustering module."""

from __future__ import annotations

import numpy as np
import pytest

from founder_radar.analysis.clustering import (
    GreedyCosineClusterer,
    build_clusterer,
    cluster_summary,
)


def _unit(v):
    """L2-normalize a 1-D vector, return as float32."""
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    if n == 0:
        return a
    return a / n


def test_cluster_empty_returns_empty() -> None:
    clusterer = GreedyCosineClusterer(similarity_threshold=0.5)
    labels = clusterer.cluster(np.zeros((0, 4), dtype=np.float32))
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


def test_cluster_single_vector_gets_one_cluster() -> None:
    clusterer = GreedyCosineClusterer(similarity_threshold=0.5)
    labels = clusterer.cluster(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    )
    assert labels.tolist() == [0]


def test_cluster_identical_vectors_share_cluster() -> None:
    """Three identical vectors should land in one cluster."""
    clusterer = GreedyCosineClusterer(similarity_threshold=0.9)
    vectors = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)
    )
    labels = clusterer.cluster(vectors)
    assert set(labels.tolist()) == {0}
    assert labels.tolist() == [0, 0, 0]


def test_cluster_orthogonal_vectors_get_separate_clusters() -> None:
    """Vectors pointing in opposite directions should not share a cluster."""
    clusterer = GreedyCosineClusterer(similarity_threshold=0.5)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = clusterer.cluster(vectors)
    # Cosine similarity of opposites is -1, well below 0.5.
    assert len(set(labels.tolist())) == 2


def test_cluster_threshold_separates_near_misses() -> None:
    """A vector just below the threshold starts a new cluster."""
    clusterer = GreedyCosineClusterer(similarity_threshold=0.99)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],  # cosine sim ~ 0.9999 — passes
            [0.5, 0.5, 0.0],    # cosine sim ~ 0.707 — fails
        ],
        dtype=np.float32,
    )
    # Normalize so the math is what we expect.
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = clusterer.cluster(vectors.astype(np.float32))
    assert labels[0] == labels[1]  # First two are similar.
    assert labels[0] != labels[2]  # Third is far from the first.


def test_cluster_is_deterministic() -> None:
    """Same input → same labels, every time."""
    clusterer = GreedyCosineClusterer(similarity_threshold=0.7)
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    a = clusterer.cluster(vectors)
    b = clusterer.cluster(vectors)
    np.testing.assert_array_equal(a, b)


def test_cluster_rejects_non_2d_input() -> None:
    clusterer = GreedyCosineClusterer(similarity_threshold=0.5)
    with pytest.raises(ValueError, match="2-D"):
        clusterer.cluster(np.zeros((4,), dtype=np.float32))


def test_threshold_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        GreedyCosineClusterer(similarity_threshold=1.5)
    with pytest.raises(ValueError, match="similarity_threshold"):
        GreedyCosineClusterer(similarity_threshold=-0.1)


def test_cluster_name_includes_threshold() -> None:
    clusterer = GreedyCosineClusterer(similarity_threshold=0.73)
    assert "0.73" in clusterer.name
    assert clusterer.similarity_threshold == 0.73


def test_build_clusterer_reads_settings(tmp_settings) -> None:
    clusterer = build_clusterer(tmp_settings)
    assert isinstance(clusterer, GreedyCosineClusterer)
    assert clusterer.similarity_threshold == tmp_settings.cluster_similarity_threshold


def test_cluster_summary_counts_per_label() -> None:
    labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    summary = cluster_summary(labels)
    assert summary == {0: 2, 1: 3, 2: 1}


def test_cluster_summary_empty() -> None:
    labels = np.array([], dtype=np.int64)
    assert cluster_summary(labels) == {}