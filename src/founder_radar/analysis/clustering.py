"""Clustering algorithms.

We ship exactly one algorithm in Phase 2: greedy single-pass threshold-
based clustering with cosine similarity. It is intentionally simple.

Algorithm:
  1. For each unassigned post i (in id order):
       - If i is the first unassigned post, start cluster 0 with i.
       - Otherwise start the next cluster with i.
       - For every later unassigned post j, if cosine_similarity(i, j)
         >= threshold, add j to the same cluster.
  2. Return labels[i] for every post.

Complexity:
  - O(n^2) time, O(n) memory. For n=1000 posts, ~1M comparisons — sub-second
    on any modern CPU. For n=10000, ~100M comparisons — tens of seconds.
    When we hit that scale we'll swap in HDBSCAN.

Trade-offs (documented for future maintainers):
  - Chaining: if A-B and B-C are similar but A-C is not, all three end up
    in one cluster. Average-link agglomerative would fix this; we accept
    the limitation for the MVP.
  - Determinism: depends on input order. We sort by post id so re-runs
    produce identical labels.
  - No "noise" concept: every post joins a cluster. DBSCAN-style noise
    would be a future enhancement.

Why a base class at all?
  - User rule: "pluggable". A future HDBSCAN wrapper is a new subclass.
  - Keeps the CLI's `cluster` command backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClusterer(ABC):
    """Abstract clustering algorithm.

    Concrete implementations take a `(N, dim)` matrix of L2-normalized
    vectors and return a `(N,)` integer array of cluster labels.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logs and CLI output."""

    @abstractmethod
    def cluster(self, vectors: np.ndarray) -> np.ndarray:
        """Assign a cluster id to each row of `vectors`.

        Args:
            vectors: `(N, dim)` float32 array, L2-normalized. `N=0`
                returns an empty array.

        Returns:
            `(N,)` int64 array of cluster labels in `[0, k)` where `k`
            is the number of distinct clusters. Labels are deterministic
            for a given input.
        """


class GreedyCosineClusterer(BaseClusterer):
    """Single-pass greedy clusterer with cosine similarity threshold.

    Args:
        similarity_threshold: Minimum cosine similarity for two posts to
            share a cluster. Higher values produce tighter, smaller
            clusters. `0.75` is a reasonable default for sentence-
            transformer embeddings.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )
        self._threshold = float(similarity_threshold)

    @property
    def name(self) -> str:
        return f"greedy-cosine-{self._threshold:.2f}"

    @property
    def similarity_threshold(self) -> float:
        return self._threshold

    def cluster(self, vectors: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D, got shape {vectors.shape!r}"
            )

        labels = np.full(n, -1, dtype=np.int64)
        next_cluster = 0

        # We compute pairwise similarities row-by-row instead of materializing
        # the full N×N matrix. For N=10000 that's 100M floats (~400MB) we
        # avoid; the row-by-row version is O(N^2) compute but O(N) memory.
        for i in range(n):
            if labels[i] != -1:
                continue

            # Seed a new cluster with row i.
            labels[i] = next_cluster

            # Compare against every later unassigned row.
            # `unassigned_j = j > i and labels[j] == -1`
            # Only compare forward; symmetric relations are handled by the
            # outer loop picking up i when it gets to j.
            later_unassigned = labels[i + 1:] == -1
            if not np.any(later_unassigned):
                next_cluster += 1
                continue

            # Slice the relevant rows once.
            later_vecs = vectors[i + 1:]
            sims = later_vecs @ vectors[i]  # cosine sim for normalized vectors

            # The threshold is a *similarity*, so we want sim >= threshold.
            join_mask = sims >= self._threshold
            join_mask &= later_unassigned

            join_indices = np.where(join_mask)[0]
            labels[i + 1 + join_indices] = next_cluster
            next_cluster += 1

        return labels


def build_clusterer(settings) -> BaseClusterer:  # type: ignore[no-untyped-def]
    """Factory: build the clusterer the CLI uses.

    Args:
        settings: `Settings` instance. Reads
            `settings.cluster_similarity_threshold`.

    Currently always returns `GreedyCosineClusterer`. New algorithms
    would extend this factory to pick the right one.
    """
    return GreedyCosineClusterer(
        similarity_threshold=settings.cluster_similarity_threshold,
    )


def cluster_summary(
    labels: np.ndarray,
    ids: np.ndarray | None = None,
) -> dict[int, int]:
    """Return `{cluster_id: post_count}` from a labels array.

    Args:
        labels: `(N,)` cluster labels from `BaseClusterer.cluster()`.
        ids: Optional `(N,)` post ids parallel to `labels`. If given,
            the returned dict keys are still integer labels; ids are not
            used directly but the parameter is accepted for symmetry
            with other helpers.

    Returns:
        Dict mapping cluster_id (int) to count (int). Cluster ids with
        zero members are not present.
    """
    if labels.size == 0:
        return {}
    # `np.bincount` is the fastest way to count occurrences.
    counts = np.bincount(labels.astype(np.int64, copy=False))
    return {int(i): int(c) for i, c in enumerate(counts) if c > 0}