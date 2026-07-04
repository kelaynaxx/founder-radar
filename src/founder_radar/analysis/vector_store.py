"""Vector store abstraction.

A vector store holds dense vectors indexed by some key (we use post id)
and supports nearest-neighbor search by cosine similarity.

Why an abstraction in Phase 2?
  - User rule: "pluggable vector store abstraction". Future phases will
    exceed in-memory capacity; swapping to FAISS / Chroma / pgvector
    should be a new file here, not a refactor.
  - Tests use the in-memory implementation; the real pipeline can use it
    in Phase 2 because post volumes are small.
  - Downstream code (`clustering.py`, `main.py --similar`) talks only
    to the base interface.

Design choices:
  - We index by integer post id (the primary key from `posts`). This is
    the natural join key.
  - We store L2-normalized vectors and use inner-product as cosine
    similarity. No need to renormalize on query.
  - `search()` returns results sorted by descending similarity. We do
    not implement approximate NN — for Phase 2 volumes exact NN over a
    numpy matrix is fast and predictable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseVectorStore(ABC):
    """Abstract vector store.

    Concrete implementations:
      - `InMemoryVectorStore` (Phase 2 default): keeps everything in
        a numpy matrix, exact nearest-neighbor via matmul.
      - Future: FAISS / Chroma / pgvector wrappers (Phase 5+).
    """

    @abstractmethod
    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        """Insert or update vectors for the given ids.

        `vectors` must be L2-normalized, shape `(len(ids), dim)`, dtype
        float32. Implementations may silently de-duplicate by id (the
        later vector wins).
        """

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Find the `k` most similar ids to `query`.

        Args:
            query: 1-D L2-normalized float32 vector of shape `(dim,)`.
            k: Number of results to return. Clamped to `len(self)`.

        Returns:
            List of `(id, similarity)` tuples sorted by similarity
            descending. Similarity is in `[-1, 1]` for cosine on
            normalized vectors. Returns an empty list if the store is
            empty or `k <= 0`.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors currently in the store."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of vectors in the store. 0 when empty."""


class InMemoryVectorStore(BaseVectorStore):
    """Numpy-backed vector store with exact nearest-neighbor search.

    Stores vectors in a `(N, dim)` float32 matrix and a parallel list of
    ids. `search()` does a single matrix-vector multiply to get all
    similarities, then argsort to get the top-k.

    Performance:
      - Insertion: O(1) amortized; we re-allocate when capacity doubles.
      - Search: O(N * dim) per query. For N=10,000 and dim=384 this is
        ~4M flops, well under 100ms on a modern CPU.
      - Memory: N * dim * 4 bytes. 10k * 384 = ~15 MB. Fine for Phase 2.

    Limitations (documented, accepted for MVP):
      - Exact NN only. Approximate methods (IVF, HNSW) come with FAISS
        when scale demands it.
      - Not thread-safe. The CLI is single-threaded; if you parallelize,
        wrap with a lock.
    """

    def __init__(self, dim: int | None = None) -> None:
        # We hold vectors as a list of (id, vector) rows. We avoid a single
        # big numpy matrix at insertion time because ids arrive in
        # arbitrary order and reallocating on every add would be wasteful.
        # We materialize a stacked matrix on demand inside `search()`.
        self._ids: list[int] = []
        self._vectors: list[np.ndarray] = []
        self._id_to_index: dict[int, int] = {}
        self._declared_dim = dim

    @property
    def dim(self) -> int:
        if self._declared_dim is not None:
            return self._declared_dim
        if self._vectors:
            return int(self._vectors[0].shape[0])
        return 0

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids ({len(ids)}) and vectors ({vectors.shape[0]}) "
                "must have the same length"
            )
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D, got shape {vectors.shape!r}"
            )
        if vectors.dtype != np.float32:
            raise ValueError(
                f"vectors must be float32, got {vectors.dtype!r}"
            )

        for i, post_id in enumerate(ids):
            vec = vectors[i]
            existing_idx = self._id_to_index.get(post_id)
            if existing_idx is not None:
                # Replace in place; no id-list reordering.
                self._vectors[existing_idx] = vec
            else:
                self._id_to_index[post_id] = len(self._ids)
                self._ids.append(post_id)
                self._vectors.append(vec)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        if query.ndim != 1:
            raise ValueError(
                f"query must be 1-D, got shape {query.shape!r}"
            )
        if query.dtype != np.float32:
            raise ValueError(
                f"query must be float32, got {query.dtype!r}"
            )

        k = min(k, len(self._ids))
        if k <= 0:
            return []

        # Stack into one matmul. `np.stack` of N small arrays is O(N*dim)
        # but that's the cost of the search anyway.
        matrix = np.stack(self._vectors, axis=0)
        # Cosine similarity on normalized vectors == dot product.
        similarities = matrix @ query
        # argpartition is O(N); faster than full sort for small k.
        top_k_idx = np.argpartition(-similarities, k - 1)[:k]
        # But argpartition is unordered; sort the top-k by score descending.
        top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]

        return [
            (int(self._ids[i]), float(similarities[i]))
            for i in top_k_idx
        ]

    def ids(self) -> list[int]:
        """Return a copy of the id list, in insertion order."""
        return list(self._ids)

    def matrix(self) -> np.ndarray:
        """Return the (N, dim) matrix of vectors, in insertion order.

        Useful for clustering and other batch operations. Returns an
        empty `(0, dim)` array when the store is empty.
        """
        if not self._vectors:
            d = self.dim if self.dim else 0
            return np.zeros((0, d), dtype=np.float32)
        return np.stack(self._vectors, axis=0).astype(np.float32, copy=False)


def load_vectors_into_store(
    store: BaseVectorStore,
    *,
    ids: Sequence[int],
    vectors: np.ndarray,
) -> None:
    """Convenience: call `store.add(ids, vectors)`.

    Lives here (not on the store itself) so we can later add logging,
    validation, or batching without changing the store's interface.
    """
    if len(ids) == 0:
        return
    store.add(ids, vectors)