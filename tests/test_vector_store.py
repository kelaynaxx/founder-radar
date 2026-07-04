"""Tests for the analysis/vector_store module."""

from __future__ import annotations

import numpy as np
import pytest

from founder_radar.analysis.vector_store import InMemoryVectorStore


def _vec(seed: int, dim: int = 4) -> np.ndarray:
    """Build a deterministic L2-normalized test vector.

    `seed` selects the random direction so each call yields a distinct
    unit vector. (Earlier versions of this test used `_vec(0.1)` style
    scalars which all normalized to the same direction; this version
    uses a seed for distinctness.)
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def test_empty_store_is_empty() -> None:
    store = InMemoryVectorStore(dim=4)
    assert len(store) == 0
    assert store.dim == 4
    assert store.search(_vec(1), k=5) == []


def test_add_increases_len() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1, 2, 3], np.stack([_vec(1), _vec(2), _vec(3)], axis=0))
    assert len(store) == 3


def test_search_returns_top_k_sorted() -> None:
    store = InMemoryVectorStore(dim=4)
    # Three distinct unit vectors along known axes:
    #   10 along +x, 20 along +y, 30 at 45 degrees in the xy plane.
    store.add(
        [10, 20, 30],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.7, 0.7, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    # Searching for +y; id=20 (pure +y) should win.
    query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, k=3)
    assert len(results) == 3
    sims = [sim for _, sim in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0][0] == 20


def test_search_clamps_k_to_len() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1], np.stack([_vec(1)], axis=0))
    results = store.search(_vec(1), k=10)
    assert len(results) == 1


def test_search_with_zero_k_returns_empty() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1], np.stack([_vec(1)], axis=0))
    assert store.search(_vec(1), k=0) == []


def test_add_replaces_existing_id() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1, 2], np.stack([_vec(1), _vec(2)], axis=0))
    store.add([1], np.stack([_vec(99)], axis=0))
    assert len(store) == 2
    results = store.search(_vec(99), k=1)
    assert results[0][0] == 1


def test_add_rejects_mismatched_lengths() -> None:
    store = InMemoryVectorStore(dim=4)
    with pytest.raises(ValueError, match="same length"):
        store.add([1, 2], np.zeros((3, 4), dtype=np.float32))


def test_add_rejects_wrong_dtype() -> None:
    store = InMemoryVectorStore(dim=4)
    with pytest.raises(ValueError, match="float32"):
        store.add([1], np.zeros((1, 4), dtype=np.float64))


def test_add_rejects_wrong_ndim() -> None:
    store = InMemoryVectorStore(dim=4)
    # 2 ids, but the vector is 3-D; the length check passes (2 == 2)
    # so the ndim check fires.
    with pytest.raises(ValueError, match="2-D"):
        store.add([1, 2], np.zeros((2, 4, 1), dtype=np.float32))


def test_search_rejects_wrong_query_ndim() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1], np.stack([_vec(1)], axis=0))
    with pytest.raises(ValueError, match="1-D"):
        store.search(np.zeros((1, 4), dtype=np.float32))


def test_search_rejects_wrong_query_dtype() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1], np.stack([_vec(1)], axis=0))
    with pytest.raises(ValueError, match="float32"):
        store.search(np.zeros((4,), dtype=np.float64))


def test_ids_returns_insertion_order() -> None:
    store = InMemoryVectorStore(dim=4)
    ids = [10, 20, 30, 40]
    store.add(ids, np.stack([_vec(i) for i in ids], axis=0))
    assert store.ids() == ids


def test_matrix_returns_stacked_array() -> None:
    store = InMemoryVectorStore(dim=4)
    store.add([1, 2], np.stack([_vec(1), _vec(2)], axis=0))
    matrix = store.matrix()
    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32


def test_matrix_empty_store_returns_zero_dim() -> None:
    store = InMemoryVectorStore(dim=8)
    matrix = store.matrix()
    assert matrix.shape == (0, 8)
    assert matrix.dtype == np.float32


def test_dim_inferred_from_first_vector() -> None:
    store = InMemoryVectorStore()  # no declared dim
    assert store.dim == 0
    store.add([1], np.zeros((1, 16), dtype=np.float32))
    assert store.dim == 16