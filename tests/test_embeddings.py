"""Tests for the analysis/embeddings module.

We only exercise `NullEmbedder` here — testing the sentence-transformers
backend would require the optional `embeddings` dep to be installed.
The Null embedder gives us full pipeline coverage without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from founder_radar.analysis.embeddings import (
    NullEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    l2_normalize,
)


def test_l2_normalize_unit_norm_rows() -> None:
    """Each row of the output should have unit L2 norm."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 16)).astype(np.float32)
    normalized = l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_preserves_direction() -> None:
    """Normalizing then dot-producting equals cosine similarity."""
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((4, 8)).astype(np.float32)
    normalized = l2_normalize(vectors)
    # If rows are unit-norm, dot product == cosine similarity.
    sims = normalized @ normalized.T
    # Diagonal should be 1.
    assert np.allclose(np.diag(sims), 1.0, atol=1e-5)


def test_l2_normalize_handles_zero_vector() -> None:
    """A zero vector stays zero (no NaN propagation)."""
    arr = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(arr)
    assert np.allclose(out, np.zeros_like(arr))
    assert not np.any(np.isnan(out))


def test_l2_normalize_empty() -> None:
    """Empty input returns empty float32 array."""
    arr = np.zeros((0, 4), dtype=np.float32)
    out = l2_normalize(arr)
    assert out.shape == (0, 4)
    assert out.dtype == np.float32


def test_null_embedder_returns_correct_shape() -> None:
    emb = NullEmbedder(dim=64)
    vectors = emb.embed_texts(["hello", "world", "another text"])
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32


def test_null_embedder_empty_input() -> None:
    emb = NullEmbedder(dim=32)
    vectors = emb.embed_texts([])
    assert vectors.shape == (0, 32)
    assert vectors.dtype == np.float32


def test_null_embedder_is_deterministic() -> None:
    """Same text always yields the same vector."""
    emb = NullEmbedder(dim=64)
    a = emb.embed_texts(["repeatable text"])[0]
    b = emb.embed_texts(["repeatable text"])[0]
    np.testing.assert_array_equal(a, b)


def test_null_embedder_output_is_l2_normalized() -> None:
    emb = NullEmbedder(dim=64)
    vectors = emb.embed_texts(["x", "y", "z"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_null_embedder_distinct_inputs_differ() -> None:
    emb = NullEmbedder(dim=128)
    vectors = emb.embed_texts(["alpha", "beta", "gamma"])
    # Pairwise: not all zero, not all the same.
    assert not np.allclose(vectors[0], vectors[1])
    assert not np.allclose(vectors[1], vectors[2])


def test_null_embedder_model_name_reflects_dim() -> None:
    emb = NullEmbedder(dim=128)
    assert emb.model_name == "null-128d"
    assert emb.dim == 128


def test_build_embedder_picks_backend(tmp_settings) -> None:
    """The factory returns the right concrete type per backend name."""
    # sentence-transformers (lazy: not loaded yet)
    tmp_settings.embedding_backend = "sentence-transformers"
    emb = build_embedder(tmp_settings)
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.model_name == tmp_settings.embedding_model

    # null
    tmp_settings.embedding_backend = "null"
    emb = build_embedder(tmp_settings)
    assert isinstance(emb, NullEmbedder)


def test_build_embedder_rejects_unknown_backend(tmp_settings) -> None:
    tmp_settings.embedding_backend = "made-up-backend"
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        build_embedder(tmp_settings)


def test_sentence_transformer_embedder_missing_dep(tmp_settings) -> None:
    """Without the optional dep installed, .dim should raise a clear error."""
    emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    # The dep is not installed in the test environment. .dim triggers a
    # model load, which should fail with a helpful message — or succeed
    # if the user happens to have the dep. We accept both: only assert
    # that no unexpected exception type leaks.
    try:
        _ = emb.dim
    except RuntimeError as exc:
        # The expected failure mode.
        assert "sentence-transformers is not installed" in str(exc) or \
               "sentence-transformers" in str(exc)
    except ImportError:
        # Also acceptable: the dep might be partially installed.
        pass