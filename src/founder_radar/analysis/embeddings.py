"""Embedder abstractions.

Every embedder implements `BaseEmbedder`. The pipeline treats them
uniformly: pass a list of texts, get back a `(N, D)` numpy array of
float32 vectors.

Why an abstraction?
  - User rule for Phase 2: "pluggable vector store" and implicit pluggable
    embedder (different teams want different models).
  - A user may have no GPU and prefer OpenAI's API; another may be fully
    offline and want a local model. Both paths produce vectors that look
    identical to downstream code.
  - Tests can swap in `NullEmbedder` (zero vectors, no model load) so the
    full pipeline is testable without a multi-GB dependency.

Conventions:
  - All vectors are L2-normalized. Downstream cosine-similarity code can
    treat `@` as cosine similarity directly. We enforce this here so
    downstream never has to think about it.
  - All vectors are float32. Matches the storage format on disk
    (see `database.repository.encode_vector`).
  - The model name returned by `model_name` is the *exact* string written
    to the `embeddings.model_name` column. Don't fabricate a different one
    per call.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedder interface.

    Implementations must:
      - Return L2-normalized float32 vectors from `embed_texts`.
      - Have a stable `model_name` property used as the storage key.
      - Be deterministic for the same input (or document if not).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable identifier for this embedder, persisted with each embedding."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimensionality."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of input strings. Empty list is allowed and
                returns an empty `(0, dim)` array.

        Returns:
            `(N, dim)` float32 numpy array. Each row is L2-normalized.

        Raises:
            RuntimeError: on backend errors. Callers should not have to
                know which backend they're using.
        """


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row of `vectors` in place-free fashion.

    Returns a new array. Zero-norm rows are left as zeros (we do not
    divide by zero — that would produce NaNs that silently poison
    cosine similarity).
    """
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid divide-by-zero: where norm is 0, leave the row as zeros.
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe_norms
    # Re-zero the rows that were zero (the divide gave 0/1 = 0 anyway,
    # but explicit is better).
    normalized = np.where(norms == 0, 0.0, normalized)
    return normalized.astype(np.float32, copy=False)


class NullEmbedder(BaseEmbedder):
    """Returns deterministic placeholder vectors without loading any model.

    Useful for:
      - Tests that exercise the full embed/cluster pipeline without the
        sentence-transformers dependency.
      - Smoke-checking the rest of the pipeline before the model has
        downloaded.

    The "vectors" are derived from a hash of each input string, mapped
    onto the unit sphere. They are L2-normalized and deterministic — same
    input always yields the same output — but they carry no semantic
    meaning. Clustering on them produces purely lexical groupings.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._model_name = f"null-{dim}d"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        # Deterministic per-text pseudo-vectors: hash text, use the bytes
        # to fill a float32 vector, then L2-normalize. This is not a real
        # embedding — it's a placeholder that lets us exercise the pipeline.
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # np.random with a seeded RandomState gives a reproducible mapping.
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self._dim).astype(np.float32)
        return l2_normalize(out)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedder backed by a sentence-transformers model.

    Loads the model lazily on first call to `embed_texts`. The library
    handles tokenization, batching, and device placement.

    Requires the optional `[embeddings]` extra:
        pip install founder-radar[embeddings]

    We import `sentence_transformers` lazily inside the class so the
    rest of the package imports cleanly even when the optional dep is
    missing.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None  # lazy

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]

    def _ensure_loaded(self) -> None:
        """Load the model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "`pip install founder-radar[embeddings]` to enable "
                "real local embeddings. Alternatively, set "
                "EMBEDDING_BACKEND=null in your .env for placeholder vectors."
            ) from exc
        logger.info("Loading sentence-transformers model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info(
            "Model loaded: dim=%d, max_seq_length=%d",
            self._model.get_sentence_embedding_dimension(),
            self._model.max_seq_length,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        # `convert_to_numpy=True` ensures we get a numpy array out.
        # `normalize_embeddings=True` matches our L2-normalized convention.
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible embeddings via the same HTTP client we already use.

    Hits `<LLM_BASE_URL>/embeddings` with the configured `LLM_API_KEY`. Set
    `EMBEDDING_BACKEND=openai` to use it.

    The model name passed in is sent verbatim as the `model` field, so
    any OpenAI-compatible endpoint that supports `/embeddings` works
    (OpenAI, Azure, Together, vLLM, etc.).
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str,
        api_key: str,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        # OpenAI's text-embedding-3-small is 1536, 3-large is 3072.
        # We don't hardcode dim here — we ask the server on first call.
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError(
                "OpenAIEmbedder.dim is unknown before the first call. "
                "Issue a dummy request to populate it, or set it explicitly."
            )
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim if self._dim else 0), dtype=np.float32)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for OpenAIEmbedder but not importable."
            ) from exc

        url = f"{self._base_url.rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model_name, "input": texts}
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc

        try:
            items = data["data"]
            vectors = np.array(
                [item["embedding"] for item in items], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected embeddings response shape: {exc}"
            ) from exc

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError(
                f"Embeddings response shape mismatch: got {vectors.shape}, "
                f"expected ({len(texts)}, ?)"
            )

        # Cache dim for next call.
        if self._dim is None:
            self._dim = int(vectors.shape[1])

        return l2_normalize(vectors)


def build_embedder(settings) -> BaseEmbedder:  # type: ignore[no-untyped-def]
    """Factory: build the embedder named by `settings.embedding_backend`.

    Used by the CLI. Tests build embedders directly.

    Args:
        settings: A `Settings` instance (typed loosely here to avoid an
            import cycle; the type annotation is documented, not enforced).
    """
    backend = settings.embedding_backend
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if backend == "null":
        return NullEmbedder()
    if backend == "openai":
        return OpenAIEmbedder(
            settings.embedding_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    raise ValueError(
        f"Unknown embedding backend: {backend!r}. "
        "Choose one of: 'sentence-transformers', 'null', 'openai'."
    )