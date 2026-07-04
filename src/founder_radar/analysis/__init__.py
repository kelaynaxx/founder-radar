"""Analysis layer.

This package holds the post-collection analytical stages. Everything
here is *data → data*: no I/O, no CLI.

Subpackages:
  - `embeddings`         : `BaseEmbedder` + 3 implementations + factory.
  - `vector_store`       : `BaseVectorStore` + `InMemoryVectorStore`.
  - `clustering`         : `BaseClusterer` + `GreedyCosineClusterer`.
  - `scoring`            : 8-factor deterministic score + weighted subscores.
  - `opportunity`        : `HeuristicExtractor` + `LLMBasedExtractor`.
  - `reality_check`      : competitor detection + saturation scoring.
  - `reality_validator`  : viability classification (Phase 3.5).
  - `trends`             : emerging/recurring cluster classification.
"""

from founder_radar.analysis.clustering import (
    BaseClusterer,
    GreedyCosineClusterer,
    build_clusterer,
    cluster_summary,
)
from founder_radar.analysis.embeddings import (
    BaseEmbedder,
    NullEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    l2_normalize,
)
from founder_radar.analysis.opportunity import (
    BaseExtractor,
    HeuristicExtractor,
    LLMBasedExtractor,
    build_extractor,
)
from founder_radar.analysis.reality_check import (
    RealityCheck,
    run_reality_check,
)
from founder_radar.analysis.reality_validator import (
    ALL_STATUSES,
    RealityAssessment,
    STATUS_COMPETITIVE,
    STATUS_SATURATED,
    STATUS_UNDERSERVED,
    STATUS_UNKNOWN,
    assess_reality,
)
from founder_radar.analysis.scoring import (
    ScoreFactors,
    compute_deterministic_scores,
)
from founder_radar.analysis.trends import (
    TrendReport,
    recency_score,
    run_trend_analysis,
)
from founder_radar.analysis.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    load_vectors_into_store,
)

__all__ = [
    # embeddings
    "BaseEmbedder",
    "NullEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
    "l2_normalize",
    # vector store
    "BaseVectorStore",
    "InMemoryVectorStore",
    "load_vectors_into_store",
    # clustering
    "BaseClusterer",
    "GreedyCosineClusterer",
    "build_clusterer",
    "cluster_summary",
    # scoring
    "ScoreFactors",
    "compute_deterministic_scores",
    # opportunity
    "BaseExtractor",
    "HeuristicExtractor",
    "LLMBasedExtractor",
    "build_extractor",
    # reality check (Phase 3+)
    "RealityCheck",
    "run_reality_check",
    # reality validator (Phase 3.5)
    "RealityAssessment",
    "assess_reality",
    "ALL_STATUSES",
    "STATUS_SATURATED",
    "STATUS_COMPETITIVE",
    "STATUS_UNDERSERVED",
    "STATUS_UNKNOWN",
    # trends (Phase 3+)
    "TrendReport",
    "run_trend_analysis",
    "recency_score",
]