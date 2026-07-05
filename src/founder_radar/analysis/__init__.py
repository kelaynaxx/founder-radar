"""Analysis layer.

This package holds the post-collection analytical stages. Everything
here is *data → data*: no I/O, no CLI.

Subpackages:
  - `embeddings`         : `BaseEmbedder` + 3 implementations + factory.
  - `vector_store`       : `BaseVectorStore` + `InMemoryVectorStore`.
  - `clustering`         : `BaseClusterer` + `GreedyCosineClusterer`.
  - `scoring`            : 8-factor deterministic score + weighted subscores.
  - `opportunity`        : `HeuristicExtractor` + `LLMBasedExtractor`.
  - `opportunity_type`   : 9-way `opportunity_type` classifier + productizability.
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
from founder_radar.analysis.opportunity_type import (
ALL_TYPES,
OpportunityTypeAssessment,
TYPE_DEVELOPER_WORKFLOW_PAIN,
TYPE_DOCUMENTATION_CONFUSION,
TYPE_INFRA_OPERATIONAL_PAIN,
TYPE_INTEGRATION_PAIN,
TYPE_MISSING_FEATURE,
TYPE_POTENTIAL_PRODUCT,
TYPE_REPO_SPECIFIC_BUG,
TYPE_SECURITY_COMPLIANCE_PAIN,
    TYPE_UNKNOWN,
TYPE_UPSTREAM_LIBRARY_BUG,
classify_opportunity,
)
from founder_radar.analysis.opportunity_review import (
    ALL_REVIEW_REASONS,
    ALL_REVIEW_VERDICTS,
    REVIEW_FAILED_TAG,
    REVIEW_REASON_DOCUMENTATION_ONLY,
    REVIEW_REASON_MAINTENANCE_CHORE,
    REVIEW_REASON_NOT_BUYER_PAIN,
    REVIEW_REASON_POSSIBLE_DEVTOOL,
    REVIEW_REASON_POSSIBLE_INFRA_TOOL,
    REVIEW_REASON_POSSIBLE_MICRO_SAAS,
    REVIEW_REASON_REPO_INTERNAL_TASK,
    REVIEW_REASON_STRONG_REPEATED_PAIN,
    REVIEW_REASON_TOO_REPO_SPECIFIC,
    REVIEW_REASON_TOO_VAGUE,
    REVIEW_REASON_UPSTREAM_BUG,
    REVIEW_VERDICT_MAYBE,
    REVIEW_VERDICT_REJECT,
    REVIEW_VERDICT_STRONG_CANDIDATE,
    ReviewVerdict,
    review_opportunity,
    review_opportunities_batch,
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
    # opportunity type classifier (Phase 4+ signal calibration)
    "OpportunityTypeAssessment",
    "classify_opportunity",
    "ALL_TYPES",
    "TYPE_REPO_SPECIFIC_BUG",
    "TYPE_UPSTREAM_LIBRARY_BUG",
    "TYPE_DOCUMENTATION_CONFUSION",
    "TYPE_MISSING_FEATURE",
    "TYPE_INTEGRATION_PAIN",
    "TYPE_DEVELOPER_WORKFLOW_PAIN",
    "TYPE_INFRA_OPERATIONAL_PAIN",
    "TYPE_SECURITY_COMPLIANCE_PAIN",
    "TYPE_POTENTIAL_PRODUCT",
    "TYPE_UNKNOWN",
    # opportunity review layer (Phase 4+ LLM-assisted triage)
    "ReviewVerdict",
    "review_opportunity",
    "review_opportunities_batch",
    "ALL_REVIEW_VERDICTS",
    "ALL_REVIEW_REASONS",
    "REVIEW_VERDICT_REJECT",
    "REVIEW_VERDICT_MAYBE",
    "REVIEW_VERDICT_STRONG_CANDIDATE",
    "REVIEW_REASON_REPO_INTERNAL_TASK",
    "REVIEW_REASON_UPSTREAM_BUG",
    "REVIEW_REASON_MAINTENANCE_CHORE",
    "REVIEW_REASON_DOCUMENTATION_ONLY",
    "REVIEW_REASON_NOT_BUYER_PAIN",
    "REVIEW_REASON_TOO_VAGUE",
    "REVIEW_REASON_TOO_REPO_SPECIFIC",
    "REVIEW_REASON_POSSIBLE_DEVTOOL",
    "REVIEW_REASON_POSSIBLE_MICRO_SAAS",
    "REVIEW_REASON_POSSIBLE_INFRA_TOOL",
    "REVIEW_REASON_STRONG_REPEATED_PAIN",
    "REVIEW_FAILED_TAG",
]