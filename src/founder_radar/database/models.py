"""SQLAlchemy ORM models.

We use the SQLAlchemy 2.x *declarative* style. Every model inherits from
`Base` (declared below) which is wired to the engine in `connection.py`.

Phase 1 added `Post`. Phase 2 adds `Embedding` and a `cluster_id` column
on `Post`. Phase 3 will add `Opportunity` and a `Cluster` join model.

Schema design notes:
  - `id` is a surrogate autoincrement integer; external systems reference
    posts by `(source, external_id)` which together are unique.
  - `raw_json` keeps the original payload so we can re-derive fields later
    (e.g. when we want to extract comment trees we previously ignored).
  - Timestamps are stored as naive UTC. SQLAlchemy's DateTime with
    `timezone=False` is portable across SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Naive UTC `datetime` for portable timestamp storage.

    SQLite has no native timezone-aware datetimes, so we standardize on
    naive UTC everywhere and document the convention in the docstring.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models.

    Having a single Base keeps metadata in one place: `Base.metadata` is what
    `connection.create_all()` uses to materialize tables.
    """


class Post(Base):
    """A single discussion item collected from any source.

    A *post* is the atomic unit the pipeline operates on: one Reddit thread,
    one Hacker News story, one GitHub issue, etc. The fields below capture
    the intersection of what every source exposes — anything source-specific
    goes into `raw_json`.

    Identity:
        A post is uniquely identified by `(source, external_id)` so we can
        safely re-collect from a source without creating duplicates even if
        the autoincrement `id` differs.
    """

    __tablename__ = "posts"
    __table_args__ = (
        # Composite uniqueness: same id under the same source is one row.
        UniqueConstraint("source", "external_id", name="uq_post_source_external_id"),
    )

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True,
        doc="Where this post came from: 'reddit', 'hackernews', 'github', ...",
    )
    external_id: Mapped[str] = mapped_column(
        String(64), nullable=False,
        doc="Source-native id, e.g. Reddit submission id 'abc123'.",
    )
    source_category: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True,
        doc="Bucket inside the source: subreddit name, HN category, repo, ...",
    )

    # -------------------------------------------------------------------------
    # Content
    # -------------------------------------------------------------------------
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(String(128), nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # -------------------------------------------------------------------------
    # Engagement
    # -------------------------------------------------------------------------
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_comments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # -------------------------------------------------------------------------
    # Timestamps
    # -------------------------------------------------------------------------
    # `created_at` is when the post was created on the source side.
    # `collected_at` is when our collector pulled it into our DB.
    # Phase 4+ (thread-aware grouping, currently HN-only) - generic
    # nullable metadata fields. Old rows have NULL here; new rows are
    # populated by the HN collector.
    thread_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        doc=(
            "Logical thread root id. For HN stories: the story's HN id. "
            "For HN comments: the root story's HN id. NULL for non-HN "
            "sources or posts predating the thread-aware change."
        ),
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
        doc=(
            "Direct parent id. For HN: the comment's parent field. "
            "NULL for top-level stories. Useful for comment sub-threads."
        ),
    )
    item_type: Mapped[str | None] = mapped_column(
        String(16), nullable=True,
        doc=(
            "Item kind from the source API: 'story', 'comment', 'job', etc. "
            "NULL for non-HN sources or posts predating the change."
        ),
    )

    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False,
    )

    # -------------------------------------------------------------------------
    # Original payload
    # -------------------------------------------------------------------------
    raw_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="Full original API response, JSON-encoded, for debugging/replay.",
    )

    # -------------------------------------------------------------------------
    # Clustering (Phase 2)
    # -------------------------------------------------------------------------
    # `cluster_id` is set by the analysis layer. NULL means the post has
    # not yet been clustered. We store the id on the row itself (rather
    # than a join table) because cluster membership is 1:1 — every post
    # belongs to at most one cluster — and the CLI inspection tools read
    # this column constantly.
    cluster_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True,
        doc="Cluster label assigned by analysis.clustering. NULL = not yet clustered.",
    )

    def __repr__(self) -> str:
        return (
            f"Post(id={self.id!r}, source={self.source!r}, "
            f"external_id={self.external_id!r}, title={self.title!r})"
        )


class Embedding(Base):
    """A dense vector representation of a post produced by an embedder.

    Stored in a separate table from `Post` for three reasons:
      1. We may swap embedding models over time. The `model_name` column
         lets us keep historical embeddings alongside newer ones.
      2. Multiple embeddings per post (one per model) are first-class.
      3. The `posts` row stays focused on source-side content.

    The vector is stored as little-endian float32 bytes. This is portable
    across SQLite and PostgreSQL. When we eventually migrate to PostgreSQL,
    this column can be replaced by a native `vector` type (pgvector) without
    changing the application code.

    Identity:
        A post may have at most one embedding per `model_name` (the
        composite unique constraint). Re-embedding the same post with the
        same model replaces the row.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("post_id", "model_name", name="uq_embedding_post_model"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True,
        doc="FK to posts.id. We do not declare a SQL FK because SQLite ON "
            "DELETE CASCADE is awkward in declarative models; the repository "
            "enforces referential integrity in Python.",
    )
    model_name: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
        doc="Embedder identifier, e.g. 'sentence-transformers/all-MiniLM-L6-v2'.",
    )
    dim: Mapped[int] = mapped_column(
        Integer, nullable=False,
        doc="Vector dimensionality. Stored explicitly so we can sanity-check "
            "vectors on load.",
    )
    vector: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False,
        doc="Vector bytes in little-endian float32. Decode with numpy.frombuffer(..., dtype=np.float32).",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"Embedding(id={self.id!r}, post_id={self.post_id!r}, "
            f"model_name={self.model_name!r}, dim={self.dim!r})"
        )
class Opportunity(Base):
    """A software business opportunity extracted from one or more posts.

    One opportunity corresponds to one *problem* identified across a
    cluster of posts. It carries the LLM- or heuristic-derived content
    fields (title, problem summary, target audience, SaaS ideas, ...)
    plus the 8-factor scoring fields.

    Scoring fields are all on `[0, 1]`. `total_score` is the mean.
    `confidence_score` is a meta-score: how much of the data was filled
    in by a real LLM (high) vs estimated by the heuristic (low).

    Lifecycle:
        `status` is one of:
          - "new"          just extracted, not yet reviewed
          - "confirmed"    human-curated, ready for action
          - "dismissed"    not worth pursuing
          - "archived"     superseded or no longer relevant

    Identity:
        We don't enforce uniqueness at the SQL level. The CLI's
        `extract` command is idempotent: it deletes prior opportunities
        for a cluster before inserting the new one.
    """

    __tablename__ = "opportunities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    problem_summary: Mapped[str] = mapped_column(Text, nullable=False)
    target_audience: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Free-form fields stored as JSON-encoded text so SQLite and PostgreSQL
    # behave identically. The repository provides typed accessors that
    # decode on read and encode on write.
    saas_ideas_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="JSON list[str] of SaaS / product ideas.",
    )
    competitors_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="JSON list[str] of existing competitor names / products.",
    )
    source_links_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="JSON list[str] of original post URLs.",
    )

    # 8-factor scoring. All on [0, 1].
    frequency_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    emotional_intensity_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    dissatisfaction_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    market_size_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    ease_of_implementation_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    recurring_revenue_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    technical_feasibility_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    novelty_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    total_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    confidence_score: Mapped[float] = mapped_column(default=0.0, nullable=False)

    # Phase 3+ re-calibrated scoring. weighted_score is the default
    # ranking key (pain-dominated per the brief). See
    # analysis/scoring.py for the formula.
    pain_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    monetization_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    weighted_score: Mapped[float] = mapped_column(
        default=0.0, nullable=False, index=True,
        doc="Pain-dominated ranking score. Higher = better opportunity.",
    )

    # Phase 3+ Reality Check fields. Saturation measures how crowded the
    # market appears (0 = open, 1 = saturated).
    saturation_score: Mapped[float] = mapped_column(default=0.0, nullable=False)
    distinct_competitor_count: Mapped[int] = mapped_column(default=0, nullable=False)
    competitor_mention_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Phase 3+ Trend classification. Set by the trend analyzer.
    # One of: 'emerging', 'stable', 'declining', 'recurring', 'unknown'.
    trend: Mapped[str] = mapped_column(
        String(16), default="unknown", nullable=False, index=True,
        doc="'emerging' | 'stable' | 'declining' | 'recurring' | 'unknown'.",
    )

    # Phase 3.5 Reality Validation Layer. Orthogonal to scoring —
    # tells us whether the opportunity is actually viable, not just
    # whether it ranks well. See analysis/reality_validator.py.
    reality_status: Mapped[str] = mapped_column(
        String(16), default="unknown", nullable=False, index=True,
        doc=(
            "'saturated' | 'competitive' | 'underserved' | 'unknown'. "
            "Computed by the Reality Validator; does NOT affect "
            "weighted_score ranking."
        ),
    )
    reality_confidence: Mapped[float] = mapped_column(
        default=0.0, nullable=False,
        doc="[0,1] How strongly the data supports the chosen reality_status.",
    )
    competitor_strength_estimate: Mapped[float] = mapped_column(
        default=0.0, nullable=False,
        doc=(
            "[0,1] Numeric summary of competitor signal (count + lexicon "
            "+ density). Independent of saturation_score."
        ),
    )
    confidence_score: Mapped[float] = mapped_column(default=0.0, nullable=False)

    # Origin
    cluster_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True,
        doc="Cluster this opportunity came from. NULL if extracted without clustering.",
    )
    mentions: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        doc="Number of source posts (number of mentions of this problem).",
    )
    extraction_method: Mapped[str] = mapped_column(
        String(32), default="heuristic", nullable=False,
        doc="'heuristic' or 'llm' — which extractor produced this row.",
    )
    llm_model: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
        doc="LLM model name if extracted via LLM. NULL otherwise.",
    )
    status: Mapped[str] = mapped_column(
        String(16), default="new", nullable=False,
        doc="'new', 'confirmed', 'dismissed', 'archived'.",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"Opportunity(id={self.id!r}, title={self.title!r}, "
            f"total_score={self.total_score!r}, status={self.status!r})"
        )


class OpportunityPost(Base):
    """M:N link between opportunities and the source posts they cite.

    In practice this is mostly 1:N (one opportunity references many
    posts), but the M:N shape leaves room for re-clustering a single
    post under multiple opportunities later.

    Composite primary key on (opportunity_id, post_id) prevents
    duplicate links.
    """

    __tablename__ = "opportunity_posts"
    __table_args__ = (
        # Composite PK — same opportunity can cite the same post at most once.
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True,
        doc="FK to opportunities.id (enforced in the repository).",
    )
    post_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True,
        doc="FK to posts.id (enforced in the repository).",
    )

    def __repr__(self) -> str:
        return (
            f"OpportunityPost(opportunity_id={self.opportunity_id!r}, "
            f"post_id={self.post_id!r})"
        )