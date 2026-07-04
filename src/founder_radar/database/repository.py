"""Repository layer.

The repository pattern keeps every database operation in one place. The rest
of the codebase asks the repository for data; it never constructs queries
or knows about SQL/SQLAlchemy.

Why?
  - One swap point if we change ORM behavior.
  - Easier to test: repositories can be mocked with an in-memory fake.
  - Avoids the failure mode where business logic slowly fills up with
    ad-hoc session.query() calls.

Phase 1 added `PostRepository`. Phase 2 adds `EmbeddingRepository` and a
handful of cluster-management methods on `PostRepository`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from founder_radar.database.models import (
    Embedding,
    Opportunity,
    OpportunityPost,
    Post,
    _utcnow,
)

import logging
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from founder_radar.database.models import Embedding, Post, _utcnow

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Vector encoding helpers (module-level so both layers can import them)
# -------------------------------------------------------------------------
def encode_vector(vector: np.ndarray) -> bytes:
    """Serialize a 1-D float32 numpy array to portable bytes.

    Uses little-endian float32 so the output is byte-identical on any
    platform. Callers must convert to float32 first if they have a
    different dtype — we do not silently coerce, to avoid silent precision
    loss surprises.
    """
    if vector.ndim != 1:
        raise ValueError(
            f"encode_vector expects a 1-D array, got shape {vector.shape!r}"
        )
    if vector.dtype != np.float32:
        raise ValueError(
            f"encode_vector expects float32, got {vector.dtype!r}; "
            "call .astype(np.float32) first to be explicit."
        )
    return np.ascontiguousarray(vector).tobytes()


def decode_vector(blob: bytes, expected_dim: int | None = None) -> np.ndarray:
    """Inverse of `encode_vector`. Returns a 1-D float32 numpy array.

    Args:
        blob: The bytes previously written by `encode_vector`.
        expected_dim: If given, validates the decoded length matches.
            Mismatches raise `ValueError` so we catch silent corruption.
    """
    arr = np.frombuffer(blob, dtype=np.float32)
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError(
            f"Decoded vector has dim {arr.shape[0]}, expected {expected_dim}"
        )
    return arr


class PostRepository:
    """Read/write access to the `posts` table.

    All methods are stateless beyond the session they wrap. Create one
    repository per session and discard it after the session closes.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    def add(self, post: Post) -> Post:
        """Insert a single post.

        Returns the post with its `id` populated. If the post already exists
        (same `source` + `external_id`), we return the existing row instead
        of inserting, so re-runs are idempotent.

        Implementation note:
            We check for existence with a SELECT before inserting rather
            than catching IntegrityError on insert. The catch-and-rollback
            approach has a subtle bug: rolling back after a duplicate error
            also rolls back earlier successful inserts in the same session.
            Pre-checking is portable across SQLite and PostgreSQL and keeps
            the session state clean.
        """
        existing = self.get_by_source_id(post.source, post.external_id)
        if existing is not None:
            # Sync the caller's reference so `post.id` matches the DB row.
            post.id = existing.id
            logger.debug(
                "Duplicate post skipped: %s/%s",
                post.source,
                post.external_id,
            )
            return existing

        self._session.add(post)
        self._session.flush()
        return post

    def add_many(self, posts: Iterable[Post]) -> int:
        """Bulk insert. Returns the number of *new* rows added.

        Duplicates are detected at flush time; we let SQLAlchemy emit
        IntegrityError, roll back, and continue. Simpler implementations
        (e.g. one commit at the end) would be cleaner, but bulk performance
        is irrelevant for Phase 1 volumes.
        """
        count = 0
        for post in posts:
            before_id = post.id
            self.add(post)
            if post.id is not None and post.id != before_id:
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------
    def get_by_id(self, post_id: int) -> Post | None:
        return self._session.get(Post, post_id)

    def get_by_source_id(self, source: str, external_id: str) -> Post | None:
        """Look up a post by its source-native id. Used for dedup."""
        stmt = select(Post).where(
            Post.source == source,
            Post.external_id == external_id,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_all(self, limit: int | None = None) -> Sequence[Post]:
        """Return all posts, newest first. Optionally capped."""
        stmt = select(Post).order_by(Post.collected_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return self._session.execute(stmt).scalars().all()

    def list_by_source(
        self,
        source: str,
        limit: int | None = None,
    ) -> Sequence[Post]:
        stmt = (
            select(Post)
            .where(Post.source == source)
            .order_by(Post.collected_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return self._session.execute(stmt).scalars().all()

    def list_by_source_category(
        self,
        source: str,
        source_category: str,
        limit: int | None = None,
    ) -> Sequence[Post]:
        """Return posts from a specific bucket, e.g. one subreddit."""
        stmt = (
            select(Post)
            .where(Post.source == source, Post.source_category == source_category)
            .order_by(Post.collected_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return self._session.execute(stmt).scalars().all()

    def list_collected_between(
        self,
        start: datetime,
        end: datetime,
    ) -> Sequence[Post]:
        stmt = (
            select(Post)
            .where(Post.collected_at >= start, Post.collected_at <= end)
            .order_by(Post.collected_at.desc())
        )
        return self._session.execute(stmt).scalars().all()

    def list_ids_without_embeddings(self, model_name: str) -> Sequence[int]:
        """Return ids of posts that have NO embedding under `model_name`.

        Implemented as a NOT EXISTS subquery so SQLite and PostgreSQL
        behave identically.
        """
        from founder_radar.database.models import Embedding as Emb

        stmt = select(Post.id).where(
            ~select(Emb.id)
            .where(Emb.post_id == Post.id, Emb.model_name == model_name)
            .exists()
        ).order_by(Post.id)
        return self._session.execute(stmt).scalars().all()

    def list_all_with_embedding(self, model_name: str) -> Sequence[Post]:
        """Return every post that has an embedding under `model_name`, ordered by id."""
        from founder_radar.database.models import Embedding as Emb

        stmt = (
            select(Post)
            .where(
                select(Emb.id)
                .where(Emb.post_id == Post.id, Emb.model_name == model_name)
                .exists()
            )
            .order_by(Post.id)
        )
        return self._session.execute(stmt).scalars().all()

    # -------------------------------------------------------------------------
    # Aggregates
    # -------------------------------------------------------------------------
    def count(self) -> int:
        return self._session.execute(
            select(func.count()).select_from(Post)
        ).scalar_one()

    # -------------------------------------------------------------------------
    # Clustering (Phase 2)
    # -------------------------------------------------------------------------
    def list_unclustered(self) -> Sequence[Post]:
        """Return posts that have not been clustered yet (`cluster_id IS NULL`)."""
        stmt = (
            select(Post)
            .where(Post.cluster_id.is_(None))
            .order_by(Post.id)
        )
        return self._session.execute(stmt).scalars().all()

    def list_by_cluster(self, cluster_id: int) -> Sequence[Post]:
        """Return every post in the given cluster."""
        stmt = (
            select(Post)
            .where(Post.cluster_id == cluster_id)
            .order_by(Post.id)
        )
        return self._session.execute(stmt).scalars().all()

    def cluster_sizes(self) -> dict[int, int]:
        """Return `{cluster_id: post_count}` for every cluster that has posts."""
        stmt = (
            select(Post.cluster_id, func.count(Post.id))
            .where(Post.cluster_id.is_not(None))
            .group_by(Post.cluster_id)
            .order_by(Post.cluster_id)
        )
        return {
            cid: count
            for cid, count in self._session.execute(stmt).all()
        }

    def reset_clusters(self) -> int:
        """Set every post's `cluster_id` to NULL. Returns row count affected.

        Called before a fresh `cluster` run so deleted clusters don't keep
        stale labels.
        """
        result = self._session.execute(
            update(Post).values(cluster_id=None)
        )
        return int(result.rowcount or 0)

    def assign_clusters(self, assignments: dict[int, int]) -> int:
        """Bulk-assign `cluster_id` for each post id in `assignments`.

        Args:
            assignments: `{post_id: cluster_id}`. Both sides are ints.

        Returns:
            Number of rows updated.
        """
        if not assignments:
            return 0
        # One UPDATE per (post_id, cluster_id) pair. We accept O(n) round-trips
        # here because the dataset at this stage is small (hundreds, not
        # millions) and simplicity beats micro-optimization.
        count = 0
        for post_id, cluster_id in assignments.items():
            result = self._session.execute(
                update(Post)
                .where(Post.id == post_id)
                .values(cluster_id=cluster_id)
            )
            count += int(result.rowcount or 0)
        return count


class EmbeddingRepository:
    """Read/write access to the `embeddings` table.

    Vectors are stored as little-endian float32 bytes. Use
    `decode_vector(emb.vector)` to get a numpy array back; use
    `encode_vector(np.ndarray)` to go the other way. Keeping that
    conversion in one place makes it trivial to swap the storage format
    later (e.g. to a native pgvector type) without touching callers.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    def upsert(
        self,
        post_id: int,
        model_name: str,
        vector: np.ndarray,
    ) -> Embedding:
        """Insert or replace an embedding for (post_id, model_name).

        We use a SELECT-then-INSERT/UPDATE pattern (rather than dialect-
        specific `INSERT ... ON CONFLICT`) so the code stays portable across
        SQLite and PostgreSQL.
        """
        vector_bytes = encode_vector(vector)
        dim = int(vector.shape[0])

        existing = self._session.execute(
            select(Embedding).where(
                Embedding.post_id == post_id,
                Embedding.model_name == model_name,
            )
        ).scalar_one_or_none()

        if existing is not None:
            existing.vector = vector_bytes
            existing.dim = dim
            existing.created_at = _utcnow()
            self._session.flush()
            return existing

        emb = Embedding(
            post_id=post_id,
            model_name=model_name,
            dim=dim,
            vector=vector_bytes,
        )
        self._session.add(emb)
        self._session.flush()
        return emb

    def upsert_many(
        self,
        items: Iterable[tuple[int, str, np.ndarray]],
    ) -> int:
        """Bulk upsert. Returns the count of *new* rows.

        Tuples are `(post_id, model_name, vector)`. Same semantics as
        `upsert()` for each item.
        """
        new_count = 0
        for post_id, model_name, vector in items:
            before = self._session.execute(
                select(Embedding.id).where(
                    Embedding.post_id == post_id,
                    Embedding.model_name == model_name,
                )
            ).scalar_one_or_none()
            self.upsert(post_id, model_name, vector)
            if before is None:
                new_count += 1
        return new_count

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------
    def get(self, post_id: int, model_name: str) -> Embedding | None:
        stmt = select(Embedding).where(
            Embedding.post_id == post_id,
            Embedding.model_name == model_name,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_for_model(self, model_name: str) -> Sequence[Embedding]:
        """Return every embedding produced by `model_name`, ordered by post_id."""
        stmt = (
            select(Embedding)
            .where(Embedding.model_name == model_name)
            .order_by(Embedding.post_id)
        )
        return self._session.execute(stmt).scalars().all()

    def count(self) -> int:
        return self._session.execute(
            select(func.count()).select_from(Embedding)
        ).scalar_one()
# -------------------------------------------------------------------------
# JSON list helpers for Opportunity columns
# -------------------------------------------------------------------------
def _encode_json_list(values: Sequence[str] | None) -> str | None:
    if values is None:
        return None
    return json.dumps(list(values), ensure_ascii=False)


def _decode_json_list(blob: str | None) -> list[str]:
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except (ValueError, TypeError) as exc:
        logger.warning("Failed to decode JSON list from DB: %s", exc)
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data]


# -------------------------------------------------------------------------
# OpportunityRepository
# -------------------------------------------------------------------------
class OpportunityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, opp: Opportunity, post_ids: Iterable[int] = ()) -> Opportunity:
        self._session.add(opp)
        self._session.flush()
        for pid in post_ids:
            self._session.add(
                OpportunityPost(
                    opportunity_id=opp.id,
                    post_id=pid,
                )
            )
        self._session.flush()
        return opp

    def add_from_dict(self, data: dict, post_ids: Iterable[int] = ()) -> Opportunity:
        kwargs = dict(data)
        if "saas_ideas" in kwargs:
            kwargs["saas_ideas_json"] = _encode_json_list(kwargs.pop("saas_ideas"))
        if "competitors" in kwargs:
            kwargs["competitors_json"] = _encode_json_list(kwargs.pop("competitors"))
        if "source_links" in kwargs:
            kwargs["source_links_json"] = _encode_json_list(kwargs.pop("source_links"))
        opp = Opportunity(**kwargs)
        return self.add(opp, post_ids=post_ids)

    def delete_for_cluster(self, cluster_id: int) -> int:
        opp_ids_subq = (
            select(Opportunity.id).where(Opportunity.cluster_id == cluster_id)
        )
        self._session.execute(
            delete(OpportunityPost).where(
                OpportunityPost.opportunity_id.in_(opp_ids_subq)
            )
        )
        opp_del = self._session.execute(
            delete(Opportunity).where(Opportunity.cluster_id == cluster_id)
        )
        return int(opp_del.rowcount or 0)

    def update_status(self, opportunity_id: int, status: str) -> bool:
        result = self._session.execute(
            update(Opportunity)
            .where(Opportunity.id == opportunity_id)
            .values(status=status)
        )
        return (result.rowcount or 0) > 0

    def get_by_id(self, opportunity_id: int) -> Opportunity | None:
        return self._session.get(Opportunity, opportunity_id)

    def list_all(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        sort_by_weighted: bool = True,
    ) -> Sequence[Opportunity]:
        # Phase 3+ default sort: weighted_score (pain-dominated).
        # Pass sort_by_weighted=False to fall back to total_score.
        sort_col = (
            Opportunity.weighted_score if sort_by_weighted
            else Opportunity.total_score
        )
        stmt = select(Opportunity).order_by(sort_col.desc())
        if status is not None:
            stmt = stmt.where(Opportunity.status == status)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self._session.execute(stmt).scalars().all()

    def list_by_cluster(self, cluster_id: int) -> Sequence[Opportunity]:
        stmt = (
            select(Opportunity)
            .where(Opportunity.cluster_id == cluster_id)
            .order_by(Opportunity.weighted_score.desc())
        )
        return self._session.execute(stmt).scalars().all()

    def list_post_ids(self, opportunity_id: int) -> list[int]:
        stmt = select(OpportunityPost.post_id).where(
            OpportunityPost.opportunity_id == opportunity_id
        ).order_by(OpportunityPost.post_id)
        return list(self._session.execute(stmt).scalars().all())

    def count(self) -> int:
        return self._session.execute(
            select(func.count()).select_from(Opportunity)
        ).scalar_one()

    def saas_ideas(self, opp: Opportunity) -> list[str]:
        return _decode_json_list(opp.saas_ideas_json)

    def competitors(self, opp: Opportunity) -> list[str]:
        return _decode_json_list(opp.competitors_json)

    def source_links(self, opp: Opportunity) -> list[str]:
        return _decode_json_list(opp.source_links_json)