"""Tests for OpportunityRepository (Phase 3)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from founder_radar.database.connection import get_session
from founder_radar.database.models import Opportunity, Post
from founder_radar.database.repository import (
    EmbeddingRepository,
    OpportunityRepository,
    PostRepository,
    encode_vector,
)


def _post(external_id: str = "p", title: str = "Title long enough to pass cleaner") -> Post:
    return Post(
        source="reddit",
        external_id=external_id,
        source_category="entrepreneur",
        title=title,
        body="Body text",
        author="op",
        url=None,
        score=1,
        num_comments=1,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


def _seed_three_posts(configured_db) -> None:
    """Insert three posts so opportunities have something to link to."""
    import numpy as np
    with get_session() as session:
        repo = PostRepository(session)
        for i in range(3):
            repo.add(_post(external_id=f"p{i}"))
        emb_repo = EmbeddingRepository(session)
        # Drop in an embedding so list_all_with_embedding has something
        # to find (not strictly needed for these tests, but it keeps
        # the fixture representative of a real Phase 2 state).
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        emb_repo.upsert(1, "null-8d", v)


# -------------------------------------------------------------------------
# add_from_dict
# -------------------------------------------------------------------------
def test_add_from_dict_basic(configured_db) -> None:
    _seed_three_posts(configured_db)
    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {
                "title": "Test opportunity",
                "problem_summary": "People need this.",
                "total_score": 0.7,
                "confidence_score": 0.5,
                "cluster_id": 0,
                "mentions": 1,
                "extraction_method": "heuristic",
                "llm_model": None,
                "status": "new",
            },
            post_ids=[1, 2],
        )
    assert opp.id is not None
    assert opp.title == "Test opportunity"


def test_add_from_dict_serializes_json_lists(configured_db) -> None:
    _seed_three_posts(configured_db)
    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {
                "title": "t",
                "problem_summary": "p",
                "saas_ideas": ["Idea A", "Idea B"],
                "competitors": ["X"],
                "source_links": ["https://a", "https://b"],
                "extraction_method": "heuristic",
            },
        )
    with get_session() as session:
        repo = OpportunityRepository(session)
        # The list accessors should decode the JSON.
        ideas = repo.saas_ideas(opp)
        comps = repo.competitors(opp)
        links = repo.source_links(opp)
    assert ideas == ["Idea A", "Idea B"]
    assert comps == ["X"]
    assert links == ["https://a", "https://b"]


def test_add_from_dict_with_none_lists(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {
                "title": "t",
                "problem_summary": "p",
                "saas_ideas": None,
                "competitors": None,
                "source_links": None,
                "extraction_method": "heuristic",
            },
        )
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.saas_ideas(opp) == []
        assert repo.competitors(opp) == []
        assert repo.source_links(opp) == []


# -------------------------------------------------------------------------
# list / get
# -------------------------------------------------------------------------
def test_list_all_orders_by_total_score_desc(configured_db) -> None:
    _seed_three_posts(configured_db)
    with get_session() as session:
        repo = OpportunityRepository(session)
        repo.add_from_dict({"title": "low", "problem_summary": "p",
                            "weighted_score": 0.2, "extraction_method": "heuristic"})
        repo.add_from_dict({"title": "high", "problem_summary": "p",
                            "weighted_score": 0.9, "extraction_method": "heuristic"})
        repo.add_from_dict({"title": "mid", "problem_summary": "p",
                            "weighted_score": 0.5, "extraction_method": "heuristic"})
    with get_session() as session:
        repo = OpportunityRepository(session)
        titles = [o.title for o in repo.list_all()]
    assert titles == ["high", "mid", "low"]


def test_list_all_with_status_filter(configured_db) -> None:
    _seed_three_posts(configured_db)
    with get_session() as session:
        repo = OpportunityRepository(session)
        repo.add_from_dict({"title": "a", "problem_summary": "p",
                            "status": "new", "extraction_method": "heuristic"})
        repo.add_from_dict({"title": "b", "problem_summary": "p",
                            "status": "dismissed", "extraction_method": "heuristic"})
    with get_session() as session:
        repo = OpportunityRepository(session)
        new = repo.list_all(status="new")
        dismissed = repo.list_all(status="dismissed")
    assert [o.title for o in new] == ["a"]
    assert [o.title for o in dismissed] == ["b"]


def test_get_by_id_returns_none_for_missing(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.get_by_id(9999) is None


def test_count_starts_at_zero(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.count() == 0


# -------------------------------------------------------------------------
# delete_for_cluster
# -------------------------------------------------------------------------
def test_delete_for_cluster_removes_matching_opportunities(configured_db) -> None:
    _seed_three_posts(configured_db)
    with get_session() as session:
        repo = OpportunityRepository(session)
        repo.add_from_dict({"title": "in-0", "problem_summary": "p",
                            "cluster_id": 0, "extraction_method": "heuristic"})
        repo.add_from_dict({"title": "in-1", "problem_summary": "p",
                            "cluster_id": 1, "extraction_method": "heuristic"})
        repo.add_from_dict({"title": "in-0-2", "problem_summary": "p",
                            "cluster_id": 0, "extraction_method": "heuristic"})

    with get_session() as session:
        repo = OpportunityRepository(session)
        removed = repo.delete_for_cluster(0)
    assert removed == 2

    with get_session() as session:
        repo = OpportunityRepository(session)
        # Only the cluster_id=1 one remains.
        assert repo.count() == 1
        assert repo.list_all()[0].title == "in-1"


def test_delete_for_cluster_no_op_when_empty(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.delete_for_cluster(99) == 0


# -------------------------------------------------------------------------
# update_status
# -------------------------------------------------------------------------
def test_update_status_changes_status(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {"title": "t", "problem_summary": "p", "status": "new",
             "extraction_method": "heuristic"}
        )
    with get_session() as session:
        repo = OpportunityRepository(session)
        ok = repo.update_status(opp.id, "confirmed")
    assert ok
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.get_by_id(opp.id).status == "confirmed"


def test_update_status_returns_false_for_missing(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.update_status(9999, "confirmed") is False


# -------------------------------------------------------------------------
# list_post_ids (the join table)
# -------------------------------------------------------------------------
def test_list_post_ids_returns_linked_posts(configured_db) -> None:
    _seed_three_posts(configured_db)
    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {"title": "t", "problem_summary": "p",
             "extraction_method": "heuristic"},
            post_ids=[1, 3],
        )
    with get_session() as session:
        repo = OpportunityRepository(session)
        ids = repo.list_post_ids(opp.id)
    assert sorted(ids) == [1, 3]


def test_list_post_ids_empty_when_no_links(configured_db) -> None:
    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {"title": "t", "problem_summary": "p",
             "extraction_method": "heuristic"}
        )
    with get_session() as session:
        repo = OpportunityRepository(session)
        assert repo.list_post_ids(opp.id) == []


# -------------------------------------------------------------------------
# Defensive decoding
# -------------------------------------------------------------------------
def test_decode_handles_corrupt_json(configured_db) -> None:
    """A corrupted JSON blob shouldn't crash the report renderer."""
    with get_session() as session:
        repo = OpportunityRepository(session)
        # Insert directly with a broken JSON blob.
        opp = Opportunity(
            title="t",
            problem_summary="p",
            saas_ideas_json="not json",
            competitors_json="also not json",
            source_links_json="[broken",
            extraction_method="heuristic",
        )
        repo.add(opp)
    with get_session() as session:
        repo = OpportunityRepository(session)
        # All return [] on corruption; never raise.
        assert repo.saas_ideas(opp) == []
        assert repo.competitors(opp) == []
        assert repo.source_links(opp) == []