"""Shared pytest fixtures.

The goal: every test runs against a fresh in-memory SQLite database and a
fresh Settings instance, with no reliance on the real filesystem or
network. This is what makes the test suite pass without API credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from founder_radar.config.settings import Settings, get_settings
from founder_radar.database.connection import get_session, init_engine, reset_for_tests
from founder_radar.database.repository import PostRepository


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Build a Settings instance that points at a temp directory.

    We deliberately construct the object instead of using `get_settings()`
    so tests never read `.env` from the project root.
    """
    return Settings(
        reddit_client_id="test_id",
        reddit_client_secret="test_secret",
        reddit_user_agent="founder-radar-test/0.1",
        database_url=f"sqlite:///{tmp_path}/test.db",
        reports_dir=tmp_path / "reports",
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        scan_limit_per_subreddit=10,
        default_subreddits="test,test2",
        log_level="WARNING",
    )


@pytest.fixture
def configured_db(tmp_settings: Settings) -> Iterator[Settings]:
    """Init the engine on a temp DB and clean up after the test.

    Yields the Settings so tests can read limits, paths, etc. without
    reaching for `get_settings()`.
    """
    # Wipe the cached settings object so nothing leaks across tests.
    get_settings.cache_clear()
    init_engine(tmp_settings.database_url)
    yield tmp_settings
    reset_for_tests()


@pytest.fixture
def repo(configured_db: Settings) -> Iterator[PostRepository]:
    """A PostRepository bound to a clean in-memory session.

    Use this in any test that needs to read/write posts. The session
    commits on successful exit of the `with` block.
    """
    with get_session() as session:
        yield PostRepository(session)


@pytest.fixture
def env_cleanup() -> Iterator[None]:
    """Snapshot/restore env vars so tests don't leak.

    Use when a test needs to set an env var to verify Settings reacts.
    """
    snapshot = os.environ.copy()
    yield
    # Restore: remove any keys added, restore originals.
    for key in list(os.environ):
        if key not in snapshot:
            os.environ.pop(key, None)
    for key, value in snapshot.items():
        os.environ[key] = value