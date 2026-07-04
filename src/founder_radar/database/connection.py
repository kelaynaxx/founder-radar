"""Database engine and session factory.

`init_engine()` reads the database URL from settings, configures engine
options appropriate for the dialect (SQLite needs `check_same_thread=False`
for multi-threaded use), and stashes the result on a module-level singleton.

Usage pattern (everywhere outside `connection.py`):

    from founder_radar.database.connection import get_session
    from founder_radar.database.repository import PostRepository

    with get_session() as session:
        repo = PostRepository(session)
        repo.add(...)

The session context manager handles commit/rollback automatically.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from founder_radar.database.models import Base

logger = logging.getLogger(__name__)

# Module-level singletons. Populated by `init_engine`. Tests can call
# `init_engine(... test_url ...)` to point at a temporary database.
_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def init_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Initialize the global engine + session factory.

    Args:
        database_url: SQLAlchemy URL, e.g. `sqlite:///data/founder_radar.db`
            or `postgresql+psycopg://...`.
        echo: If True, log every SQL statement (very noisy; debug only).

    Returns:
        The created engine. Subsequent calls return the same engine and
        merely reconfigure logging.

    Side effects:
        - Creates the engine bound to `database_url`.
        - Calls `Base.metadata.create_all(engine)` to materialize tables
          that do not yet exist. Safe to call repeatedly; existing tables
          are untouched.
    """
    global _engine, _SessionFactory

    connect_args: dict = {}
    # SQLite-specific tweaks. We detect SQLite by URL prefix rather than
    # dialect introspection because the engine hasn't been built yet.
    if database_url.startswith("sqlite"):
        # Allow the engine to be shared across threads (CLI + background jobs).
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        echo=echo,
        future=True,
        connect_args=connect_args,
    )

    _SessionFactory = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )

    # Materialize tables. We use create_all (not Alembic) intentionally in
    # Phase 1 — the schema is tiny and migrations would be over-engineering.
    # Alembic is the planned upgrade path for when schema changes start to
    # matter (Phase 3+).
    Base.metadata.create_all(_engine)

    logger.info("Database engine initialized: %s", database_url)
    return _engine


def get_engine() -> Engine:
    """Return the initialized engine. Raises if `init_engine` was not called."""
    if _engine is None:
        raise RuntimeError(
            "Database engine not initialized. Call init_engine() first "
            "(the CLI does this at startup)."
        )
    return _engine


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy Session with automatic commit/rollback.

    On normal exit: commits the transaction.
    On exception: rolls back and re-raises.

    Why a context manager?
      - Guarantees the session closes (releases connection to the pool).
      - Makes commit/rollback explicit but automatic.
      - Plays nicely with `with` for readability.
    """
    if _SessionFactory is None:
        raise RuntimeError(
            "Session factory not initialized. Call init_engine() first."
        )

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_for_tests() -> None:
    """Drop and recreate all tables. Test-only convenience.

    Lets each test start from a clean slate without juggling fixtures.
    """
    if _engine is None:
        raise RuntimeError("Call init_engine() before reset_for_tests().")
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)