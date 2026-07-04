"""Database layer.

Three concerns, three modules:

- `models`     : SQLAlchemy ORM declarative classes (Post, Opportunity, ...)
- `connection` : engine + session factory; dialect-agnostic
- `repository` : all read/write operations; everything else uses the repo

SQLite is the default in Phase 1. Swapping to PostgreSQL in a later phase is
a single environment variable change because we never write dialect-specific
SQL by hand.
"""