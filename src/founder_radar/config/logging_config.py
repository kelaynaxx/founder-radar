"""Logging configuration.

We use the standard library `logging` package so we don't pull in a heavy
dependency just to print timestamps. The configuration is intentionally
simple: a single StreamHandler with a concise formatter for human use, and
an optional FileHandler when `logs_dir` is configured.

`configure_logging()` is called once from the CLI entry point. Tests do not
need to call it.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(level: str = "INFO", logs_dir: Path | None = None) -> None:
    """Configure the root logger with sensible defaults.

    Args:
        level: Log level name (DEBUG/INFO/WARNING/ERROR). Unknown values
            silently fall back to INFO.
        logs_dir: If given, also write logs to `<logs_dir>/founder_radar.log`
            using a rotating file handler (1 MB per file, 3 backups).

    Behavior:
        - Idempotent: removes previously configured handlers so reconfiguring
          in tests does not duplicate output.
        - Sets the level on the root logger so third-party libraries
          (SQLAlchemy, urllib3, ...) respect it too.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers we previously attached (e.g. from a prior test).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Always log to stderr — convenient for CLI users and test capture.
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Optionally also write to a rotating file.
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / "founder_radar.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet down chatty libraries unless the user explicitly asked for DEBUG.
    if numeric_level > logging.DEBUG:
        for noisy in ("urllib3", "httpx", "httpcore", "sqlalchemy.engine"):
            logging.getLogger(noisy).setLevel(logging.WARNING)