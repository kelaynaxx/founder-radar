"""Abstract base for every discussion source.

A `BaseCollector` is the contract every concrete source (Reddit, Hacker News,
GitHub Issues, ...) must implement. The pipeline treats them uniformly:
`for collector in registry.all(): collector.collect(...)`.

Why a base class instead of duck typing?
  - Documents the contract in one place.
  - Static type checkers catch missing methods at "compile time".
  - Future AI agents extending the project see the exact shape they need
    to provide.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from founder_radar.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RawPost:
    """Source-agnostic representation of a discussion item.

    Fields are populated by collectors from whatever the source API returns
    and consumed by the database layer to build `Post` ORM rows.

    Using a dataclass (instead of passing PRAW's `Submission` objects
    directly) decouples downstream code from any one source library.
    """

    source: str
    external_id: str
    title: str
    body: str | None = None
    author: str | None = None
    url: str | None = None
    source_category: str | None = None  # subreddit name, repo name, ...
    score: int = 0
    num_comments: int = 0
    created_at: datetime | None = None
    raw_json: str | None = field(default=None, repr=False)

    # Phase 4+ thread metadata. The HN collector populates these;
    # other sources leave them NULL.
    thread_id: str | None = None
    parent_id: str | None = None
    item_type: str | None = None
    # Phase 4+ calibration tag. The HN collector derives this from the
    # title prefix and item type (e.g. "Ask HN:" -> "ask_hn"). Used by
    # downstream code to downrank pure Show HN launches, etc. See the
    # Post.subtype column docstring for the full taxonomy.
    subtype: str | None = None


class BaseCollector(ABC):
    """Abstract base class for every discussion source.

    Subclasses must:
        1. Set the class attribute `source_name` (e.g. "reddit").
        2. Implement `collect()` returning an iterator of `RawPost`.

    The CLI and registry discover collectors by importing the modules under
    `founder_radar.collectors` and looking for `BaseCollector` subclasses.
    """

    # Subclasses MUST override this with the source identifier (lowercase).
    # It is used as the `source` column in the database, so changing it
    # later would break dedup.
    source_name: str = ""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    @abstractmethod
    def collect(
        self,
        *,
        categories: list[str] | None = None,
        limit_per_category: int | None = None,
    ) -> Iterator[RawPost]:
        """Fetch raw posts from the source.

        Args:
            categories: Source-specific buckets to scan. For Reddit this is
                subreddit names. For Hacker News it might be category tags.
                If None, the collector should use whatever is configured in
                settings.
            limit_per_category: Cap on items per category. If None, fall
                back to `settings.scan_limit_per_subreddit` (or the
                source-equivalent default).

        Yields:
            `RawPost` instances. The collector is allowed to yield zero
            items (e.g. credentials missing, source unavailable).

        Raises:
            NotImplementedError: if the subclass did not override this.
            Source-specific errors: subclasses should wrap network errors
                in a clear `RuntimeError` with a user-friendly message.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(source_name={self.source_name!r})"


class CollectorRegistry:
    """In-memory registry of available collectors.

    We don't use a plugin system (entry points, dynamic imports) yet — there
    is exactly one collector and the complexity isn't worth it. When Phase 5
    adds more sources, we can promote this to entry-point discovery without
    touching the collectors themselves.
    """

    def __init__(self) -> None:
        self._collectors: dict[str, type[BaseCollector]] = {}

    def register(self, collector_cls: type[BaseCollector]) -> None:
        """Register a collector class. Idempotent on repeated registrations."""
        name = collector_cls.source_name
        if not name:
            raise ValueError(
                f"Collector {collector_cls.__name__} has empty source_name; "
                "set the class attribute."
            )
        self._collectors[name] = collector_cls

    def get(self, name: str) -> type[BaseCollector]:
        if name not in self._collectors:
            raise KeyError(
                f"Unknown collector: {name!r}. "
                f"Available: {sorted(self._collectors)}"
            )
        return self._collectors[name]

    def all_names(self) -> list[str]:
        return sorted(self._collectors)

    def all(self) -> list[type[BaseCollector]]:
        return [self._collectors[n] for n in self.all_names()]


# Module-level singleton. Populated by `register_builtins()` in the package
# `__init__.py`. Keeping it on the module lets tests override it cleanly.
registry = CollectorRegistry()


def register_builtins() -> None:
    """Register every collector shipped with the package.

    Imported lazily so the `__init__.py` doesn't pull heavy libraries
    (PRAW, etc.) just because someone imported the package.
    """
    from founder_radar.collectors.github import GitHubIssuesCollector
    from founder_radar.collectors.hackernews import HackerNewsCollector
    from founder_radar.collectors.reddit import RedditCollector

    registry.register(GitHubIssuesCollector)
    registry.register(HackerNewsCollector)
    registry.register(RedditCollector)

    registry.register(RedditCollector)