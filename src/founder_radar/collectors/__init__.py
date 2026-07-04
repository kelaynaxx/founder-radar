"""Collectors layer.

A *collector* is anything that fetches raw discussion items from an external
source (Reddit, Hacker News, GitHub Issues, ...). Every collector inherits
from `BaseCollector` so the pipeline can treat all sources uniformly.

Phase 1 ships with `RedditCollector`. Phase 2+ adds `HackerNewsCollector`
(public HN Firebase API, no auth). New sources are added by dropping a
new module under this package and registering it in the registry.
"""
from founder_radar.collectors.base import (
    BaseCollector,
    CollectorRegistry,
    RawPost,
    registry,
    register_builtins,
)

# Concrete collectors are re-exported for convenience. Importing this
# module does NOT trigger PRAW loading because `register_builtins()` is
# called lazily (only when the CLI or registry needs it). HN has no
# such lazy boundary (httpx is small) but we still import the class
# here for the public re-export.
from founder_radar.collectors.hackernews import (  # noqa: E402
    HackerNewsCollector,
    HN_STORY_TYPES,
)
from founder_radar.collectors.reddit import RedditCollector  # noqa: E402

__all__ = [
    "BaseCollector",
    "CollectorRegistry",
    "HackerNewsCollector",
    "HN_STORY_TYPES",
    "RawPost",
    "RedditCollector",
    "registry",
    "register_builtins",
]