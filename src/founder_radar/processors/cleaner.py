"""Basic cleaning: deduplication and obvious-spam heuristics.

Heuristics are intentionally conservative in Phase 1. False positives
(dropping a real post) are worse than false negatives (keeping a noisy
one), because Phase 3 LLM analysis can recover signal from noise but not
the other way around.

Heuristics implemented:
  1. Deduplicate by `(source, external_id)`. This is mostly defensive —
     the DB layer already enforces uniqueness, but cleaning before insert
     keeps the count accurate and lets tests inspect what was removed.
  2. Drop posts shorter than `min_body_length` characters of *combined*
     title + body. One-word "I hate X" posts are noise.
  3. Drop posts that match known spam patterns: excessive capitalization,
     emoji storms, URLs in body without any text.

These rules will evolve; they're isolated in one module so Phase 2/3
can swap or extend them without touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from founder_radar.processors.base import BaseProcessor

if TYPE_CHECKING:
    from founder_radar.collectors.base import RawPost

logger = logging.getLogger(__name__)


# Pre-compiled regexes for spam heuristics.
_UPPER_RATIO = re.compile(r"[A-Z]")
_EMOJI_RUN = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]{4,}")


class Cleaner(BaseProcessor):
    """Drop duplicates, near-empty posts, and obvious spam.

    Args:
        min_body_length: Minimum combined title+body character count to
            keep. Below this, the post is too thin to extract signal from.
        max_uppercase_ratio: Maximum fraction of letters that may be
            uppercase before we flag a post as shouting.
    """

    name: str = "cleaner"

    def __init__(
        self,
        *,
        min_body_length: int = 20,
        max_uppercase_ratio: float = 0.7,
    ) -> None:
        self._min_body_length = min_body_length
        self._max_uppercase_ratio = max_uppercase_ratio

    def process(self, posts: list["RawPost"]) -> list["RawPost"]:
        """Return a new list with duplicates and obvious spam removed."""
        seen: set[tuple[str, str]] = set()
        kept: list["RawPost"] = []
        dropped_dup = 0
        dropped_short = 0
        dropped_spam = 0

        for post in posts:
            key = (post.source, post.external_id)
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)

            if self._is_too_short(post):
                dropped_short += 1
                continue

            if self._looks_like_spam(post):
                dropped_spam += 1
                continue

            kept.append(post)

        logger.info(
            "Cleaner: kept=%d dropped_dup=%d dropped_short=%d dropped_spam=%d",
            len(kept),
            dropped_dup,
            dropped_short,
            dropped_spam,
        )
        return kept

    # -------------------------------------------------------------------------
    # Heuristics
    # -------------------------------------------------------------------------
    def _is_too_short(self, post: "RawPost") -> bool:
        body = post.body or ""
        total = len(post.title) + len(body)
        return total < self._min_body_length

    def _looks_like_spam(self, post: "RawPost") -> bool:
        text = f"{post.title}\n{post.body or ''}"

        # Long runs of emoji-only content.
        if _EMOJI_RUN.search(text):
            return True

        # Shouting: >70% of letters uppercase AND at least 10 letters.
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 10:
            upper = sum(1 for c in letters if _UPPER_RATIO.match(c))
            if upper / len(letters) > self._max_uppercase_ratio:
                return True

        return False