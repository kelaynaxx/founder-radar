"""Reddit collector.

Uses PRAW (Python Reddit API Wrapper) to fetch recent posts from one or
more subreddits.

Auth model:
    PRAW's "script" app flow is what we use: read-only access, identified
    by client_id + client_secret + a unique user_agent. We do NOT need to
    log in as a user; we just want to read public posts.

Rate limits:
    Reddit's OAuth API allows 100 requests/minute for script apps. Each
    call to `subreddit.new()` returns up to 100 items by default, so we
    can fetch ~50 with a single request — well under the limit for the
    Phase 1 scan size.

Failure modes we handle explicitly:
    - Missing credentials: we raise RuntimeError with a clear setup hint.
    - Empty subreddit: PRAW yields nothing; we propagate that as zero
      `RawPost` items (not an error).
    - Network errors: bubble up from PRAW; the CLI catches and logs them.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Iterator

import praw
from prawcore.exceptions import PrawcoreException

from founder_radar.collectors.base import BaseCollector, RawPost

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collects posts from one or more subreddits via PRAW."""

    source_name: str = "reddit"

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(settings)
        # Lazy: we don't authenticate until collect() runs. This way
        # `founder-radar report` doesn't need credentials.
        self._reddit: praw.Reddit | None = None

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def _client(self) -> praw.Reddit:
        """Return a configured PRAW client, building it on first use."""
        if self._reddit is not None:
            return self._reddit

        client_id = self._settings.reddit_client_id
        client_secret = self._settings.reddit_client_secret
        user_agent = self._settings.reddit_user_agent

        if not client_id or not client_secret:
            raise RuntimeError(
                "Reddit credentials missing. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET in your .env (see .env.example)."
            )

        # PRAW's `read_only=True` is the right mode for our use case: we
        # never post, comment, vote, or otherwise mutate state.
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )
        self._reddit.read_only = True
        logger.debug("PRAW client initialized (read-only).")
        return self._reddit

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------
    def collect(
        self,
        *,
        categories: list[str] | None = None,
        limit_per_category: int | None = None,
    ) -> Iterator[RawPost]:
        """Yield `RawPost`s from the requested subreddits.

        Args:
            categories: Subreddit names to scan. Defaults to
                `settings.subreddit_list` when None.
            limit_per_category: Max posts per subreddit. Defaults to
                `settings.scan_limit_per_subreddit`.

        Yields:
            `RawPost` instances in the order Reddit returns them
            (newest first when listing `new`).

        Raises:
            RuntimeError: on missing credentials or PRAW errors.
        """
        subreddits = categories or self._settings.subreddit_list
        limit = limit_per_category or self._settings.scan_limit_per_subreddit

        if not subreddits:
            logger.warning("No subreddits configured; nothing to collect.")
            return

        client = self._client()

        for subreddit_name in subreddits:
            logger.info(
                "Collecting up to %d posts from r/%s ...",
                limit,
                subreddit_name,
            )
            try:
                yield from self._collect_one(client, subreddit_name, limit)
            except PrawcoreException as exc:
                # Don't abort the whole run because one subreddit is private
                # or rate-limited; log and continue.
                logger.error(
                    "PRAW error while scanning r/%s: %s",
                    subreddit_name,
                    exc,
                )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _collect_one(
        self,
        client: praw.Reddit,
        subreddit_name: str,
        limit: int,
    ) -> Iterator[RawPost]:
        """Fetch up to `limit` posts from a single subreddit."""
        subreddit = client.subreddit(subreddit_name)
        # `new` gives us newest-first; suitable for catching fresh
        # complaints before they get drowned in replies.
        for submission in subreddit.new(limit=limit):
            # Defensive: skip anything that doesn't have the fields we need.
            if submission.id is None or submission.title is None:
                continue

            # PRAW exposes `created_utc` as a float; convert to naive UTC
            # datetime to match our database convention.
            created_at = None
            if submission.created_utc:
                created_at = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ).replace(tzinfo=None)

            # `selftext` is the body of a text post; `''` for link posts.
            body = submission.selftext if submission.selftext else None

            # Stash the raw payload for debugging / future enrichment.
            raw_json = json.dumps(
                {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "author": str(submission.author) if submission.author else None,
                    "url": submission.url,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "permalink": submission.permalink,
                    "subreddit": subreddit_name,
                },
                default=str,
            )

            yield RawPost(
                source=self.source_name,
                external_id=submission.id,
                title=submission.title,
                body=body,
                author=str(submission.author) if submission.author else None,
                url=submission.url,
                source_category=subreddit_name,
                score=int(submission.score or 0),
                num_comments=int(submission.num_comments or 0),
                created_at=created_at,
                raw_json=raw_json,
            )