"""Hacker News collector.

Uses the public HN Firebase API (no authentication required):

  https://hacker-news.firebaseio.com/v0/{story_type}.json
  https://hacker-news.firebaseio.com/v0/item/{id}.json

Supported story types:
  topstories, newstories, askstories, showstories, beststories, jobstories

Per-item fields we read:
  id, type, by, time, title, url, text, score, descendants, kids,
  deleted, dead

We skip items where:
  - 'deleted' or 'dead' is true
  - 'type' isn't 'story' or 'job' (when collecting stories)
  - 'title' is missing (stories and jobs must have one)

Comments (opt-in via `--include-comments`):
  - First-level comments only (kids of the story).
  - Capped at 5 comments per story to bound HTTP cost.
  - Comments don't carry titles; we synthesize one from the first
    non-empty text line so downstream rendering still works.

Failure modes we handle explicitly:
  - HTTP error on the list endpoint -> logged, loop continues.
  - HTTP error on one item fetch -> that item is skipped, others
    continue.
  - Item body `null` (deleted) or `dead` true -> skipped silently.
  - Unexpected `type` (e.g. poll) -> skipped.
  - Missing required fields -> skipped.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Iterator

import httpx

from founder_radar.collectors.base import BaseCollector, RawPost

logger = logging.getLogger(__name__)


# HN Firebase endpoints. Public, no auth.
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"

# Story-type names accepted by `collect(categories=...)`. Order matters
# only insofar as it controls the order in which feeds are processed.
HN_STORY_TYPES = (
    "topstories",
    "newstories",
    "askstories",
    "showstories",
    "beststories",
    "jobstories",
)

# Default cap on comments fetched per story when --include-comments is set.
_MAX_COMMENTS_PER_STORY = 5

_DEFAULT_USER_AGENT = (
    "founder-radar/0.1 (HN collector; see project README for contact)"
)

# Compile regexes once.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITIES: dict[str, str] = {
    "&#x27;": "'",
    "&#x2F;": "/",
    "&quot;": '"',
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
}


class HackerNewsCollector(BaseCollector):
    """Pulls stories (and optionally comments) from Hacker News.

    Designed to be a drop-in second source alongside `RedditCollector`.
    Both inherit from `BaseCollector`; both implement `collect(...)`.

    Construction:
        HackerNewsCollector(settings)
        HackerNewsCollector(settings, include_comments=True)

    The collector holds an httpx.Client lazily. Tests inject a mock by
    patching `_client` to return an httpx.Client(transport=httpx.MockTransport(...)).
    """

    source_name: str = "hackernews"

    def __init__(
        self,
        settings,
        *,
        include_comments: bool = False,
    ) -> None:
        super().__init__(settings)
        self._include_comments = include_comments
        self._session: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        """Return a configured httpx.Client, built on first use.

        Lazy so that constructing the collector does not require network
        access. Tests inject a transport via `patch.object(...,
        "_client", return_value=mock_client)`.
        """
        if self._session is not None:
            return self._session
        ua = getattr(
            self._settings, "hackernews_user_agent", _DEFAULT_USER_AGENT
        )
        self._session = httpx.Client(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": ua},
            # HN expects polite bots; one connection at a time is fine
            # for our volumes.
            limits=httpx.Limits(max_connections=1),
        )
        return self._session

    def collect(
        self,
        *,
        categories: list[str] | None = None,
        limit_per_category: int | None = None,
    ) -> Iterator[RawPost]:
        """Yield RawPost objects from each requested HN story type.

        Args:
            categories: Story type names (e.g. ["topstories", "askstories"]).
                When None or empty, falls back to
                `settings.hn_story_type_list`.
            limit_per_category: Max items to fetch per story type. When
                None, falls back to `settings.scan_limit_per_subreddit`
                (renamed semantically — keeps config surface small).
        """
        story_types = categories or self._settings.hn_story_type_list
        limit = (
            limit_per_category
            or getattr(self._settings, "scan_limit_per_subreddit", 50)
        )

        if not story_types:
            logger.warning("No HN story types configured; nothing to collect.")
            return

        client = self._client()

        for story_type in story_types:
            if story_type not in HN_STORY_TYPES:
                logger.warning(
                    "Unknown HN story type %r; skipping. "
                    "Valid types: %s",
                    story_type, list(HN_STORY_TYPES),
                )
                continue
            logger.info(
                "Collecting up to %d stories from %s ...",
                limit, story_type,
            )
            try:
                yield from self._collect_one(client, story_type, limit)
            except httpx.HTTPError as exc:
                # Network glitch on one feed shouldn't kill the whole run.
                logger.error(
                    "HN HTTP error fetching %s: %s", story_type, exc
                )
            except Exception as exc:
                # Last-resort catch — log and keep going so a single
                # weird item can't break the pipeline.
                logger.exception(
                    "Unexpected error fetching %s: %s", story_type, exc
                )

    # ---------------------------------------------------------------------
    # Per-feed collection
    # ---------------------------------------------------------------------
    def _collect_one(
        self,
        client: httpx.Client,
        story_type: str,
        limit: int,
    ) -> Iterator[RawPost]:
        """Fetch one story-type feed and yield RawPosts."""
        ids = self._fetch_story_ids(client, story_type)
        if not ids:
            return
        # Truncate up-front so we don't issue useless HTTP calls.
        for sid in ids[:limit]:
            try:
                item = self._fetch_item(client, sid)
            except httpx.HTTPError as exc:
                logger.warning("Skipping HN item %s: %s", sid, exc)
                continue
            if item is None:
                continue
            post = self._item_to_raw_post(item, story_type)
            if post is None:
                continue
            yield post

            if self._include_comments:
                yield from self._collect_comments(
                    client, item, story_type, thread_root_id=sid
                )

    def _collect_comments(
        self,
        client: httpx.Client,
        parent: dict,
        source_category: str,
        *,
        thread_root_id: int | str,
    ) -> Iterator[RawPost]:
        """Yield up to `_MAX_COMMENTS_PER_STORY` first-level comments.

        Every yielded comment has its `thread_id` set to `thread_root_id`
        so a downstream thread-aware clusterer can group all of a story's
        comments (and the story itself) under the same cluster.
        """
        kids = parent.get("kids", []) or []
        if not kids:
            return
        for kid_id in kids[:_MAX_COMMENTS_PER_STORY]:
            try:
                comment = self._fetch_item(client, kid_id)
            except httpx.HTTPError as exc:
                logger.debug("Skip comment %s: %s", kid_id, exc)
                continue
            if comment is None:
                continue
            comment_post = self._item_to_raw_post(
                comment,
                source_category,
                is_comment=True,
                thread_root_id=thread_root_id,
            )
            if comment_post is not None:
                yield comment_post

    # ---------------------------------------------------------------------
    # HTTP helpers
    # ---------------------------------------------------------------------
    def _fetch_story_ids(
        self, client: httpx.Client, story_type: str
    ) -> list[int]:
        """GET the story-type endpoint, returning a list of item ids.

        Returns an empty list on any failure (HTTP error or unexpected
        payload) so the caller can iterate zero items gracefully.
        """
        url = f"{HN_API_BASE}/{story_type}.json"
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch %s: %s", url, exc)
            return []
        try:
            data = response.json()
        except (ValueError, TypeError) as exc:
            logger.error("Bad JSON from %s: %s", url, exc)
            return []
        # The endpoint returns e.g. [123, 456, ...] or null on outages.
        if not isinstance(data, list):
            return []
        return [int(x) for x in data if isinstance(x, (int, float))]

    def _fetch_item(
        self, client: httpx.Client, item_id: int | str
    ) -> dict | None:
        """GET one item, returning the parsed dict (or None on skip).

        Returns None when:
          - The item body is `null` (deleted) — HN returns this for
            tombstoned items.
          - The item has `deleted: true` or `dead: true`.
          - The response is not a dict (defensive against odd payloads).
        """
        url = f"{HN_API_BASE}/item/{item_id}.json"
        response = client.get(url)
        response.raise_for_status()
        item = response.json()

        if not isinstance(item, dict):
            return None
        if item.get("deleted") or item.get("dead"):
            return None
        return item

    # ---------------------------------------------------------------------
    # HN item -> RawPost
    # ---------------------------------------------------------------------
    def _item_to_raw_post(
        self,
        item: dict,
        source_category: str,
        *,
        is_comment: bool = False,
        thread_root_id: int | str | None = None,
    ) -> RawPost | None:
        """Translate one HN item dict into our RawPost shape.

        Returns None if the item lacks fields we require.
        """
        item_id = item.get("id")
        title: str | None = None
        body: str | None = None

        if is_comment:
            # Comments don't carry a title. Synthesize from the first
            # non-empty line of the comment text so the report still
            # shows something useful.
            text = item.get("text", "") or ""
            if not text:
                return None
            title = _first_meaningful_line(text) or f"comment {item_id}"
            body = text
        else:
            # Stories and jobs both have `title`. Skip items without one
            # — they're either deleted or not actually stories.
            title = item.get("title")
            body = item.get("text") or None  # Ask HN / Show HN text

        if not title or item_id is None:
            return None

        # For non-comment collection we only want stories and jobs.
        # Polls, pollopts, etc. are skipped.
        item_type = item.get("type", "")
        if not is_comment and item_type not in ("story", "job"):
            return None

        # Unix timestamp -> naive UTC (our DB convention).
        hn_time = item.get("time")
        created_at = None
        if isinstance(hn_time, (int, float)):
            try:
                created_at = datetime.fromtimestamp(
                    hn_time, tz=timezone.utc
                ).replace(tzinfo=None)
            except (OverflowError, OSError, ValueError):
                created_at = None

        # Ask HN / Show HN text posts often have no external `url`.
        # The HN discussion page itself is a reasonable fallback.
        url = item.get("url") or (
            f"https://news.ycombinator.com/item?id={item_id}"
        )

        score = _safe_int(item.get("score"), default=0)
        num_comments = _safe_int(item.get("descendants"), default=0)

        # raw_json: a stable JSON string for replay / debugging.
        raw_json = None
        try:
            raw_json = json.dumps(item, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            raw_json = None

        # Phase 4+ thread metadata.
        # - Story: thread_id = own id, item_type = "story",
        #          parent_id = None.
        # - Comment: thread_id = the root story's id (passed in
        #   via `thread_root_id`), parent_id = the comment's parent
        #   (which is usually the story for top-level comments).
        if is_comment:
            final_thread_id = str(thread_root_id) if thread_root_id is not None else None
            parent_id = str(item.get("parent", "")) or None
        else:
            final_thread_id = str(item_id)
            parent_id = None
        final_item_type = item_type or None

        return RawPost(
            source=self.source_name,
            external_id=str(item_id),
            source_category=source_category,
            title=title,
            body=body,
            author=item.get("by"),
            url=url,
            score=score,
            num_comments=num_comments,
            created_at=created_at,
            raw_json=raw_json,
            thread_id=final_thread_id,
            parent_id=parent_id,
            item_type=final_item_type,
        )


# -------------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------------
def _first_meaningful_line(text: str, max_len: int = 200) -> str:
    """Return the first non-empty, non-whitespace line of `text`,
    stripped of HTML tags and the most common HTML entities. Used to
    synthesize a title for comment items (which have no `title` field).
    """
    cleaned = _HTML_TAG_RE.sub("", text)
    for entity, replacement in _HTML_ENTITIES.items():
        cleaned = cleaned.replace(entity, replacement)

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) <= max_len:
            return line
        return line[: max_len - 3] + "..."

    # Fallback: whole text stripped.
    fallback = cleaned.strip()
    if not fallback:
        return ""
    return fallback[:max_len]


def _safe_int(value, *, default: int = 0) -> int:
    """Convert `value` to int, returning `default` on any conversion error."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
