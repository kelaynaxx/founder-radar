"""GitHub Issues collector.

Uses the public GitHub REST API to fetch issues. No authentication is
required for public repositories, but setting ``GITHUB_TOKEN`` raises
the rate limit from 60/hour (anonymous) to 5,000/hour (authenticated).

Two collection modes:

1.  **Repo mode** — ``--repo owner/name`` (repeatable).

    Hits ``GET /repos/{owner}/{repo}/issues`` for each requested repo.
    This endpoint returns *both* issues and pull requests; pull requests
    are filtered out via the ``pull_request`` key that GitHub attaches
    to them.

2.  **Search mode** — ``--query "..."``.

    Hits ``GET /search/issues?q={query}``. GitHub's issue search supports
    the full qualifier language (``is:issue``, ``is:open``, ``label:bug``,
    ``repo:owner/name``, ...). Search already excludes PRs when you pass
    ``is:issue`` in the query; we still defensively check the
    ``pull_request`` key on every hit.

Subtype taxonomy (mapped onto ``RawPost.subtype``):

    bug             The issue is labelled "bug" or its title starts with
                    "[Bug]" / "Bug:" / "Bug Report".
    feature_request The issue is labelled "feature" / "feature request" /
                    "enhancement" or its title starts with "[Feature]" /
                    "Feature Request" / "FR:".
    enhancement     Specifically labelled "enhancement".
    question        Labelled "question" — usually a support request.
    bot_update      Authored by an automated bot account (dependabot,
                    renovate, github-actions, ...). Always downranked.
    unknown         Anything else.

Filtering defaults (overridable via settings or constructor):

    - Pull requests: skipped unconditionally.
    - Closed issues: skipped unless ``include_closed=True``.
    - Bot-typed issues: skipped unless ``include_bots=True``.
    - Template-only issues (empty body + generic title): skipped unless
      ``include_templates=True``.

Failure modes we handle explicitly:

    - 4xx on a single page: log and stop pagination of that repo.
    - 4xx on the search endpoint: log and return zero items.
    - Unexpected JSON shape: defensive skips, no exceptions bubble up.
    - Network error: caught at the per-request boundary, logged, and
      treated as "no more items" so the loop terminates cleanly.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx

from founder_radar.collectors.base import BaseCollector, RawPost

logger = logging.getLogger(__name__)


# GitHub REST endpoints. Public, no auth.
# - /repos/{owner}/{repo}/issues lists both issues and PRs.
# - /search/issues runs a structured search; supports qualifiers like
#   `is:issue`, `is:open`, `label:bug`, `repo:owner/name`, etc.
GITHUB_API_BASE_DEFAULT = "https://api.github.com"

# GitHub caps per_page at 100. Hard ceiling across pages: 1000 results
# (page=100 × per_page=100). For our calibration scans we never need
# that many; we stop early once we've collected `limit` items.
_PER_PAGE_MAX = 100
_PAGE_HARD_CAP = 100

# Known bot account login patterns. Matched as case-insensitive
# substrings against ``user.login``. Anything that contains one of these
# tokens is treated as a bot unless ``include_bots=True``.
_BOT_LOGIN_PATTERNS: tuple[str, ...] = (
    "dependabot",
    "renovate",
    "github-actions",
    "codecov",
    "snyk-bot",
    "sonarcloud",
    "deepsource-bot",
    "imgbot",
)

# Generic titles that signal a blank issue-form submission.
# Matched case-insensitively as exact (stripped) matches.
_TEMPLATE_TITLES: tuple[str, ...] = (
    "(no title)",
    "issue",
    "bug report",
    "feature request",
    "support request",
)


# -------------------------------------------------------------------------
# Public constants re-exported for tests
# -------------------------------------------------------------------------
GITHUB_SUBTYPE_BUG = "bug"
GITHUB_SUBTYPE_FEATURE_REQUEST = "feature_request"
GITHUB_SUBTYPE_ENHANCEMENT = "enhancement"
GITHUB_SUBTYPE_QUESTION = "question"
GITHUB_SUBTYPE_BOT_UPDATE = "bot_update"
GITHUB_SUBTYPE_UNKNOWN = "unknown"

GITHUB_SUBTYPES = (
    GITHUB_SUBTYPE_BUG,
    GITHUB_SUBTYPE_FEATURE_REQUEST,
    GITHUB_SUBTYPE_ENHANCEMENT,
    GITHUB_SUBTYPE_QUESTION,
    GITHUB_SUBTYPE_BOT_UPDATE,
    GITHUB_SUBTYPE_UNKNOWN,
)


# Compile title-prefix regexes once. We accept both the bracketed form
# ("[Bug] ...") and the bare colon form ("Bug: ..."). The trailing
# punctuation is optional so a bare "[Bug] something broke" still
# matches — many real issues use the bracket as a visual marker without
# adding a colon. After the prefix we require whitespace so we don't
# match a title that's just "[Bug]".
_BUG_TITLE_RE = re.compile(
    r"^\s*\[?\s*bug\s*\]?\s*[:\-]?\s+", re.IGNORECASE,
)
_FEATURE_TITLE_RE = re.compile(
    r"^\s*\[?\s*(feature\s*request|fr|feat|enhancement|rfe)\s*\]?\s*[:\-]?\s+",
    re.IGNORECASE,
)


class GitHubIssuesCollector(BaseCollector):
    """Collects public issues from GitHub via the REST API.

    Construction::

        GitHubIssuesCollector(settings)
        GitHubIssuesCollector(settings, include_closed=True)
        GitHubIssuesCollector(settings, include_bots=True,
                             include_templates=True)

    The collector holds an ``httpx.Client`` lazily so that instantiating
    it does not require network access. Tests inject a mock transport
    by patching ``_client`` (same pattern as the HN collector).
    """

    source_name: str = "github"

    def __init__(
        self,
        settings,
        *,
        include_closed: bool = False,
        include_bots: bool | None = None,
        include_templates: bool | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            settings: App settings. ``github_token``, ``github_user_agent``,
                ``github_api_base``, ``github_include_bots``, and
                ``github_include_templates`` are read from here.
            include_closed: When True, also fetch closed issues (and PRs
                remain filtered out as usual). Default False.
            include_bots: Override for ``settings.github_include_bots``.
                When None, defer to the setting.
            include_templates: Override for
                ``settings.github_include_templates``. When None, defer
                to the setting.
        """
        super().__init__(settings)
        self._include_closed = include_closed
        # `None` means "defer to settings at collect() time".
        self._include_bots_override = include_bots
        self._include_templates_override = include_templates
        self._session: httpx.Client | None = None

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------
    def _client(self) -> httpx.Client:
        """Return a configured httpx.Client, built on first use.

        Lazy so that constructing the collector does not require network
        access. Tests inject a transport by patching ``_client``.
        """
        if self._session is not None:
            return self._session

        headers = {
            "User-Agent": getattr(
                self._settings, "github_user_agent",
                "founder-radar/0.1 (GitHub Issues collector)",
            ),
            # GitHub returns v3 of the API by default, but be explicit
            # so a future default bump doesn't change our parsing.
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = getattr(self._settings, "github_token", "")
        if token:
            # Authenticated requests get a much higher rate limit. We
            # do NOT require a token — anonymous requests still work.
            headers["Authorization"] = f"Bearer {token}"

        self._session = httpx.Client(
            timeout=httpx.Timeout(30.0),
            headers=headers,
            limits=httpx.Limits(max_connections=1),
        )
        return self._session

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def collect(
        self,
        *,
        categories: list[str] | None = None,
        limit_per_category: int | None = None,
        repos: list[str] | None = None,
        query: str | None = None,
    ) -> Iterator[RawPost]:
        """Yield RawPost objects from GitHub issues.

        Exactly one of ``repos`` or ``query`` should be provided. If both
        are given, repo mode wins (we exhaust the repos first, then run
        the search). If neither is given, the collector returns zero
        items and logs a warning — there's no sensible default for a
        source this large.

        Args:
            categories: Accepted for source-uniformity with Reddit/HN.
                Unused on this collector; pass ``repos`` instead.
            limit_per_category: Max items to collect per repo or per
                search. When None, falls back to
                ``settings.scan_limit_per_subreddit``.
            repos: List of "owner/name" repository identifiers (e.g.
                ``["openai/openai-python", "langchain-ai/langchain"]``).
            query: GitHub search query string. Supports qualifiers like
                ``is:issue is:open label:bug``.
        """
        limit = (
            limit_per_category
            or getattr(self._settings, "scan_limit_per_subreddit", 50)
        )

        include_bots = self._include_bots_override
        if include_bots is None:
            include_bots = bool(
                getattr(self._settings, "github_include_bots", False)
            )
        include_templates = self._include_templates_override
        if include_templates is None:
            include_templates = bool(
                getattr(self._settings, "github_include_templates", False)
            )

        if not repos and not query:
            logger.warning(
                "GitHub collector needs either --repo or --query; "
                "nothing to collect."
            )
            return

        client = self._client()

        if repos:
            for repo in repos:
                if not repo or "/" not in repo:
                    logger.warning(
                        "Skipping invalid repo %r; expected 'owner/name'.", repo,
                    )
                    continue
                logger.info(
                    "Collecting up to %d issues from %s ...", limit, repo,
                )
                yield from self._collect_repo(
                    client=client,
                    repo=repo,
                    limit=limit,
                    include_bots=include_bots,
                    include_templates=include_templates,
                )

        if query:
            logger.info(
                "Collecting up to %d issues via search q=%r ...",
                limit, query,
            )
            yield from self._collect_search(
                client=client,
                query=query,
                limit=limit,
                include_bots=include_bots,
                include_templates=include_templates,
            )

    # ------------------------------------------------------------------
    # Repo mode
    # ------------------------------------------------------------------
    def _collect_repo(
        self,
        *,
        client: httpx.Client,
        repo: str,
        limit: int,
        include_bots: bool,
        include_templates: bool,
    ) -> Iterator[RawPost]:
        """Paginate ``GET /repos/{owner}/{repo}/issues`` up to ``limit``.

        The endpoint interleaves issues and pull requests; we drop PRs
        defensively (and silently) via the ``pull_request`` key.
        """
        state = "all" if self._include_closed else "open"
        per_page = min(limit, _PER_PAGE_MAX)
        collected = 0

        for page in range(1, _PAGE_HARD_CAP + 1):
            if collected >= limit:
                break
            params = {
                "state": state,
                "per_page": per_page,
                "page": page,
                # Newest issues first; these are the freshest pain signals.
                "direction": "desc",
                # "created" vs "updated" — created keeps issue age honest.
                "sort": "created",
            }
            url = f"{self._api_base()}/repos/{repo}/issues"
            try:
                resp = client.get(url, params=params)
            except httpx.HTTPError as exc:
                logger.error("GitHub repo fetch failed for %s: %s", repo, exc)
                return

            if resp.status_code == 404:
                logger.warning(
                    "GitHub returned 404 for %s; repo may be private or "
                    "renamed. Skipping.", repo,
                )
                return
            if resp.status_code == 403:
                # Most often rate-limit; the body has the exact reason.
                logger.warning(
                    "GitHub returned 403 for %s (rate-limited or "
                    "forbidden). Skipping. body=%s",
                    repo, resp.text[:200],
                )
                return
            if resp.status_code >= 400:
                logger.warning(
                    "GitHub returned HTTP %s for %s page %s; stopping.",
                    resp.status_code, repo, page,
                )
                return

            try:
                items = resp.json()
            except (ValueError, TypeError) as exc:
                logger.error(
                    "Bad JSON from GitHub %s page %s: %s",
                    repo, page, exc,
                )
                return

            if not isinstance(items, list):
                logger.error(
                    "Unexpected GitHub response shape for %s: %r",
                    repo, type(items).__name__,
                )
                return

            if not items:
                # Empty page: we've reached the end of the list.
                return

            for item in items:
                post = self._item_to_raw_post(
                    item,
                    source_category=repo,
                    include_bots=include_bots,
                    include_templates=include_templates,
                )
                if post is None:
                    continue
                yield post
                collected += 1
                if collected >= limit:
                    return

    # ------------------------------------------------------------------
    # Search mode
    # ------------------------------------------------------------------
    def _collect_search(
        self,
        *,
        client: httpx.Client,
        query: str,
        limit: int,
        include_bots: bool,
        include_templates: bool,
    ) -> Iterator[RawPost]:
        """Paginate ``GET /search/issues`` up to ``limit`` items.

        GitHub's search API returns at most 1000 results (10 pages ×
        per_page=100). For our calibration scans we never need that
        many; we stop early once we've collected ``limit`` items.

        The endpoint already excludes PRs when ``is:issue`` is in the
        query, but we still defensively skip any hit carrying the
        ``pull_request`` key.
        """
        per_page = min(limit, _PER_PAGE_MAX)
        collected = 0

        for page in range(1, _PAGE_HARD_CAP + 1):
            if collected >= limit:
                break
            params = {
                "q": query,
                "per_page": per_page,
                "page": page,
                # "created" is the natural sort for fresh pain signals.
                "sort": "created",
                "order": "desc",
            }
            url = f"{self._api_base()}/search/issues"
            try:
                resp = client.get(url, params=params)
            except httpx.HTTPError as exc:
                logger.error(
                    "GitHub search failed for q=%r: %s", query, exc
                )
                return

            if resp.status_code == 422:
                # Search validation error (bad query syntax).
                logger.warning(
                    "GitHub search rejected query %r (HTTP 422). "
                    "Check your qualifiers.", query,
                )
                return
            if resp.status_code == 403:
                logger.warning(
                    "GitHub search returned 403 for q=%r "
                    "(rate-limited). body=%s",
                    query, resp.text[:200],
                )
                return
            if resp.status_code >= 400:
                logger.warning(
                    "GitHub search returned HTTP %s for q=%r; stopping.",
                    resp.status_code, query,
                )
                return

            try:
                data = resp.json()
            except (ValueError, TypeError) as exc:
                logger.error(
                    "Bad JSON from GitHub search q=%r: %s", query, exc
                )
                return

            if not isinstance(data, dict):
                logger.error(
                    "Unexpected GitHub search response shape: %r",
                    type(data).__name__,
                )
                return

            items = data.get("items", [])
            if not isinstance(items, list):
                return
            if not items:
                return

            for item in items:
                post = self._item_to_raw_post(
                    item,
                    source_category=f"search:{query}",
                    include_bots=include_bots,
                    include_templates=include_templates,
                )
                if post is None:
                    continue
                yield post
                collected += 1
                if collected >= limit:
                    return

    # ------------------------------------------------------------------
    # Item -> RawPost
    # ------------------------------------------------------------------
    def _item_to_raw_post(
        self,
        item: dict[str, Any],
        *,
        source_category: str,
        include_bots: bool,
        include_templates: bool,
    ) -> RawPost | None:
        """Translate one GitHub issue dict into our RawPost shape.

        Returns None for items we should skip (PRs, bots, templates,
        malformed payloads).
        """
        if not isinstance(item, dict):
            return None

        # Pull requests come back through the /issues endpoint. GitHub
        # attaches a `pull_request` key to PR items — that's our
        # only reliable signal.
        if "pull_request" in item:
            return None

        # Drop closed issues when not explicitly requested.
        state = item.get("state")
        if state not in ("open", "closed"):
            # Missing or unexpected state -> skip defensively.
            return None
        if state == "closed" and not self._include_closed:
            return None

        # Title is mandatory.
        title = item.get("title")
        if not title or not isinstance(title, str):
            return None

        # Body. Many issues have no body (often just a title) — that's
        # fine, we'll fall back to the title for clustering context.
        body = item.get("body")
        if not isinstance(body, str):
            body = None
        if body is not None and not body.strip():
            body = None

        # Author.
        user = item.get("user") or {}
        author = user.get("login") if isinstance(user, dict) else None
        if not isinstance(author, str):
            author = None

        # Bot detection.
        is_bot = _is_bot_account(user if isinstance(user, dict) else {})
        if is_bot and not include_bots:
            return None

        # Template-only detection.
        is_template = _is_template_only(title=title, body=body)
        if is_template and not include_templates:
            return None

        # Subtype. Bots always get `bot_update` regardless of labels,
        # so downstream code can downrank them with one rule.
        labels = _extract_label_names(item.get("labels"))
        if is_bot:
            subtype = GITHUB_SUBTYPE_BOT_UPDATE
        else:
            subtype = _derive_subtype(title=title, body=body, labels=labels)

        # URL — the issue's html_url. Fall back to api_url + construct
        # a sane web URL only if html_url is missing.
        url = item.get("html_url")
        if not isinstance(url, str) or not url:
            url = None

        # Issue number + repo for stable external_id. ``number`` is the
        # human-friendly id GitHub exposes; we use ``{repo}#{number}``
        # so the same issue collected twice still dedups cleanly.
        repo_url = item.get("repository_url")
        repo_name = _repo_name_from_url(repo_url) if repo_url else None
        number = item.get("number")
        if number is None:
            return None
        external_id = (
            f"{repo_name}#{number}" if repo_name else str(number)
        )

        # Reactions -> score (best engagement proxy GitHub exposes).
        score = 0
        reactions = item.get("reactions")
        if isinstance(reactions, dict):
            try:
                score = int(reactions.get("total_count") or 0)
            except (TypeError, ValueError):
                score = 0

        # num_comments.
        try:
            num_comments = int(item.get("comments") or 0)
        except (TypeError, ValueError):
            num_comments = 0

        # created_at: ISO 8601 -> naive UTC.
        created_at = _parse_iso_datetime(item.get("created_at"))

        # raw_json: a stable JSON string for replay / debugging.
        raw_json = None
        try:
            raw_json = json.dumps(item, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            raw_json = None

        # thread_id: GitHub doesn't have first-class threading, but
        # issue + its comments share the issue's number. We tag the
        # thread root as the issue number so future thread-aware work
        # has something to attach to. (No comments are collected yet;
        # this is a placeholder consistent with HN's contract.)
        thread_id = str(number)
        parent_id = None
        item_type = "issue"

        return RawPost(
            source=self.source_name,
            external_id=external_id,
            source_category=source_category,
            title=title,
            body=body,
            author=author,
            url=url,
            score=score,
            num_comments=num_comments,
            created_at=created_at,
            raw_json=raw_json,
            thread_id=thread_id,
            parent_id=parent_id,
            item_type=item_type,
            subtype=subtype,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _api_base(self) -> str:
        """Return the configured GitHub API base URL (with no trailing slash)."""
        base = getattr(self._settings, "github_api_base", GITHUB_API_BASE_DEFAULT)
        return (base or GITHUB_API_BASE_DEFAULT).rstrip("/")


# -------------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------------
def _safe_int(value, *, default: int = 0) -> int:
    """Convert ``value`` to int, returning ``default`` on any error."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_label_names(labels: Any) -> list[str]:
    """Normalize ``item.labels`` to a flat list of lowercase names.

    GitHub's label objects look like ``{"name": "bug", "color": "..."}``,
    but for search API responses some clients see just strings. We
    accept both shapes.
    """
    if not isinstance(labels, list):
        return []
    out: list[str] = []
    for label in labels:
        if isinstance(label, str):
            out.append(label.lower().strip())
        elif isinstance(label, dict):
            name = label.get("name")
            if isinstance(name, str) and name:
                out.append(name.lower().strip())
    return [n for n in out if n]


def _is_bot_account(user: dict[str, Any]) -> bool:
    """Return True if the issue author looks like an automated bot.

    Detection layers:
      1. ``user.type == "Bot"`` — GitHub's official account kind.
      2. ``user.login`` contains a known bot token (dependabot, ...).
    """
    user_type = (user.get("type") or "").strip().lower()
    if user_type == "bot":
        return True
    login = (user.get("login") or "").lower()
    if not login:
        return False
    return any(token in login for token in _BOT_LOGIN_PATTERNS)


def _is_template_only(*, title: str, body: str | None) -> bool:
    """Return True when the issue is almost certainly a blank template.

    Heuristics:
      - Title matches a known generic title (case-insensitive).
      - Body is None AND title is short (≤40 chars) AND title does not
        look like a real sentence (no terminal punctuation, mostly
        lowercase, or all-caps).
    """
    normalized = title.strip().lower()
    if normalized in _TEMPLATE_TITLES:
        return True

    if body is None:
        if len(title) > 40:
            return False
        # Reject titles that look like real sentences.
        if title.rstrip().endswith((".", "?", "!")):
            return False
        # If the title has any uppercase letter and at least one space,
        # it's probably a real headline. Otherwise, treat as template.
        if " " in title and any(c.isupper() for c in title):
            return False
        # Single-word or all-caps short titles without body → template.
        return True

    return False


def _derive_subtype(
    *,
    title: str,
    body: str | None,
    labels: list[str],
) -> str:
    """Map an issue to one of the GitHub subtype tags.

    Order of precedence:
      1. Labels (most reliable — maintainers tag deliberately).
      2. Title prefix.
      3. Title/body heuristics (weakest).
      4. Unknown fallback.
    """
    # 1. Labels.
    label_set = set(labels)
    if "bug" in label_set:
        return GITHUB_SUBTYPE_BUG
    if "question" in label_set:
        return GITHUB_SUBTYPE_QUESTION
    if "enhancement" in label_set:
        return GITHUB_SUBTYPE_ENHANCEMENT
    if any(
        tok in label_set for tok in ("feature", "feature request", "feat", "rfe")
    ):
        return GITHUB_SUBTYPE_FEATURE_REQUEST

    # 2. Title prefix.
    if _BUG_TITLE_RE.match(title):
        return GITHUB_SUBTYPE_BUG
    if _FEATURE_TITLE_RE.match(title):
        return GITHUB_SUBTYPE_FEATURE_REQUEST

    # 3. Heuristics.
    if body and "?" in body and len(body) < 600:
        # Short body with a question mark — likely a support question.
        return GITHUB_SUBTYPE_QUESTION

    return GITHUB_SUBTYPE_UNKNOWN


def _repo_name_from_url(url: str) -> str | None:
    """Extract ``owner/name`` from a GitHub ``repository_url`` field.

    The repository_url looks like ``https://api.github.com/repos/{owner}/{name}``.
    """
    if not isinstance(url, str) or "/repos/" not in url:
        return None
    try:
        tail = url.split("/repos/", 1)[1]
    except IndexError:
        return None
    parts = [p for p in tail.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Parse GitHub's ISO 8601 timestamp into a naive UTC datetime.

    Returns None on any parse error so the field stays NULL.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        # Python's fromisoformat in 3.11+ handles "Z" suffix natively.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt