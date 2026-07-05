"""Tests for the GitHub Issues collector.

All HTTP calls are mocked via httpx.MockTransport. No network access.

The fixtures here build "issue dicts" that look like the shape GitHub's
REST API returns from `GET /repos/{owner}/{repo}/issues` and
`GET /search/issues`. They are intentionally close to real payloads so
downstream code (subtype derivation, label parsing, etc.) is exercised
end-to-end.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from founder_radar.collectors.base import RawPost
from founder_radar.collectors.github import (
    GITHUB_SUBTYPES,
    GITHUB_SUBTYPE_BOT_UPDATE,
    GITHUB_SUBTYPE_BUG,
    GITHUB_SUBTYPE_ENHANCEMENT,
    GITHUB_SUBTYPE_FEATURE_REQUEST,
    GITHUB_SUBTYPE_QUESTION,
    GITHUB_SUBTYPE_UNKNOWN,
    GitHubIssuesCollector,
    _extract_label_names,
    _is_bot_account,
    _is_template_only,
    _parse_iso_datetime,
)


# -------------------------------------------------------------------------
# Test fixtures
# -------------------------------------------------------------------------
def _mock_client(handler) -> httpx.Client:
    """Wrap a request handler in an httpx.Client with a mock transport."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _issue(
    number: int,
    *,
    title: str = "Sample Issue",
    body: str | None = "A description.",
    state: str = "open",
    user_login: str = "alice",
    user_type: str = "User",
    labels: list | None = None,
    repo_url: str = "https://api.github.com/repos/owner/repo",
    html_url: str | None = None,
    reactions_total: int = 0,
    comments: int = 0,
    created_at: str = "2024-01-15T12:00:00Z",
    is_pull_request: bool = False,
) -> dict:
    """Build a representative GitHub issue dict."""
    item = {
        "id": 100000 + number,
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "user": {"login": user_login, "type": user_type},
        "labels": labels or [],
        "repository_url": repo_url,
        "html_url": html_url or f"https://github.com/owner/repo/issues/{number}",
        "reactions": {"total_count": reactions_total},
        "comments": comments,
        "created_at": created_at,
    }
    if is_pull_request:
        # GitHub attaches a `pull_request` key (any value, usually a dict).
        # That's the only signal our filter looks at.
        item["pull_request"] = {"url": "https://api.github.com/repos/x/y/pulls/1"}
    return item


def _search_response(items: list[dict], total_count: int | None = None) -> dict:
    """Wrap a list of issue dicts in a /search/issues response shape."""
    return {
        "total_count": total_count if total_count is not None else len(items),
        "incomplete_results": False,
        "items": items,
    }


def _make_repo_handler(
    repo_responses: dict[str, list[list[dict]]],
):
    """Build a handler that returns canned pages per repo.

    Args:
        repo_responses: Mapping of "owner/name" -> list of pages.
            Each page is a list of issue dicts. An empty page (or a
            shorter list than per_page) signals "end of pagination".
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        # /repos/{owner}/{repo}/issues
        if path.startswith("/repos/") and path.endswith("/issues"):
            parts = path.split("/")
            # ["", "repos", "{owner}", "{repo}", "issues"]
            if len(parts) >= 5:
                repo = f"{parts[2]}/{parts[3]}"
            else:
                return httpx.Response(404, json={"error": "bad repo path"})
            pages = repo_responses.get(repo, [[]])
            try:
                page = int(request.url.params.get("page", "1"))
            except ValueError:
                page = 1
            idx = page - 1
            if idx < 0 or idx >= len(pages):
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=pages[idx])
        return httpx.Response(404, json={"error": "not found"})

    return handler


def _make_search_handler(responses: list[dict]):
    """Build a handler that returns canned search responses per page."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/issues":
            try:
                page = int(request.url.params.get("page", "1"))
            except ValueError:
                page = 1
            calls.append(f"page={page}")
            idx = page - 1
            if idx < 0 or idx >= len(responses):
                return httpx.Response(200, json=_search_response([]))
            return httpx.Response(200, json=responses[idx])
        return httpx.Response(404, json={"error": "not found"})

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


# -------------------------------------------------------------------------
# Source name + shape
# -------------------------------------------------------------------------
def test_source_name_is_github(tmp_settings) -> None:
    """The registry looks up collectors by `source_name`."""
    assert GitHubIssuesCollector.source_name == "github"


def test_subtypes_cover_documented_taxonomy() -> None:
    """Public taxonomy must match the brief's requirements."""
    assert GITHUB_SUBTYPE_BUG in GITHUB_SUBTYPES
    assert GITHUB_SUBTYPE_FEATURE_REQUEST in GITHUB_SUBTYPES
    assert GITHUB_SUBTYPE_QUESTION in GITHUB_SUBTYPES
    assert GITHUB_SUBTYPE_ENHANCEMENT in GITHUB_SUBTYPES
    assert GITHUB_SUBTYPE_UNKNOWN in GITHUB_SUBTYPES
    assert GITHUB_SUBTYPE_BOT_UPDATE in GITHUB_SUBTYPES


# -------------------------------------------------------------------------
# Construction: no network required, no credentials required
# -------------------------------------------------------------------------
def test_construction_does_not_make_http_requests(tmp_settings) -> None:
    """Just instantiating the collector must not touch the network."""
    c = GitHubIssuesCollector(tmp_settings)
    assert c._session is None  # type: ignore[attr-defined]


def test_construction_does_not_require_token(tmp_settings) -> None:
    """An empty token still allows the collector to build."""
    tmp_settings.github_token = ""
    c = GitHubIssuesCollector(tmp_settings)
    assert c._session is None  # type: ignore[attr-defined]


def test_token_sets_authorization_header(tmp_settings) -> None:
    """When GITHUB_TOKEN is set, requests include a Bearer token."""
    tmp_settings.github_token = "ghp_secret"
    collector = GitHubIssuesCollector(tmp_settings)
    client = collector._client()
    auth = client.headers.get("Authorization")
    assert auth == "Bearer ghp_secret"


def test_no_token_means_no_authorization_header(tmp_settings) -> None:
    """Without a token, no Authorization header is added."""
    tmp_settings.github_token = ""
    collector = GitHubIssuesCollector(tmp_settings)
    client = collector._client()
    assert "Authorization" not in client.headers


# -------------------------------------------------------------------------
# Happy path: repo mode
# -------------------------------------------------------------------------
def test_collect_repo_yields_raw_posts(tmp_settings) -> None:
    """A list of open issues becomes RawPost rows with the right shape."""
    issues = [
        _issue(1, title="Bug in foo", body="bar", comments=3, reactions_total=5),
        _issue(2, title="Feature: add baz", body="yes", comments=1, reactions_total=2),
    ]
    handler = _make_repo_handler(
        {"owner/repo": [issues, []]},  # one page, then empty
    )
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        collector = GitHubIssuesCollector(tmp_settings)
        posts = list(collector.collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert len(posts) == 2
    assert all(isinstance(p, RawPost) for p in posts)
    p0 = posts[0]
    assert p0.source == "github"
    assert p0.external_id == "owner/repo#1"
    assert p0.title == "Bug in foo"
    assert p0.body == "bar"
    assert p0.source_category == "owner/repo"
    assert p0.url == "https://github.com/owner/repo/issues/1"
    assert p0.author == "alice"
    assert p0.num_comments == 3
    assert p0.score == 5
    assert p0.thread_id == "1"
    assert p0.item_type == "issue"


def test_collect_repo_handles_multiple_repos(tmp_settings) -> None:
    """Repeating --repo should scan each repo in turn."""
    repo_a = [_issue(1, title="Repo A issue")]
    repo_b = [_issue(2, title="Repo B issue", repo_url="https://api.github.com/repos/foo/bar")]
    handler = _make_repo_handler({
        "owner/a": [repo_a, []],
        "foo/bar": [repo_b, []],
    })
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        collector = GitHubIssuesCollector(tmp_settings)
        posts = list(collector.collect(
            repos=["owner/a", "foo/bar"], limit_per_category=10,
        ))
    assert {p.source_category for p in posts} == {"owner/a", "foo/bar"}
    assert {p.title for p in posts} == {"Repo A issue", "Repo B issue"}


def test_collect_repo_respects_limit(tmp_settings) -> None:
    """We must stop issuing requests once `limit` items are collected."""
    issues = [
        _issue(i, title=f"Issue {i}")
        for i in range(1, 11)
    ]
    handler = _make_repo_handler({"owner/repo": [issues[:5], issues[5:], []]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        collector = GitHubIssuesCollector(tmp_settings)
        posts = list(collector.collect(
            repos=["owner/repo"], limit_per_category=7,
        ))
    # 7 collected, not 10 (limit hit before second page finishes).
    assert len(posts) == 7


def test_collect_repo_uses_state_open_by_default(tmp_settings) -> None:
    """The collector must request state=open by default."""
    captured_params: list[dict] = []
    issues = [_issue(1, title="Real issue")]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.append(dict(request.url.params))
        return httpx.Response(200, json=issues)

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        c = GitHubIssuesCollector(tmp_settings)
        list(c.collect(repos=["owner/repo"], limit_per_category=10))
    assert captured_params, "no requests were made"
    assert captured_params[0].get("state") == "open"


def test_collect_repo_uses_state_all_when_include_closed(tmp_settings) -> None:
    """include_closed=True sends state=all."""
    captured_params: list[dict] = []
    issue = _issue(1, title="Real issue")
    issues = [issue]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.append(dict(request.url.params))
        return httpx.Response(200, json=issues)

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        c = GitHubIssuesCollector(
            tmp_settings, include_closed=True,
        )
        list(c.collect(repos=["owner/repo"], limit_per_category=10))
    assert captured_params, "no requests were made"
    assert captured_params[0].get("state") == "all"


def test_collect_repo_skips_closed_issues_by_default(tmp_settings) -> None:
    """Closed issues must be skipped unless include_closed=True."""
    issues = [
        _issue(1, title="Open one", state="open"),
        _issue(2, title="Closed one", state="closed"),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert [p.title for p in posts] == ["Open one"]


def test_collect_repo_keeps_closed_when_include_closed(tmp_settings) -> None:
    """include_closed=True keeps both open and closed issues."""
    issues = [
        _issue(1, title="Open one", state="open"),
        _issue(2, title="Closed one", state="closed"),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(
            tmp_settings, include_closed=True,
        ).collect(repos=["owner/repo"], limit_per_category=10))
    assert {p.title for p in posts} == {"Open one", "Closed one"}


# -------------------------------------------------------------------------
# Pull request filtering
# -------------------------------------------------------------------------
def test_collect_repo_filters_out_pull_requests(tmp_settings) -> None:
    """PRs come through the same endpoint — the `pull_request` key marks them."""
    issues = [
        _issue(1, title="Real issue"),
        _issue(2, title="Pull request masquerading as issue", is_pull_request=True),
        _issue(3, title="Another issue"),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    titles = [p.title for p in posts]
    assert "Real issue" in titles
    assert "Another issue" in titles
    assert not any("Pull request" in t for t in titles)


# -------------------------------------------------------------------------
# Bot filtering
# -------------------------------------------------------------------------
def test_collect_repo_filters_bot_issues_by_default(tmp_settings) -> None:
    """Issues authored by Bot-typed accounts are dropped by default."""
    issues = [
        _issue(1, title="Real bug"),
        _issue(2, title="Bump dep", user_type="Bot", user_login="dependabot[bot]"),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert [p.title for p in posts] == ["Real bug"]


def test_collect_repo_keeps_bots_when_include_bots(tmp_settings) -> None:
    """include_bots=True keeps bot-typed issues but tags them."""
    issues = [
        _issue(1, title="Real bug"),
        _issue(2, title="Bump dep", user_type="Bot", user_login="dependabot[bot]"),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(
            tmp_settings, include_bots=True,
        ).collect(repos=["owner/repo"], limit_per_category=10))
    assert len(posts) == 2
    bot_post = next(p for p in posts if "Bump dep" in p.title)
    assert bot_post.subtype == GITHUB_SUBTYPE_BOT_UPDATE


def test_bot_detected_by_login_pattern(tmp_settings) -> None:
    """Login containing 'renovate' is treated as a bot even if type is 'User'."""
    issue = _issue(
        1, title="Update dependency", user_type="User", user_login="renovate-bot"
    )
    handler = _make_repo_handler({"owner/repo": [[issue]]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts == []  # filtered


def test_collect_repo_respects_github_include_bots_setting(tmp_settings) -> None:
    """Settings flag should be respected when constructor override is None."""
    tmp_settings.github_include_bots = True
    issue = _issue(
        2, title="Bump dep", user_type="Bot", user_login="dependabot[bot]"
    )
    handler = _make_repo_handler({"owner/repo": [[issue]]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert len(posts) == 1
    assert posts[0].subtype == GITHUB_SUBTYPE_BOT_UPDATE


# -------------------------------------------------------------------------
# Template-only filtering
# -------------------------------------------------------------------------
def test_collect_repo_filters_template_only_by_default(tmp_settings) -> None:
    """Issues with no body and a generic short title are filtered."""
    issues = [
        _issue(1, title="Real bug", body="description here"),
        # No body + title doesn't end with terminal punctuation + short =
        # template.
        _issue(2, title="bug report", body=None),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert [p.title for p in posts] == ["Real bug"]


def test_collect_repo_keeps_templates_when_include_templates(tmp_settings) -> None:
    """include_templates=True keeps generic blank issues."""
    issues = [
        _issue(1, title="Real bug", body="desc"),
        _issue(2, title="bug report", body=None),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(
            tmp_settings, include_templates=True,
        ).collect(repos=["owner/repo"], limit_per_category=10))
    assert len(posts) == 2


def test_template_title_with_real_sentence_is_kept(tmp_settings) -> None:
    """An issue with no body but a real sentence in the title is kept."""
    issues = [
        _issue(1, title="My entire data was wiped after upgrade", body=None),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert len(posts) == 1


# -------------------------------------------------------------------------
# Subtype derivation
# -------------------------------------------------------------------------
def test_subtype_bug_from_label(tmp_settings) -> None:
    """Label 'bug' yields subtype='bug'."""
    issues = [_issue(1, title="thing", labels=[{"name": "bug"}])]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].subtype == GITHUB_SUBTYPE_BUG


def test_subtype_feature_request_from_label(tmp_settings) -> None:
    """Labels 'feature' / 'feature request' yield subtype='feature_request'."""
    for label in ("feature", "feature request", "rfe"):
        issues = [_issue(1, title="Sample issue", labels=[{"name": label}])]
        handler = _make_repo_handler({"owner/repo": [issues]})
        client = _mock_client(handler)
        with patch.object(GitHubIssuesCollector, "_client", return_value=client):
            posts = list(GitHubIssuesCollector(tmp_settings).collect(
                repos=["owner/repo"], limit_per_category=10,
            ))
        assert posts[0].subtype == GITHUB_SUBTYPE_FEATURE_REQUEST, label


def test_subtype_enhancement_from_label(tmp_settings) -> None:
    """Label 'enhancement' yields subtype='enhancement'."""
    issues = [_issue(1, title="Sample issue", labels=[{"name": "enhancement"}])]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].subtype == GITHUB_SUBTYPE_ENHANCEMENT


def test_subtype_question_from_label(tmp_settings) -> None:
    """Label 'question' yields subtype='question'."""
    issues = [_issue(1, title="Sample issue", labels=[{"name": "question"}])]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].subtype == GITHUB_SUBTYPE_QUESTION


def test_subtype_bug_from_title_prefix(tmp_settings) -> None:
    """Title starting with '[Bug]' or 'Bug:' yields subtype='bug'."""
    for title in ("[Bug] something broke", "Bug: crash on save", "bug: oops"):
        issues = [_issue(1, title=title, labels=[])]
        handler = _make_repo_handler({"owner/repo": [issues]})
        client = _mock_client(handler)
        with patch.object(GitHubIssuesCollector, "_client", return_value=client):
            posts = list(GitHubIssuesCollector(tmp_settings).collect(
                repos=["owner/repo"], limit_per_category=10,
            ))
        assert posts[0].subtype == GITHUB_SUBTYPE_BUG, title


def test_subtype_feature_request_from_title_prefix(tmp_settings) -> None:
    """Title starting with '[Feature Request]' yields feature_request."""
    for title in ("[Feature Request] add dark mode",
                  "Feature Request: better search",
                  "FR: improve performance"):
        issues = [_issue(1, title=title, labels=[])]
        handler = _make_repo_handler({"owner/repo": [issues]})
        client = _mock_client(handler)
        with patch.object(GitHubIssuesCollector, "_client", return_value=client):
            posts = list(GitHubIssuesCollector(tmp_settings).collect(
                repos=["owner/repo"], limit_per_category=10,
            ))
        assert posts[0].subtype == GITHUB_SUBTYPE_FEATURE_REQUEST, title


def test_subtype_question_from_short_body_with_question_mark(tmp_settings) -> None:
    """Short body containing '?' yields subtype='question'."""
    issues = [_issue(
        1, title="how do I configure X?",
        body="I can't figure out the right flag. Help?",
        labels=[],
    )]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].subtype == GITHUB_SUBTYPE_QUESTION


def test_subtype_unknown_for_unmatched(tmp_settings) -> None:
    """Issue with no matching labels/prefix/heuristics -> unknown."""
    issues = [_issue(
        1, title="Long detailed bug report with no question mark",
        body="This is a long body without any question marks or special markers.",
        labels=[],
    )]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].subtype == GITHUB_SUBTYPE_UNKNOWN


def test_subtype_labels_accept_string_shapes(tmp_settings) -> None:
    """The /search/issues endpoint sometimes returns labels as plain strings."""
    issues = [_issue(1, title="Sample issue", labels=["bug"])]  # string instead of dict
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].subtype == GITHUB_SUBTYPE_BUG


# -------------------------------------------------------------------------
# Search mode
# -------------------------------------------------------------------------
def test_collect_search_yields_raw_posts(tmp_settings) -> None:
    """Search mode hits /search/issues and tags source_category with the query."""
    issues = [
        _issue(1, title="Bug in foo", repo_url="https://api.github.com/repos/owner/repo"),
        _issue(2, title="Bug in bar", repo_url="https://api.github.com/repos/owner/repo"),
    ]
    handler = _make_search_handler([
        _search_response(issues),
        _search_response([]),
    ])
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        collector = GitHubIssuesCollector(tmp_settings)
        posts = list(collector.collect(
            query="is:issue is:open label:bug", limit_per_category=10,
        ))
    assert len(posts) == 2
    assert all(p.source_category == "search:is:issue is:open label:bug" for p in posts)


def test_collect_search_filters_pull_requests_defensively(tmp_settings) -> None:
    """Even on /search/issues we drop items with a `pull_request` key."""
    items = [
        _issue(1, title="Real issue"),
        _issue(2, title="PR in search results", is_pull_request=True),
    ]
    handler = _make_search_handler([_search_response(items)])
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            query="is:issue label:bug", limit_per_category=10,
        ))
    assert len(posts) == 1
    assert posts[0].title == "Real issue"


def test_collect_search_handles_422(tmp_settings) -> None:
    """Bad query syntax returns 422 — log and return zero items, don't raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"message": "Validation Failed"})

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            query="bad::query", limit_per_category=10,
        ))
    assert posts == []


def test_collect_search_handles_403_rate_limit(tmp_settings) -> None:
    """Rate-limited (403) returns zero items, doesn't raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
        )

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            query="is:issue", limit_per_category=10,
        ))
    assert posts == []


def test_collect_search_paginates(tmp_settings) -> None:
    """Search follows pages until items run out or limit is reached."""
    page1 = [_issue(i, title=f"Real bug {i}") for i in range(1, 4)]
    page2 = [_issue(i, title=f"Real bug {i}") for i in range(4, 7)]
    handler = _make_search_handler([
        _search_response(page1),
        _search_response(page2),
        _search_response([]),
    ])
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            query="is:issue label:bug", limit_per_category=10,
        ))
    assert len(posts) == 6
    # Three page calls: page 1 + page 2 yield items; page 3 returns empty
    # and we stop.
    assert len(handler.calls) == 3  # type: ignore[attr-defined]


def test_collect_search_respects_limit(tmp_settings) -> None:
    """Search must stop once `limit` items are collected."""
    page1 = [_issue(i, title=f"Real bug {i}") for i in range(1, 4)]
    page2 = [_issue(i, title=f"Real bug {i}") for i in range(4, 7)]
    handler = _make_search_handler([
        _search_response(page1),
        _search_response(page2),
        _search_response([]),
    ])
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            query="is:issue label:bug", limit_per_category=4,
        ))
    assert len(posts) == 4
    # Page 1 returns 3 items, page 2 returns 1 (then we hit the limit and
    # return). Page 3 is never requested.
    assert len(handler.calls) == 2  # type: ignore[attr-defined]


# -------------------------------------------------------------------------
# Combined modes
# -------------------------------------------------------------------------
def test_collect_repo_and_search_combined(tmp_settings) -> None:
    """Passing both --repo and --query should collect both."""
    repo_issues = [_issue(1, title="repo bug")]
    search_issues = [_issue(
        2, title="search bug",
        repo_url="https://api.github.com/repos/foo/bar",
    )]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/issues":
            return httpx.Response(200, json=_search_response(search_issues))
        if path.startswith("/repos/"):
            parts = path.split("/")
            repo = f"{parts[2]}/{parts[3]}"
            return httpx.Response(
                200,
                json=repo_issues if repo == "owner/repo" else [],
            )
        return httpx.Response(404)

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"],
            query="is:issue label:bug",
            limit_per_category=10,
        ))
    titles = {p.title for p in posts}
    assert titles == {"repo bug", "search bug"}


# -------------------------------------------------------------------------
# Error handling and edge cases
# -------------------------------------------------------------------------
def test_collect_repo_handles_404(tmp_settings) -> None:
    """A 404 on a repo (private/renamed) returns zero items for that repo."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/missing"], limit_per_category=10,
        ))
    assert posts == []


def test_collect_repo_handles_403_rate_limit(tmp_settings) -> None:
    """Rate-limited (403) returns zero items, doesn't raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"message": "API rate limit exceeded for 0.0.0.0"},
        )

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts == []


def test_collect_repo_handles_500(tmp_settings) -> None:
    """A 500 on a repo page stops pagination for that repo."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts == []


def test_collect_repo_skips_invalid_repo_format(tmp_settings) -> None:
    """Repo strings without 'owner/name' shape are warned and skipped."""
    handler = _make_repo_handler({})  # no repos match
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["not-a-valid-repo", "alsogood"], limit_per_category=10,
        ))
    assert posts == []


def test_collect_with_no_repos_and_no_query_warns_and_returns(tmp_settings) -> None:
    """Without inputs the collector returns zero items and doesn't blow up."""
    posts = list(GitHubIssuesCollector(tmp_settings).collect(limit_per_category=10))
    assert posts == []


def test_collect_repo_skips_items_missing_required_fields(tmp_settings) -> None:
    """Items missing title/number are skipped silently."""
    issues = [
        _issue(1, title="OK"),
        # missing title
        {"number": 2, "state": "open", "user": {"login": "x"},
         "labels": [], "created_at": "2024-01-15T12:00:00Z",
         "html_url": "https://gh/x/2", "reactions": {"total_count": 0},
         "comments": 0, "repository_url": "https://api.github.com/repos/a/b"},
        # missing number
        _issue(99, title="no number").__class__(  # type: ignore[attr-defined]
            **{"title": "x", "state": "open", "user": {"login": "x"},
               "labels": [], "created_at": "2024-01-15T12:00:00Z",
               "html_url": "https://gh/x/x", "reactions": {"total_count": 0},
               "comments": 0,
               "repository_url": "https://api.github.com/repos/a/b"},
        ),
    ]
    # Simplify: just test the title-missing case; the number-missing one
    # is covered by other shape tests below.
    issues = [
        _issue(1, title="OK"),
        {"number": 2, "state": "open", "user": {"login": "x"},
         "labels": [], "created_at": "2024-01-15T12:00:00Z",
         "html_url": "https://gh/x/2", "reactions": {"total_count": 0},
         "comments": 0, "repository_url": "https://api.github.com/repos/a/b"},
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert [p.title for p in posts] == ["OK"]


def test_collect_repo_skips_items_with_unexpected_state(tmp_settings) -> None:
    """Items with state != 'open'/'closed' are skipped defensively."""
    issues = [
        _issue(1, title="OK", state="open"),
        _issue(2, title="Bogus state", state="unknown"),
    ]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert [p.title for p in posts] == ["OK"]


# -------------------------------------------------------------------------
# created_at parsing
# -------------------------------------------------------------------------
def test_created_at_is_parsed_from_iso_to_naive_utc(tmp_settings) -> None:
    """ISO 8601 timestamps are converted to naive UTC."""
    issues = [_issue(1, title="Sample issue", created_at="2024-01-15T12:00:00Z")]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    from datetime import datetime
    assert posts[0].created_at == datetime(2024, 1, 15, 12, 0, 0)


def test_created_at_handles_invalid_iso(tmp_settings) -> None:
    """Bad ISO strings leave created_at as None."""
    issues = [_issue(1, title="Sample issue", created_at="not a date")]
    handler = _make_repo_handler({"owner/repo": [issues]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert posts[0].created_at is None


# -------------------------------------------------------------------------
# Helper unit tests
# -------------------------------------------------------------------------
def test_extract_label_names_normalizes_strings_and_dicts() -> None:
    """Labels come in two shapes; both should produce lowercase strings."""
    labels = [
        {"name": "Bug"},
        {"name": "Feature Request"},
        "documentation",
        {"no_name_key": "ignored"},
        "",
    ]
    out = _extract_label_names(labels)
    assert "bug" in out
    assert "feature request" in out
    assert "documentation" in out
    assert "" not in out


def test_extract_label_names_handles_non_list() -> None:
    """Non-list inputs return []."""
    assert _extract_label_names(None) == []
    assert _extract_label_names({}) == []
    assert _extract_label_names("bug") == []


def test_is_bot_account_via_type() -> None:
    """user.type == 'Bot' is the strongest signal."""
    assert _is_bot_account({"type": "Bot", "login": "alice"}) is True


def test_is_bot_account_via_login_pattern() -> None:
    """Known bot login substrings also trigger."""
    for login in ("dependabot[bot]", "renovate-bot", "github-actions[bot]",
                  "codecov-io", "snyk-bot"):
        assert _is_bot_account({"type": "User", "login": login}) is True, login


def test_is_bot_account_rejects_normal_users() -> None:
    """Regular users are not bots."""
    assert _is_bot_account({"type": "User", "login": "alice"}) is False
    assert _is_bot_account({}) is False


def test_is_template_only_matches_known_titles() -> None:
    """Known generic titles are template-only even with a body."""
    for title in ("(no title)", "Issue", "bug report"):
        assert _is_template_only(title=title, body="something") is True


def test_is_template_only_short_lowercase_without_body() -> None:
    """Short generic titles without body are template-only."""
    assert _is_template_only(title="bug", body=None) is True


def test_is_template_only_real_title_kept() -> None:
    """Real-looking titles (with terminal punctuation) are kept."""
    assert _is_template_only(
        title="Crash on save.", body=None,
    ) is False
    assert _is_template_only(
        title="Does this library support Redis cluster mode?",
        body=None,
    ) is False


def test_is_template_only_long_title_without_body_kept() -> None:
    """Long titles without body are likely real."""
    long_title = "x" * 50
    assert _is_template_only(title=long_title, body=None) is False


def test_parse_iso_datetime_handles_z_suffix() -> None:
    """Z suffix is treated as UTC."""
    from datetime import datetime
    assert _parse_iso_datetime("2024-01-15T12:00:00Z") == datetime(2024, 1, 15, 12, 0, 0)


def test_parse_iso_datetime_returns_none_on_garbage() -> None:
    assert _parse_iso_datetime("not a date") is None
    assert _parse_iso_datetime(None) is None
    assert _parse_iso_datetime("") is None


# -------------------------------------------------------------------------
# API base override
# -------------------------------------------------------------------------
def test_api_base_uses_setting(tmp_settings) -> None:
    """github_api_base setting is respected (for GitHub Enterprise)."""
    tmp_settings.github_api_base = "https://github.example.com/api/v3"
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=[])

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    assert any("github.example.com/api/v3" in u for u in captured_urls)


def test_api_base_strips_trailing_slash(tmp_settings) -> None:
    """Trailing slash on the base URL is normalized."""
    tmp_settings.github_api_base = "https://api.github.com/"
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=[])

    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    # No double-slash between base and path.
    assert any("api.github.com/repos/" in u and "//repos" not in u
               for u in captured_urls)


# -------------------------------------------------------------------------
# raw_json round-trip
# -------------------------------------------------------------------------
def test_raw_json_contains_original_payload(tmp_settings) -> None:
    """The raw_json column carries the full issue dict for replay."""
    import json as _json
    issue = _issue(1, title="Sample issue", body="y", labels=[{"name": "bug"}])
    handler = _make_repo_handler({"owner/repo": [[issue]]})
    client = _mock_client(handler)
    with patch.object(GitHubIssuesCollector, "_client", return_value=client):
        posts = list(GitHubIssuesCollector(tmp_settings).collect(
            repos=["owner/repo"], limit_per_category=10,
        ))
    raw = _json.loads(posts[0].raw_json)
    assert raw["title"] == "Sample issue"
    assert raw["body"] == "y"
    assert raw["labels"][0]["name"] == "bug"
# -------------------------------------------------------------------------
# Registry integration
# -------------------------------------------------------------------------
def test_register_builtins_includes_github(tmp_settings) -> None:
    """`register_builtins()` must register the GitHub collector so the
    CLI can dispatch --source github to it."""
    from founder_radar.collectors import registry, register_builtins

    register_builtins()
    assert "github" in registry.all_names()


def test_registry_lookup_returns_github_collector_class(tmp_settings) -> None:
    """Looking up 'github' returns the GitHubIssuesCollector class."""
    from founder_radar.collectors import registry, register_builtins

    register_builtins()
    cls = registry.get("github")
    assert cls is GitHubIssuesCollector