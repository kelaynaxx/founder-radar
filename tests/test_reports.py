"""Tests for the markdown report."""

from __future__ import annotations

from datetime import datetime, timezone

from founder_radar.database.models import Post
from founder_radar.reports.markdown_report import MarkdownReport


def _post(**overrides) -> Post:
    defaults = dict(
        source="reddit",
        external_id="x",
        source_category="entrepreneur",
        title="Sample title",
        body="Sample body with enough characters to be kept by the cleaner.",
        author="op",
        url="https://reddit.com/r/entrepreneur/comments/x",
        score=5,
        num_comments=2,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        collected_at=datetime.now(timezone.utc).replace(tzinfo=None),

    )
    defaults.update(overrides)
    return Post(**defaults)


def test_render_handles_empty_post_list() -> None:
    out = MarkdownReport().render([])
    assert "Founder Radar" in out
    assert "Total posts: **0**" in out


def test_render_includes_post_title_and_url() -> None:
    p = _post(title="A specific title", url="https://example.com/post")
    out = MarkdownReport().render([p])
    assert "A specific title" in out
    assert "https://example.com/post" in out


def test_render_groups_by_source_and_category() -> None:
    p1 = _post(source="reddit", source_category="entrepreneur",
               external_id="r1")
    p2 = _post(source="reddit", source_category="startups",
               external_id="r2")
    p3 = _post(source="hackernews", source_category=None,
               external_id="h1")
    out = MarkdownReport().render([p1, p2, p3])
    assert "## Source: `reddit`" in out
    assert "## Source: `hackernews`" in out
    assert "### entrepreneur" in out
    assert "### startups" in out
    assert "### (uncategorized)" in out


def test_render_truncates_long_body() -> None:
    p = _post(body="a" * 1000)
    out = MarkdownReport().render([p])
    assert "..." in out


def test_render_includes_totals() -> None:
    p1 = _post(source="reddit", external_id="a", score=10, num_comments=3)
    p2 = _post(source="reddit", external_id="b", score=20, num_comments=7)
    out = MarkdownReport().render([p1, p2])
    assert "## Totals" in out
    assert "Total score across all posts: **30**" in out
    assert "Total comments across all posts: **10**" in out


def test_write_creates_file(tmp_path) -> None:
    p = _post()
    path = MarkdownReport().write([p], tmp_path / "report.md")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Sample title" in content