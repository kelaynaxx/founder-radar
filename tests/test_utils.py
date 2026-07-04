"""Tests for the small text utilities."""

from __future__ import annotations

from founder_radar.utils.text import extract_first_url, slugify


def test_slugify_basic() -> None:
    assert slugify("Hello World") == "hello-world"


def test_slugify_handles_punctuation() -> None:
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_collapses_whitespace() -> None:
    assert slugify("  Multi   spaces  ") == "multi-spaces"


def test_slugify_strips_accents() -> None:
    assert slugify("naïve café") == "naive-cafe"


def test_slugify_empty_returns_untitled() -> None:
    assert slugify("") == "untitled"
    assert slugify("!!!") == "untitled"


def test_slugify_truncates_at_word_boundary() -> None:
    long = "a-very-long-title-that-exceeds-the-default-max-length-easily"
    slug = slugify(long, max_length=20)
    assert len(slug) <= 20
    assert not slug.endswith("-")


def test_extract_first_url_returns_url() -> None:
    text = "Check this out: https://example.com/foo and http://bar.com"
    assert extract_first_url(text) == "https://example.com/foo"


def test_extract_first_url_returns_none_when_absent() -> None:
    assert extract_first_url("no urls here") is None
    assert extract_first_url("") is None