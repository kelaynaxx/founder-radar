"""Tests for the configuration layer."""

from __future__ import annotations

import pytest

from founder_radar.config.settings import Settings, get_settings


def test_settings_have_safe_defaults() -> None:
    """A freshly constructed Settings object must have sensible defaults."""
    s = Settings()
    assert s.scan_limit_per_subreddit >= 1
    assert s.default_subreddits  # non-empty
    assert s.llm_model
    assert s.database_url.startswith("sqlite")


def test_subreddit_list_parses_commas() -> None:
    s = Settings(default_subreddits="a, b ,, c")
    assert s.subreddit_list == ["a", "b", "c"]


def test_subreddit_list_empty_when_blank() -> None:
    s = Settings(default_subreddits="")
    assert s.subreddit_list == []


def test_settings_rejects_invalid_scan_limit() -> None:
    """`scan_limit_per_subreddit` is bounded [1, 1000]."""
    with pytest.raises(Exception):
        Settings(scan_limit_per_subreddit=0)
    with pytest.raises(Exception):
        Settings(scan_limit_per_subreddit=10_000)


def test_get_settings_is_memoized() -> None:
    """`get_settings()` returns the same object on repeated calls."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_get_settings_respects_env(
    env_cleanup: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting env vars is reflected in the next Settings instance."""
    get_settings.cache_clear()
    monkeypatch.setenv("REDDIT_CLIENT_ID", "env_id")
    monkeypatch.setenv("SCAN_LIMIT_PER_SUBREDDIT", "123")
    s = get_settings()
    assert s.reddit_client_id == "env_id"
    assert s.scan_limit_per_subreddit == 123
    get_settings.cache_clear()