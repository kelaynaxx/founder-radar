"""Tests for the CLI.

We don't run the full pipeline end-to-end (that's an integration concern).
Instead we test that each subcommand wires the right pieces together and
exits cleanly with the expected shape.

To keep tests isolated from the real `data/founder_radar.db`, the autouse
`_isolated_cli_env` fixture overrides the env vars that `get_settings()`
reads so the CLI uses the temp DB initialized by `configured_db`.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from founder_radar.config.settings import get_settings
from founder_radar.database.connection import get_engine
from founder_radar.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cli_env(tmp_path, configured_db, monkeypatch):
    """Force every CLI invocation in this file to use the temp DB.

    Steps:
      1. `configured_db` initializes the engine on the temp DB.
      2. We read the engine's URL (the temp DB) and monkey-patch env
         vars so `get_settings()` reproduces the same URL.
      3. We clear the cached `Settings` so the next `get_settings()`
         call re-reads the env vars.
    """
    engine = get_engine()
    url = str(engine.url)
    monkeypatch.setenv("DATABASE_URL", url)
    # Redirect runtime dirs to tmp_path so `info` doesn't touch the real
    # reports/ or logs/ directories.
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_helpFlag() -> None:
    """Sanity check: `--help` works and mentions the project."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Founder Radar" in result.stdout or "founder-radar" in result.stdout


def test_info_subcommand_prints_config() -> None:
    """`info` should always succeed and include the version + DB stats."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.stdout
    assert "Founder Radar" in result.stdout
    # Phase 2 added embedding + cluster stats to info.
    assert "embedding_backend" in result.stdout
    assert "total embeddings" in result.stdout


def test_report_subcommand_handles_empty_db(tmp_path) -> None:
    """`report` against an empty DB must produce a valid Markdown file."""
    output = tmp_path / "empty.md"
    result = runner.invoke(app, ["report", "--output", str(output)])
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    assert "Total posts: **0**" in output.read_text(encoding="utf-8")


def test_collect_rejects_unknown_source() -> None:
    """Asking for a source that doesn't exist should exit with code 2."""
    result = runner.invoke(app, ["collect", "--source", "myspace"])
    assert result.exit_code == 2


def test_embed_with_no_posts_is_noop() -> None:
    """`embed` against an empty DB exits cleanly with a no-op message."""
    result = runner.invoke(app, ["embed", "--backend", "null"])
    assert result.exit_code == 0
    assert "Nothing to embed" in result.stdout


def test_cluster_with_no_embeddings_warns() -> None:
    """`cluster` against an empty DB exits with code 1 and a clear hint."""
    result = runner.invoke(app, ["cluster"])
    assert result.exit_code == 1
    assert "No posts have embeddings" in result.stdout


def test_similar_requires_query_or_post_id() -> None:
    """`similar` without either --query or --post-id exits with code 2."""
    result = runner.invoke(app, ["similar"])
    assert result.exit_code == 2

def test_trends_requires_clusters() -> None:
    result = runner.invoke(app, ["trends"])
    assert result.exit_code == 0
    assert "No clusters yet" in result.stdout


def test_validate_rejects_missing_opportunity() -> None:
    result = runner.invoke(app, ["validate", "999"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_competitors_rejects_missing_opportunity() -> None:
    result = runner.invoke(app, ["competitors", "999"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_cluster_history_rejects_missing_cluster() -> None:
    result = runner.invoke(app, ["cluster-history", "999"])
    assert result.exit_code == 0
    assert "has no posts" in result.stdout


def test_trends_filter_arg_accepted() -> None:
    """The --trend filter should be accepted without error even with no matches."""
    result = runner.invoke(app, ["trends", "--trend", "emerging"])
    assert result.exit_code == 0

def test_reality_with_no_opportunities_is_noop() -> None:
    result = runner.invoke(app, ["reality"])
    assert result.exit_code == 0
    assert "No opportunities" in result.stdout or "reality" in result.stdout.lower()


def test_reality_filter_status() -> None:
    """--status filter should not error even with no matches."""
    result = runner.invoke(app, ["reality", "--status", "underserved"])
    assert result.exit_code == 0


def test_reality_top_flag_accepted() -> None:
    """--top N should be accepted."""
    result = runner.invoke(app, ["reality", "--top", "5"])
    assert result.exit_code == 0

def test_audit_reality_rejects_missing_opportunity() -> None:
    result = runner.invoke(app, ["audit-reality", "999"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_audit_reality_runs_with_no_opportunities() -> None:
    """Empty DB: audit-reality should still respond cleanly."""
    result = runner.invoke(app, ["audit-reality", "1"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_audit_reality_help_flag() -> None:
    result = runner.invoke(app, ["audit-reality", "--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout.lower()


# -------------------------------------------------------------------------
# Hacker News: end-to-end CLI without Reddit credentials
# -------------------------------------------------------------------------

def test_collect_hackernews_runs_without_reddit_credentials(
    monkeypatch, tmp_path,
) -> None:
    """`founder-radar collect --source hackernews` works with empty Reddit creds.

    This is the central guarantee of the no-Reddit path. We point the
    test at an empty env, an empty DB, and a mock HN transport so the
    full CLI flow runs end-to-end without touching the network or
    requiring Reddit credentials.
    """
    import httpx

    # Make sure no Reddit creds leak in.
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)

    # Mock the HN network entirely.
    items = {
        1: {
            "id": 1, "type": "story", "by": "alice", "time": 1_700_000_000,
            "title": "Test HN story",
            "score": 50, "descendants": 3, "url": "https://example.com/1",
        },
    }

    def handler(request):
        if request.url.path == "/v0/askstories.json":
            return httpx.Response(200, json=[1])
        if request.url.path == "/v0/item/1.json":
            return httpx.Response(200, json=items[1])
        return httpx.Response(404, json=None)

    # Patch the collector to use a mock client. The CLI builds it via
    # the registry so we patch the class method.
    from founder_radar.collectors.hackernews import HackerNewsCollector

    def fake_client(self):
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(HackerNewsCollector, "_client", fake_client)
    # Also redirect DB writes to a temp path so we don't pollute the
    # real data dir.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/hn.db")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # Need to clear the cached settings so the env vars take effect.
    from founder_radar.config.settings import get_settings
    from founder_radar.database.connection import init_engine
    get_settings.cache_clear()

    result = runner.invoke(app, [
        "collect",
        "--source", "hackernews",
        "--story-type", "askstories",
        "--limit", "1",
    ])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Collected 1" in result.stdout or "Inserted 1" in result.stdout


def test_collect_hackernews_with_unknown_story_type_warns(
    monkeypatch, tmp_path,
) -> None:
    """An unknown --story-type should warn, not crash."""
    import httpx

    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/hn_unknown.db")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()

    def handler(request):
        return httpx.Response(200, json=[])

    from founder_radar.collectors.hackernews import HackerNewsCollector
    monkeypatch.setattr(
        HackerNewsCollector, "_client",
        lambda self: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = runner.invoke(app, [
        "collect",
        "--source", "hackernews",
        "--story-type", "fake_story_type",
        "--limit", "5",
    ])
    # Empty list of items -> 0 collected, no crash.
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "0 raw posts" in result.stdout or "Nothing to do" in result.stdout


def test_collect_hackernews_help_text_lists_source(monkeypatch) -> None:
    """`--help` should mention hackernews as a valid --source value."""
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    result = runner.invoke(app, ["collect", "--help"])
    assert result.exit_code == 0
    assert "hackernews" in result.stdout
    assert "--story-type" in result.stdout
    assert "--include-comments" in result.stdout

    result = runner.invoke(app, ["audit-reality", "--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout.lower()



def test_tune_clusters_does_not_raise_nameerror_on_empty_db() -> None:
    """Regression: tune-clusters must import GreedyCosineClusterer.

    A previous calibration pass added the symbol as a free reference
    in the function body without an import. This test fails fast if
    anyone breaks the import again — we'd get a NameError instead of
    the expected SystemExit(1) for 'no embeddings'.
    """
    result = runner.invoke(app, ["tune-clusters"])
    # `raise typer.Exit(code=1)` shows up as a SystemExit(1) in the
    # runner's exception slot — that's the *expected* normal exit, not
    # a real exception. We assert that the *type* is the expected
    # one. Any other exception type (NameError, AttributeError, ...)
    # means the import is broken.
    exc = result.exception
    assert exc is None or isinstance(exc, SystemExit), (
        f"tune-clusters raised unexpected {type(exc).__name__}: {exc}"
    )
    # Output should mention the "no embeddings" message, not a traceback.
    assert "NameError" not in result.output
    assert "Traceback" not in result.output
    assert "No posts have embeddings" in result.output
    assert result.exit_code == 1


def test_tune_clusters_symbol_is_imported_in_main() -> None:
    """Direct check: the symbol must be importable from the CLI module.

    This is a more obvious regression guard than running the full
    command — a one-line import-check is enough to catch a missing
    import that would otherwise only surface at runtime.
    """
    import founder_radar.main as cli_main
    assert hasattr(cli_main, "GreedyCosineClusterer")
    # The class should be the same one we use elsewhere.
    from founder_radar.analysis.clustering import GreedyCosineClusterer
    assert cli_main.GreedyCosineClusterer is GreedyCosineClusterer
