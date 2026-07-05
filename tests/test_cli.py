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

def test_collect_hn_alias_routes_to_hackernews(monkeypatch, tmp_path) -> None:
    """`--source hn` should behave identically to `--source hackernews`."""
    import httpx
    from founder_radar.main import app
    from founder_radar.database.connection import get_session, init_engine

    db_path = tmp_path / "hn.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    init_engine(f"sqlite:///{db_path}")

    def handler(request):
        if "askstories" in str(request.url):
            return httpx.Response(200, json=[1])
        if request.url.path == "/v0/item/1.json":
            return httpx.Response(200, json={
                "id": 1, "type": "story", "by": "alice",
                "time": 1_700_000_000, "title": "Ask HN: what is the best way to scale Postgres?",
                "score": 50, "descendants": 3,
            })
        return httpx.Response(404, json=None)

    from founder_radar.collectors.hackernews import HackerNewsCollector
    monkeypatch.setattr(
        HackerNewsCollector, "_client",
        lambda self: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    from typer.testing import CliRunner
    result = CliRunner().invoke(app, [
        "collect", "--source", "hn", "--story-type", "askstories", "--limit", "1",
    ])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    # The hn alias must have stored a post via the canonical hackernews
    # collector, so the row's source column is "hackernews" (not "hn").
    with get_session() as session:
        from founder_radar.database.repository import PostRepository
        rows = PostRepository(session).list_all()
    assert len(rows) == 1
    assert rows[0].source == "hackernews"


def test_collect_hn_short_story_type_aliases(
    monkeypatch, tmp_path,
) -> None:
    """--story-type ask / show / top / launch should be accepted."""
    import httpx
    from founder_radar.main import app
    from founder_radar.database.connection import get_session, init_engine

    db_path = tmp_path / "hn_alias.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    from founder_radar.config.settings import get_settings
    get_settings.cache_clear()
    init_engine(f"sqlite:///{db_path}")

    # Story-type short alias -> HN feed.
    alias_to_endpoint = {
        "ask": "askstories", "show": "showstories",
        "top": "topstories", "launch": "showstories",
    }
    for short, endpoint in alias_to_endpoint.items():
        def make_handler(ep):
            def handler(request):
                if ep in str(request.url) and "/v0/" in str(request.url)                         and not "/item/" in str(request.url):
                    return httpx.Response(200, json=[42])
                if request.url.path == "/v0/item/42.json":
                    return httpx.Response(200, json={
                        "id": 42, "type": "story", "by": "u",
                        "time": 1_700_000_000, "title": f"{short} a real long-form title for testing the cleaner",
                        "score": 1, "descendants": 0,
                    })
                return httpx.Response(404, json=None)
            return handler
        from founder_radar.collectors.hackernews import HackerNewsCollector
        monkeypatch.setattr(
            HackerNewsCollector, "_client",
            lambda self, ep=endpoint: httpx.Client(
                transport=httpx.MockTransport(make_handler(ep))
            ),
        )
        result = CliRunner().invoke(app, [
            "collect", "--source", "hackernews",
            "--story-type", short, "--limit", "1",
        ])
        assert result.exit_code == 0, result.stdout + (result.stderr or "")
        with get_session() as session:
            from founder_radar.database.repository import PostRepository
            rows = PostRepository(session).list_all()
        # 1 post per iteration; the table accumulates across them.
        assert any(r.external_id == "42" for r in rows),             f"short alias {short!r} did not produce a post"


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


# -------------------------------------------------------------------------
# productizable (Phase 4+ signal calibration)
# -------------------------------------------------------------------------
def test_productizable_help_works() -> None:
    """`productizable --help` exits cleanly and documents the filters."""
    result = runner.invoke(app, ["productizable", "--help"])
    assert result.exit_code == 0
    assert "opportunity_type" in result.stdout
    assert "--type" in result.stdout
    assert "--top" in result.stdout
    assert "--min-score" in result.stdout
    assert "--recalculate" in result.stdout
    # The valid types are listed in the help text.
    assert "potential_product" in result.stdout
    assert "integration_pain" in result.stdout
    assert "developer_workflow_pain" in result.stdout


def test_productizable_on_empty_db() -> None:
    """On an empty DB, the command exits cleanly with a 'no opportunities' hint."""
    result = runner.invoke(app, ["productizable"])
    assert result.exit_code == 0
    assert "No opportunities" in result.stdout


def test_productizable_rejects_unknown_type() -> None:
    """Unknown --type values exit with code 2 (CLI usage error)."""
    result = runner.invoke(app, ["productizable", "--type", "totally-bogus"])
    assert result.exit_code == 2


def test_productizable_accepts_exclude_flag() -> None:
    """V2: --exclude TYPE (repeatable) is in --help and doesn't crash on empty DB."""
    result = runner.invoke(app, ["productizable", "--help"])
    assert result.exit_code == 0
    assert "--exclude" in result.stdout
    # On an empty DB, --exclude should be accepted and produce the
    # "no opportunities" message.
    result = runner.invoke(app, [
        "productizable", "--exclude", "upstream_library_bug",
        "--exclude", "repo_specific_bug",
    ])
    assert result.exit_code == 0
    assert "No opportunities" in result.stdout


def test_productizable_rejects_unknown_exclude_type() -> None:
    """V2: unknown --exclude values exit with code 2."""
    result = runner.invoke(
    app, ["productizable", "--exclude", "totally-bogus"],
    )
    assert result.exit_code == 2


# -------------------------------------------------------------------------
# review-opportunities (Phase 4+ LLM-assisted review)
# -------------------------------------------------------------------------
def test_review_opportunities_help_works() -> None:
    """`review-opportunities --help` exits cleanly and documents the flags."""
    result = runner.invoke(app, ["review-opportunities", "--help"])
    assert result.exit_code == 0
    assert "--verdict" in result.stdout
    assert "--exclude-rejected" in result.stdout
    assert "--rerun-all" in result.stdout
    assert "--use-heuristic" in result.stdout
    # The valid verdicts are listed in the help text.
    assert "strong_candidate" in result.stdout
    assert "maybe" in result.stdout
    assert "reject" in result.stdout


def test_review_opportunities_on_empty_db() -> None:
    """On an empty DB, the command exits cleanly with a hint."""
    result = runner.invoke(app, ["review-opportunities", "--use-heuristic"])
    assert result.exit_code == 0
    assert "No opportunities" in result.stdout


def test_review_opportunities_rejects_unknown_verdict() -> None:
    """Unknown --verdict values exit with code 2."""
    result = runner.invoke(
        app, ["review-opportunities", "--use-heuristic", "--verdict", "great_product"],
    )
    assert result.exit_code == 2


def test_review_opportunities_heuristic_rejects_non_potential_product() -> None:
    """--use-heuristic returns 'reject' for non-`potential_product` clusters
    and 'maybe' for `potential_product` clusters."""
    from founder_radar.database.connection import get_session
    from founder_radar.database.models import Base
    from founder_radar.database.repository import OpportunityRepository
    from founder_radar.database.connection import get_engine
    from founder_radar.config.settings import get_settings as _get_settings
    import json as _json

    # Initialize the schema (already done by configured_db fixture).
    get_engine()  # ensures engine

    with get_session() as session:
        repo = OpportunityRepository(session)
        # Two opportunities: one potential_product, one repo_specific_bug.
        potential = repo.add_from_dict(
            {
                "title": "Cross-tool workflow pain",
                "problem_summary": "Pain",
                "mentions": 5,
                "opportunity_type": "potential_product",
                "productizability_score": 0.75,
                "weighted_score": 0.7,
            }
        )
        bug = repo.add_from_dict(
            {
                "title": "TypeError on save",
                "problem_summary": "Bug",
                "mentions": 3,
                "opportunity_type": "repo_specific_bug",
                "productizability_score": 0.15,
                "weighted_score": 0.3,
            }
        )

    result = runner.invoke(
        app, ["review-opportunities", "--use-heuristic", "--top", "10"],
    )
    assert result.exit_code == 0
    # The `potential_product` op should land at 'maybe'.
    assert "verdict=maybe" in result.stdout
    assert "Cross-tool workflow pain" in result.stdout
    # The repo_specific_bug op should be 'reject'.
    assert "verdict=reject" in result.stdout
    assert "TypeError on save" in result.stdout


def test_review_opportunities_exclude_rejected_filter() -> None:
    """--exclude-rejected hides reject verdicts."""
    from founder_radar.database.connection import get_session
    from founder_radar.database.repository import OpportunityRepository
    from sqlalchemy import select as _sel
    from founder_radar.database.models import Opportunity as _Op

    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.add_from_dict(
            {
                "title": "Cross-tool workflow pain",
                "problem_summary": "Pain",
                "mentions": 5,
                "opportunity_type": "potential_product",
                "productizability_score": 0.75,
            }
        )
        repo.set_review(
            opp.id,
            verdict="maybe",
            reasons=["possible_micro_saas"],
            summary="manual: maybe",
            confidence=0.5,
        )

    result = runner.invoke(
        app, [
            "review-opportunities", "--use-heuristic",
            "--rerun-all",
            "--verdict", "maybe",
            "--exclude-rejected",
        ],
    )
    assert result.exit_code == 0
    # Even though no --verdict reject, --exclude-rejected filters
    # them out anyway. The maybe row should be shown.
    assert "Cross-tool workflow pain" in result.stdout
    assert "filter: --exclude-rejected" in result.stdout


def test_review_opportunities_rejects_unknown_verdict_quietly() -> None:
    """--verdict is validated against the canonical set."""
    # Without the right API key, --use-heuristic lets the CLI proceed.
    result = runner.invoke(
        app, ["review-opportunities", "--use-heuristic", "--verdict", "unknown_type"],
    )
    assert result.exit_code == 2
