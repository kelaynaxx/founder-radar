"""Command-line entry point.

We expose a single `founder-radar` command with multiple subcommands:

Phase 1:
  founder-radar collect  --subreddit entrepreneur --limit 25
  founder-radar report   --output reports/latest.md
  founder-radar run      --subreddit entrepreneur
  founder-radar info

Phase 2:
  founder-radar embed
  founder-radar cluster
  founder-radar clusters  [--cluster N]  [--limit K]
  founder-radar similar  --query "..."   OR   --post-id 123

Phase 3:
  founder-radar extract
  founder-radar opportunities  [--limit K]  [--status STATUS]
  founder-radar opportunity  ID

Phase 3+ (Reality Check + Trends + Weighted Scoring):
  founder-radar trends
  founder-radar cluster-history  CLUSTER_ID
  founder-radar validate  OPPORTUNITY_ID
  founder-radar competitors  OPPORTUNITY_ID
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from founder_radar import __version__
from founder_radar.analysis import (
    ALL_REVIEW_VERDICTS,
ALL_TYPES as ALL_OPPORTUNITY_TYPES,
HeuristicExtractor,
InMemoryVectorStore,
TYPE_POTENTIAL_PRODUCT,
build_clusterer,
build_embedder,
build_extractor,
    classify_opportunity,
    review_opportunity,
    review_opportunities_batch,
run_reality_check,
run_trend_analysis,
)
from founder_radar.analysis.clustering import GreedyCosineClusterer
from founder_radar.collectors import (
    HackerNewsCollector,
    RawPost,
    RedditCollector,
    register_builtins,
    registry,
)
from founder_radar.config.logging_config import configure_logging
from founder_radar.config.settings import get_settings
from founder_radar.database.connection import get_session, init_engine
from founder_radar.database.models import Post
from founder_radar.database.repository import (
    EmbeddingRepository,
    OpportunityRepository,
    PostRepository,
    decode_vector,
)
from founder_radar.processors import Cleaner
from founder_radar.llm.base import LLMMessage
from founder_radar.llm.openai_provider import OpenAICompatibleProvider
from founder_radar.reports import MarkdownReport
# `app` is the Typer application. The pyproject.toml entry point binds it
# to the `founder-radar` console script.
app = typer.Typer(
    name="founder-radar",
    help="Discover software business opportunities from public discussions.",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared bootstrap
# =============================================================================
def _bootstrap() -> None:
    """Initialize logging, paths, and DB engine from current settings."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir)
    settings.ensure_paths()
    init_engine(settings.database_url)


def _raw_to_orm(raw: RawPost) -> Post:
    """Convert a `RawPost` (collector output) to a `Post` ORM row."""
    return Post(
        source=raw.source,
        external_id=raw.external_id,
        source_category=raw.source_category,
        title=raw.title,
        body=raw.body,
        author=raw.author,
        url=raw.url,
        score=raw.score,
        num_comments=raw.num_comments,
        created_at=raw.created_at,
        raw_json=raw.raw_json,
        # Phase 4+ thread metadata. The HN collector populates
        # these; other sources leave them NULL.
        thread_id=raw.thread_id,
        parent_id=raw.parent_id,
        item_type=raw.item_type,
    )


# =============================================================================
# collect (Phase 1)
# =============================================================================
@app.command()
def collect(
    source: str = typer.Option(
        "reddit", "--source", "-s",
        help=(
            "Which source to collect from. "
            "Registered: 'reddit', 'hackernews', 'github'. "
            "Add new sources via collectors/ and register_builtins()."
        ),
    ),
    subreddit: list[str] = typer.Option(
        None, "--subreddit", "-r",
        help="(Reddit) Subreddit(s) to scan. Repeat the flag for multiple.",
    ),
    story_type: list[str] = typer.Option(
        None, "--story-type", "-t",
        help=(
            "(Hacker News) Story type(s) to scan. Repeat the flag. "
            "Valid: topstories, newstories, askstories, showstories, "
            "beststories, jobstories. "
            "Default (when this flag is omitted): settings.default_hn_story_types."
        ),
    ),
    repo: list[str] = typer.Option(
        None, "--repo",
        help=(
            "(GitHub) One or more 'owner/name' repositories to scan for "
            "issues. Repeat the flag for multiple, e.g. "
            "`--repo openai/openai-python --repo langchain-ai/langchain`. "
            "When set, /repos/{owner}/{name}/issues is fetched."
        ),
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q",
        help=(
            "Full-text search via the source's search API. "
            "For Hacker News: the Algolia HN Search API (e.g. "
            "`--query \"pain problem frustrated\"`). "
            "For GitHub: the public /search/issues endpoint. Supports "
            "qualifiers like `is:issue is:open label:bug`, `repo:owner/name`, "
            "and free-text terms (e.g. `--query \"is:issue is:open \\\"feature request\\\" automation\"`)."
        ),
    ),
    include_comments: bool = typer.Option(
        False, "--include-comments",
        help=(
            "(Hacker News) Also fetch up to 5 first-level comments per story. "
            "Disabled by default — stories only."
        ),
    ),
    include_closed: bool = typer.Option(
        False, "--include-closed",
        help=(
            "(GitHub) Include closed issues as well as open ones. "
            "Default behavior collects open issues only."
        ),
    ),
    include_bots: bool = typer.Option(
        False, "--include-bots",
        help=(
            "(GitHub) Keep issues authored by automated bot accounts "
            "(dependabot, renovate, github-actions, ...). Default "
            "behavior filters them out at collection time."
        ),
    ),
    include_templates: bool = typer.Option(
        False, "--include-templates",
        help=(
            "(GitHub) Keep template-only issues (no body, generic title). "
            "Default behavior filters them out at collection time."
        ),
    ),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
    skip_clean: bool = typer.Option(False, "--skip-clean"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Collect posts from the configured source and store them in the DB.

    Examples:
        founder-radar collect --subreddit entrepreneur --limit 25
        founder-radar collect --source hackernews --story-type askstories --limit 100
        founder-radar collect --source hackernews --include-comments
        founder-radar collect --source github --repo openai/openai-python --limit 100
        founder-radar collect --source github --repo langchain-ai/langchain --limit 100
        founder-radar collect --source github --query "is:issue is:open label:bug" --limit 100
        founder-radar collect --source github --query 'is:issue is:open "feature request" automation' --limit 100
    """
    _bootstrap()
    settings = get_settings()

    # Discover available collectors via the registry.
    register_builtins()
    # Source aliases: "hn" is the short form for "hackernews".
    source_aliases = {"hn": "hackernews"}
    source = source_aliases.get(source, source)
    if source not in registry.all_names():
        typer.echo(
            f"Source {source!r} not available. "
            f"Registered sources: {registry.all_names()}",
            err=True,
        )
        raise typer.Exit(code=2)

    # Dispatch to the right collector. Reddit uses --subreddit; HN uses
    # --story-type; GitHub uses --repo and/or --query. Unknown collectors
    # fall through to a generic path that uses `categories` as-is.
    if source == "reddit":
        from founder_radar.collectors import RedditCollector
        categories = subreddit or None
        collector = RedditCollector(settings)
    elif source in ("hackernews", "hn"):
        # Source alias: "hn" is the short form the user uses in their
        # calibration workflow; "hackernews" is the canonical name.
        from founder_radar.collectors import HackerNewsCollector
        # Short story-type aliases the user asked for: ask, show, top,
        # new, best, job, launch. launch maps to showstories (HN has
        # no separate launch feed; launches go in showstories).
        type_aliases = {
            "ask": "askstories", "show": "showstories",
            "top": "topstories", "new": "newstories",
            "best": "beststories", "job": "jobstories",
            "launch": "showstories",
        }
        categories = (
            [type_aliases.get(s.strip().lower(), s.strip())
             for s in (story_type or [])]
            if story_type else None
        )
        if query:
            # When --query is set, pass it through to the collector.
            # The collector routes to the Algolia path; the Firebase
            # story-list endpoints are bypassed.
            collector = HackerNewsCollector(
                settings, include_comments=include_comments
            )
            raw_posts = list(collector.collect(
                categories=categories,
                limit_per_category=limit,
                query=query,
            ))
            typer.echo(
                f"Collected {len(raw_posts)} raw post(s) from hackernews "
                f"(Algolia query={query!r})."
            )
            if not raw_posts:
                typer.echo("Nothing to do.")
                return
            cleaned = raw_posts if skip_clean else Cleaner().process(raw_posts)
            typer.echo(f"After cleaning: {len(cleaned)} posts.")
            if dry_run:
                typer.echo("Dry run: not writing to database.")
                return
            with get_session() as session:
                repo = PostRepository(session)
                inserted = repo.add_many(_raw_to_orm(r) for r in cleaned)
            typer.echo(
                f"Inserted {inserted} new post(s) into the database."
            )
            return
        collector = HackerNewsCollector(
            settings, include_comments=include_comments
        )
    elif source == "github":
        from founder_radar.collectors import GitHubIssuesCollector
        if not repo and not query:
            typer.echo(
                "GitHub collector needs --repo and/or --query. "
                "Example: --repo openai/openai-python or "
                "--query \"is:issue is:open label:bug\".",
                err=True,
            )
            raise typer.Exit(code=2)
        collector = GitHubIssuesCollector(
            settings,
            include_closed=include_closed,
            include_bots=include_bots,
            include_templates=include_templates,
        )
        raw_posts = list(collector.collect(
            limit_per_category=limit,
            repos=repo or None,
            query=query,
        ))
        typer.echo(
            f"Collected {len(raw_posts)} raw post(s) from github."
        )
        if not raw_posts:
            typer.echo("Nothing to do.")
            return
        cleaned = raw_posts if skip_clean else Cleaner().process(raw_posts)
        typer.echo(f"After cleaning: {len(cleaned)} posts.")
        if dry_run:
            typer.echo("Dry run: not writing to database.")
            return
        with get_session() as session:
            db_repo = PostRepository(session)
            inserted = db_repo.add_many(_raw_to_orm(r) for r in cleaned)
        typer.echo(
            f"Inserted {inserted} new post(s) into the database."
        )
        return
    else:
        # Generic fallback for future collectors.
        collector_cls = registry.get(source)
        categories = subreddit or story_type or None
        collector = collector_cls(settings)

    raw_posts: list[RawPost] = list(
        collector.collect(
            categories=categories,
            limit_per_category=limit,
        )
    )
    typer.echo(f"Collected {len(raw_posts)} raw posts from {source}.")

    if not raw_posts:
        typer.echo("Nothing to do.")
        return

    cleaned = raw_posts if skip_clean else Cleaner().process(raw_posts)
    typer.echo(f"After cleaning: {len(cleaned)} posts.")

    if dry_run:
        typer.echo("Dry run: not writing to database.")
        return

    with get_session() as session:
        repo = PostRepository(session)
        inserted = repo.add_many(_raw_to_orm(r) for r in cleaned)
    typer.echo(f"Inserted {inserted} new post(s) into the database.")


# =============================================================================
# report (Phase 1 + Phase 3)
# =============================================================================
@app.command()
def report(
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
) -> None:
    """Render a Markdown report from the database."""
    _bootstrap()
    settings = get_settings()

    with get_session() as session:
        post_repo = PostRepository(session)
        if source:
            posts = post_repo.list_by_source(source, limit=limit)
        else:
            posts = post_repo.list_all(limit=limit)
        opp_repo = OpportunityRepository(session)
        opportunities = opp_repo.list_all(limit=20)

    typer.echo(
        f"Rendering report from {len(posts)} post(s) "
        f"and {len(opportunities)} opportunit"
        f"{'y' if len(opportunities) == 1 else 'ies'}."
    )

    if output is None:
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).strftime(
            "%Y%m%d-%H%M%S"
        )
        output = settings.reports_dir / f"report-{timestamp}.md"

    path = MarkdownReport().write(
        list(posts), output, opportunities=list(opportunities)
    )
    typer.echo(f"Wrote report: {path}")


# =============================================================================
# run (Phase 1 — collect + report convenience)
# =============================================================================
@app.command()
def run(
    source: str = typer.Option("reddit", "--source", "-s"),
    subreddit: list[str] = typer.Option(None, "--subreddit", "-r"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Collect posts and immediately render a report. Convenience command."""
    ctx = app.make_context("founder-radar", ["run"])
    try:
        collect.callback(  # type: ignore[attr-defined]
            source=source,
            subreddit=subreddit or None,
            limit=limit,
            skip_clean=False,
            dry_run=False,
        )
        report.callback(  # type: ignore[attr-defined]
            output=output,
            source=None,
            limit=None,
        )
    finally:
        ctx.close()


# =============================================================================
# embed (Phase 2)
# =============================================================================
@app.command()
def embed(
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b"),
) -> None:
    """Generate embeddings for every post that doesn't have one yet."""
    _bootstrap()
    settings = get_settings()

    embedder = build_embedder(settings)
    if backend is not None:
        new_settings = settings.model_copy(update={"embedding_backend": backend})
        embedder = build_embedder(new_settings)

    typer.echo(f"Embedder: {embedder.model_name} (dim={embedder.dim})")

    with get_session() as session:
        post_repo = PostRepository(session)
        ids_to_embed = post_repo.list_ids_without_embeddings(embedder.model_name)

    if limit is not None:
        ids_to_embed = ids_to_embed[:limit]

    if not ids_to_embed:
        typer.echo("Nothing to embed; every post already has an embedding.")
        return

    typer.echo(f"Embedding {len(ids_to_embed)} post(s)...")

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)
        posts = [post_repo.get_by_id(pid) for pid in ids_to_embed]
        posts = [p for p in posts if p is not None]
        texts = [(p.title + "\n\n" + (p.body or "")).strip() for p in posts]
        vectors = embedder.embed_texts(texts)
        new_count = emb_repo.upsert_many(
            (p.id, embedder.model_name, vectors[i])
            for i, p in enumerate(posts)
        )

    typer.echo(
        f"Embedded {len(posts)} post(s). New rows: {new_count}. "
        f"Updated rows: {len(posts) - new_count}."
    )


# =============================================================================
# cluster (Phase 2)
# =============================================================================
@app.command()
def cluster(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t"),
    reset: bool = typer.Option(False, "--reset"),
    mode: str = typer.Option(
        "embedding", "--mode", "-m",
        help=(
            "Clustering mode. 'embedding' (default) groups posts by "
            "cosine similarity over their embeddings. 'thread-aware' "
            "groups posts by their thread_id (HN story root) without "
            "using embeddings - useful for HN data where one story + "
            "many comments should be one cluster. Requires a fresh HN "
            "collect so thread_id is populated on every post."
        ),
    ),
) -> None:
    """Cluster all posts that have an embedding for the configured model.

    Modes:
      embedding    - default. Greedy cosine clustering.
      thread-aware - TODO. Will group HN comments by their story root
                    once parent_id is stored on Post. For now falls
                    back to 'embedding' with a one-line warning.
    """
    _bootstrap()
    settings = get_settings()

    # TODO(phase-4+): implement --mode thread-aware properly. The current
    # HN collector does not store parent_id on Post, so we cannot yet
    # group comments by story root. We accept the flag so user scripts
    # don't break, but fall back to embedding-based clustering with a
    # one-time warning.
    if mode not in ("embedding", "thread-aware"):
        typer.echo(
            f"Unknown --mode {mode!r}; expected 'embedding' or 'thread-aware'.",
            err=True,
        )
        raise typer.Exit(code=2)
    if mode == "thread-aware":
        # No embedding is needed for this path. The HN collector
        # already populated thread_id (story's own id) and the
        # comment's thread_id (root story id) when the post was
        # collected. We group by thread_id and assign sequential
        # cluster_ids.
        from collections import Counter
        with get_session() as session:
            post_repo = PostRepository(session)
            posts = post_repo.list_all()
            if not posts:
                typer.echo("No posts to cluster.")
                raise typer.Exit(code=1)
            # Group by thread_id; only consider posts where it's set.
            thread_groups: dict = {}
            missing_thread_id = 0
            for p in posts:
                if p.thread_id is None:
                    missing_thread_id += 1
                    continue
                thread_groups.setdefault(p.thread_id, []).append(p.id)
            if missing_thread_id > 0:
                typer.echo(
                    f"WARNING: {missing_thread_id} post(s) have no "
                    f"thread_id (likely pre-thread-aware or non-HN). "
                    f"Those will be left without a cluster_id."
                )
            if not thread_groups:
                typer.echo(
                    "No posts have thread_id set. Re-run collection "
                    "(`founder-radar collect --source hackernews "
                    "--include-comments`) so the HN collector can "
                    "populate thread_id."
                )
                raise typer.Exit(code=1)
            if reset:
                cleared = post_repo.reset_clusters()
                typer.echo(f"Cleared cluster_id on {cleared} post(s).")
            # One cluster_id per unique thread. Use the smallest
            # available post id in each thread as a stable, deterministic
            # cluster id (avoids a separate sequence).
            new_assignments: dict = {}
            for thread_id, pids in thread_groups.items():
                cid = min(pids)
                for pid in pids:
                    new_assignments[pid] = cid
            updated = post_repo.assign_clusters(new_assignments)
        typer.echo(
            f"Thread-aware clustering: assigned {updated} post(s) to "
            f"{len(thread_groups)} thread(s) "
            f"({sum(1 for v in new_assignments.values() if v == min(pids))} unique "
            f"clusters)."
        )
        return

    model_name = model or settings.embedding_model
    clusterer = build_clusterer(settings)
    if threshold is not None:
        clusterer = GreedyCosineClusterer(similarity_threshold=threshold)

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)

        posts = post_repo.list_all_with_embedding(model_name)
        if not posts:
            typer.echo(
                f"No posts have embeddings for {model_name!r}. "
                f"Run `founder-radar embed` first."
            )
            raise typer.Exit(code=1)

        embeddings = emb_repo.list_for_model(model_name)
        vec_by_post = {
            emb.post_id: decode_vector(emb.vector, expected_dim=emb.dim)
            for emb in embeddings
        }
        ids = [p.id for p in posts]
        matrix = np.stack(
            [vec_by_post[pid] for pid in ids], axis=0,
        ).astype(np.float32, copy=False)

        typer.echo(
            f"Clustering {len(posts)} post(s) with {clusterer.name!r} ..."
        )

        if reset:
            cleared = post_repo.reset_clusters()
            typer.echo(f"Cleared cluster_id on {cleared} post(s).")

        labels = clusterer.cluster(matrix)
        assignments = {pid: int(labels[i]) for i, pid in enumerate(ids)}
        updated = post_repo.assign_clusters(assignments)

    typer.echo(
        f"Assigned {updated} post(s) to {len(set(labels.tolist()))} cluster(s)."
    )


# =============================================================================
@app.command()
def clusters(
    cluster_id: Optional[int] = typer.Option(None, "--cluster", "-c"),
    sample_size: int = typer.Option(3, "--sample", "-s"),
) -> None:
    """Inspect clusters: sizes and representative posts."""
    _bootstrap()

    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()

        if not sizes:
            typer.echo("No clusters yet. Run `founder-radar cluster` first.")
            return

        if cluster_id is not None:
            if cluster_id not in sizes:
                typer.echo(f"Cluster {cluster_id} does not exist.")
                raise typer.Exit(code=1)
            posts = post_repo.list_by_cluster(cluster_id)
            typer.echo(f"Cluster {cluster_id} — {len(posts)} post(s):")
            typer.echo("")
            for p in posts:
                typer.echo(f"  - [id={p.id}] {p.title}")
                if p.url:
                    typer.echo(f"    {p.url}")
            return

        typer.echo(f"{len(sizes)} cluster(s); {sum(sizes.values())} post(s) total.")
        typer.echo("")
        for cid in sorted(sizes):
            typer.echo(f"--- Cluster {cid} (size={sizes[cid]}) ---")
            posts = post_repo.list_by_cluster(cid)
            for p in posts[:sample_size]:
                typer.echo(f"  • {p.title}")
                typer.echo(f"    id={p.id}  score={p.score}  comments={p.num_comments}")
            if len(posts) > sample_size:
                typer.echo(f"  ... and {len(posts) - sample_size} more")
            typer.echo("")


# =============================================================================
# similar (Phase 2 — inspection)
# =============================================================================
@app.command()
def similar(
    query: Optional[str] = typer.Option(None, "--query", "-q"),
    post_id: Optional[int] = typer.Option(None, "--post-id", "-p"),
    k: int = typer.Option(10, "--limit", "-k"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
) -> None:
    """Find posts semantically similar to a query string or post id."""
    if (query is None) == (post_id is None):
        typer.echo("Provide exactly one of --query / -q or --post-id / -p.")
        raise typer.Exit(code=2)

    _bootstrap()
    settings = get_settings()
    model_name = model or settings.embedding_model

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)

        embeddings = emb_repo.list_for_model(model_name)
        if not embeddings:
            typer.echo(
                f"No embeddings for model {model_name!r}. "
                "Run `founder-radar embed` first."
            )
            raise typer.Exit(code=1)

        store = InMemoryVectorStore()
        store.add(
            [emb.post_id for emb in embeddings],
            np.stack(
                [decode_vector(emb.vector, expected_dim=emb.dim) for emb in embeddings],
                axis=0,
            ).astype(np.float32, copy=False),
        )

        if query is not None:
            embedder = build_embedder(settings)
            vec = embedder.embed_texts([query])[0].astype(np.float32, copy=False)
            typer.echo(f"Query: {query!r}")
        else:
            assert post_id is not None
            emb = emb_repo.get(post_id, model_name)
            if emb is None:
                typer.echo(
                    f"Post {post_id} has no embedding for model {model_name!r}."
                )
                raise typer.Exit(code=1)
            vec = decode_vector(emb.vector, expected_dim=emb.dim).astype(
                np.float32, copy=False
            )
            source_post = post_repo.get_by_id(post_id)
            typer.echo(
                f"Query: post id={post_id} — "
                f"{source_post.title if source_post else '<unknown>'!r}"
            )

        results = store.search(vec, k=k)

    typer.echo(f"Top {len(results)} similar post(s):")
    typer.echo("")
    with get_session() as session:
        post_repo = PostRepository(session)
        for pid, sim in results:
            p = post_repo.get_by_id(pid)
            if p is None:
                continue
            typer.echo(f"  • [id={p.id}] (sim={sim:.3f}) {p.title}")
            if p.url:
                typer.echo(f"    {p.url}")


# =============================================================================
# extract (Phase 3)
# =============================================================================
@app.command()
def extract(
    cluster: Optional[int] = typer.Option(None, "--cluster", "-c"),
    min_cluster_size: int = typer.Option(
        None, "--min-cluster-size",
        help=(
            "Minimum posts per cluster required to create an opportunity. "
            "Default: settings.extract_min_cluster_size (currently "
            "${default}). Singletons (size=1) are skipped unless "
            "--include-singletons is set."
        ),
    ),
    include_singletons: bool = typer.Option(
        False, "--include-singletons",
        help=(
            "Include clusters with only 1 post. Off by default — "
            "a 1-post 'opportunity' is not a real opportunity."
        ),
    ),
    force_heuristic: bool = typer.Option(
        False, "--heuristic", hidden=True,
        help=typer.style(
            "(deprecated) use --method heuristic instead",
            fg="yellow",
        ),
    ),
    method: Optional[str] = typer.Option(
        None, "--method", "-m",
        help=(
            "Extraction method. One of: 'heuristic', 'llm', 'auto'. "
            "'auto' (default): uses LLM if LLM_API_KEY is set, else "
            "heuristic. Pass --method heuristic to FORCE the heuristic "
            "extractor even when an LLM key is present (useful when "
            "debugging or when the LLM is rate-limiting)."
        ),
    ),
) -> None:
    """Run opportunity extraction over clustered posts.

    By default, only clusters with at least N posts become
    opportunities. This prevents the "564 posts -> 551 fake
    opportunities" pathology where every singleton post is mistaken
    for a market signal.

    Use --method heuristic to skip the LLM call entirely, even when
    LLM_API_KEY is set. Use --method llm to force the LLM path even
    when --heuristic was the default (e.g. to override a missing
    LLM key configuration for a one-off run).
    """
    _bootstrap()
    settings = get_settings()
    threshold = min_cluster_size if min_cluster_size is not None else settings.extract_min_cluster_size

    # Resolve extraction method.
    # - --heuristic is the legacy flag; maps to --method heuristic.
    # - --method heuristic forces heuristic extractor.
    # - --method llm forces LLM extractor (errors if LLM not configured).
    # - --method auto (default) or None: use LLM if key present, else heuristic.
    if force_heuristic and method and method != "heuristic":
        typer.echo(
            "Conflicting flags: --heuristic and --method. Using --method.",
            err=True,
        )
    requested = "heuristic" if force_heuristic else (method or "auto")
    if requested == "auto":
        requested = "llm" if settings.llm_api_key else "heuristic"
    if requested == "heuristic":
        extractor: object = HeuristicExtractor()
    elif requested == "llm":
        if not settings.llm_api_key:
            typer.echo(
                "--method llm requires LLM_API_KEY. Either set it in .env "
                "or pass --method heuristic.",
                err=True,
            )
            raise typer.Exit(code=2)
        extractor = build_extractor(settings)
    else:
        typer.echo(
            f"Unknown --method {requested!r}. Expected: heuristic, llm, auto.",
            err=True,
        )
        raise typer.Exit(code=2)
    typer.echo(f"Extractor: {getattr(extractor, 'name', '?')} (requested: {requested})")
    typer.echo(f"Min cluster size: {threshold} (--include-singletons: {include_singletons})")

    with get_session() as session:
        post_repo = PostRepository(session)
        opp_repo = OpportunityRepository(session)
        sizes = post_repo.cluster_sizes()
        if not sizes:
            typer.echo("No clusters found. Run `founder-radar cluster` first.")
            raise typer.Exit(code=1)

        # Decide which clusters qualify.
        target_clusters = [cluster] if cluster is not None else sorted(sizes)
        qualifying = [
            cid for cid in target_clusters
            if include_singletons or sizes[cid] >= threshold
        ]
        skipped = [cid for cid in target_clusters if cid not in qualifying]

        if skipped:
            typer.echo(
                f"Skipping {len(skipped)} cluster(s) with size < {threshold}. "
                f"Pass --include-singletons to include them."
            )

        # If the user picked ONE cluster by id and it doesn't qualify,
        # exit cleanly with a clear message — don't extract noise.
        if cluster is not None and not qualifying:
            typer.echo(
                f"Cluster {cluster} has only {sizes[cluster]} post(s); "
                f"needs at least {threshold}. Pass --include-singletons to override."
            )
            return

        # If NOTHING qualifies across the whole DB, that's a real signal:
        # clustering is too fragmented. Warn loudly.
        if not qualifying:
            typer.echo(
                f"⚠ WARNING: No clusters meet min_cluster_size={threshold}.",
                err=True,
            )
            typer.echo(
                "  Clustering is too fragmented. Try one of:",
                err=True,
            )
            typer.echo(
                "    • `founder-radar tune-clusters` to find a better threshold",
                err=True,
            )
            typer.echo(
                "    • Lower --threshold when re-running `cluster` (e.g. 0.65)",
                err=True,
            )
            typer.echo(
                "    • Pass --include-singletons to extract any cluster",
                err=True,
            )
            typer.echo(
                "    • Use `--mode thread-aware` for HN (groups comments per story)",
                err=True,
            )
            return

        produced = 0
        for cid in qualifying:
            posts = list(post_repo.list_by_cluster(cid))
            opp_repo.delete_for_cluster(cid)
            data = extractor.extract(cluster_id=cid, posts=posts)
            opp = opp_repo.add_from_dict(data, post_ids=[p.id for p in posts])
            produced += 1
            typer.echo(
                f"  cluster {cid} ({sizes[cid]} posts): {opp.title[:60]!r}  "
                f"(weighted={opp.weighted_score:.2f}, "
                f"conf={opp.confidence_score:.2f})"
            )

    typer.echo(f"Produced {produced} opportunity row(s).")


# =============================================================================
# cluster-stats (calibration diagnostic)
# =============================================================================
@app.command()
def cluster_stats() -> None:
    """Diagnose cluster fragmentation.

    Shows total posts / clusters / singletons, the size distribution,
    and a warning when >70% of clusters are singletons — a strong
    signal that the similarity threshold is too tight.

    Diagnostic only. Does not write to the DB.
    """
    _bootstrap()
    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
        total_posts = post_repo.count()

    if not sizes:
        typer.echo("No clusters yet. Run `founder-radar cluster` first.")
        return

    total_clusters = len(sizes)
    singleton_count = sum(1 for s in sizes.values() if s == 1)
    singleton_pct = (singleton_count / total_clusters) * 100.0
    largest = max(sizes.values())
    avg = sum(sizes.values()) / total_clusters

    distribution = {}
    for size in sizes.values():
        bucket = size if size <= 5 else "6+"
        distribution[bucket] = distribution.get(bucket, 0) + 1

    top5 = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]

    typer.echo(f"Total posts:        {total_posts}")
    typer.echo(f"Total clusters:     {total_clusters}")
    typer.echo(f"Singleton clusters: {singleton_count}  ({singleton_pct:.1f}%)")
    typer.echo(f"Largest cluster:    {largest} post(s)")
    typer.echo(f"Average cluster:    {avg:.2f} post(s)")
    typer.echo("")
    typer.echo("Largest clusters (top 5):")
    for cid, size in top5:
        typer.echo(f"  cluster {cid}: {size} post(s)")
    typer.echo("")
    typer.echo("Size distribution:")
    for k in sorted(distribution.keys(), key=lambda x: (isinstance(x, str), x)):
        typer.echo(f"  {k} post(s): {distribution[k]} cluster(s)")

    if singleton_pct > 70:
        typer.echo("")
        typer.echo("=" * 60)
        typer.echo(f"WARNING: {singleton_pct:.1f}% of clusters are singletons.")
        typer.echo("  Clustering is too fragmented — most posts end up alone.")
        typer.echo("  Next steps:")
        typer.echo("    1. `founder-radar tune-clusters` — find a better threshold.")
        typer.echo("    2. `founder-radar cluster --threshold 0.65` — re-cluster lower.")
        typer.echo("    3. For HN: `founder-radar cluster --mode thread-aware`")
        typer.echo("       (groups all comments per story root).")
        typer.echo("=" * 60)


# =============================================================================
# tune-clusters (calibration helper)
# =============================================================================
@app.command()
def tune_clusters(
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Embedding model to use. Default: settings.embedding_model.",
    ),
    apply_threshold: Optional[float] = typer.Option(
        None, "--apply-threshold",
        help=(
            "Apply this threshold and write cluster_id to the DB. "
            "Without --apply-threshold the command is a dry run. "
            "Must be one of: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75."
        ),
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="Clear existing cluster_id values before applying.",
    ),
) -> None:
    """Try several similarity thresholds and print cluster stats.

    The default cluster similarity threshold (0.75) often produces too
    many singletons on real data. This command runs the clusterer at
    six thresholds and shows the resulting fragmentation so you can
    pick the right one for your dataset.

    EMBEDDING MODE ONLY. For HN data where story + comments should
    be one cluster, use `founder-radar cluster --mode thread-aware`
    instead - that does not depend on embedding similarity at all.

    Does NOT modify the DB unless --apply-threshold T is given.
    """
    _bootstrap()
    settings = get_settings()
    model_name = model or settings.embedding_model

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)
        posts = post_repo.list_all_with_embedding(model_name)
        if not posts:
            typer.echo(
                f"No posts have embeddings for {model_name!r}. "
                "Run `founder-radar embed` first."
            )
            raise typer.Exit(code=1)
            typer.echo(
                f"No posts have embeddings for {model_name!r}. "
                "Run `founder-radar embed` first."
            )
            return
        embeddings = emb_repo.list_for_model(model_name)
        vec_by_post = {
            e.post_id: decode_vector(e.vector, expected_dim=e.dim)
            for e in embeddings
        }
        ids = [p.id for p in posts]
        matrix = np.stack(
            [vec_by_post[i] for i in ids], axis=0
        ).astype(np.float32, copy=False)

    typer.echo(
        f"Tuning cluster thresholds across {len(ids)} embedded post(s) "
        f"(model={model_name!r}). Note: EMBEDDING MODE ONLY — for "
        f"thread-aware grouping (HN), use `cluster --mode thread-aware`."
    )
    typer.echo("")
    typer.echo(
        f"  {'Threshold':>10}  {'Clusters':>9}  {'Singletons':>11}  "
        f"{'% sing':>8}  {'Largest':>8}  {'Avg':>6}"
    )
    typer.echo("  " + "-" * 65)

    rows = []
    for t in thresholds:
        clusterer = GreedyCosineClusterer(similarity_threshold=t)
        labels = clusterer.cluster(matrix).tolist()
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        n_clusters = len(counts)
        singletons = sum(1 for c in counts.values() if c == 1)
        largest = max(counts.values()) if counts else 0
        avg = (sum(counts.values()) / n_clusters) if n_clusters else 0.0
        pct_sing = (singletons / n_clusters) * 100 if n_clusters else 0.0
        rows.append((t, labels, n_clusters, singletons, pct_sing, largest, avg))
        typer.echo(
            f"  {t:>10.2f}  {n_clusters:>9}  {singletons:>11}  "
            f"{pct_sing:>7.1f}%  {largest:>8}  {avg:>6.2f}"
        )

    if rows:
        eligible = [r for r in rows if r[4] < 50]
        if eligible:
            best = max(eligible, key=lambda r: r[6])
            typer.echo("")
            typer.echo(
                f"  Recommendation: threshold {best[0]:.2f} gives "
                f"{best[2]} clusters with avg size {best[6]:.2f} "
                f"and {best[4]:.1f}% singletons."
            )
            typer.echo(
                f"  Apply with: `founder-radar tune-clusters --apply-threshold {best[0]:.2f}`"
            )
        else:
            typer.echo("")
            typer.echo(
                "  WARNING: All tested thresholds produce >=50% singletons."
            )
            typer.echo(
                "    This dataset may need a fundamentally different approach"
            )
            typer.echo(
                "    (e.g. thread-aware grouping for HN, or a denser embedder)."
            )

    if apply_threshold is not None:
        matching = [
            r for r in rows if abs(r[0] - apply_threshold) < 1e-6
        ]
        if not matching:
            typer.echo(
                f"\n  --apply-threshold {apply_threshold} is not in the tested "
                f"set ({thresholds}). Run without --apply-threshold first to see options."
            )
            return
        t, labels, *_ = matching[0]
        typer.echo(f"\nApplying threshold {t} ...")
        with get_session() as session:
            post_repo = PostRepository(session)
            if reset:
                cleared = post_repo.reset_clusters()
                typer.echo(f"  Cleared cluster_id on {cleared} post(s).")
            assignments = {
                pid: int(labels[i]) for i, pid in enumerate(ids)
            }
            updated = post_repo.assign_clusters(assignments)
        typer.echo(
            f"  Wrote cluster_id for {updated} post(s) at threshold {t}."
        )
        typer.echo(
            "  Next: `founder-radar extract` "
            "(uses settings.extract_min_cluster_size)."
        )


# =============================================================================
# opportunities (Phase 3 — inspection, sorted by weighted_score)
# =============================================================================# =============================================================================
# opportunities (Phase 3 — inspection, sorted by weighted_score)
# =============================================================================
@app.command()
def opportunities(
    limit: int = typer.Option(20, "--limit", "-k"),
    status: Optional[str] = typer.Option(None, "--status"),
) -> None:
    """List opportunities ranked by weighted_score (pain-dominated)."""
    _bootstrap()

    with get_session() as session:
        repo = OpportunityRepository(session)
        opps = repo.list_all(status=status, limit=limit)

    if not opps:
        typer.echo("No opportunities yet. Run `founder-radar extract` first.")
        return

    typer.echo(
        f"{len(opps)} opportunit"
        f"{'y' if len(opps) == 1 else 'ies'} (sorted by weighted_score):"
    )
    typer.echo("")
    for opp in opps:
        typer.echo(
            f"[id={opp.id}] (weighted={opp.weighted_score:.2f}, "
            f"pain={opp.pain_score:.2f}, mono={opp.monetization_score:.2f}, "
            f"conf={opp.confidence_score:.2f}, mentions={opp.mentions}) "
            f"{opp.title}"
        )
        if opp.trend != "unknown":
            typer.echo(f"    trend: {opp.trend}")
        if opp.saturation_score >= 0.5:
            typer.echo(f"    saturation: {opp.saturation_score:.2f}")
        typer.echo("")


# =============================================================================
# productizable (Phase 4+ — opportunity-type calibration)
# =============================================================================
@app.command()
def productizable(
    top: int = typer.Option(30, "--top", "-k", help="Number of opportunities to show."),
    type: Optional[str] = typer.Option(  # noqa: A002 (intentional shadow)
        None, "--type", "-t",
        help=(
            "Filter by opportunity_type. One of: repo_specific_bug, "
            "upstream_library_bug, documentation_confusion, missing_feature, "
            "integration_pain, developer_workflow_pain, "
            "infra_operational_pain, security_compliance_pain, "
            "potential_product, unknown."
        ),
    ),
    exclude: list[str] = typer.Option(
        None, "--exclude", "-X",
        help=(
            "Exclude one or more opportunity_type values from the result. "
            "Repeat the flag. Useful for hiding noise like "
            "`--exclude upstream_library_bug --exclude repo_specific_bug` "
            "to focus on real product opportunities."
        ),
    ),
    min_productizability: float = typer.Option(
        0.0, "--min-score", "-s",
        help="Only show opportunities with productizability_score >= this.",
    ),
    recalculate: bool = typer.Option(
        False, "--recalculate",
        help=(
            "Re-run the classifier on every opportunity and persist the "
            "result. Without this flag, the command just reads what's "
            "already on disk."
        ),
    ),
) -> None:
    """Show opportunities by opportunity_type and productizability.

    Output columns (one row per opportunity):
      id, title, weighted_score, reality_status, opportunity_type,
      productizability_score, productizability_reason

    Filter by type:
      founder-radar productizable --type potential_product
      founder-radar productizable --type integration_pain
      founder-radar productizable --type developer_workflow_pain

    Exclude noise:
      founder-radar productizable --exclude upstream_library_bug
      founder-radar productizable --exclude upstream_library_bug --exclude repo_specific_bug
    """
    _bootstrap()

    # Validate --type and --exclude early.
    if type is not None and type not in ALL_OPPORTUNITY_TYPES:
        typer.echo(
            f"Unknown --type {type!r}. "
            f"Valid: {', '.join(ALL_OPPORTUNITY_TYPES)}",
            err=True,
        )
        raise typer.Exit(code=2)
    if exclude:
        for ex in exclude:
            if ex not in ALL_OPPORTUNITY_TYPES:
                typer.echo(
                    f"Unknown --exclude {ex!r}. "
                    f"Valid: {', '.join(ALL_OPPORTUNITY_TYPES)}",
                    err=True,
                )
                raise typer.Exit(code=2)

    with get_session() as session:
        opp_repo = OpportunityRepository(session)
        opps = list(opp_repo.list_all(limit=None))
        if not opps:
            typer.echo("No opportunities yet. Run `founder-radar extract` first.")
            raise typer.Exit(code=0)

        if recalculate:
            from sqlalchemy import select  # noqa: PLC0415
            from founder_radar.database.models import Post  # noqa: PLC0415
            post_ids_flat: set[int] = set()
            for opp in opps:
                post_ids_flat.update(opp_repo.list_post_ids(opp.id))
            # Batch-load the posts.
            posts_by_id: dict = {}
            if post_ids_flat:
                post_rows = session.execute(
                    select(Post).where(Post.id.in_(post_ids_flat))
                ).scalars().all()
                posts_by_id = {p.id: p for p in post_rows}

            updated = 0
            for opp in opps:
                ids = opp_repo.list_post_ids(opp.id)
                posts = [posts_by_id[i] for i in ids if i in posts_by_id]
                if not posts:
                    continue
                assessment = classify_opportunity(opp, posts)
                opp.opportunity_type = assessment.opportunity_type
                opp.productizability_score = assessment.productizability_score
                opp.productizability_reason = assessment.productizability_reason
                updated += 1
            typer.echo(f"Recalculated types for {updated} opportunit"
                       f"{'y' if updated == 1 else 'ies'}.")

    # Apply filters.
    exclude_set = set(exclude or [])
    filtered = []
    for opp in opps:
        if type is not None and opp.opportunity_type != type:
            continue
        if opp.opportunity_type in exclude_set:
            continue
        if (opp.productizability_score or 0.0) < min_productizability:
            continue
        filtered.append(opp)
    # Sort: by productizability_score desc, then weighted_score desc.
    filtered.sort(
        key=lambda o: (
            -(o.productizability_score or 0.0),
            -(o.weighted_score or 0.0),
        )
    )
    shown = filtered[:top]

    if not shown:
        typer.echo("No opportunities match the current filters.")
        raise typer.Exit(code=0)

    typer.echo(
        f"{len(shown)} opportunit{'y' if len(shown) == 1 else 'ies'} "
        f"(of {len(filtered)} match{'es' if len(filtered) != 1 else ''}, "
        f"{len(opps)} total)"
    )
    if type is not None:
        typer.echo(f"  filter: type={type}")
    if exclude:
        typer.echo(f"  filter: exclude={','.join(exclude)}")
    if min_productizability > 0:
        typer.echo(f"  filter: min_score>={min_productizability}")
    typer.echo("")
    for opp in shown:
        typer.echo(
            f"[id={opp.id}] (weighted={opp.weighted_score:.2f}, "
            f"reality={opp.reality_status}) "
            f"{opp.title}"
        )
        typer.echo(
            f"    type={opp.opportunity_type:<26} "
            f"productizability={opp.productizability_score:.2f}"
        )
        if opp.productizability_reason:
            typer.echo(f"    reason: {opp.productizability_reason}")
        typer.echo("")

@app.command()
def review_opportunities(
    top: int = typer.Option(50, "--top", "-k", help="Number of opportunities to review."),
    verdict: Optional[str] = typer.Option(  # noqa: A002
        None, "--verdict",
        help=(
            "Filter to opportunities with this review verdict "
            "(strong_candidate, maybe, reject). "
            "Default: show all verdicts."
        ),
    ),
    exclude_rejected: bool = typer.Option(
        False, "--exclude-rejected",
        help="Hide rows whose review_verdict is 'reject'. "
             "Useful for showing only potential survivors.",
    ),
    rerun_all: bool = typer.Option(
        False, "--rerun-all",
        help=(
            "Re-review every opportunity, even ones that already have "
            "a verdict. Without this flag, only un-reviewed opportunities "
            "are sent to the LLM."
        ),
    ),
    use_heuristic: bool = typer.Option(
        False, "--use-heuristic",
        help=(
            "Run the review layer against the heuristic classifier "
            "instead of the LLM. Useful for smoke tests without a "
            "configured LLM endpoint. Always returns 'reject' for "
            "non-`potential_product` clusters and 'maybe' for the rest."
        ),
    ),
) -> None:
    """LLM-assisted opportunity review (triage filter).

    Acts like a strict startup-analyst filter on top of the
    deterministic `opportunity_type` + `productizability_score`. The
    classifier says "this cluster LOOKS productizable"; this command
    asks an LLM (default) or a heuristic (--use-heuristic) "is this
    cluster ACTUALLY a product opportunity?"

    The LLM is configured via Settings.llm_api_key / llm_base_url.
    With no LLM_API_KEY set, the command exits with an error
    unless --use-heuristic is passed.

    Output columns (one row per opportunity):
      id, title, opportunity_type, productizability_score,
      review_verdict, review_reasons, review_summary, review_confidence

    Filters:
      --verdict strong_candidate
      --verdict maybe
      --verdict reject
      --exclude-rejected        hide all reject verdicts
    """
    _bootstrap()
    settings = get_settings()

    if verdict is not None and verdict not in ALL_REVIEW_VERDICTS:
        typer.echo(
            f"Unknown --verdict {verdict!r}. "
            f"Valid: {', '.join(ALL_REVIEW_VERDICTS)}",
            err=True,
        )
        raise typer.Exit(code=2)

    # Resolve LLM provider.
    llm = None
    if not use_heuristic:
        if not settings.llm_api_key:
            typer.echo(
                "LLM_API_KEY is not set. Either set it in .env or pass "
                "--use-heuristic to run a deterministic fallback.",
                err=True,
            )
            raise typer.Exit(code=2)
        llm = OpenAICompatibleProvider(settings)
        typer.echo(
            f"Using LLM provider: {llm.name} "
            f"(model={settings.llm_model})"
        )
    else:
        typer.echo("Using --use-heuristic fallback (no LLM).")

    with get_session() as session:
        opp_repo = OpportunityRepository(session)
        opps = list(opp_repo.list_all(limit=None))
        if not opps:
            typer.echo(
                "No opportunities yet. Run `founder-radar extract` first."
            )
            raise typer.Exit(code=0)

        # Build the post-id -> list[Post] map once.
        from sqlalchemy import select as _sa_select  # noqa: PLC0415
        from founder_radar.database.models import (  # noqa: PLC0415
            OpportunityPost, Post as _PostModel,
        )
        opp_ids = [o.id for o in opps if o.id is not None]
        posts_by_id: dict = {oid: [] for oid in opp_ids}
        if opp_ids:
            link_rows = session.execute(
                _sa_select(
                    OpportunityPost.opportunity_id,
                    OpportunityPost.post_id,
                )
            ).all()
            post_ids_flat = {row[1] for row in link_rows}
            if post_ids_flat:
                post_rows = session.execute(
                    _sa_select(_PostModel).where(
                        _PostModel.id.in_(post_ids_flat)
                    )
                ).scalars().all()
                posts_map = {p.id: p for p in post_rows}
            else:
                posts_map = {}
            for opp_id, post_id in link_rows:
                if opp_id in posts_by_id and post_id in posts_map:
                    posts_by_id[opp_id].append(posts_map[post_id])

        # Decide which opps need (re-)review.
        targets = []
        for opp in opps:
            if opp.id is None:
                continue
            if rerun_all or opp.review_verdict is None:
                targets.append(opp)
        if not targets:
            typer.echo(
                f"All {len(opps)} opportunities are already reviewed. "
                "Pass --rerun-all to re-review."
            )
        else:
            typer.echo(
                f"Reviewing {len(targets)} opportunit"
                f"{'y' if len(targets) == 1 else 'ies'} "
                f"(of {len(opps)} total) ..."
            )
            if llm is not None:
                results = review_opportunities_batch(
                    targets, posts_by_id, llm, progress=False,
                )
            else:
                results = _heuristic_review_batch(targets, posts_by_id)
            for opp_id, verdict_obj in results.items():
                opp_repo.set_review(
                    opp_id,
                    verdict=verdict_obj.verdict,
                    reasons=verdict_obj.reasons,
                    summary=verdict_obj.summary,
                    confidence=verdict_obj.confidence,
                )

    # Display: re-fetch from DB so we see persisted state.
    with get_session() as session:
        opp_repo = OpportunityRepository(session)
        all_opps = list(opp_repo.list_all(limit=None))

        def _verdict_of(opp) -> str:
            return opp.review_verdict or "reject"

        if verdict is not None:
            filtered = [o for o in all_opps if _verdict_of(o) == verdict]
        else:
            filtered = list(all_opps)
        if exclude_rejected:
            filtered = [o for o in filtered if _verdict_of(o) != "reject"]
        verdict_order = {
            "strong_candidate": 0, "maybe": 1, "reject": 2,
        }
        filtered.sort(
            key=lambda o: (
                verdict_order.get(_verdict_of(o), 3),
                -(o.productizability_score or 0.0),
            )
        )
        shown = filtered[:top]

    if not shown:
        typer.echo("No opportunities match the current filters.")
        raise typer.Exit(code=0)

    n_rejected = sum(1 for o in shown if _verdict_of(o) == "reject")
    n_maybe = sum(1 for o in shown if _verdict_of(o) == "maybe")
    n_strong = sum(1 for o in shown if _verdict_of(o) == "strong_candidate")
    typer.echo(
        f"{len(shown)} opportunit{'y' if len(shown) == 1 else 'ies'} "
        f"(strong={n_strong}, maybe={n_maybe}, reject={n_rejected}, "
        f"of {len(filtered)} match, {len(all_opps)} total)"
    )
    if verdict is not None:
        typer.echo(f"  filter: verdict={verdict}")
    if exclude_rejected:
        typer.echo("  filter: --exclude-rejected")
    typer.echo("")

    for opp in shown:
        v = _verdict_of(opp)
        typer.echo(
            f"[id={opp.id}] (type={opp.opportunity_type}, "
            f"productizability={opp.productizability_score:.2f}, "
            f"verdict={v}, conf={opp.review_confidence:.2f}) "
            f"{opp.title}"
        )
        reasons = opp_repo.review_reasons(opp)
        if reasons:
            typer.echo(f"    reasons: {','.join(reasons)}")
        if opp.review_summary:
            typer.echo(f"    summary: {opp.review_summary[:300]}")
        typer.echo("")


def _heuristic_review_batch(opps, posts_by_id):
    """Deterministic fallback when --use-heuristic is passed.

    Returns {opp_id: ReviewVerdict}. Always says 'reject' for
    non-`potential_product` clusters and 'maybe' for the rest.
    Used for smoke tests and offline operation; never returns
    'strong_candidate' (that's reserved for the LLM).
    """
    from founder_radar.analysis.opportunity_review import (
        ReviewVerdict,
        REVIEW_VERDICT_MAYBE,
        REVIEW_VERDICT_REJECT,
    )
    out: dict = {}
    for opp in opps:
        if opp.opportunity_type == "potential_product":
            out[opp.id] = ReviewVerdict(
                verdict=REVIEW_VERDICT_MAYBE,
                reasons=["possible_micro_saas"],
                summary="heuristic: potential_product cluster (no LLM verdict)",
                confidence=0.0,
            )
        else:
            out[opp.id] = ReviewVerdict(
                verdict=REVIEW_VERDICT_REJECT,
                reasons=["too_repo_specific"],
                summary="heuristic: not a potential_product cluster",
                confidence=0.0,
            )
    return out


@app.command()
def opportunity(
    opportunity_id: int = typer.Argument(..., help="Opportunity id to show."),
) -> None:
    """Show one opportunity in full."""
    _bootstrap()

    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.get_by_id(opportunity_id)

    if opp is None:
        typer.echo(f"Opportunity {opportunity_id} does not exist.")
        raise typer.Exit(code=1)

    typer.echo(f"# {opp.title}")
    typer.echo("")
    typer.echo(f"**Cluster:** {opp.cluster_id}")
    typer.echo(f"**Status:** {opp.status}")
    typer.echo(f"**Method:** {opp.extraction_method} ({opp.llm_model or 'n/a'})")
    typer.echo(f"**Mentions:** {opp.mentions}")
    typer.echo("")
    typer.echo("## Scores (each 0..1)")
    typer.echo(f"- frequency:           {opp.frequency_score:.2f}")
    typer.echo(f"- emotional_intensity: {opp.emotional_intensity_score:.2f}")
    typer.echo(f"- dissatisfaction:     {opp.dissatisfaction_score:.2f}")
    typer.echo(f"- market_size:         {opp.market_size_score:.2f}")
    typer.echo(f"- ease_of_implementation: {opp.ease_of_implementation_score:.2f}")
    typer.echo(f"- recurring_revenue:   {opp.recurring_revenue_score:.2f}")
    typer.echo(f"- technical_feasibility: {opp.technical_feasibility_score:.2f}")
    typer.echo(f"- novelty:             {opp.novelty_score:.2f}")
    typer.echo("")
    typer.echo("## Phase 3+ weighted")
    typer.echo(f"- pain_score:        {opp.pain_score:.2f}")
    typer.echo(f"- monetization_score: {opp.monetization_score:.2f}")
    typer.echo(f"- weighted_score:     {opp.weighted_score:.2f}  <- rank key")
    typer.echo(f"- total_score:        {opp.total_score:.2f}")
    typer.echo(f"- confidence_score:   {opp.confidence_score:.2f}")
    typer.echo("")
    typer.echo("## Reality Check")
    typer.echo(f"- saturation_score:          {opp.saturation_score:.2f}")
    typer.echo(f"- distinct_competitors:      {opp.distinct_competitor_count}")
    typer.echo(f"- competitor_mentions:       {opp.competitor_mention_count}")
    typer.echo(f"- trend:                     {opp.trend}")

    # Phase 3.5 Reality Validation: one line per opportunity.
    # Use `founder-radar audit-reality N` for the full reason + raw signals.
    typer.echo(
        f"- reality_status:           {opp.reality_status} "
        f"(confidence={opp.reality_confidence:.2f}, "
        f"competitor_strength={opp.competitor_strength_estimate:.2f})"
    )
    typer.echo(f"- saturation_score:          {opp.saturation_score:.2f}")
    typer.echo(f"- distinct_competitors:      {opp.distinct_competitor_count}")
    typer.echo(f"- competitor_mentions:       {opp.competitor_mention_count}")
    typer.echo(f"- trend:                     {opp.trend}")
    typer.echo("")
    typer.echo("## Problem")
    typer.echo(opp.problem_summary)
    if opp.target_audience:
        typer.echo("")
        typer.echo(f"**Audience:** {opp.target_audience}")

    with get_session() as session:
        repo = OpportunityRepository(session)
        ideas = repo.saas_ideas(opp)
        comps = repo.competitors(opp)
        links = repo.source_links(opp)
        post_ids = repo.list_post_ids(opportunity_id)

    if ideas:
        typer.echo("")
        typer.echo("## SaaS ideas")
        for idea in ideas:
            typer.echo(f"- {idea}")

    if comps:
        typer.echo("")
        typer.echo("## Competitors")
        for c in comps:
            typer.echo(f"- {c}")

    if links:
        typer.echo("")
        typer.echo(f"## Source posts ({len(post_ids)} linked)")
        for url in links[:10]:
            typer.echo(f"- {url}")
        if len(links) > 10:
            typer.echo(f"- ... and {len(links) - 10} more")


# =============================================================================
# trends (Phase 3+ — inspection)
# =============================================================================
@app.command()
def trends(
    trend: Optional[str] = typer.Option(
        None, "--trend", "-t",
        help="Filter by trend label: 'emerging', 'stable', 'declining', 'recurring'.",
    ),
    sort_by: str = typer.Option(
        "recency", "--sort", "-s",
        help="Sort by 'recency' or 'size'.",
    ),
) -> None:
    """Show cluster trend classifications."""
    _bootstrap()

    with get_session() as session:
        post_repo = PostRepository(session)
        sizes = post_repo.cluster_sizes()
        if not sizes:
            typer.echo("No clusters yet. Run `founder-radar cluster` first.")
            return

        rows = []
        for cid in sizes:
            posts = list(post_repo.list_by_cluster(cid))
            if not posts:
                continue
            tr = run_trend_analysis(posts)
            rows.append((cid, len(posts), tr))

        if trend is not None:
            rows = [r for r in rows if r[2].trend == trend]

        if sort_by == "size":
            rows.sort(key=lambda r: -r[1])
        else:
            rows.sort(key=lambda r: -r[2].posts_last_7d)

    typer.echo(f"{len(rows)} cluster(s):")
    typer.echo("")
    for cid, size, tr in rows:
        typer.echo(
            f"Cluster {cid:>3}  size={size:>3}  trend={tr.trend:<10}  {tr.label}"
        )
    typer.echo("")


# =============================================================================
# cluster-history (Phase 3+ — inspection)
# =============================================================================
@app.command()
def cluster_history(
    cluster_id: int = typer.Argument(..., help="Cluster id to inspect."),
) -> None:
    """Show a cluster's posts ordered by time + trend + reality check."""
    _bootstrap()

    with get_session() as session:
        post_repo = PostRepository(session)
        posts = list(post_repo.list_by_cluster(cluster_id))

    if not posts:
        typer.echo(f"Cluster {cluster_id} has no posts.")
        return

    ordered = sorted(posts, key=lambda p: p.created_at or datetime.min)
    tr = run_trend_analysis(posts)
    rc = run_reality_check(posts)

    typer.echo(f"Cluster {cluster_id}: {len(posts)} post(s)")
    typer.echo(f"Trend: {tr.label}")
    typer.echo(
        f"  last 7d: {tr.posts_last_7d} · prior 30d: {tr.posts_prior_30d} · "
        f"growth: {tr.growth_rate:.2f}x"
    )
    typer.echo(
        f"Reality check: {rc.distinct_competitor_count} competitor(s), "
        f"saturation={rc.saturation_score:.2f}"
    )
    typer.echo("")
    typer.echo("Timeline (oldest -> newest):")
    for p in ordered:
        ts = p.created_at.strftime("%Y-%m-%d") if p.created_at else "unknown"
        typer.echo(f"  {ts}  [id={p.id}] score={p.score:>3}  {p.title}")


# =============================================================================
# validate (Phase 3+ — inspection)
# =============================================================================
@app.command()
def validate(
    opportunity_id: int = typer.Argument(..., help="Opportunity id to validate."),
) -> None:
    """Reality-check one opportunity: competitors, saturation, trend, scores."""
    _bootstrap()

    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.get_by_id(opportunity_id)

    if opp is None:
        typer.echo(f"Opportunity {opportunity_id} does not exist.")
        raise typer.Exit(code=1)

    typer.echo(f"# {opp.title}")
    typer.echo("")
    typer.echo("## Reality Check")
    saturated = " ⚠ SATURATED" if opp.saturation_score >= 0.7 else ""
    typer.echo(f"- saturation_score: {opp.saturation_score:.2f}{saturated}")
    typer.echo(f"- distinct_competitor_count: {opp.distinct_competitor_count}")
    typer.echo(f"- competitor_mention_count: {opp.competitor_mention_count}")
    with get_session() as session:
        repo = OpportunityRepository(session)
        comps = repo.competitors(opp)
    if comps:
        typer.echo("- competitors found:")
        for c in comps[:10]:
            typer.echo(f"    - {c}")
        if len(comps) > 10:
            typer.echo(f"    - ... and {len(comps) - 10} more")
    else:
        typer.echo("- competitors found: (none)")
    typer.echo("")
    typer.echo("## Trend")
    typer.echo(f"- trend: {opp.trend}")
    typer.echo(f"- mentions: {opp.mentions}")
    typer.echo("")
    typer.echo("## Score Breakdown (Phase 3+ weighted)")
    typer.echo(f"- pain_score:        {opp.pain_score:.2f}  (50% weight)")
    typer.echo(f"- monetization:      {opp.monetization_score:.2f}  (40% weight)")
    typer.echo(f"- novelty:           {opp.novelty_score:.2f}  (10% weight)")
    typer.echo(f"- weighted_score:    {opp.weighted_score:.2f}  <- rank key")
    typer.echo(f"- total_score:       {opp.total_score:.2f}  (legacy unweighted)")
    typer.echo(f"- confidence_score:  {opp.confidence_score:.2f}")


# =============================================================================
# competitors (Phase 3+ — inspection)
# =============================================================================
@app.command()
def competitors(
    opportunity_id: int = typer.Argument(..., help="Opportunity id to inspect."),
) -> None:
    """List competitors extracted for one opportunity."""
    _bootstrap()

    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.get_by_id(opportunity_id)

    if opp is None:
        typer.echo(f"Opportunity {opportunity_id} does not exist.")
        raise typer.Exit(code=1)

    with get_session() as session:
        repo = OpportunityRepository(session)
        comps = repo.competitors(opp)

    if not comps:
        typer.echo(f"Opportunity {opportunity_id}: no competitors detected.")
        return
    typer.echo(
        f"Opportunity {opportunity_id}: {len(comps)} competitor(s) detected "
        f"({opp.competitor_mention_count} total mention(s))."
    )
    typer.echo("")
    for c in comps:
        typer.echo(f"  - {c}")


# =============================================================================
# audit-reality (Phase 3.5 — calibration inspection)
# =============================================================================
# Deep inspection of a single opportunity's RealityAssessment. Surfaces
# every raw signal the validator saw (pain_density, competitor_strength,
# dissatisfaction_hits, distinct_competitor_count) and the reason string
# that explains the classification decision. Use this to verify a
# borderline case hit the right branch.
@app.command()
def audit_reality(
    opportunity_id: int = typer.Argument(..., help="Opportunity id to audit."),
) -> None:
    """Calibration audit of one opportunity's Reality classification.

    Shows every raw signal and the exact reason string used to pick
    the status. Use this to verify a borderline case hit the right
    branch — for example, why a particular cluster was classified
    'underserved' instead of 'unknown'.
    """
    _bootstrap()

    with get_session() as session:
        repo = OpportunityRepository(session)
        opp = repo.get_by_id(opportunity_id)

    if opp is None:
        typer.echo(f"Opportunity {opportunity_id} does not exist.")
        raise typer.Exit(code=1)

    # Re-derive the full assessment so the audit reflects the *current*
    # state of the cluster's posts. The DB stores summary fields
    # (status, confidence) but not the raw signals or reason — those
    # are recomputed here for inspection.
    assessment = None
    if opp.cluster_id is not None:
        try:
            from founder_radar.analysis.reality_validator import assess_reality
            from founder_radar.database.connection import get_session as _gs
            from founder_radar.database.repository import PostRepository as _PR

            with _gs() as s:
                posts = list(_PR(s).list_by_cluster(opp.cluster_id))
            if posts:
                assessment = assess_reality(
                    posts,
                    competitors=None,
                    distinct_competitor_count=opp.distinct_competitor_count,
                    competitor_mention_count=opp.competitor_mention_count,
                )
        except Exception as exc:
            logger.debug("Could not recompute assessment for opp %s: %s", opp.id, exc)

    # Stored values (what the DB has).
    typer.echo(f"# Audit: opportunity {opp.id} — {opp.title!r}")
    typer.echo("")
    typer.echo("## Stored assessment (from DB)")
    typer.echo(f"  reality_status:           {opp.reality_status}")
    typer.echo(f"  reality_confidence:       {opp.reality_confidence:.2f}")
    typer.echo(f"  weighted_score:           {opp.weighted_score:.2f}")
    typer.echo(f"  total_score:              {opp.total_score:.2f}")
    typer.echo("")

    if assessment is None:
        typer.echo("(Could not recompute full assessment — cluster not found.)")
        return

    # Recomputed raw signals (what the classifier actually saw).
    typer.echo("## Raw signals (recomputed)")
    typer.echo(f"  pain_density:             {assessment.pain_density:.3f}")
    typer.echo(f"  dissatisfaction_hits:     {assessment.dissatisfaction_hits}")
    typer.echo(f"  distinct_competitor_count:{assessment.distinct_competitor_count}")
    typer.echo(f"  competitor_strength:     {assessment.competitor_strength_estimate:.3f}")
    typer.echo(f"  saturation_confidence:    {assessment.saturation_confidence:.3f}")
    typer.echo("")

    # Status (recomputed).
    typer.echo("## Status (recomputed)")
    typer.echo(f"  status:    {assessment.status}")
    typer.echo(f"  is_viable:  {assessment.is_viable}")
    typer.echo("")

    # The reason string.
    typer.echo("## Reason")
    typer.echo(f"  {assessment.reason}")
    typer.echo("")

    # Evidence bullets.
    if assessment.evidence:
        typer.echo("## Evidence")
        for line in assessment.evidence:
            typer.echo(f"  - {line}")
        typer.echo("")


# =============================================================================
# reality (Phase 3.5 — inspection)
# =============================================================================


# =============================================================================
# reality (Phase 3.5 — inspection)
# =============================================================================
@app.command()
def reality(
    top: int = typer.Option(
        20, "--top", "-n",
        help="How many top opportunities (by weighted_score) to show.",
    ),
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by reality status: 'saturated', 'competitive', 'underserved', 'unknown'.",
    ),
) -> None:
    """Show the Reality View: which opportunities are actually viable.

    The ranking view (weighted_score) tells you what *looks* promising.
    The reality view tells you what *is* viable given competitor
    presence and user dissatisfaction signals. Use both — they answer
    different questions.
    """
    _bootstrap()

    with get_session() as session:
        repo = OpportunityRepository(session)
        # Ranking by weighted_score: same key the opportunities command uses.
        opps = repo.list_all(limit=top)

    if status is not None:
        opps = [o for o in opps if o.reality_status == status]

    if not opps:
        typer.echo("No opportunities to show.")
        typer.echo("Run `founder-radar extract` first, or adjust --top / --status.")
        return

    typer.echo(
        f"Reality view: top {len(opps)} opportunit"
        f"{'y' if len(opps) == 1 else 'ies'} (ranking key: weighted_score)."
    )
    typer.echo("")

    # Group by status for readability.
    by_status: dict[str, list] = {}
    for opp in opps:
        by_status.setdefault(opp.reality_status, []).append(opp)

    status_order = ("underserved", "competitive", "saturated", "unknown")
    status_emoji = {
        "underserved": "[OPPORTUNITY]",
        "competitive": "[FRAGMENTED]",
        "saturated": "[SATURATED]",
        "unknown": "[UNKNOWN]",
    }

    for st in status_order:
        if st not in by_status:
            continue
        typer.echo(f"{status_emoji[st]} {st.upper()}  ({len(by_status[st])} opportunit{'y' if len(by_status[st]) == 1 else 'ies'})")
        for opp in by_status[st]:
            _render_reality_entry(opp)
        typer.echo("")


def _render_reality_entry(opp) -> None:
    """Render one opportunity's reality section."""
    typer.echo(f"  - id={opp.id}  weighted={opp.weighted_score:.2f}  "
               f"confidence={opp.reality_confidence:.2f}")
    typer.echo(f"    title: {opp.title}")
    typer.echo(
        f"    competitor_strength: {opp.competitor_strength_estimate:.2f}  "
        f"distinct_competitors: {opp.distinct_competitor_count}  "
        f"mentions: {opp.competitor_mention_count}"
    )
    # Evidence is derived, not persisted — re-run the validator to get it.
    # We pass precomputed competitor counts so we don't pay the regex
    # cost a second time when the extractor already did it.
    # Warning flags.
    if opp.reality_status == "saturated":
        typer.echo("    [WARN] Saturated market — many competitors exist.")
    if opp.competitor_strength_estimate >= 0.7 and opp.distinct_competitor_count >= 5:
        typer.echo("    [WARN] 5+ competitors named; high entry barrier.")



def _render_evidence(opp) -> None:
    """Re-derive and render the evidence list for one opportunity.

    Evidence is a derived view from the posts + competitor info, so we
    recompute it on the fly. Cheap for the cluster sizes we expect.
    """
    if opp.cluster_id is None:
        return
    try:
        from founder_radar.analysis.reality_validator import assess_reality
        from founder_radar.database.connection import get_session
        from founder_radar.database.repository import PostRepository

        with get_session() as session:
            posts = list(PostRepository(session).list_by_cluster(opp.cluster_id))
        if not posts:
            return
        ra = assess_reality(
            posts,
            competitors=None,
            distinct_competitor_count=opp.distinct_competitor_count,
            competitor_mention_count=opp.competitor_mention_count,
        )
        if not ra.evidence:
            return
        typer.echo("    evidence:")
        for line in ra.evidence[:5]:
            typer.echo(f"      - {line}")
        if len(ra.evidence) > 5:
            typer.echo(f"      - ... and {len(ra.evidence) - 5} more")
    except Exception as exc:
        logger.debug("Could not render evidence for opp %s: %s", opp.id, exc)

# =============================================================================
# info
# =============================================================================


# =============================================================================
# info
# =============================================================================
@app.command()
def info() -> None:
    """Print configuration and database stats. Useful for debugging."""
    settings = get_settings()
    _bootstrap()

    typer.echo(f"Founder Radar v{__version__}")
    typer.echo("")
    typer.echo("Configuration:")
    typer.echo(f"  database_url       = {settings.database_url}")
    typer.echo(f"  reports_dir        = {settings.reports_dir}")
    typer.echo(f"  data_dir           = {settings.data_dir}")
    typer.echo(f"  scan_limit         = {settings.scan_limit_per_subreddit}")
    typer.echo(f"  default_subreddits = {settings.subreddit_list}")
    typer.echo(f"  embedding_backend  = {settings.embedding_backend}")
    typer.echo(f"  embedding_model    = {settings.embedding_model}")
    typer.echo(f"  cluster_threshold  = {settings.cluster_similarity_threshold}")
    typer.echo(f"  llm_model          = {settings.llm_model}")
    typer.echo(
        f"  reddit_credentials = "
        f"{'configured' if settings.reddit_client_id else 'MISSING'}"
    )

    with get_session() as session:
        post_repo = PostRepository(session)
        emb_repo = EmbeddingRepository(session)
        opp_repo = OpportunityRepository(session)
        typer.echo("")
        typer.echo("Database:")
        typer.echo(f"  total posts         = {post_repo.count()}")
        typer.echo(f"  total embeddings    = {emb_repo.count()}")
        sizes = post_repo.cluster_sizes()
        typer.echo(f"  total clusters      = {len(sizes)}")
        if sizes:
            typer.echo(f"  largest cluster     = {max(sizes.values())} post(s)")
        typer.echo(f"  total opportunities = {opp_repo.count()}")
        if settings.llm_api_key:
            typer.echo("  llm_extraction      = enabled (LLM_API_KEY set)")
        else:
            typer.echo("  llm_extraction      = disabled (heuristic only)")


@app.command()
def llm_smoke_test(
    debug_request: bool = typer.Option(
        False, "--debug-request",
        help=(
            "Print a sanitized view of the request payload that would "
            "be sent to the provider (model, response_format, "
            "reasoning_split, thinking, reasoning, timeout). Does NOT "
            "make an HTTP call. Use this to verify provider-specific "
            "fields (e.g. MiniMax-M3 reasoning_split=true) are wired "
            "correctly before spending a real call."
        ),
    ),
) -> None:
    """Smoke-test the configured LLM provider end-to-end.

    Exercises:
      - Provider connectivity (one chat completion call).
      - Model name surfacing.
      - Strict JSON response (response_format=json_object).
      - <think>...</think> block stripping (reasoning-model behaviour).
      - reasoning_split flag handling (top-level pass-through).
      - Retry-repair pass when the first parse fails.

    With no LLM_API_KEY set, the command exits with code 2 and a clear
    message. `--debug-request` is supported WITHOUT an API key because
    it does not make a network call.

    Findings semantics:
      - PASS: the response parses as a JSON dict (initial or after repair).
      - WARN: a <think>...</think> block was present but JSON parsed after
        stripping — the model is leaking its reasoning trace. Consider
        setting LLM_REASONING_SPLIT=true or LLM_THINKING_MODE=disabled.
      - FAIL: the response did not parse as JSON even after think-stripping
        AND retry-repair.

    The point of this command is to make reasoning-model debugging
    cheap: one call (or zero, with --debug-request), one report, no DB
    writes. Run it after editing LLM_REASONING_SPLIT /
    LLM_THINKING_MODE / LLM_RESPONSE_FORMAT and BEFORE running the full
    `extract` or `review-opportunities`.
    """
    _bootstrap()
    settings = get_settings()

    if debug_request:
        provider = OpenAICompatibleProvider(settings)
        test_messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a JSON echo. Respond with one object: "
                    '{"ok": true, "model_acknowledged": "<model-name>"} '
                    "with no commentary, no markdown fences, no "
                    "<think>...</think> blocks."
                ),
            ),
            LLMMessage(role="user", content="Smoke test."),
        ]
        payload = provider.build_request_payload(
            test_messages,
            temperature=0.0,
            max_tokens=200,
        )
        typer.echo("---- sanitized request payload ----")
        typer.echo(
            f"  base_url:                  "
            f"{settings.llm_base_url!r}"
        )
        typer.echo(f"  url:                       {provider._url()}")
        typer.echo(f"  model:                     {payload.get('model')!r}")
        typer.echo(
            f"  response_format:           "
            f"{payload.get('response_format', '(omitted)')!r}"
        )
        typer.echo(
            f"  reasoning_split:           "
            f"{payload.get('reasoning_split', '(omitted)')!r}"
        )
        typer.echo(
            f"  thinking:                  "
            f"{payload.get('thinking', '(omitted)')!r}"
        )
        typer.echo(
            f"  reasoning:                 "
            f"{payload.get('reasoning', '(omitted)')!r}"
        )
        typer.echo(
            f"  extra_body:                "
            f"{payload.get('extra_body', '(none — fields at top level)')!r}"
        )
        typer.echo(f"  timeout_seconds:           {provider._timeout}")
        typer.echo(f"  max_tokens:                200")
        typer.echo(f"  temperature:               0.0")
        typer.echo(
            f"  authorization_header:      "
            f"{'present (api_key configured)' if settings.llm_api_key else 'omitted'}"
        )
        typer.echo(
            f"  api_key:                   (never printed)"
        )
        typer.echo(
            f"  llm_strip_thinking_always: {settings.llm_strip_thinking_always}"
        )
        typer.echo("----------------------------------")
        return

    if not settings.llm_api_key:
        typer.echo(
            "LLM_API_KEY is not set. Set it in .env or your shell, then "
            "re-run this command. (The smoke test always makes one real "
            "chat-completion call to the configured endpoint.) "
            "Pass --debug-request to print the payload without calling.",
            err=True,
        )
        raise typer.Exit(code=2)

    provider = OpenAICompatibleProvider(settings)

    # Single chat completion asking for strict JSON. We use a minimal
    # schema so the call is cheap regardless of model.
    test_messages = [
        LLMMessage(
            role="system",
            content=(
                "You are a JSON echo. Respond with one object: "
                '{"ok": true, "model_acknowledged": "<model-name>"} '
                "with no commentary, no markdown fences, no "
                "<think>...</think> blocks."
            ),
        ),
        LLMMessage(role="user", content="Smoke test."),
    ]

    typer.echo(f"Provider: {provider.name}")
    typer.echo(f"Endpoint: {settings.llm_base_url}")
    typer.echo(f"Model: {settings.llm_model}")
    typer.echo(
        f"Settings: reasoning_split={settings.llm_reasoning_split} "
        f"thinking_mode={settings.llm_thinking_mode!r} "
        f"response_format={settings.llm_response_format!r} "
        f"strip_thinking_always={settings.llm_strip_thinking_always} "
        f"timeout={provider._timeout}s"
    )
    typer.echo("Calling /chat/completions once...")

    try:
        resp = provider.complete(
            test_messages,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:
        typer.echo(f"FAIL: provider call failed: {exc}", err=True)
        raise typer.Exit(code=2)

    content = resp.content or ""
    # When strip_thinking_always=True, the provider has already stripped
    # <think> blocks before returning. We respect that and don't print
    # the raw preview — only show its length.
    if settings.llm_strip_thinking_always:
        typer.echo(
            f"Response received (length={len(content)} chars; "
            "thinking blocks already stripped by provider)."
        )
    else:
        typer.echo("---- response content (first 400 chars) ----")
        typer.echo(content[:400])
        typer.echo("--------------------------------------------")

    # Use the shared JSON pipeline so the smoke test mirrors what
    # `extract` and `review-opportunities` actually see.
    from founder_radar.analysis.llm_json import (
        extract_json,
        make_repair_callback,
        parse_with_repair,
    )

    # Detect thinking blocks in the *raw* response (pre-stripping) so
    # we can WARN even when strip_thinking_always is on. The provider
    # set LLMResponse.content from message.content, so we look there.
    # Note: if strip_thinking_always=True, the provider has already
    # stripped content before we got it; we can still detect whether
    # the original model emitted a block by checking the response
    # length or the "raw_response" attribute on LLMResponse. The
    # base provider doesn't carry raw_response, so for the smoke test
    # we approximate with the stripped content.
    _TAG_RE = re.compile(r"</?(think|thinking|reasoning)\b", re.IGNORECASE)

    has_think_in_visible = bool(_TAG_RE.search(content))
    findings: list[str] = []
    parsed = None
    parse_error: str | None = None
    try:
        parsed = extract_json(content)
    except Exception as exc:
        parse_error = str(exc)

    if parsed is not None:
        if has_think_in_visible:
            findings.append(
                "WARN: response contains <think>...</think> blocks but "
                "still parses as JSON after stripping. The provider is "
                "leaking its reasoning trace. Consider "
                "LLM_REASONING_SPLIT=true or LLM_THINKING_MODE=disabled."
            )
        else:
            findings.append(
                "PASS: response parses as a JSON dict without "
                "thinking-block contamination."
            )
    else:
        # First parse failed. Try the retry-repair path so we exercise
        # the same recovery that `extract` and `review-opportunities`
        # use in production.
        findings.append(
            f"INFO: initial parse failed ({parse_error}); "
            "sending repair prompt..."
        )

        def _llm_complete(messages: list[LLMMessage]) -> str:
            return provider.complete(
                messages,
                temperature=0.0,
                max_tokens=300,
            ).content

        repair = make_repair_callback(
            _llm_complete,
            schema_hint='{"ok": true, "model_acknowledged": "<model-name>"}',
        )
        result = parse_with_repair(content, repair=repair)
        if result.value is not None:
            findings.append(
                "PASS: repair path recovered a valid JSON dict."
            )
        else:
            findings.append(
                f"FAIL: response did not parse as JSON, and the "
                f"repair-retry also failed: {result.last_error}"
            )

    typer.echo("")
    typer.echo("---- findings ----")
    for f in findings:
        typer.echo(f)
    typer.echo("------------------")

    if any(f.startswith("FAIL") for f in findings):
        raise typer.Exit(code=2)
    typer.echo("OK: LLM smoke test passed.")


if __name__ == "__main__":
    # Allows `python -m founder_radar.main` for debugging.
    app()