"""Centralized, typed application settings.

This module is the *only* place that should read environment variables or
`.env` files. Every other module gets a `Settings` instance injected (or
imports `get_settings()`) and reads typed attributes off it.

Why Pydantic Settings instead of plain `os.getenv`?
  - Type coercion and validation (e.g. SCAN_LIMIT_PER_SUBREDDIT becomes int)
  - Single object exposes the whole config to introspection / docs / tests
  - Default values are colocated with the field, easy to evolve

Adding a new setting:
  1. Add a typed attribute below with a sensible default.
  2. Document it in `.env.example`.
  3. Read it via `settings.your_new_field` from anywhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root once at module import. We do this at module
# level (not inside the class) because lambdas inside `default_factory`
# cannot see names defined in the class body — Python's class body is
# not a normal closure scope.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application-wide settings loaded from .env + environment variables.

    Pydantic Settings reads environment variables first, then falls back to
    values declared in the `.env` file (if present). Defaults below apply
    only when neither is set, so a fresh clone "just works" for local dev.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Reddit
    # -------------------------------------------------------------------------
    reddit_client_id: str = Field(default="", description="Reddit OAuth client id")
    reddit_client_secret: str = Field(default="", description="Reddit OAuth client secret")
    reddit_user_agent: str = Field(
        default="founder-radar/0.1",
        description="User-Agent string. Reddit requires a unique one per app.",
    )

    # -------------------------------------------------------------------------
    # LLM (OpenAI-compatible)
    # -------------------------------------------------------------------------
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible endpoint. Override for local servers.",
    )
    llm_api_key: str = Field(default="", description="API key for the LLM endpoint")
    llm_model: str = Field(default="gpt-4o-mini", description="Default LLM model name")

    # Provider-specific extras (V2.1: MiniMax-M3 reasoning support).
    # These are intentionally generic so they cover MiniMax-M3, DeepSeek,
    # Anthropic Claude with extended thinking, OpenAI o1-family, and any
    # future reasoning model that speaks the OpenAI Chat Completions shape.
    #
    # LLM_REASONING_SPLIT=true asks providers like MiniMax-M3 to return
    # the reasoning trace in a SEPARATE field (reasoning_content /
    # reasoning_text) instead of inlining it into message.content. The
    # provider then sets content="" + a parallel reasoning field, which
    # keeps strict-JSON contracts intact.
    #
    # LLM_THINKING_MODE controls the reasoning-mode field. Allowed:
    # "disabled" (no reasoning), "adaptive" (model decides),
    # "enabled" (always reason), or "empty" (send the field with an
    # empty value to disable). The actual JSON key the provider expects
    # varies — see `_build_extra_body()` in openai_provider.py.
    #
    # LLM_RESPONSE_FORMAT is the value for the OpenAI `response_format`
    # field. Allowed: "json_object" (constrain output to valid JSON) or
    # "none" (skip the field). Default "json_object".
    llm_reasoning_split: bool = Field(
        default=False,
        description=(
            "Send `extra_body.reasoning_split=true` to the provider. "
            "MiniMax-M3, DeepSeek, and similar reasoning models "
            "honour this to keep reasoning out of message.content."
        ),
    )
    llm_thinking_mode: str = Field(
        default="empty",
        description=(
            "Reasoning-mode field. One of: 'disabled', 'adaptive', "
            "'enabled', 'empty'. 'empty' means send the field with an "
            "empty value (MiniMax-M3 convention) to disable reasoning."
        ),
    )
    llm_response_format: str = Field(
        default="json_object",
        description=(
            "Value for the OpenAI `response_format` field. One of: "
            "'json_object' (constrain output to valid JSON) or 'none' "
            "(skip the field). Default 'json_object'."
        ),
    )

    # HTTP timeout for LLM provider requests. Reasoning-model responses can
    # be slow (and the provider may be a local server); 120s is a
    # comfortable default. Set lower for quick smoke tests, higher if the
    # provider routinely takes >60s.
    llm_timeout_seconds: float = Field(
        default=120.0, ge=1.0, le=3600.0,
        description=(
            "HTTP timeout for LLM provider requests in seconds. "
            "Default 120. Reasoning-model responses can be slow; "
            "raise this if you see httpx.TimeoutException."
        ),
    )

    # When True (default), the provider strips <think>...</think> blocks
    # from the model's response.content before returning it to callers.
    # Reasoning models (MiniMax-M3, DeepSeek R1, o1-family) inline their
    # chain-of-thought into message.content when reasoning_split=true is
    # NOT honored by the provider. Stripping here is a safety net so the
    # downstream JSON parser sees clean content.
    llm_strip_thinking_always: bool = Field(
        default=True,
        description=(
            "Strip <think>...</think> blocks from the model's response "
            "before returning it. Default True for reliability with "
            "reasoning models. Set False only when you want to inspect "
            "the raw response (e.g. in llm-smoke-test --debug-request)."
        ),
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="sqlite:///data/founder_radar.db",
        description="SQLAlchemy URL. SQLite in Phase 1, PostgreSQL later.",
    )

    # -------------------------------------------------------------------------
    # Embeddings (Phase 2)
    # -------------------------------------------------------------------------
    # The backend name selects which `BaseEmbedder` implementation the CLI
    # uses. "sentence-transformers" is the local default; "null" returns
    # zero vectors (useful for tests and for users who haven't installed
    # the heavy ML deps yet); "openai" hits OpenAI's embeddings API.
    embedding_backend: str = Field(
        default="sentence-transformers",
        description="One of: 'sentence-transformers', 'openai', 'null'.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name passed to the embedder backend. For sentence-transformers, this is the HF hub id.",
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=512,
        description="How many texts to embed per batch. Larger = faster but more RAM.",
    )

    # -------------------------------------------------------------------------
    # Clustering (Phase 2)
    # -------------------------------------------------------------------------
    cluster_similarity_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Cosine similarity threshold. Posts with similarity >= this are in the same cluster. Higher = tighter clusters.",
    )
    # Calibration: opportunity extraction should only convert clusters
    # with at least N posts into opportunities, otherwise we get a flood
    # of fake one-post "opportunities". Set to 2 by default; CLI can
    # override per-run with --min-cluster-size.
    extract_min_cluster_size: int = Field(
        default=2, ge=1, le=100,
        description=(
            "Minimum posts per cluster required to create an opportunity. "
            "Singletons are skipped by default to prevent fake 'opportunities'."
        ),
    )

    # -------------------------------------------------------------------------
    # Pipeline limits
    # -------------------------------------------------------------------------
    scan_limit_per_subreddit: int = Field(
        default=50, ge=1, le=1000, description="Posts to fetch per subreddit per run."
    )
    default_subreddits: str = Field(
        default="entrepreneur,startups,SaaS,smallbusiness,indiehackers",
        description="Comma-separated subreddits to scan when none are specified.",
    )

    # -------------------------------------------------------------------------
    # Hacker News (no-auth, public HN Firebase API)
    # -------------------------------------------------------------------------
    default_hn_story_types: str = Field(
        default="topstories,newstories,askstories,showstories",
        description=(
            "Comma-separated HN story types to scan when --story-type is not "
            "specified. One of: topstories, newstories, askstories, showstories, "
            "beststories, jobstories."
        ),
    )
    hackernews_user_agent: str = Field(
        default="founder-radar/0.1 (HN collector; contact via project README)",
        description=(
            "User-Agent string for HN API requests. HN asks that bot operators "
            "identify themselves with a contact URL or address."
        ),
    )

    # -------------------------------------------------------------------------
    # GitHub Issues (no-auth by default; token optional for higher rate limits)
    # -------------------------------------------------------------------------
    # GitHub's public REST API allows 60 unauthenticated requests/hour per IP.
    # Setting GITHUB_TOKEN raises the limit to 5,000/hour. We never *require*
    # the token — collectors work without it — but rate-limit-aware users
    # should set one. See README for how to mint a token.
    github_token: str = Field(
        default="",
        description=(
            "Optional GitHub personal access token. Raises the REST API "
            "rate limit from 60/hour (anonymous) to 5000/hour (authenticated)."
        ),
    )
    github_user_agent: str = Field(
        default="founder-radar/0.1 (GitHub Issues collector; see project README)",
        description=(
            "User-Agent string for GitHub API requests. GitHub requires a "
            "unique UA and recommends including a contact URL."
        ),
    )
    github_api_base: str = Field(
        default="https://api.github.com",
        description="Base URL for GitHub's REST API. Override for GitHub Enterprise.",
    )
    # When True, issues authored by automated bot accounts (dependabot,
    # renovate, github-actions, etc.) are kept and tagged with a
    # subtype='bot_update' marker so downstream code can downrank them.
    # When False (default), bot-typed issues are silently skipped at
    # collection time.
    github_include_bots: bool = Field(
        default=False,
        description=(
            "Keep issues authored by automated bot accounts. "
            "When False (default), bot issues are filtered out at collection."
        ),
    )
    # When True, "template-only" issues (no body, no useful title) are kept.
    # When False (default), they are filtered out — these are usually blank
    # issue-form submissions and rarely carry actionable pain signals.
    github_include_templates: bool = Field(
        default=False,
        description=(
            "Keep template-only issues (no body, generic title). "
            "When False (default), these are filtered at collection time."
        ),
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Defaults resolve against the project root (two levels up from this file:
    # src/founder_radar/config/settings.py -> ../../..). Tests override these
    # by passing explicit values to the Settings() constructor.
    reports_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "reports")
    data_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data")
    logs_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "logs")

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Root log level (DEBUG/INFO/...).")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @property
    def subreddit_list(self) -> list[str]:
        """Return `default_subreddits` as a clean list.

        Strips whitespace and drops empty entries so a trailing comma in the
        env file does not produce an empty subreddit name.
        """
        return [s.strip() for s in self.default_subreddits.split(",") if s.strip()]

    @property
    def hn_story_type_list(self) -> list[str]:
        """Return `default_hn_story_types` as a clean list."""
        return [s.strip() for s in self.default_hn_story_types.split(",") if s.strip()]

    def ensure_paths(self) -> None:
        """Create the on-disk directories the app expects to exist.

        Idempotent. Called once at CLI startup.
        """
        for path in (self.reports_dir, self.data_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a memoized `Settings` instance.

    Cached so repeated reads (very common across modules) don't re-parse the
    environment every time. Reset the cache in tests with
    `get_settings.cache_clear()` after mutating env vars.
    """
    return Settings()