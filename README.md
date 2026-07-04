# Founder Radar

> Discover software business opportunities by automatically analyzing public discussions.

Founder Radar continuously scans public conversations (Reddit today; Hacker News,
GitHub Issues, Stack Overflow, and more tomorrow), removes noise, and (in later
phases) clusters complaints into ranked software-business opportunities.

This repository is intentionally built in **phases**. **Phase 1** ships the full
data pipeline skeleton for two sources (Reddit + Hacker News) end-to-end: collect → clean →
store → Markdown report. Phase 2 adds embeddings and clustering; Phase 3 adds
LLM opportunity extraction. The per-phase design notes are kept
inside the development workspace; see the git history for the
full architectural decision log.

---

## Why this codebase looks the way it does

Founder Radar is meant to be **maintained over years**, possibly by different
human developers and AI agents. To make that work:

- **One layer per folder.** Every concern (config, database, collectors, …)
  has its own package, base class, and tests. New sources are *new files*,
  not edits to existing ones.
- **One configuration object.** Nothing reads `os.environ` directly. Add a
  field to `Settings` and to `.env.example` and the rest of the code follows.
- **One LLM abstraction.** Swap models by changing `LLM_BASE_URL` /
  `LLM_MODEL`. Swap providers by implementing `BaseLLMProvider`.
- **One dialect-agnostic ORM.** SQLite in Phase 1, PostgreSQL later — no
  query changes, just the URL.
- **No skipping ahead.** Phase 1 *does not* call an LLM. The interface
  exists, the implementation does not. Phase 3 wires it up without
  restructuring anything that came before.

---

## Architecture at a glance

```
              ┌──────────────────────────────────────────┐
              │            CLI (Typer)                  │
              │   collect · clean · report · run · info │
              └────────────────────┬─────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌──────────────┐          ┌─────────────────┐         ┌──────────────┐
│ collectors/  │  ──────▶ │  processors/    │ ──────▶ │  reports/    │
│  reddit.py   │          │   cleaner.py    │         │  markdown_*  │
│  base.py     │          │   base.py       │         │  base.py     │
└──────────────┘          └─────────────────┘         └──────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                          ┌─────────────────┐
                          │   database/     │
                          │  models.py      │
                          │  connection.py  │
                          │  repository.py  │
                          └─────────────────┘
                                   ▲
                                   │
                          ┌─────────────────┐
                          │      llm/       │  (defined, unused in Phase 1)
                          │  base.py        │
                          │  openai_*       │
                          └─────────────────┘
```

Everything is glued by `founder_radar.config.settings.Settings` (typed config)
and `founder_radar.database.connection.get_session` (DB session context
manager). See the source for detailed comments in each module.

---

## Project layout

```
src/founder_radar/
├── main.py                     # Typer CLI
├── config/                     # Settings + logging
├── collectors/                 # Source plugins (Reddit, Hacker News)
├── processors/                 # Data transforms (Cleaner for now)
├── analysis/                   # Embeddings, vector store, clustering, scoring, opportunity extraction
├── llm/                        # LLM provider abstraction
├── database/                   # SQLAlchemy models + connection + repo
├── reports/                    # Output renderers (Markdown for now)
└── utils/                      # Tiny pure helpers
tests/                          # pytest suite, all external I/O mocked
data/                           # SQLite DB lives here
reports/                        # Generated Markdown reports
logs/                           # Rotating log files
.env.example                    # Copy to .env and fill in
pyproject.toml                  # Build + dependencies + CLI entry point
```

---

## Quick start

### 1. Install

Requires Python 3.11+.

```bash
# Clone / cd into the project
cd FounderRadar

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 2. Configure

```bash
# Copy the template
cp .env.example .env            # macOS/Linux
copy .env.example .env          # Windows
```

**Reddit credentials are optional.** Founder Radar ships with two
collectors: Reddit (requires API creds) and Hacker News (no auth). See
[Run without Reddit](#run-without-reddit) below to start with HN only.

To enable Reddit, add your credentials to `.env`:

```ini
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=founder-radar/0.1 (by /u/your_username)
```

To get Reddit credentials:

1. Go to <https://www.reddit.com/prefs/apps>
2. Click **create another app…**
3. Name: anything (e.g. `founder-radar`)
4. Type: **script**
5. Redirect URI: `http://localhost:8080`
6. Copy the client id (the string under the app name) and the secret.

LLM credentials are optional in Phase 1 — only needed from Phase 3.

### 3. Run the pipeline

```bash
# Quick check: see config and DB stats
founder-radar info

# Collect posts from one subreddit
founder-radar collect --subreddit entrepreneur --limit 25

# Generate a Markdown report from what's in the DB
founder-radar report

# Or do it all in one go
founder-radar run --subreddit startups --limit 50
```

Reports are written to `reports/report-YYYYMMDD-HHMMSS.md`.


### Run without Reddit

If you don't have Reddit API credentials — or Reddit's app approval
is blocking your first scan — start with **Hacker News**. The HN
collector uses the **public HN Firebase API** and requires **no API key,
no auth, no signup**. Every page of the run that doesn't mention
Reddit will work.

```bash
# Tell the CLI to use HN — no Reddit needed.
founder-radar collect   --source hackernews   --story-type askstories   --limit 50
```

You can substitute any of: `topstories`, `newstories`, `askstories`,
`showstories`, `beststories`, `jobstories`. Repeat the flag for
multiple feeds, or set `DEFAULT_HN_STORY_TYPES` in `.env`.

Add `--include-comments` to also fetch up to 5 first-level comments
per story (off by default; stories only):

```bash
founder-radar collect --source hackernews --include-comments --limit 25
```

The rest of the pipeline (embed, cluster, extract, opportunities,
reality) works identically on HN data — only the collector source
changes.


### 3b. Phase 2 — Embed, cluster, search

```bash
# Generate embeddings for every post that doesn't have one yet.
# Default backend is sentence-transformers (requires [embeddings] extra).
# Override with --backend null for placeholder vectors (no model load).
founder-radar embed --backend null

# Cluster all embedded posts using the configured similarity threshold.
founder-radar cluster --threshold 0.75

# Inspect the resulting clusters: sizes + representative posts.
founder-radar clusters

# Or focus on one cluster:
founder-radar clusters --cluster 0

# Find posts semantically similar to a free-text query:
founder-radar similar --query "I can't find my first paying customer"

# ...or to an existing post (useful for "more like this"):
founder-radar similar --post-id 42 --limit 5
```

To use real embeddings instead of the placeholder, install the optional
extra:

```bash
pip install founder-radar[embeddings]
```

### 3c. Phase 3 — Extract, score, inspect opportunities

```bash
# Generate one opportunity per cluster. Uses the LLM if LLM_API_KEY is
# set, otherwise the heuristic (still useful).
founder-radar extract
founder-radar extract --heuristic           # skip the LLM entirely
founder-radar extract --cluster 0           # one cluster only

# Inspect ranked opportunities.
founder-radar opportunities
founder-radar opportunities --status new --limit 10

# Detail view of one opportunity.
founder-radar opportunity 1
```

The default report now includes a Top Opportunities section sorted by
total score. Each opportunity shows its 8-factor breakdown.


### 3d. Phase 3+ — Reality check, trends, weighted ranking

```bash
# Phase 3+ ranks opportunities by **pain-dominated weighted_score**:
#   weighted = pain * 0.5 + monetization * 0.4 + novelty * 0.1
# where pain = dissatisfaction * 0.4 + emotional_intensity * 0.4 + frequency * 0.2.
#
# This means a high-pain problem in a saturated market outranks a fresh
# topic with low engagement. Novelty is intentionally weighted lowest
# (10%) per the brief.

# Reality check: competitors + saturation
founder-radar validate 1                    # full reality check breakdown
founder-radar competitors 1                 # just the competitors

# Trend classification
founder-radar trends                        # all clusters, with trend labels
founder-radar trends --trend emerging       # only emerging
founder-radar trends --sort size            # most posts first
founder-radar cluster-history 3             # timeline + reality check for one cluster
```

The default report and `opportunities` list now sort by
`weighted_score` instead of the unweighted mean. The Reality Check
runs deterministically (no LLM) on every cluster.

### 3. Run the pipeline

```bash
# Quick check: see config and DB stats
founder-radar info

# Collect posts from one subreddit
founder-radar collect --subreddit entrepreneur --limit 25

# Generate a Markdown report from what's in the DB
founder-radar report

# Or do it all in one go
founder-radar run --subreddit startups --limit 50
```

Reports are written to `reports/report-YYYYMMDD-HHMMSS.md`.

### 4. Run the tests
```

### 3. Run the pipeline

```bash
# Quick check: see config and DB stats
founder-radar info

# Collect posts from one subreddit
founder-radar collect --subreddit entrepreneur --limit 25

# Generate a Markdown report from what's in the DB
founder-radar report

# Or do it all in one go
founder-radar run --subreddit startups --limit 50
```

Reports are written to `reports/report-YYYYMMDD-HHMMSS.md`.

### 4. Run the tests

The full test suite uses **mocked** Reddit and LLM calls, so no credentials
are required:

```bash
pytest
```

---

## Configuration reference

All settings live in `.env` (copy from `.env.example`). Read them in code via
`from founder_radar.config.settings import get_settings; s = get_settings()`.

| Variable | Default | Purpose |
|---|---|---|
| `REDDIT_CLIENT_ID` | _(empty)_ | OAuth client id from reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | _(empty)_ | OAuth client secret |
| `REDDIT_USER_AGENT` | `founder-radar/0.1` | Reddit requires a unique UA |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | _(empty)_ | Bearer token; optional for local servers |
| `LLM_MODEL` | `gpt-4o-mini` | Model name to call |
| `DATABASE_URL` | `sqlite:///data/founder_radar.db` | SQLAlchemy URL (PostgreSQL later) |
| `SCAN_LIMIT_PER_SUBREDDIT` | `50` | Posts per subreddit per run |
| `DEFAULT_SUBREDDITS` | `entrepreneur,startups,SaaS,smallbusiness,indiehackers` | Comma-separated |
| `REPORTS_DIR` | `reports` | Where Markdown reports are written |
| `DATA_DIR` | `data` | Where the SQLite DB lives |
| `LOGS_DIR` | `logs` | Where rotating log files go |
| `LOG_LEVEL` | `INFO` | Root logger level |

---

## Phase roadmap

- [x] **Phase 1** — Reddit collector + SQLite + Markdown report
- [x] **Phase 2** — Embeddings, clustering, semantic search
- [x] **Phase 3** — LLM opportunity extraction + scoring
- [x] **Phase 3+** — Reality check (competitors + saturation) + trend analysis + pain-dominated weighted ranking
- [ ] **Phase 4** — Web dashboard
- [ ] **Phase 5** — Multi-source support (HN, GH Issues, ...)
- [ ] **Phase 6** — Continuous scanning / scheduler

Each phase is fully validated before the next starts. The architecture is
designed so no phase forces a rewrite of an earlier one.

---

## For future AI agents

If you are reading this to extend Founder Radar, the rules of the road are:

1. **Add, don't rewrite.** New sources → new file under `collectors/`. New
   processing → new file under `processors/`. New output format → new file
   under `reports/`.
2. **Update `.env.example`** whenever you add a `Settings` field.
3. **Update the registry** in `collectors/__init__.py::register_builtins`
   when you add a new source so the CLI can find it.
4. **Write tests for the new module** that follow the existing patterns
   (mock external I/O, use a temp SQLite, assert on observable behavior).
5. **Never write type-suppressing annotations** (`as any`, `# type: ignore`)
   — fix the underlying types instead.
6. **Never commit `.env`, the `data/*.db` file, or generated reports.**
   The `.gitignore` already excludes them; keep it that way.

The per-phase design notes (kept in the development workspace) capture the rationale
behind every Phase 1 decision so you don't have to re-derive them.

---

## License

MIT.