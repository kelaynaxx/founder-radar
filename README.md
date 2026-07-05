# Founder Radar

> Discover software business opportunities by automatically analyzing public discussions.

Founder Radar continuously scans public conversations (Reddit, Hacker News,
GitHub Issues, Stack Overflow, and more tomorrow), removes noise, and
(in later phases) clusters complaints into ranked software-business
opportunities.
This repository is intentionally built in **phases**. The first
release ships the full data pipeline skeleton for three sources
(Reddit + Hacker News + GitHub Issues) end-to-end: collect → clean →
store → Markdown report.
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
│  hackernews  │          │   base.py       │         │  base.py     │
│  github.py   │          │                 │         │              │
│  base.py     │          │                 │         │              │
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
├── collectors/                 # Source plugins (Reddit, Hacker News, GitHub Issues)
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

### Using HN while Reddit credentials are pending

If your Reddit Data Access approval is still pending (or you just
want to test the pipeline end-to-end), HN is a fully functional
secondary source. The HN collector:

- Uses the public HN Firebase API and the public Algolia HN Search
  API. **No API key, no auth, no signup.**
- Is well-suited for calibration: Ask HN posts surface user pain
  questions, Show HN surfaces product launches.
- Has a built-in `subtype` tag (`ask_hn`, `show_hn`, `regular_story`,
  `regular_comment`, `job`) that downstream code can use to
  downrank pure launches.

The rest of the pipeline (embed, cluster, extract, opportunities,
reality) is source-agnostic. HN is treated as a first-class source
alongside Reddit — only the collector is different.

### Using GitHub Issues while Reddit access is pending

If your Reddit Data Access approval is still pending — or HN calibration
produced too little signal — GitHub Issues is the highest-quality
secondary source available. Public issues are filled with explicit
pain signals: bug reports, feature requests, missing functionality,
broken workflows, and tool limitations. The GitHub collector:

- Uses the **public GitHub REST API** (`api.github.com`). **No auth
  required by default.**
- Optionally accepts a `GITHUB_TOKEN` (personal access token) for a
  higher rate limit: 60 requests/hour (anonymous) → 5,000 requests/hour
  (authenticated). Mint a token at
  <https://github.com/settings/tokens> (classic) or
  <https://github.com/settings/personal-access-tokens> (fine-grained).
  No scopes are needed for public-issue reads.
- Supports **two collection modes**:
  1. **Repo mode** — list issues for one or more `owner/name` repos
     via `GET /repos/{owner}/{repo}/issues`.
  2. **Search mode** — search across all of GitHub via
     `GET /search/issues`. Supports the full qualifier language
     (`is:issue`, `is:open`, `label:bug`, `repo:owner/name`, etc.).
- Carries a built-in `subtype` tag taxonomy
  (`bug`, `feature_request`, `enhancement`, `question`, `bot_update`,
  `unknown`) so downstream code can target specific issue kinds.
- **Filters out** pull requests (PRs come through the same endpoint
  and are marked with a `pull_request` key) and **defaults to open
  issues** (use `--include-closed` to opt in to closed).
- **Filters out** automated bot issues (dependabot, renovate,
  github-actions, …) and template-only issues by default. Use
  `--include-bots` and `--include-templates` to keep them.

#### Quick start

```bash
# Scan a single repo
founder-radar collect --source github --repo openai/openai-python --limit 100

# Scan multiple repos (repeat the flag)
founder-radar collect --source github --repo openai/openai-python --repo langchain-ai/langchain --limit 100

# Search GitHub for open bug reports
founder-radar collect --source github --query "is:issue is:open label:bug" --limit 100

# Search for feature requests in a topic
founder-radar collect --source github --query 'is:issue is:open "feature request" automation' --limit 100

# Include closed issues too
founder-radar collect --source github --repo openai/openai-python --include-closed --limit 100
```

#### Optional: raise the rate limit

```bash
# In .env:
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The collector will pick this up automatically; no code change needed.
60 requests/hour is enough for ~5 calibration scans of 100 issues
each. With a token you can comfortably scan dozens of repos per hour.

#### What it does NOT do

- **No comment collection.** Issues are collected as a flat stream;
  the comments thread is left for a future pass. This is the same
  trade-off HN's `--include-comments` opt-in makes.
- **No pull requests.** PRs are filtered at the `pull_request` key.
  This is intentional: PRs rarely surface end-user pain.
- **No scoring changes.** GitHub issues go into the same downstream
  pipeline as Reddit and HN. Downstream code can use the
  `subtype='bot_update'` tag to downrank dependency-update noise.

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

### 3e. Phase 4+ — Opportunity type calibration (`productizable`)

V2 (calibration pass 2) tightened the classifier after a real GitHub
run produced false-positive `potential_product` rows for SDK / API /
type-mismatch errors (e.g. `BadRequestError max_tokens` in the openai
SDK, `annotation typed as object instead of Annotation`). The new
rules are:

- **`upstream_library_bug`** is a new label. SDK / API / type / schema
  / serialization failures land here, not in `integration_pain` or
  `potential_product`.
- **`potential_product` requires `reality_status` in
  {`underserved`, `competitive`}** — never `unknown`, never `saturated`.
  `unknown` was the #1 source of false positives.
- **`upstream_library_bug` blocks `potential_product`.** Even when all
  four structural conditions are met, a cluster dominated by upstream
  SDK cues is demoted.
- **Per-type score caps.** A `repo_specific_bug` can never reach
  `productizability_score > 0.25`; `upstream_library_bug` is capped at
  `0.30`; `potential_product` requires `>= 0.70`.
- **Security lexicon tightened.** `401` / `permission denied` /
  `unauthorized` no longer trigger `security_compliance_pain`. Only
  explicit security/privacy/compliance/vuln cues do.

The classifier maps each `Opportunity` to one of ten labels and a
`productizability_score` in `[0, 1]`. It does NOT change
`weighted_score` or any other ranking.

**The ten `opportunity_type` labels:**

| Type | What it means | Score cap |
|---|---|---|
| `repo_specific_bug` | Stack-trace / regression / `TypeError` issues scoped to one repo | 0.25 |
| `upstream_library_bug` | SDK / API / type / schema / serialization errors (`BadRequestError`, `typed as object`, `pydantic v2`, …) | 0.30 |
| `documentation_confusion` | "where are the docs?", "how do I install?", missing tutorials | 0.25 |
| `missing_feature` | "please add X", "feature request", "I wish it had Y" | 0.45 |
| `integration_pain` | API / SDK / webhook / OAuth / connector issues (not library bugs) | 0.65 |
| `developer_workflow_pain` | Repetitive / tedious / CI / deploy friction | 0.55 |
| `infra_operational_pain` | Rate limit / retry / timeout / monitoring / scaling | 0.60 |
| `security_compliance_pain` | Explicit GDPR / SOC2 / PII / XSS / SQL injection / RCE / vulnerability | 0.55 |
| `potential_product` | Cross-tool pain + clear buyer + underserved\|competitive reality + score >= 0.70 | 1.0 (floor 0.70) |
| `unknown` | Signal too thin to classify | 0.10 |

`potential_product` is the strictest. All four structural conditions
must pass **and** the cluster must NOT be primarily an upstream library
bug **and** the resulting score must be >= 0.70:

1. **cross-cutting** (multiple sources OR multiple tools mentioned)
2. **repeated** (`mentions >= 5`)
3. **open market** (`reality_status in {underserved, competitive}` AND
   `competitor_strength < 0.55`)
4. **real pain** (or buyer language) **AND NOT primarily upstream**

High `weighted_score` alone is explicitly NOT enough.

**CLI usage:**

```bash
# Top 30 productizable opportunities (sorted by productizability_score)
founder-radar productizable --top 30

# Filter by a single type
founder-radar productizable --type potential_product
founder-radar productizable --type integration_pain
founder-radar productizable --type developer_workflow_pain
founder-radar productizable --type missing_feature
founder-radar productizable --type upstream_library_bug

# Exclude noise (repeatable)
founder-radar productizable --exclude upstream_library_bug
founder-radar productizable --exclude upstream_library_bug --exclude repo_specific_bug

# Combine type + minimum score
founder-radar productizable --type integration_pain --min-score 0.5

# Recompute types after editing posts (otherwise reads from the DB)
founder-radar productizable --recalculate
```
Output columns (one row per opportunity):
`id`, `title`, `weighted_score`, `reality_status`, `opportunity_type`,
`productizability_score`, `productizability_reason`.

### 3f. Phase 4+ — LLM-assisted opportunity review (`review-opportunities`)

The deterministic `opportunity_type` + `productizability_score` says
"this cluster LOOKS productizable." The review layer is a separate
optional step that asks an LLM (default) or a deterministic
fallback (`--use-heuristic`) "is this cluster ACTUALLY a product
opportunity?" It's a strict startup-analyst filter.

The deterministic layer is still the source of truth. The review
layer is a triage filter, not a generator. By default it
**rejects**. To keep an opportunity after review, the LLM must
demonstrate clear repeated pain, a clear buyer/persona, and a
plausible standalone tool/service.

**Review verdicts (3):**

| Verdict | Meaning |
|---|---|
| `reject` | Default. Not a real product opportunity. |
| `maybe` | Plausible but unproven. Could become a tool. |
| `strong_candidate` | Strong evidence of real product opportunity. |

**Review reasons (11):** `repo_internal_task`, `upstream_bug`,
`maintenance_chore`, `documentation_only`, `not_buyer_pain`,
`too_vague`, `too_repo_specific`, `possible_devtool`,
`possible_micro_saas`, `possible_infra_tool`, `strong_repeated_pain`.

**Safety nets:**
- An LLM that returns `strong_candidate` for a non-`potential_product`
  cluster is **demoted to `maybe`** (the deterministic classifier
  wins).
- An LLM that returns `maybe` with all-reject-class reasons is
  **demoted to `reject`**.
- An LLM that returns invalid JSON, prose, fences, or raises an
  exception — the verdict is `reject` with reason `review_failed`.
  The CLI never crashes on a bad LLM response.

**CLI usage:**

```bash
# Default: review un-reviewed opportunities with the LLM.
founder-radar review-opportunities --top 50

# Filter by verdict
founder-radar review-opportunities --verdict strong_candidate
founder-radar review-opportunities --verdict maybe
founder-radar review-opportunities --verdict reject

# Hide all reject verdicts
founder-radar review-opportunities --exclude-rejected

# Re-review every opportunity, even ones with existing verdicts
founder-radar review-opportunities --rerun-all

# Smoke test without an LLM endpoint (deterministic fallback).
# Always returns 'reject' for non-`potential_product` clusters and
# 'maybe' for the rest. NEVER returns 'strong_candidate'.
founder-radar review-opportunities --use-heuristic
```

Output columns: `id`, `title`, `opportunity_type`,
`productizability_score`, `review_verdict`, `review_reasons`,
`review_summary`, `review_confidence`. The CLI never requires
the LLM to run — without `LLM_API_KEY` set, the command exits with
an error unless `--use-heuristic` is passed.

The LLM is configured via `Settings.llm_api_key` / `llm_base_url` /
`llm_model` — same settings the `extract` command uses. Any
OpenAI-compatible endpoint works (LM Studio, Ollama with openai
shim, vLLM, etc.).

### 3g. Phase 4+ — Reasoning-model LLM support (MiniMax-M3, DeepSeek R1, o1)

V2.1 added first-class support for LLMs that inline a chain-of-thought
trace into `message.content` before the JSON answer. By default the
classifier expected a clean JSON response — reasoning models like
**MiniMax-M3** broke that contract, producing output like:

    <think>Let me analyze this cluster carefully...
</think>

    {"verdict":"reject"}

That JSON looks valid after stripping the <think>...</think> block,
but `json.loads(raw)` raised `JSONDecodeError` and the CLI fell back
to a heuristic verdict with reason `review_failed`. V2.1 fixes this
in three layers:

1. **Robust JSON extraction** (`analysis/llm_json.py`):
   - Strips `<think>...</think>` blocks (including unterminated ones).
   - Strips markdown `\`\`\`json ... \`\`\`` fences.
   - Finds the first balanced `{...}` block.
   - Tries each candidate in order; raises `LLMJsonError` with the
     first `ERROR_PREVIEW_CHARS` (300) chars on total failure.

2. **One retry-repair pass**: when the first parse fails, the
   extractor/reviewer sends a follow-up prompt asking the LLM to
   convert its previous output into valid JSON. Only ONE repair pass
   is attempted — we accept defeat after that.

3. **Provider-specific extras** (`OpenAICompatibleProvider`):
   - `response_format` (default `json_object`) — standard OpenAI field.
   - `extra_body.reasoning_split` — MiniMax-M3 / DeepSeek convention
     for splitting the reasoning trace into a separate field.
   - `extra_body.thinking.type` and `extra_body.reasoning.effort` —
     Anthropic and OpenAI-compatible conventions for reasoning-mode.
   - Recovery: when `content=""` and the provider puts the answer in
     `reasoning_content` / `reasoning_text` / `reasoning`, we copy it
     back into `content` so downstream parsers see it.

**Three new settings** (see `.env.example` for the full list):

| Setting | Default | Purpose |
|---|---|---|
| `LLM_REASONING_SPLIT` | `false` | Send `extra_body.reasoning_split=true` to the provider. |
| `LLM_THINKING_MODE` | `empty` | `disabled` / `adaptive` / `enabled` / `empty` (MiniMax-M3 convention). |
| `LLM_RESPONSE_FORMAT` | `json_object` | OpenAI `response_format` field; set to `none` to skip. |

**Worked example for MiniMax-M3:**

```bash
# In your .env:
LLM_BASE_URL=https://api.MiniMax.chat/v1
LLM_API_KEY=<your MiniMax-M3 key>
LLM_MODEL=MiniMax-M3
LLM_REASONING_SPLIT=true
LLM_THINKING_MODE=disabled
LLM_RESPONSE_FORMAT=json_object
```

After changing these, verify with one cheap call BEFORE running the
full extract / review:

```bash
founder-radar llm-smoke-test
```

The smoke test makes one chat-completion call asking for a minimal
strict-JSON response and reports:

  - Provider name + endpoint + model
  - Whether `<think>...</think>` blocks appear in content (FAIL if yes)
  - Whether the response parses as a JSON dict (PASS/FAIL)
  - Whether the response was wrapped in markdown fences (INFO)

Exit code is 2 on any FAIL, 0 on PASS.

**CLI changes:**

```bash
# extract: explicit --method option (defaults to 'auto')
founder-radar extract --method heuristic        # force heuristic even if LLM key set
founder-radar extract --method llm             # force LLM (errors if no key)
founder-radar extract --method auto             # use LLM if key present, else heuristic
founder-radar extract                          # same as --method auto

# llm-smoke-test: one-call diagnostic for reasoning-model debugging
founder-radar llm-smoke-test
```

`--heuristic` on `extract` is kept as a deprecated hidden alias for
`--method heuristic`.




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
| `GITHUB_TOKEN` | _(empty)_ | Optional. Raises REST rate limit from 60/hr to 5,000/hr |
| `GITHUB_USER_AGENT` | `founder-radar/0.1 (...)` | GitHub requires a unique UA |
| `GITHUB_API_BASE` | `https://api.github.com` | Override for GitHub Enterprise |
| `GITHUB_INCLUDE_BOTS` | `false` | Keep bot-authored issues (dependabot, renovate, …) |
| `GITHUB_INCLUDE_TEMPLATES` | `false` | Keep template-only issues (no body, generic title) |
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
- [x] **Phase 4a** — Opportunity-type calibration V1 (`productizable` CLI; nine `opportunity_type` labels + `productizability_score`)
- [x] **Phase 4b** — Opportunity-type calibration V2 (new `upstream_library_bug` label; reality_status gate; per-type score caps; tightened security lexicon; `--exclude` filter)
- [x] **Phase 4c** — LLM-assisted opportunity review (`review-opportunities` CLI; three `review_verdict`s + 11 reasons; safety nets; `--use-heuristic` fallback)
- [x] **Phase 5a** — Hacker News + GitHub Issues collectors (multi-source; Reddit remains the primary)
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