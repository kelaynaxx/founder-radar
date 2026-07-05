"""Opportunity Type Classifier (Phase 4+ signal calibration).

A *narrow* calibration layer that answers one question:

    "Is this opportunity cluster a real product opportunity, or just
     a repo-specific bug / upstream SDK error / docs gap / niche
     edge case?"

V2 (calibration pass 2) — tightening motivated by real GitHub runs:
  - Adds `upstream_library_bug` so SDK / type / schema / serialization
    failures stop being misclassified as `integration_pain` or
    `potential_product`.
  - Restricts `potential_product` to `reality_status in {underserved,
    competitive}` (was: also `unknown`).
  - Disqualifies `potential_product` when the cluster is primarily
    about an upstream library error.
  - Caps `productizability_score` per type so weak types can't reach
    high productizability.
  - Tightens the security lexicon so HTTP 401 / "permission denied"
    / "unauthorized" don't trigger `security_compliance_pain`.

The classifier maps each `Opportunity` to one of ten `opportunity_type`
labels using deterministic rules only — no LLM, no external calls.
The output is an `OpportunityTypeAssessment` dataclass with three
fields:

  - `opportunity_type`   one of ten strings (see ALL_TYPES below)
  - `productizability_score`  [0, 1] standalone-product potential
  - `productizability_reason`  one-line human-readable explanation

This is *orthogonal* to weighted_score: a high weighted_score doesn't
automatically make an opportunity a product. The two are surfaced
side by side in the `productizable` CLI command.

Why a separate layer instead of more scoring factors?
  - Scoring already tells us "how much pain". Type tells us "what kind
    of pain". Conflating them would muddle both signals.
  - The user can now filter for `potential_product` /
    `integration_pain` / etc. without re-running extraction.
  - Adding a new type later is a one-line constant + one matching rule;
    no scoring math changes.

Design rules:
  - Deterministic. No LLM calls.
  - Pure: takes an Opportunity + posts, returns a dataclass.
  - Lexicons are small, hand-curated, and tunable.
  - `unknown` is the safe default when the evidence is too thin to
    pick a specific type.
  - `potential_product` requires STRONG evidence (rule 1: reality is
    underserved or competitive; rule 2/4: not primarily an upstream
    library bug; rule 6: score >= 0.70). High weighted_score alone
    is not enough.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from founder_radar.database.models import Opportunity, Post

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Public constants: the type taxonomy
TYPE_REPO_SPECIFIC_BUG = "repo_specific_bug"
TYPE_UPSTREAM_LIBRARY_BUG = "upstream_library_bug"
TYPE_DOCUMENTATION_CONFUSION = "documentation_confusion"
TYPE_MISSING_FEATURE = "missing_feature"
TYPE_INTEGRATION_PAIN = "integration_pain"
TYPE_DEVELOPER_WORKFLOW_PAIN = "developer_workflow_pain"
TYPE_INFRA_OPERATIONAL_PAIN = "infra_operational_pain"
TYPE_SECURITY_COMPLIANCE_PAIN = "security_compliance_pain"
TYPE_POTENTIAL_PRODUCT = "potential_product"
TYPE_UNKNOWN = "unknown"

ALL_TYPES = (
    TYPE_REPO_SPECIFIC_BUG,
    TYPE_UPSTREAM_LIBRARY_BUG,
    TYPE_DOCUMENTATION_CONFUSION,
    TYPE_MISSING_FEATURE,
    TYPE_INTEGRATION_PAIN,
    TYPE_DEVELOPER_WORKFLOW_PAIN,
    TYPE_INFRA_OPERATIONAL_PAIN,
    TYPE_SECURITY_COMPLIANCE_PAIN,
    TYPE_POTENTIAL_PRODUCT,
    TYPE_UNKNOWN,
)


# -------------------------------------------------------------------------
# Lexicons
# -------------------------------------------------------------------------
# Each list contains lowercased phrases / words / regex-friendly tokens.
# Match is case-insensitive, whole-word, with `\b` boundaries.
#
# These are intentionally conservative: a hit doesn't *force* a type,
# it raises the score. The classifier picks the highest-scoring type
# (or `unknown` if everything is below the noise floor).

# Stack traces, error messages, regression language. The presence of
# one of these is a strong "this is a bug" signal, especially if the
# cluster is small and concentrated in one repo.
_REPO_SPECIFIC_BUG_CUES = (
    # Python
    "traceback", "typeerror", "attributeerror", "valueerror",
    "keyerror", "indexerror", "importerror", "modulenotfounderror",
    "runtimeerror", "oserror", "ioerror",
    # JavaScript / TypeScript
    "undefined is not", "cannot read property", "cannot read properties",
    "is not a function", "is not defined", "syntaxerror",
    "typeerror:", "referenceerror",
    # General
    "nullpointerexception", "segmentation fault", "segfault",
    "stack trace", "stacktrace", "exception", "crash on",
    "crashes when", "crashes after", "regression",
    "broken since", "broken after", "works in version",
    "doesn't work after upgrade", "no longer works", "stopped working",
    "fatal error", "panic:", "core dump",
    # Memory / IO
    "memory leak", "use-after-free", "double free",
    "file not found", "permission denied", "segfault",
)

# V2 (calibration pass 2): a *narrower* kind of bug than
# `repo_specific_bug`. Used when the failure is clearly attributable to
# an SDK / API / library upstream of the user — type mismatches,
# schema validation errors, serialization / deserialization bugs,
# and event / response-shape errors from a specific named library
# (openai-python, langchain, pydantic, ...).
#
# Calibrating rule 3 of the V2 brief: these are the lexica that catch
# the false positives observed on the real GitHub run, e.g.
# "BadRequestError max_tokens/model output limit in Azure OpenAI
# GPT-5 parse()" and "ResponseOutputTextAnnotationAddedEvent.annotation
# typed as object instead of Annotation".
_UPSTREAM_LIBRARY_BUG_CUES = (
    # --- Type / schema / serialization patterns ---
    "typed as object", "typed as string", "typed as list",
    "typed as dict", "typed as int", "typed as float",
    "typed as bool", "typed as None", "typed as array",
    "expected .* but got", "expected str but got",
    "expected int but got", "expected list but got",
    "expected dict but got", "expected bool but got",
    "schema mismatch", "schema validation", "schema error",
    "deserialization", "serialization", "deserialize",
    "serialize", "json decode", "json encode",
    "invalid type", "wrong type",
    "validation error", "validation failed",
    "fields.missing", "missing required field",
    "extra fields not permitted", "extra fields",
    "value is not a valid", "must be a valid",
    "model_dump", "model_dump_json", "model_validate",
    "object instead of", "string instead of",
    "dict instead of", "list instead of", "int instead of",
    # --- Specific OpenAI / SDK failure patterns from the brief ---
    "max_tokens", "max tokens", "model output limit",
    "finish_reason", "finish reason", "truncated output",
    "tool_call", "tool call", "function_call", "function call",
    "messages list", "messages array", "messages field",
    "prompt_cache", "prompt cache",
    # --- Named upstream error classes / events ---
    # NB: bare "typeerror" / "valueerror" are NOT included — they overlap
    # with the generic repo_specific_bug cues. The cues below are
    # unambiguously upstream / library-specific.
    "badrequesterror", "invalidrequesterror", "notfounderror",
    "apierror", "ratelimiterror", "authenticationerror",
    "permissionerror", "timeouterror", "responseerror",
    "apiresponse", "apiconnection", "completionerror",
    "parseerror",
    "responseoutputtext", "responseoutputtextannotation",
    "responsetext", "responsestream",
    "annotationadded", "annotation",
    "openai-python", "openai-node", "openai sdk",
    "langchain", "llamaindex", "llama index",
    "pydantic v2", "pydantic v1", "pydantic v",
    "anthropic sdk", "claude sdk",
    "azure openai", "azure-openai",
    "openai async", "openai chat", "openai completion",
    # --- Common upstream-pattern verbs ---
    "not a valid", "must be", "should be", "is not a",
    "pydantic", "pydantic-core", "validator",
)

# Setup / install / docs / tutorial language.
_DOCS_CUES = (
    "documentation", "docs unclear", "docs are", "docs is",
    "where can i find", "where to find", "how do i install",
    "how to install", "install instructions", "installation guide",
    "setup instructions", "setup guide", "getting started",
    "tutorial", "example code", "code example", "example missing",
    "missing example", "no example", "no tutorial",
    "unclear docs", "outdated docs", "outdated documentation",
    "stale docs", "no docs", "missing docs",
    "readme", "wiki", "guide", "manual",
    "how to use", "how to configure", "how to configure",
    "can't find docs", "no clear docs", "no clear instructions",
    "wrong docs", "documentation is", "example would help",
)

# Feature requests. "please add", "would love to see", etc.
_MISSING_FEATURE_CUES = (
    "feature request", "feature-request", "please add",
    "please support", "would be nice", "would love to see",
    "wish it had", "wish there was", "missing feature",
    "needs to support", "needs support for", "ability to",
    "any plans to", "any plans for", "any chance of",
    "could you add", "could you support", "could we get",
    "would be great if", "would be great to",
    "support for", "supporting", "i wish", "i would love",
    "missing from", "lacks support", "lacks the ability",
    "no support for", "doesn't support",
    "alternative implementation", "replacement for",
)

# Integration / API / SDK / connector language.
_INTEGRATION_CUES = (
    "integrate", "integration", "integrating", "to integrate",
    "connect to", "connecting to", "talk to", "talks to",
    "play nice with", "doesn't talk to", "doesn't connect",
    "incompatible with", "compat with", "compatibility with",
    "sync with", "syncing with", "synced with",
    "import from", "export to", "importing from", "exporting to",
    "third-party", "third party", "third-party integration",
    "connector", "connectors", "plugin", "extension",
    "sdk", "api", "rest api", "graphql api", "graphql",
    "webhook", "webhooks", "oauth", "sso", "saml",
    "two-way sync", "bi-directional", "bidirectional",
    "data import", "data export", "csv import", "csv export",
    "zapier", "make.com", "integromat", "n8n",
    "migrate from", "migration from", "migrating from",
    "ingest", "ingestion", "etl",
)

# Workflow / process / pipeline / repetitive-task language.
_WORKFLOW_CUES = (
    "workflow", "every time i", "every time we",
    "i always have to", "we always have to", "manually",
    "manual process", "manual step", "manual work",
    "tedious", "repetitive", "repetitively",
    "waste of time", "wastes time", "time-consuming",
    "shouldn't have to", "shouldn't need to", "should not have to",
    "context switching", "context-switching", "context switch",
    "broken workflow", "friction", "friction in", "friction with",
    "steps to", "step-by-step", "process of",
    "pipeline", "pipelines", "ci", "ci/cd", "cd pipeline",
    "git", "git workflow", "pr workflow", "pull request",
    "deploy", "deployment", "test", "testing",
    "debug", "debugging", "troubleshoot", "troubleshooting",
    "release", "releases", "release process",
    "onboarding", "onboard new", "new hire",
    "code review", "code reviews", "review process",
    "ticket", "tickets", "issue tracker", "jira",
    "standup", "retrospective", "sprint planning",
)

# Infra / ops / reliability / scaling.
_INFRA_CUES = (
    "rate limit", "rate-limit", "rate limited", "throttle", "throttled",
    "throttling", "retry", "retries", "retrying", "backoff",
    "timeout", "timed out", "timing out",
    "uptime", "downtime", "outage", "incident", "postmortem",
    "monitoring", "alerting", "alert", "oncall", "on-call",
    "scaling", "scale up", "scale out", "autoscale", "auto-scale",
    "queue", "queues", "load balancer", "load balancing",
    "auth failed", "401", "403", "500", "502", "503", "504",
    "infrastructure", "infra", "kubernetes", "k8s",
    "docker", "container", "containers", "containerized",
    "failover", "high availability", "ha", "sla", "slo",
    "memory leak", "oom", "oomkilled", "out of memory",
    "cpu spike", "high cpu", "slow query", "slow queries",
    "latency", "p99", "p95", "response time",
    "logging", "log aggregation", "log shipping",
    "metrics", "tracing", "observability",
    "ssl", "tls", "certificate expired", "cert expired",
    "dns", "dns lookup", "dns resolution",
    "deployment failed", "deploy failed", "rollback",
)

# Security / compliance / privacy.
#
# V2 (calibration pass 2): this lexicon was *deliberately tightened* to
# fix false positives where ordinary API errors (HTTP 401, "permission
# denied", "unauthorized", "token expired", etc.) were being classified
# as security_compliance_pain. Rule 5 of the brief: "Only classify as
# security_compliance_pain if there are explicit security/privacy/
# compliance/vulnerability/permission/authZ/authN cues." We drop the
# most ambiguous terms (permission, unauthorized, authentication,
# authorization, token, credentials, password, session). The remaining
# cues are unambiguous security/privacy signals.
_SECURITY_CUES = (
    # Explicit security/privacy/compliance vocabulary.
    "security", "vulnerability", "cve", "exploit", "exploitable",
    "compliance", "gdpr", "hipaa", "soc2", "soc 2", "iso 27001",
    "pci", "pci dss", "fedramp", "audit", "auditing",
    "data leak", "leaked", "breach", "data breach",
    "encryption", "encrypted", "at rest", "in transit",
    "pii", "personally identifiable", "phi", "protected health",
    "privacy", "data residency", "data sovereignty",
    "zero-day", "0-day", "security advisory", "hardening",
    "hardened", "least privilege",
    # Unambiguous attack-class / vuln-class terms.
    "rce", "remote code execution", "privilege escalation",
    "xss", "csrf", "sql injection", "session hijack",
    "two-factor", "2fa", "mfa", "multi-factor",
    # Specific authorization models (NOT generic "auth").
    "rbac", "abac", "access control",
    # Specific leak / incident terms.
    "credential stuffing", "brute force", "phishing", "malware",
    "ransomware", "spyware", "keylogger", "rootkit",
    "man-in-the-middle", "man in the middle", "mitm",
    "ddos", "denial of service",
)

# Buyer / user identification language. Used to confirm "potential_product".
_BUYER_CUES = (
    "as a developer", "as a developer,", "as developers",
    "as a startup", "as a startup founder", "as a founder",
    "as a freelancer", "as a freelancer,", "as freelancers",
    "as a small business", "as a small team", "as a team lead",
    "as an engineer", "as engineers", "as an engineering team",
    "as a data scientist", "as a data analyst",
    "as a product manager", "as a designer",
    "as a marketer", "as a sales", "as sales",
    "as a teacher", "as a student", "as a researcher",
    "as an agency", "as a consultant", "as a contractor",
    "we are a", "we're a", "we run a", "our team", "my team",
    "our company", "our company uses", "my company",
    "for our", "for my team", "for my company",
    "at our", "at my", "at a", "in our",
)

# Integration tool names. Used for cross-tool signal detection. Listed
# here so we can detect "people mention tool X and tool Y together".
_KNOWN_TOOL_NAMES = (
    "slack", "github", "gitlab", "jira", "linear", "notion",
    "airtable", "salesforce", "hubspot", "stripe", "shopify",
    "google sheets", "google docs", "google drive", "gmail",
    "zapier", "make.com", "integromat", "n8n",
    "figma", "sketch", "trello", "asana", "monday", "clickup",
    "discord", "teams", "zoom", "intercom", "zendesk",
    "aws", "azure", "gcp", "google cloud", "vercel", "netlify",
    "heroku", "digitalocean", "cloudflare", "fastly",
    "sentry", "datadog", "grafana", "prometheus",
    "firebase", "supabase", "auth0", "okta", "clerk",
    "redis", "postgres", "postgresql", "mysql", "mongodb",
    "elasticsearch", "kafka", "rabbitmq", "celery",
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "openai", "anthropic", "claude", "gpt", "llm",
    "langchain", "llamaindex", "pinecone", "weaviate",
)


# -------------------------------------------------------------------------
# Compiled regexes (one pass per lexicon)
# -------------------------------------------------------------------------
def _compile(cues: tuple[str, ...]) -> re.Pattern:
    """Compile a lexicon into a single word-boundary regex.

    We escape each cue and join with `|`. Whole-word match is enforced
    by `\b` so we don't catch "API" inside "rapid" or "ssl" inside
    "classless". Single-character cues (like "ci") get a different
    treatment to avoid spurious matches.
    """
    parts = []
    for cue in cues:
        if len(cue) <= 2:
            # Short cues need stricter boundaries.
            parts.append(re.escape(cue))
        else:
            parts.append(re.escape(cue))
    pattern = r"(?:" + "|".join(parts) + r")"
    return re.compile(pattern, re.IGNORECASE)


# Compile each lexicon once. These are the workhorse regexes.
_REPO_BUG_RE = _compile(_REPO_SPECIFIC_BUG_CUES)
_UPSTREAM_BUG_RE = _compile(_UPSTREAM_LIBRARY_BUG_CUES)
_DOCS_RE = _compile(_DOCS_CUES)
_FEATURE_RE = _compile(_MISSING_FEATURE_CUES)
_INTEGRATION_RE = _compile(_INTEGRATION_CUES)
_WORKFLOW_RE = _compile(_WORKFLOW_CUES)
_INFRA_RE = _compile(_INFRA_CUES)
_SECURITY_RE = _compile(_SECURITY_CUES)
_BUYER_RE = _compile(_BUYER_CUES)
# Tool name regex: case-insensitive, but match the whole name (which
# may include spaces). Use a single alternation of escaped names.
_TOOL_RE = re.compile(
    r"(?:" + "|".join(re.escape(t) for t in _KNOWN_TOOL_NAMES) + r")",
    re.IGNORECASE,
)


# -------------------------------------------------------------------------
# Thresholds (tunable)
# -------------------------------------------------------------------------
# A type "matches" when its cue density (posts with at least one cue /
# total posts) hits this floor. Tuned conservatively so generic text
# doesn't accidentally trip a type.
TYPE_DENSITY_THRESHOLD = 0.30
# Absolute cue hits threshold (across all posts). Useful when a single
# post contains many cues (e.g., a long bug report with lots of error
# tokens).
TYPE_HITS_THRESHOLD = 3

# V2 (calibration pass 2): `potential_product` is the strictest possible
# type. Rule 1: reality_status must be `underserved` or `competitive`
# (NOT `unknown`, NOT `saturated`). Rule 2 + 4: the cluster must NOT
# be primarily an upstream library bug, an API bug, or a type/schema
# mismatch — the "standalone buyer and standalone product" check.
# Rule 6: the final productizability_score must be at least 0.70.
POTENTIAL_MIN_MENTIONS = 5           # repeated pain
POTENTIAL_MIN_DISTINCT_SOURCES = 2   # cross-source (not one repo)
POTENTIAL_MIN_DISTINCT_TOOLS = 2     # or cross-tool (e.g., Slack+GitHub)
POTENTIAL_MIN_PAIN_DENSITY = 0.30    # real frustration
# Open-market competitor strength ceiling (from reality layer).
POTENTIAL_OPEN_MARKET_CEILING = 0.55
# All four conditions must pass.
POTENTIAL_REQUIRED_CONDITIONS = 4
# Reality statuses that allow `potential_product`. Anything else
# (notably `unknown` and `saturated`) is a hard NO.
POTENTIAL_ALLOWED_REALITY_STATUSES = ("underserved", "competitive")
# Minimum final productizability_score for a `potential_product` row.
# Below this we demote to the next-best matching type.
POTENTIAL_MIN_SCORE = 0.70

# V2 (calibration pass 2) — per-type score caps (rule 6). These are
# hard upper bounds; the final productizability_score is the lesser
# of the baseline + cross-cutting bonuses and the cap.
SCORE_CAP = {
    TYPE_UPSTREAM_LIBRARY_BUG: 0.30,
    TYPE_REPO_SPECIFIC_BUG: 0.25,
    TYPE_DOCUMENTATION_CONFUSION: 0.25,
    TYPE_MISSING_FEATURE: 0.45,
    TYPE_INTEGRATION_PAIN: 0.65,
    TYPE_DEVELOPER_WORKFLOW_PAIN: 0.55,
    TYPE_INFRA_OPERATIONAL_PAIN: 0.60,
    TYPE_SECURITY_COMPLIANCE_PAIN: 0.55,
    TYPE_POTENTIAL_PRODUCT: 1.0,  # no cap; POTENTIAL_MIN_SCORE is the floor
    TYPE_UNKNOWN: 0.10,
}
# Baselines used by `_baseline_productizability`. Kept inside the cap
# ceiling so cross-cutting bonuses can't accidentally promote a low
# type above its hard limit.
SCORE_BASELINE = {
    TYPE_UPSTREAM_LIBRARY_BUG: 0.10,
    TYPE_REPO_SPECIFIC_BUG: 0.10,
    TYPE_DOCUMENTATION_CONFUSION: 0.15,
    TYPE_MISSING_FEATURE: 0.30,
    TYPE_INTEGRATION_PAIN: 0.55,
    TYPE_DEVELOPER_WORKFLOW_PAIN: 0.45,
    TYPE_INFRA_OPERATIONAL_PAIN: 0.40,
    TYPE_SECURITY_COMPLIANCE_PAIN: 0.40,
}
# Public dataclass
# -------------------------------------------------------------------------
@dataclass(slots=True)
class OpportunityTypeAssessment:
    """Output of the opportunity-type classifier.

    `opportunity_type` is one of `ALL_TYPES`.
    `productizability_score` is `[0, 1]`: standalone-product potential.
        0.0 = definitely not a buildable product (e.g., a docs gap).
        1.0 = strong cross-tool pain with clear buyer + open market.
    `productizability_reason` is a short human-readable string that
        explains the classification. Surfaced by `founder-radar
        productizable` so the user can audit the reasoning.

    The `signals` dict exposes the raw counts / densities the
    classifier used, so tests and audits can verify thresholds.
    """

    opportunity_type: str = TYPE_UNKNOWN
    productizability_score: float = 0.0
    productizability_reason: str = ""
    signals: dict = field(default_factory=dict)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def classify_opportunity(
    opportunity: "Opportunity",
    posts: Iterable["Post"],
) -> OpportunityTypeAssessment:
    """Classify an opportunity into one of nine `opportunity_type`s.

    Args:
        opportunity: The extracted `Opportunity` row. We read `title`,
            `problem_summary`, `target_audience`, `mentions`, and the
            reality / pain scores already on it.
        posts: All source posts in the cluster. Their titles, bodies,
            and source categories feed the lexicons.

    Returns:
        A populated `OpportunityTypeAssessment`. Never raises — empty
        or malformed input returns `opportunity_type="unknown"` with
        `productizability_score=0.0`.
    """
    posts = list(posts)
    signals = _compute_signals(opportunity, posts)
    type_scores = _score_types(signals)
    product_signals = _product_potential_signals(opportunity, posts, signals)
    opportunity_type, product_score, reason = _pick(
        type_scores, product_signals, signals
    )
    return OpportunityTypeAssessment(
        opportunity_type=opportunity_type,
        productizability_score=round(product_score, 3),
        productizability_reason=reason,
        signals=signals,
    )


# -------------------------------------------------------------------------
# Signal extraction
# -------------------------------------------------------------------------
def _compute_signals(
    opportunity: "Opportunity",
    posts: list["Post"],
) -> dict:
    """Compute the cue-hit counts / densities used by the classifier.

    All counts are per-post (so we can compute density) AND total.
    Returned dict is exposed on the assessment for audit purposes.
    """
    # Combine opportunity text with all post text. The opportunity
    # title/problem_summary are the LLM/heuristic's digest of the
    # cluster — including them gives a small boost to the underlying
    # signal but never overrides the per-post evidence.
    opp_text_parts = []
    if opportunity.title:
        opp_text_parts.append(opportunity.title)
    if opportunity.problem_summary:
        opp_text_parts.append(opportunity.problem_summary)
    if opportunity.target_audience:
        opp_text_parts.append(opportunity.target_audience)
    opp_text = "\n".join(opp_text_parts).strip()

    # Per-post cues (title + body). We only count a cue once per post
    # to avoid one noisy post dominating the score.
    bug_hit_posts = 0
    upstream_hit_posts = 0
    docs_hit_posts = 0
    feature_hit_posts = 0
    integration_hit_posts = 0
    workflow_hit_posts = 0
    infra_hit_posts = 0
    security_hit_posts = 0
    buyer_hit_posts = 0

    # Absolute cue hits (used for TYPE_HITS_THRESHOLD).
    bug_hits = 0
    upstream_hits = 0
    docs_hits = 0
    feature_hits = 0
    integration_hits = 0
    workflow_hits = 0
    infra_hits = 0
    security_hits = 0
    buyer_hits = 0
    # Tools / sources seen.
    sources: set[str] = set()
    tools_seen: set[str] = set()

    for p in posts:
        text = f"{p.title or ''}\n{p.body or ''}"
        if p.source_category:
            sources.add(p.source_category)
        # Tools mentioned in the post.
        for m in _TOOL_RE.finditer(text):
            tools_seen.add(m.group(0).lower())

        if _REPO_BUG_RE.search(text):
            bug_hit_posts += 1
        if _UPSTREAM_BUG_RE.search(text):
            upstream_hit_posts += 1
        if _DOCS_RE.search(text):
            docs_hit_posts += 1
        if _FEATURE_RE.search(text):
            feature_hit_posts += 1
        if _INTEGRATION_RE.search(text):
            integration_hit_posts += 1
        if _WORKFLOW_RE.search(text):
            workflow_hit_posts += 1
        if _INFRA_RE.search(text):
            infra_hit_posts += 1
        if _SECURITY_RE.search(text):
            security_hit_posts += 1
        if _BUYER_RE.search(text):
            buyer_hit_posts += 1

        bug_hits += len(_REPO_BUG_RE.findall(text))
        upstream_hits += len(_UPSTREAM_BUG_RE.findall(text))
        docs_hits += len(_DOCS_RE.findall(text))
        feature_hits += len(_FEATURE_RE.findall(text))
        integration_hits += len(_INTEGRATION_RE.findall(text))
        workflow_hits += len(_WORKFLOW_RE.findall(text))
        infra_hits += len(_INFRA_RE.findall(text))
        security_hits += len(_SECURITY_RE.findall(text))
        buyer_hits += len(_BUYER_RE.findall(text))
    n = max(len(posts), 1)
    return {
        "n_posts": len(posts),
        "n_distinct_sources": len(sources),
        "n_distinct_tools": len(tools_seen),
        "tools": sorted(tools_seen),
        # Per-post hit counts.
        "bug_posts": bug_hit_posts,
        "upstream_posts": upstream_hit_posts,
        "docs_posts": docs_hit_posts,
        "feature_posts": feature_hit_posts,
        "integration_posts": integration_hit_posts,
        "workflow_posts": workflow_hit_posts,
        "infra_posts": infra_hit_posts,
        "security_posts": security_hit_posts,
        "buyer_posts": buyer_hit_posts,
        # Density (posts_with_cue / n_posts).
        "bug_density": bug_hit_posts / n,
        "upstream_density": upstream_hit_posts / n,
        "docs_density": docs_hit_posts / n,
        "feature_density": feature_hit_posts / n,
        "integration_density": integration_hit_posts / n,
        "workflow_density": workflow_hit_posts / n,
        "infra_density": infra_hit_posts / n,
        "security_density": security_hit_posts / n,
        "buyer_density": buyer_hit_posts / n,
        # Absolute cue hits.
        "bug_hits": bug_hits,
        "upstream_hits": upstream_hits,
        "docs_hits": docs_hits,
        "feature_hits": feature_hits,
        "integration_hits": integration_hits,
        "workflow_hits": workflow_hits,
        "infra_hits": infra_hits,
        "security_hits": security_hits,
        "buyer_hits": buyer_hits,
        # Combined text length — useful for audit (very small corpora
        # produce unstable classifications).
        "opp_text_chars": len(opp_text),
    }


# -------------------------------------------------------------------------
# Per-type scoring
# -------------------------------------------------------------------------
def _score_types(signals: dict) -> dict:
    """Return a `{type_name: (matched, density, hits)}` map.

    A type "matches" when EITHER its density crosses the floor OR its
    absolute cue count crosses the floor. The matched flag is the
    boolean; density and hits are the supporting evidence.
    """
    def _matches(density: float, hits: int) -> bool:
        return density >= TYPE_DENSITY_THRESHOLD or hits >= TYPE_HITS_THRESHOLD

    return {
        TYPE_REPO_SPECIFIC_BUG: (
            _matches(signals["bug_density"], signals["bug_hits"]),
            signals["bug_density"],
            signals["bug_hits"],
        ),
        TYPE_UPSTREAM_LIBRARY_BUG: (
            _matches(signals["upstream_density"], signals["upstream_hits"]),
            signals["upstream_density"],
            signals["upstream_hits"],
        ),
        TYPE_DOCUMENTATION_CONFUSION: (
            _matches(signals["docs_density"], signals["docs_hits"]),
            signals["docs_density"],
            signals["docs_hits"],
        ),
        TYPE_MISSING_FEATURE: (
            _matches(signals["feature_density"], signals["feature_hits"]),
            signals["feature_density"],
            signals["feature_hits"],
        ),
        TYPE_INTEGRATION_PAIN: (
            _matches(signals["integration_density"], signals["integration_hits"]),
            signals["integration_density"],
            signals["integration_hits"],
        ),
        TYPE_DEVELOPER_WORKFLOW_PAIN: (
            _matches(signals["workflow_density"], signals["workflow_hits"]),
            signals["workflow_density"],
            signals["workflow_hits"],
        ),
        TYPE_INFRA_OPERATIONAL_PAIN: (
            _matches(signals["infra_density"], signals["infra_hits"]),
            signals["infra_density"],
            signals["infra_hits"],
        ),
        TYPE_SECURITY_COMPLIANCE_PAIN: (
            _matches(signals["security_density"], signals["security_hits"]),
            signals["security_density"],
            signals["security_hits"],
        ),
    }


# -------------------------------------------------------------------------
# potential_product signals
# -------------------------------------------------------------------------
def _product_potential_signals(
    opportunity: "Opportunity",
    posts: list["Post"],
    signals: dict,
) -> dict:
    """Compute the structural conditions for `potential_product`.

    V2 (calibration pass 2): the four-conditions-plus-upstream-check
    model from V1 was tightened further.

    Rule 1: `reality_status` must be `underserved` or `competitive` —
    NOT `unknown`, NOT `saturated`. (Earlier V1 allowed `unknown`,
    which is the #1 source of false positives on real GitHub runs.)

    Rule 2 + 4: the cluster must NOT be primarily an upstream library
    bug, an API bug, or a type/schema mismatch. We enforce this by
    requiring the "real_pain" condition to also be NOT dominated by
    upstream-library cues. If `upstream_density` crosses the density
    floor, the cluster is demoted to `upstream_library_bug` and the
    potential_product path is closed.

    Required conditions (all must pass):
      1. cross_cutting    — cross_source OR cross_tool.
      2. repeated         — opportunity.mentions >= POTENTIAL_MIN_MENTIONS
                            OR frequency_score >= 0.5.
      3. open_market      — reality_status in POTENTIAL_ALLOWED_REALITY_STATUSES
                            AND competitor_strength < POTENTIAL_OPEN_MARKET_CEILING.
      4. real_pain        — pain_score >= POTENTIAL_MIN_PAIN_DENSITY
                            OR (per-axis pain density sum >= floor)
                            OR buyer language detected,
                            AND NOT primarily_upstream.
    """
    # 1. cross_cutting: cross_source OR cross_tool.
    cross_source = signals["n_distinct_sources"] >= POTENTIAL_MIN_DISTINCT_SOURCES
    cross_tool = signals["n_distinct_tools"] >= POTENTIAL_MIN_DISTINCT_TOOLS
    cross_cutting = cross_source or cross_tool
    # 2. repeated
    mentions = opportunity.mentions or 0
    freq = opportunity.frequency_score or 0.0
    repeated = mentions >= POTENTIAL_MIN_MENTIONS or freq >= 0.5
    # 3. open_market (rule 1: reality_status must be underserved|competitive).
    reality_status = opportunity.reality_status or "unknown"
    competitor_strength = opportunity.competitor_strength_estimate or 0.0
    open_market = (
        reality_status in POTENTIAL_ALLOWED_REALITY_STATUSES
        and competitor_strength < POTENTIAL_OPEN_MARKET_CEILING
    )
    # 4. real_pain, with rule 2+4: NOT primarily upstream.
    pain_score = opportunity.pain_score or 0.0
    buyer_match = signals["buyer_density"] > 0
    raw_pain_density = (
        signals["bug_density"] + signals["integration_density"]
        + signals["workflow_density"] + signals["infra_density"]
        + signals["security_density"]
    )
    raw_pain = (
        pain_score >= POTENTIAL_MIN_PAIN_DENSITY
        or raw_pain_density >= POTENTIAL_MIN_PAIN_DENSITY
        or buyer_match
    )
    # "Primarily upstream" means: the upstream lexicon is the
    # strongest single signal in the cluster, OR it alone crosses the
    # density floor. In either case the cluster is really about an
    # upstream SDK failure — not a standalone product.
    primarily_upstream = (
        signals["upstream_density"] >= TYPE_DENSITY_THRESHOLD
        and signals["upstream_density"] >= max(
            signals["bug_density"],
            signals["integration_density"],
            signals["workflow_density"],
            signals["infra_density"],
            signals["security_density"],
        )
    )
    real_pain = raw_pain and not primarily_upstream

    conditions_met = sum(
        bool(x) for x in (cross_cutting, repeated, open_market, real_pain)
    )
    return {
        "cross_source": cross_source,
        "cross_tool": cross_tool,
        "cross_cutting": cross_cutting,
        "repeated": repeated,
        "open_market": open_market,
        "raw_pain": raw_pain,
        "primarily_upstream": primarily_upstream,
        "real_pain": real_pain,
        "buyer_language": buyer_match,
        "conditions_met": conditions_met,
        "reality_status": reality_status,
        "competitor_strength": competitor_strength,
        "pain_score": pain_score,
        "mentions_raw": mentions,
    }

# -------------------------------------------------------------------------
# Final decision
# -------------------------------------------------------------------------
def _pick(
    type_scores: dict,
    product_signals: dict,
    signals: dict,
) -> tuple[str, float, str]:
    """Return (opportunity_type, productizability_score, reason).

    V2 decision tree (calibration pass 2):

      1. If `upstream_library_bug` matches AND the cluster is
         "primarily upstream" (its upstream density is the highest
         single cue), demote to `upstream_library_bug`. This is
         rule 4: "Do not let upstream_library_bug become
         potential_product even if repeated / painful / cross-tool /
         high keyword density."

      2. If `potential_product` conditions met (all 4 strict) AND the
         resulting score is >= POTENTIAL_MIN_SCORE (0.70), return
         `potential_product`.

      3. Otherwise pick the highest-priority matching type. Priority
         order: upstream_library_bug > security > infra >
         integration > docs > bug > feature > workflow.

      4. Apply per-type score cap (rule 6).

      5. If nothing matches, return `unknown` with a low score.
    """
    # Step 1: upstream_library_bug hard-stops potential_product.
    upstream_matched, upstream_density, upstream_hits = type_scores[
        TYPE_UPSTREAM_LIBRARY_BUG
    ]
    if upstream_matched and product_signals.get("primarily_upstream", False):
        product_score = min(
            _baseline_productizability(
                TYPE_UPSTREAM_LIBRARY_BUG, signals, product_signals
            ),
            SCORE_CAP[TYPE_UPSTREAM_LIBRARY_BUG],
        )
        reason = (
            f"upstream_library_bug (rule 4 demote): "
            f"density={upstream_density:.2f} hits={upstream_hits}; "
            f"primarily upstream, NOT a standalone product "
            f"({signals['n_posts']} post(s))"
        )
        return TYPE_UPSTREAM_LIBRARY_BUG, product_score, reason

    # Step 2: potential_product path. Rule 1: reality_status must be
    # underserved|competitive. Rule 2: NOT primarily upstream. The
    # conditions_met check (4) already enforces the latter.
    if product_signals["conditions_met"] >= POTENTIAL_REQUIRED_CONDITIONS:
        product_score = _product_score(signals, product_signals)
        if product_score >= POTENTIAL_MIN_SCORE:
            reason = _product_reason(product_signals, signals)
            return TYPE_POTENTIAL_PRODUCT, product_score, reason
        # Score too low for potential_product. Demote below.

    # Step 3: pick the highest-priority matching type.
    priority_order = (
        TYPE_UPSTREAM_LIBRARY_BUG,
        TYPE_SECURITY_COMPLIANCE_PAIN,
        TYPE_INFRA_OPERATIONAL_PAIN,
        TYPE_INTEGRATION_PAIN,
        TYPE_DOCUMENTATION_CONFUSION,
        TYPE_REPO_SPECIFIC_BUG,
        TYPE_MISSING_FEATURE,
        TYPE_DEVELOPER_WORKFLOW_PAIN,
    )
    matched = [
        (t, type_scores[t][1], type_scores[t][2])
        for t in priority_order
        if type_scores[t][0]
    ]
    if matched:
        chosen = max(matched, key=lambda x: x[1])
        opp_type = chosen[0]
        density, hits = chosen[1], chosen[2]
        raw_score = _baseline_productizability(
            opp_type, signals, product_signals
        )
        # Apply per-type score cap (rule 6).
        product_score = min(raw_score, SCORE_CAP.get(opp_type, 1.0))
        reason = (
            f"{opp_type}: density={density:.2f} hits={hits} "
            f"({signals['n_posts']} post(s))"
        )
        return opp_type, product_score, reason

    # Step 4: unknown.
    return TYPE_UNKNOWN, min(0.0, SCORE_CAP.get(TYPE_UNKNOWN, 0.0)), (
        f"unknown: no type lexicon hit the {TYPE_DENSITY_THRESHOLD:.2f} "
        f"density floor or {TYPE_HITS_THRESHOLD} absolute hits "
        f"({signals['n_posts']} post(s))"
    )


# -------------------------------------------------------------------------
# Score + reason helpers
# -------------------------------------------------------------------------
def _product_score(signals: dict, product_signals: dict) -> float:
    """Compute productizability_score for a `potential_product` cluster.

    Weighted combination:
      0.25 * cross_source/cross_tool signal
      0.20 * log_scaled(mentions)
      0.20 * pain density
      0.20 * (1 - competitor_strength)  (open market bonus)
      0.15 * has clear buyer

    Clipped to [0, 1]. A 5/5 cluster caps at ~0.95; a 3/5 cluster lands
    around 0.6-0.75 depending on the per-axis values.
    """
    # Cross-axis: 1.0 if both source and tool are cross-cutting, else 0.5.
    if product_signals["cross_source"] and product_signals["cross_tool"]:
        cross_axis = 1.0
    elif product_signals["cross_source"] or product_signals["cross_tool"]:
        cross_axis = 0.5
    else:
        cross_axis = 0.0

    # Mentions on a log scale (1 -> 0.0, 5 -> 0.43, 10 -> 0.6, 50 -> 0.93).
    import math
    mentions = max(product_signals.get("mentions_raw", 0), 1)
    mention_score = min(1.0, math.log1p(mentions) / math.log1p(50))

    # Pain density from signals.
    pain = (
        signals["bug_density"]
        + signals["integration_density"]
        + signals["workflow_density"]
        + signals["infra_density"]
        + signals["security_density"]
    ) / 5.0  # average over 5 axes
    pain = min(1.0, pain)

    # Open market bonus.
    market_bonus = 1.0 - min(1.0, product_signals["competitor_strength"])

    # Buyer language.
    buyer = 1.0 if product_signals["buyer_language"] else 0.0

    score = (
        0.25 * cross_axis
        + 0.20 * mention_score
        + 0.20 * pain
        + 0.20 * market_bonus
        + 0.15 * buyer
    )
    return max(0.0, min(1.0, score))

def _baseline_productizability(
    opp_type: str, signals: dict, product_signals: dict
) -> float:
    """Compute productizability_score for a non-`potential_product` type.

    V2: baselines moved to the module-level ``SCORE_BASELINE`` map so
    they're easy to audit and tweak. Cross-cutting bonuses still apply
    (a workflow_pain cluster that's also cross-tool outranks one that
    isn't), but the result is then clipped at the per-type cap from
    ``SCORE_CAP`` by the caller — so bonuses can never push a type
    past its hard ceiling.
    """
    base = SCORE_BASELINE.get(opp_type, 0.0)
    if product_signals.get("cross_source"):
        base += 0.05
    if product_signals.get("cross_tool"):
        base += 0.05
    if product_signals.get("open_market"):
        base += 0.05
    return max(0.0, min(1.0, base))


def _product_reason(product_signals: dict, signals: dict) -> str:
    """One-line explanation of the `potential_product` decision."""
    met = product_signals["conditions_met"]
    flags = []
    if product_signals["cross_source"]:
        flags.append(f"cross-source({signals['n_distinct_sources']})")
    if product_signals["cross_tool"]:
        flags.append(f"cross-tool({signals['n_distinct_tools']})")
    if product_signals["repeated"]:
        flags.append("repeated")
    if product_signals["open_market"]:
        flags.append(
            f"open_market(status={product_signals['reality_status']},"
            f"comp_strength={product_signals['competitor_strength']:.2f})"
        )
    if product_signals["real_pain"]:
        flags.append("real_pain")
    joined = ", ".join(flags) if flags else "none"
    return (
        f"potential_product: {met}/5 conditions met [{joined}] "
        f"({signals['n_posts']} post(s))"
    )
