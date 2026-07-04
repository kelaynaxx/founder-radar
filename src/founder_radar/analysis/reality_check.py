"""Reality Check Layer.

Two related concerns:

  1. Competitor detection: extract named competitors from cluster posts.
     We use two complementary strategies:
       a) Match against a small built-in lexicon of well-known SaaS
          names. This catches the obvious cases without needing NLP.
       b) Extract capitalized noun phrases following competitive cue
          words ("alternative to X", "vs X", "switched from X"). This
          catches niche products the lexicon misses.

  2. Saturation scoring: how crowded is this market? Combines the
     distinct competitor count with mention density into a single
     `[0, 1]` score. The brief says novelty matters less than pain;
     saturation is the "beaten-to-death" warning signal — useful but
     never the primary driver.

This module is *pure*: takes a list of posts, returns a `RealityCheck`
dataclass. No I/O.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from founder_radar.database.models import Post

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Built-in competitor lexicon
# -------------------------------------------------------------------------
# A short list of well-known SaaS / consumer-tech products. We don't try
# to be exhaustive — the regex extractor below picks up niche names.
# Curated by hand from common Reddit mentions in entrepreneur/startup
# subreddits. Kept lowercase for case-insensitive matching.
KNOWN_COMPETITORS: tuple[str, ...] = (
    # CRM / sales
    "salesforce", "hubspot", "pipedrive", "zoho", "freshsales",
    "mailchimp", "convertkit", "activecampaign", "sendinblue",
    "klaviyo", "drip", "mailerlite", "constant contact",
    # Productivity
    "notion", "evernote", "onenote", "obsidian", "roam",
    "todoist", "things", "ticktick", "any.do",
    "trello", "asana", "monday", "clickup", "jira", "linear",
    "airtable", "coda", "google sheets", "google docs",
    # Communication
    "slack", "discord", "teams", "zoom", "meet", "webex",
    "telegram", "whatsapp", "signal", "messenger",
    # Project / design
    "figma", "sketch", "adobe xd", "invision", "canva",
    "framer", "webflow",
    # Developer tools
    "github", "gitlab", "bitbucket", "jira",
    "vscode", "pycharm", "sublime", "vim",
    # Storage / cloud
    "dropbox", "google drive", "onedrive", "box",
    "aws", "azure", "gcp", "heroku", "vercel", "netlify",
    # Analytics
    "google analytics", "mixpanel", "amplitude", "segment",
    "heap", "plausible", "fathom",
    # Payments
    "stripe", "paypal", "square", "braintree", "paddle", "lemonsqueezy",
    # Auth
    "auth0", "okta", "clerk", "supabase", "firebase",
    # E-commerce
    "shopify", "woocommerce", "magento", "bigcommerce", "squarespace",
    "wix", "webflow",
    # Customer support
    "zendesk", "intercom", "freshdesk", "helpscout", "drift",
    # Video
    "youtube", "vimeo", "loom", "wistia",
    # Scheduling
    "calendly", "acuity", "doodle", "scheduleonce",
    # Forms
    "typeform", "google forms", "jotform", "tally",
)


# -------------------------------------------------------------------------
# Regex extractors
# -------------------------------------------------------------------------
# Phrases that put a competitor in context. We capture the word(s)
# immediately after the cue.
_COMPETITIVE_CUE_RE = re.compile(
    r"\b(?:alternative(?:s)?\s+to|instead\s+of|better\s+than|worse\s+than|"
    r"switched?\s+(?:from|to|away\s+from)|migrated?\s+(?:from|to|away\s+from)|"
    r"using\s+instead\s+of|replaced?\s+(?:with|by)|"
    r"vs\.?|versus|compared?\s+(?:to|with))\s+"
    r"(?:[A-Z][\w&.-]*"
    r"(?:\s+[A-Z][\w&.-]*){0,3})",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RealityCheck:
    """Output of the Reality Check Layer.

    All fields are computed deterministically from posts.

    `competitors` is a list of distinct competitor names, sorted by
    mention count descending. `competitor_mention_count` is the total
    number of times *any* competitor was mentioned (a single post can
    mention multiple competitors).

    `saturation_score` is in `[0, 1]`. 0 = wide open (no competitors
    found), 1 = completely saturated (6+ distinct competitors, lots
    of mentions).
    """

    competitors: list[str] = field(default_factory=list)
    distinct_competitor_count: int = 0
    competitor_mention_count: int = 0
    saturation_score: float = 0.0

    @property
    def has_real_competition(self) -> bool:
        """True when at least 2 distinct competitors are mentioned."""
        return self.distinct_competitor_count >= 2

    @property
    def is_saturated(self) -> bool:
        """True when saturation_score is high enough to flag as warning."""
        return self.saturation_score >= 0.7


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def run_reality_check(posts: Iterable["Post"]) -> RealityCheck:
    """Compute the Reality Check for a set of posts.

    Args:
        posts: Iterable of `Post` rows. We read `title` and `body`.

    Returns:
        A `RealityCheck` with distinct competitors and a saturation
        score.
    """
    posts = list(posts)
    if not posts:
        return RealityCheck()

    # (competitor_lower, display_name) -> mention_count
    counts: dict[str, int] = {}
    # Display names as they appeared in text — preserve original casing
    # where possible for nicer display in reports.
    display: dict[str, str] = {}

    for post in posts:
        text = f"{post.title or ''}\n{post.body or ''}"
        for name, display_name in _extract_competitors(text):
            key = name.lower()
            counts[key] = counts.get(key, 0) + 1
            # Keep the first display name we see (preserves original
            # casing for nicer report rendering).
            display.setdefault(key, display_name)

    # Sort by mention count, then alphabetically for determinism.
    sorted_keys = sorted(
        counts.keys(),
        key=lambda k: (-counts[k], k),
    )
    competitors = [display[k] for k in sorted_keys]
    distinct = len(competitors)
    mentions = sum(counts.values())

    # Saturation formula:
    #   - 0 competitors -> 0.0
    #   - 1-2 competitors -> small bump
    #   - 3-5 competitors -> moderate
    #   - 6+ competitors -> saturated
    #   - Mention density adds a bonus (more mentions per post ->
    #     higher density = more crowded).
    if distinct == 0:
        saturation = 0.0
    else:
        # Base saturation from distinct count, capped at 0.85.
        base = min(0.85, distinct / 6.0)
        # Density bonus: mentions per post. 0 mentions/post -> 0,
        # 1+ mentions/post -> +0.15.
        density = min(0.15, mentions / max(len(posts), 1) * 0.15)
        saturation = min(1.0, base + density)

    return RealityCheck(
        competitors=competitors,
        distinct_competitor_count=distinct,
        competitor_mention_count=mentions,
        saturation_score=saturation,
    )


# -------------------------------------------------------------------------
# Extractors (private helpers)
# -------------------------------------------------------------------------
def _extract_competitors(text: str) -> Iterable[tuple[str, str]]:
    """Yield (name_lower, display_name) pairs found in `text`.

    Two strategies combined:
      - Lexicon match against `KNOWN_COMPETITORS`.
      - Regex match against `_COMPETITIVE_CUE_RE`.
    """
    if not text:
        return
    text_lower = text.lower()

    # Strategy A: lexicon.
    for comp in KNOWN_COMPETITORS:
        # Word-boundary match so "note" doesn't match "notion".
        if re.search(r"(?<![\w])" + re.escape(comp) + r"(?![\w])", text_lower):
            yield comp, comp.title() if comp.islower() else comp

    # Strategy B: regex cue phrases.
    for match in _COMPETITIVE_CUE_RE.finditer(text):
        phrase = match.group(0)
        # Strip the leading cue word(s) so we're left with the candidate.
        # We do a cheap split: find the index after the cue verb.
        stripped = _strip_cue(phrase)
        if not stripped:
            continue
        if len(stripped) < 2 or len(stripped) > 40:
            continue
        # Skip if it's just punctuation / stop words.
        if not re.search(r"[A-Za-z]", stripped):
            continue
        # Don't yield lexicon names twice (already covered above).
        if stripped.lower() in {c.lower() for c in KNOWN_COMPETITORS}:
            continue
        # Filter out very common capitalized phrases that aren't
        # products. Heuristic: must start with a capital letter.
        if not stripped[0].isupper():
            continue
        yield stripped.lower(), stripped


_CUE_LEADING_WORDS = (
    "versus",
    "alternative to", "alternatives to",
    "instead of",
    "better than", "worse than",
    "switched from", "switched to", "switched away from",
    "switch from", "switch to", "switch away from",
    "migrated from", "migrated to", "migrated away from",
    "migrate from", "migrate to",
    "using instead of",
    "replaced with", "replaced by",
    "compared to", "compared with",
    "vs",
)


def _strip_cue(phrase: str) -> str:
    """Strip the leading competitive cue from a regex match.

    Tries each known cue prefix (longest first) and returns what comes
    after. Cues are matched case-insensitively so the original casing
    of the candidate is preserved in the returned string.
    """
    text = phrase.lower()
    for cue in sorted(_CUE_LEADING_WORDS, key=len, reverse=True):
        if text.startswith(cue):
            return phrase[len(cue):].strip().rstrip(",.;:")
    # Fallback: drop the first word.
    parts = phrase.split(None, 1)
    if len(parts) < 2:
        return ""
    return parts[1].strip().rstrip(",.;:")
    """Remove the leading competitive cue word(s) from a regex match."""
    # The regex captures the cue + the candidate. We split on the first
    # whitespace after the cue verb and return the rest.
    parts = phrase.split(None, 1)
    if len(parts) < 2:
        return ""
    return parts[1].strip().rstrip(",.;:")