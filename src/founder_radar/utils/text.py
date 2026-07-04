"""Small dependency-free text helpers.

Anything heavier than this (language detection, embeddings, sentiment)
belongs in its own module so we don't bloat imports across the codebase.
"""

from __future__ import annotations

import re
import unicodedata

# Pre-compiled patterns. Compiling once at import time is the standard
# micro-optimization for hot regexes; we also benefit from clearer stack traces.
_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM_DASHES = re.compile(r"-+")


def slugify(text: str, max_length: int = 80) -> str:
    """Convert arbitrary text into a filesystem/URL-safe slug.

    Examples:
        >>> slugify("Hello, World!")
        'hello-world'
        >>> slugify("  Multi   spaces  ")
        'multi-spaces'

    Args:
        text: Input text. Unicode is normalized to ASCII where possible.
        max_length: Hard cap on output length. Slugs longer than this are
            truncated at the last word boundary.

    Returns:
        Lowercase, dash-separated, ASCII-only string. Returns "untitled"
        if the input has no usable characters.
    """
    if not text:
        return "untitled"

    # NFKD decomposes accents (e.g. "é" -> "e" + combining acute), then we
    # drop the combining marks. This keeps "naïve" -> "naive" instead of
    # keeping the special character.
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")

    lowered = ascii_only.lower()
    dashed = _SLUG_NON_ALNUM.sub("-", lowered)
    trimmed = _SLUG_TRIM_DASHES.sub("-", dashed).strip("-")

    if not trimmed:
        return "untitled"

    if len(trimmed) > max_length:
        trimmed = trimmed[:max_length].rsplit("-", 1)[0] or trimmed[:max_length]

    return trimmed


_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)


def extract_first_url(text: str) -> str | None:
    """Return the first http(s) URL found in `text`, or None.

    Used to pull a primary link out of a post body that may contain markdown
    links, bare URLs, or both.
    """
    if not text:
        return None
    match = _URL_PATTERN.search(text)
    return match.group(0) if match else None