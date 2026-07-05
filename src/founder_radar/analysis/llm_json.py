"""Robust JSON extraction helper for LLM responses.

V2.1 (MiniMax-M3 reasoning support). This module centralises the
"give me a JSON object out of an LLM response" pipeline that used
to live inline in `opportunity.py` and `opportunity_review.py`. The
two call sites had drifted apart and neither handled reasoning-model
output (`<!-- raw HTML removed -->...<!-- raw HTML removed -->` blocks), so we unified them here.

The extraction pipeline (in order):
  1. Strip leading/trailing thinking-tag blocks (`<!-- raw HTML removed -->`,
     `<thinking>`, `<reasoning>`, case-insensitive, multi-line).
  2. Strip markdown code fences (`\\`\\`json ... \\`\\``).
  3. Find the first balanced `{...}` block in the remaining text.
  4. Try `json.loads()` on each candidate. Return the first dict.

If none of the above works, raise `LLMJsonError` with the first
300 characters of the original content. The caller decides whether
to attempt a retry-repair pass or fail safely.

The `parse_with_repair()` helper handles the retry-repair loop:
on a parse failure, it sends a second prompt to the LLM asking it
to convert its previous output to valid JSON, then re-parses. If
that also fails, the helper raises with the combined error.

Design rules:
  - Pure functions where possible. The retry-repair helper takes
    a callable `repair` so callers can plug in any retry strategy.
  - Never mutate `content` in-place — return a new string.
  - Always include the first 300 chars of the failing content in
    the error so the CLI / logs can surface "this is what the LLM
    actually said".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from founder_radar.llm.base import LLMMessage

logger = logging.getLogger(__name__)


# Length of the original content we keep when reporting a parse
# failure. Short enough to fit in a log line, long enough to give a
# human (or another LLM) something to work with.
ERROR_PREVIEW_CHARS = 300


# Maximum size of the extracted JSON object we'll keep around for
# error reporting. Reasoning models can emit huge traces — we don't
# want to print megabytes when something goes wrong.
MAX_REPORTABLE_DICT_KEYS = 64


class LLMJsonError(ValueError):
    """Raised when the LLM's output cannot be parsed as JSON.

    The string form of the exception is the first
    `ERROR_PREVIEW_CHARS` chars of the original content (or the
    parser-attempted content) for log-friendly diagnostics.
    """

    def __init__(self, message: str, *, preview: str = "") -> None:
        super().__init__(message)
        self.preview = preview


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

# Reasoning-model tag names to strip. Order matters: longest first so
# that "<thinking>" is matched before any shorter alias. The primary
# MiniMax-M3 / DeepSeek R1 / OpenAI o1-style block is `<!-- raw HTML removed -->...<!-- raw HTML removed -->`,
# but other tags appear in the wild and the stripper must cope with all
# of them (and unterminated variants).
_THINKING_TAGS: tuple[str, ...] = ("thinking", "think", "reasoning")

# Keywords that flag an HTML comment block as a reasoning trace.
# Unrelated comments are intentionally left alone; see the gated
# fallback in `_strip_html_comments_if_reasoning`.
_REASONING_KEYWORDS = ("reason", "think", "step", "analysis")

# Pre-compiled at module load so we don't recompile per LLM response.
# Each open pattern matches an opening tag with optional attributes
# (e.g. `<think role="x">`); each close pattern matches the closing tag.
_OPEN_TAG_PATTERNS: dict[str, "re.Pattern[str]"] = {
    tag: re.compile(rf"<{tag}\b[^>]*>", re.IGNORECASE)
    for tag in _THINKING_TAGS
}
_CLOSE_TAG_PATTERNS: dict[str, "re.Pattern[str]"] = {
    tag: re.compile(rf"</{tag}\s*>", re.IGNORECASE)
    for tag in _THINKING_TAGS
}
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_one_tag(content: str, tag: str) -> str:
    """Strip one tag family (e.g. ``<!-- raw HTML removed -->...<!-- raw HTML removed -->``).

    Handles three shapes:
      1. Balanced: `<tag>...</tag>` -- drop the tag pair.
      2. Unterminated opening (`<tag>` + json at EOF, no `</tag>`) --
         drop from the opening tag to end-of-string.
      3. Stray closing (`</tag>` only, no opening) -- drop the stray close.

    Loops because stripping can expose a second occurrence of the same
    tag further down the string.
    """
    open_pat = _OPEN_TAG_PATTERNS[tag]
    close_pat = _CLOSE_TAG_PATTERNS[tag]
    while True:
        open_match = open_pat.search(content)
        if open_match is None:
            # No more opening tags. Strip any stray closing tags.
            if close_pat.search(content):
                content = close_pat.sub("", content)
            return content
        close_match = close_pat.search(content, open_match.end())
        if close_match is None:
            # Unterminated: drop from the opening tag to EOF.
            return content[: open_match.start()]
        # Balanced: drop from opening tag to end of closing tag.
        content = content[: open_match.start()] + content[close_match.end():]


def _strip_html_comments_if_reasoning(content: str) -> str:
    """Defensive fallback: strip HTML comments that look like reasoning.

    Only strips comments whose body contains one of
    `_REASONING_KEYWORDS`. Other HTML comments are preserved verbatim --
    this avoids accidentally eating user-authored `<!-- TODO -->` markers
    or other tool-emitted comments.

    This is NOT the primary MiniMax-M3 case; literal
    `<!-- raw HTML removed -->...<!-- raw HTML removed -->`, `<thinking>`, and `<reasoning>` tags are stripped
    by the primary pass above. This is a defense-in-depth fallback for
    future model variants.
    """
    def _replace(match: "re.Match[str]") -> str:
        body = match.group(0).lower()
        if any(keyword in body for keyword in _REASONING_KEYWORDS):
            return ""
        return match.group(0)
    return _HTML_COMMENT_PATTERN.sub(_replace, content)


def strip_thinking_blocks(content: str) -> str:
    """Remove reasoning-model inline blocks from `content`.

    Strips three tag families (`<!-- raw HTML removed -->`, `<thinking>`,
    `<reasoning>`) and their unterminated variants, all case-insensitive.
    As a defensive fallback, HTML-comment blocks are also stripped when
    their body looks like a reasoning trace (contains one of `reason`,
    `think`, `step`, `analysis`).

    Reasoning models (MiniMax-M3, DeepSeek R1, OpenAI o1, etc.)
    occasionally inline their chain-of-thought into `message.content`
    before the real answer. The block can be multiline; we strip
    greedily. Tolerant: if the closing tag is missing, we drop from
    the opening tag to the end of the content.

    If no thinking-tag signal is present in `content`, it is returned
    unchanged.
    """
    if not content:
        return content
    stripped = content
    for tag in _THINKING_TAGS:
        stripped = _strip_one_tag(stripped, tag)
    stripped = _strip_html_comments_if_reasoning(stripped)
    return stripped.strip()


def strip_markdown_fences(content: str) -> str:
    """Strip surrounding markdown ``` / ```json fences.

    Tolerant: any leading line that starts with ``` (with optional
    `json` after) is treated as an opening fence; the LAST trailing
    line that starts with ``` is treated as the closing fence.
    """
    if not content:
        return content
    text = content.strip()

    # Opening fence: first non-empty line starts with ```
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        # Drop the opening fence line.
        lines = lines[1:]
    # Closing fence: last non-empty line starts with ```
    while lines and lines[-1].lstrip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _balanced_json_object(text: str) -> str | None:
    """Find the first balanced `{...}` substring in `text`.

    Tracks string boundaries (so braces inside JSON strings don't
    confuse the depth counter). Returns the substring or None.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        i = start
        in_string = False
        escape = False
        while i < len(text):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            i += 1
        # No balanced object found starting at this `start`. Try the
        # next `{` after this position.
        start = text.find("{", start + 1)
    return None


def extract_json(content: str) -> dict:
    """Best-effort JSON extraction from an LLM response.

    Pipeline (each step runs only if the previous failed):
      1. Direct `json.loads(content)` (works for clean JSON).
      2. Strip thinking-tag blocks, then direct parse.
      3. Strip markdown fences, then direct parse.
      4. Find the first balanced `{...}` substring, then parse.
      5. Combine strips (1 + 2 + 3), then try balanced extraction.

    On success returns the parsed dict. On failure raises
    `LLMJsonError` with a preview of the original content.

    IMPORTANT: this function never returns None. A parse failure is
    always an exception. Callers that want to swallow it must catch.
    """
    if not content or not content.strip():
        raise LLMJsonError(
            "LLM returned empty content (no message.content).",
            preview="<empty>",
        )

    preview = content[:ERROR_PREVIEW_CHARS]

    # Pipeline 1: direct parse.
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Pipeline 2: strip thinking blocks, try again.
    stripped = strip_thinking_blocks(content)
    if stripped != content:
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Pipeline 3: strip markdown fences, try again.
    fenced = strip_markdown_fences(content)
    if fenced != content:
        try:
            data = json.loads(fenced)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Pipeline 4: extract balanced { ... } block from the content.
    candidate = _balanced_json_object(content)
    if candidate is not None:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Pipeline 5: strip BOTH thinking and fences, then balanced.
    both = strip_markdown_fences(stripped)
    candidate = _balanced_json_object(both)
    if candidate is not None:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    raise LLMJsonError(
        f"Could not extract a JSON object from LLM response "
        f"(first {ERROR_PREVIEW_CHARS} chars shown in `preview`).",
        preview=preview,
    )


def try_extract_json(content: str) -> dict | None:
    """Same as `extract_json` but returns None instead of raising.

    Useful for code paths that have a separate fallback when JSON is
    missing. The brief: NEVER crash the CLI on bad LLM output — this
    function is the safe form of `extract_json`.
    """
    try:
        return extract_json(content)
    except LLMJsonError:
        return None


@dataclass(slots=True)
class RepairResult:
    """Result of a `parse_with_repair` retry-repair pass.

    `value` is the parsed dict on success, None on failure.
    `attempts` is the number of LLM calls made (1 = first call, 2 = one
    repair, etc.).
    `last_error` is the most recent parse error (None on success).
    `last_content` is the most recent raw content the parser saw.
    """

    value: dict | None
    attempts: int
    last_error: str | None
    last_content: str


def parse_with_repair(
    initial_content: str,
    *,
    repair: Callable[[str], str | None],
) -> RepairResult:
    """Parse `initial_content`; on failure, ask `repair` for a fix.

    The `repair` callable takes the previous failed content (or its
    preview) and returns a new content string. Return None from
    `repair` to abort the retry loop (caller gives up).

    Behaviour:
      1. Try to parse `initial_content`.
      2. On failure, call `repair(preview)` once.
      3. If `repair` returned non-None content, parse it.
      4. If parse still fails, return RepairResult(value=None,
         attempts=2, last_error=..., last_content=repaired_content).

    The retry budget is intentionally hard-coded to ONE repair pass.
    Beyond that we accept defeat — we'd rather fail safely than spin.
    """
    try:
        return RepairResult(
            value=extract_json(initial_content),
            attempts=1,
            last_error=None,
            last_content=initial_content,
        )
    except LLMJsonError as first_err:
        first_preview = first_err.preview or initial_content[:ERROR_PREVIEW_CHARS]
        logger.info(
            "First parse failed; asking repair callable to fix "
            "(first %d chars: %r)",
            ERROR_PREVIEW_CHARS, first_preview,
        )
        repaired = repair(first_preview)
        if repaired is None:
            return RepairResult(
                value=None,
                attempts=1,
                last_error=str(first_err),
                last_content=initial_content,
            )
        try:
            return RepairResult(
                value=extract_json(repaired),
                attempts=2,
                last_error=None,
                last_content=repaired,
            )
        except LLMJsonError as second_err:
            return RepairResult(
                value=None,
                attempts=2,
                last_error=str(second_err),
                last_content=repaired,
            )


def make_repair_callback(
    llm_complete: Callable[[list[LLMMessage]], str],
    *,
    schema_hint: str = "",
) -> Callable[[str], str | None]:
    """Build a repair callable for `parse_with_repair`.

    The callback wraps `llm_complete` (a function that takes a list
    of messages and returns content). It sends a repair prompt asking
    the LLM to convert the failed content into valid JSON matching
    `schema_hint`.

    Returns None from `llm_complete` to signal "give up" -- the
    callback will return None and the repair loop aborts.
    """
    def repair(failed_preview: str) -> str | None:
        schema_block = (
            f"The repaired JSON MUST match this schema:\n{schema_hint}"
            if schema_hint.strip()
            else "The repaired JSON should follow the same shape as "
                 "the original prompt asked for."
        )
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a strict JSON repair assistant. "
                    "Given the assistant's previous response, output "
                    "ONLY valid JSON -- no commentary, no markdown "
                    "fences, no thinking-block tags of any kind, no prose."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"The previous assistant response could not be "
                    f"parsed as JSON. Convert it into valid JSON. "
                    f"{schema_block}\n\n"
                    f"Previous response (first {ERROR_PREVIEW_CHARS} "
                    f"chars):\n{failed_preview}"
                ),
            ),
        ]
        try:
            content = llm_complete(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Repair call failed: %s", exc)
            return None
        if content is None:
            return None
        return content

    return repair
