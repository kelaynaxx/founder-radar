"""Abstract base for every report format.

A report consumes a list of stored posts (and optionally opportunities)
and emits a human-readable artifact. Phase 1 ships `MarkdownReport`;
later phases add `HTMLReport` and `JSONReport`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from founder_radar.database.models import Opportunity, Post


class BaseReport(ABC):
    """Base class for all report generators."""

    extension: str = ""  # subclasses set, e.g. ".md"

    @abstractmethod
    def render(
        self,
        posts: list["Post"],
        opportunities: list["Opportunity"] | None = None,
    ) -> str:
        """Return the rendered report as a string.

        `opportunities` is optional; when None, only the posts section
        is rendered. When provided, an opportunities section is appended.
        """

    def write(
        self,
        posts: list["Post"],
        path: Path,
        opportunities: list["Opportunity"] | None = None,
    ) -> Path:
        """Render and write the report to `path`.

        Convenience method that ensures the parent directory exists and
        the file is written as UTF-8. Returns the resolved path on disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.render(posts, opportunities=opportunities),
            encoding="utf-8",
        )
        return path