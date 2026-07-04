"""Abstract base for every processor stage.

A processor is a pure data transformation: it takes a list of `RawPost`
and returns a (possibly shorter) list. The pipeline composes processors
left-to-right; each is independent.

Phase 1 ships only `Cleaner`. Phase 2 will add `Embedder`; Phase 3 will
add `Clusterer`. None of them will need to modify `BaseProcessor`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from founder_radar.collectors.base import RawPost


class BaseProcessor(ABC):
    """Pure-data transformation stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, human-readable identifier. Used in logs and reports."""

    @abstractmethod
    def process(self, posts: list["RawPost"]) -> list["RawPost"]:
        """Transform `posts`. Must not mutate the input list.

        Returning a new list makes composition order-independent and
        prevents subtle bugs from shared references.
        """