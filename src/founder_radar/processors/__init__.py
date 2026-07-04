"""Processors layer.

A *processor* transforms a list of raw posts into a (smaller, cleaner) list
of posts. Processors compose into a pipeline; each one is independent and
order-independent except where the design demands otherwise.

Phase 1 ships with `Cleaner` (deduplication + spam heuristics). Future
phases will add embedder, clusterer, scorer as additional processor stages.
"""
from founder_radar.processors.base import BaseProcessor
from founder_radar.processors.cleaner import Cleaner

__all__ = ["BaseProcessor", "Cleaner"]