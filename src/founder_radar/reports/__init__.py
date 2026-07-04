"""Reports layer.

A *report* consumes the database (or a snapshot of it) and emits a
human-readable artifact: Markdown for now, HTML and JSON in later phases.
Every reporter inherits from `BaseReport` so the CLI can pick one
generically.
"""
from founder_radar.reports.base import BaseReport
from founder_radar.reports.markdown_report import MarkdownReport

__all__ = ["BaseReport", "MarkdownReport"]