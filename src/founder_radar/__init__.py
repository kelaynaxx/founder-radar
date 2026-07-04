"""Founder Radar — opportunity discovery from public discussions.

This package is organized as a pipeline of pluggable layers:

    collectors  -> processors -> analysis -> reports
                            \\-> database   <-/

See README.md for the architecture overview and the deepwork notes at
`.slim/deepwork/phase-1.md` for Phase 1 design decisions.
"""

from __future__ import annotations

# Semantic version. Bump on every architectural change.
__version__ = "0.1.0"