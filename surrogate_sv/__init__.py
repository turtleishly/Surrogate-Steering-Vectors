"""Surrogate steering vectors package.

Phase 1 module extraction from notebook helpers.
"""

from .config import COLLECTIONS, DEFAULT_MONITOR_SYSTEM_PROMPT
from .paths import resolve_output_root

__all__ = [
    "COLLECTIONS",
    "DEFAULT_MONITOR_SYSTEM_PROMPT",
    "resolve_output_root",
]
