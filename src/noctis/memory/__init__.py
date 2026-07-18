"""Noctis memory — the agent's own long-term notes.

A small seam (append findings, remember dead ends, close-phase upkeep) with an in-memory
implementation. The human-readable ``MEMORY.md`` store implements the same protocol.
"""

from __future__ import annotations

from noctis.memory.base import InMemoryMemory, Memory
from noctis.memory.store import MemoryStore

__all__ = ["Memory", "InMemoryMemory", "MemoryStore"]
