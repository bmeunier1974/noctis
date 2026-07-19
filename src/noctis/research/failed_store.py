"""Capped on-disk record of rejected coder attempts — a bad session, inspectable from disk.

Every coder attempt the write gate rejects (a validation error or a non-code reply) is
persisted here by the toolbox, so a failed authoring session is diagnosable from files on disk
instead of terminal scrollback (the #13 epic's motivation: the one bad session that started it
was only readable from scrollback). The store lives under the strategy library's working tier
— ``<__tmp>/failed/`` — the failure twin of the drafts and rejects beside it, and is
**capped**: only the most recent ~50 attempts are kept, oldest evicted, so observability can
never grow unbounded on disk.

One file per attempt carries BOTH halves a human needs — a comment header with the strategy
name, attempt number, UTC timestamp, and the gate error, then the exact attempted source below
— so opening one file shows what the coder wrote and why the gate refused it. Files are named
with a zero-padded monotonic sequence (``000042-<name>-attempt<N>.py``) so insertion order is
total: eviction always drops the lowest sequence, and the next sequence keeps climbing even
after a rollover (a number is never reused). The store is **stateless across calls** — the next
sequence and the eviction set are both re-read from the directory each time — so it holds no
cross-session memory and two stores over the same root stay consistent.

The ``failed/`` folder is a subdirectory of ``__tmp``, and library discovery globs each tier
non-recursively (``__tmp/*.py``), so a persisted attempt never masquerades as a real strategy.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

# Keep roughly the last N rejected attempts on disk, oldest evicted — observability, not history.
DEFAULT_CAP = 50

_SEQ_RE = re.compile(r"^(\d+)-")
_UNSAFE_RE = re.compile(r"[^a-z0-9_]+")


class FailedAttemptStore:
    """A capped folder of rejected coder attempts under the working tier's ``failed/`` area.

    Deep module, one-call surface: :meth:`record` writes one attempt (source + gate error) and
    evicts the oldest over the cap. Construct with the ``failed/`` root and an optional cap.
    """

    def __init__(self, root: Path | str, *, cap: int = DEFAULT_CAP) -> None:
        self._root = Path(root)
        self._cap = max(1, cap)

    @property
    def root(self) -> Path:
        return self._root

    def record(self, name: str, attempt: int, source: str, error: str) -> Path:
        """Persist one rejected attempt; return its file path. Evicts the oldest over the cap."""
        self._root.mkdir(parents=True, exist_ok=True)
        seq = self._next_seq()
        path = self._root / f"{seq:06d}-{self._safe(name)}-attempt{attempt}.py"
        path.write_text(self._render(name, attempt, source, error), encoding="utf-8")
        self._evict()
        return path

    # ── sequence + eviction (both re-read from disk, so the store carries no state) ──
    def _next_seq(self) -> int:
        return max((self._seq_of(p) for p in self._files()), default=0) + 1

    def _evict(self) -> None:
        files = sorted(self._files(), key=self._seq_of)
        for stale in files[: max(0, len(files) - self._cap)]:
            stale.unlink(missing_ok=True)

    def _files(self) -> list[Path]:
        if not self._root.is_dir():
            return []
        return [p for p in self._root.glob("*.py") if _SEQ_RE.match(p.name)]

    @staticmethod
    def _seq_of(path: Path) -> int:
        match = _SEQ_RE.match(path.name)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _safe(name: str) -> str:
        """A filesystem-safe token from a driver-supplied strategy name (never trusted raw)."""
        cleaned = _UNSAFE_RE.sub("-", (name or "").strip().lower()).strip("-")
        return cleaned or "unnamed"

    @staticmethod
    def _render(name: str, attempt: int, source: str, error: str) -> str:
        """One inspectable file: a commented header (name, attempt, time, error) then source."""
        stamp = datetime.now(UTC).isoformat()
        error_lines = (error or "").splitlines() or [""]
        header = [
            f"# rejected coder attempt — strategy {name!r}, attempt {attempt}",
            f"# recorded: {stamp}",
            "# gate error:",
            *[f"#   {line}" for line in error_lines],
            "# --- attempted source below ---",
        ]
        return "\n".join(header) + "\n" + (source or "")
