"""Capped, content-addressed sidecars — the body behind a hash a ledger row carries (story #180).

Noctis records *counts, not content*: a ledger row says an authoring job ran and how it ended, but
the brief the model was actually given is constructed, passed, and dropped. This store is the other
half — the call site hands a payload here, gets a content hash back, and writes only that hash into
its (small, append-only) row, so the body stays findable on disk without bloating the ledger. The
store is the epic's one capture primitive: adding capture at a new call site should be one line,
not a design exercise.

**Content-addressed, so deduplication is free.** The filename carries the payload's sha256, which
means the same body captured twice — two episodes under identical knobs, one briefing repeated
across a round — is one file, and the hash is a stable name for it across instances, processes and
runs. The digest is over the payload's UTF-8 bytes and nothing else (no timestamp, no kind, no
run id), so a hash written into a ledger today names the same content forever.

**Per-kind subfolders, one global cap.** ``<root>/<kind>/`` keeps call sites apart when a human
goes looking, but the cap bounds the *whole* area: the promise is that ``qa/`` cannot grow without
bound, and one number is the honest way to keep it. The cap defaults generously — captured bodies
are tens of KB and a session emits dozens, not thousands — because it is a safety net, not a
budget. Files are named ``<seq>-<sha256>.txt`` with a zero-padded monotonic sequence, so insertion
order is total (mtimes are not, at the resolution a fast session writes at): eviction always drops
the lowest sequence, and the sequence keeps climbing after a rollover so a number is never reused.
Like :class:`~noctis.research.failed_store.FailedAttemptStore`, the store is **stateless across
calls** — the next sequence and the eviction set are both re-read from disk — so a fresh instance
over an existing tree evicts the tree's oldest, not merely its own.

**The fail-safe latch**, the same posture as the debug recorder and the run store: capture is
strictly secondary, so the *first* write failure is logged **exactly once**, latches the store off,
and returns ``None``; every later call is a silent no-op, and nothing ever raises into a research
session. A latched :meth:`store` returning ``None`` rather than a hash is the honest signal — the
caller omits the field rather than recording a hash whose body was never written.

Sidecars land under the run's gitignored ``qa/`` area, so nothing captured ever reaches git
(AGENTS.md rule 6), and the payloads a caller hands over inherit the run record's secrets
discipline: it is the call site's job never to capture a credential.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

__all__ = ["DEFAULT_CAP", "CaptureStore"]

# The module logger the fail-safe latch warns through, once, when it trips.
logger = logging.getLogger(__name__)

# Keep roughly the last N sidecars across every kind, oldest evicted. Generous on purpose: bodies
# are tens of KB and a session emits dozens, so this bounds disk without ever binding a real run.
DEFAULT_CAP = 500

_SIDECAR_SUFFIX = ".txt"
_SEQ_RE = re.compile(r"^(\d+)-")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_RE = re.compile(r"[^a-z0-9_.-]+")


class CaptureStore:
    """A capped, content-addressed sidecar area — ``store(kind, payload) -> hash``, and read back.

    Deep module, one-call surface: :meth:`store` hashes the payload, writes it under
    ``<root>/<kind>/`` unless that exact content is already there, evicts the oldest over the cap,
    and returns the hash (``None`` if capture is latched off). :meth:`read` retrieves a payload by
    ``(kind, hash)`` for tests and post-mortems. Construct with the capture root — callers pass the
    run's ``qa/`` area — and an optional cap.
    """

    def __init__(self, root: Path | str, *, cap: int = DEFAULT_CAP) -> None:
        self._root = Path(root)
        self._cap = max(1, cap)
        self._disabled = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def disabled(self) -> bool:
        """Whether the fail-safe latch has tripped. Once ``True`` it never clears."""
        return self._disabled

    def store(self, kind: str, payload: str) -> str | None:
        """Capture ``payload`` under ``kind``; return its content hash, or ``None`` if latched off.

        Identical content stored again is the same file and the same hash — no rewrite, and no cap
        room consumed — so a repeated body cannot evict its neighbours. ``None`` means nothing was
        written (the store has latched off after a failure): the caller should omit the hash rather
        than record a name for a body that is not on disk.
        """
        if self._disabled:
            return None
        try:
            digest = _content_hash(payload)
            folder = self._root / _safe_kind(kind)
            if self._find(folder, digest) is not None:
                return digest
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{self._next_seq():06d}-{digest}{_SIDECAR_SUFFIX}").write_text(
                payload, encoding="utf-8"
            )
            self._evict()
            return digest
        except Exception as exc:
            self._trip(exc)
            return None

    def read(self, kind: str, digest: str) -> str | None:
        """The payload stored under ``(kind, digest)``, or ``None`` if it was never stored or has
        since been evicted — a post-mortem helper, so a missing body is an answer, not an error.

        Deliberately not gated on the latch: the latch stops *writing*, and the sidecars written
        before a failure are exactly what a post-mortem wants to read.
        """
        try:
            path = self._find(self._root / _safe_kind(kind), digest)
            return None if path is None else path.read_text(encoding="utf-8")
        except OSError:
            return None

    # ── sequence + eviction (both re-read from disk, so the store carries no state) ──
    def _next_seq(self) -> int:
        return max((_seq_of(p) for p in self._files()), default=0) + 1

    def _evict(self) -> None:
        files = sorted(self._files(), key=_seq_of)
        for stale in files[: max(0, len(files) - self._cap)]:
            stale.unlink(missing_ok=True)

    def _files(self) -> list[Path]:
        """Every sidecar under the root, across every kind — the cap is global by design."""
        if not self._root.is_dir():
            return []
        return [p for p in self._root.glob(f"*/*{_SIDECAR_SUFFIX}") if _SEQ_RE.match(p.name)]

    @staticmethod
    def _find(folder: Path, digest: str) -> Path | None:
        """The sidecar holding ``digest`` in ``folder``, whatever sequence it was written with.

        A caller-supplied digest is never trusted into a glob: anything but a bare sha256 hex
        string is not a name this store can have written, so it simply has no match.
        """
        if not _HASH_RE.match(digest or "") or not folder.is_dir():
            return None
        return next(iter(sorted(folder.glob(f"*-{digest}{_SIDECAR_SUFFIX}"))), None)

    # ── the fail-safe latch ──
    def _trip(self, exc: BaseException) -> None:
        """Disable capture for good on the first failure — a LATCH, not a retry.

        Exactly one warning, then silence. Capture is observability: a body that cannot be written
        must never take down the research session that produced it, and an intermittent disk must
        not produce a half-captured session that reads as a whole one.
        """
        if self._disabled:
            return
        self._disabled = True
        logger.warning(
            "capture store %s self-disabled after a write failure (%s: %s); no further inputs "
            "will be captured this session",
            self._root,
            type(exc).__name__,
            exc,
        )


def _content_hash(payload: str) -> str:
    """The sha256 of the payload's UTF-8 bytes — the sidecar's name and the ledger's reference."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_kind(kind: str) -> str:
    """A filesystem-safe folder token from a caller-supplied kind slug (never trusted raw)."""
    cleaned = _UNSAFE_RE.sub("-", (kind or "").strip().lower()).strip("-.")
    return cleaned or "unnamed"


def _seq_of(path: Path) -> int:
    match = _SEQ_RE.match(path.name)
    return int(match.group(1)) if match else 0
