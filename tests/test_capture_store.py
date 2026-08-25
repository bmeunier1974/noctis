"""The capped, content-addressed capture store — sidecars on disk, retrievable by hash.

External behavior only (story #180, epic #168): what files land under the capture root, what a
read-back by hash returns, that identical payloads collapse to one file, that the cap evicts
oldest-first even for a store instance that never saw the earlier writes, and that a write
failure latches capture off with exactly one warning instead of raising into a research session.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from noctis.observability.capture import DEFAULT_CAP, CaptureStore
from noctis.observability.failsafe import write_text

CAPTURE_LOGGER = "noctis.observability.capture"


def _files(root: Path) -> list[Path]:
    """Every sidecar under the capture root, whatever kind folder it landed in."""
    return sorted(p for p in root.rglob("*") if p.is_file())


# ── round-trip ────────────────────────────────────────────────────────────────────────────────


def test_stored_payload_reads_back_identical(tmp_path):
    store = CaptureStore(tmp_path / "capture")
    payload = "briefing body — ünicode, \ttabs\nand newlines\n"

    digest = store.store("briefing", payload)

    assert digest is not None
    assert store.read("briefing", digest) == payload


def test_first_store_creates_the_area_lazily_under_a_per_kind_folder(tmp_path):
    # No captures ⇒ no capture tree; the first store materializes it, one folder per kind.
    root = tmp_path / "capture"
    store = CaptureStore(root)
    assert not root.exists()

    store.store("author-brief", "brief body")
    store.store("knobs", "knob snapshot")

    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ["author-brief", "knobs"]
    assert len(list((root / "author-brief").iterdir())) == 1
    assert len(list((root / "knobs").iterdir())) == 1


def test_read_of_an_unknown_hash_returns_none(tmp_path):
    store = CaptureStore(tmp_path / "capture")
    store.store("briefing", "a body")

    assert store.read("briefing", "0" * 64) is None
    assert store.read("no-such-kind", "0" * 64) is None


# ── content addressing ────────────────────────────────────────────────────────────────────────


def test_same_payload_twice_yields_one_file_and_the_same_hash(tmp_path):
    root = tmp_path / "capture"
    store = CaptureStore(root)

    first = store.store("briefing", "identical body")
    second = store.store("briefing", "identical body")

    assert first == second
    assert len(_files(root)) == 1  # deduplicated for free
    assert store.read("briefing", first) == "identical body"


def test_different_payloads_yield_different_hashes_and_separate_files(tmp_path):
    root = tmp_path / "capture"
    store = CaptureStore(root)

    a = store.store("briefing", "body A")
    b = store.store("briefing", "body B")

    assert a != b
    assert len(_files(root)) == 2
    assert store.read("briefing", a) == "body A"
    assert store.read("briefing", b) == "body B"


def test_the_same_payload_under_two_kinds_stays_in_its_own_folder(tmp_path):
    # Per-kind subfolders keep call sites apart: the hash is the payload's, the folder is the
    # site's, so a body captured at two sites is readable under each.
    root = tmp_path / "capture"
    store = CaptureStore(root)

    brief = store.store("author-brief", "shared body")
    knobs = store.store("knobs", "shared body")

    assert brief == knobs
    assert len(_files(root)) == 2
    assert store.read("author-brief", brief) == "shared body"
    assert store.read("knobs", knobs) == "shared body"


# ── cap eviction (state re-read from disk, never held in memory) ──────────────────────────────


def test_cap_keeps_the_newest_sidecars_and_evicts_the_oldest(tmp_path):
    root = tmp_path / "capture"
    store = CaptureStore(root, cap=10)

    digests = [store.store("briefing", f"body {i}") for i in range(14)]

    assert len(_files(root)) == 10
    assert [store.read("briefing", d) for d in digests[:4]] == [None] * 4  # oldest 4 evicted
    assert [store.read("briefing", d) for d in digests[4:]] == [f"body {i}" for i in range(4, 14)]


def test_a_fresh_store_over_an_existing_tree_evicts_correctly(tmp_path):
    # The store carries NO in-memory state: a second instance over the same root re-reads the
    # ordering from disk, so it evicts the *earlier* instance's oldest sidecars, not its own.
    root = tmp_path / "capture"
    first_digests = [CaptureStore(root, cap=10).store("briefing", f"body {i}") for i in range(10)]

    fresh = CaptureStore(root, cap=10)
    later_digests = [fresh.store("briefing", f"later {i}") for i in range(3)]

    assert len(_files(root)) == 10
    assert [fresh.read("briefing", d) for d in first_digests[:3]] == [None] * 3
    assert [fresh.read("briefing", d) for d in first_digests[3:]] == [
        f"body {i}" for i in range(3, 10)
    ]
    assert [fresh.read("briefing", d) for d in later_digests] == [f"later {i}" for i in range(3)]


def test_the_cap_spans_every_kind(tmp_path):
    # The cap bounds the whole capture area, not one folder: qa/ cannot grow without bound
    # however many sites capture into it.
    root = tmp_path / "capture"
    store = CaptureStore(root, cap=8)

    for i in range(6):
        store.store("briefing", f"briefing {i}")
    for i in range(6):
        store.store("knobs", f"knobs {i}")

    assert len(_files(root)) == 8


def test_re_storing_a_payload_does_not_consume_cap_room(tmp_path):
    # A repeat store is one file, so a chatty site repeating one body cannot evict the others.
    root = tmp_path / "capture"
    store = CaptureStore(root, cap=4)

    kept = store.store("briefing", "the stable body")
    others = [store.store("knobs", f"other {i}") for i in range(3)]
    for _ in range(5):
        store.store("briefing", "the stable body")

    assert len(_files(root)) == 4
    assert store.read("briefing", kept) == "the stable body"
    assert [store.read("knobs", d) for d in others] == [f"other {i}" for i in range(3)]


def test_default_cap_is_generous_and_holds(tmp_path):
    # The cap is a safety net, not a budget: a session emits dozens, the default allows hundreds.
    assert DEFAULT_CAP >= 200

    root = tmp_path / "capture"
    store = CaptureStore(root)  # default cap
    for i in range(DEFAULT_CAP + 5):
        store.store("briefing", f"body {i}")

    assert len(_files(root)) == DEFAULT_CAP


# ── the fail-safe latch ───────────────────────────────────────────────────────────────────────
#
# The seam these tests reach through is the store's own WRITER (story #348) — the callable handed
# in at construction, standing exactly where the store writes. A failing disk is injected there,
# never monkeypatched process-wide, so nothing but the store under test is ever made to fail.


class Disk:
    """The writer seam as a test double: writes for real until ``fail`` is set, then refuses."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.attempts = 0

    def __call__(self, path: Path, payload: str) -> None:
        self.attempts += 1
        if self.fail:
            raise OSError("simulated disk failure on the capture store's write")
        write_text(path, payload)


def _capture_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == CAPTURE_LOGGER and r.levelno == logging.WARNING]


def test_a_write_failure_warns_once_and_latches_capture_off(tmp_path, caplog):
    root = tmp_path / "capture"
    disk = Disk(fail=True)
    store = CaptureStore(root, writer=disk)

    with caplog.at_level(logging.WARNING, logger=CAPTURE_LOGGER):
        first = store.store("briefing", "body one")  # fails → warns once, latches
        later = [store.store("briefing", f"body {i}") for i in range(3)]  # silent no-ops

    assert first is None  # nothing was captured, and the caller is told so
    assert later == [None, None, None]
    assert store.disabled is True
    assert len(_capture_warnings(caplog)) == 1  # exactly one warning — no retries, no spam
    assert disk.attempts == 1  # no write was even attempted after the latch tripped


def test_a_latched_store_never_raises_and_reads_back_nothing(tmp_path, caplog):
    root = tmp_path / "capture"
    store = CaptureStore(root, writer=Disk(fail=True))

    with caplog.at_level(logging.WARNING, logger=CAPTURE_LOGGER):
        store.store("briefing", "body one")
        assert store.store("knobs", "another body") is None  # a different kind, still a no-op
        assert store.read("briefing", "0" * 64) is None  # a read after the latch never raises

    assert _files(root) == []


def test_sidecars_written_before_the_latch_stay_readable(tmp_path, caplog):
    # The latch stops writing, not reading: whatever was captured before the failure is exactly
    # what a post-mortem wants back.
    root = tmp_path / "capture"
    disk = Disk()
    store = CaptureStore(root, writer=disk)
    kept = store.store("briefing", "captured while the disk was healthy")

    disk.fail = True
    with caplog.at_level(logging.WARNING, logger=CAPTURE_LOGGER):
        assert store.store("briefing", "never lands") is None

    assert kept is not None
    assert store.read("briefing", kept) == "captured while the disk was healthy"


def test_capture_survives_a_failure_by_staying_off_for_the_rest_of_the_session(tmp_path, caplog):
    # The latch never clears: once capture has failed it stays off even after the filesystem
    # recovers, so a session sees one warning and one behavior, not intermittent half-capture.
    root = tmp_path / "capture"
    disk = Disk(fail=True)
    store = CaptureStore(root, writer=disk)

    with caplog.at_level(logging.WARNING, logger=CAPTURE_LOGGER):
        store.store("briefing", "body one")
        disk.fail = False  # the disk recovers
        assert store.store("briefing", "body two") is None

    assert _files(root) == []
    assert len(_capture_warnings(caplog)) == 1


# ── hash stability across processes ───────────────────────────────────────────────────────────


_SUBPROCESS_STORE = """
import sys
from noctis.observability.capture import CaptureStore

root, payload = sys.argv[1], sys.argv[2]
print(CaptureStore(root).store("briefing", payload))
"""


def test_the_hash_is_stable_across_instances_and_interpreters(tmp_path):
    # Same payload → same hash, in another instance and in another interpreter, so a hash
    # recorded in a ledger today still names the same sidecar tomorrow.
    root = tmp_path / "capture"
    payload = "a briefing whose hash must not move"

    from_child = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_STORE, str(root), payload],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    same_root = CaptureStore(root).store("briefing", payload)
    other_root = CaptureStore(tmp_path / "elsewhere").store("briefing", payload)

    assert from_child == same_root == other_root
    assert len(_files(root)) == 1  # the parent deduped onto the child's file
    assert CaptureStore(root).read("briefing", from_child) == payload
