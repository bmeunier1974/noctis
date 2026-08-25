"""Test doubles for the capture store's writer seam (story #348).

Capture is strictly secondary at every call site, so several suites need a store that has latched
itself off and then assert the *session* is unharmed. The store writes through an injected writer,
so such a store is built by handing it one that refuses — never by patching ``Path.write_text``
out from under the whole process, which catches pytest's own I/O as readily as the code under test.
"""

from __future__ import annotations

from pathlib import Path

from noctis.observability.capture import CaptureStore


def refuses(path: Path, payload: str) -> None:
    """A writer that fails every write — the injected disk failure."""
    raise OSError("simulated disk failure on the capture store's write")


def failing_capture_store(root: Path) -> CaptureStore:
    """A capture store whose writer refuses, so it latches off on its very first capture."""
    return CaptureStore(root, writer=refuses)
