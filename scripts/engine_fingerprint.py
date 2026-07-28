#!/usr/bin/env python
"""Engine fingerprint ratchet — check or regenerate ``engine_fingerprint.json`` (story #128).

    uv run python scripts/engine_fingerprint.py            # check (what CI and pre-commit run)
    uv run python scripts/engine_fingerprint.py --write     # regenerate, then commit the file

Arbiter drift (``gates``, ``backtest``) with no ``ENGINE_VERSION`` bump fails the check — and
``--write`` **refuses** to record it (writes nothing, exits 1), because regenerating rewrites every
component at once and so cannot also be how an undeclared arbiter move gets committed (story #145).

A thin I/O wrapper, exactly like ``scripts/parity_harness.py``: the rule, the comparison and the
report live in :mod:`noctis.observability.engine_ratchet` (inside mypy's scope and covered by
``tests/test_engine_ratchet.py``); this file only puts them on a command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    # Let the check run from a source checkout without an editable install (pre-commit's
    # `language: system` hook runs the repo's plain interpreter).
    sys.path.insert(0, str(_SRC))

try:
    from noctis.observability.engine_ratchet import main  # noqa: E402 (after the sys.path shim)
except ImportError as exc:  # a bare interpreter, outside the project environment
    raise SystemExit(
        f"engine fingerprint ratchet: {exc}\n"
        "Run it inside the project environment: uv run python scripts/engine_fingerprint.py"
    ) from exc

if __name__ == "__main__":
    raise SystemExit(main())
