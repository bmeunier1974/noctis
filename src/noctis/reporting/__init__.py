"""Noctis reporting — the close-of-day report, and the run record.

Two artifacts with different lifetimes live here. ``report`` is the per-day close report an
operator reads each morning. The **run record** — one self-describing
``workspace/runs/<run_id>/run.json`` per run, accumulating across every process invocation that
works on that run — is the ``run_tree`` *package* (all of its I/O: five modules over one narrow
read, layered ``record ← {address, index, lock, evidence} ← store``) plus the pure ``run_record``
builder and the pure ``schema`` validator. They are imported as submodules on purpose — the
record's modules are not re-exported here, so importing the day report never drags the run tree
(or the engine fingerprint) along with it.
"""

from __future__ import annotations

from noctis.reporting.report import (
    ReportData,
    Trade,
    latest_report,
    render_report,
    sweep_stale_reports,
    today_str,
    write_report,
    write_report_json,
    write_reports,
)

__all__ = [
    "ReportData",
    "Trade",
    "latest_report",
    "render_report",
    "sweep_stale_reports",
    "today_str",
    "write_report",
    "write_report_json",
    "write_reports",
]
