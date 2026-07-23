"""Close-of-day report generation.

Renders a Markdown report to ``<reports_dir>/YYYY-MM-DD.md`` covering trades and rationales, P&L,
open positions, champion changes, the research summary, and notable events (data degradation,
risk halts, integrity repairs, feed-drift flags). Markdown is built in-house (no template
engine dependency); an empty day still produces a valid report with every section.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str
    quantity: float
    price: float
    rationale: str = ""


@dataclass
class ReportData:
    as_of: str
    mode: str = "paper"
    start_equity: float = 0.0
    end_equity: float = 0.0
    realized_pnl: float = 0.0
    # The continuous paper account's curve since inception (equity − starting cash);
    # None when no account exists yet, so old reports and fresh installs render unchanged.
    cumulative_pnl: float | None = None
    account_opened: str | None = None
    # Per-champion forward track record (live-holdout plan 5): each a dict with family,
    # forward_pnl (= realized + current unrealized), realized_pnl, unrealized_pnl,
    # sessions_traded, opened_session. Empty until a champion trades a live-holdout session.
    forward: list[dict] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    promotions: list[dict] = field(default_factory=list)
    demotions: list[dict] = field(default_factory=list)
    champions: list[dict] = field(default_factory=list)
    research: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


def _fmt_pct(a: float, b: float) -> str:
    if not a:
        return "n/a"
    return f"{(b / a - 1.0) * 100:.2f}%"


def _fmt_counts(counts: dict) -> str:
    """``a=1 b=2`` (sorted, deterministic) or ``none`` — the by-kind/by-stage/by-model formatter."""
    return " ".join(f"{k}={v}" for k, v in sorted((counts or {}).items())) or "none"


def _render_research_sessions(lines: list[str], sessions: list) -> None:
    """Render each episodic session's rollup + per-candidate trail (story #74). Called only from
    the Research section; appends nothing when ``sessions`` is empty, so a ledgerless report is
    byte-identical to today."""
    for s in sessions:
        rollup = s.get("rollup") or {}
        lines.append(f"- Session {s.get('session_id', '')}:")
        lines.append(f"  - Theses formulated: {rollup.get('theses', 0)}")
        lines.append(f"  - Files authored: {rollup.get('authored', 0)}")
        lines.append(f"  - Validation failures: {rollup.get('validation_failures', 0)}")
        lines.append(f"  - Trials run: {rollup.get('trials', 0)}")
        lines.append(f"  - Verdicts: {_fmt_counts(rollup.get('verdicts', {}))}")
        lines.append(f"  - Undecided: {rollup.get('undecided', 0)}")
        lines.append(f"  - Escalations: {rollup.get('escalations', 0)}")
        lines.append(f"  - Tokens by stage: {_fmt_counts(rollup.get('tokens_by_stage', {}))}")
        lines.append(f"  - Tokens by model: {_fmt_counts(rollup.get('tokens_by_model', {}))}")
        candidates = s.get("candidates") or []
        if candidates:
            lines.append("  - Candidate trail:")
            for c in candidates:
                trail = " → ".join(["formulate", *c.get("stages", [])])
                metric = c.get("best_metric")
                metric_note = f", best={metric:.4f}" if isinstance(metric, (int, float)) else ""
                lines.append(
                    f"    - {c.get('strategy', '')} [{c.get('outcome', '')}]: {trail} "
                    f"(trials={c.get('trials', 0)}{metric_note})"
                )
                thesis = c.get("thesis")
                if thesis:
                    lines.append(f"      thesis: {thesis}")
                oracle = c.get("oracle")
                if oracle:
                    # The fixed spec's scenarios the candidate was gated against (#86) — so a
                    # post-mortem audits which oracle each candidate met, not just its outcome.
                    lines.append(f"      oracle: {', '.join(oracle)}")


def render_report(data: ReportData) -> str:
    lines: list[str] = []
    lines.append(f"# Close-of-day report — {data.as_of}")
    lines.append("")
    lines.append(f"**Mode:** {data.mode}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Start equity: {data.start_equity:,.2f}")
    lines.append(f"- End equity: {data.end_equity:,.2f}")
    lines.append(f"- Session return: {_fmt_pct(data.start_equity, data.end_equity)}")
    lines.append(f"- Realized P&L: {data.realized_pnl:,.2f}")
    if data.cumulative_pnl is not None and data.account_opened is not None:
        lines.append(f"- Cumulative P&L since {data.account_opened}: {data.cumulative_pnl:+,.2f}")
    lines.append("")

    lines.append("## Trades")
    lines.append("")
    if data.trades:
        lines.append("| Symbol | Side | Qty | Price | Rationale |")
        lines.append("|---|---|---:|---:|---|")
        for t in data.trades:
            lines.append(
                f"| {t.symbol} | {t.side} | {t.quantity:g} | {t.price:.4f} | {t.rationale} |"
            )
    else:
        lines.append("_No trades this session._")
    lines.append("")

    lines.append("## Open positions")
    lines.append("")
    if data.positions:
        for sym, qty in sorted(data.positions.items()):
            lines.append(f"- {sym}: {qty:g}")
    else:
        lines.append("_No open positions._")
    lines.append("")

    lines.append("## Champion changes")
    lines.append("")
    if data.promotions or data.demotions:
        for p in data.promotions:
            rationale = p.get("rationale", "")
            lines.append(f"- PROMOTED {p.get('family')} {p.get('params', {})} — {rationale}")
        for d in data.demotions:
            demoted = d.get("demoted", {})
            lines.append(f"- DEMOTED {demoted.get('family')} {demoted.get('params', {})}")
    else:
        lines.append("_No champion changes._")
    lines.append("")

    lines.append("## Current champions")
    lines.append("")
    if data.champions:
        for c in data.champions:
            lines.append(
                f"- {c.get('family')} {c.get('params', {})} — "
                f"test={c.get('test_metric', 0):.4f} gap={c.get('gap', 0):.4f}"
            )
    else:
        lines.append("_No champions yet._")
    lines.append("")

    lines.append("## Forward track record (live-holdout)")
    lines.append("")
    if data.forward:
        lines.append("| Champion | Forward P&L | Realized | Unrealized | Sessions | Since |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in data.forward:
            lines.append(
                f"| {r.get('family')} | {r.get('forward_pnl', 0.0):+,.2f} | "
                f"{r.get('realized_pnl', 0.0):+,.2f} | {r.get('unrealized_pnl', 0.0):+,.2f} | "
                f"{r.get('sessions_traded', 0)} | {r.get('opened_session', '')} |"
            )
        lines.append("")
        lines.append(
            "_Realized is attributed to whoever held each symbol that session; open-position "
            "unrealized follows the current assignee._"
        )
    else:
        lines.append("_No forward record yet._")
    lines.append("")

    lines.append("## Research")
    lines.append("")
    r = data.research or {}
    lines.append(f"- Candidates tried: {r.get('iterations', 0)}")
    lines.append(f"- Promotions: {r.get('promotions', 0)}")
    lines.append(f"- Rejections: {r.get('rejections', 0)}")
    lines.append(f"- Dead ends: {r.get('dead_ends', 0)}")
    undecided = r.get("undecided", [])
    if undecided:
        lines.append("- Undecided (authored, no verdict):")
        for name in undecided:
            lines.append(f"  - {name}")
    findings = r.get("findings", [])
    if findings:
        lines.append("- Notable findings:")
        for f in findings:
            lines.append(f"  - {f}")
    # Episodic sessions (story #74): a per-session rollup + per-candidate stage trail, derived
    # from each session's ledger. Absent (conversation-loop / legacy sessions carry no ledger) ⇒
    # nothing is appended, so the render stays byte-identical to a ledgerless report.
    _render_research_sessions(lines, r.get("sessions") or [])
    lines.append("")

    lines.append("## Notable events")
    lines.append("")
    if data.events:
        for e in data.events:
            lines.append(f"- {e}")
    else:
        lines.append("_No notable events._")
    lines.append("")

    return "\n".join(lines)


def _archive_if_differs(path: Path, new_content: str) -> Path | None:
    """Archive ``path`` before it is overwritten with *different* content; else no-op.

    Mirrors :meth:`AccountStore.reset`: the canonical path (``<reports_dir>/<as_of>.md``) stays
    put, but a prior report for the same date moves to ``archive/<name>.<mtime>.<ext>``
    (stamped with when it was written) rather than clobbered — no evidence destroyed silently.
    An identical rewrite is a no-op (returns ``None``), so re-running CLOSE for a date never
    churns the archive. Returns the archive path when it moved a file, else ``None``.
    """
    if not path.is_file() or path.read_text(encoding="utf-8") == new_content:
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
    archive = archive_dir / f"{path.stem}.{stamp}{path.suffix}"
    n = 1
    while archive.exists():  # keep earlier archives written in the same second
        archive = archive_dir / f"{path.stem}.{stamp}.{n}{path.suffix}"
        n += 1
    path.replace(archive)
    return archive


def write_report(data: ReportData, reports_dir: str | Path) -> Path:
    """Render ``data`` and write it to ``<reports_dir>/<as_of>.md``. Returns the path.

    If a *differing* report already exists for the date, the prior is archived first (see
    :func:`_archive_if_differs`) and an ``Overwrote existing report`` event is added so the
    overwrite is visible in the day's own report. An identical rewrite is a silent no-op.
    """
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data.as_of}.md"
    content = render_report(data)
    if _archive_if_differs(path, content) is not None:
        note = f"Overwrote existing report for {data.as_of} (prior archived)"
        if note not in data.events:
            data.events.append(note)
        content = render_report(data)  # re-render so the note lands in the report itself
    path.write_text(content, encoding="utf-8")
    return path


def write_report_json(data: ReportData, reports_dir: str | Path) -> Path:
    """Write the structured report to ``<reports_dir>/<as_of>.json`` alongside the Markdown so a
    frontend can consume it later. ``dataclasses.asdict`` recurses into the nested ``Trade``
    dataclasses; the ``research`` dict (minted/promoted specs, findings) is already JSON-safe.

    Like :func:`write_report`, a differing prior for the date is archived, not clobbered. The
    overwrite event is not added here — :func:`write_report` runs first in CLOSE and already
    stamped ``data.events``, which ``asdict`` carries into this JSON.
    """
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data.as_of}.json"
    content = json.dumps(asdict(data), indent=2, sort_keys=True)
    _archive_if_differs(path, content)
    path.write_text(content, encoding="utf-8")
    return path


def sweep_stale_reports(reports_dir: str | Path, *, apply: bool = False) -> list[Path]:
    """Report files dated **after wall-clock today** — the fingerprint of old simulated-clock
    runs (as-of = simulated date, e.g. out to 2027-02).

    Returns the stale ``.md``/``.json`` paths, sorted. When ``apply`` is True, moves each into
    the ``archive/`` subfolder first. Opt-in, never automatic: a legitimately simulated run writes
    future-dated as-ofs on purpose, so an automatic sweep would delete its own outputs. Only
    genuinely date-named reports are considered; anything else is left untouched.
    """
    directory = Path(reports_dir)
    if not directory.is_dir():
        return []
    today = date.today()
    stale: list[Path] = []
    for p in directory.iterdir():
        if not p.is_file() or p.suffix not in (".md", ".json"):
            continue
        try:
            as_of = date.fromisoformat(p.stem)
        except ValueError:
            continue  # not a YYYY-MM-DD report file
        if as_of > today:
            stale.append(p)
    stale.sort()
    if apply:
        archive_dir = directory / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for p in stale:
            dest = archive_dir / p.name
            n = 1
            while dest.exists():
                dest = archive_dir / f"{p.stem}.{n}{p.suffix}"
                n += 1
            p.replace(dest)
    return stale


def latest_report(reports_dir: str | Path) -> Path | None:
    directory = Path(reports_dir)
    if not directory.is_dir():
        return None
    reports = sorted(directory.glob("*.md"))
    return reports[-1] if reports else None


def today_str() -> str:
    return date.today().isoformat()
