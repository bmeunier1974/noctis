"""Parity metrics — the evidence gate for flipping ``auto`` to episodic (epic #62, story #75).

Both research loops now live behind one seam (the conversation transcript,
:func:`noctis.research.agent.run_agent_research`, and the deterministic episodic driver,
:func:`noctis.research.driver.run_episodic_research`). Before ``auto`` can prefer episodic on a
small window (#76), an operator needs legible evidence that episodic is *at least as effective per
session at materially lower spend*. This module is the computation half of that evidence: pure
functions from ``(ResearchSummary, episodic-rollup-dict | None)`` pairs — one pair per session — to
a typed per-loop metrics row and a side-by-side rendering. It parses **no transcript**: every
number comes from the :class:`~noctis.engine.research.ResearchSummary` both loops already return and
the :class:`~noctis.research.ledger.SessionLedger` rollup the episodic driver already writes. The
thin orchestrator that assembles the paid runs is ``scripts/parity_harness.py``; this module is what
the tests cover.

**Metric definitions (each computed identically for both loops, or rendered ``n/a`` for the loop
that cannot honestly supply it — never a fabricated number).**

* **verdicts / session** *(primary)* — ``(promotions + rejections)`` summed across the loop's
  sessions, divided by the session count. A *verdict* is a spent promote/reject: the gate arbitrated
  it. ``dead_ends`` is a *legacy*-loop concept (:func:`noctis.engine.research.run_research`) that
  neither agent loop sets, so it is excluded — both agent loops fill ``promotions``/``rejections``
  from the same toolbox counters, so this counts the same thing on both sides.

* **tokens / verdict** *(the decision row)* — ``tokens_total`` summed across sessions, divided by
  total verdicts. ``ResearchSummary.tokens_total`` is one comparable spend axis both loops now fill
  from usage they already track: the four neutral usage fields (input + output + cache-creation +
  cache-read) across every completion the loop's own judgment model made, retries included — the
  conversation loop from its per-round usage totals, the episodic driver from its ledger's per-
  episode token sums. Coder-authoring completions run on a *separate* client and are excluded from
  both, so this is apples-to-apples. ``n/a`` when a loop reached zero verdicts (no division).

* **validator first-attempt %** *(a gate-pass rate)* — of the strategies a session tried to author,
  the fraction that passed the write gate on the first attempt. Episodic-only: the ledger rollup
  derives ``validation_failures`` (author stages that never reached OPTIMIZE) from ``authored``
  (stages that did), so the rate is ``authored / (authored + validation_failures)``. The
  conversation loop writes no ledger and its summary carries no such split, so it is ``n/a`` — the
  honest move, not an invented number. ``n/a`` too when there were no author attempts at all.

* **promotion-gate reach %** *(a gate-pass rate)* — the fraction of strategies worked on that
  reached a gated verdict: ``verdicts / candidates``, where ``candidates`` is
  ``len(summary.candidates)``
  (the strategies the session touched, filled identically by both loops). ``n/a`` with no
  candidates.

* **undecided** — ``len(summary.undecided)`` summed: strategies authored but never carried to a
  verdict (archived after the TTL), surfaced by both loops.

**The flip criterion** (:func:`assess_flip`, made legible by :func:`render_comparison`): episodic
meets it when it holds **verdicts/session** (episodic ≥ conversation) **and** spends *materially*
fewer **tokens/verdict** — at least :data:`MATERIAL_TOKEN_REDUCTION` fewer. When a tokens/verdict is
``n/a`` for either loop the token half is *inconclusive* rather than a pass or a fail, so a
zero-verdict run never reads as evidence either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from noctis.engine.research import ResearchSummary

# The two loop labels, matching ``noctis.bootstrap.resolve_research_loop``.
CONVERSATION = "conversation"
EPISODIC = "episodic"

# "Materially lower" for the flip criterion, stated so an operator can judge it: episodic must spend
# at least this fraction fewer tokens per verdict than the conversation loop (0.30 ⇒ a ≥30% cut).
# The whole point of the episodic driver is a small context that stops re-billing a growing
# transcript, so a real win shows as a large tokens/verdict gap, not a rounding-margin one.
MATERIAL_TOKEN_REDUCTION = 0.30

# One session's parity inputs: the summary both loops return, plus the episodic ledger rollup dict
# (``SessionLedger.rollup().to_dict()``) or ``None`` for the conversation loop, which writes none.
SessionPair = tuple[ResearchSummary, dict[str, Any] | None]


@dataclass(frozen=True)
class LoopMetrics:
    """The computed parity row for one loop, aggregated across its sessions. A ``None`` field is a
    metric the loop cannot honestly supply (rendered ``n/a``); it is never a fabricated zero."""

    loop: str
    sessions: int
    verdicts: int
    verdicts_per_session: float
    tokens_total: int
    tokens_per_verdict: float | None
    validator_first_attempt_pct: float | None
    promotion_gate_reach_pct: float | None
    undecided: int


def compute_loop_metrics(loop: str, sessions: Sequence[SessionPair]) -> LoopMetrics:
    """Aggregate one loop's ``(summary, rollup | None)`` session pairs into a :class:`LoopMetrics`.

    Pure and total: an empty session list yields honest zeros with the ratios ``n/a``, and a metric
    a loop cannot supply (validator% without a ledger, any ratio with a zero denominator) is
    ``None`` — never a divide-by-zero and never an invented number."""
    n = len(sessions)
    summaries = [s for s, _ in sessions]
    rollups = [r for _, r in sessions if r is not None]

    verdicts = sum(s.promotions + s.rejections for s in summaries)
    tokens_total = sum(s.tokens_total for s in summaries)
    candidates = sum(len(s.candidates) for s in summaries)
    undecided = sum(len(s.undecided) for s in summaries)

    authored = sum(int(r.get("authored", 0)) for r in rollups)
    validation_failures = sum(int(r.get("validation_failures", 0)) for r in rollups)
    author_attempts = authored + validation_failures

    return LoopMetrics(
        loop=loop,
        sessions=n,
        verdicts=verdicts,
        verdicts_per_session=verdicts / n if n else 0.0,
        tokens_total=tokens_total,
        tokens_per_verdict=tokens_total / verdicts if verdicts else None,
        validator_first_attempt_pct=(
            100.0 * authored / author_attempts if author_attempts else None
        ),
        promotion_gate_reach_pct=100.0 * verdicts / candidates if candidates else None,
        undecided=undecided,
    )


def rollup_for(summary: ResearchSummary) -> dict[str, Any] | None:
    """The episodic ledger rollup dict for one summary, or ``None`` when it wrote no ledger.

    The episodic driver stamps ``summary.ledger_path``; the conversation loop leaves it ``None``.
    Reads the ledger back through the public :meth:`~noctis.research.ledger.SessionLedger.from_path`
    / :meth:`~noctis.research.ledger.SessionLedger.rollup` API — no transcript, no re-parsing. A
    missing/empty ledger file yields ``None`` so a metric never rests on a phantom rollup."""
    if not summary.ledger_path:
        return None
    from noctis.research.ledger import SessionLedger

    ledger = SessionLedger.from_path(summary.ledger_path)
    if not ledger.records():
        return None
    return ledger.rollup().to_dict()


# ── the flip-criterion assessment ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FlipAssessment:
    """Whether episodic meets the flip criterion, and why — the decision :func:`render_comparison`
    prints. ``tokens_materially_lower`` is ``None`` when a tokens/verdict is ``n/a`` (inconclusive,
    neither pass nor fail); ``token_reduction`` is the observed fraction ``(conv - epi) / conv`` or
    ``None`` when it cannot be computed."""

    verdicts_ok: bool
    tokens_materially_lower: bool | None
    token_reduction: float | None
    meets_flip_criterion: bool
    summary: str


def assess_flip(conversation: LoopMetrics, episodic: LoopMetrics) -> FlipAssessment:
    """Judge the episodic-over-conversation flip criterion (#76's gate): episodic holds
    verdicts/session AND spends at least :data:`MATERIAL_TOKEN_REDUCTION` fewer tokens/verdict.

    A ``n/a`` tokens/verdict on either side (a zero-verdict run) makes the token half *inconclusive*
    rather than deciding it, so an empty run is never mistaken for evidence."""
    verdicts_ok = episodic.verdicts_per_session >= conversation.verdicts_per_session

    conv_tpv = conversation.tokens_per_verdict
    epi_tpv = episodic.tokens_per_verdict
    if conv_tpv is None or epi_tpv is None or conv_tpv <= 0:
        tokens_lower: bool | None = None
        reduction: float | None = None
    else:
        reduction = (conv_tpv - epi_tpv) / conv_tpv
        tokens_lower = reduction >= MATERIAL_TOKEN_REDUCTION

    meets = verdicts_ok and tokens_lower is True
    if meets:
        assert reduction is not None  # tokens_lower is True ⇒ reduction was computed
        summary = (
            f"PASS — episodic holds verdicts/session ({episodic.verdicts_per_session:.2f} "
            f">= {conversation.verdicts_per_session:.2f}) and cuts tokens/verdict by "
            f"{reduction * 100:.0f}% (>= {MATERIAL_TOKEN_REDUCTION * 100:.0f}%). Evidence supports "
            f"flipping auto to episodic on this fixture (#76)."
        )
    elif tokens_lower is None:
        summary = (
            "INCONCLUSIVE — a tokens/verdict is n/a (a loop reached zero verdicts), so spend "
            "cannot be compared. Re-run with a fixture/mandate that yields verdicts on both loops."
        )
    else:
        reasons = []
        if not verdicts_ok:
            reasons.append(
                f"verdicts/session fell ({episodic.verdicts_per_session:.2f} < "
                f"{conversation.verdicts_per_session:.2f})"
            )
        if tokens_lower is False:
            assert reduction is not None  # tokens_lower is False ⇒ reduction was computed
            reasons.append(
                f"tokens/verdict cut only {reduction * 100:.0f}% "
                f"(< {MATERIAL_TOKEN_REDUCTION * 100:.0f}%)"
            )
        summary = "FAIL — " + "; ".join(reasons) + ". Evidence does not yet support the flip (#76)."

    return FlipAssessment(
        verdicts_ok=verdicts_ok,
        tokens_materially_lower=tokens_lower,
        token_reduction=reduction,
        meets_flip_criterion=meets,
        summary=summary,
    )


# ── side-by-side rendering ─────────────────────────────────────────────────────────────────────
def _fmt(value: Any) -> str:
    """A cell: ``n/a`` for ``None``, two decimals for a float, the value itself otherwise."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_comparison(conversation: LoopMetrics, episodic: LoopMetrics) -> str:
    """A side-by-side text table of both loops' parity rows plus the flip-criterion verdict.

    The decision rows (verdicts/session, tokens/verdict) sit at the top; a loop that cannot supply a
    metric shows ``n/a`` in its column. The trailing block states the flip criterion and
    :func:`assess_flip`'s verdict, so an operator reads the whole gate off one page."""
    label_w = 26
    col_w = 16
    rows: list[tuple[str, Any, Any]] = [
        ("Sessions", conversation.sessions, episodic.sessions),
        ("Verdicts (total)", conversation.verdicts, episodic.verdicts),
        ("Verdicts / session", conversation.verdicts_per_session, episodic.verdicts_per_session),
        ("Tokens (total)", conversation.tokens_total, episodic.tokens_total),
        ("Tokens / verdict", conversation.tokens_per_verdict, episodic.tokens_per_verdict),
        (
            "Validator 1st-attempt %",
            conversation.validator_first_attempt_pct,
            episodic.validator_first_attempt_pct,
        ),
        (
            "Promotion-gate reach %",
            conversation.promotion_gate_reach_pct,
            episodic.promotion_gate_reach_pct,
        ),
        ("Undecided (total)", conversation.undecided, episodic.undecided),
    ]

    lines = [
        "Parity: conversation vs episodic",
        "",
        f"{'metric':<{label_w}}{CONVERSATION:>{col_w}}{EPISODIC:>{col_w}}",
        f"{'-' * label_w}{'-' * col_w}{'-' * col_w}",
    ]
    for name, conv_val, epi_val in rows:
        lines.append(f"{name:<{label_w}}{_fmt(conv_val):>{col_w}}{_fmt(epi_val):>{col_w}}")

    assessment = assess_flip(conversation, episodic)
    lines += [
        "",
        (
            "Flip criterion (#76): episodic >= conversation on verdicts/session AND tokens/verdict "
            f"at least {MATERIAL_TOKEN_REDUCTION * 100:.0f}% lower."
        ),
        assessment.summary,
    ]
    return "\n".join(lines)
