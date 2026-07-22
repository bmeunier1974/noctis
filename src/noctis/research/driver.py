"""The episodic research driver — a deterministic session machine that owns the research
protocol and invokes the model only at narrow judgment points (epic #62, story #68).

Where the conversation loop (:func:`noctis.research.agent.run_agent_research`) hands the whole
protocol to one long tool-use transcript, this driver *owns* the protocol in Python and drives
FORMULATE → MATCH → AUTHOR → OPTIMIZE → DECIDE per strategy, calling the model only for the two
judgment episodes (formulate, decide) and executing everything else through the same gated
:class:`~noctis.research.tools.ResearchToolbox` methods the conversation loop uses. It imports no
LLM client or provider SDK: the :class:`~noctis.research.episode.EpisodeRunner` that holds the
client is injected *behind* the formulate/decide callables, so the whole stage machine is testable
end to end with plain fakes and zero LLM involvement.

**The stages (minimal where follow-ups deepen them).**

* **FORMULATE** — one episode proposes a falsifiable thesis (a :class:`FormulateOutput`). Its
  thesis is recorded to the session ledger (so the next formulate's briefing tail already knows
  what this session tried) before any authoring.
* **MATCH** — a passthrough in this story: the fit panel is the symbols the composition root
  chose. Deterministic structural screening lands in #69.
* **AUTHOR** — the formulate output is mapped onto a :class:`~noctis.research.author.StrategyBrief`
  and committed through ``toolbox.tool_write_strategy(brief=…)``; the coder author engine,
  fresh-subprocess validation, and thesis journaling all live behind that one gated method. (Brief
  authoring needs the coder engine — ``research.agent.coder_model``; without it the write is
  refused and the strategy is skipped, exactly like any other author failure.)
* **OPTIMIZE** — a baseline ``tool_run_backtest`` plus one ``tool_run_sweep`` to give DECIDE a real
  journal. A completed sweep clears the ``min_trials`` exhaustion floor; the multi-fidelity recipe
  lands in #70.
* **DECIDE** — one episode proposes a verdict (a :class:`DecideOutput`); the driver *submits it
  through* the gated ``tool_evaluate_vs_champion`` / ``tool_reject_strategy`` methods, so the
  exhaustion floor and evidence checks — not the episode — dispose of it. A verdict the journal
  cannot support comes back as a structured refusal and is handled by the DECIDE policy below.

**The per-stage failed-episode policies (honest, documented, and deterministically tested).**

* **FORMULATE failure ends the session** (``stopped_reason="formulate_failed"``): without a thesis
  there is nothing to author, and a persistently misfiring formulate model would otherwise
  busy-loop against the episode budget rather than do useful work.
* **AUTHOR failure skips the strategy**: a write-gate rejection is a code bug in one draft, not a
  verdict on the thesis, so the driver drops it and formulates the next idea. (The refused draft
  was never added to the undecided set, so nothing lingers.)
* **DECIDE failure re-asks once, then leaves the strategy undecided**: a failed decide episode
  *or* a refused verdict triggers exactly one re-ask, with the failure note / the method's refusal
  folded in as corrective context. If the re-ask still fails or is still refused, the strategy is
  left undecided (the toolbox already holds it there from AUTHOR time, so the session-end rollup
  and the summary surface it, and the TTL sweep archives it later — never a silent loss). A
  ``revise`` verdict is not terminal in this skeleton: the strategy is left undecided for a later
  story to thread back into another optimize round.

**Budgets.** ``max_episodes`` (mapped from ``max_iterations``) is enforced off the injected
``completions`` counter — the episode runner's own per-completion tally, retries included — checked
at every stage boundary; ``budget_minutes`` wall-clock is checked at the same boundaries; a
``stop_event`` (market open / time limit) is honored between stages. No work is interrupted
mid-stage, so the journal, registry, and ledger are always left consistent.

**Contract.** Returns the same :class:`~noctis.engine.research.ResearchSummary` the conversation
loop returns, assembled from the toolbox's own counters, so the runtime, the ``research`` CLI, and
the phase machine are untouched. Every episode output is persisted to the session ledger before the
driver acts on it; stage transitions, verdicts, and a session-end rollup are ledgered too.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from noctis.engine.research import ResearchSummary, StopEvent
from noctis.research.briefings import decide_briefing, formulate_briefing
from noctis.research.episode import EmitContract, EpisodeResult

if TYPE_CHECKING:
    from noctis.research.episode import EpisodeRunner
    from noctis.research.ledger import SessionLedger
    from noctis.research.mandate import Mandate
    from noctis.research.tools import ResearchToolbox

logger = logging.getLogger("noctis.research.driver")

# Stage labels — the ``stage`` string ledgered at each transition and read back by reports.
FORMULATE = "formulate"
MATCH = "match"
AUTHOR = "author"
OPTIMIZE = "optimize"
DECIDE = "decide"

# The verdict vocabulary of a DECIDE episode.
_APPROVE, _REJECT, _REVISE = "approve", "reject", "revise"

# One re-ask on a failed/refused DECIDE, then undecided (see the module docstring).
_DECIDE_ATTEMPTS = 2

# Folded into a re-asked decide briefing so the model sees why its last verdict did not land.
_CORRECTIVE_HEADER = "PREVIOUS VERDICT DID NOT LAND — correct and re-decide:"


# ─────────────────────────────────────────────────────────────────────────────
# Parsed episode outputs (typed, frozen) + their emit contracts
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FormulateOutput:
    """The typed record a FORMULATE episode emits: one falsifiable thesis with the cost
    arithmetic, timeframe, symbol character, scenario intent, and param-space sketch that make it
    authorable, plus optional pivot lineage."""

    thesis: str
    style: str
    class_tag: str
    timeframe: str
    cost_arithmetic: str
    symbol_character: str
    scenario_intent: str
    param_space_sketch: str
    parent_thesis: str | None = None
    pivot_rationale: str | None = None


@dataclass(frozen=True)
class DecideOutput:
    """The typed record a DECIDE episode emits: the proposed verdict and its rationale, the
    class-exhaustion post-mortem, and (for an approve) the nominated symbol-holdout names. The
    verdict is only a *proposal* — the gated toolbox method disposes of it."""

    verdict: str
    reason: str
    class_exhausted: bool
    class_tag: str
    holdout_symbols: tuple[str, ...] = ()
    new_lever: str | None = None


def _require(payload: dict[str, Any], key: str) -> Any:
    """Read a required field; a missing/empty value raises so the episode runner reads it as a
    schema misfire and re-prompts (never a silently-degraded record)."""
    value = payload.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"missing required field {key!r}")
    return value


def _opt(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value else None


def parse_formulate(payload: dict[str, Any]) -> FormulateOutput:
    """The single typed parse both episode transports meet at for FORMULATE."""
    return FormulateOutput(
        thesis=str(_require(payload, "thesis")),
        style=str(_require(payload, "style")),
        class_tag=str(_require(payload, "class_tag")),
        timeframe=str(_require(payload, "timeframe")),
        cost_arithmetic=str(_require(payload, "cost_arithmetic")),
        symbol_character=str(_require(payload, "symbol_character")),
        scenario_intent=str(_require(payload, "scenario_intent")),
        param_space_sketch=str(_require(payload, "param_space_sketch")),
        parent_thesis=_opt(payload, "parent_thesis"),
        pivot_rationale=_opt(payload, "pivot_rationale"),
    )


def parse_decide(payload: dict[str, Any]) -> DecideOutput:
    """The single typed parse both episode transports meet at for DECIDE."""
    verdict = str(_require(payload, "verdict")).strip().lower()
    if verdict not in (_APPROVE, _REJECT, _REVISE):
        raise ValueError(f"verdict {verdict!r} not one of approve/reject/revise")
    holdout = payload.get("holdout_symbols") or []
    if not isinstance(holdout, list):
        raise ValueError("holdout_symbols must be a list of symbols")
    return DecideOutput(
        verdict=verdict,
        reason=str(_require(payload, "reason")),
        class_exhausted=bool(payload.get("class_exhausted", False)),
        class_tag=str(payload.get("class_tag") or ""),
        holdout_symbols=tuple(str(s) for s in holdout if str(s).strip()),
        new_lever=_opt(payload, "new_lever"),
    )


_FORMULATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thesis": {"type": "string", "description": "The market inefficiency, falsifiable."},
        "style": {"type": "string", "description": "momentum / mean-reversion / breakout / …"},
        "class_tag": {"type": "string", "description": "Short approach label (the class key)."},
        "timeframe": {"type": "string", "description": "Bar granularity the thesis needs."},
        "cost_arithmetic": {
            "type": "string",
            "description": "Captured move per trade vs the round-trip cost (aim >= 3x).",
        },
        "symbol_character": {"type": "string", "description": "The KIND of symbol it needs."},
        "scenario_intent": {
            "type": "string",
            "description": "Each known-outcome tape's shape and the behavior it must prove.",
        },
        "param_space_sketch": {"type": "string", "description": "Tunable knobs + ranges."},
        "parent_thesis": {"type": "string", "description": "Optional: the thesis this pivots off."},
        "pivot_rationale": {"type": "string", "description": "Optional: why it pivots."},
    },
    "required": [
        "thesis",
        "style",
        "class_tag",
        "timeframe",
        "cost_arithmetic",
        "symbol_character",
        "scenario_intent",
        "param_space_sketch",
    ],
}

_DECIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [_APPROVE, _REJECT, _REVISE]},
        "reason": {"type": "string", "description": "Why, in the metric's units."},
        "class_exhausted": {
            "type": "boolean",
            "description": "True when the whole class — not just these params — is a dead end.",
        },
        "class_tag": {"type": "string", "description": "The class label the verdict concerns."},
        "new_lever": {"type": "string", "description": "Optional: a genuinely new lever to try."},
        "holdout_symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Profile-matching names kept OUT of all tuning (approve only).",
        },
    },
    "required": ["verdict", "reason", "class_exhausted", "class_tag", "holdout_symbols"],
}

FORMULATE_CONTRACT: EmitContract[FormulateOutput] = EmitContract(
    name="emit_formulation",
    description="Emit ONE falsifiable strategy thesis for this session as a structured object.",
    schema=_FORMULATE_SCHEMA,
    parse=parse_formulate,
)

DECIDE_CONTRACT: EmitContract[DecideOutput] = EmitContract(
    name="emit_verdict",
    description="Emit the verdict for this strategy as a structured object.",
    schema=_DECIDE_SCHEMA,
    parse=parse_decide,
)

# Episode system prompts — one line of role framing; the briefing (rebuilt fresh from disk by the
# builders below) carries every fact and the task. Kept tiny so a small-context backend has room.
_FORMULATE_SYSTEM = (
    "You are a quantitative strategy researcher. Read the briefing and emit exactly one "
    "falsifiable strategy thesis through the provided tool. Do the cost arithmetic first."
)
_DECIDE_SYSTEM = (
    "You are a quantitative strategy researcher. Read the strategy's gate-facing evidence and "
    "emit exactly one verdict through the provided tool. The journaled evidence is the arbiter."
)


# ─────────────────────────────────────────────────────────────────────────────
# Episode callables — production wiring (tests inject plain fakes instead)
# ─────────────────────────────────────────────────────────────────────────────
FormulateEpisode = Callable[[], EpisodeResult[FormulateOutput]]
DecideEpisode = Callable[..., EpisodeResult[DecideOutput]]


def make_episodes(
    *,
    runner: EpisodeRunner,
    toolbox: ResearchToolbox,
    ledger: SessionLedger,
    mandate: Mandate | None,
    context_window: int,
) -> tuple[FormulateEpisode, DecideEpisode]:
    """Bind the formulate/decide episode callables the driver takes to the real
    :class:`~noctis.research.episode.EpisodeRunner`, the two emit contracts, and the briefing
    builders. Each call rebuilds its prompt fresh from disk (the ledger tail + shared digests), so
    there is no accumulated transcript to overflow a small window. The runner (which holds the LLM
    client) is injected here, never in the driver — the driver only ever sees these callables."""

    def formulate() -> EpisodeResult[FormulateOutput]:
        briefing = formulate_briefing(
            toolbox, ledger, mandate=mandate, context_window=context_window
        )
        return runner.run(contract=FORMULATE_CONTRACT, system=_FORMULATE_SYSTEM, briefing=briefing)

    def decide(strategy: str, *, corrective: str | None = None) -> EpisodeResult[DecideOutput]:
        briefing = decide_briefing(
            toolbox, ledger, strategy, mandate=mandate, context_window=context_window
        )
        if corrective:
            briefing = f"{briefing}\n\n{_CORRECTIVE_HEADER}\n{corrective}"
        return runner.run(contract=DECIDE_CONTRACT, system=_DECIDE_SYSTEM, briefing=briefing)

    return formulate, decide


# ─────────────────────────────────────────────────────────────────────────────
# The deterministic protocol machine
# ─────────────────────────────────────────────────────────────────────────────
def _utcnow() -> datetime:
    return datetime.now(UTC)


class _NeverStop:
    def is_set(self) -> bool:
        return False


def _slug(text: str) -> str:
    """A lower_snake_case slug for a strategy name, derived from the thesis class tag."""
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def _invoke(fn: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    """Call a gated toolbox method, turning any raise into a structured error result — research
    must never crash the runtime, and a tool error is data the driver reasons about, not a fault."""
    try:
        result = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 — a tool error is data, never a crash
        logger.warning("toolbox call %s failed: %s", getattr(fn, "__name__", fn), exc)
        return {"error": str(exc)}
    return result if isinstance(result, dict) else {"error": "tool returned no dict result"}


def _entry_exit_brief(fo: FormulateOutput) -> str:
    """Compose the brief's precise-rules field from the formulate output (skeleton mapping; the
    author engine turns it into code). Deepened when MATCH/screening lands (#69)."""
    return (
        f"Author precise long/short/flat rules that make this thesis falsifiable at the "
        f"{fo.timeframe} timeframe. Target symbol character: {fo.symbol_character}. The captured "
        f"move per trade must clear the round-trip cost: {fo.cost_arithmetic}."
    )


def _brief_from_formulate(fo: FormulateOutput, symbols: Sequence[str]) -> dict[str, Any]:
    """Map a FORMULATE output onto the strategy author's brief (thesis, entry/exit, param space,
    scenarios). Passed to ``tool_write_strategy(brief=…)`` — the coder author engine translates it
    into one validated file."""
    return {
        "thesis": fo.thesis,
        "entry_exit": _entry_exit_brief(fo),
        "param_space": fo.param_space_sketch,
        "scenarios": fo.scenario_intent,
        "style": fo.style,
        "symbols": list(symbols),
    }


def run_episodic_research(
    *,
    toolbox: ResearchToolbox,
    ledger: SessionLedger,
    formulate: FormulateEpisode,
    decide: DecideEpisode,
    fit_symbols: Sequence[str],
    budget_minutes: float,
    max_episodes: int,
    completions: Callable[[], int],
    stop_event: StopEvent | None = None,
    now: Callable[[], datetime] = _utcnow,
    mandate_source: str | None = None,
    models: dict[str, Any] | None = None,
    sweep_trials: int | None = None,
) -> ResearchSummary:
    """Run one episodic research session; returns the same summary shape as the conversation loop.

    ``formulate`` / ``decide`` are the two model-judgment episodes (injected — the LLM client is
    behind them, never handed here); every other stage runs through the gated ``toolbox`` methods.
    ``completions`` returns the episode runner's per-completion count (retries included) that
    ``max_episodes`` budgets against; ``budget_minutes`` and ``stop_event`` bound wall-clock and
    interruption. ``fit_symbols`` is the MATCH panel (a passthrough this story). See the module
    docstring for the stage protocol and the per-stage failed-episode policies.
    """
    stop_event = stop_event or _NeverStop()
    summary = ResearchSummary()
    start = now()
    budget_seconds = budget_minutes * 60.0
    formulated = 0

    ledger.record_session_start(
        mandate=mandate_source,
        budgets={"max_episodes": max_episodes, "budget_minutes": budget_minutes},
        models=dict(models or {}),
    )

    def _budget_stop() -> str | None:
        if stop_event.is_set():
            return "stop_event"
        if (now() - start).total_seconds() >= budget_seconds:
            return "time_budget"
        if completions() >= max_episodes:
            return "max_episodes"
        return None

    def _record_episode(stage: str, result: EpisodeResult[Any]) -> None:
        ledger.record_episode(
            stage=stage,
            model=result.model,
            outcome=result.outcome,
            tokens=result.tokens,
            misfires=result.misfires,
        )

    while True:
        stop = _budget_stop()
        if stop:
            summary.stopped_reason = stop
            break

        # ── FORMULATE ────────────────────────────────────────────────────────
        ledger.record_stage(FORMULATE)
        f_result = formulate()
        _record_episode(FORMULATE, f_result)
        if not f_result.ok or f_result.value is None:
            # No thesis ⇒ nothing to author; end the session rather than busy-loop the budget.
            summary.stopped_reason = "formulate_failed"
            break
        fo = f_result.value
        formulated += 1
        name = f"{_slug(fo.class_tag) or 'strategy'}_{formulated}"
        ledger.record_thesis(
            name, fo.thesis, parent_thesis=fo.parent_thesis, pivot_rationale=fo.pivot_rationale
        )

        # ── MATCH (passthrough; deterministic screening lands in #69) ─────────
        ledger.record_stage(MATCH, strategy=name)
        symbols = list(fit_symbols)

        # ── AUTHOR ────────────────────────────────────────────────────────────
        ledger.record_stage(AUTHOR, strategy=name)
        write = _invoke(
            toolbox.tool_write_strategy,
            name=name,
            brief=_brief_from_formulate(fo, symbols),
            class_tag=fo.class_tag,
            thesis=fo.thesis,
            parent_thesis=fo.parent_thesis,
            pivot_rationale=fo.pivot_rationale,
        )
        if "error" in write:
            # A write-gate rejection is a code bug in one draft, not a verdict — skip and move on.
            logger.info("author skipped %s: %s", name, write["error"])
            continue

        # ── OPTIMIZE (baseline backtest + one sweep to clear the exhaustion floor) ──
        ledger.record_stage(OPTIMIZE, strategy=name)
        _invoke(toolbox.tool_run_backtest, name=name, symbols=list(symbols))
        _invoke(toolbox.tool_run_sweep, name=name, symbols=list(symbols), n_trials=sweep_trials)

        # ── DECIDE ────────────────────────────────────────────────────────────
        stop = _budget_stop()
        if stop:
            summary.stopped_reason = stop
            break  # authored + optimized but out of budget — left undecided, honestly
        ledger.record_stage(DECIDE, strategy=name)
        _decide_stage(toolbox, ledger, decide, name, symbols, _record_episode)

    summary.iterations = formulated
    summary.promotions = int(getattr(toolbox, "promotions", 0))
    summary.rejections = int(getattr(toolbox, "rejections", 0))
    summary.candidates = list(getattr(toolbox, "strategies_touched", []))
    summary.author_calls = int(getattr(toolbox, "author_calls", 0))
    summary.undecided = sorted(getattr(toolbox, "undecided", set()))
    ledger.record_session_end(
        formulated=formulated,
        promoted=summary.promotions,
        rejected=summary.rejections,
        note=summary.stopped_reason or None,
    )
    if summary.undecided:
        logger.warning(
            "%d strategies left undecided — archived after the TTL: %s",
            len(summary.undecided),
            ", ".join(summary.undecided),
        )
    logger.info(
        "episodic research session finished: %d formulated, %d promotions, %d rejections (%s)",
        formulated,
        summary.promotions,
        summary.rejections,
        summary.stopped_reason,
    )
    return summary


def _decide_stage(
    toolbox: ResearchToolbox,
    ledger: SessionLedger,
    decide: DecideEpisode,
    name: str,
    symbols: Sequence[str],
    record_episode: Callable[[str, EpisodeResult[Any]], None],
) -> None:
    """Run DECIDE for one strategy: propose a verdict, submit it through the gated toolbox method,
    and on a failed episode or a refused verdict re-ask exactly once (with the note/refusal as
    corrective context) before leaving the strategy undecided. A ``revise`` verdict is left
    undecided too (not terminal in this skeleton)."""
    corrective: str | None = None
    for _ in range(_DECIDE_ATTEMPTS):
        result = decide(name, corrective=corrective)
        record_episode(DECIDE, result)
        if not result.ok or result.value is None:
            corrective = (
                f"The previous decide episode produced no valid verdict "
                f"({result.note or result.outcome}). Re-read the evidence and emit a verdict."
            )
            continue
        verdict = result.value
        if verdict.verdict == _REVISE:
            # A new-lever call — not a terminal verdict here; leave undecided for a later round.
            logger.info("decide %s: revise (left undecided this story)", name)
            return
        outcome = _submit_verdict(toolbox, name, symbols, verdict)
        if "error" not in outcome:
            _record_verdict(ledger, name, verdict, outcome)
            return
        # The gate disposed of the proposal — re-ask once with the refusal as corrective context.
        logger.info("decide %s refused by the gate: %s", name, outcome["error"])
        corrective = str(outcome["error"])
    # Re-ask exhausted: leave the strategy undecided (the toolbox still holds it there).
    logger.info("decide %s left undecided after the re-ask", name)


def _submit_verdict(
    toolbox: ResearchToolbox, name: str, symbols: Sequence[str], verdict: DecideOutput
) -> dict[str, Any]:
    """Submit a proposed verdict through the matching gated toolbox method — the method, not the
    episode, disposes of it (min-trials floor + evidence checks refuse an unsupported verdict)."""
    if verdict.verdict == _REJECT:
        return _invoke(
            toolbox.tool_reject_strategy,
            name=name,
            reason=verdict.reason,
            class_tag=verdict.class_tag or None,
            class_exhausted=verdict.class_exhausted,
        )
    # approve: challenge the champion board with the best-observed params from the journal.
    log = _invoke(toolbox.tool_get_experiment_log, name=name)
    trials = log.get("top_trials") or []
    params = trials[0].get("params", {}) if trials else {}
    return _invoke(
        toolbox.tool_evaluate_vs_champion,
        name=name,
        symbols=list(symbols),
        params=params,
        holdout_symbols=list(verdict.holdout_symbols) or None,
    )


def _record_verdict(
    ledger: SessionLedger, name: str, verdict: DecideOutput, outcome: dict[str, Any]
) -> None:
    """Ledger one spent verdict with its class-level lesson and (for an approve) whether it was
    promoted."""
    if verdict.verdict == _REJECT:
        ledger.record_verdict(name, verdict=_REJECT, lesson=verdict.reason, promoted=False)
    else:
        ledger.record_verdict(
            name, verdict=_APPROVE, lesson=verdict.reason, promoted=bool(outcome.get("promoted"))
        )
