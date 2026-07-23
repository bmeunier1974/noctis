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
* **MATCH** — *deterministic* structural screening in driver code, no model call (#69). The
  formulate output's ``symbol_character`` prose is mapped (:func:`character_to_profile`) onto the
  screener's trend/volatility/liquidity band profile, and the gated, budget-aware
  ``toolbox.tool_screen_symbols`` ranks the lake for it. Its ``suggested_fit`` becomes the fit set
  AUTHOR/OPTIMIZE tune on; its ``reserved_holdout`` is held out *by code* — those names never
  enter a write/backtest/sweep and are submitted at DECIDE time as the symbol-holdout nominees.
  What was a soft prompt rule in the conversation loop ("keep reserved_holdout out of tuning") is
  now a structural guarantee. An empty screen (no lake match) falls back to the composition-root
  panel exactly as the pre-#69 passthrough did, ledgered as a fallback; the discover-episode that
  would replace that fallback is out of this epic's scope for now.
* **AUTHOR** — the formulate output is mapped onto a :class:`~noctis.research.author.StrategyBrief`
  and committed through ``toolbox.tool_write_strategy(brief=…)``; the coder author engine,
  fresh-subprocess validation, and thesis journaling all live behind that one gated method. (Brief
  authoring needs the coder engine — ``research.agent.coder_model``; without it the write is
  refused and the strategy is skipped, exactly like any other author failure.)
* **OPTIMIZE** — the v1 multi-fidelity tuning recipe (#70), run with **zero LLM calls** entirely
  through the gated toolbox methods (:func:`_optimize_stage`): a full-panel **baseline** backtest →
  a **cheap** exploration sweep (a subset of the fit panel at a truncated recent window) → a
  **full-panel confirm** of that sweep's best params → **≤ 2 narrowed re-tune rounds**, each earned
  by a meaningful improvement and hard-capped at two. Every backtest/sweep journals its trials
  exactly as before, so a completed cheap sweep clears the ``min_trials`` exhaustion floor and the
  best-observed params reach DECIDE through the same journal. The recipe is code with
  data-dependent branches (promising vs weak baseline, re-tune improvement vs stall) — the exact
  branch points a later ``interpret`` episode could slot into, deliberately out of scope here — and
  it handles a budget refusal at any step honestly: it stops tuning and hands DECIDE whatever
  evidence exists (the gates dispose).
* **DECIDE** — one episode proposes a verdict (a :class:`DecideOutput`); the driver *submits it
  through* the gated ``tool_evaluate_vs_champion`` / ``tool_reject_strategy`` methods, so the
  exhaustion floor and evidence checks — not the episode — dispose of it. A verdict the journal
  cannot support comes back as a structured refusal and is handled by the DECIDE policy below.
  The decide schema still carries a ``holdout_symbols`` field, but with deterministic MATCH the
  *code* owns the reservation: the driver submits the MATCH-reserved holdout names and **ignores
  (logging) any model nomination** that disagrees, so a model proposal can never overwrite the
  structural reservation. When MATCH fell back (no reservation), the holdout nomination is left
  empty and the toolbox's own out-of-fit fallback picks the symbol holdout, exactly as before.

**Driver-side sanity checks on episode outputs (story #71).** A small model can emit
schema-valid nonsense that would burn an authoring call or a verdict attempt; three cheap,
advisory-corrective checks catch it first. They are a *first line*, not a new gate — the promotion
gates still arbitrate quality; a failing check only earns one corrective re-ask (a message naming
exactly what was wrong) before the stage's own failed-episode policy applies:

* **FORMULATE — cost arithmetic must cite the digest.** At least one number in the formulate
  output's ``cost_arithmetic`` must appear in the MARKET ECONOMICS digest the episode was shown
  (:func:`_check_cost_arithmetic`, numeric-token overlap; a number-free sketch fails). The digest
  source is the same one the briefing embeds, injected as ``market_digest``.
* **FORMULATE — the proposed class must not be exhausted.** ``class_tag`` is checked against the
  exhausted-class registry (:func:`_check_class_tag`, mirroring the write-gate guard). FORMULATE
  carries no ``new_lever`` escape, so an exhausted tag simply fails — the honest move is a
  genuinely different class.
* **DECIDE — ``revise`` is capped.** A ``revise`` verdict earns the one corrective re-ask (naming
  the cap); a second ``revise`` for the same strategy exhausts it and leaves the strategy undecided.

Each check's outcome rides the ledger's ``episode`` line (a ``checks`` payload of
``{check, result}`` — ``reask`` when it earned the correction, ``exhausted`` when it fired again),
so which check fired and how its re-ask resolved is visible to reports.

**The per-stage failed-episode policies (honest, documented, and deterministically tested).**

* **FORMULATE failure ends the session** (``stopped_reason="formulate_failed"``): without a thesis
  there is nothing to author, and a persistently misfiring formulate model would otherwise
  busy-loop against the episode budget rather than do useful work. A FORMULATE sanity check that
  fails, then fails again on its single re-ask, ends the session the same way — the model was shown
  what was wrong and did not fix it.
* **AUTHOR failure skips the strategy**: a write-gate rejection is a code bug in one draft, not a
  verdict on the thesis, so the driver drops it and formulates the next idea. (The refused draft
  was never added to the undecided set, so nothing lingers.)
* **DECIDE failure re-asks once, then leaves the strategy undecided**: a failed decide episode, a
  refused verdict, *or* a capped ``revise`` triggers exactly one re-ask, with the failure note /
  refusal / cap folded in as corrective context. The two-attempt cap is shared across all three
  causes — one corrective re-ask total, never more — so a mix (e.g. a ``revise`` then a gate
  refusal) still spends only the one re-ask. If the re-ask still fails, is still refused, or
  ``revise``s again, the strategy is left undecided (the toolbox already holds it there from AUTHOR
  time, so the session-end rollup and the summary surface it, and the TTL sweep archives it later —
  never a silent loss).

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
from noctis.observability.events import Event, stage_event, tool_event
from noctis.research import digests
from noctis.research.briefings import decide_briefing, formulate_briefing
from noctis.research.episode import EmitContract, EpisodeResult
from noctis.strategies import library
from noctis.strategies.scenario_spec import (
    Behavior,
    LegSpec,
    ScenarioSpec,
    SpecError,
    SpecSuite,
    compile_spec,
    describe_spec,
)
from noctis.strategies.scenarios import Scenario

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

# One re-ask on a failed/refused DECIDE, then undecided (see the module docstring). The same
# two-attempt cap absorbs a `revise` verdict: the first revise earns the corrective re-ask, a
# second revise for the same strategy exhausts it (undecided).
_DECIDE_ATTEMPTS = 2

# One re-ask on a failed FORMULATE sanity check, then end the session (the FORMULATE policy).
_FORMULATE_ATTEMPTS = 2

# Folded into a re-asked briefing so the model sees why its last output did not land.
_CORRECTIVE_HEADER = "PREVIOUS VERDICT DID NOT LAND — correct and re-decide:"
_FORMULATE_CORRECTIVE_HEADER = (
    "PREVIOUS THESIS FAILED A DRIVER SANITY CHECK — correct and re-formulate:"
)

# ── driver-side sanity checks on episode outputs (story #71) ─────────────────────────────────
# Three cheap, advisory-corrective checks catch schema-valid nonsense from a small model *before*
# it burns an authoring call or a verdict attempt. Each is a FIRST LINE, not a gate: a failing
# check earns exactly one corrective re-ask (a message naming what was wrong), and if the re-ask
# still fails the stage's own failed-episode policy applies. The promotion gates absorb whatever
# slips past. The check ids and result labels below ride the ledger episode line's ``checks``
# payload so a reader sees which check fired and how the re-ask resolved.
_COST_CHECK = "cost_arithmetic"  # cost_arithmetic cites no number from the digest shown
_CLASS_CHECK = "class_tag_exhausted"  # the proposed class is already a declared dead end
_REVISE_CHECK = "revise_cap"  # a `revise` verdict — capped at one re-ask per strategy
_REASK, _EXHAUSTED = "reask", "exhausted"  # earned the re-ask vs fired again and exhausted it

# A maximal integer/decimal run — the cheap numeric-token extractor the cost check overlaps on.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_REVISE_CORRECTIVE = (
    "a 'revise' verdict is capped — this session does not spend another tuning round on your "
    "say-so. Decide on the current evidence: approve (challenge the board), reject (a class-level "
    "dead end), or revise once more only if the class genuinely continues."
)

# ── the deterministic MATCH lexicon ─────────────────────────────────────────────────────────
# A pure keyword map from the formulate output's prose ``symbol_character`` onto the screener's
# three band dimensions (each low | high | any). Per dimension the low/negative markers are tested
# BEFORE the high/positive ones — so "illiquid" reads low (not high on the "liquid" substring) and
# "low volatility" reads low (not high on "volatility") — and a dimension with no marker stays
# "any" (the screen then ranks that axis by liquidity, most-tradable first). It is deliberately
# best-effort and side-effect free: the tickers are the lake's job (:func:`tool_screen_symbols`);
# this only turns the thesis's character sketch into the band request the screen understands.
_CHARACTER_LEXICON: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "trend": (
        (
            "chop",
            "range-bound",
            "rangebound",
            "ranging",
            "mean-revert",
            "mean revert",
            "reversion",
            "sideways",
            "oscillat",
        ),
        ("trend", "momentum", "directional", "breakout", "persistent"),
    ),
    "volatility": (
        ("calm", "quiet", "low-vol", "low vol", "stable", "tight", "placid", "sleepy"),
        ("volatile", "volatility", "high-vol", "high vol", "wide", "explosive"),
    ),
    "liquidity": (
        (
            "illiquid",
            "small-cap",
            "small cap",
            "micro-cap",
            "thin",
            "low-volume",
            "low volume",
            "lightly traded",
        ),
        (
            "liquid",
            "large-cap",
            "large cap",
            "mega-cap",
            "mega cap",
            "high-volume",
            "high volume",
            "deep",
            "heavily traded",
        ),
    ),
}


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    """Whether any marker opens a word in ``text``. A leading word boundary (never a bare
    substring) is what keeps "thin" out of "anything" and "liquid" out of "illiquid"; the trailing
    end is left open so a marker still catches its plurals ("mega-cap" ⇒ "mega-caps", "trend" ⇒
    "trending")."""
    return any(re.search(r"\b" + re.escape(m), text) for m in markers)


def _band_for(text: str, low_markers: tuple[str, ...], high_markers: tuple[str, ...]) -> str:
    if _any_marker(text, low_markers):
        return "low"
    if _any_marker(text, high_markers):
        return "high"
    return "any"


def character_to_profile(symbol_character: str) -> dict[str, str]:
    """Map a formulate output's prose ``symbol_character`` onto the screener's band profile.

    Pure and deterministic (no LLM, no I/O): the same sketch always yields the same
    ``{trend, volatility, liquidity}`` bands, each ``low`` / ``high`` / ``any`` (see
    :data:`_CHARACTER_LEXICON`). The bands are exactly the keyword args
    :meth:`~noctis.research.tools.ResearchToolbox.tool_screen_symbols` takes."""
    text = (symbol_character or "").lower()
    return {dim: _band_for(text, low, high) for dim, (low, high) in _CHARACTER_LEXICON.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Parsed episode outputs (typed, frozen) + their emit contracts
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FormulateOutput:
    """The typed record a FORMULATE episode emits: one falsifiable thesis with the cost
    arithmetic, timeframe, symbol character, a structured **scenario spec**, and a param-space
    sketch that make it authorable, plus optional pivot lineage.

    The scenario oracle is inverted (epic #78): FORMULATE emits a structured
    :class:`~noctis.strategies.scenario_spec.SpecSuite` (``scenario_spec``) in the #82 vocabulary —
    the model reasons about tape shape and one behavior tag per scenario and never writes a bar
    index. The driver compiles it at parse time (a structural validity check at ``warm=0``) into
    the ``scenarios`` tuple; the write gate (#84) re-compiles the *same* spec at the strategy's real
    declared warmup. Both the parsed suite and the parse-time compilation are carried so downstream
    stages (author brief #84/#85, write gate #84) consume the fixed oracle rather than re-parsing
    free prose."""

    thesis: str
    style: str
    class_tag: str
    timeframe: str
    cost_arithmetic: str
    symbol_character: str
    scenario_spec: SpecSuite
    scenarios: tuple[Scenario, ...]
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


# The representative warmup FORMULATE compiles a spec at *parse* time — purely a structural
# validity check of the spec's shape (the write gate #84 re-compiles at the strategy's real
# declared warmup). Zero, so a directional entry leg begins right after the setup pad.
PARSE_WARM = 0

# The allowed leg kinds (the #82 segment builders) the FORMULATE schema advertises to the model.
_LEG_KINDS = ("flat", "trend", "selloff", "recovery", "chop", "vol_spike", "gap")


def _build_leg(payload: Any, scenario_name: str, index: int) -> LegSpec:
    """Construct one frozen :class:`LegSpec` from the model's JSON leg; raise on a malformed shape
    (a non-object leg, a missing kind, a non-integer length) so it re-prompts as a schema misfire.
    ``pct``/``amplitude``/``period`` default to 0 and are ignored per kind by the compiler."""
    if not isinstance(payload, dict):
        raise ValueError(f"scenario {scenario_name!r} leg {index}: each leg must be an object")
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"scenario {scenario_name!r} leg {index}: a leg 'kind' is required")
    bars = payload.get("bars", 0)
    if isinstance(bars, bool) or not isinstance(bars, int):
        raise ValueError(
            f"scenario {scenario_name!r} leg {index}: 'bars' must be an integer length (0 for gap)"
        )
    return LegSpec(
        kind=kind,
        bars=bars,
        pct=float(payload.get("pct", 0.0) or 0.0),
        amplitude=float(payload.get("amplitude", 0.0) or 0.0),
        period=int(payload.get("period", 0) or 0),
    )


def _build_behavior(value: Any, scenario_name: str) -> Behavior:
    """Map the model's behavior string onto the #82 :class:`Behavior` tag; raise with the allowed
    vocabulary on an unknown/missing tag so it re-prompts as a schema misfire."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"scenario {scenario_name!r}: a 'behavior' tag is required")
    try:
        return Behavior(value)
    except ValueError:
        allowed = ", ".join(b.value for b in Behavior)
        raise ValueError(
            f"scenario {scenario_name!r}: unknown behavior {value!r}; use one of {allowed}"
        ) from None


def _build_scenario(payload: Any, index: int) -> ScenarioSpec:
    """Construct one frozen :class:`ScenarioSpec` from the model's JSON scenario; raise on any
    malformed shape (missing name/legs, a non-integer leg reference) so it re-prompts as a
    misfire."""
    if not isinstance(payload, dict):
        raise ValueError(f"scenario {index}: each scenario must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"scenario {index}: a non-empty 'name' is required")
    raw_legs = payload.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        raise ValueError(f"scenario {name!r}: a non-empty 'legs' list is required")
    legs = tuple(_build_leg(leg, name, i) for i, leg in enumerate(raw_legs))
    behavior = _build_behavior(payload.get("behavior"), name)
    leg_ref = payload.get("leg")
    if leg_ref is not None and (isinstance(leg_ref, bool) or not isinstance(leg_ref, int)):
        raise ValueError(f"scenario {name!r}: 'leg' must be an integer leg index or omitted")
    return ScenarioSpec(name=name, legs=legs, behavior=behavior, leg=leg_ref)


def _parse_scenario_spec(payload: dict[str, Any]) -> tuple[SpecSuite, tuple[Scenario, ...]]:
    """Parse the structured ``scenario_spec`` into a :class:`SpecSuite` and compile it at
    :data:`PARSE_WARM` as a structural validity check.

    Any malformed shape raises :class:`ValueError`; an uncompilable spec re-raises the compiler's
    precise :class:`SpecError` message as a :class:`ValueError`. Both are caught by the episode
    runner as a schema misfire (exactly like a missing field), and the message rides into the
    corrective so the model can fix the spec on the re-prompt."""
    raw = _require(payload, "scenario_spec")
    if not isinstance(raw, dict):
        raise ValueError("scenario_spec must be an object with a 'scenarios' list")
    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenario_spec.scenarios must be a non-empty list of scenarios")
    suite = SpecSuite(scenarios=tuple(_build_scenario(s, i) for i, s in enumerate(raw_scenarios)))
    try:
        compiled = compile_spec(suite, PARSE_WARM)
    except SpecError as exc:
        raise ValueError(str(exc)) from exc
    return suite, compiled


def parse_formulate(payload: dict[str, Any]) -> FormulateOutput:
    """The single typed parse both episode transports meet at for FORMULATE.

    The structured ``scenario_spec`` is parsed into the #82 dataclasses and compiled at parse time
    (:func:`_parse_scenario_spec`); a malformed or uncompilable spec raises here, so the episode
    runner re-prompts it as a schema misfire exactly like any missing/invalid field."""
    scenario_spec, scenarios = _parse_scenario_spec(payload)
    return FormulateOutput(
        thesis=str(_require(payload, "thesis")),
        style=str(_require(payload, "style")),
        class_tag=str(_require(payload, "class_tag")),
        timeframe=str(_require(payload, "timeframe")),
        cost_arithmetic=str(_require(payload, "cost_arithmetic")),
        symbol_character=str(_require(payload, "symbol_character")),
        scenario_spec=scenario_spec,
        scenarios=scenarios,
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


# The structured scenario_spec the model emits — a 1:1 mapping onto the #82 vocabulary. The model
# reasons about tape SHAPE (legs) and ONE behavior tag per scenario; it NEVER writes a bar index —
# the compiler derives every window from the leg boundaries and the strategy's declared warmup.
_LEG_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": list(_LEG_KINDS),
            "description": "The segment shape of this leg.",
        },
        "bars": {
            "type": "integer",
            "description": "The leg's LENGTH in decision bars (never a bar index); 0 for a gap.",
        },
        "pct": {
            "type": "number",
            "description": "Signed total move for trend/selloff/recovery/gap (0.05 = +5%).",
        },
        "amplitude": {
            "type": "number",
            "description": "Oscillation amplitude for chop / vol_spike (e.g. 0.03).",
        },
        "period": {"type": "integer", "description": "Wave length in bars for chop (default 8)."},
    },
    "required": ["kind", "bars"],
}

_SCENARIO_SPEC_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "A unique name for this scenario tape."},
        "legs": {
            "type": "array",
            "items": _LEG_SPEC_SCHEMA,
            "description": "The ordered legs of the tape, in decision-bar lengths.",
        },
        "behavior": {
            "type": "string",
            "enum": [b.value for b in Behavior],
            "description": "The ONE behavior this tape must prove (the thesis's contribution).",
        },
        "leg": {
            "type": "integer",
            "description": "0-based index into 'legs' the behavior targets; omit for never_trade.",
        },
    },
    "required": ["name", "legs", "behavior"],
}

_SCENARIO_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": _SCENARIO_SPEC_ITEM_SCHEMA,
            "description": (
                "2-8 known-outcome tapes: at least one directional entry (enter/hold "
                "long/short) and at least one never_trade tape. You author tape SHAPE and "
                "behavior only — never a bar index."
            ),
        },
    },
    "required": ["scenarios"],
}

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
        "scenario_spec": _SCENARIO_SPEC_SCHEMA,
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
        "scenario_spec",
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
# Driver-side sanity checks on episode outputs (story #71)
# ─────────────────────────────────────────────────────────────────────────────
def _numbers(text: str) -> set[float]:
    """The set of numeric-token values in ``text`` — the cheap, formatting-agnostic basis for the
    cost-arithmetic overlap check (``4bp`` and ``4.0`` both read ``4.0``)."""
    out: set[float] = set()
    for token in _NUMBER_RE.findall(text or ""):
        try:
            out.add(float(token))
        except ValueError:  # pragma: no cover — the regex only yields parseable numbers
            continue
    return out


@dataclass(frozen=True)
class SanityCheck:
    """One driver-side sanity-check outcome: ``ok`` when it passed, else the ``check`` id that
    fired and the ``corrective`` message naming what was wrong (folded into the one re-ask)."""

    ok: bool
    check: str = ""
    corrective: str = ""


_CHECK_OK = SanityCheck(True)


def _check_cost_arithmetic(fo: FormulateOutput, digest_text: str) -> SanityCheck:
    """The cost arithmetic must cite at least one number that actually appears in the MARKET
    digest the episode was shown; a number-free or wholly-invented cost sketch fails. Cheap and
    honest — a first line against schema-valid nonsense, never a parser of the arithmetic."""
    if _numbers(fo.cost_arithmetic) & _numbers(digest_text):
        return _CHECK_OK
    return SanityCheck(
        False,
        _COST_CHECK,
        "your cost_arithmetic cites no number from the MARKET ECONOMICS digest you were shown — "
        "redo it against the real round-trip cost and per-bar move numbers in the briefing.",
    )


def _check_class_tag(fo: FormulateOutput, exhausted: Any) -> SanityCheck:
    """The proposed ``class_tag`` must not already be a declared dead end (mirrors the write-gate
    exhausted-class guard). FORMULATE carries no ``new_lever`` escape, so an exhausted tag simply
    fails here — the honest move is a genuinely different class, not a loosened guard."""
    if exhausted is None:
        return _CHECK_OK
    dead = exhausted.is_exhausted(fo.class_tag)
    if dead is None:
        return _CHECK_OK
    reason = dead.get("reason", "") if isinstance(dead, dict) else ""
    return SanityCheck(
        False,
        _CLASS_CHECK,
        f"the class {fo.class_tag!r} was already declared a dead end by a prior session "
        f"({reason}) — do not re-mine it; propose a genuinely different class.",
    )


def _formulate_checks(fo: FormulateOutput, digest_text: str, exhausted: Any) -> SanityCheck:
    """Run the FORMULATE sanity checks in order (cost arithmetic, then exhausted class) and return
    the first that fires, or the passing result when both are clean."""
    for check in (_check_cost_arithmetic(fo, digest_text), _check_class_tag(fo, exhausted)):
        if not check.ok:
            return check
    return _CHECK_OK


# ─────────────────────────────────────────────────────────────────────────────
# Episode callables — production wiring (tests inject plain fakes instead)
# ─────────────────────────────────────────────────────────────────────────────
FormulateEpisode = Callable[..., EpisodeResult[FormulateOutput]]
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

    def formulate(*, corrective: str | None = None) -> EpisodeResult[FormulateOutput]:
        briefing = formulate_briefing(
            toolbox, ledger, mandate=mandate, context_window=context_window
        )
        if corrective:
            briefing = f"{briefing}\n\n{_FORMULATE_CORRECTIVE_HEADER}\n{corrective}"
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


# ── observability: stage boundaries + deterministic tool lines (story #73) ───────────────────
# The driver calls the gated toolbox methods directly (never through the conversation loop's
# ``dispatch``), so the tool/result line the loop emits after each dispatch has no counterpart
# here. The driver emits it itself, through the SAME builder the loop uses
# (:func:`noctis.observability.events.tool_event`), so an episodic session's -v feed reads as one
# continuous narrative — screens, backtests, sweeps, writes, and verdict submissions all as tool
# lines, interleaved with the stage boundaries and the episode think/say/usage the runner tees.
# ``ToolEmitter`` is ``(short_tool_name, args, result) -> None``; ``_NO_EMIT`` is the no-sink no-op
# so a bare run stays byte-identical.
ToolEmitter = Callable[[str, dict[str, Any], dict[str, Any]], None]


def _no_emit(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    return None


def _tool_emitter(on_event: Callable[[Event | str], None] | None, toolbox: Any) -> ToolEmitter:
    """Build the driver's tool-line emitter, or the no-op when no sink is wired. Each line uses the
    toolbox's own ``result_brief`` (the gate-facing slice) so the numbers a promotion/rejection
    turns on print exactly as they do in the conversation loop's feed."""
    if on_event is None:
        return _no_emit

    def emit_tool(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        brief = toolbox.result_brief(result) if isinstance(result, dict) else {}
        on_event(tool_event(name, args, result, brief))

    return emit_tool


def _entry_exit_brief(fo: FormulateOutput) -> str:
    """Compose the brief's precise-rules field from the formulate output (skeleton mapping; the
    author engine turns it into code). The prose ``symbol_character`` still frames the rules; MATCH
    (#69) additionally maps it onto a band profile to screen the fit set the brief tunes on."""
    return (
        f"Author precise long/short/flat rules that make this thesis falsifiable at the "
        f"{fo.timeframe} timeframe. Target symbol character: {fo.symbol_character}. The captured "
        f"move per trade must clear the round-trip cost: {fo.cost_arithmetic}."
    )


@dataclass(frozen=True)
class MatchResult:
    """The deterministic MATCH outcome for one thesis: the ``fit`` set AUTHOR/OPTIMIZE tune on, the
    ``reserved`` symbol-holdout names held out by code (never tuned; DECIDE's holdout nominees), the
    screened band ``profile``, and a ``fallback`` reason (``None`` on a real screen, a short string
    when the screen found no lake match and the composition-root panel was used instead)."""

    fit: list[str]
    reserved: list[str]
    profile: dict[str, str]
    fallback: str | None = None

    def ledger_detail(self) -> dict[str, Any]:
        return {
            "profile": dict(self.profile),
            "fit": list(self.fit),
            "reserved_holdout": list(self.reserved),
            "fallback": self.fallback,
        }


def _match_stage(
    toolbox: ResearchToolbox,
    fo: FormulateOutput,
    *,
    fallback_panel: Sequence[str],
    emit_tool: ToolEmitter = _no_emit,
) -> MatchResult:
    """Deterministic MATCH: screen the lake for the thesis's symbol character and reserve the
    symbol holdout — all in driver code, zero LLM. The band profile comes from
    :func:`character_to_profile`; the gated, budget-aware ``tool_screen_symbols`` (the same method
    the conversation loop used) ranks the lake and splits ``suggested_fit`` / ``reserved_holdout``
    by the configured sizes. A screen error or an empty match falls back to ``fallback_panel``
    (the composition-root fit set) with no reservation, so the toolbox's own out-of-fit holdout
    fallback still supplies a symbol holdout at verdict time (rule 4 stays honored)."""
    profile = character_to_profile(fo.symbol_character)
    screen_args = {
        "trend": profile["trend"],
        "volatility": profile["volatility"],
        "liquidity": profile["liquidity"],
    }
    screen = _invoke(toolbox.tool_screen_symbols, **screen_args)
    emit_tool("screen_symbols", screen_args, screen)
    if "error" in screen:
        return MatchResult(
            list(fallback_panel), [], profile, fallback=f"screen_error: {screen['error']}"
        )
    fit = [str(s) for s in (screen.get("suggested_fit") or [])]
    reserved = [str(s) for s in (screen.get("reserved_holdout") or [])]
    if not fit:
        return MatchResult(list(fallback_panel), [], profile, fallback="no_lake_match")
    return MatchResult(fit, reserved, profile)


# The AUTHOR episode-line outcomes. An ESCALATED authoring job (story #72) records whether the
# paid fallback authored the file (``ok``) or also failed the gate (``author_failed``). The
# needs-more-history case (#85) — a write rejected because the candidate's declared warmup is too
# large for the fixed oracle and the lookback defaults cannot honestly shrink — records
# ``refined_brief`` instead: the honest exit is a lighter thesis, so the next formulate round can
# propose one, never a bent gate or a mutated tape. A local (non-escalated) author that simply
# lands or fails generically records NO episode line, so the stream stays byte-identical to before
# unless an escalation OR a refined-brief exit earns one.
_AUTHOR_OK = "ok"
_AUTHOR_FAILED = "author_failed"
_AUTHOR_REFINED = "refined_brief"


def _record_author_outcome(
    ledger: SessionLedger, write: dict[str, Any], *, coder_model: str
) -> None:
    """Ledger one AUTHOR episode line for the two outcomes that earn one: an escalation to the paid
    fallback (story #72) or a refined-brief exit (#85). Everything else — a plain local success or
    a generic local gate rejection — records no line, so the episode stream is unchanged from
    before unless one of those two fired.

    ``tool_write_strategy`` marks an escalated write ``escalated=True`` and names the model that
    authored it (``author_model`` = the fallback model). A refined-brief exit is detected from the
    gate error via :func:`noctis.strategies.library.is_warmup_too_large`; it can ride either an
    escalated or a local write, so it takes precedence over the generic failure label. The line's
    model is the escalated fallback model when escalated, else the session's local coder."""
    escalated = bool(write.get("escalated"))
    errored = "error" in write
    refined = errored and library.is_warmup_too_large(str(write.get("error") or ""))
    if not escalated and not refined:
        return
    if refined:
        outcome = _AUTHOR_REFINED
    elif errored:
        outcome = _AUTHOR_FAILED
    else:
        outcome = _AUTHOR_OK
    ledger.record_episode(
        stage=AUTHOR,
        model=str(write.get("author_model") or coder_model or ""),
        outcome=outcome,
        escalated=escalated,
    )


def _brief_from_formulate(fo: FormulateOutput, symbols: Sequence[str]) -> dict[str, Any]:
    """Map a FORMULATE output onto the strategy author's brief (thesis, entry/exit, param space,
    scenarios). Passed to ``tool_write_strategy(brief=…, spec=…)`` — the coder author engine
    translates it into one validated file and the write gate (#84) owns the oracle. The
    ``scenarios`` field is the fixed oracle rendered faithfully from the FORMULATE
    ``scenario_spec`` (:func:`~noctis.strategies.scenario_spec.describe_spec` — tape shapes,
    behaviors, target legs), never free-prose scenario intent (#85): the same ``SpecSuite`` is
    threaded to the gate as ``spec`` so the coder authors no ``scenarios()`` block at all."""
    return {
        "thesis": fo.thesis,
        "entry_exit": _entry_exit_brief(fo),
        "param_space": fo.param_space_sketch,
        "scenarios": describe_spec(fo.scenario_spec),
        "style": fo.style,
        "symbols": list(symbols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZE — the v1 multi-fidelity tuning recipe (story #70), zero LLM
# ─────────────────────────────────────────────────────────────────────────────
# Hard cap on narrowed re-tune rounds — the recipe never spends more than two, however far the
# metric keeps climbing (AGENTS.md rule 2: search effort is bounded, the gates are the arbiter).
_MAX_RETUNES = 2

# Cheap-fidelity bar cap for the exploration sweep: it tunes on at most this many recent bars per
# symbol (a truncated recent window), so a broad first pass is cheap. Kept well above the toolbox's
# ``_MAX_BARS_FLOOR`` (walk-forward needs train+test+holdout room); on a shorter series it is a
# no-op (``tail`` returns the whole series). The full-panel confirm re-checks the best on all bars.
_CHEAP_MAX_BARS = 2000

# A re-tune must clear this relative bar over the previous round's confirm to earn another round —
# a scale-free "meaningful improvement" gate for a positive metric; a flat/negative metric only has
# to strictly increase (moving it up at all is progress the search earned). See :func:`_improved`.
_IMPROVE_MARGIN = 0.05


def _panel_metric(result: dict[str, Any]) -> float | None:
    """The panel test metric a backtest/confirm exposes (``avg_test_metric``), or ``None`` when the
    call errored or scored nothing — the same metric the journal ranks trials by, so baseline,
    confirm, and sweep numbers sit on one scale-free footing."""
    value = result.get("avg_test_metric")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _best_params(sweep: dict[str, Any]) -> dict[str, Any]:
    """The best (top-ranked) param set from a sweep return, or ``{}`` when it journaled none."""
    trials = sweep.get("top_trials") or []
    return dict(trials[0].get("params", {})) if trials else {}


def _improved(new: float | None, prev: float | None) -> bool:
    """Scale-free "meaningful improvement" of a panel test metric over the prior round's confirm.

    An un-scored new round never counts; a first round (no prior) always does. With a positive
    prior the new confirm must clear it by ``_IMPROVE_MARGIN`` (relative); with a flat/negative
    prior any strict increase counts, since nudging a losing metric upward is real search progress
    the recipe should let run one more round (still hard-capped)."""
    if new is None:
        return False
    if prev is None:
        return True
    if prev > 0:
        return new >= prev * (1.0 + _IMPROVE_MARGIN)
    return new > prev


def _narrow_ranges(best_params: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """A narrowed ``ranges`` request for a re-tune sweep — a neighborhood around each numeric best
    param (half its magnitude either side, positive ints floored at 1). Categorical/bool params
    have no numeric neighborhood and are skipped. Returns ``None`` when nothing numeric can be
    narrowed, which the recipe reads as "no re-tune to run" and stops honestly. The toolbox keeps
    each param's original kind/step/choices and only swaps the low/high we send (see
    ``_sweep_space``)."""
    ranges: dict[str, dict[str, Any]] = {}
    for pname, value in best_params.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            span_i = max(1, abs(value) // 2)
            low = value - span_i
            ranges[pname] = {"low": max(1, low) if value > 0 else low, "high": value + span_i}
        elif isinstance(value, float):
            span_f = abs(value) * 0.5 or 1.0
            ranges[pname] = {"low": value - span_f, "high": value + span_f}
    return ranges or None


def _cheap_subset(fit: Sequence[str]) -> list[str]:
    """The subset of the fit panel the cheap exploration sweep tunes on: the first half (rounded
    up, at least one). A strict subset for a panel of two or more — the other cheap-fidelity lever
    beside the truncated bar window; the confirm re-checks the best on the whole panel."""
    half = max(1, (len(fit) + 1) // 2)
    return list(fit[:half])


def _optimize_stage(
    toolbox: ResearchToolbox,
    name: str,
    fit_symbols: Sequence[str],
    *,
    sweep_trials: int | None,
    emit_tool: ToolEmitter = _no_emit,
) -> dict[str, Any]:
    """Run the v1 multi-fidelity OPTIMIZE recipe for one strategy and return the ledger detail.

    Zero LLM: every step is a gated ``toolbox`` call, journaled exactly as today. The shape is
    baseline → cheap subset sweep → full-panel confirm → ≤ 2 narrowed re-tune rounds, with two
    data-dependent branch axes (see the module docstring): the baseline's sign sizes the cheap
    sweep (promising vs weak — the weak branch still sweeps to feed the exhaustion floor, just
    smaller), and each confirm's improvement over the prior round earns or ends re-tuning. A budget
    refusal at any step stops the recipe and hands DECIDE whatever evidence the journal already
    holds — the gates dispose. Adds no gate and invents no metric; it only allocates search effort.
    """
    fit = list(fit_symbols)
    detail: dict[str, Any] = {
        "retune_rounds": 0,
        "sweeps": 0,  # completed sweeps (each journals its trials)
        "backtests": 0,  # completed backtests (each journals one trial)
        "trials": 0,  # trials journaled this recipe (backtests + sweep trials)
        "weak_baseline": False,
        "best_metric": None,
        "stopped": "complete",
    }

    def run_backtest(**kwargs: Any) -> dict[str, Any]:
        result = _invoke(toolbox.tool_run_backtest, **kwargs)
        emit_tool("run_backtest", kwargs, result)
        if "error" not in result:
            detail["backtests"] += 1
            detail["trials"] += 1
        return result

    def run_sweep(**kwargs: Any) -> dict[str, Any]:
        result = _invoke(toolbox.tool_run_sweep, **kwargs)
        emit_tool("run_sweep", kwargs, result)
        if "error" not in result:
            detail["sweeps"] += 1
            detail["trials"] += int(result.get("n_trials") or 0)
        return result

    def note_best(metric: float | None) -> None:
        if metric is not None and (detail["best_metric"] is None or metric > detail["best_metric"]):
            detail["best_metric"] = metric

    # 1. Baseline — the shipped defaults on the full fit panel.
    baseline = run_backtest(name=name, symbols=fit)
    if "error" in baseline:
        detail["stopped"] = "budget"
        return detail
    baseline_metric = _panel_metric(baseline)
    note_best(baseline_metric)

    # 2. Cheap exploration sweep — a subset of the panel at a truncated recent window. A weak
    #    baseline still sweeps (the floor needs trials) but is sized down (the interpret slot).
    weak = baseline_metric is None or baseline_metric <= 0.0
    detail["weak_baseline"] = weak
    base_trials = sweep_trials or int(getattr(toolbox, "default_sweep_trials", 0) or 0) or None
    cheap_trials = max(1, base_trials // 2) if (weak and base_trials) else base_trials
    sweep = run_sweep(
        name=name,
        symbols=_cheap_subset(fit),
        n_trials=cheap_trials,
        max_bars=_CHEAP_MAX_BARS,
    )
    if "error" in sweep:
        detail["stopped"] = "budget"
        return detail
    best_params = _best_params(sweep)

    # 3. Full-panel confirm of the cheap sweep's best.
    confirm = run_backtest(name=name, symbols=fit, params=best_params or None)
    if "error" in confirm:
        detail["stopped"] = "budget"
        return detail
    confirm_metric = _panel_metric(confirm)
    note_best(confirm_metric)

    # 4. Up to two narrowed re-tune rounds — each earned by a meaningful improvement, hard-capped.
    prev_metric = baseline_metric
    while detail["retune_rounds"] < _MAX_RETUNES:
        if not _improved(confirm_metric, prev_metric):
            detail["stopped"] = "stall"
            break
        ranges = _narrow_ranges(best_params)
        if ranges is None:
            detail["stopped"] = "no_narrowing"
            break
        retune = run_sweep(name=name, symbols=fit, ranges=ranges)
        if "error" in retune:
            detail["stopped"] = "budget"
            break
        detail["retune_rounds"] += 1
        best_params = _best_params(retune) or best_params
        prev_metric = confirm_metric
        confirm = run_backtest(name=name, symbols=fit, params=best_params or None)
        if "error" in confirm:
            detail["stopped"] = "budget"
            break
        confirm_metric = _panel_metric(confirm)
        note_best(confirm_metric)
    else:
        detail["stopped"] = "hard_cap"

    return detail


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
    market_digest: Callable[[], str] | None = None,
    on_event: Callable[[Event | str], None] | None = None,
) -> ResearchSummary:
    """Run one episodic research session; returns the same summary shape as the conversation loop.

    ``formulate`` / ``decide`` are the two model-judgment episodes (injected — the LLM client is
    behind them, never handed here); every other stage runs through the gated ``toolbox`` methods.
    ``completions`` returns the episode runner's per-completion count (retries included) that
    ``max_episodes`` budgets against; ``budget_minutes`` and ``stop_event`` bound wall-clock and
    interruption. ``fit_symbols`` is now the MATCH *fallback* panel — deterministic screening
    (:func:`_match_stage`) chooses the per-thesis fit set and reserves the symbol holdout, and only
    an empty screen falls back to ``fit_symbols``. ``market_digest`` supplies the MARKET ECONOMICS
    digest text the FORMULATE cost-arithmetic sanity check overlaps against — the same source the
    formulate briefing embeds; it defaults to the toolbox's own digest. See the module docstring
    for the stage protocol, the per-stage failed-episode policies, and the sanity checks.

    ``on_event`` (story #73) makes one observable episodic session read *better* than the
    conversation loop's inferred narration: a ``stage`` boundary Event opens each of
    FORMULATE/MATCH/AUTHOR/OPTIMIZE/DECIDE, the driver's deterministic toolbox actions emit the
    same ``tool`` lines the loop does, and the injected episodes (via the episode runner's own
    ``on_event``) tee the think/say/usage the model returned — so the whole arc interleaves on one
    sink. ``None`` (a bare run) ⇒ no emission, byte-identical to before.
    """
    stop_event = stop_event or _NeverStop()
    digest_source = market_digest or (lambda: digests.market_digest(toolbox))
    exhausted = getattr(toolbox, "exhausted", None)
    # The local coder model names the AUTHOR episode line for a non-escalated refined-brief exit
    # (#85); an escalated write names its own paid fallback model. Read off the session models map.
    coder_model = str((models or {}).get("coder") or "")
    summary = ResearchSummary()
    start = now()
    budget_seconds = budget_minutes * 60.0
    formulated = 0

    # Observability seam (#73): a `stage` boundary before each stage's work and a `tool` line per
    # deterministic toolbox action, both no-ops without a sink so a bare run is unchanged.
    emit_tool = _tool_emitter(on_event, toolbox)

    def emit_stage(stage: str, strategy: str | None = None) -> None:
        if on_event is not None:
            on_event(stage_event(stage, strategy))

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

    def _record_episode(
        stage: str,
        result: EpisodeResult[Any],
        checks: list[dict[str, Any]] | None = None,
    ) -> None:
        ledger.record_episode(
            stage=stage,
            model=result.model,
            outcome=result.outcome,
            tokens=result.tokens,
            misfires=result.misfires,
            checks=checks,
        )

    while True:
        stop = _budget_stop()
        if stop:
            summary.stopped_reason = stop
            break

        # ── FORMULATE (+ driver-side sanity checks, #71) ──────────────────────
        emit_stage(FORMULATE)
        ledger.record_stage(FORMULATE)
        fo = _formulate_stage(formulate, digest_source(), exhausted, _record_episode)
        if fo is None:
            # No usable thesis (a failed episode, or a sanity check the one re-ask never fixed);
            # end the session rather than busy-loop the budget.
            summary.stopped_reason = "formulate_failed"
            break
        formulated += 1
        name = f"{_slug(fo.class_tag) or 'strategy'}_{formulated}"
        ledger.record_thesis(
            name, fo.thesis, parent_thesis=fo.parent_thesis, pivot_rationale=fo.pivot_rationale
        )

        # ── MATCH (deterministic screening + symbol-holdout reservation, #69) ──
        # The boundary Event opens the stage before its screen runs; the ledger line lands after,
        # once the screen's profile/fit/reservation detail exists to carry.
        emit_stage(MATCH, name)
        match = _match_stage(toolbox, fo, fallback_panel=fit_symbols, emit_tool=emit_tool)
        ledger.record_stage(MATCH, strategy=name, detail=match.ledger_detail())
        symbols = match.fit  # AUTHOR/OPTIMIZE tune on the fit set ONLY
        reserved_holdout = match.reserved  # held out by code — never tuned, DECIDE's holdout

        # ── AUTHOR ────────────────────────────────────────────────────────────
        # The FORMULATE spec is the FIXED ORACLE: it is threaded into the write gate's spec path
        # (the coder authors no scenarios(); the gate stamps it) and the same suite renders the
        # brief's oracle summary (#85). The gate replays it at the candidate's own declared warmup.
        emit_stage(AUTHOR, name)
        ledger.record_stage(AUTHOR, strategy=name)
        write = _invoke(
            toolbox.tool_write_strategy,
            name=name,
            brief=_brief_from_formulate(fo, symbols),
            class_tag=fo.class_tag,
            thesis=fo.thesis,
            parent_thesis=fo.parent_thesis,
            pivot_rationale=fo.pivot_rationale,
            spec=fo.scenario_spec,
        )
        # The write's own coder-attempt `author` events already emitted from inside the toolbox;
        # this tool line is the write's outcome, mirroring the loop's post-dispatch line.
        emit_tool("write_strategy", {"name": name, "class_tag": fo.class_tag}, write)
        _record_author_outcome(ledger, write, coder_model=coder_model)
        if "error" in write:
            # A write-gate rejection is a code bug in one draft, not a verdict on the thesis — skip
            # and formulate the next idea. The one exception is the needs-more-history signal: a
            # declared warmup too large for the fixed oracle ended this strategy in a REFINED BRIEF
            # (recorded above), so the next formulate can propose a thesis needing less history.
            refined = library.is_warmup_too_large(str(write["error"]))
            verb = "refined-brief (needs less history)" if refined else "skipped"
            logger.info("author %s %s: %s", verb, name, write["error"])
            continue

        # ── OPTIMIZE (v1 multi-fidelity tuning recipe, #70 — zero LLM) ──────────
        emit_stage(OPTIMIZE, name)
        optimize_detail = _optimize_stage(
            toolbox, name, symbols, sweep_trials=sweep_trials, emit_tool=emit_tool
        )
        ledger.record_stage(OPTIMIZE, strategy=name, detail=optimize_detail)

        # ── DECIDE ────────────────────────────────────────────────────────────
        stop = _budget_stop()
        if stop:
            summary.stopped_reason = stop
            break  # authored + optimized but out of budget — left undecided, honestly
        emit_stage(DECIDE, name)
        ledger.record_stage(DECIDE, strategy=name)
        _decide_stage(
            toolbox,
            ledger,
            decide,
            name,
            symbols,
            reserved_holdout,
            _record_episode,
            emit_tool=emit_tool,
        )

    summary.iterations = formulated
    summary.promotions = int(getattr(toolbox, "promotions", 0))
    summary.rejections = int(getattr(toolbox, "rejections", 0))
    summary.candidates = list(getattr(toolbox, "strategies_touched", []))
    summary.author_calls = int(getattr(toolbox, "author_calls", 0))
    summary.escalations = int(getattr(toolbox, "escalations", 0))
    summary.undecided = sorted(getattr(toolbox, "undecided", set()))
    summary.ledger_path = str(ledger.path)
    ledger.record_session_end(
        formulated=formulated,
        promoted=summary.promotions,
        rejected=summary.rejections,
        note=summary.stopped_reason or None,
    )
    # A legible rollup at session end (story #74): the same at-a-glance numbers the CLOSE report
    # renders, derived from the ledger's typed records — theses, files authored, validation
    # failures, trials, verdicts by kind, undecided, escalations, tokens by stage and by model.
    rollup = ledger.rollup()
    # The one comparable spend axis (story #75): total judgment-model tokens this session, summed
    # off the ledger's per-episode token counts (the same four usage fields the conversation loop
    # totals), so the parity harness reads tokens/verdict off both loops honestly. Escalated
    # coder-authoring runs on a separate client and is excluded, as in the conversation loop.
    summary.tokens_total = sum(rollup.tokens_by_stage.values())
    logger.info("session rollup — %s", rollup.log_line())
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


def _formulate_stage(
    formulate: FormulateEpisode,
    digest_text: str,
    exhausted: Any,
    record_episode: Callable[..., None],
) -> FormulateOutput | None:
    """Run FORMULATE with the driver-side sanity checks (#71): propose a thesis, run the
    cost-arithmetic and exhausted-class checks, and on a failing check re-ask exactly once (naming
    what was wrong) before the FORMULATE failure policy applies (return ``None`` ⇒ end the
    session). A failed episode (no thesis at all) ends immediately with no sanity re-ask — a
    persistently misfiring model is not corrected by a check note. Each episode line is ledgered
    with the check that fired (``reask`` when it earned the one correction, ``exhausted`` when it
    fired again after it), so a reader sees which check fired and how the re-ask resolved."""
    corrective: str | None = None
    for attempt in range(_FORMULATE_ATTEMPTS):
        result = formulate(corrective=corrective)
        if not result.ok or result.value is None:
            record_episode(FORMULATE, result)
            return None
        fo = result.value
        check = _formulate_checks(fo, digest_text, exhausted)
        if check.ok:
            record_episode(FORMULATE, result)
            return fo
        last = attempt == _FORMULATE_ATTEMPTS - 1
        record_episode(
            FORMULATE, result, [{"check": check.check, "result": _EXHAUSTED if last else _REASK}]
        )
        if last:
            logger.info("formulate failed %s after the re-ask — ending session", check.check)
            return None
        logger.info("formulate failed the %s check — re-asking once", check.check)
        corrective = check.corrective
    return None  # pragma: no cover — the loop always returns within _FORMULATE_ATTEMPTS


def _decide_stage(
    toolbox: ResearchToolbox,
    ledger: SessionLedger,
    decide: DecideEpisode,
    name: str,
    symbols: Sequence[str],
    reserved_holdout: Sequence[str],
    record_episode: Callable[..., None],
    *,
    emit_tool: ToolEmitter = _no_emit,
) -> None:
    """Run DECIDE for one strategy: propose a verdict, submit it through the gated toolbox method,
    and on a failed episode, a refused verdict, or a (capped) ``revise`` re-ask exactly once before
    leaving the strategy undecided. The two-attempt cap is shared across all three causes — one
    corrective re-ask total, whatever fired it — so a ``revise`` earns the one re-ask (naming the
    cap) and a second ``revise`` for the same strategy exhausts it (undecided). ``reserved_holdout``
    is the MATCH-reserved symbol holdout the driver submits at verdict time — a code reservation the
    model proposal never overwrites."""
    corrective: str | None = None
    for attempt in range(_DECIDE_ATTEMPTS):
        result = decide(name, corrective=corrective)
        if not result.ok or result.value is None:
            record_episode(DECIDE, result)
            corrective = (
                f"The previous decide episode produced no valid verdict "
                f"({result.note or result.outcome}). Re-read the evidence and emit a verdict."
            )
            continue
        verdict = result.value
        if verdict.verdict == _REVISE:
            # `revise` is capped: the first earns the one corrective re-ask, a second exhausts it.
            last = attempt == _DECIDE_ATTEMPTS - 1
            record_episode(
                DECIDE, result, [{"check": _REVISE_CHECK, "result": _EXHAUSTED if last else _REASK}]
            )
            if last:
                logger.info("decide %s: revise cap reached — left undecided", name)
                return
            corrective = _REVISE_CORRECTIVE
            continue
        record_episode(DECIDE, result)
        outcome = _submit_verdict(
            toolbox, name, symbols, reserved_holdout, verdict, emit_tool=emit_tool
        )
        if "error" not in outcome:
            _record_verdict(ledger, name, verdict, outcome)
            return
        # The gate disposed of the proposal — re-ask once with the refusal as corrective context.
        logger.info("decide %s refused by the gate: %s", name, outcome["error"])
        corrective = str(outcome["error"])
    # Re-ask exhausted: leave the strategy undecided (the toolbox still holds it there).
    logger.info("decide %s left undecided after the re-ask", name)


def _submit_verdict(
    toolbox: ResearchToolbox,
    name: str,
    symbols: Sequence[str],
    reserved_holdout: Sequence[str],
    verdict: DecideOutput,
    *,
    emit_tool: ToolEmitter = _no_emit,
) -> dict[str, Any]:
    """Submit a proposed verdict through the matching gated toolbox method — the method, not the
    episode, disposes of it (min-trials floor + evidence checks refuse an unsupported verdict).

    For an approve, the symbol holdout is the MATCH-reserved names (``reserved_holdout``), not the
    model's nomination: the reservation was made in code at MATCH time and those names were kept
    out of every backtest/sweep, so they are the honest symbol holdout. A model nomination that
    disagrees is ignored (and logged). An empty reservation (MATCH fell back) submits no holdout,
    so the toolbox's own out-of-fit fallback supplies one.

    The verdict submission is the narrated action here (#73): its tool line — a reject or an
    evaluate-vs-champion — is emitted, so it is the last tool line the DECIDE stage prints. The
    internal ``get_experiment_log`` read that fetches the best params is plumbing, not an action,
    so it is not narrated."""
    if verdict.verdict == _REJECT:
        reject_args = {
            "name": name,
            "reason": verdict.reason,
            "class_tag": verdict.class_tag or None,
            "class_exhausted": verdict.class_exhausted,
        }
        result = _invoke(toolbox.tool_reject_strategy, **reject_args)
        emit_tool("reject_strategy", reject_args, result)
        return result
    # approve: challenge the champion board with the best-observed params from the journal.
    nominated = [s for s in verdict.holdout_symbols if s]
    if nominated and set(nominated) != set(reserved_holdout):
        logger.info(
            "decide %s: ignoring model-nominated holdout %s in favor of the MATCH reservation %s",
            name,
            nominated,
            list(reserved_holdout),
        )
    log = _invoke(toolbox.tool_get_experiment_log, name=name)
    trials = log.get("top_trials") or []
    params = trials[0].get("params", {}) if trials else {}
    evaluate_args = {
        "name": name,
        "symbols": list(symbols),
        "params": params,
        "holdout_symbols": list(reserved_holdout) or None,
    }
    result = _invoke(toolbox.tool_evaluate_vs_champion, **evaluate_args)
    emit_tool("evaluate_vs_champion", evaluate_args, result)
    return result


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
