"""Briefing builders for the v1 episodic stages — formulate and decide — with a build-time
fit assertion that replaces mid-session eviction.

The episodic research driver (epic #62) invokes the model only at narrow judgment points and
rebuilds each episode's prompt *fresh from disk* — there is no accumulated transcript to evict.
So the discipline that keeps a session inside its context window moves to *build time*: every
builder here renders the same facts the conversation loop shows (out of the shared digest
builders in :mod:`noctis.research.digests`) plus the session-ledger tail, then asserts the
rendered prompt fits the configured window. Nothing carries between calls: a builder is a pure
function of what is on disk when it runs.

The fit assertion trims only *advisory* blocks, in a fixed priority order — memory tail →
library stubs → digest breadth — and never touches a gate-facing number (the cost arithmetic,
the exhausted-class guard, the champion board, the ledger narrative, or, for decide, the
candidate's journaled trial evidence). A briefing that still does not fit after every advisory
block is trimmed fails loudly with :class:`BriefingTooLargeError` rather than truncating
silently: silent truncation of a gate-facing number is structurally impossible here.

Builders take explicit collaborators (the toolbox, the session ledger, an optional mandate) and
the window size as parameters — no Settings reads. Token size is estimated with the loop's own
provider-neutral ~4-chars/token heuristic (:func:`noctis.research.agent._estimate_tokens`), so
there is one token accounting across the codebase, not two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from noctis.research import digests
from noctis.research.agent import _estimate_tokens
from noctis.research.ledger import SessionLedger
from noctis.research.mandate import Mandate

# The fixed advisory-trim priority order: memory tail first, then the library stubs, then the
# per-symbol digest breadth. Everything NOT keyed here is core (gate-facing) and never trimmed.
_TRIM_ORDER = ("memory", "library", "breadth")

# How many top-ranked trials the decide evidence surfaces (mirrors get_experiment_log's cap).
_TOP_TRIALS = 10

_FORMULATE_HEADER = (
    "Propose ONE falsifiable strategy thesis for this session. Before any code, state the cost "
    "arithmetic against MARKET ECONOMICS below: the move captured per trade must clear the "
    "round-trip cost by a comfortable multiple (aim >=3x), which fixes the timeframe. Emit "
    "thesis, style, class_tag, timeframe, cost_arithmetic, symbol_character, scenario_spec, "
    "and param_space_sketch (plus parent_thesis / pivot_rationale when this pivots off an "
    "earlier idea). Do NOT re-propose an idea under ALREADY TRIED THIS SESSION, and do not "
    "re-mine a class under EXHAUSTED CLASSES.\n\n"
    "scenario_spec is a STRUCTURED known-outcome test suite — you reason about tape SHAPE and the "
    "behavior each tape must prove; you NEVER write a bar index (the compiler derives every window "
    'from the leg lengths and the strategy\'s warmup). Emit {"scenarios": [ ... ]} with 2-8 '
    "scenarios; each scenario is {name, legs, behavior, leg?}. A leg is {kind, bars, pct?, "
    "amplitude?, period?}: kind is one of flat / trend / selloff / recovery / chop / vol_spike / "
    "gap; bars is the leg's LENGTH in decision bars (0 for a gap); pct is the signed total move "
    "for trend/selloff/recovery/gap (0.05 = +5%); amplitude/period shape chop and vol_spike. "
    "behavior is exactly one of enter_long_during_leg / enter_short_during_leg / "
    "hold_long_through_leg / hold_short_through_leg / flat_by_end_of_leg / never_trade, and 'leg' "
    "is the 0-based index into that scenario's legs the behavior targets (omit it for "
    "never_trade). The suite MUST include at least one directional entry (enter/hold long/short) "
    "and at least one never_trade tape."
)


def _decide_header(strategy: str) -> str:
    return (
        f"Reach a verdict for strategy {strategy!r} from its gate-facing evidence below: approve "
        f"(challenge the champion board), reject (a class-level dead end recorded to memory), or "
        f"revise (a genuinely new lever worth another round). Emit verdict, reason, "
        f"class_exhausted, class_tag, holdout_symbols (profile-matching names NOT in the tuned "
        f"off-limits set), and new_lever when the class continues. The verdict is earned by the "
        f"journaled evidence — the min_trials gate and the holdout metrics are the arbiter."
    )


class BriefingTooLargeError(RuntimeError):
    """A briefing whose un-trimmable core still overflows the window after every advisory block
    is trimmed. Raised loudly so a driver never ships a silently-truncated gate-facing prompt."""


@dataclass(frozen=True)
class _Section:
    """One labeled briefing block. ``key`` names an advisory block for the trim order; a core
    (gate-facing) block uses a key absent from :data:`_TRIM_ORDER`, so it is never dropped."""

    key: str
    label: str
    body: str


def _json(obj: Any) -> str:
    """Deterministic, compact-but-readable serialization for a briefing block."""
    return json.dumps(obj, sort_keys=True, default=str)


def _tokens(text: str) -> int:
    """The loop's provider-neutral size estimate (~4 chars/token) — one shared accounting."""
    return _estimate_tokens(len(text), [])


def _mandate_body(mandate: Mandate | None) -> str:
    if mandate is None:
        return "(unconstrained — no operator mandate this session)"
    lines = [mandate.summary or mandate.text.strip()]
    if mandate.symbols:
        lines.append(f"declared symbols: {', '.join(mandate.symbols)}")
    return "\n".join(lines)


def _ledger_tail(ledger: SessionLedger) -> list[dict[str, Any]]:
    """The 'already tried this session' narrative: each journaled thesis paired with the verdict
    outcome and class-level lesson it earned, so the model never re-proposes what just failed.

    This is core narrative, not an advisory block — it participates in no trim level. If it (with
    the rest of the core) makes the briefing overflow, the builder fails loudly rather than
    silently dropping the session's own memory of what it already ruled out."""
    latest_verdict = {verdict.strategy: verdict for verdict in ledger.verdicts()}
    tail: list[dict[str, Any]] = []
    for thesis in ledger.theses():
        entry: dict[str, Any] = {"strategy": thesis.strategy, "thesis": thesis.text}
        if thesis.pivot_rationale:
            entry["pivot_rationale"] = thesis.pivot_rationale
        outcome = latest_verdict.get(thesis.strategy)
        if outcome is not None:
            entry["verdict"] = outcome.verdict
            if outcome.lesson:
                entry["lesson"] = outcome.lesson
        tail.append(entry)
    return tail


def _market_parts(toolbox: Any) -> tuple[str, str, str]:
    """Split the shared market digest into (cost-facts core, exhausted-classes core, breadth).

    The facts come from the one shared builder (:func:`digests.market_digest`) — parsed back so
    both loops render the same market numbers by construction. The per-symbol ``symbols`` breadth
    is the trimmable 'digest breadth' block; the cost arithmetic and the exhausted-class guard are
    gate-facing and stay in core."""
    digest = json.loads(digests.market_digest(toolbox))
    breadth = digest.pop("symbols", {})
    exhausted = digest.pop("exhausted_classes", [])
    return _json(digest), _json(exhausted), _json(breadth)


def _decide_evidence(toolbox: Any, strategy: str) -> dict[str, Any]:
    """The candidate's gate-facing evidence: exhaustion stats, the ranked top trials with their
    train/test/gap/holdout metrics, journaled verdicts, and the holdout-taint symbol set — the
    same digest ``get_experiment_log`` surfaces, so the decide episode reasons on identical
    numbers. Never trimmed."""
    journal = toolbox.journal
    stats = journal.stats(strategy)
    trials = journal.trials_by_test(strategy)[:_TOP_TRIALS]
    thesis = journal.thesis(strategy)
    return {
        "strategy": strategy,
        "thesis": thesis.text if thesis is not None else None,
        "class_tag": journal.class_tag(strategy),
        "n_trials": stats.n_trials,
        "n_distinct_params": stats.n_distinct_params,
        "sweep_completed": stats.sweep_completed,
        "min_trials_gate": toolbox.min_trials,
        "top_trials": [
            {
                "params": trial.params,
                "symbols": trial.symbols,
                "source": trial.source,
                **({"max_bars": trial.max_bars} if trial.max_bars else {}),
                **trial.metrics,
            }
            for trial in trials
        ],
        "verdicts": journal.verdicts(strategy),
        "tuned_off_limits_for_holdout": sorted(journal.touched_symbols(strategy)),
    }


def _render(sections: list[_Section], dropped: set[str]) -> str:
    return "\n\n".join(
        f"{section.label}:\n{section.body}" for section in sections if section.key not in dropped
    )


def _fit_or_raise(sections: list[_Section], *, window: int, kind: str) -> str:
    """Render ``sections`` within ``window`` tokens, dropping advisory blocks in the fixed trim
    order until it fits, and raising :class:`BriefingTooLargeError` if the core still overflows."""
    dropped: set[str] = set()
    text = _render(sections, dropped)
    for key in _TRIM_ORDER:
        if _tokens(text) <= window:
            return text
        dropped.add(key)
        text = _render(sections, dropped)
    if _tokens(text) > window:
        raise BriefingTooLargeError(
            f"{kind} briefing needs {_tokens(text)} tokens but the context window is {window}; "
            f"every advisory block (memory tail, library stubs, digest breadth) is already "
            f"trimmed and only gate-facing content remains. Widen research.agent.context_window "
            f"or reduce the gate-facing state — this is a loud failure by design, never a silent "
            f"truncation of a number the model must reason against."
        )
    return text


def formulate_briefing(
    toolbox: Any,
    ledger: SessionLedger,
    *,
    mandate: Mandate | None = None,
    context_window: int,
) -> str:
    """The FORMULATE episode briefing, rebuilt fresh from disk and asserted to fit the window.

    Embeds the mandate summary, the market cost arithmetic, the exhausted-class guard, the
    champion board, the session-ledger tail (what this session already tried and why it failed),
    and — as advisory, trimmable blocks — the distilled memory tail, the library index (rejected
    entries stubbed), and the per-symbol digest breadth. See the module docstring for the trim
    contract."""
    market_core, exhausted, breadth = _market_parts(toolbox)
    findings, dead_ends = digests.memory_block(toolbox.memory)
    sections = [
        _Section("header", "FORMULATE TASK", _FORMULATE_HEADER),
        _Section("mandate", "OPERATOR MANDATE", _mandate_body(mandate)),
        _Section("market", "MARKET ECONOMICS (cost arithmetic)", market_core),
        _Section("breadth", "MARKET BREADTH (per-symbol character)", breadth),
        _Section("exhausted", "EXHAUSTED CLASSES (do not re-mine)", exhausted),
        _Section("champions", "CHAMPION BOARD (beat the weakest)", _json(_champions(toolbox))),
        _Section("ledger", "ALREADY TRIED THIS SESSION", _json(_ledger_tail(ledger))),
        _Section(
            "memory", "MEMORY (advisory)", _json({"findings": findings, "dead_ends": dead_ends})
        ),
        _Section("library", "STRATEGY LIBRARY (rejected stubbed)", _json(_library(toolbox))),
    ]
    return _fit_or_raise(sections, window=context_window, kind="formulate")


def decide_briefing(
    toolbox: Any,
    ledger: SessionLedger,
    strategy: str,
    *,
    mandate: Mandate | None = None,
    context_window: int,
) -> str:
    """The DECIDE episode briefing for one ``strategy``, rebuilt fresh from disk and asserted to
    fit ``context_window``.

    Carries the candidate's gate-facing evidence (the ranked journaled trials/stats, the
    min_trials floor, journaled verdicts, the holdout-taint set), the market cost arithmetic, the
    exhausted-class guard, the champion board, and the ledger tail. Advisory, trimmable blocks are
    the distilled memory tail, the library index, and the per-symbol digest breadth."""
    market_core, exhausted, breadth = _market_parts(toolbox)
    findings, dead_ends = digests.memory_block(toolbox.memory)
    sections = [
        _Section("header", "DECIDE TASK", _decide_header(strategy)),
        _Section("mandate", "OPERATOR MANDATE", _mandate_body(mandate)),
        _Section(
            "evidence",
            f"EVIDENCE FOR {strategy} (gate-facing)",
            _json(_decide_evidence(toolbox, strategy)),
        ),
        _Section("market", "MARKET ECONOMICS (cost arithmetic)", market_core),
        _Section("breadth", "MARKET BREADTH (per-symbol character)", breadth),
        _Section("exhausted", "EXHAUSTED CLASSES (do not re-mine)", exhausted),
        _Section("champions", "CHAMPION BOARD (beat the weakest)", _json(_champions(toolbox))),
        _Section("ledger", "ALREADY TRIED THIS SESSION", _json(_ledger_tail(ledger))),
        _Section(
            "memory", "MEMORY (advisory)", _json({"findings": findings, "dead_ends": dead_ends})
        ),
        _Section("library", "STRATEGY LIBRARY (rejected stubbed)", _json(_library(toolbox))),
    ]
    return _fit_or_raise(sections, window=context_window, kind="decide")


def _champions(toolbox: Any) -> list[dict[str, Any]]:
    return digests.champion_digest(toolbox.registry)


def _library(toolbox: Any) -> list[dict[str, Any]]:
    return digests.library_index(toolbox.strategies_dir)
