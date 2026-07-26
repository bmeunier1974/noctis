"""The deny-by-default settings overlay — classify every knob, apply a patch by re-validating.

This module sits beside the safety gate because it is a statement about *settings*: it owns
no mandate, agent, or LLM concepts, so it is testable without a mandate file and reusable by
any caller that wants to overlay a flat ``{"dotted.path": value}`` patch onto a loaded
:class:`~noctis.config.settings.Settings`.

Two properties make it safe:

* **Completeness, not omission.** Every leaf dotted path in ``Settings`` is classified
  exactly once — :data:`ALLOWED` (tier A, run-shaping), :data:`CLAMPED` (tier B, legal in
  one direction only), or :data:`REFUSED` (tier C, with the reason that goes in the error).
  A field added to ``Settings`` tomorrow belongs to none of the three, so it classifies as
  *unknown* and is refused — and the suite's ratchet fails until someone classifies it
  deliberately. The old flat allowlist refused a new field only by accident of not being
  listed.
* **Validation from pydantic, not by hand.** :func:`apply_patch` groups surviving keys by
  their owning top-level section, deep-merges them into that section's dump, and rebuilds
  the section through ``model_validate``. The metric parser, the fill-cost floor, ``Literal``
  membership, and ``int | None`` coercion therefore all run exactly as they do for
  ``config.yaml`` — no per-knob hand checks, one place for the rules.

Re-validating *sections* (never the whole settings object) is deliberate: rebuilding the
root would re-read the environment, ``.env``, and the YAML file mid-assembly, and enabling
``validate_assignment`` globally would change behavior for every existing in-place
assignment in the codebase.

**Errors are loud.** A refused key, an unknown key, a clamped key moved the wrong way, or a
value the owning section rejects raises :class:`OverlayError` — one error listing *every*
problem in the patch, so fixing a bad overlay is one pass rather than a fix-one-rerun loop.
A clamp is checked against the value **config resolved** for that path (the live ``settings``
value), and the error names it, so the direction and the number to clear are both in the
message. Nothing is clipped silently: a mandate says what it means or it does not run.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from noctis.config.settings import Settings


class OverlayError(ValueError):
    """A patch that cannot be applied: a refused/unknown key, a wrong-direction clamp, or a
    value the owning config section's own validators reject. Callers catch this one type."""


Tier = Literal["allowed", "clamped", "refused", "unknown"]
Direction = Literal["raise_only", "lower_only"]


@dataclass(frozen=True)
class Verdict:
    """How the overlay treats one dotted settings path.

    ``tier`` is the classification; ``reason`` is the operator-facing "why not" carried into
    the error for everything but ``allowed``; ``direction`` is set for ``clamped`` paths only.
    """

    path: str
    tier: Tier
    reason: str = ""
    direction: Direction | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Tier A — ALLOWED: run-shaping knobs with no gate contact.
# ─────────────────────────────────────────────────────────────────────────────
# A mandate configures the whole *run* — which model thinks, what it may spend, how big the
# prompt gets, what "good" is scored as — while ``config.yaml`` keeps the *arena*: the safety
# mode, the fill-cost floor, the promotion thresholds, the holdout geometry, the paths, the
# secrets. Every path below is one of four kinds and says which in its own comment:
# a **compatibility lever** (fit the backend the operator actually has), a **spend ceiling**
# (bound what one session may burn), a **prompt-size lever** (shape what the model is told),
# or **housekeeping** (display + retention). None of them is read by the promotion gates
# (``champions/promotion.py``), the split geometry (``backtest/splits.py``), or the safety
# gate (``config/gate.py``) — that is the property that makes them settable at all.
#
# Widening this set is a deliberate, owner-gated edit here, and the suite's sample-value
# ratchet refuses a widening that ships untested.

# Model / provider seam — WHICH model runs the session and where it is reached. Pure
# compatibility levers: a mandate written for a local 30B coder and one written for a hosted
# frontier model are different research *personalities*, and the personality file is the
# honest place to say which brain it needs. A model choice cannot reach a gate — every
# candidate it produces is scored by the same pipeline and judged by the same thresholds.
_MODEL_SEAM: frozenset[str] = frozenset(
    {
        # The LiteLLM ``provider/model`` string the research driver runs on.
        "research.model",
        # The endpoint override for OpenAI-compatible/local backends (vLLM, Ollama, a proxy).
        "research.base_url",
        # The fallback driver model used when ``research.model`` is unset.
        "research.agent.model",
        # The dedicated authoring ("coder") model write_strategy briefs are sent to.
        "research.agent.coder_model",
        # The paid coder-fallback model a spent local author escalates to.
        "research.agent.coder_fallback_model",
        # The driver's provider-native reasoning dial (a watch-session switch).
        "research.agent.thinking",
        # The local coder's own reasoning dial — sized to the coder, not to the driver.
        "research.agent.coder_thinking",
        # The escalated coder's reasoning dial — sized to the strong fallback model.
        "research.agent.coder_fallback_thinking",
        # Which research loop drives the session (conversation transcript vs. episodic
        # driver). A backend-shape choice — the episodic driver exists for small-context
        # models — and both loops return the same ResearchSummary through the same tools.
        "research.agent.loop",
    }
)

# Spend + compatibility ceilings — HOW MUCH one session may burn, and how big a single
# request may get. Ceilings only bound resources; none of them decides whether a candidate is
# good. The exhaustion floor (``research.min_trials``) is deliberately NOT here: that one is
# the discipline rule, not a ceiling, so it belongs to the direction-clamped tier.
_SPEND_CEILINGS: frozenset[str] = frozenset(
    {
        # The engine-level cost knob that scales every Class-B budget together.
        "research.cost_profile",
        # Tool-use rounds / episodes per session.
        "research.agent.max_iterations",
        # Backtest calls + individual sweep trials per session.
        "research.agent.max_backtests",
        # Default Optuna trials for one run_sweep call (a per-call ceiling, never a floor:
        # the exhaustion gate still counts journaled param sets before any verdict).
        "research.agent.sweep_trials",
        # Coder completions per session — the authoring bill.
        "research.agent.max_author_calls",
        # How many failed local authoring attempts may reach the paid fallback model.
        "research.agent.max_escalations",
        # The driver's per-completion output ceiling — a compatibility lever for backends
        # that bound prompt+max_tokens by the context window.
        "research.agent.max_tokens",
        # The coder's own output ceiling — the strategy file's budget.
        "research.agent.coder_max_tokens",
        # The whole-request context budget: a compatibility lever for small-context backends.
        # Everything the loop evicts stays re-fetchable through the same tools (the on-disk
        # experiment journal is the ground truth), so no gate can be affected.
        "research.agent.context_window",
        # Corrective retries per episode when the model misfires — a robustness ceiling.
        "research.agent.episode_retries",
        # Server-side web-search grounding during FORMULATE/MATCH, and its per-session cap:
        # latency + tool-use spend, and grounding never bypasses a gate (a searched-up idea
        # is validated exactly like an invented one).
        "research.agent.web_search",
        "research.agent.max_web_searches",
        # Worker processes for parallel evaluation, and the memory guard that scales them
        # down. Compute shape only — results are identical whatever the worker count.
        "research.agent.sweep_workers",
        "research.agent.worker_bar_budget",
        # Wall-clock ceiling on one research phase.
        "research_time_budget_minutes",
        # Wall-clock ceiling on the whole run (the loop's global stop).
        "time_limit_hours",
    }
)

# Search shape — what the session is pointed at and how big its prompt gets. These are the
# knobs that make a mandate a *search prior* rather than a preference: the metric is the
# operator's risk appetite, and the rest shape the prompt and the tuning objective.
_SEARCH_SHAPE: frozenset[str] = frozenset(
    {
        # The scoring metric — the operator's risk appetite, and the surface's original knob.
        # It reinterprets the promotion thresholds' *units*; it never loosens one (a champion
        # scored under a different metric is treated as stale, not as comparable).
        "promotion.metric",
        # The cap on the symbols enumerated into each session's prompt. A prompt-size lever
        # only: symbols beyond the cap stay tradeable and re-fetchable through the tools.
        "research.focus_size",
        # The λ subtracted (× cross-symbol dispersion) from the *tuning* objective only —
        # never from the election score, so it shapes parameters, not champion selection.
        "research.tuning_dispersion_penalty",
        # How long an abandoned working-tier draft lingers before it is archived. Pure
        # housekeeping over bytes — it never touches a verdict.
        "research.draft_ttl_hours",
        # The memory-distillation cadence (a CLOSE-phase prompt-shaping step).
        "research.memory_distill_every",
    }
)

# Data acquisition — the lookback window a mandate's own declared symbols are fetched over,
# and whether a run backfills them at all. A mandate that names symbols is the thing that
# knows how much history its thesis needs. The *spend* side of data stays out of tier A:
# ``budget_usd`` is direction-clamped below, and ``lake_dir`` is state redirection.
_DATA_ACQUISITION: frozenset[str] = frozenset(
    {
        # The one-time backfill's lookback window for not-yet-ready symbols.
        "data.history_days",
        # Whether that backfill runs at all (still budget-gated by ``data.budget_usd``).
        "data.auto_backfill",
    }
)

# Display / housekeeping — read by no decision path at all.
_HOUSEKEEPING: frozenset[str] = frozenset(
    {
        # Live TRADING heartbeat cadence for an unattended run's verbose feed.
        "observability.heartbeat_polls",
        # How many --debug QA run folders survive prune-on-start.
        "qa.keep_last_runs",
    }
)

ALLOWED: frozenset[str] = (
    _MODEL_SEAM | _SPEND_CEILINGS | _SEARCH_SHAPE | _DATA_ACQUISITION | _HOUSEKEEPING
)


# ─────────────────────────────────────────────────────────────────────────────
# Tier B — CLAMPED: legal only in the direction that adds discipline.
# ─────────────────────────────────────────────────────────────────────────────
# Two knobs a mandate may move, but only *towards* discipline — the direction IS the guard
# that makes them settable at all, so a wrong-direction value is fatal rather than clipped
# (silently clamping a number would let a mandate say one thing while the run does another).
#
# The bound is whatever config resolved for the path at overlay time (``config.yaml``, ``.env``
# and the environment together), so a clamp constrains **the overlay and nothing else**: the
# config file's own semantics are untouched, and an operator who wants a lower exhaustion floor
# or a bigger vendor budget still sets it there. A second overlay in the same process therefore
# ratchets from the first one's result.
#
# ``None`` — "no bound": an unlimited budget, no exhaustion floor at all — is the
# least-disciplined end of *either* scale, so an overlay may replace it with a number and
# never the reverse.
CLAMPED: dict[str, Direction] = {
    # The exhaustion floor: how many distinct param sets must reach the experiment journal
    # before a verdict tool will answer. A count, not a metric — raising it demands strictly
    # more evidence per verdict (a tune-first personality asking for 20 is legitimate
    # steering), and lowering it is exactly the loosening AGENTS.md rule 5 forbids.
    "research.min_trials": "raise_only",
    # The vendor spend cap the data preflight bounds every fetch against. A mandate may spend
    # less of the operator's money; the ceiling config set stays the ceiling.
    "data.budget_usd": "lower_only",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tier C — REFUSED: fatal, with the reason in the error.
# ─────────────────────────────────────────────────────────────────────────────
_LIVE_MONEY = (
    "live-money double gate — real orders are reachable only when the config mode and the "
    "ALLOW_LIVE environment gate are independently open, so neither may come from an overlay"
)
_COST_FLOOR = (
    "arena difficulty — the enforced fill-cost floor; an overlay steers what to look for, "
    "never how forgiving the arena is"
)
_PROMOTION_GATES = (
    "promotion gates + metric robustness — the thresholds are read in the units of "
    "promotion.metric, so they are not comparable across a metric change and an overlay can "
    "never loosen them"
)
_HOLDOUT_GEOMETRY = (
    "two-axis holdout geometry — the fit-set/symbol-holdout split is the out-of-sample "
    "guarantee; resizing it defeats a gate instead of clearing it"
)
_BOARD_SIZE = (
    "board size — the number of champion slots changes how easy it is to beat the weakest champion"
)
_STATE_IO = (
    "state / IO redirection — moving an output root moves the experiment journal the "
    "exhaustion gate counts, the champion board, and the strategy tiers; that is structural, "
    "not a setting"
)
_SECRETS = "secrets live in the environment only and never travel in a config overlay"
_SELF_SELECTION = (
    "self-selection / recursion — an overlay may not choose which overlay is read or which "
    "research engine runs"
)
_TRADING_LOOP = "trading loop, not research — an overlay steers research only"
_VENDOR = (
    "vendor + feed infrastructure — the lake and every catalog are built under one vendor "
    "and dataset"
)
_SESSION_CLOCK = (
    "session clock — the lake and every catalog are aligned to one calendar and timezone"
)
_IDEATION = "legacy ideation path — deferred, and not run-shaping"

# Everything destined for tiers A/B in a later stage: refused today, because the surface is
# deny-by-default and a knob becomes settable only when it is deliberately classified.
_NOT_YET = (
    "not yet in the overlay surface — the overlay is deny-by-default, so a knob is settable "
    "only once it has been classified as run-shaping"
)

REFUSED: dict[str, str] = {
    # Live-money double gate.
    "mode": _LIVE_MONEY,
    "allow_live": _LIVE_MONEY,
    # Arena difficulty — the enforced cost floor.
    "backtest.fee_bps": _COST_FLOOR,
    "backtest.slippage_bps": _COST_FLOOR,
    # Promotion gates + metric robustness (every promotion.* except `metric`).
    "promotion.max_gap": _PROMOTION_GATES,
    "promotion.min_test_metric": _PROMOTION_GATES,
    "promotion.min_holdout_metric": _PROMOTION_GATES,
    "promotion.min_symbol_holdout_metric": _PROMOTION_GATES,
    "promotion.min_symbol_consistency": _PROMOTION_GATES,
    "promotion.min_test_activity": _PROMOTION_GATES,
    "promotion.annualization_cap": _PROMOTION_GATES,
    "promotion.max_period_ratio": _PROMOTION_GATES,
    "promotion.max_reverse_gap": _PROMOTION_GATES,
    "promotion.max_test_metric": _PROMOTION_GATES,
    # Two-axis holdout geometry.
    "research.fit_set_size": _HOLDOUT_GEOMETRY,
    "research.symbol_holdout_size": _HOLDOUT_GEOMETRY,
    # Board size.
    "champion_count": _BOARD_SIZE,
    # State / IO redirection.
    "workspace_dir": _STATE_IO,
    "state_dir": _STATE_IO,
    "reports_dir": _STATE_IO,
    "memory_path": _STATE_IO,
    "qa_dir": _STATE_IO,
    "strategies_dir": _STATE_IO,
    "mandate_dir": _STATE_IO,
    "data.lake_dir": _STATE_IO,
    # Secrets.
    "databento_api_key": _SECRETS,
    "anthropic_api_key": _SECRETS,
    "openai_api_key": _SECRETS,
    # Self-selection / recursion.
    "research.mandate": _SELF_SELECTION,
    "research.mode": _SELF_SELECTION,
    # Trading loop, not research.
    "risk.max_position_pct": _TRADING_LOOP,
    "risk.max_gross_exposure_pct": _TRADING_LOOP,
    "risk.max_daily_loss_pct": _TRADING_LOOP,
    "trading.max_catchup_sessions": _TRADING_LOOP,
    "trading.min_order_notional": _TRADING_LOOP,
    "trading.rebalance_band_pct": _TRADING_LOOP,
    "trading.execution": _TRADING_LOOP,
    "live_feed.poll_interval_s": _TRADING_LOOP,
    # Vendor + feed infrastructure.
    "data.provider": _VENDOR,
    "data.dataset": _VENDOR,
    # Session clock.
    "session.calendar": _SESSION_CLOCK,
    "session.timezone": _SESSION_CLOCK,
    # Legacy ideation path.
    "ideation.enabled": _IDEATION,
    "ideation.specs_per_round": _IDEATION,
    "ideation.cadence": _IDEATION,
    "ideation.model": _IDEATION,
    "ideation.max_indicators": _IDEATION,
    "ideation.web_search": _IDEATION,
    "ideation.max_web_searches": _IDEATION,
    # Not yet in the surface — run-shaping in kind, but carrying a guard that does not exist
    # yet, and a knob without its guard is a hole: shrinking ``universe`` below the fit-set +
    # symbol-holdout sizes would starve the symbol-holdout axis, defeating a gate by emptying
    # it rather than by clearing it, so it lands with its starvation guard.
    "universe": _NOT_YET,
}

_UNKNOWN = "not a setting — check the spelling and the dotted path against config.example.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────
def classify(path: str) -> Verdict:
    """Classify one dotted settings path. Total over *any* string: a path the tables don't
    carry is ``unknown`` (and therefore refused), never silently permitted."""
    if path in ALLOWED:
        return Verdict(path=path, tier="allowed")
    direction = CLAMPED.get(path)
    if direction is not None:
        adds_discipline = "raised" if direction == "raise_only" else "lowered"
        return Verdict(
            path=path,
            tier="clamped",
            reason=f"may only be {adds_discipline} by an overlay",
            direction=direction,
        )
    reason = REFUSED.get(path)
    if reason is not None:
        return Verdict(path=path, tier="refused", reason=reason)
    return Verdict(path=path, tier="unknown", reason=_UNKNOWN)


# ─────────────────────────────────────────────────────────────────────────────
# Applying a patch
# ─────────────────────────────────────────────────────────────────────────────
def apply_patch(settings: Settings, patch: Mapping[str, object]) -> list[str]:
    """Apply a flat ``{"dotted.path": value}`` patch to ``settings`` in place.

    Classifies **every** key first — and direction-checks every :data:`CLAMPED` one against
    the value ``settings`` currently holds — then raises one :class:`OverlayError` listing all
    the violations, so an operator sees every problem at once. Survivors (allowed keys and
    clamped keys moving the legal way, treated identically from here) are grouped by their
    owning top-level section, deep-merged into that section's dump, and re-validated through
    ``model_validate`` (top-level scalars go through a ``TypeAdapter`` over the field's own
    annotation), so values are checked by exactly the validators ``config.yaml`` gets. Any
    pydantic ``ValidationError`` is re-raised as an :class:`OverlayError`.

    Returns the sorted ``"path=value"`` echo lines for what was applied, read back off the
    validated objects so the echo shows the value the run will actually use.
    """
    violations: list[str] = []
    survivors: dict[str, object] = {}
    for path, value in patch.items():
        violation = _violation(settings, classify(path), value)
        if violation is None:
            survivors[path] = value
        else:
            violations.append(violation)
    if violations:
        raise OverlayError(_refusal_message(violations))
    if not survivors:
        return []

    sections = _sections()
    grouped: dict[str | None, dict[str, object]] = {}
    for path, value in survivors.items():
        head = path.split(".", 1)[0]
        grouped.setdefault(head if head in sections else None, {})[path] = value

    # Build every replacement before assigning any of it: a patch that fails validation must
    # leave the settings object exactly as it found it.
    rebuilt: dict[str, Any] = {}
    for section, entries in grouped.items():
        if section is None:
            rebuilt.update(_validated_scalars(entries))
        else:
            rebuilt[section] = _validated_section(settings, section, entries)
    for name, value in rebuilt.items():
        setattr(settings, name, value)
    return sorted(f"{path}={_read_path(settings, path)}" for path in survivors)


def _violation(settings: Settings, verdict: Verdict, value: object) -> str | None:
    """The one error line for a key that may not be applied, or ``None`` when it may."""
    if verdict.tier == "allowed":
        return None
    if verdict.tier == "clamped" and verdict.direction is not None:
        return _clamp_violation(settings, verdict, verdict.direction, value)
    return f"{verdict.path}: {verdict.reason}"


def _clamp_violation(
    settings: Settings, verdict: Verdict, direction: Direction, value: object
) -> str | None:
    """Direction-check one :data:`CLAMPED` value, naming the config value it tried to cross.

    The comparison is against the live ``settings`` value — what ``config.yaml``, ``.env``,
    and the environment settled on — so the clamp constrains the overlay and nothing else.
    A value that cannot be read as a number (on either side) is not a direction question at
    all: it is left to the section rebuild, where pydantic gives the honest type diagnosis.
    """
    proposed = _discipline_rank(value, direction)
    if proposed is None:
        return None
    configured = _read_path(settings, verdict.path)
    current = _discipline_rank(configured, direction)
    if current is None or proposed >= current:
        return None
    crossed = "below" if direction == "raise_only" else "above"
    return (
        f"{verdict.path}: {verdict.reason} — "
        f"{_show(value)} is {crossed} the configured {_show(configured)}"
    )


def _discipline_rank(value: object, direction: Direction) -> float | None:
    """Where ``value`` sits on the path's *discipline* scale (higher is stricter), or ``None``
    when it is not a number the clamp can read.

    ``None`` is "no bound" — an unlimited budget, no exhaustion floor — which is the
    least-disciplined end of either scale, hence ``-inf`` whichever way the clamp points.
    """
    if value is None:
        return float("-inf")
    number = _as_number(value)
    if number is None:
        return None
    return number if direction == "raise_only" else -number


def _as_number(value: object) -> float | None:
    """``value`` as the number the owning section would coerce it to, or ``None`` if it isn't.

    As permissive as pydantic's lax mode on purpose: a mandate's YAML front matter can hand
    over a quoted ``"2"`` that validates into ``2``, so the clamp has to read it as ``2`` too
    — otherwise the direction check would have a hole one quote character wide.
    """
    if isinstance(value, bool | int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _show(value: object) -> str:
    """A value as it reads in an operator-facing message (``None`` says what it means)."""
    return "null (no bound)" if value is None else repr(value)


def _sections() -> frozenset[str]:
    """The top-level ``Settings`` fields that are themselves config models."""
    return frozenset(
        name
        for name, field in Settings.model_fields.items()
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
    )


def _validated_section(settings: Settings, section: str, entries: Mapping[str, object]) -> Any:
    """Rebuild one config section with ``entries`` deep-merged into its current dump."""
    current = getattr(settings, section)
    data = current.model_dump()
    for path, value in entries.items():
        _deep_set(data, path.split(".")[1:], value)
    try:
        return type(current).model_validate(data)
    except ValidationError as exc:
        raise OverlayError(_invalid_message(sorted(entries), exc)) from exc


def _validated_scalars(entries: Mapping[str, object]) -> dict[str, Any]:
    """Validate top-level scalars (no owning section) against their own annotations."""
    out: dict[str, Any] = {}
    for path, value in entries.items():
        annotation = Settings.model_fields[path].annotation
        try:
            out[path] = TypeAdapter(annotation).validate_python(value)
        except ValidationError as exc:
            raise OverlayError(_invalid_message([path], exc)) from exc
    return out


def _deep_set(data: dict[str, Any], parts: list[str], value: object) -> None:
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _read_path(settings: Any, path: str) -> Any:
    node = settings
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _refusal_message(violations: list[str]) -> str:
    count = len(violations)
    head = f"{count} config override{'' if count == 1 else 's'} refused"
    return "\n".join([f"{head}:", *(f"  - {line}" for line in sorted(violations))])


def _invalid_message(paths: list[str], exc: ValidationError) -> str:
    return f"invalid config override for {', '.join(paths)}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# The gate-unmoved assertion
# ─────────────────────────────────────────────────────────────────────────────
def gate_snapshot(settings: Settings) -> dict[str, Any]:
    """Dump the whole :data:`REFUSED` subtree of ``settings``.

    Derived from the refusal table itself, so classifying a new path as refused extends the
    assertion automatically. Deep-copied, so a later in-place mutation of a list value can
    never make a snapshot agree with itself by aliasing.
    """
    return {path: copy.deepcopy(_read_path(settings, path)) for path in sorted(REFUSED)}


def assert_gates_unmoved(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    """Raise :class:`OverlayError` if any refused setting differs between two snapshots.

    The message names the moved paths but deliberately **not** their values: the refused
    subtree carries the API keys, and a diagnostic is no place for a credential.
    """
    moved = sorted(
        path
        for path in set(before) | set(after)
        if path not in before or path not in after or before[path] != after[path]
    )
    if moved:
        raise OverlayError(
            "refused settings moved during an overlay — this is a bug in the overlay "
            f"allowlist, not operator input: {', '.join(moved)}"
        )
