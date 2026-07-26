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

**Errors are loud.** A refused key, an unknown key, or a value the owning section rejects
raises :class:`OverlayError` — one error listing *every* problem in the patch, so fixing a
bad overlay is one pass rather than a fix-one-rerun loop.
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
# The scoring metric — an operator's risk appetite — is the whole of the surface today.
# Widening it is a deliberate, owner-gated edit here, and the suite's sample-value ratchet
# refuses a widening that ships untested.
ALLOWED: frozenset[str] = frozenset({"promotion.metric"})


# ─────────────────────────────────────────────────────────────────────────────
# Tier B — CLAMPED: legal only in the direction that adds discipline.
# ─────────────────────────────────────────────────────────────────────────────
# Empty today; the direction clamps land with the widened surface.
CLAMPED: dict[str, Direction] = {}


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
    # Not yet in the surface — the model/provider seam, the spend and compatibility
    # ceilings, the prompt-shape levers, the data-acquisition window, and the two knobs
    # destined for direction clamps.
    "research.model": _NOT_YET,
    "research.base_url": _NOT_YET,
    "research.agent.model": _NOT_YET,
    "research.agent.coder_model": _NOT_YET,
    "research.agent.coder_fallback_model": _NOT_YET,
    "research.agent.thinking": _NOT_YET,
    "research.agent.coder_thinking": _NOT_YET,
    "research.agent.coder_fallback_thinking": _NOT_YET,
    "research.agent.loop": _NOT_YET,
    "research.cost_profile": _NOT_YET,
    "research.agent.max_iterations": _NOT_YET,
    "research.agent.max_backtests": _NOT_YET,
    "research.agent.sweep_trials": _NOT_YET,
    "research.agent.max_author_calls": _NOT_YET,
    "research.agent.max_escalations": _NOT_YET,
    "research.agent.max_tokens": _NOT_YET,
    "research.agent.coder_max_tokens": _NOT_YET,
    "research.agent.context_window": _NOT_YET,
    "research.agent.episode_retries": _NOT_YET,
    "research.agent.web_search": _NOT_YET,
    "research.agent.max_web_searches": _NOT_YET,
    "research.agent.sweep_workers": _NOT_YET,
    "research.agent.worker_bar_budget": _NOT_YET,
    "research_time_budget_minutes": _NOT_YET,
    "time_limit_hours": _NOT_YET,
    "universe": _NOT_YET,
    "research.focus_size": _NOT_YET,
    "research.tuning_dispersion_penalty": _NOT_YET,
    "research.draft_ttl_hours": _NOT_YET,
    "research.memory_distill_every": _NOT_YET,
    "data.history_days": _NOT_YET,
    "data.auto_backfill": _NOT_YET,
    "observability.heartbeat_polls": _NOT_YET,
    "qa.keep_last_runs": _NOT_YET,
    "research.min_trials": _NOT_YET,
    "data.budget_usd": _NOT_YET,
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
            reason=f"may only be {adds_discipline} relative to the configured value",
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

    Classifies **every** key first and raises one :class:`OverlayError` listing all the
    violations, so an operator sees every problem at once. Survivors are grouped by their
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
        verdict = classify(path)
        if verdict.tier == "allowed":
            survivors[path] = value
        else:
            violations.append(f"{path}: {verdict.reason}")
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
