"""Typed application configuration.

Layered sources, highest priority first: constructor args, process environment, ``.env``,
then ``config.yaml``. So operational knobs come from ``config.yaml`` while secrets and
overrides come from the environment (environment always wins over the YAML file).

One layer sits *above* every source built here, and is applied by the composition root to the
object this module hands back: the active mandate's ``config:`` overlay
(:mod:`noctis.config.overlay`). The effective chain is therefore::

    CLI flags > mandate overlay > environment > .env > config.yaml > built-in defaults

For the run-shaping subset a mandate may bind, that **inverts** the environment-beats-YAML rule
above. It is deliberate: a mandate is a per-run *selection* an operator makes on purpose, not
ambient environment, and pinning one is meant to configure the run. Secrets and the
``ALLOW_LIVE`` gate are refused by the overlay, so the environment stays their only source.

Point at an alternate YAML file with the ``NOCTIS_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "SPY",
    "QQQ",
]


class SessionConfig(BaseModel):
    """Market session / clock configuration."""

    calendar: str = "XNYS"
    timezone: str = "America/New_York"


# The shipped baseline — 1bp fee + 1bp slippage per side (a 4bp round trip) — is also the
# enforced minimum. See :class:`BacktestConfig` for why the knob may only be raised.
_COST_FLOOR_BPS = 1.0


class BacktestConfig(BaseModel):
    """Simulated fill costs — the ONE source every consumer shares.

    ``fee_bps`` and ``slippage_bps`` are charged **per side** (enter and exit each pay), so
    the round-trip cost the research agent reasons about is ``2 × (fee_bps + slippage_bps)``.
    The single value is threaded from the composition root into the coarse pre-filter,
    walk-forward validation, the agent's cost hint, and the paper-fill broker, so those four
    can never disagree on what a trade costs.

    **The floor is load-bearing.** The cost model is the system's main difficulty knob:
    dialing it below the shipped baseline is the cheapest way to manufacture champions that
    would die on real fills, so the knob may only make the world *harsher* (or per-venue
    realistic), never cheaper than the baseline. A value below ``_COST_FLOOR_BPS`` is a hard
    startup error (like ``mode: live`` without ``ALLOW_LIVE``), never a silent clamp. This
    whole section is deliberately **refused by the mandate overlay**
    (:mod:`noctis.config.overlay`) — a research personality steers what to look for, never how
    forgiving the arena is.
    """

    fee_bps: float = _COST_FLOOR_BPS
    slippage_bps: float = _COST_FLOOR_BPS

    @field_validator("fee_bps", "slippage_bps")
    @classmethod
    def _enforce_floor(cls, value: float, info) -> float:
        if value < _COST_FLOOR_BPS:
            raise ValueError(
                f"backtest.{info.field_name}={value} is below the enforced minimum of "
                f"{_COST_FLOOR_BPS} bp per side (the shipped baseline). The simulated cost "
                f"model is the system's main difficulty knob; it may only be raised toward "
                f"per-venue realism, never lowered below the baseline (that is overfitting "
                f"with extra steps). Set backtest.{info.field_name} to at least "
                f"{_COST_FLOOR_BPS}."
            )
        return value


class RiskConfig(BaseModel):
    """Risk limits enforced in the TRADING loop (percent of account equity)."""

    max_position_pct: float = 10.0
    max_gross_exposure_pct: float = 100.0
    max_daily_loss_pct: float = 3.0


class TradingConfig(BaseModel):
    """TRADING-phase catalog replay (the rolling live-holdout).

    Each day the replay trades only the newest lake session(s) past the persisted
    high-water mark (``state/trading_sessions.json``) — one risk-managed session per
    session date, so the "daily" loss limit stays daily. There is deliberately no knob to
    opt out of slicing: replaying the full catalog would just resurrect the in-sample
    replay bug.
    """

    # Cap on unseen sessions replayed in one TRADING phase after downtime (chronological,
    # newest kept). Sessions truncated by the cap are reported explicitly, so a first run
    # after a long gap never re-becomes "replay all of history".
    max_catchup_sessions: int = 5

    # Rebalance dead-band (live-holdout plan 3). A held, same-direction position is re-trued
    # only when the drift clears one of these; opens, exits, and flips always execute. This
    # quiets the sub-share dust a held champion would otherwise emit nearly every bar as
    # equity/price drift. Both default 0.0 = off = today's every-bar re-truing. Applied in the
    # live/replay session ONLY, never in the backtest fills path, so no scorecard or gate moves.
    min_order_notional: float = 0.0  # skip same-direction adjustments below this $ notional
    rebalance_band_pct: float = 0.0  # skip same-direction adjustments below this % of target

    # Which TRADING driver runs (live-holdout plan 4). ``auto`` (default) derives from
    # ``data.provider`` — yfinance → live feed, anything else → catalog replay (today's
    # behavior). ``replay`` forces catalog replay even under ``data.provider: yfinance``
    # (offline live-holdout testing). ``live`` declares intent to stream; if the feed can't be
    # built the day still falls back to replay, but the mismatch is logged at WARNING, never
    # silent. Selects only WHICH driver runs — never whether real orders are reachable (the two
    # live-money gates, ``mode: live`` + ``ALLOW_LIVE``, are untouched).
    execution: Literal["auto", "replay", "live"] = "auto"


class DataConfig(BaseModel):
    """Market-data lake configuration.

    ``provider`` also selects the TRADING-phase live data source: ``yfinance`` opts in to the
    free, ~15-min-delayed Yahoo Finance feed (closed intraday bars, no credentials), while
    anything else keeps TRADING on offline catalog replay. A bare run therefore never contacts
    a live feed. (The research/backtest lake is a separate seam and still ingests from DataBento
    via ``DATABENTO_API_KEY`` regardless of this setting.)
    """

    provider: str = "databento"
    budget_usd: float = 125.0
    dataset: str = "EQUS.MINI"
    lake_dir: str = "data_lake/"
    # Opt-in: when true, ``run`` backfills missing history for any not-yet-ready universe
    # symbol before entering the loop (budget-gated; off by default so a bare run fetches
    # nothing). ``history_days`` is the lookback window for that one-time backfill.
    auto_backfill: bool = False
    history_days: int = 365


class LiveFeedConfig(BaseModel):
    """Live-feed loop pacing (only used when ``data.provider: yfinance``)."""

    # Seconds between streaming-loop polls. The yfinance feed self-throttles its actual Yahoo
    # fetches, so this is only how often the loop checks for a newly-closed bar to act on.
    poll_interval_s: float = 2.0


class AgentResearchConfig(BaseModel):
    """The agent-driven research session (``research.mode: agent``).

    Claude drives the four-phase protocol (formulate → match → optimize → decide) through
    the curated tool registry; these knobs bound one session. Needs ``ANTHROPIC_API_KEY``
    + the ``[llm]`` extra at runtime — without them research falls back to the legacy loop.
    """

    model: str = "claude-opus-4-8"
    # Which research loop drives a session (episodic-research epic #62). ``conversation`` = the
    # one long tool-use transcript; ``episodic`` = the deterministic session driver that owns the
    # protocol and calls the model only at narrow judgment episodes (for a small-context local
    # model). ``auto`` (the default) is the evidence-gated flip (#76): episodic when
    # ``context_window`` is declared at or below the documented threshold, conversation for
    # larger/unset windows. Loop selection is resolved in the composition root
    # (``bootstrap.resolve_research_loop``), never here.
    loop: Literal["auto", "conversation", "episodic"] = "auto"
    # Dedicated authoring model for write_strategy (same LiteLLM ``provider/model`` grammar as
    # research.model — the provider prefix picks the API key). ``None`` (the default) = the driver
    # writes full source itself, today's behavior bit for bit. Set it to pair a cheap/local driver
    # that runs the session with a strong hosted coder that only turns structured briefs into
    # validated strategy files. Built stateless at the composition root; a missing provider
    # key/extra degrades loudly back to driver-authored mode, never a mid-session failure.
    coder_model: str | None = None
    # The paid coder-fallback model for cheapest-first authoring (episodic-research epic #62,
    # story #72). ``coder_model`` (typically a cheap/local coder) attempts EVERY authoring job
    # first; only when its validator-retry budget is spent — an honest write-gate failure, never a
    # budget refusal — the SAME brief escalates to this model with the full validator-retry budget.
    # ``None`` (the default) = no escalation path: a failed local author is skipped exactly as
    # today. Same LiteLLM ``provider/model`` grammar as ``coder_model``; built stateless at the
    # composition root beside the local coder, and a missing provider key/extra degrades loudly to
    # no fallback (never a mid-session failure). The paid coder is a counted fallback triggered by
    # validator failure, never a default — its per-session spend is bounded by ``max_escalations``.
    coder_fallback_model: str | None = None
    # How many failed local authoring attempts may escalate to ``coder_fallback_model`` per session
    # (story #72). ``0`` (the default) disables escalation entirely — the paid coder is never
    # reached even when ``coder_fallback_model`` is set — so escalation is strictly opt-in bounded
    # spend. Each escalation (whether the paid coder then authors the file or also fails) counts
    # against this cap; once spent, a further local failure is skipped without touching the paid
    # model. Formulate/decide stay local; only authoring escalates. Inert without a configured
    # ``coder_fallback_model`` (there is nothing to escalate to).
    max_escalations: int = 0
    # The coder's own thinking dial (#17), default ON — authoring (scenario-window + warmup
    # arithmetic) is the reasoning-heavy sub-task, so the dedicated coder client reasons through it
    # instead of repeating an error it was just shown. Separate from the driver's ``thinking`` watch
    # dial below and marked a *deliberate* decision at the composition root, so it turns on even a
    # Sonnet coder (whose driver-side thinking stays the cheap-path pin). Its cost is already
    # bounded by ``max_author_calls``; set ``off`` to opt a coder out. Inert without a coder_model.
    coder_thinking: Literal["off", "on"] = "on"
    # The ESCALATED (paid-fallback) coder's own thinking dial (#98), default OFF. The fallback is
    # by definition the strong model, and ``coder_thinking`` above is a dial tuned for weak local
    # coders: the first field exercise of escalation (#98) showed a thinking sonnet-5 fallback both
    # outrunning the transport timeout and spending the shared output ceiling on thinking until no
    # file fit — the insurance never paid out. Off, the escalated call spends its whole ceiling on
    # the file. Set ``on`` to opt the fallback into the same deliberate adaptive thinking as the
    # local coder (streamed authoring and the author engine's thinking allowance make that
    # survivable). Inert without a configured ``coder_fallback_model``.
    coder_fallback_thinking: Literal["off", "on"] = "off"
    # The coder's output-token ceiling — the FILE's budget. ``None`` (the default) defers to the
    # author engine's built-in ceiling (``StrategyAuthor._MAX_TOKENS``, sized so a full strategy
    # file never truncates mid-source); a number pins it. A coder client that runs provider
    # thinking gets the engine's thinking allowance added ON TOP (on Anthropic models thinking and
    # text share ``max_tokens`` — #98), so this ceiling is all text either way. A
    # compatibility/sizing lever — resize it for a coder backend whose output window differs — NOT
    # a cost budget: output tokens are billed as generated, so unused headroom costs nothing (spend
    # is bounded by ``max_author_calls``, not this). Inert without a configured ``coder_model``.
    coder_max_tokens: int | None = None
    # The coder's sampling temperature (#222). ``None`` (the default) sends NO temperature at all —
    # today's request, byte for byte — and a number is offered to the client, which forwards it
    # ONLY where the provider seam declares the capability (``llm.Capabilities.temperature``). On a
    # provider without it the knob is a clean no-op: the parameter is simply not sent, never an
    # error and never a fake promise. Current Anthropic models reject a temperature beside the
    # thinking dial this seam pins (a 400) and hosted OpenAI's reasoning family fixes it, so the
    # real lever is the local/OpenAI-compatible backend (vLLM, Ollama, llama.cpp). Lowering it does
    # NOT make a coder deterministic — it narrows the sampler, nothing more; repetitions and paired
    # statistics are the defence against run-to-run variance, never this knob. Inert without a
    # configured ``coder_model``.
    coder_temperature: float | None = None
    # The coder's sampling seed (#222), same contract as ``coder_temperature``: ``None`` (the
    # default) sends nothing, a number is forwarded only where the provider seam declares the
    # ``seed`` capability. Current Anthropic models expose no seed parameter whatsoever, so it is a
    # capability no-op there; local/OpenAI-compatible servers accept it as a real sampler control.
    # Even where it IS sent a seed buys *repeatability at best effort*, never determinism —
    # batching, kernel/quantization nondeterminism and a moved model snapshot all still change the
    # output — so it is a variance-reduction lever, not an identity. Inert without a coder model.
    coder_seed: int | None = None
    # Private validator re-prompts per authoring job (#222): after the first attempt, how many times
    # the coder may be shown its own gate error and asked again before the job fails typed. ``None``
    # (the default) defers to the author engine's built-in budget (``StrategyAuthor._CODER_RETRIES``
    # = 2, i.e. initial + 2 ≤ 3 coder completions per job); a number pins it. Every attempt —
    # landing or not — spends one coder completion, so raising this raises the authoring bill in the
    # Class-B ``max_author_calls`` ceiling that still bounds the session. A robustness/experiment
    # knob (it is what a coder-benchmark ablation varies), never a gate: an attempt still has to
    # pass the same write gate. Inert without a configured ``coder_model``.
    coder_retries: int | None = None
    # Provider-native reasoning dial (verbose-observability P2), default OFF. ``"on"`` opts a
    # *watch* session into provider-native reasoning where it exists: for the Anthropic (non-Sonnet)
    # fallback model it sends adaptive thinking with a summarized display, so the loop emits
    # ``think`` events. This is the ONE observability knob that spends more (adaptive-thinking
    # output tokens) and the only one that changes a request parameter at all — leave it ``"off"``
    # for unattended runs. No-op on OpenAI/local (no thinking dial) and on Sonnet (its thinking
    # stays the deliberate cheap-path OFF under both settings). Adaptive thinking has no tunable
    # budget — the model picks depth — so this is a binary watch-session switch, not a spend dial.
    thinking: Literal["off", "on"] = "off"
    # Class-B research budgets (#12). ``None`` ⇒ read the value from the active ``cost_profile``
    # (the table in noctis/research/cost.py); set a number/bool here to PIN one budget regardless
    # of profile (an explicit per-knob override). The profile — not these defaults — is the source.
    max_iterations: int | None = None  # tool-use rounds per session
    max_backtests: int | None = None  # run_backtest calls + individual run_sweep trials
    sweep_trials: int | None = None  # default Optuna trials for one run_sweep call
    # Coder-model completions per session (coder-model split): every write_strategy brief the
    # coder authors — private validation retries included — spends one; one authored or revised
    # file ≈ one call. Bounds coder spend so an unbounded driver can't run up the bill. ``None`` ⇒
    # the active cost_profile's value (20/12/6 full/balanced/economy); a number here pins it.
    # Inert without a configured coder_model (source-based writes never touch this budget).
    max_author_calls: int | None = None
    # Worker processes for parallel evaluation (1 = fully sequential): sweep trials run
    # concurrently, and a panel run_backtest/evaluate_vs_champion evaluates its symbols
    # concurrently. Capped by CPU/task count; falls back to sequential if the pool breaks.
    # NOT a Class-B budget — parallelism/compute, not tokens — so it stays a plain default.
    sweep_workers: int = 8
    # Memory guard on the above: each worker holds a full copy of the panel bars + per-trial
    # intermediates, so peak RAM scales with the panel's TOTAL bar count — a 1m panel is ~60× a
    # 1h one. sweep_workers is a ceiling; the effective count is scaled down so workers × total
    # bars stays under this budget (fine/large panels shed workers toward sequential; coarse/small
    # ones keep them all). Prevents the OOM-killed-worker pool hang without hand-tuning per run.
    worker_bar_budget: int = 6_000_000
    # Server-side web_search grounding during FORMULATE/MATCH (same tool ideation uses).
    web_search: bool | None = None
    max_web_searches: int | None = None
    # Per-completion output-token ceiling. NOT a cost_profile budget — output is billed as
    # generated, so a high cap costs nothing unused; this is a compatibility lever for backends
    # that bound prompt+max_tokens by the model's context window (vLLM and other local/
    # OpenAI-compatible servers). ``None`` ⇒ the built-in default (8000, sized so a full
    # write_strategy file never truncates mid-generation). Lower it only to fit a small-context
    # model, knowing an oversized strategy file would then truncate and end the session.
    max_tokens: int | None = None
    # Whole-request context budget in tokens (system + tools + history, ~4 chars/token). Like
    # max_tokens, a compatibility lever for small-context backends — NOT a cost budget. When
    # set, the loop tiers per-result caps down, evicts the oldest tool-result bodies to fixed
    # pointer lines, and collapses a decided strategy's history at its verdict; everything
    # replaced stays re-fetchable through the same tools (the on-disk experiment journal is the
    # ground truth, so no gate is affected). ``None`` ⇒ unlimited (history byte-identical).
    context_window: int | None = None
    # Corrective retries per episode when the model misfires (episode runner, #66). An episode is
    # one forced structured-emit call; a misfire — markup instead of a native tool call, an
    # output-limit truncation, a thinking-only stall, or a payload that fails the schema — is
    # re-prompted with the classifier's corrective up to this many times before the episode fails
    # typed and the driver decides. Small by design: a persistent misfirer should fail fast, not
    # burn the session budget. Default 2 (initial + 2 retries = 3 completions), matching the
    # coder engine's private-retry budget. NOT a Class-B token budget — a robustness knob.
    episode_retries: int = 2


class ModelPriceConfig(BaseModel):
    """One model prefix's four ``$/Mtok`` rates — the shape ``research.pricing`` entries take.

    All four are required on purpose: input, output, cache-write and cache-read bill separately,
    and a half-stated price would silently value the unstated fields at nothing. The record calls
    everything derived from these an *estimate* (see ``noctis/research/pricing.py``), because list
    prices ignore discounts, batch tiers and mid-month changes.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_write_usd_per_mtok: float
    cache_read_usd_per_mtok: float


class ResearchConfig(BaseModel):
    """Cross-sectional (panel) research configuration.

    Research evaluates every candidate on a **panel** of universe symbols instead of a
    single series: the first ``fit_set_size`` ready universe symbols form the fit set
    (tuning + election), and the next ``symbol_holdout_size`` ready symbols are reserved
    as a symbol holdout — never seen by tuning or selection, fixed for the whole run.
    """

    # Who drives research: "agent" = Claude runs the formulate→match→optimize→decide
    # protocol through tools; "legacy" = the proposer/Optuna loop over registered families.
    # Agent mode needs an ANTHROPIC_API_KEY; without one it degrades to the legacy loop.
    mode: Literal["agent", "legacy"] = "agent"
    # Provider seam (issues #9/#10): a LiteLLM ``provider/model`` string. The four operator-chosen
    # models switch here with no code change — "openai/gpt-5.4" (default), "openai/gpt-5.5",
    # "anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8" — plus any "ollama/…"/local model.
    # The provider prefix picks the .env API key (openai/* → OPENAI_API_KEY, anthropic/* →
    # ANTHROPIC_API_KEY) and the capability set. ``None`` falls back to ``research.agent.model``.
    model: str | None = "openai/gpt-5.4"
    # Optional endpoint override for OpenAI-compatible / local backends (vLLM, Ollama, a proxy).
    base_url: str | None = None
    # Engine-level cost knob (#12): scales the Class-B research budgets together via the profile
    # table in noctis/research/cost.py. "balanced" (default) = today's ceilings (no behavior
    # change on upgrade); "economy" = reduced ceilings; "full" = maximums, and the automatic
    # choice on a free/local provider (overridable). Binds resource ceilings only — never a
    # promotion gate or the min_trials exhaustion floor (AGENTS.md rules 2/4).
    cost_profile: Literal["full", "balanced", "economy"] = "balanced"
    # The active mandate under mandate_dir: a profile name, "MANDATE" (mandate/MANDATE.md),
    # "auto" (agent picks a profile per session), or null (unconstrained). Its config: block
    # overlays every path noctis.config.overlay classifies as run-shaping (and may move the two
    # clamped ones in the disciplined direction only); a refused, unknown, or invalid key is
    # fatal at startup — see docs/configuration.md. This selector is itself refused there: an
    # overlay may not choose which overlay is read. Under "auto" the agent picks its profile
    # mid-session, long after assembly, so that profile's config: block never applies at all
    # (startup warns). CLI --mandate/--directive override the selector for one session.
    mandate: str | None = None
    # Exhaustion gate: verdict tools (evaluate_vs_champion / reject_strategy) refuse until
    # the strategy's journal shows this many distinct param sets or one completed sweep.
    min_trials: int = 8
    # Working-tier housekeeping (story #56): on each research-session assembly, undecided
    # (draft/candidate) top-level drafts in workspace/strategies/__tmp/ whose mtime predates
    # this many hours are swept into __tmp/archive/ before the library loads, so a session
    # never inherits a stale corpse. Bounds only how long an abandoned draft lingers; ``None``
    # or ``0`` disables the sweep entirely. Pure housekeeping — it moves bytes verbatim and
    # never touches a verdict, a promotion gate, or the exhaustion floor (AGENTS.md rule 2).
    draft_ttl_hours: float | None = 48.0
    # Agent-session knobs (model, iteration/backtest budgets, web search).
    agent: AgentResearchConfig = Field(default_factory=AgentResearchConfig)
    # Symbols in the fit panel (walk-forward + election). 0 would disable research.
    fit_set_size: int = 6
    # Ready symbols reserved as the cross-sectional holdout; 0 disables the symbol gate.
    symbol_holdout_size: int = 2
    # Cap on the research *focus set* — the symbols enumerated into each session's prompt
    # (the MARKET REALITY digest): fit set + symbol-holdout names + mandate-declared symbols.
    # Purely a prompt-size lever: symbols beyond the cap stay tradeable (the trading roster
    # never shrinks) and re-fetchable via preview_bars/list_symbols — they just aren't
    # broadcast into every prompt as the lake grows.
    focus_size: int = 10
    # Optional λ subtracted (× cross-symbol dispersion) from the Optuna tuning objective
    # only — never from the election score. 0.0 = off (the shipped default). Within-strategy
    # shaping only: it tunes parameters, so it never touches between-strategy champion election.
    tuning_dispersion_penalty: float = 0.0
    # Stage-2 memory distillation (context plan P3): every N completed research sessions, one
    # LLM call at CLOSE folds the full findings history into MEMORY.md's machine-owned
    # "Distilled lessons" block; sessions then embed that block + the 3 newest raw entries.
    # 0 = off (the default). Degrades to the always-on code-side consolidation without a
    # client; never runs inside a research session's own loop.
    memory_distill_every: int = 0
    # Price overrides for the run record's spend estimate (story #140), keyed by **model prefix**
    # — ``{"anthropic/claude-opus-4": {input_usd_per_mtok: 5.0, …}}``. Empty (the default) means
    # the shipped table in ``noctis/research/pricing.py`` under its own version. An override adds
    # a model the table never heard of or restates one it did; the resulting table identifies
    # itself as ``<version>+custom.<digest>`` in the record, so a reader can always tell whether
    # the numbers came from the engine's own prices. Pure accounting: nothing here is read by a
    # gate, a budget or a research decision — it only changes what the record *reports* a run
    # cost, which is why the mandate overlay refuses it (an experiment may not restate its bill).
    pricing: dict[str, ModelPriceConfig] = Field(default_factory=dict)


class ObservabilityConfig(BaseModel):
    """Config mirrors for the inline verbose feeds (verbose-observability P4).

    Purely display-level — nothing here is read by a decision path (observability is read-only,
    in the spirit of AGENTS.md's invariants). The interactive surface is the ``-v``/``-vv``/
    ``--show-reasoning`` flags; these knobs let an unattended (cron) run tune the same feeds.
    """

    # Live TRADING heartbeat cadence: every N streaming polls the ``-vv`` trading feed emits one
    # ``heartbeat`` event (poll count, open positions, mark-to-market equity) — the "is it alive?"
    # signal a long unattended session needs. 0 disables it. Only the live driver polls (replay is
    # a single instantaneous pass), so this is a no-op under catalog replay. At the default 2s poll
    # interval, 60 polls ≈ a heartbeat every ~2 minutes.
    heartbeat_polls: int = 60


class QAConfig(BaseModel):
    """Retention for the ``--debug`` QA run tree (``workspace/qa/``; epic #36, story #42).

    Purely a housekeeping knob — nothing here is read by a decision path. The QA area holds one
    folder per debug-recorded run; left unbounded it grows forever, so the only policy is
    prune-on-start (see :func:`noctis.observability.debug.prune_qa_dir`), keeping the newest N.
    """

    # On the start of a debug-recorded run, prune the QA area to the newest this-many run
    # folders (by run-id name order). 0 keeps nothing; the pruner clamps a negative to 0.
    keep_last_runs: int = 20


class PromotionConfig(BaseModel):
    """Scoring metric + challenger→champion promotion thresholds (all in the metric's units).

    **Refused by the mandate overlay, and why.** Every field below except ``metric`` is
    classified REFUSED in :mod:`noctis.config.overlay`: these thresholds *are* the promotion
    gates, and a mandate that could move them would be a permission slip rather than a search
    prior. They are refused outright rather than clamped "tighten-only" like the exhaustion
    floor, because across a metric change the direction is not even well defined — read in
    ``metric``'s units, a ``max_gap`` of 0.5 is not a stricter 1.0, it is a different scale.
    That is the same incomparability that makes a champion scored under a different metric
    *stale* (displaceable) instead of beatable. ``metric`` itself is allowed: it is the
    operator's risk appetite, and it reinterprets the thresholds' units without loosening one.
    """

    # The objective every research stage scores on — your risk appetite. ``sharpe`` penalizes
    # all volatility (risk-averse); ``sortino`` penalizes only downside; ``total_return``
    # ignores volatility (raw profit, most risk-seeking). Drives the pipeline score AND every
    # gate below, so changing it reinterprets the thresholds — re-tune them for the new units.
    metric: str = "sharpe"
    # Reject a challenger whose train−test metric gap exceeds this (overfit guard).
    max_gap: float = 1.0
    # A challenger must clear this out-of-sample test metric to take a free slot.
    min_test_metric: float = 0.0
    # Forward-holdout gate: a challenger must clear this metric on the reserved most-recent
    # slice the search never touched. Enforced only when a holdout was reserved (enough bars).
    min_holdout_metric: float = 0.0
    # Symbol-holdout gate: a challenger must clear this metric on the reserved held-out
    # symbols (names never used in tuning/selection). Enforced only when the scorecard
    # carries a symbol_holdout_metric (panel research with symbol_holdout_size > 0).
    min_symbol_holdout_metric: float = 0.0
    # Optional breadth gate: minimum fraction of fit symbols with a positive per-symbol
    # test metric (e.g. 0.6). 0.0 disables it (the default — specialization is legitimate).
    min_symbol_consistency: float = 0.0
    # Activity floor: minimum fraction of test splits with market exposure. A strategy that
    # almost never trades can post a positive average metric on a few lucky windows and sit
    # unbeatable at the top of the registry. 0.0 disables it.
    min_test_activity: float = 0.0
    # ── Metric robustness (scoring): bound Sharpe/Sortino so noise can't sit unbeatable atop the
    # registry. Both feed the pipeline score AND every gate above (units follow from them).
    # Annualize no finer than this bars/year ceiling (252 = daily): annualizing sub-daily returns
    # by sqrt(intraday periods) inflates the ratio 20-300x (1m x313 vs daily x16).
    annualization_cap: int = 252
    # Clamp the per-period risk-adjusted ratio (mean/std, mean/downside-std) to +/- this. A
    # per-BAR Sharpe/Sortino above ~1 is degeneracy (a split with near-zero downside), not edge;
    # unclamped it annualizes into the tens of thousands. Raise it to loosen, never below realism.
    max_period_ratio: float = 1.0
    # ── Degeneracy backstops (promotion gates): reject a challenger whose test metric implausibly
    # EXCEEDS train — a large negative train−test gap, the mirror of the max_gap overfit guard.
    # A hugely-better-out-of-sample result is a noise signal, not a robust edge. 0.0 disables.
    max_reverse_gap: float = 1.0
    # Reject a challenger whose |test metric| exceeds this sane ceiling (a second net beyond the
    # per-period clamp, for when it is loosened). 0.0 disables.
    max_test_metric: float = 0.0

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        # Call-time import: config can't import the backtest package at module scope
        # (backtest → broker → live_stub → config.gate closes a cycle), and validators
        # only fire on loaded values, never on this class's defaults.
        from noctis.backtest.scorecard import Metric

        return Metric.parse(value).value


class IdeationConfig(BaseModel):
    """LLM ideation of new ``StrategySpec`` families (opt-in; needs the ``[llm]`` extra + a key
    for the model's provider — none for a local backend). ``enabled`` is the config switch; the
    Ideator additionally requires a usable client at runtime, so a bare run mints nothing
    regardless."""

    enabled: bool = True
    # New specs requested per ideation round (capped by max_tokens / the tool schema).
    specs_per_round: int = 3
    # Ideate on the seed round and every ``cadence`` research iterations thereafter.
    cadence: int = 5
    # Same provider seam grammar as research.model: bare id or ``provider/model``.
    model: str = "claude-opus-4-8"
    # Upper bound on features per minted spec (the ideation validation gate).
    max_indicators: int = 12
    # Let the ideation agent use the provider's server-side web_search tool to ground new
    # structures in published quantitative research (auto-disables where the provider lacks
    # it). Safe because minted specs are still parity-gated and evaluated causally; see the
    # forward-holdout gate for the backstop.
    web_search: bool = True
    # Cap on web searches per ideation round (bounds latency + tool-use cost).
    max_web_searches: int = 5


def _yaml_path() -> Path:
    """Resolve the config YAML path (overridable via ``NOCTIS_CONFIG``)."""
    return Path(os.environ.get("NOCTIS_CONFIG", "config.yaml"))


def _workspace_subpath(workspace: object, *parts: str) -> str:
    return str(Path(str(workspace)).joinpath(*parts))


# The credential fields, named once. Everything that must keep a secret out of an artifact reads
# this set: the ``--debug`` manifest's config digest (``bootstrap._digest_excluded_fields``) and the
# run record's frozen inputs (``config.rehydrate``), which is also why a resumed run takes its keys
# from the live ``.env`` rather than from the record (AGENTS.md rule 6). Kept beside the fields
# themselves so adding a credential is one edit; the overlay's refusal table declares the same three
# under :data:`~noctis.config.overlay.SECRETS`, and a test pins the two together.
SECRET_FIELDS: frozenset[str] = frozenset(
    {"databento_api_key", "anthropic_api_key", "openai_api_key"}
)

# The reserved run id every invocation that has **not** opened a run reads: the read-only verbs
# (``status``, ``champions``, ``account``, ``report``, ``backtest``). It is also the run
# ``noctis migrate`` adopts a pre-run-scoped ``workspace/state/`` into, and that coincidence is the
# point: an operator who migrates finds their champions, account and reports exactly where those
# verbs already look, instead of a silently-empty board beside abandoned state. The two verbs that
# *work* a run — ``noctis run`` and ``noctis research`` (story #137) — never use it: each opens a
# run, mints or addresses its id, and binds it here through :func:`bind_run_dir`.
DEFAULT_RUN_ID = "legacy"

# The per-artifact paths one run OWNS, and the subpath each takes under the run dir. Champions,
# the paper account, the forward ledger, specs, sessions and experiment journals (``state``), the
# per-day close reports, the ``--debug`` QA tree and the agent's live memory all belong to the run
# that produced them — two runs sharing them would crown champions onto one board and trade one
# paper account, so neither run's numbers would mean anything (epic #126, D5). The data lake is
# deliberately NOT here: vendor data is expensive, reproducible and run-neutral, so it stays
# workspace-level and shared by every run.
_RUN_SCOPED_SUBPATHS: dict[str, tuple[str, ...]] = {
    "state_dir": ("state",),
    "reports_dir": ("reports",),
    "memory_path": ("memory", "MEMORY.md"),
    "qa_dir": ("qa",),
}


def run_scoped_paths(run_dir: object) -> dict[str, str]:
    """The four per-artifact paths ``run_dir`` implies — the one derivation, stated once."""
    return {
        field: _workspace_subpath(run_dir, *parts) for field, parts in _RUN_SCOPED_SUBPATHS.items()
    }


def bind_run_dir(settings: Settings, run_dir: str | os.PathLike[str]) -> Settings:
    """Re-point ``settings`` at one run's tree, in place. Called once a run's id is minted.

    The composition root's half of the run-scoping change (:func:`noctis.bootstrap.open_run_store`
    is its only production caller): settings are assembled before a run exists, so the run dir
    they derived from is the reserved default, and opening a run rebinds them onto that run's own
    tree — every collaborator built afterwards then reads the run's state with no path arithmetic
    in any command body.

    A path an operator set **explicitly** (YAML, env, or constructor) is left exactly where they
    pointed it: only a path still equal to what the *current* run dir derives is re-derived, so an
    explicit override stays the absolute override this module has always promised. Idempotent, and
    it never touches ``data.lake_dir`` — the lake is shared across runs by design.
    """
    derived = run_scoped_paths(settings.run_dir)
    for field, value in run_scoped_paths(run_dir).items():
        if getattr(settings, field) == derived[field]:
            setattr(settings, field, value)
    settings.run_dir = str(run_dir)
    return settings


class Settings(BaseSettings):
    """Root application settings.

    Knobs default to safe values so the app is runnable with no configuration at all.
    Secrets default to ``None``/``False`` and come from the environment only.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- Operational knobs (config.yaml) ---
    mode: Literal["paper", "live"] = "paper"
    universe: list[str] = Field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    session: SessionConfig = Field(default_factory=SessionConfig)
    research_time_budget_minutes: int = 60
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    live_feed: LiveFeedConfig = Field(default_factory=LiveFeedConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    ideation: IdeationConfig = Field(default_factory=IdeationConfig)
    champion_count: int = 3
    # The wall-clock ceiling on ONE process — how long tonight lasts. Live tier: a run is stopped
    # each morning and resumed each night, so this is the operator's call every time, never a
    # decision the run made weeks ago.
    time_limit_hours: float | None = None
    # The compute ceiling on the whole RUN, across every segment (story #136). Frozen at creation
    # like everything else that decides what the accumulated results mean: it is what makes two
    # runs comparable on **equal compute** — a mandate given 100 research hours and one given 30
    # are not the same experiment. Once the run's cumulative runtime breaches it the loop stops
    # between phases (the same shutdown path ``time_limit_hours`` uses) and the run is `completed`:
    # terminal, so a published result can never quietly gain another segment. ``null`` = uncapped.
    run_limit_hours: float | None = None
    # Archive this run whole: embed **every** candidate's strategy source in the run record, not
    # just its champions' (story #141, `noctis run --embed-all-sources`). Off by default, because
    # the champions-only policy is what holds a fortnight's record to a couple of hundred
    # kilobytes instead of megabytes — every other candidate is still named there, by a
    # run-relative path plus a content hash. Frozen at run creation like the compute cap above: it
    # says what the run's artifact *is*, and a record whose contents depended on how the last
    # segment happened to be invoked would quietly lose what an earlier one embedded.
    embed_all_sources: bool = False
    # ── The one output root. Everything the engine writes lands under this directory
    # (gitignored): run state, the data lake, reports, agent memory. The per-artifact knobs
    # below derive from it when not explicitly set (see ``_derive_workspace_paths``); an
    # explicit value — YAML, env, or constructor — is an absolute override. Env override:
    # ``NOCTIS_WORKSPACE`` (the plain pydantic-derived name is deliberately unsupported,
    # mirroring ``ALLOW_LIVE``).
    workspace_dir: str = Field(default="workspace/", alias="NOCTIS_WORKSPACE")
    # The run tree: one ``<run_id>/`` folder per run, each holding that run's ``run.json`` record
    # and its liveness lock. Every ``noctis run`` mints a new run here (identity is minted, never
    # derived from the config), so this directory is the run history of the whole workspace.
    runs_dir: str = "workspace/runs"
    # ── The ONE run root. A run owns its state (epic #126, D5), so the four knobs below derive
    # from this directory rather than from the workspace: two runs in one workspace can no longer
    # crown champions onto one board or trade one paper account. Defaults to the reserved
    # ``runs/<DEFAULT_RUN_ID>/`` run — what an invocation that has not opened a run reads — and
    # ``noctis run`` rebinds it to its own minted run through :func:`bind_run_dir`.
    run_dir: str = "workspace/runs/legacy"
    # Directory for this run's state (champion registry, ledgers, journals, specs); gitignored.
    state_dir: str = "workspace/runs/legacy/state"
    # This run's daily reports (YYYY-MM-DD.md/.json + archive/); gitignored.
    reports_dir: str = "workspace/runs/legacy/reports"
    # This run's long-term agent memory file (seeded from the committed MEMORY.seed.md at run
    # creation, so one run's lessons never leak into another's trajectory).
    memory_path: str = "workspace/runs/legacy/memory/MEMORY.md"
    # Hour-segmented QA run reports (the --debug tree); gitignored like everything under workspace/.
    qa_dir: str = "workspace/runs/legacy/qa"
    # The one-file strategy library root: committed seeds + TEMPLATE.py, plus the gitignored
    # __tmp/ (working files) and champions/ (local champions) tiers. See strategies/README.md.
    strategies_dir: str = "strategies/"
    # The operator's input surface. Only the scaffold is committed (MANDATE.md.example, the five
    # shipped profiles/, tune-first.md, README, one reference example); the human's own MANDATE.md,
    # custom personalities, and personal references are gitignored so steering never pollutes git.
    mandate_dir: str = "mandate/"
    # The committed benchmark corpus: the curated buckets a review shipped to every user, read-only
    # input exactly like the strategy seeds beside it. The engine's own tier is <workspace>/cases/,
    # and a case id in both is the workspace's. See cases/README.md.
    cases_dir: str = "cases/"

    # --- Secrets / env-only switches ---
    databento_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Resolved per provider prefix by the LLM seam: openai/* → this key, anthropic/* → the above.
    openai_api_key: str | None = None
    # The live-execution env gate. Sourced from ALLOW_LIVE. One of two required gates.
    allow_live: bool = Field(default=False, alias="ALLOW_LIVE")

    @model_validator(mode="before")
    @classmethod
    def _derive_workspace_paths(cls, data):
        """Inject the derived defaults for the per-artifact paths when absent.

        Two roots, one chain: the **workspace** owns the run tree and the shared data lake, and
        the **run dir** (``runs/<run_id>/``, the reserved default until a run is opened) owns the
        four paths a run's own artifacts live under. Runs in mode ``"before"`` on the merged raw
        data (init > env > .env > YAML), so an absent knob is distinguishable from an explicit one
        and every public path field stays a plain ``str`` — no ``Optional`` ripple through
        consumers. The nested ``data.lake_dir`` is normalized here too, and stays workspace-level:
        vendor data is run-neutral and shared.
        """
        if not isinstance(data, dict):
            return data
        lowered = {key.lower(): value for key, value in data.items() if isinstance(key, str)}
        # The alias (env) key wins over the field name when both are present, matching the
        # env > YAML source order pydantic resolves the field itself with.
        workspace = lowered.get("noctis_workspace") or lowered.get("workspace_dir") or "workspace/"
        runs_dir = lowered.get("runs_dir") or _workspace_subpath(workspace, "runs")
        data.setdefault("runs_dir", runs_dir)
        run_dir = lowered.get("run_dir") or _workspace_subpath(runs_dir, DEFAULT_RUN_ID)
        data.setdefault("run_dir", run_dir)
        for field, value in run_scoped_paths(run_dir).items():
            data.setdefault(field, value)
        derived_lake = _workspace_subpath(workspace, "data_lake")
        raw_data = data.get("data")
        if raw_data is None:
            data["data"] = {"lake_dir": derived_lake}
        elif isinstance(raw_data, dict):
            raw_data.setdefault("lake_dir", derived_lake)
        elif isinstance(raw_data, DataConfig) and "lake_dir" not in raw_data.model_fields_set:
            data["data"] = raw_data.model_copy(update={"lake_dir": derived_lake})
        return data

    @field_validator("allow_live", mode="before")
    @classmethod
    def _blank_allow_live_is_false(cls, value):
        # An empty ``ALLOW_LIVE=`` (as shipped in .env.example) means "not set" → paper.
        # Without this, an empty string would raise a bool-parse error at startup.
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return False
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env > .env > config.yaml, with constructor args highest of all."""
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        yaml_file = _yaml_path()
        if yaml_file.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file))
        return tuple(sources)


def load_settings(config_path: str | os.PathLike[str] | None = None, **overrides) -> Settings:
    """Load :class:`Settings`.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file. When given, it takes effect for this load
        (via the ``NOCTIS_CONFIG`` environment variable).
    **overrides:
        Field overrides passed straight to the constructor (highest priority). Handy for
        tests and programmatic use.
    """
    if config_path is not None:
        os.environ["NOCTIS_CONFIG"] = str(config_path)
    return Settings(**overrides)
