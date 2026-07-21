"""Typed application configuration.

Layered sources, highest priority first: constructor args, process environment, ``.env``,
then ``config.yaml``. So operational knobs come from ``config.yaml`` while secrets and
overrides come from the environment (environment always wins over the YAML file).

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
    section is deliberately **not** in the mandate ``config:`` overlay allowlist — a research
    personality steers what to look for, never how forgiving the arena is.
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
    # Dedicated authoring model for write_strategy (same LiteLLM ``provider/model`` grammar as
    # research.model — the provider prefix picks the API key). ``None`` (the default) = the driver
    # writes full source itself, today's behavior bit for bit. Set it to pair a cheap/local driver
    # that runs the session with a strong hosted coder that only turns structured briefs into
    # validated strategy files. Built stateless at the composition root; a missing provider
    # key/extra degrades loudly back to driver-authored mode, never a mid-session failure.
    coder_model: str | None = None
    # The coder's own thinking dial (#17), default ON — authoring (scenario-window + warmup
    # arithmetic) is the reasoning-heavy sub-task, so the dedicated coder client reasons through it
    # instead of repeating an error it was just shown. Separate from the driver's ``thinking`` watch
    # dial below and marked a *deliberate* decision at the composition root, so it turns on even a
    # Sonnet coder (whose driver-side thinking stays the cheap-path pin). Its cost is already
    # bounded by ``max_author_calls``; set ``off`` to opt a coder out. Inert without a coder_model.
    coder_thinking: Literal["off", "on"] = "on"
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
    # "auto" (agent picks a profile per session), or null (unconstrained). Its config: block may
    # override promotion.metric ONLY — see docs/operator-mandate.md. Replaces the old directive
    # string; CLI --mandate/--directive override it for one session.
    mandate: str | None = None
    # Exhaustion gate: verdict tools (evaluate_vs_champion / reject_strategy) refuse until
    # the strategy's journal shows this many distinct param sets or one completed sweep.
    min_trials: int = 8
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


class PromotionConfig(BaseModel):
    """Scoring metric + challenger→champion promotion thresholds (all in the metric's units)."""

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
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    ideation: IdeationConfig = Field(default_factory=IdeationConfig)
    champion_count: int = 3
    time_limit_hours: float | None = None
    # ── The one output root. Everything the engine writes lands under this directory
    # (gitignored): run state, the data lake, reports, agent memory. The per-artifact knobs
    # below derive from it when not explicitly set (see ``_derive_workspace_paths``); an
    # explicit value — YAML, env, or constructor — is an absolute override. Env override:
    # ``NOCTIS_WORKSPACE`` (the plain pydantic-derived name is deliberately unsupported,
    # mirroring ``ALLOW_LIVE``).
    workspace_dir: str = Field(default="workspace/", alias="NOCTIS_WORKSPACE")
    # Directory for run state (champion registry, ledgers, journals, specs); gitignored.
    state_dir: str = "workspace/state"
    # Daily reports (YYYY-MM-DD.md/.json + archive/); gitignored.
    reports_dir: str = "workspace/reports"
    # The agent's long-term memory file (seeded from the committed MEMORY.seed.md).
    memory_path: str = "workspace/memory/MEMORY.md"
    # Hour-segmented QA run reports (the --debug tree); gitignored like everything under workspace/.
    qa_dir: str = "workspace/qa"
    # The one-file strategy library root: committed seeds + TEMPLATE.py, plus the gitignored
    # __tmp/ (working files) and champions/ (local champions) tiers. See strategies/README.md.
    strategies_dir: str = "strategies/"
    # The operator's input surface. Only the scaffold is committed (MANDATE.md.example, the five
    # shipped profiles/, tune-first.md, README, one reference example); the human's own MANDATE.md,
    # custom personalities, and personal references are gitignored so steering never pollutes git.
    mandate_dir: str = "mandate/"

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
        """Inject workspace-derived defaults for the per-artifact paths when absent.

        Runs in mode ``"before"`` on the merged raw data (init > env > .env > YAML), so an
        absent knob is distinguishable from an explicit one and every public path field
        stays a plain ``str`` — no ``Optional`` ripple through consumers. The nested
        ``data.lake_dir`` is normalized here too.
        """
        if not isinstance(data, dict):
            return data
        lowered = {key.lower(): value for key, value in data.items() if isinstance(key, str)}
        # The alias (env) key wins over the field name when both are present, matching the
        # env > YAML source order pydantic resolves the field itself with.
        workspace = lowered.get("noctis_workspace") or lowered.get("workspace_dir") or "workspace/"
        data.setdefault("state_dir", _workspace_subpath(workspace, "state"))
        data.setdefault("reports_dir", _workspace_subpath(workspace, "reports"))
        data.setdefault("memory_path", _workspace_subpath(workspace, "memory", "MEMORY.md"))
        data.setdefault("qa_dir", _workspace_subpath(workspace, "qa"))
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
