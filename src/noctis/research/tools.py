"""The curated tool registry the research agent drives — grid-mng's copilot tools, re-seamed.

Each tool wraps an **existing seam** (library loader, lake, pipeline ``evaluate``, champion
registry, memory); the registry is curated, not dumped. The loop's discipline is structural:

* **Experiment journal** — every ``run_backtest`` call and every ``run_sweep`` trial appends
  one JSON line to ``state/experiments/<strategy>.jsonl`` via :class:`ExperimentJournal`
  (:mod:`noctis.research.journal` owns the record schema); ``get_experiment_log`` reads it
  back ranked (grid-mng's per-job leaderboard, reborn per-strategy).
* **Exhaustion gate** — the verdict tools (``evaluate_vs_champion`` / ``reject_strategy``)
  refuse until the journal shows ≥ ``research.min_trials`` distinct parameter sets or one
  completed sweep. ``write_strategy`` for a *new* name while another strategy sits undecided
  returns a soft warning, never a block.
* **Anti-overfit surface** — ``run_backtest`` returns aggregate :class:`Scorecard` numbers
  only (never per-bar/per-trade series) and ``preview_bars`` is capped to the training
  window (never the reserved forward-holdout tail).
* **Budget caps** — data spend is gated inside ``lake.ensure_coverage`` (cost preflight);
  backtests/sweep trials count against a per-session budget.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.backtest.pool import evaluation_time_limit, scale_workers
from noctis.data.aggregate import (
    NATIVE_TIMEFRAME,
    TIMEFRAMES,
    aggregate_bars,
    bars_per_year,
    validate_timeframe,
)
from noctis.data.types import ns_to_date, ns_to_timestamp, to_ns, to_ns_end_inclusive
from noctis.observability.events import Event, render_plain
from noctis.research import websearch
from noctis.research.author import AuthoringError, StrategyAuthor, StrategyBrief
from noctis.research.cost import resolve_budgets
from noctis.research.exhaustion_registry import ExhaustedClassRegistry
from noctis.research.failed_store import FailedAttemptStore
from noctis.research.journal import ExperimentJournal
from noctis.research.sweep import SweepRunner
from noctis.research.symbols import BANDS, SymbolScreener, screen, validate_profile
from noctis.strategies import library
from noctis.strategies.base import ParamSpec
from noctis.strategies.families import FamilyRegistry

if TYPE_CHECKING:
    from noctis.data.seam import MarketData
    from noctis.memory.base import Memory

logger = logging.getLogger("noctis.research.tools")

_PREVIEW_ROW_CAP = 60
_LOG_LIMIT_CAP = 50
# Floor for run_sweep's max_bars fidelity cap — below this the walk-forward split
# geometry (train + test + holdout) has no room and every trial degenerates.
_MAX_BARS_FLOOR = 200
# Fixation backstop: after this many CONSECUTIVE write-gate rejections with zero backtests
# run this session, the rejection result gains a redirect toward tuning the existing library.
# A hint in a tool result, never a gate change — the gate keeps rejecting exactly what it
# rejected, and max_iterations still bounds a model that ignores it. Small local backends
# were observed burning whole sessions on failed authoring without ever producing evidence.
_WRITE_FIXATION_THRESHOLD = 3


def _offending_line(source: str, error: str) -> str | None:
    """The draft line a ``line N`` validation error points at, echoed back verbatim.

    Models were observed abandoning a whole thesis over a one-line syntax error; showing the
    exact line makes the repair path cheaper than starting over.
    """
    match = re.search(r"\bline (\d+)\b", error)
    if match is None:
        return None
    lines = source.splitlines()
    n = int(match.group(1))
    if not 1 <= n <= len(lines):
        return None
    return lines[n - 1].strip() or None


def _round_trip_cost_bp(fee_bps: float, slippage_bps: float) -> float:
    """Cost of one full trade (enter + exit) in bp from the configured per-side fill costs.

    Both sides pay ``fee_bps + slippage_bps``, so a round trip is ``2 ×`` that. The caller
    passes the operator-configured ``backtest`` costs, so the agent's hint reflects exactly
    what the pipeline and paper fills charge — never a stale hardcoded default."""
    return 2.0 * (fee_bps + slippage_bps)


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _tool(name: str, description: str, properties: dict | None = None, required=None) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": list(required or []),
        },
    }


class _CoderCallCounter:
    """Counting proxy over the coder LLM client so the toolbox tallies every completion the
    authoring engine spends — private validation retries included.

    The :class:`~noctis.research.author.StrategyAuthor` engine stays toolbox-state-free: it drives
    the coder through the plain ``complete()`` seam while this wrapper does the accounting the
    toolbox owns (AGENTS.md: "the toolbox keeps the accounting"). Every ``complete()`` — the one
    place a coder completion actually happens — bumps the count before the call, so a completion
    that raises still counts as spend. Any other attribute forwards to the wrapped client.
    """

    def __init__(self, client, on_complete):
        self._client = client
        self._on_complete = on_complete

    def __getattr__(self, item):
        return getattr(self._client, item)

    def complete(self, **kwargs):
        self._on_complete()
        return self._client.complete(**kwargs)


class ResearchToolbox:
    """Session-scoped tool registry + dispatcher for the agent research loop.

    Holds the loop-side protocol state: which strategies were written this session and
    remain undecided (feeds the new-strategy soft nudge), how many backtests the session
    has spent, and the promote/reject counters the summary reports.
    """

    # ── tool semantics, declared beside the tools (consumed by the agent loop) ──
    # The verdict tools end a strategy: their successful results are the session's durable
    # conclusions — the context budget never replaces them, and each triggers the
    # verdict-boundary compaction of the strategy it decides.
    VERDICT_TOOLS: ClassVar[frozenset[str]] = frozenset({"evaluate_vs_champion", "reject_strategy"})
    # Tools whose per-strategy results are re-fetchable from the experiment journal
    # (state/experiments/<name>.jsonl — every trial they report is journaled), so the
    # context budget may collapse their history to pointer lines without losing anything
    # durable: the exhaustion gate reads the journal, never the context.
    STRATEGY_HISTORY_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {"run_backtest", "run_sweep", "get_experiment_log"}
    )
    # The result keys worth one feed line: the gate-facing numbers a promotion/rejection
    # actually turns on — the train/test overfit `gap`, the temporal `holdout_metric`, the
    # cross-sectional `symbol_holdout_metric` — so the -v feed alone tells the story
    # without -vv. `n_failed` rides alongside `n_trials` so a sweep's burned budget shows
    # on the same line (only sweep results carry it). Neutral numbers only; no
    # verdict-shaped editorializing (AGENTS.md rule 2 / the honesty spirit).
    TOOL_LINE_KEYS: ClassVar[tuple[str, ...]] = (
        "promoted",
        "rationale",
        "stage",
        "avg_test_metric",
        "gap",
        "holdout_metric",
        "symbol_holdout_metric",
        "n_trials",
        "n_failed",
        "status",
        "ok",
    )

    def __init__(
        self,
        *,
        settings,
        lake: MarketData,
        registry,
        families: FamilyRegistry,
        memory: Memory,
        rules,
        mandate_source=None,
        mandate=None,
        coder_client=None,
        on_event=None,
    ):
        self.settings = settings
        self.lake = lake
        self.families = families
        self.registry = registry
        self.memory = memory
        self.rules = rules
        # Provenance of the active mandate this session (or None). Stamped onto every
        # champion the verdict path crowns, so `auto` selection can attribute champions.
        self.mandate_source = mandate_source
        # The resolved mandate itself (or None): its declared symbols join the research
        # focus set (the prompt digest + holdout candidate pool enumeration).
        self.mandate = mandate
        # The dedicated strategy-authoring ("coder") LLM client, or None (the default) when the
        # driver authors full source itself. Built at the composition root from
        # research.agent.coder_model. When set, write_strategy switches to brief mode: the
        # driver commits a StrategyBrief and this client authors the file (see _author_source).
        self.coder_client = coder_client
        # The coder model string that pairs with coder_client — stamped onto every authoring
        # event so a watch session names the model that wrote (or failed to write) the file.
        self.coder_model = settings.research.agent.coder_model
        # The session event channel this toolbox emits authoring observability through (#9): the
        # SAME on_event sink the agent loop and console already use, threaded in at the composition
        # root. None (tests, bare loops) falls back to the logger, like the agent's default sink.
        self.on_event = on_event
        # The three library tier roots (seeds committed, __tmp/champions under the
        # workspace); every library call takes it opaquely, incl. the sweep workers.
        self.strategies_dir = library.LibraryPaths.from_settings(settings)
        # Coder completions spent this session (author path only): the counting proxy below bumps
        # it on every engine completion, private retries included — the Class-B author budget
        # meters on it and the session summary surfaces it. Source-based writes never move it.
        self.author_calls = 0
        # The brief-authoring engine — built only in coder mode. Toolbox-state-free: it turns
        # a brief into validated source through the SAME library.write_strategy gate the source
        # path uses, so both authoring paths converge on one result shape and one set of guards.
        # The coder client is wrapped so every completion the engine spends counts against the
        # author budget (retries included) — the accounting stays toolbox-side.
        self.author_engine = (
            StrategyAuthor(
                client=_CoderCallCounter(coder_client, self._bump_author_calls),
                strategies_dir=self.strategies_dir,
                families=self.families,
            )
            if coder_client is not None
            else None
        )
        # Every coder attempt the write gate rejects is persisted here — a capped folder under
        # the working tier (<__tmp>/failed/) — so a bad authoring session is inspectable from
        # disk, not just terminal scrollback (#18). The attempt sink writes the attempted source
        # + gate error on every rejection; a landing attempt writes nothing, and the store evicts
        # oldest over its cap so observability never grows unbounded. Only the coder path reaches
        # it (the engine's per-attempt seam is the sole caller); source-based writes never do.
        self.failed_store = FailedAttemptStore(self.strategies_dir.tmp / "failed")
        self.state_dir = Path(settings.state_dir)
        # The durable evidence record every gate reads — see noctis.research.journal.
        self.journal = ExperimentJournal(self.state_dir)
        # Cross-session guard against re-mining a class the agent already proved dead.
        self.exhausted = ExhaustedClassRegistry(self.state_dir / "exhausted_classes.json")
        self.dataset = settings.data.dataset
        self.schema = "ohlcv-1m"
        # Structural feature store for screen_symbols + the digest's character block
        # (session-cached; recomputes automatically when ensure_data extends a series).
        self.screener = SymbolScreener(lake, self.dataset, self.schema)
        self.min_trials = settings.research.min_trials
        # Class-B budgets come from the active cost_profile (#12), with any explicit
        # research.agent per-knob value pinning its own budget. min_trials is NOT here —
        # the exhaustion floor is quality, not cost, and the profile never touches it.
        budgets = resolve_budgets(settings.research)
        self.max_backtests = budgets.max_backtests
        self.default_sweep_trials = budgets.sweep_trials
        # Coder-model Class-B budget: every brief the coder authors (retries included) counts;
        # once spent, brief authoring is refused (source-based writes stay open). Inert without
        # a coder_model — no completion ever happens, so the count stays 0.
        self.max_author_calls = budgets.max_author_calls
        self.sweep_workers = settings.research.agent.sweep_workers
        self.worker_bar_budget = settings.research.agent.worker_bar_budget
        # run_sweep's execution engine — sampler, fork pool, stall guard — behind its own
        # interface (noctis.research.sweep). The toolbox keeps the accounting (budget,
        # journal, ranking); tests substitute a fake runner here.
        self.sweep_runner = SweepRunner(
            strategies_dir=self.strategies_dir,
            workers=self.sweep_workers,
            bar_budget=self.worker_bar_budget,
            evaluate_fn=self._evaluate,
        )

        # Session protocol state.
        self.backtests_run = 0
        self.promotions = 0
        self.rejections = 0
        self.strategies_touched: list[str] = []
        self.undecided: set[str] = set()
        # Consecutive write-gate rejections (reset by any successful write) — feeds the
        # fixation backstop in tool_write_strategy.
        self._write_gate_failures = 0
        # Last name whose write was rejected and never repaired — a successful write under a
        # DIFFERENT name means the model abandoned a fixable draft, and draws a nudge back.
        self._last_failed_write: str | None = None

        # The library is this toolbox's world — make sure it is registered.
        library.load_and_register(self.strategies_dir, self.families)

    # ── exhaustion gate ──────────────────────────────────────────────────────
    def _exhaustion_block(self, name: str) -> str | None:
        stats = self.journal.stats(name)
        if stats.n_distinct_params >= self.min_trials or stats.sweep_completed:
            return None
        return (
            f"exhaustion gate: {name!r} has only {stats.n_distinct_params} distinct parameter "
            f"set(s) in its journal and no completed sweep. Explore the parameter space first "
            f"— at least {self.min_trials} distinct sets via run_backtest, or one run_sweep — "
            f"then decide."
        )

    def _is_champion(self, name: str) -> bool:
        """True when ``name`` is a family currently seated in the champion registry.

        The file's ``status:`` header and the registry are separate stores; only the registry
        governs what actually trades. Rejecting a seated champion would only re-stamp the file
        while it keeps trading — a split-brain — so :meth:`tool_reject_strategy` refuses it.
        """
        return any(entry.family == name for entry in self.registry.list())

    # ── data helpers ─────────────────────────────────────────────────────────
    def _bars_for(
        self, symbols: list[str], timeframe: str = NATIVE_TIMEFRAME
    ) -> dict[str, pd.DataFrame]:
        symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        if not symbols:
            raise ValueError("no symbols given")
        missing = [s for s in symbols if not self.lake.check_symbol_ready(s)]
        if missing:
            raise ValueError(
                f"symbols not ready in the lake: {missing}; use list_symbols to see coverage "
                f"and ensure_data to fetch history"
            )
        raw = self.lake.get_bars(self.dataset, self.schema, symbols, 0, 2**63 - 1)
        bars = {s: aggregate_bars(df, timeframe) for s, df in raw.items() if len(df) > 0}
        empty = sorted(set(symbols) - set(bars))
        if empty:
            raise ValueError(f"no catalog bars for: {empty}")
        return bars

    def _timeframe_for(self, name: str) -> str:
        """The bar granularity the strategy declares (the lake stays 1m; we aggregate)."""
        return validate_timeframe(self.families.get_class(name).timeframe)

    def _pipeline_config(
        self, bars_by_symbol: dict[str, pd.DataFrame], timeframe: str = NATIVE_TIMEFRAME
    ) -> PipelineConfig:
        # One geometry/metric home (PipelineConfig.auto), sized from the shortest series.
        # Sizes are in the strategy's own timeframe (bars arrive here already aggregated);
        # metric annualization follows the timeframe.
        return PipelineConfig.auto_from_settings(
            self.settings,
            min(len(df) for df in bars_by_symbol.values()),
            periods_per_year=bars_per_year(timeframe),
            prefilter_min_score=None,  # tools always want the full scorecard
        )

    def _require_strategy(self, name: str) -> None:
        if library.strategy_path(self.strategies_dir, name) is None:
            raise ValueError(
                f"no strategy named {name!r} in the library; write_strategy it first "
                f"(list_strategies shows what exists)"
            )
        if name not in self.families:
            library.load_and_register(self.strategies_dir, self.families)

    def _resolved_params(self, name: str, params: dict | None) -> dict:
        strategy = self.families.create(name, dict(params or {}))
        return strategy.params_dict()

    def _window(self, bars: dict[str, pd.DataFrame]) -> dict:
        first = min(int(df["ts_event"].iloc[0]) for df in bars.values())
        last = max(int(df["ts_event"].iloc[-1]) for df in bars.values())
        return {
            "bars": min(len(df) for df in bars.values()),
            "start": ns_to_date(first).isoformat(),
            "end": ns_to_date(last).isoformat(),
        }

    def _evaluate(
        self,
        name: str,
        params: dict,
        bars: dict[str, pd.DataFrame],
        symbol_holdout: dict[str, pd.DataFrame] | None = None,
    ):
        config = self._pipeline_config(bars, self._timeframe_for(name))
        # Scale workers by the panel's total bar count so a memory-heavy 1m panel sheds workers
        # (each holds bars + intermediates) rather than OOM-killing them; 1h+ panels keep them all.
        total_bars = sum(len(df) for df in bars.values()) + sum(
            len(df) for df in (symbol_holdout or {}).values()
        )
        # The wall-clock ceiling is the sequential sibling of the pool stall guard: with
        # workers scaled to 1 (the common case on a big 1m panel) no pool guard exists, and
        # an agent-authored strategy that hangs on a pathological param set would otherwise
        # wedge the research loop forever. An EvaluationTimeout surfaces to the caller — a
        # tool error for the agent, an ended sweep for the SweepRunner.
        with evaluation_time_limit():
            return evaluate(
                Candidate(name, params),
                bars,
                config=config,
                symbol_holdout=symbol_holdout,
                workers=scale_workers(
                    self.sweep_workers, total_bars, budget=self.worker_bar_budget
                ),
                families=self.families,
            )

    @staticmethod
    def _test_split_mean(card, name: str) -> float:
        splits = [s for ss in card.symbols.values() for s in ss.splits]
        if not splits:
            return 0.0
        return sum(s.test.get(name) for s in splits) / len(splits)

    def _card_summary(self, card, bars: dict[str, pd.DataFrame] | None = None) -> dict:
        out = {
            "stage": card.stage,
            "metric": card.metric_name,
            "avg_train_metric": _round(card.avg_train_metric),
            "avg_test_metric": _round(card.avg_test_metric),
            "gap": _round(card.gap),
            "holdout_metric": _round(card.holdout_metric),
        }
        # Trade economics: is the metric built on real activity, and what do costs eat?
        # A near-zero activity means the aggregate rests on a handful of trades (noise);
        # turnover × round-trip cost is the drag the signal must out-earn.
        out["trade_economics"] = {
            "test_activity": _round(card.test_activity),
            "avg_test_exposure": _round(self._test_split_mean(card, "exposure")),
            "avg_test_turnover": _round(self._test_split_mean(card, "turnover"), 6),
            "round_trip_cost_bp": _round(
                _round_trip_cost_bp(
                    self.settings.backtest.fee_bps, self.settings.backtest.slippage_bps
                ),
                2,
            ),
        }
        if bars:
            # The do-nothing benchmark over the same evaluated window (full window, not
            # per-split): the strategy must justify itself against simply holding.
            hold = {
                sym: _round(float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1.0)
                for sym, df in bars.items()
                if len(df) > 1 and float(df["close"].iloc[0]) != 0.0
            }
            if hold:
                out["buy_hold_full_window"] = {
                    "per_symbol": hold,
                    "mean": _round(sum(hold.values()) / len(hold)),
                }
        per_symbol = card.symbol_test_metrics()
        if per_symbol:
            out["symbol_test_metrics"] = {s: _round(v) for s, v in per_symbol.items()}
            out["symbol_holdout_metric"] = _round(card.symbol_holdout_metric)
            out["panel_dispersion"] = _round(card.panel_dispersion)
        if card.dropped_symbols:
            out["dropped_symbols"] = card.dropped_symbols
        return out

    def market_context(self) -> dict:
        """Session-start economics digest: bar granularity, the cost hurdle, and what
        buy-and-hold did per research-focus symbol over the full lake. This is the context
        the formulating agent needs to do cost arithmetic BEFORE writing a strategy.

        Enumerates the capped :func:`~noctis.engine.runtime.research_focus` set — not the
        whole growing trading roster — so the digest stays bounded as discoveries
        accumulate; any lake symbol remains inspectable via preview_bars/list_symbols.
        Per-symbol numbers stay neutral (no viability flags); ``cost_hurdle`` is one
        code-computed ratio the agent would otherwise redo by hand, never a verdict."""
        from noctis.engine.runtime import research_focus

        backtest = self.settings.backtest
        round_trip_bp = _round_trip_cost_bp(backtest.fee_bps, backtest.slippage_bps)
        timeframes = sorted(TIMEFRAMES, key=lambda t: TIMEFRAMES[t])
        symbols: dict[str, dict] = {}
        # Per timeframe: how many focus symbols' median abs bar move exceeds the round trip.
        clears: dict[str, int] = dict.fromkeys(timeframes, 0)
        for sym in research_focus(self.settings, self.lake, self.mandate):
            if not self.lake.check_symbol_ready(sym):
                continue
            try:
                bars = self._bars_for([sym])[sym]
            except ValueError:
                continue
            close = bars["close"].astype("float64")
            if len(close) < 2 or float(close.iloc[0]) == 0.0:
                continue
            returns = close.pct_change().dropna()
            symbols[sym] = {
                "bars": int(len(bars)),
                "buy_hold_return": _round(float(close.iloc[-1] / close.iloc[0] - 1.0)),
                "median_abs_bar_move_bp": _round(float(returns.abs().median()) * 1e4, 2),
                "bar_move_std_bp": _round(float(returns.std()) * 1e4, 2),
            }
            # The same structural read screen_symbols ranks on (training-window bars),
            # inlined so MATCH-phase profile choices are grounded before any tool call.
            features = self.screener.features_from_bars(sym, bars)
            if features is not None:
                symbols[sym]["character"] = self._feature_brief(features)
            for tf in timeframes:
                agg_close = (
                    close if tf == NATIVE_TIMEFRAME else aggregate_bars(bars, tf)["close"]
                ).astype("float64")
                if len(agg_close) < 2:
                    continue
                median_bp = float(agg_close.pct_change().dropna().abs().median()) * 1e4
                if median_bp > round_trip_bp:
                    clears[tf] += 1
        return {
            "bar_schema": self.schema,
            "supported_timeframes": timeframes,
            "fee_bps_per_side": backtest.fee_bps,
            "slippage_bps_per_side": backtest.slippage_bps,
            "round_trip_cost_bp": _round(round_trip_bp, 2),
            "symbols": symbols,
            # Neutral arithmetic, not advice: per timeframe, the fraction of the focus
            # symbols whose median abs bar move (on actually-aggregated bars) exceeds the
            # round-trip cost. Which side of the ratio a thesis needs is still yours to argue.
            "cost_hurdle": {
                "median_bar_move_clears_round_trip": {
                    tf: f"{clears[tf]}/{len(symbols)}" for tf in timeframes
                }
            },
            # Classes prior sessions proved dead — do NOT re-mine these (see write_strategy's
            # class_tag/new_lever guard); reuse a listed class_tag verbatim if you extend one.
            "exhausted_classes": self.exhausted.summary(),
        }

    def _spend_backtests(self, n: int = 1) -> str | None:
        if self.backtests_run + n > self.max_backtests:
            return (
                f"backtest budget exhausted ({self.backtests_run}/{self.max_backtests} spent "
                f"this session); reach a verdict with what the journal already shows"
            )
        return None

    def _bump_author_calls(self) -> None:
        """Count one coder completion (the counting proxy calls this per engine ``complete()``)."""
        self.author_calls += 1

    def _author_budget_block(self) -> str | None:
        """Refuse to START a new brief-authoring job once the coder budget is spent.

        Meters on completions, so a job already running may overrun the cap with its private
        retries — this only refuses to begin a *new* one, mirroring the codebase's "check before
        you start" budget idiom. A refusal with explicit driver guidance, never a silent failure;
        the hand-written ``source`` path is deliberately not gated (it spends no coder completion).
        """
        if self.author_calls >= self.max_author_calls:
            return (
                f"author-call budget exhausted ({self.author_calls}/{self.max_author_calls} coder "
                f"completions spent this session); further brief authoring is refused. Revise an "
                f"existing strategy by hand (submit `source` — the hand-written path stays open) "
                f"or proceed to a verdict (evaluate_vs_champion / reject_strategy) with what the "
                f"journal already shows."
            )
        return None

    # ── tool definitions (Anthropic tool-use schema) ─────────────────────────
    def tool_specs(self) -> list[dict]:
        sym_array = {"type": "array", "items": {"type": "string"}, "description": "Ticker symbols."}
        params_obj = {
            "type": "object",
            "description": "Strategy parameters (Params fields); omitted fields use defaults.",
        }
        return [
            _tool(
                "list_strategies",
                "The strategy library index: every authored .py with its thesis, status "
                "(draft/candidate/champion/rejected), style, researched symbols, current "
                "Params defaults, and param space.",
            ),
            _tool(
                "get_strategy",
                "Full Python source of one library strategy.",
                {"name": {"type": "string", "description": "Strategy (file) name."}},
                ["name"],
            ),
            _tool(
                "list_symbols",
                "Data-lake inventory: every tracked symbol with its bar count and covered "
                "date span, plus the configured universe.",
            ),
            _tool(
                "preview_bars",
                "Summary statistics and the most recent rows of one symbol's TRAINING window "
                "(the reserved forward-holdout tail is never shown). For sanity-checking a "
                "symbol's character against a thesis, not for fitting. Pass `timeframe` to "
                "preview at the granularity your strategy will actually see.",
                {
                    "symbol": {"type": "string"},
                    "rows": {
                        "type": "integer",
                        "description": f"Recent rows (default 20, max {_PREVIEW_ROW_CAP}).",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": f"Bar granularity (default '1m'): {sorted(TIMEFRAMES)}.",
                    },
                },
                ["symbol"],
            ),
            _tool(
                "screen_symbols",
                "Deterministic structural screen: state the symbol character the thesis needs "
                "and get the lake symbols that express it — the thesis picks the KIND of "
                "symbol, the data picks the tickers. Per-symbol features (trend efficiency, "
                "annualized volatility, dollar-volume liquidity) are computed on "
                "training-window bars and banded low/medium/high relative to the pool; "
                "matches are ranked by how strongly they express the profile, with a "
                "suggested fit panel and reserved_holdout names — keep those out of ALL "
                "tuning so you can nominate them as holdout_symbols at verdict time. Screens "
                "structure only, never strategy PnL: a match is character, not edge. Only "
                "lake-ready symbols are screened — discover + ensure_data new names first, "
                "then re-screen.",
                {
                    "trend": {
                        "type": "string",
                        "enum": [*BANDS, "any"],
                        "description": "Trend efficiency band: 'high' = one-way movers, "
                        "'low' = range-bound/choppy (mean-reversion character).",
                    },
                    "volatility": {
                        "type": "string",
                        "enum": [*BANDS, "any"],
                        "description": "Annualized realized-volatility band.",
                    },
                    "liquidity": {
                        "type": "string",
                        "enum": [*BANDS, "any"],
                        "description": "Median daily dollar-volume band.",
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit pool; default = every lake-ready "
                        "universe symbol.",
                    },
                },
            ),
            _tool(
                "get_champions",
                "The current champion board: family, params, test metric, gap, fit symbols.",
            ),
            websearch.client_tool_spec(
                "Search the public web for current, external context (news, filings, macro "
                "prints, corporate actions) to ground a thesis. Returns ranked "
                "{title, url, snippet} hits. Grounding only — never a substitute for the "
                "backtest gates. Returns an error result if the local search sidecar is down."
            ),
            _tool(
                "get_experiment_log",
                "One strategy's journaled trials ranked by out-of-sample test metric, plus "
                "counts (distinct param sets, sweep completed) toward the exhaustion gate. "
                "Pass `symbols` to compare like-for-like (only trials run on exactly that "
                "panel — e.g. baseline vs tuned on the full fit set).",
                {
                    "name": {"type": "string"},
                    "limit": {"type": "integer", "description": "Top trials (default 10)."},
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only trials run on exactly this symbol set.",
                    },
                },
                ["name"],
            ),
            _tool(
                "ensure_data",
                "Fetch missing history so [start, end] is covered for the symbols "
                "(coverage-diffed: already-covered ranges cost $0; refuses over the budget "
                "cap). Fetched symbols join the persistent universe automatically — the "
                "lake is the universe, so discovered names stay researchable and tradeable.",
                {
                    "symbols": sym_array,
                    "start": {"type": "string", "description": "ISO date, inclusive."},
                    "end": {"type": "string", "description": "ISO date, inclusive."},
                },
                ["symbols", "start", "end"],
            ),
            self._write_strategy_spec(),
            _tool(
                "run_backtest",
                "Evaluate one parameter set over the symbols (walk-forward + forward holdout); "
                "returns the aggregate scorecard only and journals the trial.",
                {
                    "name": {"type": "string"},
                    "symbols": sym_array,
                    "params": params_obj,
                },
                ["name", "symbols"],
            ),
            _tool(
                "run_sweep",
                "Optuna sweep over the strategy's param_space (optionally narrowed/overridden "
                "by `ranges`); every trial is journaled; returns the ranked top trials. "
                "Completing a sweep satisfies the exhaustion gate. Sweeps are cheapest on 2-3 "
                "representative symbols and a recent `max_bars` window — that is exploration "
                "fidelity, not judgment fidelity: confirm the winner with one run_backtest on "
                "the full fit panel before a verdict.",
                {
                    "name": {"type": "string"},
                    "symbols": sym_array,
                    "n_trials": {"type": "integer", "description": "Trials (default from config)."},
                    "ranges": {
                        "type": "object",
                        "description": "Optional per-param overrides: "
                        '{"param": {"low": ..., "high": ..., "step": ...}}.',
                    },
                    "max_bars": {
                        "type": "integer",
                        "description": "Optional speed cap: evaluate each trial on only the "
                        f"most recent N bars per symbol (min {_MAX_BARS_FLOOR}).",
                    },
                },
                ["name", "symbols"],
            ),
            _tool(
                "evaluate_vs_champion",
                "The APPROVE verdict: full-gate evaluation (walk-forward + temporal holdout + "
                "symbol holdout) and a champion-registry decision; on promotion the tuned "
                "params and status are written back into the strategy file. Refused until the "
                "parameter space is exhausted (min distinct trials or a completed sweep). "
                "Optionally nominate holdout_symbols — profile-matching names you deliberately "
                "kept OUT of all tuning; refused if any appears in this strategy's journal or "
                "the fit set. Omitted → the first ready universe names outside the fit set.",
                {
                    "name": {"type": "string"},
                    "symbols": sym_array,
                    "params": params_obj,
                    "holdout_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional symbol-holdout nominees: lake-ready names "
                        "never used in any of this strategy's backtests/sweeps.",
                    },
                },
                ["name", "symbols", "params"],
            ),
            _tool(
                "reject_strategy",
                "The REJECT verdict: stamp `status: rejected` in the file and record the dead "
                "end in memory so future sessions don't re-mine it. Refused until the "
                "parameter space is exhausted, and refused outright for a current champion "
                "(displace it with a better challenger via evaluate_vs_champion instead). If "
                "your post-mortem concludes the whole CLASS is "
                "dead (not just this parameterization), set class_exhausted=true and pass "
                "class_tag — that registers the class so the governor blocks re-mining it in "
                "future sessions (unless a genuinely new lever is named).",
                {
                    "name": {"type": "string"},
                    "reason": {"type": "string", "description": "Why the idea failed."},
                    "class_tag": {
                        "type": "string",
                        "description": "The class label (matching the write_strategy class_tag) "
                        "to register as exhausted. Falls back to the journaled tag if omitted.",
                    },
                    "class_exhausted": {
                        "type": "boolean",
                        "description": "True when the whole class — not just these params — is a "
                        "proven dead end. Registers class_tag against future re-mining.",
                    },
                },
                ["name", "reason"],
            ),
        ]

    def _write_strategy_spec(self) -> dict:
        """The write_strategy tool spec, switched by whether a coder model is configured.

        Exactly one authoring mode is ever visible to the driver. Default (no coder): the
        driver hand-writes ``source`` (required) — today's behavior, bit-for-bit. Coder mode:
        ``source`` is replaced as the required input by a ``brief`` object, so the driver must
        commit thesis/rules/params/scenarios before any code exists; ``source`` stays
        accepted-but-optional so a capable driver can still hand-write a revision. ``name`` /
        ``class_tag`` / ``new_lever`` and every write guard are shared by both modes.
        """
        base = (
            "Submit a strategy .py (new, or a revision of an existing one). Validated in "
            "an isolated interpreter: clean import, exactly one TraderStrategy subclass "
            "whose `name` equals the file name, docstring header, smoke replay on a "
            "synthetic fixture, signals/on_bar parity, and a replay of the file's own "
            "known-outcome scenarios: a `scenarios()` classmethod returning 2-8 Scenario "
            "objects built from the noctis.strategies.scenarios DSL (segments: flat/"
            "trend/selloff/recovery/chop/vol_spike/gap; expectations: flat_until/"
            "long_within/holds_long_through/short_within/holds_short_through/flat_by/"
            "always_flat). Targets are signed (+1 long, -1 short, 0 flat); include at "
            "least one tape demanding a directional entry (long OR short) and one "
            "always_flat() no-trade tape. Derive the tapes and windows from the THESIS "
            "(and from your Params defaults) before writing on_bar — code that violates "
            "its own declared scenarios is rejected. "
            "Rejected sources leave no file, and a rejection is a repairable code bug, NOT "
            "a verdict on the thesis: fix the reported error and resubmit the SAME name "
            "rather than authoring a new strategy. Tag every submission with a short "
            "`class_tag` "
            "naming the approach (e.g. 'per-symbol long/flat MA-cross overlay'); if that "
            "class was already declared exhausted (see MARKET REALITY exhausted_classes), the "
            "write is refused unless you pass `new_lever` naming what is materially new "
            "(a short leg, a session-time gate, a different symbol character)."
        )
        class_tag = {
            "type": "string",
            "description": "Short label for this strategy's CLASS/approach — the "
            "governor keys on it to prevent re-mining a proven dead end. Reuse an "
            "existing exhausted_classes tag verbatim when you deliberately extend it.",
        }
        new_lever = {
            "type": "string",
            "description": "Only when class_tag names an exhausted class: the "
            "materially-new dimension this attempt adds that the post-mortem did "
            "not cover (short leg / session-time gate / different symbol character).",
        }
        name = {"type": "string", "description": "lower_snake_case strategy name."}
        if self.coder_client is None:
            return _tool(
                "write_strategy",
                base,
                {
                    "name": name,
                    "source": {"type": "string", "description": "Complete Python source."},
                    "class_tag": class_tag,
                    "new_lever": new_lever,
                },
                ["name", "source"],
            )
        brief = {
            "type": "object",
            "description": "The research judgment a dedicated coder model turns into the file. "
            "It IS the research — commit thesis, rules, parameters, and expected scenario "
            "behavior here; the coder translates, never invents.",
            "properties": {
                "thesis": {
                    "type": "string",
                    "description": "The market inefficiency, in one paragraph.",
                },
                "entry_exit": {
                    "type": "string",
                    "description": "Precise long/short/flat rules.",
                },
                "param_space": {
                    "type": "string",
                    "description": "Tunable knobs + sensible ranges.",
                },
                "scenarios": {
                    "type": "string",
                    "description": "Sketch of the 2-8 known-outcome tapes the file must honor — "
                    "INTENT, not tape dictation: state each tape's shape and the behavior it must "
                    "prove, then leave tape construction to the coder. Do NOT dictate "
                    "indicator-level tape properties (exact amplitudes, warmup lengths, or that a "
                    "scale-free percentile rule 'stays quiet') the coder cannot honor.",
                },
                "reference": {
                    "type": "string",
                    "description": "Optional: a library strategy to adapt.",
                },
                "style": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["thesis", "entry_exit", "param_space", "scenarios"],
        }
        coder_desc = (
            base + " A dedicated coder model is configured: submit a `brief` (thesis, "
            "entry_exit, param_space, scenarios) and it authors the file for you — you never "
            "write source. A gate rejection after its private retries comes back as a "
            "repairable bug: refine the brief and resubmit the SAME name. `source` stays "
            "optional if you would rather hand-write a revision."
        )
        return _tool(
            "write_strategy",
            coder_desc,
            {
                "name": name,
                "brief": brief,
                "source": {
                    "type": "string",
                    "description": "Optional complete Python source — hand-write a revision "
                    "instead of briefing the coder.",
                },
                "class_tag": class_tag,
                "new_lever": new_lever,
            },
            ["name", "brief"],
        )

    def result_brief(self, result) -> dict:
        """The gate-facing slice of one tool result, in :data:`TOOL_LINE_KEYS` order — what
        the one-line feed prints and structures into the event meta. ``test_activity`` (is
        the metric real activity or a handful of noise trades?) lives one level down under
        ``trade_economics`` and is pulled up here."""
        if not isinstance(result, dict):
            return {}
        brief = {key: result[key] for key in self.TOOL_LINE_KEYS if key in result}
        econ = result.get("trade_economics")
        if isinstance(econ, dict) and "test_activity" in econ:
            brief["test_activity"] = econ["test_activity"]
        return brief

    # ── dispatch ─────────────────────────────────────────────────────────────
    def dispatch(self, name: str, args: dict) -> dict:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return handler(**(args or {}))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 — tool errors go back to the agent, not up
            logger.warning("tool %s failed: %s", name, exc)
            return {"error": str(exc)}

    # ── read tools ───────────────────────────────────────────────────────────
    def tool_list_strategies(self) -> dict:
        return {"strategies": library.list_strategies(self.strategies_dir)}

    def tool_web_search(self, query: str, max_results: int = 5) -> dict:
        """Client-side web grounding via the noctis-ollama search sidecar (localhost).

        Provider-neutral: any tool-capable model — including a keyless local Ollama one — can
        call it. Shared with ideation; the sidecar is the only thing that touches the network,
        so Noctis gains no new dependency (a stdlib HTTP GET lives in ``websearch``).
        """
        return websearch.search(query, max_results)

    def tool_get_strategy(self, name: str) -> dict:
        return {"name": name, "source": library.strategy_source(self.strategies_dir, name)}

    def tool_list_symbols(self) -> dict:
        records = []
        coverage = getattr(self.lake, "coverage", None)
        if coverage is not None:
            for rec in coverage.all():
                records.append(
                    {
                        "symbol": rec.symbol,
                        "dataset": rec.dataset,
                        "schema": rec.schema,
                        "rows": rec.row_count,
                        "status": rec.status,
                        "first": ns_to_date(rec.first_ts).isoformat() if rec.first_ts else None,
                        "last": ns_to_date(rec.last_ts).isoformat() if rec.last_ts else None,
                    }
                )
        return {"tracked": records, "universe": list(self.settings.universe)}

    @staticmethod
    def _feature_brief(features) -> dict:
        """One symbol's structural character, rounded for a tool result / the digest.
        Same keys in screen_symbols and market_context so the two reads can't diverge."""
        return {
            "trend_efficiency": _round(features.trend),
            "ann_volatility": _round(features.volatility),
            "day_dollar_volume_m": _round(features.liquidity / 1e6, 2),
        }

    def tool_screen_symbols(
        self,
        trend: str = "any",
        volatility: str = "any",
        liquidity: str = "any",
        symbols: list[str] | None = None,
    ) -> dict:
        from noctis.engine.runtime import trading_roster

        profile = validate_profile(
            {"trend": trend, "volatility": volatility, "liquidity": liquidity}
        )
        pool = [s.strip().upper() for s in (symbols or []) if s and s.strip()]
        if not pool:
            pool = trading_roster(self.settings, self.lake)
        pool = list(dict.fromkeys(pool))
        unready = [s for s in pool if not self.lake.check_symbol_ready(s)]
        computed = self.screener.features([s for s in pool if s not in set(unready)])
        unscreenable = sorted(s for s, f in computed.items() if f is None)
        result = screen([f for f in computed.values() if f is not None], profile)
        ranked = [m.features.symbol for m in result.matched]
        fit_n = self.settings.research.fit_set_size
        holdout_n = self.settings.research.symbol_holdout_size
        out = {
            "profile": profile,
            "pool_size": len(pool),
            "cutoffs": {
                dim: {edge: _round(v) for edge, v in cuts.items()}
                for dim, cuts in result.cutoffs.items()
            },
            "matched": [
                {
                    "symbol": m.features.symbol,
                    "strength": _round(m.strength),
                    "bands": m.bands,
                    "features": {
                        **self._feature_brief(m.features),
                        "timeframe": m.features.timeframe,
                        "bars": m.features.bars,
                    },
                }
                for m in result.matched
            ],
            "rejected_bands": result.rejected,
            "suggested_fit": ranked[:fit_n],
            "reserved_holdout": ranked[fit_n : fit_n + holdout_n],
            "note": (
                "Structural screen only — bands are pool-relative terciles of "
                "training-window character, never strategy PnL; a match is character, not "
                "edge. Keep reserved_holdout out of every backtest/sweep so they stay "
                "journal-clean to nominate as holdout_symbols at verdict time."
            ),
        }
        if unready:
            out["unready"] = {
                "symbols": unready,
                "hint": "not in the lake; ensure_data their history first, then re-screen",
            }
        if unscreenable:
            out["unscreenable"] = {
                "symbols": unscreenable,
                "hint": "training window too short for stable character features",
            }
        return out

    def tool_preview_bars(
        self, symbol: str, rows: int = 20, timeframe: str = NATIVE_TIMEFRAME
    ) -> dict:
        timeframe = validate_timeframe(timeframe)
        bars = self._bars_for([symbol], timeframe)[symbol.strip().upper()]
        config = self._pipeline_config({symbol: bars}, timeframe)
        cut = len(bars) - config.holdout_size if config.holdout_size else len(bars)
        train = bars.iloc[:cut]
        close = train["close"].astype("float64")
        returns = close.pct_change().dropna()
        rows = max(1, min(int(rows), _PREVIEW_ROW_CAP))
        recent = [
            {
                "ts": ns_to_timestamp(int(r.ts_event)).isoformat(),
                "open": round(float(r.open), 4),
                "high": round(float(r.high), 4),
                "low": round(float(r.low), 4),
                "close": round(float(r.close), 4),
                "volume": int(r.volume),
            }
            for r in train.tail(rows).itertuples(index=False)
        ]
        return {
            "symbol": symbol.strip().upper(),
            "timeframe": timeframe,
            "training_bars": int(len(train)),
            "holdout_bars_reserved": int(len(bars) - len(train)),
            "span": {
                "start": ns_to_date(int(train["ts_event"].iloc[0])).isoformat(),
                "end": ns_to_date(int(train["ts_event"].iloc[-1])).isoformat(),
            },
            "close": {
                "first": _round(close.iloc[0]),
                "last": _round(close.iloc[-1]),
                "min": _round(close.min()),
                "max": _round(close.max()),
            },
            "per_bar_return": {
                "mean": _round(returns.mean(), 6),
                "std": _round(returns.std(), 6),
            },
            "avg_volume": _round(train["volume"].mean(), 1),
            "recent_rows": recent,
        }

    def tool_get_champions(self) -> dict:
        entries = [
            {
                "family": e.family,
                "params": e.params,
                "test_metric": _round(e.test_metric),
                "gap": _round(e.gap),
                "crowned_at": e.crowned_at,
                "fit_symbols": e.fit_symbols,
            }
            for e in self.registry.list()
        ]
        return {"champions": entries, "capacity": self.registry.capacity}

    def tool_get_experiment_log(
        self, name: str, limit: int = 10, symbols: list[str] | None = None
    ) -> dict:
        trials = self.journal.trials_by_test(name)
        if symbols:
            wanted = sorted(s.strip().upper() for s in symbols if s and s.strip())
            trials = [t for t in trials if sorted(t.symbols) == wanted]
        stats = self.journal.stats(name)
        limit = max(1, min(int(limit), _LOG_LIMIT_CAP))
        out = {
            "strategy": name,
            "n_trials": stats.n_trials,
            "n_distinct_params": stats.n_distinct_params,
            "sweep_completed": stats.sweep_completed,
            "min_trials_gate": self.min_trials,
            "top_trials": [
                {
                    "params": t.params,
                    "symbols": t.symbols,
                    "source": t.source,
                    **({"max_bars": t.max_bars} if t.max_bars else {}),
                    **t.metrics,
                }
                for t in trials[:limit]
            ],
            "verdicts": self.journal.verdicts(name),
        }
        if symbols:
            out["filtered_symbols"] = wanted
            out["n_matching_trials"] = len(trials)
        return out

    # ── data tools ───────────────────────────────────────────────────────────
    def tool_ensure_data(self, symbols: list[str], start: str, end: str) -> dict:
        syms = [s.strip().upper() for s in symbols if s and s.strip()]
        if not syms:
            return {"error": "no symbols given"}
        results = self.lake.ensure_coverage(
            self.dataset, self.schema, syms, to_ns(start), to_ns_end_inclusive(end)
        )
        return {
            "results": {
                sym: {
                    "status": res.status,
                    "fetch_calls": res.fetch_calls,
                    "rows_added": res.rows_added,
                    "cost_usd": _round(res.padded_cost),
                    "detail": res.detail,
                }
                for sym, res in results.items()
            }
        }

    # ── write / experiment tools ─────────────────────────────────────────────
    def tool_write_strategy(
        self,
        name: str,
        source: str | None = None,
        brief: dict | None = None,
        class_tag: str | None = None,
        new_lever: str | None = None,
    ) -> dict:
        is_new = library.strategy_path(self.strategies_dir, name) is None
        # The guards that fire before any source exists (exhausted-class, undecided nudge) run
        # first for BOTH authoring paths — so the exhausted-class block spends zero coder
        # completions and delegation to the coder can never reopen a proven dead end.
        if class_tag and not new_lever:
            dead = self.exhausted.is_exhausted(class_tag)
            if dead is not None:
                return {
                    "error": (
                        f"exhausted-class guard: the class {class_tag!r} was already declared a "
                        f"dead end by a prior session ({dead.get('reason', '')}). Do not re-mine "
                        f"it. Either pursue a genuinely different class, or — if this attempt adds "
                        f"a lever the post-mortem did NOT cover (e.g. a short leg, a session-time "
                        f"gate, a different symbol character) — resubmit with new_lever naming "
                        f"exactly what is materially new."
                    )
                }
        # The coder Class-B budget gates only the brief path (a coder completion is about to be
        # spent). Checked before any completion — a refusal to START, source-based writes are
        # never gated — so an exhausted budget still leaves the driver a hand-written escape.
        if brief is not None and self.author_engine is not None:
            over_budget = self._author_budget_block()
            if over_budget:
                return {"error": over_budget}
        warning = None
        if is_new:
            pending = sorted(
                u for u in self.undecided if u != name and self._exhaustion_block(u) is not None
            )
            if pending:
                warning = (
                    f"strategy {pending[0]!r} is still undecided with an unexhausted parameter "
                    f"space — the protocol is to optimize it to a verdict "
                    f"(evaluate_vs_champion / reject_strategy) before formulating a new one. "
                    f"Proceeding anyway; make sure this is deliberate."
                )
        # Which authoring path this write took, computed the SAME way _author_source routes it —
        # a brief only goes to the coder when an engine exists. It steers the retry-exhaustion
        # repair guidance below: the driver holds source in the hand-written path but not in
        # brief mode, so "fix the source" is actionable in one and not the other.
        brief_mode = brief is not None and self.author_engine is not None
        # Both authoring paths converge here on a validated write result: a driver-supplied
        # `source` goes straight to the write gate, while a `brief` (coder mode) is turned into
        # source and validated by the StrategyAuthor engine (its private retries invisible).
        # Both surface a StrategyValidationError on a gate rejection, so the fixation/REPAIR
        # handling below and the success bookkeeping are shared, never duplicated.
        try:
            result = self._author_source(name, source, brief)
        except library.StrategyValidationError as exc:
            self._write_gate_failures += 1
            self._last_failed_write = name
            error = f"validation failed: {exc}"
            offending = _offending_line(source or "", str(exc))
            if offending:
                error += f" | offending line: {offending!r}"
            if self.backtests_run == 0 and self._write_gate_failures >= _WRITE_FIXATION_THRESHOLD:
                error += (
                    f" NOTE: {self._write_gate_failures} consecutive write-gate rejections and "
                    "no backtest yet this session — authoring is not converging. Switch to the "
                    "EXISTING library: pick a non-rejected strategy via list_strategies, "
                    "run_backtest its current params, then run_sweep it toward a verdict. "
                    "Revising a passing file is easier than writing one from scratch."
                )
            elif brief_mode:
                # Brief mode: the coder authored (and privately retried) the source — the driver
                # never sees it, so "fix the source" is not actionable. Point the driver at the
                # one artifact it does hold: the brief, and the scenario sketch in particular,
                # whose expected behavior most often contradicts the entry rules the coder honored.
                error += (
                    " REPAIR, don't abandon: the coder exhausted its private retries, but a "
                    "write-gate rejection is a code bug, not a verdict on the thesis — and in "
                    "brief mode you hold no source to fix. Refine the BRIEF and resubmit "
                    "write_strategy with the SAME name; look hardest at the scenario sketch, "
                    "whose expected behavior may contradict the entry rules the coder was given. "
                    "Do not start a new strategy over a validation error."
                )
            else:
                error += (
                    " REPAIR, don't abandon: a write-gate rejection is a code bug in this "
                    "draft, not a verdict on the thesis. Fix the reported error and resubmit "
                    "write_strategy with the SAME name and the full corrected source (nothing "
                    "was saved). Do not start a new strategy over a validation error."
                )
            return {"error": error}
        self._write_gate_failures = 0
        abandoned = self._last_failed_write
        self._last_failed_write = None
        if abandoned is not None and abandoned != name:
            note = (
                f"draft {abandoned!r} was left failing and unrepaired — a validation error is "
                f"a code bug, not a verdict on its thesis; if that thesis still stands, fix "
                f"and resubmit it."
            )
            warning = f"{warning} Also: {note}" if warning else note
        self.undecided.add(name)
        if name not in self.strategies_touched:
            self.strategies_touched.append(name)
        if class_tag:
            # Persist the class so a later reject_strategy can attribute it across sessions.
            self.journal.record_class_tag(name, class_tag)
        out = {"ok": True, **result}
        if warning:
            out["warning"] = warning
        return out

    def _author_source(self, name: str, source: str | None, brief: dict | None) -> dict:
        """Materialize a validated write result from a brief (coder mode) or hand-written source.

        Returns the :func:`library.write_strategy` result (name/path/header) either path lands,
        so :meth:`tool_write_strategy` runs one shared set of guards and one bookkeeping block.
        A brief authors through the coder engine; anything else requires source. Raises
        :class:`library.StrategyValidationError` on a gate rejection (brief mode re-surfaces the
        engine's final validation error), routing both paths through the shared REPAIR handling.
        """
        if brief is not None and self.author_engine is not None:
            return self._author_from_brief(name, brief)
        if not source:
            raise library.StrategyValidationError(
                "write_strategy needs `source` (or a `brief` when a coder model is configured)"
            )
        return library.write_strategy(self.strategies_dir, name, source, self.families)

    def _author_from_brief(self, name: str, brief: dict) -> dict:
        """Delegate to the coder engine: the driver's brief in, a validated file out.

        The engine makes stateless coder completions with private retries and lands the file
        through the same ``library.write_strategy`` gate the source path uses. Its
        :class:`AuthoringError` (retries exhausted) re-surfaces as the final
        :class:`library.StrategyValidationError`, so the caller's shared fixation/REPAIR path
        steers the driver to refine the brief and resubmit the same name.
        """
        engine = self.author_engine
        assert engine is not None  # only reached in coder mode (guarded by _author_source)
        parsed = StrategyBrief(
            thesis=brief["thesis"],
            entry_exit=brief["entry_exit"],
            param_space=brief["param_space"],
            scenarios=brief["scenarios"],
            reference=brief.get("reference"),
            style=brief.get("style"),
            symbols=tuple(brief.get("symbols") or ()),
        )
        try:
            return engine.author(name, parsed, on_attempt=self._author_attempt_sink(name))
        except AuthoringError as exc:
            error = exc.validation_error or library.StrategyValidationError(str(exc))
            raise error from exc

    def _author_attempt_sink(self, name: str):
        """Adapt the engine's per-attempt hook into (a) a session authoring event and (b) an
        on-disk failure record for ``name``.

        The engine calls this once per coder completion — private retries included — after that
        attempt's validation resolves, passing the attempt number, its outcome, and the attempted
        source. The toolbox owns both the event channel (the #9 telemetry, unchanged) and the
        capped ``failed/`` store: a rejected attempt's source and gate error are persisted so a
        bad session is inspectable from disk, while a landing attempt writes no failure record.
        """

        def sink(attempt: int, error: Exception | None, source: str) -> None:
            self._emit_author_event(name, attempt, error)
            if error is not None:
                self.failed_store.record(name, attempt, source, str(error))

        return sink

    def _emit_author_event(self, name: str, attempt: int, error: Exception | None) -> None:
        """Emit one ``author`` :class:`Event` for a resolved coder completion (#9).

        ``error is None`` ⇒ the attempt landed (``ok``); otherwise the validation error (a gate
        rejection or a non-code reply) is the attempt's outcome. So a watch session (``research
        -v``) sees authoring happen instead of a silent gap where a file appears from nowhere.
        """
        ok = error is None
        model = self.coder_model or "coder"
        outcome = "ok" if ok else str(error)
        text = f"author {name} · attempt {attempt} · {model} -> {outcome}"
        self._emit(
            Event(
                "author",
                text,
                meta={
                    "model": model,
                    "strategy": name,
                    "attempt": attempt,
                    "ok": ok,
                    "outcome": outcome,
                },
                level=1,
            )
        )

    def _emit(self, event: Event) -> None:
        """Hand one event to the session channel, or the logger when no sink is wired — the same
        Event-or-log contract the agent loop's default ``on_event`` sink uses."""
        if self.on_event is not None:
            self.on_event(event)
        else:
            logger.info("%s", render_plain(event))

    def tool_run_backtest(self, name: str, symbols: list[str], params: dict | None = None) -> dict:
        blocked = self._spend_backtests(1)
        if blocked:
            return {"error": blocked}
        self._require_strategy(name)
        resolved = self._resolved_params(name, params)
        bars = self._bars_for(symbols, self._timeframe_for(name))
        card = self._evaluate(name, resolved, bars)
        self.backtests_run += 1
        self.journal.record_trial(
            name,
            source="backtest",
            symbols=sorted(bars),
            params=resolved,
            window=self._window(bars),
            card=card,
        )
        stats = self.journal.stats(name)
        return {
            "strategy": name,
            "symbols": sorted(bars),
            "params": resolved,
            **self._card_summary(card, bars=bars),
            "journal": {
                "n_trials": stats.n_trials,
                "n_distinct_params": stats.n_distinct_params,
                "min_trials_gate": self.min_trials,
            },
            "backtests_remaining": self.max_backtests - self.backtests_run,
        }

    def tool_run_sweep(
        self,
        name: str,
        symbols: list[str],
        n_trials: int | None = None,
        ranges: dict | None = None,
        max_bars: int | None = None,
    ) -> dict:
        self._require_strategy(name)
        n = int(n_trials or self.default_sweep_trials)
        n = min(n, self.max_backtests - self.backtests_run)
        if n < 1:
            return {"error": self._spend_backtests(1)}
        space = self._sweep_space(name, ranges or {})
        bars = self._bars_for(symbols, self._timeframe_for(name))
        if max_bars is not None:
            max_bars = int(max_bars)
            if max_bars < _MAX_BARS_FLOOR:
                return {
                    "error": f"max_bars must be >= {_MAX_BARS_FLOOR} "
                    f"(walk-forward needs room for train+test+holdout); got {max_bars}"
                }
            bars = {s: df.tail(max_bars).reset_index(drop=True) for s, df in bars.items()}
        window = self._window(bars)

        results = []
        n_failed = 0
        config = self._pipeline_config(bars, self._timeframe_for(name))
        for params, card in self.sweep_runner.run(name, space, bars, n, config=config):
            self.backtests_run += 1
            if card is None:  # the trial itself errored; spend budget, learn nothing
                n_failed += 1
                continue
            self.journal.record_trial(
                name,
                source="sweep",
                symbols=sorted(bars),
                params=params,
                window=window,
                card=card,
                max_bars=max_bars,
            )
            results.append(
                {
                    "params": params,
                    "test": _round(card.avg_test_metric),
                    "gap": _round(card.gap),
                    "holdout": _round(card.holdout_metric),
                    "stage": card.stage,
                }
            )
        self.journal.record_sweep_complete(
            name, n_trials=len(results), symbols=sorted(bars), max_bars=max_bars
        )
        results.sort(
            key=lambda r: r["test"] if r["test"] is not None else float("-inf"), reverse=True
        )
        stats = self.journal.stats(name)
        out = {
            "strategy": name,
            "symbols": sorted(bars),
            "n_trials": len(results),
            "n_failed": n_failed,  # trials that errored: budget spent, nothing learned
            "sweep_completed": True,
            "top_trials": results[:10],
            "journal": {
                "n_trials": stats.n_trials,
                "n_distinct_params": stats.n_distinct_params,
            },
            "backtests_remaining": self.max_backtests - self.backtests_run,
        }
        if max_bars is not None:
            out["max_bars"] = max_bars
            out["note"] = (
                "trials ran on a truncated recent window — confirm the best params with "
                "run_backtest on the full fit panel before a verdict"
            )
        return out

    def _sweep_space(self, name: str, ranges: dict) -> list[ParamSpec]:
        space = {s.name: s for s in self.families.param_space(name)}
        unknown = set(ranges) - set(space)
        if unknown:
            raise ValueError(
                f"ranges names unknown params {sorted(unknown)}; param space has {sorted(space)}"
            )
        out = []
        for pname, spec in space.items():
            if pname in ranges:
                r = ranges[pname]
                spec = ParamSpec(
                    pname,
                    spec.kind,
                    low=r.get("low", spec.low),
                    high=r.get("high", spec.high),
                    step=r.get("step", spec.step),
                    choices=spec.choices,
                )
            out.append(spec)
        return out

    # ── verdict tools ────────────────────────────────────────────────────────
    def _symbol_holdout_for(
        self, fit_symbols: list[str], timeframe: str = NATIVE_TIMEFRAME
    ) -> dict[str, pd.DataFrame] | None:
        """Fallback held-out symbols: the first ``symbol_holdout_size`` ready names outside
        the fit set — data the tuning and symbol choice provably never touched. Aggregated
        to the challenger's timeframe so holdout and fit scores are like-for-like.

        Candidates are drawn from the research focus set first (the session's own
        enumeration), then the full trading roster as a backstop — narrowing the prompt's
        focus must never leave the gate short of holdout names (rule 4)."""
        from noctis.engine.runtime import research_focus, trading_roster

        size = self.settings.research.symbol_holdout_size
        if size <= 0:
            return None
        fit = {s.upper() for s in fit_symbols}
        pool = research_focus(self.settings, self.lake, self.mandate)
        seen = {s.upper() for s in pool}
        pool += [s for s in trading_roster(self.settings, self.lake) if s.upper() not in seen]
        held = [s for s in pool if s.upper() not in fit and self.lake.check_symbol_ready(s)][:size]
        if not held:
            return None
        raw = self.lake.get_bars(self.dataset, self.schema, held, 0, 2**63 - 1)
        out = {s: aggregate_bars(df, timeframe) for s, df in raw.items() if len(df) > 0}
        return out or None

    def _validate_nominated_holdout(
        self,
        name: str,
        holdout_symbols: list[str],
        fit: set[str],
        timeframe: str = NATIVE_TIMEFRAME,
    ) -> dict[str, pd.DataFrame]:
        """Structural check on agent-nominated holdout names: a nominee must be lake-ready,
        outside the fit set, and absent from EVERY journaled trial for this strategy — the
        journal is the proof the search never touched it."""
        nominated = [s.strip().upper() for s in holdout_symbols if s and s.strip()]
        overlap = sorted(set(nominated) & fit)
        if overlap:
            raise ValueError(
                f"holdout_symbols {overlap} are in the fit set — a holdout must be a symbol "
                f"the strategy was never tuned on"
            )
        tainted = sorted(set(nominated) & self.journal.touched_symbols(name))
        if tainted:
            raise ValueError(
                f"holdout_symbols {tainted} appear in {name!r}'s experiment journal — they "
                f"were used in tuning and cannot serve as a holdout; nominate names you "
                f"kept out of every backtest/sweep"
            )
        return self._bars_for(nominated, timeframe)  # errors on unready names

    def _unconfirmed_warning(self, name: str, resolved: dict, symbols: set[str]) -> str | None:
        """Soft nudge: the submitted params were never journaled on this full panel.

        A full-fidelity trial is one with no ``max_bars`` truncation whose symbol set covers
        the submitted panel and whose resolved params match exactly. Without one, the verdict
        is being spent on a score the agent never actually observed (subset/short-window
        sweeps are exploration fidelity, not judgment fidelity).
        """
        canonical = json.dumps(resolved, sort_keys=True)
        for trial in self.journal.trials(name):
            if trial.max_bars or not symbols <= set(trial.symbols):
                continue
            try:
                trial_resolved = self._resolved_params(name, trial.params)
            except Exception:  # noqa: BLE001 — a stale/malformed journal line can't confirm
                continue
            if json.dumps(trial_resolved, sort_keys=True) == canonical:
                return None
        return (
            "these params were never tested on this full panel — run_backtest them on the "
            "full fit set (baseline vs tuned) before spending a verdict"
        )

    def tool_evaluate_vs_champion(
        self,
        name: str,
        symbols: list[str],
        params: dict,
        holdout_symbols: list[str] | None = None,
    ) -> dict:
        self._require_strategy(name)
        blocked = self._exhaustion_block(name)
        if blocked:
            return {"error": blocked}
        resolved = self._resolved_params(name, params)
        timeframe = self._timeframe_for(name)
        bars = self._bars_for(symbols, timeframe)
        warning = self._unconfirmed_warning(name, resolved, set(bars))
        symbol_holdout: dict[str, pd.DataFrame] | None
        if holdout_symbols:
            symbol_holdout = self._validate_nominated_holdout(
                name, holdout_symbols, set(bars), timeframe
            )
        else:
            symbol_holdout = self._symbol_holdout_for(list(bars), timeframe)
        # Plan the whole promotion write-back BEFORE the registry can crown anything:
        # tuned params that violate the file's declared known-outcome scenarios must
        # refuse the verdict, not strand a crowned champion on a failed rewrite. The
        # plan gate-checks the finished champion file once; commit below installs it.
        try:
            plan = library.plan_promotion(
                self.strategies_dir,
                name,
                resolved,
                symbols=sorted(bars),
                tuned=datetime.now(UTC).date().isoformat(),
            )
        except library.StrategyValidationError as exc:
            return {
                "error": f"tuned params fail the strategy's declared known-outcome "
                f"scenarios: {exc}. Revise the strategy (write_strategy) or verdict "
                f"with params that honor the thesis behavior the file itself declares."
            }
        card = self._evaluate(name, resolved, bars, symbol_holdout=symbol_holdout)
        decision = self.registry.consider(card, self.rules, mandate_source=self.mandate_source)
        self.journal.record_approval(
            name,
            promoted=decision.promote,
            rationale=decision.rationale,
            params=resolved,
            symbols=sorted(bars),
            holdout_symbols=sorted(symbol_holdout) if symbol_holdout else [],
        )
        if decision.promote:
            self.promotions += 1
            self.undecided.discard(name)
            # The winning file lands in champions/ (moved out of the gitignored working
            # area, or copied out of the committed seed) with the tuned defaults and the
            # re-stamped header — pre-validated bytes, so the install cannot fail.
            plan.commit(self.families)
            self.memory.append_finding(
                f"PROMOTED {name} {json.dumps(resolved, sort_keys=True)} — {decision.rationale}"
            )
        out = {
            "strategy": name,
            "promoted": decision.promote,
            "rationale": decision.rationale,
            **self._card_summary(card, bars=bars),
            "symbol_holdout_symbols": sorted(symbol_holdout) if symbol_holdout else [],
        }
        if warning:
            out["warning"] = warning
        return out

    def tool_reject_strategy(
        self, name: str, reason: str, class_tag: str | None = None, class_exhausted: bool = False
    ) -> dict:
        self._require_strategy(name)
        if self._is_champion(name):
            return {
                "error": (
                    f"{name!r} is a current champion in the registry and cannot be rejected. A "
                    f"champion leaves the board only by being displaced — promote a better "
                    f"challenger with evaluate_vs_champion — never by reject_strategy. Rejecting "
                    f"it would stamp the file 'rejected' while it keeps trading, a split-brain "
                    f"the registry would ignore."
                )
            }
        blocked = self._exhaustion_block(name)
        if blocked:
            return {"error": blocked}
        trials = self.journal.trials_by_test(name)
        best_params = trials[0].params if trials else {}
        library.set_header(self.strategies_dir, name, families=self.families, status="rejected")
        self.memory.record_rejected(name, best_params, reason=reason)
        self.memory.append_finding(f"REJECTED strategy {name} — {reason}")
        self.journal.record_rejection(name, reason=reason, best_params=best_params)
        self.rejections += 1
        self.undecided.discard(name)
        result = {"ok": True, "strategy": name, "status": "rejected", "best_params": best_params}
        if class_exhausted:
            tag = class_tag or self.journal.class_tag(name)
            if tag:
                rec = self.exhausted.record(tag, reason, example=name)
                result["class_exhausted"] = rec["class_tag"]
            else:
                result["warning"] = (
                    "class_exhausted=True but no class_tag was given or journaled, so the class "
                    "was NOT registered — pass class_tag to record the dead class for future "
                    "sessions."
                )
        return result
