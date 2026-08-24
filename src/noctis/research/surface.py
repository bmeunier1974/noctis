"""The research-toolbox surface — one typed seam every reader of the toolbox holds.

The agent session carries exactly one object for its whole life: the research toolbox
(:class:`noctis.research.tools.ResearchToolbox`). Briefings, the system prompt, the episodic
driver, the conversation loop, the CLI and the eval layer all read *facts* off it — the champion
board, the library index, the market economics, a candidate's journaled evidence — and until this
module existed they each reached *through* it into whichever collaborator happened to hold the
answer (``toolbox.journal``, ``toolbox.registry.capacity``, ``toolbox.lake.preflight.budget_usd``),
often behind a ``getattr(…, default)`` probe that quietly invented a fact when the reach missed.

The seam is declared here in **two tiers**, because two different things are being asked for:

* :class:`ResearchFacts` — what a *renderer* reads. A briefing, the system prompt or an eval
  fixture wants the derived facts and nothing else: no tools, no dispatch, no session mutation.
  Every fact is **always answerable** — a lake with no cost preflight answers ``None``, a lake
  that cannot list its coverage answers an empty inventory — so a reader never probes and never
  fabricates.
* :class:`Toolbox` — what a *driver* holds: the facts, plus the tool registry it dispatches
  through, the capture seams it records inputs with, and the session counters it reports.

Two dataclasses carry values that were previously read as loose scalars — :class:`ResearchLimits`
(the four Class-B ceilings, resolved once at construction) and :class:`ChampionBoard` (rows,
crowned families and the slot count that were three separate reaches) — and :class:`SessionCounters`
is a **snapshot**: the toolbox's live counters stay mutable attributes its own tools bump, and each
call here freezes their current values, so a reporter cannot hold a view that mutates under it.

Two properties of this module are load-bearing:

* **It is a leaf.** Dataclasses and typing only — it imports nothing from the engine. The eval
  layer imports it, and the engine never imports the eval layer
  (:mod:`noctis.eval.guard`, ``tests/test_eval_boundary.py``), so a shared vocabulary between the
  two has to live somewhere neither owns.
* **It declares a surface, not a hiding place.** The collaborators (journal, registry, memory,
  lake, families, library paths, capture store, settings, …) stay plain public attributes on the
  toolbox: "private" here means "not on the surface", never an underscore rename. A test still
  seeds state by writing through the real collaborator, which is why the surface tests can drive
  the production object rather than a stub of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

__all__ = [
    "ChampionBoard",
    "ResearchFacts",
    "ResearchLimits",
    "SessionCounters",
    "Toolbox",
]


@dataclass(frozen=True)
class ResearchLimits:
    """The session's four ceilings, resolved once and read as one value.

    ``min_trials`` is the exhaustion floor — *quality*, not cost, so no cost profile ever touches
    it (AGENTS.md rule 2). The other three are the Class-B budgets the active ``cost_profile``
    scales together (:mod:`noctis.research.cost`). They travel as one frozen value because every
    reader wants a consistent set: a briefing that showed one session's backtest ceiling beside
    another's trial floor would be describing a session that does not exist.
    """

    min_trials: int
    max_backtests: int
    sweep_trials: int
    max_author_calls: int


@dataclass(frozen=True)
class ChampionBoard:
    """The board a candidate must beat: the rows, the families already holding a slot, and how
    many slots there are.

    The three used to be three separate reaches into the champion registry, which is how a
    surface renders a board whose row count and capacity disagree. ``crowned_families`` is not
    derivable from ``rows`` by a caller that only has the rendered rows — it is deduplicated and
    sorted, so a board that seats one family twice still names it once — and it is gate-facing:
    the one-slot-per-family rule refuses a re-tune of any name in it.
    """

    rows: tuple[dict[str, Any], ...] = ()
    crowned_families: tuple[str, ...] = ()
    capacity: int = 0


@dataclass(frozen=True)
class SessionCounters:
    """What one research session has spent and concluded, frozen at the moment of asking.

    A snapshot, deliberately: the toolbox's live counters are mutable attributes its own tools
    bump as the session runs, and a summary, a ledger row or a run record that captured "the
    counters" by reference would silently re-read them later and report a different session than
    the one it was describing. ``strategies_touched`` and ``undecided`` freeze into a tuple and a
    frozenset for the same reason — the collections themselves keep growing on the toolbox.

    ``undecided`` is the session's **names with no verdict spent yet**: a candidate joins it when
    it is authored and leaves it the moment a verdict is arbitrated, whatever that verdict said —
    a rejection, or an approve the promotion gates then refused. What leaves a name undecided is
    that nothing judged it, not that nothing crowned it (epic #326).
    """

    backtests_run: int = 0
    promotions: int = 0
    rejections: int = 0
    author_calls: int = 0
    escalations: int = 0
    strategies_touched: tuple[str, ...] = ()
    undecided: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class ResearchFacts(Protocol):
    """What a renderer reads: the derived facts, and nothing that could change the session.

    Everything here is a *read* — safe to call from a briefing builder, the system prompt, or an
    eval fixture rebuilding a past prompt. Every method answers: there is no probe to write on the
    caller's side and no fact that is sometimes missing, because the honest degradations (no cost
    preflight, an unreadable coverage registry) are answers, not absences.
    """

    #: The session's four ceilings, fixed at construction.
    limits: ResearchLimits

    def market_context(self) -> dict:
        """The MARKET REALITY economics digest: cost hurdle, timeframes, per-symbol character."""
        ...

    def journal_evidence(self, name: str) -> dict:
        """One candidate's gate-facing evidence out of the experiment journal: exhaustion stats,
        the ranked top trials, journaled verdicts and the holdout-taint symbol set."""
        ...

    def champion_board(self) -> ChampionBoard:
        """The board, the crowned families, and the slot count — as one consistent value."""
        ...

    def library_index(self) -> list[dict]:
        """The strategy library index, every ``rejected`` entry collapsed to a stub."""
        ...

    def template_text(self) -> str:
        """The shipped ``TEMPLATE.py`` seed the strategy-file contract embeds, or ``"(none)"``."""
        ...

    def memory_tail(self, *, prefix_trim: bool = False) -> tuple[list, list]:
        """The advisory memory tail as ``(findings, dead_ends)``; ``prefix_trim`` caps it."""
        ...

    def lake_inventory(self, *, limit: int = 60) -> list[str]:
        """The tickers this lake can already research — sorted, readiness-filtered, capped."""
        ...

    def data_budget(self) -> float | None:
        """The configured data-spend budget in USD, or ``None`` when the lake has no preflight."""
        ...


@runtime_checkable
class Toolbox(ResearchFacts, Protocol):
    """What a driver holds: the facts, plus the tools, the capture seams and the counters.

    The tool methods are named individually rather than hidden behind :meth:`dispatch` because the
    episodic driver calls them directly — a stage's deterministic action is a method call, not a
    model-chosen tool name — and a seam that only promised "some dispatch" would let a rename of a
    tool the driver depends on typecheck.
    """

    #: The tools that end a strategy: their results are the session's durable conclusions.
    VERDICT_TOOLS: ClassVar[frozenset[str]]
    #: The tools whose per-strategy results are re-fetchable from the experiment journal.
    STRATEGY_HISTORY_TOOLS: ClassVar[frozenset[str]]

    # ── the registry the model drives ────────────────────────────────────────
    def tool_specs(self) -> list[dict]:
        """The tool schemas offered to the model this session."""
        ...

    def dispatch(self, name: str, args: dict) -> dict:
        """Run one model-named tool; an unknown name or a failure comes back as ``{"error": …}``."""
        ...

    def result_brief(self, result: Any) -> dict:
        """The gate-facing slice of one tool result — what the one-line feed prints."""
        ...

    # ── the tools a driver calls by name ─────────────────────────────────────
    def tool_screen_symbols(
        self,
        trend: str = "any",
        volatility: str = "any",
        liquidity: str = "any",
        symbols: list[str] | None = None,
    ) -> dict: ...

    def tool_ensure_data(self, symbols: list[str], start: str, end: str) -> dict: ...

    def tool_write_strategy(
        self,
        name: str,
        source: str | None = None,
        brief: dict | None = None,
        class_tag: str | None = None,
        new_lever: str | None = None,
        thesis: str | None = None,
        parent_thesis: str | None = None,
        pivot_rationale: str | None = None,
        spec: Any | None = None,
    ) -> dict: ...

    def tool_run_backtest(
        self, name: str, symbols: list[str], params: dict | None = None
    ) -> dict: ...

    def tool_run_sweep(
        self,
        name: str,
        symbols: list[str],
        n_trials: int | None = None,
        ranges: dict | None = None,
        max_bars: int | None = None,
    ) -> dict: ...

    def tool_get_experiment_log(
        self, name: str, limit: int = 10, symbols: list[str] | None = None
    ) -> dict: ...

    def tool_evaluate_vs_champion(
        self,
        name: str,
        symbols: list[str],
        params: dict,
        holdout_symbols: list[str] | None = None,
    ) -> dict: ...

    def tool_reject_strategy(
        self, name: str, reason: str, class_tag: str | None = None, class_exhausted: bool = False
    ) -> dict: ...

    # ── authoring preflight + the observability capture seams ────────────────
    def can_author_brief(self) -> bool:
        """Whether the coder budget could still fund one more authoring job."""
        ...

    def capture_brief(self, brief: Mapping[str, Any], spec: Any | None = None) -> dict[str, str]:
        """Persist one authoring job's inputs; returns the hashes a ledger row records."""
        ...

    def capture_episode(self, briefing: str, knobs: Mapping[str, Any]) -> dict[str, str]:
        """Persist one episode's rendered briefing and knob snapshot; returns their hashes."""
        ...

    # ── the two delegating facts a driver gates on ───────────────────────────
    def symbol_ready(self, symbol: str) -> bool:
        """Whether the lake's coverage for ``symbol`` is complete enough to research on."""
        ...

    def class_exhausted(self, tag: str) -> dict | None:
        """The record proving ``tag``'s class was declared a dead end, or ``None``."""
        ...

    # ── what the session spent ───────────────────────────────────────────────
    def session_counters(self) -> SessionCounters:
        """A fresh frozen snapshot of the live session counters."""
        ...
