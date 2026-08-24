"""One RESEARCH entry, behind its own interface — the night's work and the one summary it returns.

Dispatching a RESEARCH phase is one cohesive job, and this is the one module that does it:
choose the path — the agent session the composition root assembles, or the legacy
proposer/Optuna loop that is also its no-key fallback — drive it over the panel the runtime
hands in, and count the completed session toward the periodic memory distillation. Both paths
return the same :class:`~noctis.engine.research.ResearchSummary`, which is exactly what lets one
interface stand in front of them (AGENTS.md, "two research paths, one contract"); the runtime
copies that summary into its cycle and its counters and never looks inside the loops.

**The bars are an argument, never state.** :class:`ResearchPanel` is the frozen triple a session
researches on — the fit set, the symbol holdout, and the split geometry/metric both are scored
under — which the runtime rebuilds from a fresh catalog read at each entry, the same way TRADING
rebuilds its bars. A session that follows a CLOSE-phase T+1 sync therefore researches the data
that sync brought in, and no session can be driven on a stale panel by construction. The two
sets stay deterministic from universe order, so a rebuild never rotates a held-out name into
selection (AGENTS.md rule 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.backtest.pool import evaluation_time_limit
from noctis.backtest.scorecard import Scorecard
from noctis.champions.promotion import PromotionRules
from noctis.engine.research import ResearchSummary, run_research
from noctis.observability import NULL_SINK, EventSink
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.proposer import CandidateProposer

if TYPE_CHECKING:
    from noctis.data.seam import MarketData
    from noctis.memory.base import Memory

# The RESEARCH phase narrates on the runtime's logger, like TRADING does: what an operator
# reads is one loop's story, not one story per phase module.
logger = logging.getLogger("noctis.runtime")


@dataclass(frozen=True)
class ResearchPanel:
    """The bars one RESEARCH entry researches on, and the geometry they are scored under.

    ``fit`` is the fit set — the symbols a candidate is tuned, validated and elected across.
    ``symbol_holdout`` are names the search never touches: scored, never selected on, so the
    promotion gate can tell an edge from a fit-set artifact. ``config`` is the one
    :class:`~noctis.backtest.PipelineConfig` (split geometry + election metric, sized from the
    fit set's shortest series) every symbol in both sets is scored under, which is what keeps
    per-symbol numbers comparable.

    Frozen because a session's view must not move underneath it: the runtime rebuilds the whole
    panel at the next entry rather than mutating this one.
    """

    fit: dict[str, pd.DataFrame] = field(default_factory=dict)
    symbol_holdout: dict[str, pd.DataFrame] = field(default_factory=dict)
    config: PipelineConfig = field(default_factory=PipelineConfig)


class ResearchPhase:
    """Run one RESEARCH entry: pick the path, drive it over the panel, count the session."""

    def __init__(
        self,
        *,
        settings,
        market_lake: MarketData,
        registry,
        families: FamilyRegistry,
        memory: Memory,
        proposer: CandidateProposer,
        rules: PromotionRules,
        mandate=None,
        ideator=None,
        research_max_iters: int | None = None,
        on_event: EventSink = NULL_SINK,
        stop_event=None,
    ):
        self.settings = settings
        self.market_lake = market_lake
        self.registry = registry
        self.families = families
        self.memory = memory
        self.proposer = proposer
        self.rules = rules
        # The resolved operator mandate (or None), threaded to each agent research session.
        self.mandate = mandate
        # LLM ideation seam (clientless/no-op when no key or [llm] extra). None → seed-only.
        self.ideator = ideator
        self.research_max_iters = research_max_iters
        # Where this phase's research feed goes — always a real sink, handed straight to the
        # session the composition root assembles. A bare run holds the quiet :data:`NULL_SINK`.
        self.on_event = on_event
        self.stop_event = stop_event

    def run(self, panel: ResearchPanel) -> ResearchSummary:
        """One research session over ``panel`` — agent if it can, legacy otherwise."""
        # The budget is real research time (backtests are wall-clock work even when the
        # session clock jumps), so both loops keep their default wall clock and the
        # wall-clock budget governs. research_max_iters is None in production (unbounded
        # for the legacy loop; the agent loop then uses its config cap); tests pass an
        # explicit cap to bound loops that finish instantly.
        summary = None
        if self.settings.research.mode == "agent":
            summary = self.run_agent_session()
        if summary is None:
            # Legacy proposer/Optuna loop — also the fallback when agent mode has no client.
            summary = run_research(
                proposer=self.proposer,
                evaluate_fn=lambda candidate: self.evaluate(candidate, panel),
                registry=self.registry,
                rules=self.rules,
                memory=self.memory,
                budget_minutes=self.settings.research_time_budget_minutes,
                stop_event=self.stop_event,
                max_iterations=self.research_max_iters,
                ideate=self.ideator.run if self.ideator is not None else None,
            )
        # One completed session toward the periodic memory distillation (runs at CLOSE, not
        # here — a research session's own loop never carries the summarization call).
        from noctis.research.distill import bump_research_session

        bump_research_session(self.settings.state_dir)
        return summary

    def run_agent_session(self) -> ResearchSummary | None:
        """One agent-driven session, or ``None`` to fall back to the legacy loop (no key).

        Public because it is an entry in its own right — ``run`` is one caller, and the tests
        that drive an agent session without a whole night around it are the others.
        """
        from noctis.bootstrap import build_research_session

        # The composition root assembles the same session bundle `noctis research` runs.
        # on_event tees the research feed into the run's console (a bare run's null sink drops
        # it): `run -v` shows the tool feed, `-vv`/`--show-reasoning` opens think/say — the
        # same streams `noctis research` surfaces, now visible from the day/night loop.
        session = build_research_session(
            settings=self.settings,
            lake=self.market_lake,
            registry=self.registry,
            families=self.families,
            memory=self.memory,
            mandate=self.mandate,
            rules=self.rules,
            on_event=self.on_event,
        )
        if session is None:
            logger.info("research.mode=agent but no research client; using legacy loop")
            return None
        # The provenance comes off the session's own resolved mandate (#260) — the session
        # already carries it, so the line names what steered this session rather than a copy
        # read back out of its toolbox.
        logger.info(
            "agent research session: mandate=%s, metric=%s",
            (session.mandate.source if session.mandate else None) or "(none)",
            self.settings.promotion.metric,
        )
        return session.run(
            max_iterations=self.research_max_iters,
            stop_event=self.stop_event,
        )

    def evaluate(self, candidate: Candidate, panel: ResearchPanel) -> Scorecard:
        """Score one candidate against ``panel`` — the legacy loop's ``evaluate_fn``.

        Same hang insurance as the toolbox's evaluation: a sequential (workers=1)
        evaluation has no pool stall guard, so bound it in wall-clock time. The legacy
        loop absorbs the EvaluationTimeout as a dead end and keeps running.
        """
        with evaluation_time_limit():
            return evaluate(
                candidate,
                panel.fit,
                config=panel.config,
                symbol_holdout=panel.symbol_holdout,
                workers=self.settings.research.agent.sweep_workers,
                families=self.families,
            )
