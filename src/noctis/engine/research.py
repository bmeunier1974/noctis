"""The legacy RESEARCH loop — propose → screen → validate → elect → remember.

Runs while the market is closed until the research time budget is spent or a stop event
fires (market open / time limit). Each iteration proposes a candidate, evaluates it through
the two-stage pipeline, applies the champion promotion rules, feeds the objective back to
the proposer, and records noteworthy findings. It orchestrates through seams only — no
engine/vectorbt imports — so it stays cheap and interruptible between iterations.

The runtime selects between this loop and the agent loop
(:func:`noctis.research.agent.run_agent_research`) via ``research.mode`` — both return the
same :class:`ResearchSummary`. This loop remains the fallback when agent mode has no
Anthropic client.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from noctis.backtest.pool import EvaluationTimeout
from noctis.backtest.scorecard import Scorecard
from noctis.champions.promotion import PromotionRules
from noctis.champions.registry import ChampionRegistry
from noctis.memory.base import Memory
from noctis.strategies.candidate import Candidate
from noctis.strategies.proposer import CandidateProposer

logger = logging.getLogger("noctis.research")


class StopEvent(Protocol):
    def is_set(self) -> bool: ...


class _NeverStop:
    def is_set(self) -> bool:
        return False


@dataclass
class ResearchSummary:
    iterations: int = 0
    promotions: int = 0
    rejections: int = 0
    dead_ends: int = 0
    stopped_reason: str = ""
    candidates: list[str] = field(default_factory=list)
    minted_specs: list[str] = field(default_factory=list)
    # Coder-model completions spent this session (0 without a configured coder_model). Surfaced
    # alongside the backtest count so a session report shows how much authoring the split did.
    author_calls: int = 0
    # Strategies authored this session but never carried to a verdict (promote/reject) — the agent
    # loop fills this from the toolbox's undecided set at session end (sorted); empty by default so
    # the legacy loop and existing constructors are unaffected. They are archived after the TTL.
    undecided: list[str] = field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def run_research(
    *,
    proposer: CandidateProposer,
    evaluate_fn: Callable[[Candidate], Scorecard],
    registry: ChampionRegistry,
    rules: PromotionRules,
    memory: Memory,
    budget_minutes: float,
    now: Callable[[], datetime] = _utcnow,
    stop_event: StopEvent | None = None,
    max_iterations: int | None = None,
    ideate: Callable[[int], list[str]] | None = None,
) -> ResearchSummary:
    """Run the research loop until the time budget, iteration cap, or a stop event.

    ``evaluate_fn`` maps a candidate to a ``Scorecard`` (normally
    ``noctis.backtest.evaluate`` bound to catalog bars + config). ``now`` and ``stop_event``
    are injectable so the state machine (and tests) can control timing and interruption.

    ``ideate`` (the :class:`~noctis.research.ideation.Ideator`) is called at each iteration
    boundary with the iteration index — once on the seed round (0) and every ``cadence``
    iters thereafter (the cadence lives in the Ideator, so this stays a thin orchestrator).
    It mints new spec-families as a side effect and returns their names, folded into the
    summary. Kept at the boundary so the loop stays interruptible.
    """
    stop_event = stop_event or _NeverStop()
    summary = ResearchSummary()
    start = now()
    budget_seconds = budget_minutes * 60.0

    while True:
        # Stop conditions are checked at the iteration boundary — no work is interrupted
        # mid-flight, so the registry and memory are always left consistent.
        if stop_event.is_set():
            summary.stopped_reason = "stop_event"
            break
        elapsed = (now() - start).total_seconds()
        if elapsed >= budget_seconds:
            summary.stopped_reason = "time_budget"
            break
        if max_iterations is not None and summary.iterations >= max_iterations:
            summary.stopped_reason = "max_iterations"
            break

        # Ideation runs at the boundary (before proposing) so a family minted this round can
        # be tuned in the very same iteration. The Ideator gates itself on cadence.
        if ideate is not None:
            summary.minted_specs.extend(ideate(summary.iterations))

        candidate = proposer.propose()
        try:
            scorecard = evaluate_fn(candidate)
        except EvaluationTimeout as exc:
            # A hung evaluation, bounded by the wall-clock guard. Nothing above this loop
            # catches research exceptions, so absorb it here: record the candidate as a
            # dead end and keep the loop alive — the budget/stop checks still govern.
            logger.warning("candidate %s evaluation hung (%s); rejecting", candidate.key(), exc)
            summary.iterations += 1
            summary.candidates.append(candidate.key())
            summary.dead_ends += 1
            memory.record_rejected(candidate.family, candidate.params, reason="hung_evaluation")
            proposer.reject(candidate)
            memory.append_finding(f"DEAD END {candidate.key()} — evaluation hung (timed out)")
            continue
        decision = registry.consider(scorecard, rules)
        proposer.tell(candidate, scorecard.avg_test_metric)

        summary.iterations += 1
        summary.candidates.append(candidate.key())

        if decision.promote:
            summary.promotions += 1
            memory.append_finding(f"PROMOTED {candidate.key()} — {decision.rationale}")
        elif getattr(scorecard, "stage", "") == "prefilter_rejected":
            summary.dead_ends += 1
            memory.record_rejected(candidate.family, candidate.params, reason="prefilter")
            proposer.reject(candidate)
            memory.append_finding(f"DEAD END {candidate.key()} — killed at pre-filter")
        else:
            summary.rejections += 1

        logger.info(
            "research iter=%d candidate=%s stage=%s promote=%s rationale=%s",
            summary.iterations,
            candidate.key(),
            getattr(scorecard, "stage", "?"),
            decision.promote,
            decision.rationale,
        )

    logger.info(
        "research loop finished: %d iters, %d promotions, %d rejections, %d dead ends (%s)",
        summary.iterations,
        summary.promotions,
        summary.rejections,
        summary.dead_ends,
        summary.stopped_reason,
    )
    return summary
