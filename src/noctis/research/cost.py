"""The ``cost_profile`` knob (#12): one engine-level lever that scales the **Class-B**
research budgets together.

The throttle values live *here*, in a profile table — never hardcoded lower in the defaults.
"Reset to full research" is a single config line (``research.cost_profile: full``); nothing
about a throttled run is baked in anywhere else. The knob binds *resource ceilings* only —
rounds, backtests/sweep-trials, web searches, reasoning effort, prompt-prefix trim — and
**never** touches a promotion gate or the ``research.min_trials`` exhaustion floor (those are
quality, not cost; AGENTS.md rules 2/4).

Three profiles (all provider-neutral; provider-specific levers like ``effort`` stay additionally
capability-gated at send time):

* ``balanced`` — the shipped default; **exactly today's ceilings**, so adding the knob changes
  nothing until an operator opts into ``economy``.
* ``economy`` — the reduced ceilings for minimizing paid-API spend.
* ``full`` — maximums; may only *raise* budgets. Auto-selected on a **free/local** provider
  (a ``$0``/token backend has no reason to throttle), overridable by an explicit config value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from noctis.research.llm import provider_of


@dataclass(frozen=True)
class CostProfile:
    """A named set of Class-B research ceilings. Immutable — the table is the single source."""

    name: str
    max_iterations: int  # tool-use rounds per session
    max_backtests: int  # run_backtest calls + individual run_sweep trials
    sweep_trials: int  # default trials for one run_sweep call
    web_search: bool  # offer server-side web_search (still capability-gated per provider)
    max_web_searches: int  # cap on server web searches per session
    effort: str  # reasoning effort ("high" | "medium"); capability-gated at send
    prefix_trim: bool  # cap the memory/library slices embedded in the system prefix


# The table. ``full`` == ``balanced`` numerically (the PRD lists full as "40 (or higher)/200+");
# they differ only in intent and in ``full`` being the free/local auto-default. ``full`` never
# sits *below* ``balanced`` on any budget — it may only raise, never lower (see rules below).
FULL = CostProfile(
    name="full",
    max_iterations=40,
    max_backtests=200,
    sweep_trials=20,
    web_search=True,
    max_web_searches=8,
    effort="high",
    prefix_trim=False,
)
BALANCED = CostProfile(
    name="balanced",
    max_iterations=40,
    max_backtests=200,
    sweep_trials=20,
    web_search=True,
    max_web_searches=8,
    effort="high",
    prefix_trim=False,
)
ECONOMY = CostProfile(
    name="economy",
    max_iterations=20,
    max_backtests=80,
    sweep_trials=10,
    web_search=True,
    max_web_searches=4,
    effort="medium",
    prefix_trim=True,
)

PROFILES: dict[str, CostProfile] = {"full": FULL, "balanced": BALANCED, "economy": ECONOMY}


def is_free_local(provider: str) -> bool:
    """Whether the provider is a ``$0``/token local/self-hosted backend (no paid cloud key).

    The same signal #11 uses to build a keyless client: anything that is not one of the two paid
    clouds — ``ollama/…``, ``vllm/…``, ``lm_studio/…``, any OpenAI-compatible custom prefix."""
    return provider not in ("anthropic", "openai")


def resolve_cost_profile(research) -> CostProfile:
    """Pick the active :class:`CostProfile` for a research config.

    ``research.cost_profile`` names it (default ``balanced``). On a **free/local** provider the
    *default* ``balanced`` — the paid ceilings — auto-upgrades to ``full``: a ``$0``/token model
    has no reason to run the throttled paid profile (and ``full`` is numerically ≥ ``balanced``,
    so the upgrade never removes research). An **explicit** ``economy`` or ``full`` is honored, so
    an operator can still throttle a free model on purpose."""
    model = getattr(research, "model", None) or research.agent.model
    name = research.cost_profile
    if name == "balanced" and is_free_local(provider_of(model)):
        name = "full"
    return PROFILES[name]


def resolve_budgets(research) -> CostProfile:
    """The effective Class-B budgets for one session: the active profile, with any
    explicitly-set ``research.agent`` budget (a non-``None`` field) pinning that one knob.
    ``effort``/``prefix_trim`` are profile-only (no per-knob override field). Same
    :class:`CostProfile` shape — ``name`` still says which profile the budgets came from."""
    profile = resolve_cost_profile(research)
    agent = research.agent

    def pinned(field: str, default):
        value = getattr(agent, field, None)
        return value if value is not None else default

    return replace(
        profile,
        max_iterations=pinned("max_iterations", profile.max_iterations),
        max_backtests=pinned("max_backtests", profile.max_backtests),
        sweep_trials=pinned("sweep_trials", profile.sweep_trials),
        web_search=pinned("web_search", profile.web_search),
        max_web_searches=pinned("max_web_searches", profile.max_web_searches),
    )
