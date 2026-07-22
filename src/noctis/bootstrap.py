"""The composition root — every entrypoint assembles its session here.

Before this module, session assembly was scattered: the ``promotion.metric`` precedence
chain (``config.yaml`` → mandate overlay → ``--metric`` flag) spanned four files with its
ordering enforced by comments, :class:`~noctis.champions.promotion.PromotionRules` was
hand-built from settings in two places, and the CLI and the runtime each wired their own
copy of the agent research session (client + budgets + toolbox + loop kwargs).

Everything here is plain assembly, no policy: the safety gate, the overlay allowlist, and
the budget tables all stay with their owners (``config.gate``, ``research.mandate``,
``research.cost``). This module only fixes the *order* in one place and hands back built
collaborators. Errors are typed, never printed — the CLI maps them to red text + exit
codes; a library caller sees ordinary exceptions.

Heavy collaborators import at call time, mirroring the CLI convention (fast ``--help``)
and keeping test monkeypatching on the owning modules effective.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noctis.config import Settings, load_settings, resolve_execution_mode

logger = logging.getLogger("noctis.bootstrap")

if TYPE_CHECKING:
    from noctis.champions.promotion import PromotionRules
    from noctis.data.seam import MarketDataLake
    from noctis.engine.research import ResearchSummary
    from noctis.observability import Console, Event, EventTee
    from noctis.research import CostProfile, Mandate, ResearchToolbox
    from noctis.strategies.families import FamilyRegistry


class MissingVendorKey(RuntimeError):
    """A command that must fetch data was started without a vendor credential."""


class UsageError(ValueError):
    """Mutually-exclusive or unknown session flags. Distinct from :class:`ValueError` so a
    CLI handler never mistakes a pydantic ``ValidationError`` (also a ValueError) for usage."""


@dataclass(frozen=True)
class SessionInputs:
    """The resolved inputs of one session: settings after every override, plus provenance."""

    settings: Settings
    # The gate-resolved execution mode, or None when the entrypoint didn't ask for the gate
    # (research/report never place orders, so they don't arm it).
    mode: Literal["paper", "live"] | None
    mandate: Mandate | None
    # "k=v" echo lines for each mandate override actually applied (the CLI prints them).
    overrides: list[str]


def resolve_session(
    config_path: str | None = None,
    *,
    directive: str | None = None,
    mandate: str | None = None,
    metric: str | None = None,
    time_limit_hours: float | None = None,
    require_gate: bool = False,
) -> SessionInputs:
    """Resolve one session's settings by the one precedence order (docs/operator-mandate §5).

    ``load_settings`` → safety gate (when ``require_gate``) → ``resolve_mandate`` →
    ``apply_overrides`` → explicit CLI flags last, so a one-off ``--metric`` still wins over
    a mandate's overlay. Raises :class:`UsageError` on bad flags (both mandate selectors,
    an unknown metric), :class:`~noctis.research.MandateError` on an unresolvable selector,
    and :class:`~noctis.config.SafetyGateError` when the gate refuses — all before any
    long-running work starts.
    """
    from noctis.backtest.scorecard import Metric
    from noctis.research import apply_overrides, resolve_mandate

    if directive is not None and mandate is not None:
        raise UsageError("Pass either --directive or --mandate, not both.")
    if metric is not None:
        try:
            Metric.parse(metric)
        except ValueError as exc:  # the one diagnosis, re-typed as a usage error
            raise UsageError(str(exc)) from None

    settings = load_settings(config_path=config_path)
    mode = resolve_execution_mode(settings) if require_gate else None
    active = resolve_mandate(settings, cli_directive=directive, cli_mandate=mandate)
    overrides = apply_overrides(settings, active)
    if metric is not None:
        settings.promotion.metric = metric
    if time_limit_hours is not None:
        settings.time_limit_hours = time_limit_hours
    return SessionInputs(settings=settings, mode=mode, mandate=active, overrides=overrides)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy-layout guard
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LegacyArtifact:
    """A pre-workspace artifact the current configuration would silently abandon."""

    legacy: Path  # where the old (pre-workspace) layout kept it
    configured: Path  # where settings now point


def detect_legacy_layout(settings) -> list[LegacyArtifact]:
    """Find legacy (pre-workspace) artifacts the configured layout would orphan.

    Legacy artifacts are looked for next to the config file — the project root in the
    run-in-place model (``_yaml_path().parent``, so ``NOCTIS_CONFIG`` moves the search
    with it). An artifact is flagged when the old default path exists, the configured
    location differs, and the configured location does not exist: exactly the naive-
    upgrade case where a run would start against a silently-empty champion board while
    the real data sits abandoned. Explicitly pointing a knob at the legacy path is
    honored (nothing flagged). Callers map a non-empty result to a refusal that names
    ``noctis migrate``; ``status`` only warns.
    """
    from noctis.config.settings import _yaml_path

    root = _yaml_path().parent
    pairs = (
        (root / "state", Path(settings.state_dir)),
        (root / "data_lake", Path(settings.data.lake_dir)),
        (root / "reports", Path(settings.reports_dir)),
        (root / "MEMORY.md", Path(settings.memory_path)),
    )
    found: list[LegacyArtifact] = []
    for legacy, configured in pairs:
        if legacy.resolve() == configured.resolve():
            continue  # explicitly configured to the legacy location — intentional
        if legacy.exists() and not configured.exists():
            found.append(LegacyArtifact(legacy=legacy, configured=configured))
    return found


def scaffold_init(settings) -> list[str]:
    """Idempotent operator scaffold: local input files + the workspace. Never overwrites.

    Copies each committed template (config, env, mandate) to its local, gitignored name
    when — and only when — the local file doesn't exist yet, and creates the workspace
    root. Returns one human-readable line per action (created / kept / no template),
    which the CLI prints verbatim.
    """
    from noctis.config.settings import _yaml_path

    root = _yaml_path().parent
    lines: list[str] = []
    pairs = (
        (root / "config.example.yaml", root / "config.yaml"),
        (root / ".env.example", root / ".env"),
        (root / "mandate" / "MANDATE.md.example", root / "mandate" / "MANDATE.md"),
    )
    for template, target in pairs:
        if target.exists():
            lines.append(f"kept     {target} (already exists — your edits are safe)")
        elif not template.is_file():
            lines.append(f"skipped  {target} (no template {template.name} here)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(template.read_bytes())
            lines.append(f"created  {target} (from {template.name})")
    workspace = Path(settings.workspace_dir)
    if workspace.is_dir():
        lines.append(f"kept     {workspace} (already exists)")
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        lines.append(f"created  {workspace} (the one output root)")
    return lines


@dataclass(frozen=True)
class MigrationPlan:
    """What `noctis migrate` would do: clean moves, blocking conflicts, pinned skips."""

    moves: list[LegacyArtifact]
    conflicts: list[LegacyArtifact]  # legacy AND workspace copy both exist — refuse
    pinned: list[Path]  # a knob explicitly points at the legacy path — left in place


def plan_migration(settings) -> MigrationPlan:
    """Plan the one-shot move of every legacy artifact into the workspace.

    Covers the six legacy artifacts (state, lake, reports, root memory file, and the two
    strategy tiers beside the seeds), anchored next to the config file like
    :func:`detect_legacy_layout`. The local config never moves — it stays at the root,
    merely untracked. Pure planning: nothing on disk changes here.
    """
    from noctis.config.settings import _yaml_path
    from noctis.strategies.library import CHAMPIONS_SUBDIR, TMP_SUBDIR, LibraryPaths

    root = _yaml_path().parent
    tiers = LibraryPaths.from_settings(settings)
    pairs = (
        (root / "state", Path(settings.state_dir)),
        (root / "data_lake", Path(settings.data.lake_dir)),
        (root / "reports", Path(settings.reports_dir)),
        (root / "MEMORY.md", Path(settings.memory_path)),
        (root / "strategies" / TMP_SUBDIR, tiers.tmp),
        (root / "strategies" / CHAMPIONS_SUBDIR, tiers.champions),
    )
    moves: list[LegacyArtifact] = []
    conflicts: list[LegacyArtifact] = []
    pinned: list[Path] = []
    for legacy, configured in pairs:
        if not legacy.exists():
            continue
        if legacy.resolve() == configured.resolve():
            pinned.append(legacy)
        elif configured.exists():
            conflicts.append(LegacyArtifact(legacy=legacy, configured=configured))
        else:
            moves.append(LegacyArtifact(legacy=legacy, configured=configured))
    return MigrationPlan(moves=moves, conflicts=conflicts, pinned=pinned)


def execute_migration(plan: MigrationPlan) -> None:
    """Perform the planned moves. Call only on a conflict-free plan."""
    import shutil

    for artifact in plan.moves:
        artifact.configured.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(artifact.legacy), str(artifact.configured))


# ─────────────────────────────────────────────────────────────────────────────
# Collaborators
# ─────────────────────────────────────────────────────────────────────────────
def build_memory(settings):
    """The agent's long-term memory store (pure file I/O; LLM upkeep is the distillation step).

    Resolves ``settings.memory_path`` (workspace-derived unless overridden) and, on first
    run, seeds it from the committed ``MEMORY.seed.md`` — found next to the config file,
    like every committed input. The copy happens *before* the store constructs, because
    ``MemoryStore.load`` auto-creates its blank template for a missing file and would win
    the race. No seed ⇒ the blank template; never an error.
    """
    from noctis.config.settings import _yaml_path
    from noctis.memory import MemoryStore

    memory_path = Path(settings.memory_path)
    seed = _yaml_path().parent / "MEMORY.seed.md"
    if not memory_path.exists() and seed.is_file():
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_bytes(seed.read_bytes())
    return MemoryStore(memory_path)


def build_families(settings) -> FamilyRegistry:
    """The one strategy-family hydration: seeds → persisted spec-families → library files.

    The order is the correctness contract, fixed here once: spec-families re-register
    before anything reads the champion board (``champions.json`` stores only ``{family,
    params}``, so a promoted spec-family's class must exist before a champion builds),
    and the library files load last because they are the canonical versions of their
    families — tuned defaults live in the file and must win over any in-repo seed.
    """
    from noctis.strategies.families import FamilyRegistry
    from noctis.strategies.library import LibraryPaths
    from noctis.strategies.library import load_and_register as load_library
    from noctis.strategies.spec import load_and_register as load_specs

    families = FamilyRegistry()
    load_specs(settings.state_dir, families)
    load_library(LibraryPaths.from_settings(settings), families)
    return families


def build_lake(settings, *, require_vendor: bool = False) -> MarketDataLake:
    """Build a MarketDataLake, choosing a vendor from available credentials.

    Without a ``DATABENTO_API_KEY``: read-only commands get a placeholder vendor that
    refuses any fetch; a command that must fetch (``require_vendor=True``) raises
    :class:`MissingVendorKey` instead.
    """
    from noctis.data import MarketDataLake

    vendor: Any  # duck-typed seam: real vendor client or the read-only placeholder
    if settings.databento_api_key:
        from noctis.data.databento_provider import DataBentoVendorClient

        vendor = DataBentoVendorClient(settings.databento_api_key)
    elif require_vendor:
        raise MissingVendorKey("No DATABENTO_API_KEY set — cannot fetch data. Add it to .env.")
    else:
        vendor = _ReadOnlyVendor()
    return MarketDataLake(
        settings.data.lake_dir, vendor, settings.data.budget_usd, settings.session.calendar
    )


class _ReadOnlyVendor:
    """Placeholder vendor for read-only commands (status). Refuses any fetch."""

    def get_cost(self, **_kwargs):  # noqa: D401
        raise RuntimeError("read-only: no vendor configured")

    def fetch_bars(self, **_kwargs):
        raise RuntimeError("read-only: no vendor configured")


def _build_console(verbose: int, *, show_reasoning: bool = False) -> Console | None:
    """The level-aware console for ``-v``/``-vv``/``--show-reasoning``, or ``None`` on a
    quiet run — so downstream ``on_event=None`` keeps the loops on their own logger sinks."""
    if not verbose and not show_reasoning:
        return None
    from noctis.observability import Console

    return Console(verbose, show_reasoning=show_reasoning)


def build_event_sink(
    verbose: int,
    *,
    show_reasoning: bool = False,
    secondary: Callable[[Event | str], None] | None = None,
) -> Console | EventTee | None:
    """The session's ``on_event`` sink: the level-aware console, optionally teed to a recorder.

    With no ``secondary`` this is exactly the old console builder — a :class:`Console` when
    ``-v``/``-vv``/``--show-reasoning`` asks for one, else ``None`` so the loops fall back to their
    own logger sinks. With a ``secondary`` (a recorder-style event sink) it returns an
    :class:`~noctis.observability.EventTee` that renders on the console *and* feeds the recorder —
    **even when the console is absent**, so a quiet ``--debug`` run (no ``-v``, primary ``None``)
    still records every event. The secondary is typed generically as any event callable, so no
    recorder needs to exist yet."""
    console = _build_console(verbose, show_reasoning=show_reasoning)
    if secondary is None:
        return console
    from noctis.observability import EventTee

    return EventTee(console, secondary)


def build_console(verbose: int, *, show_reasoning: bool = False) -> Console | None:
    """Thin back-compat alias for :func:`build_event_sink` with no secondary — the level-aware
    console for ``-v``/``-vv``/``--show-reasoning``, or ``None`` on a quiet run. Existing callers
    and tests that only want a console keep this exact name, signature, and behavior."""
    return _build_console(verbose, show_reasoning=show_reasoning)


# ─────────────────────────────────────────────────────────────────────────────
# The --debug QA recorder
# ─────────────────────────────────────────────────────────────────────────────
# API keys the config digest must never fold in: the manifest lands under workspace/qa (gitignored),
# but digesting a vendor/LLM credential would still be leaking a secret (AGENTS.md rule 6).
_DIGEST_SECRET_FIELDS = frozenset({"databento_api_key", "anthropic_api_key", "openai_api_key"})


def build_recorder(settings, *, argv: list[str], mode: str | None):
    """Assemble the ``--debug`` QA recorder — the one place the run tree is minted (story #45).

    Prune-on-start first (retention per ``qa.keep_last_runs``), then mint a fresh run id and
    construct a :class:`~noctis.observability.debug.Recorder` under ``settings.qa_dir`` with a UTC
    wall-clock and the manifest fields the recorder cannot know itself: the CLI ``argv``, the run
    ``mode``, a deterministic config digest, and the noctis/python versions. The recorder owns run
    id and the started/stopped/duration stamps; everything else is injected here. The digest is
    taken over the *resolved* settings with API keys excluded (:data:`_DIGEST_SECRET_FIELDS`) so a
    credential can never ride into the report tree.
    """
    import hashlib
    import platform
    from datetime import UTC, datetime
    from importlib import metadata

    from noctis.observability.debug import Recorder, new_run_id, prune_qa_dir

    prune_qa_dir(settings.qa_dir, settings.qa.keep_last_runs)

    dump = settings.model_dump_json(exclude=set(_DIGEST_SECRET_FIELDS))
    config_digest = hashlib.sha256(dump.encode("utf-8")).hexdigest()[:12]

    try:
        noctis_version = metadata.version("noctis")
    except Exception:  # not pip-installed (editable/source tree) — fall back to the package literal
        from noctis import __version__ as noctis_version

    manifest = {
        "argv": list(argv),
        "mode": mode,
        "config_digest": config_digest,
        "versions": {"noctis": noctis_version, "python": platform.python_version()},
    }
    return Recorder(
        settings.qa_dir,
        run_id=new_run_id(),
        clock=lambda: datetime.now(UTC),
        manifest=manifest,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The agent research session
# ─────────────────────────────────────────────────────────────────────────────
# The episodic memory-distillation default (episodic-research epic #62): when the episodic loop
# is selected and the operator left ``research.memory_distill_every`` at its global-default 0,
# distillation defaults ON at this modest cadence. Applied as a per-session *effective value* on
# the shared settings instance in the loop-selection path (:meth:`ResearchSession.run`) — never a
# change to the class default — so a conversation-loop session's behavior stays bit-identical.
_EPISODIC_DISTILL_DEFAULT = 1

# The context window the episodic briefings assert against when the operator left
# ``research.agent.context_window`` unset. Generous so the build-time fit assertion is effectively
# inert (matching the conversation loop's unlimited history); an operator on a small-context
# backend sets ``context_window`` to engage the real discipline (the evidence-gated flip is #76).
_EPISODIC_CONTEXT_WINDOW = 128_000


def resolve_research_loop(settings) -> str:
    """Which research loop this session runs — ``"conversation"`` | ``"episodic"`` — from
    ``research.agent.loop``.

    ``"episodic"`` selects the deterministic episodic driver; ``"auto"`` (the default) and
    anything else resolve to the conversation loop in this story. The evidence-gated flip of
    ``"auto"`` to episodic-on-small-window lands in #76 — the one place that decision moves is
    this function, so the entrypoints never learn about it.
    """
    return "episodic" if settings.research.agent.loop == "episodic" else "conversation"


def effective_memory_distill_every(settings) -> int:
    """The memory-distillation cadence for this session: the operator's ``memory_distill_every``
    when set, otherwise the episodic default (#62) when the episodic loop is selected, else off.

    Pure — the loop-selection path applies it to the shared settings so the CLOSE-phase
    distillation reads the effective value with no change to the global default (the
    conversation loop keeps ``0`` = off, bit-identical to today)."""
    configured = int(settings.research.memory_distill_every or 0)
    if configured:
        return configured
    if resolve_research_loop(settings) == "episodic":
        return _EPISODIC_DISTILL_DEFAULT
    return 0


@dataclass
class ResearchSession:
    """One assembled agent research session: client + budgets + toolbox, ready to run.

    Built by :func:`build_research_session`; ``noctis research`` and the runtime's RESEARCH
    phase both run exactly this bundle, so their loop kwargs can never drift apart again. The
    loop that actually drives the session — the conversation transcript or the episodic driver —
    is resolved from ``research.agent.loop`` inside :meth:`run`, so both entrypoints follow the
    same selection without a code change.
    """

    settings: Settings
    toolbox: ResearchToolbox
    client: Any
    budgets: CostProfile
    mandate: Mandate | None
    on_event: Callable | None

    @property
    def model(self) -> str:
        """The resolved provider/model string this session will drive."""
        return self.settings.research.model or self.settings.research.agent.model

    def run(self, *, max_iterations: int | None = None, stop_event=None) -> ResearchSummary:
        """Run the session behind the ``research.agent.loop`` selector. ``max_iterations`` falls
        back to the cost-profile budget for either loop."""
        if resolve_research_loop(self.settings) == "episodic":
            # Apply the episodic memory-distillation default as a per-session effective value on
            # the shared settings (never the global default), so CLOSE distills on the episodic
            # cadence while a conversation session stays bit-identical.
            self.settings.research.memory_distill_every = effective_memory_distill_every(
                self.settings
            )
            return self._run_episodic(max_iterations=max_iterations, stop_event=stop_event)
        return self._run_conversation(max_iterations=max_iterations, stop_event=stop_event)

    def _run_conversation(self, *, max_iterations: int | None, stop_event) -> ResearchSummary:
        """The conversation loop — one long tool-use transcript. Unchanged from before the loop
        knob: byte-identical kwargs, so ``auto``/unset selects exactly today's behavior."""
        from noctis.research import run_agent_research

        agent_cfg = self.settings.research.agent
        return run_agent_research(
            toolbox=self.toolbox,
            client=self.client,
            budget_minutes=self.settings.research_time_budget_minutes,
            max_iterations=max_iterations or self.budgets.max_iterations,
            max_tokens=agent_cfg.max_tokens,
            context_window=agent_cfg.context_window,
            stop_event=stop_event,
            web_search=self.budgets.web_search,
            max_web_searches=self.budgets.max_web_searches,
            prefix_trim=self.budgets.prefix_trim,
            on_event=self.on_event,
            mandate=self.mandate,
        )

    def _run_episodic(self, *, max_iterations: int | None, stop_event) -> ResearchSummary:
        """The episodic driver — a deterministic session machine that calls the model only at
        narrow judgment episodes and executes everything else through the gated toolbox. The
        episode runner (which holds the client) and the ledger are assembled here and injected;
        the driver itself never sees the client. Returns the same summary shape as the
        conversation loop, so the runtime and the CLI are untouched."""
        from noctis.engine.runtime import trading_roster
        from noctis.research.driver import make_episodes, run_episodic_research
        from noctis.research.episode import EpisodeRunner
        from noctis.research.ledger import SessionLedger

        settings = self.settings
        agent_cfg = settings.research.agent
        runner_kwargs: dict[str, Any] = {}
        if agent_cfg.max_tokens:
            runner_kwargs["max_tokens"] = agent_cfg.max_tokens
        runner = EpisodeRunner(
            client=self.client,
            retries=agent_cfg.episode_retries,
            on_event=self.on_event,
            **runner_kwargs,
        )
        ledger = SessionLedger(settings.state_dir)
        context_window = agent_cfg.context_window or _EPISODIC_CONTEXT_WINDOW
        formulate, decide = make_episodes(
            runner=runner,
            toolbox=self.toolbox,
            ledger=ledger,
            mandate=self.mandate,
            context_window=context_window,
        )
        lake = self.toolbox.lake
        ready = [s for s in trading_roster(settings, lake) if lake.check_symbol_ready(s)]
        fit_symbols = ready[: settings.research.fit_set_size]
        return run_episodic_research(
            toolbox=self.toolbox,
            ledger=ledger,
            formulate=formulate,
            decide=decide,
            fit_symbols=fit_symbols,
            budget_minutes=settings.research_time_budget_minutes,
            max_episodes=max_iterations or self.budgets.max_iterations,
            completions=lambda: runner.completions,
            stop_event=stop_event,
            mandate_source=self.mandate.source if self.mandate else None,
            models={"driver": self.model, "coder": agent_cfg.coder_model},
            sweep_trials=self.toolbox.default_sweep_trials,
            on_event=self.on_event,
        )


def _build_coder_client(settings):
    """The dedicated strategy-authoring ("coder") client for ``research.agent.coder_model``, or
    ``None`` — inert in this story, threaded into the toolbox for a follow-up to consume.

    Unset (the default) ⇒ ``None``: the session driver authors full strategy source itself, and
    session assembly is unchanged. Set ⇒ a second, stateless per-model client is built alongside
    the driver via the shared :func:`~noctis.research.client_for` constructor. Thinking flips ON
    here (``research.agent.coder_thinking``, default on) because authoring — the scenario-window
    and warmup arithmetic — is the reasoning-heavy sub-task (#17); it is a *deliberate*, budgeted
    decision (``deliberate=True``), so even a Sonnet coder reasons, while the driver loop's own
    thinking pin is untouched (its cost stays bounded by the Class-B ``max_author_calls`` budget).
    If that client can't be built (its provider's key or the ``[llm]`` extra is missing) the
    degradation is loud, never silent: warn and fall back to ``None`` (driver-authored mode), so
    the session still assembles — the same graceful-degradation contract as the rest of the LLM
    seam, never a mid-session failure."""
    from noctis.research import client_for

    coder_model = settings.research.agent.coder_model
    if not coder_model:
        return None
    coder = client_for(
        settings,
        coder_model,
        thinking=settings.research.agent.coder_thinking,
        deliberate=True,
    )
    if coder is None:
        logger.warning(
            "coder_model %r is configured but no coder client could be built (its provider's "
            "API key or the [llm] extra is missing) — assembling in driver-authored mode; the "
            "session driver will write full strategy source itself. See docs/configuration.md.",
            coder_model,
        )
    return coder


def _build_coder_fallback_client(settings):
    """The PAID coder-fallback client for ``research.agent.coder_fallback_model`` (story #72), or
    ``None`` — the counted escalation target a spent local author falls back to.

    Escalation is a fallback FROM local authoring, so this is built only when BOTH a local
    ``coder_model`` and a ``coder_fallback_model`` are configured; either unset ⇒ ``None`` (no
    escalation path, and no wasted client). Built stateless beside the local coder through the
    shared :func:`~noctis.research.client_for` constructor with the same *deliberate*, budgeted
    thinking decision (``coder_thinking``, ``deliberate=True``) — the paid coder reasons through
    authoring just like the local one. If that client can't be built (its provider's key or the
    ``[llm]`` extra is missing) the degradation is loud, never silent: warn and fall back to
    ``None``, so the session still assembles and a failed local author is simply skipped as today
    — the same graceful-degradation contract as :func:`_build_coder_client`, never a mid-session
    failure. Bounded per session by ``research.agent.max_escalations`` (0 = never escalate)."""
    from noctis.research import client_for

    agent = settings.research.agent
    if not agent.coder_model or not agent.coder_fallback_model:
        return None
    fallback = client_for(
        settings,
        agent.coder_fallback_model,
        thinking=agent.coder_thinking,
        deliberate=True,
    )
    if fallback is None:
        logger.warning(
            "coder_fallback_model %r is configured but no fallback client could be built (its "
            "provider's API key or the [llm] extra is missing) — assembling with no escalation "
            "path; a failed local author will be skipped as today. See docs/configuration.md.",
            agent.coder_fallback_model,
        )
    return fallback


def build_research_session(
    *,
    settings,
    lake,
    registry,
    families: FamilyRegistry,
    memory,
    mandate: Mandate | None = None,
    rules: PromotionRules | None = None,
    on_event: Callable | None = None,
) -> ResearchSession | None:
    """Assemble one agent research session, or ``None`` when no LLM client is buildable
    (no key for the configured provider / the ``[llm]`` extra missing) — the caller decides
    whether that means an error (CLI) or the legacy-loop fallback (runtime)."""
    from noctis.champions.promotion import PromotionRules
    from noctis.research import ResearchToolbox, build_llm_client, resolve_budgets
    from noctis.strategies.library import LibraryPaths, prune_stale_drafts

    # Working-tier housekeeping (story #56): sweep stale, still-undecided drafts out of
    # __tmp/ into __tmp/archive/ *before* the toolbox constructs. The toolbox's init loads and
    # registers the library, so pruning first guarantees no session ever observes a stale
    # corpse mid-assembly. This is session assembly, so it runs regardless of which research
    # path (agent or legacy) the caller ends up choosing. Bounded by research.draft_ttl_hours;
    # None/0 is a no-op. Pure housekeeping — never a verdict or a gate (AGENTS.md rule 2).
    # Prior art: prune_qa_dir in build_recorder.
    archived = prune_stale_drafts(
        LibraryPaths.from_settings(settings).tmp,
        ttl_hours=settings.research.draft_ttl_hours,
    )
    if archived:
        logger.info(
            "pruned %d stale working-tier draft(s) before research assembly: %s",
            len(archived),
            ", ".join(archived),
        )

    client = build_llm_client(settings)
    if client is None:
        return None
    toolbox = ResearchToolbox(
        settings=settings,
        lake=lake,
        registry=registry,
        families=families,
        memory=memory,
        rules=rules if rules is not None else PromotionRules.from_settings(settings),
        mandate_source=mandate.source if mandate else None,
        mandate=mandate,
        coder_client=_build_coder_client(settings),
        coder_fallback_client=_build_coder_fallback_client(settings),
        on_event=on_event,
    )
    return ResearchSession(
        settings=settings,
        toolbox=toolbox,
        client=client,
        budgets=resolve_budgets(settings.research),
        mandate=mandate,
        on_event=on_event,
    )
