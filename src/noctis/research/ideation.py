"""LLM ideation of new ``StrategySpec`` families.

The model invents new *strategy structures* — emitted as ``StrategySpec`` JSON through a
forced function tool — which are validated, parity-gated, registered as ordinary families,
and then flow through the same proposer/Optuna → prefilter → validate → promotion pipeline
as any seed family. The LLM never sees the data or the objective; it only proposes structure.

Provider-neutral by the same seam as agent research: everything here talks to
:class:`~noctis.research.llm.LLMClient` (``ideation.model`` picks provider/model), so the
emit tool runs on any backend. Web-search grounding follows the provider — Anthropic serves
it server-side; every other backend (OpenAI, any keyless local model) uses the client-side
sidecar tool (see :mod:`noctis.research.websearch`), dispatched in ``_run_ideation_call``.

Graceful by construction. No ``[llm]`` extra, or no key for the configured provider → no
client → :func:`propose_specs` returns ``[]`` and the :class:`Ideator` mints nothing, so
research runs seed-families-only exactly as before. This mirrors the no-key short-circuit +
broad try/except in ``noctis/memory/store.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from noctis.research import websearch
from noctis.research.llm import WEB_SEARCH_TOOL_TYPE, Turn, client_for, effective_web_search
from noctis.strategies.base import replay_targets
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.library import fixture_frame
from noctis.strategies.spec.schema import (
    DEFAULT_MAX_INDICATORS,
    StrategySpec,
    validate_spec,
)
from noctis.strategies.spec.strategy import (
    family_class_from_spec,
    persisted_spec_json,
    register_spec,
)

logger = logging.getLogger("noctis.ideation")

_TOOL_NAME = "emit_strategies"
_MAX_TOKENS = 8000


# ─────────────────────────────────────────────────────────────────────────────
# Prompt material — the primitive vocabulary + few-shot StrategySpec examples
# (ported from grid-mng ai/spec/specFixtures.ts, trimmed to the +1/0 subset).
# ─────────────────────────────────────────────────────────────────────────────
VOCABULARY = """\
A StrategySpec is a directed graph: sources → features → signals → entries. Wire nodes by id
(a bare "id", or "id:port" for multi-output features). The compiled strategy is LONG/FLAT only
(target 1 = long, 0 = flat) — never propose shorts.

sources:      {"id","schema"}  e.g. {"id":"src","schema":"ohlcv-1m"}. A source ref yields close.
features (each has "id","kind","input"):
  sma|ema|rsi|atr|vwap : + "period"
  macd                 : + "fastPeriod","slowPeriod","signalPeriod"; ports :macd :signal :histogram
  rollingExtreme       : + "mode"(max|min),"period","field"(high|low|close)
  zScore               : + "lookback","upperThreshold","lowerThreshold"; boolean ports :above :below
  seriesOp             : + "op","a","b" (elementwise add|sub|mul|div)
signals (boolean):
  condition : {"id","kind":"condition","op","a", and ("b" | "threshold")}
              op ∈ >,>=,<,<=,cross_above,cross_below
  ensemble  : {"id","kind":"ensemble","method"(and|or),"inputs":[signal ids]}
entries:      {"id","enter":<boolean signal or zScore:below>,
               "exit":<boolean signal or zScore:above>}
parameters:   {"id","kind"(int|float),"value"} — a tunable feature/threshold field may be the
              param id STRING instead of a number, which binds it to that parameter.
optimizations:{"id","parameters":[{"param","type"(int|float),"min","max","step"}]} — search ranges.

Rules: every ref must resolve; no cycles; an entry's enter/exit must be a boolean node; keep the
design small (a handful of features). Prefer ideas that differ structurally from the current
champions and avoid re-proposing known dead ends."""

FEW_SHOT = [
    {
        "version": 1,
        "id": "example_sma_crossover",
        "name": "SMA crossover",
        "description": "Long while a fast SMA is above a slow SMA.",
        "sources": [{"id": "src", "schema": "ohlcv-1m"}],
        "parameters": [
            {"id": "fast", "kind": "int", "value": 10},
            {"id": "slow", "kind": "int", "value": 30},
        ],
        "features": [
            {"id": "f_fast", "kind": "sma", "input": "src", "period": "fast"},
            {"id": "f_slow", "kind": "sma", "input": "src", "period": "slow"},
        ],
        "signals": [
            {"id": "enter", "kind": "condition", "op": ">", "a": "f_fast", "b": "f_slow"},
            {"id": "exit", "kind": "condition", "op": "<=", "a": "f_fast", "b": "f_slow"},
        ],
        "entries": [{"id": "e", "enter": "enter", "exit": "exit"}],
        "optimizations": [
            {
                "id": "opt",
                "parameters": [
                    {"param": "fast", "type": "int", "min": 3, "max": 30, "step": 1},
                    {"param": "slow", "type": "int", "min": 20, "max": 100, "step": 1},
                ],
            }
        ],
    },
    {
        "version": 1,
        "id": "example_zscore_reversion",
        "name": "RSI z-score reversion",
        "description": "Buy when RSI is unusually low (z<-2), exit when unusually high (z>+2).",
        "sources": [{"id": "src", "schema": "ohlcv-1m"}],
        "features": [
            {"id": "r", "kind": "rsi", "input": "src", "period": 14},
            {
                "id": "z",
                "kind": "zScore",
                "input": "r",
                "lookback": 20,
                "upperThreshold": 2.0,
                "lowerThreshold": -2.0,
            },
        ],
        "signals": [],
        "entries": [{"id": "e", "enter": "z:below", "exit": "z:above"}],
    },
]


@dataclass
class IdeationContext:
    """What the prompt needs about the current research state."""

    champions: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    # Family ids already taken (seeds + minted specs) — a colliding id would clobber an
    # existing family's class, so the prompt forbids them and Ideator.run skips them.
    existing_families: list[str] = field(default_factory=list)
    max_indicators: int = DEFAULT_MAX_INDICATORS


# ─────────────────────────────────────────────────────────────────────────────
# The LLM call (through the provider-neutral seam)
# ─────────────────────────────────────────────────────────────────────────────
def _tool_schema(n: int) -> dict:
    """Wrap the StrategySpec JSON schema as a forced-tool input: ``{"strategies": [...]}``.

    Pydantic emits ``$defs`` at the schema root; hoist them to the tool-schema root so the
    ``$ref`` pointers still resolve inside the single tool document."""
    spec_schema = StrategySpec.model_json_schema(by_alias=True)
    defs = spec_schema.pop("$defs", {})
    return {
        "type": "object",
        "properties": {
            "strategies": {
                "type": "array",
                "description": f"{n} newly invented StrategySpec strategies.",
                "items": spec_schema,
            }
        },
        "required": ["strategies"],
        "$defs": defs,
    }


def _build_prompt(context: IdeationContext, n: int, *, web_search: bool = False) -> str:
    champions = json.dumps(context.champions, sort_keys=True)
    rejected = json.dumps(context.rejected, sort_keys=True)
    taken = json.dumps(sorted(context.existing_families))
    examples = "\n".join(json.dumps(ex) for ex in FEW_SHOT)
    # With web search on, ground ideas in durable priors — never in period-specific outcomes.
    # Minted specs run through a lookahead-free causal backtest, so outcome knowledge of any
    # given date range cannot help and only invites overfitting.
    web_guidance = (
        "You may use the web_search tool to ground your designs in established quantitative "
        "research: known factors, indicators, and market-microstructure effects. Search for "
        "durable, timeless techniques and their economic rationale — NOT for what happened in "
        "any specific historical period. The strategies are evaluated causally on held-out "
        "data the search cannot influence, so period-specific outcomes cannot help and only "
        "risk overfitting. When you have gathered enough, emit the strategies.\n\n"
        if web_search
        else ""
    )
    return (
        f"You design quantitative trading strategies as StrategySpec JSON.\n\n{VOCABULARY}\n\n"
        f"Two valid examples (shape only — invent genuinely different structures):\n{examples}\n\n"
        f"{web_guidance}"
        f"Current champions (beat these with structurally different ideas): {champions}\n"
        f"Known dead ends (do not re-propose): {rejected}\n"
        f"Family ids already taken (a colliding id is discarded): {taken}\n\n"
        f"Invent {n} NEW long/flat StrategySpec strategies, each with a unique id not in the "
        f"taken list, using at most {context.max_indicators} features. You MUST return them by "
        f"calling the {_TOOL_NAME} tool — do not describe them in text."
    )


def _extract_specs(turn: Turn | None) -> list[dict]:
    """Pull the ``strategies`` array out of the emit tool call."""
    for tc in turn.tool_calls if turn is not None else []:
        if tc.name == _TOOL_NAME:
            got = tc.arguments.get("strategies", []) if isinstance(tc.arguments, dict) else []
            return list(got)
    return []


def _run_ideation_call(
    client, *, tools, tool_choice, prompt, max_turns=6, max_web_searches=0
) -> Turn | None:
    """One ideation exchange, resuming when the model pauses for a tool.

    Server-side web search runs a loop on the provider's side inside a single completion; if
    that loop hits its iteration cap the turn comes back with ``stop_reason == "pause_turn"``
    and must be re-sent verbatim to continue — the same resume idiom as the agent research
    loop. A *client-side* ``web_search`` (OpenAI, any keyless local backend) instead comes back
    as a function tool call we execute against the sidecar, appending the result before
    continuing. Either way we loop until the model emits (or gives up). Returns the final turn.
    """
    messages: list[dict] = [{"role": "user", "content": prompt}]
    searches_left = max_web_searches
    turn = None
    for _ in range(max_turns):
        turn = client.complete(
            system=None,
            tools=tools,
            messages=messages,
            max_tokens=_MAX_TOKENS,
            tool_choice=tool_choice,
        )
        if turn.stop_reason == "pause_turn":
            # Server-tool loop paused mid-turn; resume it verbatim.
            messages = messages + [turn.assistant_message]
            continue
        if any(tc.name == _TOOL_NAME for tc in turn.tool_calls):
            return turn  # the model emitted — done
        searches = [tc for tc in turn.tool_calls if tc.name == websearch.TOOL_NAME]
        if not searches:
            return turn  # a plain/misfired turn — let the caller extract (likely nothing)
        # Client-side web search: dispatch each call to the sidecar, append results, continue.
        messages = messages + [turn.assistant_message]
        for tc in searches:
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            if searches_left > 0:
                searches_left -= 1
                result = websearch.search(args.get("query", ""), args.get("max_results", 5))
            else:
                result = {
                    "error": "web_search budget exhausted; emit the strategies now",
                    "results": [],
                }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str, separators=(",", ":")),
                }
            )
    return turn


def propose_specs(
    *,
    context: IdeationContext,
    n: int,
    client,
    web_search: bool = False,
    max_web_searches: int = 5,
) -> list[StrategySpec]:
    """Ask the LLM for ``n`` new specs; return only those that validate AND pass the parity gate.

    ``client is None`` (no key / no ``[llm]`` extra) → ``[]``. Any API/parse failure also
    degrades to ``[]`` so the research loop never breaks on ideation.

    ``web_search`` requests the provider's server-side web_search tool so the agent can ground
    new structures in published research; it auto-disables (idea-grounding degrades, nothing
    else) where the provider lacks the capability. Forcing the emit tool would skip the search,
    so with web search active the emit tool is *offered* (``tool_choice`` auto) and the prompt
    requires the agent to finish by calling it; if it never does, ``_extract_specs`` yields
    ``[]``."""
    if client is None or n <= 0:
        return []
    emit_tool = {
        "name": _TOOL_NAME,
        "description": "Return newly invented StrategySpec trading strategies.",
        "input_schema": _tool_schema(n),
    }
    tools: list[dict] = [emit_tool]
    tool_choice: dict | None = {"type": "function", "function": {"name": _TOOL_NAME}}
    web_search_active = effective_web_search(web_search, client.capabilities)  # server-side
    use_client_search = web_search and not web_search_active  # keyless fallback via the sidecar
    if web_search_active:
        tools.append(
            {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": max_web_searches}
        )
        tool_choice = None
    elif use_client_search:
        logger.info(
            "ideation web_search: local backend — grounding via the local web_search sidecar on "
            ":11435 (noctis-ollama scripts/search.sh; degrades cleanly if it is down)"
        )
        tools.append(
            websearch.client_tool_spec(
                "Search the public web to ground new designs in established quantitative "
                "research — known factors, indicators, and market-microstructure effects. "
                "Returns {title, url, snippet} hits. When you have gathered enough, emit the "
                "strategies."
            )
        )
        tool_choice = None
    try:
        turn = _run_ideation_call(
            client,
            tools=tools,
            tool_choice=tool_choice,
            prompt=_build_prompt(context, n, web_search=web_search_active or use_client_search),
            max_web_searches=max_web_searches if use_client_search else 0,
        )
    except Exception as exc:  # noqa: BLE001 — ideation is best-effort; never crash research
        logger.warning("ideation call failed (%s); minting nothing this round", exc)
        return []

    admitted: list[StrategySpec] = []
    for raw in _extract_specs(turn):
        try:
            spec = StrategySpec.model_validate(raw)
            validate_spec(spec, max_indicators=context.max_indicators)
            if not _passes_parity(spec):
                logger.info("ideation: %s rejected by parity gate", raw.get("id", "?"))
                continue
        except Exception as exc:  # noqa: BLE001 — a malformed idea is silently dropped
            logger.info("ideation: dropped invalid spec (%s)", exc)
            continue
        admitted.append(spec)
    return admitted


# ─────────────────────────────────────────────────────────────────────────────
# Parity admission gate — a minted spec is admitted only if its vectorised
# signals() and incremental on_bar() agree on a fixture (the base.py contract).
# The fixture and the on_bar replay were lifted into the strategy layer
# (library.fixture_frame / base.replay_targets) so the authored-strategy write
# gate exercises the same code.
# ─────────────────────────────────────────────────────────────────────────────
def _passes_parity(spec: StrategySpec) -> bool:
    """Build the family and assert signals() == on_bar() on the fixture; any error rejects."""
    try:
        cls = family_class_from_spec(spec)
        frame = fixture_frame()
        params = cls.params_cls()
        vectorised = [int(x) for x in cls.signals(frame, params)]
        event = replay_targets(cls(params), frame)
        return vectorised == event
    except Exception:  # noqa: BLE001 — an un-runnable spec is not admissible
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Ideator — the loop seam: propose → register → add_family → remember
# ─────────────────────────────────────────────────────────────────────────────
class Ideator:
    """Mints new spec-families at a fixed cadence and wires them into the research pipeline.

    ``run(iteration)`` proposes on the seed round (``iteration == 0``) and every ``cadence``
    iterations thereafter; each admitted spec is registered as a family (persisted to
    ``state/specs.json``), added to the proposer's rotation (so Optuna starts tuning it), and
    noted in memory. Returns the family names minted this call (``[]`` when idle or clientless)."""

    def __init__(
        self,
        *,
        client,
        config,
        registry,
        families: FamilyRegistry,
        proposer,
        memory,
        state_dir: str | Path,
    ) -> None:
        self.client = client
        self.config = config
        self.registry = registry
        self.families = families
        self.proposer = proposer
        self.memory = memory
        self.state_dir = state_dir

    def _should_run(self, iteration: int) -> bool:
        if self.client is None or not getattr(self.config, "enabled", True):
            return False
        cadence = max(1, int(getattr(self.config, "cadence", 1)))
        return iteration % cadence == 0

    def _context(self) -> IdeationContext:
        champions = [{"family": e.family, "params": e.params} for e in self.registry.list()]
        rejected = self.memory.rejected_ideas()
        return IdeationContext(
            champions=champions,
            rejected=rejected,
            existing_families=self.families.names(),
            max_indicators=int(getattr(self.config, "max_indicators", DEFAULT_MAX_INDICATORS)),
        )

    def run(self, iteration: int) -> list[str]:
        if not self._should_run(iteration):
            return []
        specs = propose_specs(
            context=self._context(),
            n=int(getattr(self.config, "specs_per_round", 3)),
            client=self.client,
            web_search=bool(getattr(self.config, "web_search", True)),
            max_web_searches=int(getattr(self.config, "max_web_searches", 5)),
        )
        minted: list[str] = []
        for spec in specs:
            # Collision guard: registry registration silently overwrites, and a clobbered family
            # breaks any champion holding its old params. An identical re-proposal of a spec
            # we already minted is idempotent (already registered + persisted) — just make
            # sure it is in the proposer's rotation; anything else is skipped.
            if spec.id in self.families:
                if persisted_spec_json(self.state_dir, spec.id) == spec.model_dump(
                    mode="json", by_alias=True
                ):
                    logger.info("ideation: %s re-proposed unchanged; already registered", spec.id)
                    self.proposer.add_family(spec.id)
                else:
                    logger.warning(
                        "ideation: skipped %s — id collides with an existing family", spec.id
                    )
                continue
            register_spec(spec, self.state_dir, self.families)
            self.proposer.add_family(spec.id)
            label = spec.name or spec.description or spec.id
            self.memory.append_finding(f"MINTED spec family {spec.id} — {label}")
            minted.append(spec.id)
        if minted:
            logger.info("ideation minted %d new families: %s", len(minted), ", ".join(minted))
        return minted


# ─────────────────────────────────────────────────────────────────────────────
# Wiring helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_ideator(
    *, settings, registry, families: FamilyRegistry, proposer, memory, state_dir: str | Path
) -> Ideator:
    """Construct the :class:`Ideator` from settings; clientless when no key/extra present.

    ``ideation.model`` rides the same provider seam grammar as ``research.model`` (#10): a bare
    id (``claude-opus-4-8``) or a ``provider/model`` string. Any provider works — the emit tool
    is an ordinary function tool — and web-search grounding auto-disables where the provider
    lacks it (:func:`propose_specs`)."""
    config = settings.ideation
    client = client_for(settings, config.model) if config.enabled else None
    return Ideator(
        client=client,
        config=config,
        registry=registry,
        families=families,
        proposer=proposer,
        memory=memory,
        state_dir=state_dir,
    )
