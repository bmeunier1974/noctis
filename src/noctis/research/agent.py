"""The agent research loop — Claude formulates strategy; the world provides tools and gates.

One session = one Anthropic tool-use conversation driving the four-phase protocol
(FORMULATE → MATCH → OPTIMIZE → DECIDE) against the curated
:class:`~noctis.research.tools.ResearchToolbox`. The loop enforces only budgets and
transport; every research invariant (exhaustion gate, journal, validation gate,
aggregate-only scorecards, budget caps) is structural inside the tools, so a creative
agent cannot wander around them.

Same summary contract as :func:`noctis.engine.research.run_research`, so the runtime calls
either loop behind one seam. Degrades to a no-op without a client (no ``ANTHROPIC_API_KEY``
or no ``anthropic`` extra) — the runtime then falls back to the legacy loop.

The loop's two big collaborators live next door: :mod:`noctis.research.prompt` assembles
the session system prompt, :mod:`noctis.research.misfire` classifies the model stumbles
the loop corrects and retries instead of reading as a conclusion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from noctis.engine.research import ResearchSummary, StopEvent
from noctis.observability.events import Event, render_plain
from noctis.research.llm import WEB_SEARCH_TOOL_TYPE, cached_system, effective_web_search
from noctis.research.misfire import classify_completion_error, classify_turn
from noctis.research.prompt import build_system_prompt

if TYPE_CHECKING:
    from noctis.research.llm import LLMClient
    from noctis.research.mandate import Mandate

logger = logging.getLogger("noctis.research.agent")

_MAX_TOKENS = 8000
_RESULT_CHAR_CAP = 20_000  # hard cap per tool result so one dump can't flood the context

# Context-budget levers (plan P5) — all inert unless research.agent.context_window is set.
# The size estimate is provider-neutral (chars // 4 ≈ tokens): it must work on any backend,
# including ones that report no usage at all.
_APPROX_CHARS_PER_TOKEN = 4
_EVICT_AT_FRACTION = 0.9  # evict when the next request's estimate crosses this fraction…
_EVICT_TO_FRACTION = 0.8  # …in one oldest-first batch down to this (no per-round cache thrash)
_RESULT_CAP_FLOOR = 2_000  # the tiered per-result cap never drops below this (a scorecard fits)
_RESULT_CAP_WINDOW_DIVISOR = 8  # tiered cap ≈ one eighth of the window, in chars

# The token-usage fields we roll up per session. Cache fields are 0 until caching lands
# (issues #5/#6); reading every field defensively keeps a fake/no-usage client from raising.
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _accumulate_usage(totals: dict[str, int], usage: dict | None) -> None:
    """Fold one completion's token ``usage`` (the neutral four-field dict on a :class:`Turn`)
    into the running per-session totals.

    Defensive by design: a fake client or a provider that omits a field contributes 0 rather
    than raising — this is the measurement floor, it must never break the loop it measures.
    """
    if not usage:
        return
    for field in _USAGE_FIELDS:
        totals[field] += int(usage.get(field, 0) or 0)


def _with_moving_breakpoint(messages: list[dict], *, cache: bool = True) -> list[dict]:
    """Return ``messages`` with one *moving* cache_control breakpoint on the last content block
    of the last message — in this loop that message is the user turn carrying the round's
    ``tool_result`` dicts, so the growing history caches up to there and reads (not re-bills) it
    next round.

    The persistent ``messages`` list is never mutated: the breakpoint lives only on a shallow
    copy built fresh each round, so no stale breakpoint survives. Paired with the one static
    breakpoint on ``system``, that keeps exactly two live breakpoints — well under the 4-cap.
    (The ~20-block cache lookback matters only if a single round appended >~18 parallel
    tool_results; such batches are small here, and a miss merely re-writes, never misdecides.)

    ``cache=False`` (auto-caching or no-caching provider) returns ``messages`` unchanged, and a
    plain-string turn like the kickoff has no block to annotate, so it is left as-is.
    """
    if not cache or not messages:
        return messages
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str) and content:
        # A plain-string turn (OpenAI tool/user message): wrap it in one cached text part.
        new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content:
        new_content = list(content)
        new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
    else:
        return messages
    return messages[:-1] + [{**last, "content": new_content}]


def _cache_hit_ratio(totals: dict[str, int]) -> float:
    """Fraction of input tokens served from cache: cache_read / (cache_read + non_cached_input),
    where non_cached_input is freshly-billed input plus cache-creation writes. 0.0 when idle."""
    cache_read = totals["cache_read_input_tokens"]
    non_cached_input = totals["input_tokens"] + totals["cache_creation_input_tokens"]
    denom = cache_read + non_cached_input
    return cache_read / denom if denom else 0.0


class _NeverStop:
    def is_set(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# The tool-use conversation loop
# ─────────────────────────────────────────────────────────────────────────────
def _tool_result_content(result: dict, cap: int = _RESULT_CHAR_CAP) -> str:
    # Compact separators drop the incidental whitespace that would otherwise be re-sent on
    # every round's accumulated tool-result history (Class-A: same content, fewer bytes).
    text = json.dumps(result, default=str, separators=(",", ":"))
    if len(text) > cap:
        text = text[:cap] + '... (truncated)"}'
    return text


def _tiered_result_cap(context_window: int | None) -> int:
    """The per-result char cap: the flat default without a budget; scaled to the window with
    one, floored so results stay usable (P5.1). Scorecards/logs are already aggregates — on a
    small window one result must not eat a large fraction of it."""
    if context_window is None:
        return _RESULT_CHAR_CAP
    scaled = context_window * _APPROX_CHARS_PER_TOKEN // _RESULT_CAP_WINDOW_DIVISOR
    return min(_RESULT_CHAR_CAP, max(_RESULT_CAP_FLOOR, scaled))


def _estimate_tokens(base_chars: int, messages: list[dict]) -> int:
    """Provider-neutral size estimate of the next request: prefix chars + serialized history,
    at ~4 chars/token. Deliberately independent of provider usage reports."""
    chars = base_chars + sum(len(json.dumps(m, default=str)) for m in messages)
    return chars // _APPROX_CHARS_PER_TOKEN


class _ContextBudget:
    """Bounded-history discipline (plan P5): every method no-ops unless a window is set, so
    the unset path stays byte-identical to the unbounded loop.

    The loop touches it at exactly three points: :meth:`evict_to_fit` on the persistent
    history before each request, :meth:`observe` on the usage each completion reports, and
    :meth:`tool_result` once per executed tool call. Everything else — the output-headroom
    reserve, the request-prefix size, the tiered per-result cap, verdict-boundary
    compaction — lives inside.

    Structural rules: never remove or reorder a message and never touch an assistant turn —
    only a tool-role message's ``content`` string is swapped for a FIXED pointer line (no
    summarization path exists, so nothing can leak holdout data forward), keeping every
    tool_call_id pairing intact on every provider. All replacements happen on the persistent
    ``messages`` list before the per-round cache-breakpoint copy is taken. The experiment
    journal on disk is the ground truth for everything replaced (plan principle 4): the
    exhaustion gate and verdict tools read the journal, never the context.

    Which tools are verdicts (durable conclusions, never replaced) and which produce
    journal-backed per-strategy history (safe to collapse to pointers) is the *toolbox's*
    knowledge — passed in as ``verdict_tools`` / ``history_tools``
    (:class:`~noctis.research.tools.ResearchToolbox` declares them beside the tools).
    """

    def __init__(
        self,
        *,
        context_window: int | None,
        max_tokens: int,
        system: str,
        tools: list[dict],
        verdict_tools: frozenset[str],
        history_tools: frozenset[str],
    ):
        self.verdict_tools = verdict_tools
        self.history_tools = history_tools
        # The response shares the backend's physical window with the prompt, so the prompt-
        # side budget reserves max_tokens of headroom — a prompt allowed to fill the whole
        # window leaves its reply no room and can only truncate (finish_reason="length").
        # The reserve is capped at half the window so a degenerate config (window ≤
        # max_tokens) still keeps a working budget.
        self.window = (
            context_window - min(max_tokens, context_window // 2) if context_window else None
        )
        # The fixed request prefix: system prompt + final tool list (web_search included).
        self.base_chars = len(system) + len(json.dumps(tools, default=str))
        # P5.1 per-result char cap: flat without a window, tiered to the window with one.
        self.result_cap = _tiered_result_cap(self.window)
        self._meta: dict[str, dict] = {}  # tool_call_id → {tool, strategy, verdict, replaced}
        self._scale = 1.0  # calibration: worst observed (real prompt tokens / char estimate)

    def observe(self, messages: list[dict], usage: dict) -> None:
        """Calibrate the chars/4 heuristic against the prompt size the backend actually
        reported for the request just sent (``messages`` exactly as completed). The char
        estimate systematically under-counts — ~1.1× on prose, worse on code-heavy history —
        and an under-counting budget lets the prompt crowd the reply out of a shared physical
        window (observed as finish_reason="length"). Worst-ratio keeps the budget honest on
        any backend; a provider that reports no usage contributes nothing and the heuristic
        stands as-is."""
        if self.window is None:
            return
        actual = sum(
            int(usage.get(f, 0) or 0)
            for f in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        raw = _estimate_tokens(self.base_chars, messages)
        if actual <= 0 or raw <= 0:
            return
        self._scale = max(self._scale, actual / raw)

    def record(self, call_id: str, tool: str, args: dict) -> None:
        if self.window is None:
            return
        self._meta[call_id] = {
            "tool": tool,
            "strategy": args.get("name"),
            "verdict": tool in self.verdict_tools,
            "replaced": False,
        }

    def _replaceable(self, msg: dict) -> dict | None:
        if msg.get("role") != "tool":
            return None
        meta = self._meta.get(msg.get("tool_call_id", ""))
        if meta is None or meta["verdict"] or meta["replaced"]:
            return None  # verdict lines are the durable conclusions — never replaced
        return meta

    def tool_result(
        self, call_id: str, tool: str, args: dict, result, messages: list[dict]
    ) -> dict:
        """The one per-call entry: record the call for later compaction/eviction, collapse a
        just-decided strategy's optimization history to pointers (the verdict result itself,
        returned here, survives in full), and hand back the capped tool-role message."""
        self.record(call_id, tool, args)
        if tool in self.verdict_tools and isinstance(result, dict) and "error" not in result:
            self.compact_decided_strategy(args.get("name"), tool, messages)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": _tool_result_content(result, cap=self.result_cap),
        }

    def compact_decided_strategy(
        self, strategy: str | None, verdict_tool: str, messages: list[dict]
    ) -> None:
        """P5.3 verdict-boundary compaction: once ``strategy`` is decided, its optimization
        back-and-forth collapses to pointer lines so the next strategy starts near-fresh."""
        if self.window is None or not strategy:
            return
        pointer = (
            f'(superseded by the {verdict_tool} verdict on "{strategy}"; full trial history '
            f'via get_experiment_log(name="{strategy}"))'
        )
        for i, msg in enumerate(messages):
            meta = self._replaceable(msg)
            if (
                meta is None
                or meta["tool"] not in self.history_tools
                or meta["strategy"] != strategy
                or len(msg.get("content") or "") <= len(pointer)
            ):
                continue
            messages[i] = {**msg, "content": pointer}
            meta["replaced"] = True

    def evict_to_fit(self, messages: list[dict]) -> None:
        """P5.2 evict-and-point: when the next request's estimate nears the window, replace
        the oldest completed rounds' tool-result bodies with pointers, in one batch down to
        a comfortable margin. The freshest round's results (after the last assistant turn)
        are never evicted — the model has not acted on them yet."""
        if self.window is None:
            return
        estimate = _estimate_tokens(self.base_chars, messages) * self._scale
        if estimate <= self.window * _EVICT_AT_FRACTION:
            return
        target = self.window * _EVICT_TO_FRACTION
        last_assistant = max(
            (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
            default=len(messages),
        )
        for i, msg in enumerate(messages):
            if estimate <= target:
                break
            if i > last_assistant:
                break  # the current round's unseen results — protected
            meta = self._replaceable(msg)
            if meta is None:
                continue
            strategy = meta["strategy"]
            if meta["tool"] in self.history_tools and strategy:
                pointer = (
                    f"(evicted to fit the context budget; re-fetch via "
                    f'get_experiment_log(name="{strategy}"))'
                )
            else:
                pointer = (
                    f"(evicted to fit the context budget; re-fetch via {meta['tool']} "
                    f"if still needed)"
                )
            saved = len(msg.get("content") or "") - len(pointer)
            if saved <= 0:
                continue
            messages[i] = {**msg, "content": pointer}
            meta["replaced"] = True
            estimate -= saved / _APPROX_CHARS_PER_TOKEN * self._scale
        if estimate > self.window:
            logger.warning(
                "context budget: history still ~%d tokens against a %d-token window after "
                "eviction; the request may not fit this backend",
                estimate,
                self.window,
            )


def run_agent_research(
    *,
    toolbox,
    client: LLMClient | None,
    budget_minutes: float,
    max_iterations: int | None = None,
    max_tokens: int | None = None,
    context_window: int | None = None,
    stop_event: StopEvent | None = None,
    now: Callable[[], datetime] = _utcnow,
    web_search: bool = False,
    max_web_searches: int = 8,
    prefix_trim: bool = False,
    on_event: Callable[[Event | str], None] | None = None,
    mandate: Mandate | None = None,
) -> ResearchSummary:
    """Run one agent research session; returns the same summary shape as ``run_research``.

    ``client`` is a provider-neutral :class:`~noctis.research.llm.LLMClient`; ``None`` (no key /
    no ``[llm]`` extra) → no-op summary, so the caller can fall back to the legacy loop. Every
    provider-specific lever is gated on ``client.capabilities`` — prompt-cache breakpoints and
    the server-side ``web_search`` tool are sent only when the provider supports them. Budgets:
    wall-clock ``budget_minutes``, ``max_iterations`` tool rounds, and the toolbox's own backtest
    budget. ``max_tokens`` caps output per completion (``None`` ⇒ the built-in default) — a
    compatibility lever for small-context backends, not a spend control. ``context_window``
    (``None`` ⇒ unlimited, byte-identical history) bounds the whole request the same way:
    per-result caps tier down, the oldest tool-result bodies evict to fixed pointer lines, and
    a decided strategy's optimization history collapses at its verdict — all re-fetchable
    through the same tools, with the on-disk journal untouched. ``on_event`` receives typed
    :class:`~noctis.observability.events.Event`s (the model's reasoning, its narration, one line
    per tool call with the gate-facing numbers, per-round usage) plus the occasional legacy
    pre-formatted string; the CLI renders them level-gated, the default sink logs them. It only
    ever surfaces what a completion already returned — zero extra model calls or tokens. When that
    sink is a level-2 :class:`~noctis.observability.console.Console` on a TTY (and the provider can
    stream), its ``delta`` renderer receives reasoning/content deltas so the turn types out in
    place (P5); every other sink runs the loop non-streaming and byte-identically.
    ``mandate`` is the optional resolved operator
    mandate embedded in the system prompt; its one-line ``summary`` (not the full body) is echoed
    into the kickoff turn.
    """
    summary = ResearchSummary()
    if client is None:
        summary.stopped_reason = "no_client"
        return summary

    caps = client.capabilities
    stop_event = stop_event or _NeverStop()
    max_iterations = max_iterations or 40
    # max_tokens is a compat lever (settings.research.agent.max_tokens), not a spend: the default
    # is sized so a whole write_strategy file fits, because a "length" truncation here doesn't
    # degrade — it breaks the tool-call JSON or reads as end_turn and kills the session.
    max_tokens = max_tokens or _MAX_TOKENS
    # The default sink logs; an Event is flattened to one line, a legacy string logged as-is.
    emit = on_event or (
        lambda item: logger.info("%s", item if isinstance(item, str) else render_plain(item))
    )
    # P5 token streaming: when the sink is a level-2 Console it exposes a delta() renderer — tee
    # reasoning/content deltas to it so they type out in place. Any other sink (logger, plain
    # callable, a -v-only console) yields None and the loop runs non-streaming exactly as before;
    # the client itself further gates on provider capability, so an unset path can't stream.
    on_delta = getattr(on_event, "delta", None) if getattr(on_event, "verbose", 0) >= 2 else None
    # P6 heartbeat: a live in-place activity line so -v isn't silent while a model call or a tool
    # sweep blocks for minutes. Duck-typed like on_delta — a plain-callable sink (tests) has no
    # activity() and the nullcontext keeps the loop byte-identical.
    activity = getattr(on_event, "activity", None)
    start = now()
    budget_seconds = budget_minutes * 60.0

    system = build_system_prompt(
        toolbox,
        budget_minutes=budget_minutes,
        max_iterations=max_iterations,
        mandate=mandate,
        prefix_trim=prefix_trim,
    )
    # The system prompt is byte-stable within a session; cache it once (gated on prompt_cache)
    # so tools + system are written to cache on round 1 and read, not re-billed, thereafter.
    system_prompt = cached_system(system, cache=caps.prompt_cache)
    tools = list(toolbox.tool_specs())
    # Web search is one tool name with two implementations, chosen by provider. Anthropic
    # serves `web_search` server-side; every other provider (OpenAI, any $0 local/self-hosted
    # backend) uses the toolbox's client-side `web_search`, which calls the local search
    # sidecar (noctis-ollama). The operator `web_search` flag governs availability; the two
    # implementations are never declared together.
    web_search_active = effective_web_search(web_search, caps)
    if not web_search:
        # Operator disabled web search — withdraw the always-declared client tool too.
        tools = [t for t in tools if t.get("name") != "web_search"]
    elif web_search_active:
        # Anthropic's server-side web_search supersedes the client tool of the same name.
        tools = [t for t in tools if t.get("name") != "web_search"]
        tools.append(
            {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": max_web_searches}
        )
    else:
        # Requested, but this provider has no server-side search — the client sidecar tool
        # (already in the toolbox spec) stands in; grounding depends on the sidecar being up.
        emit(
            "web_search: local backend — grounding via the local web_search sidecar on :11435 "
            "(noctis-ollama scripts/search.sh; degrades cleanly if it is down)"
        )
    tool_names = {t["name"] for t in tools}
    # P5 context budget — reserve math, result cap, and compaction all live inside; every
    # knob is a no-op while research.agent.context_window is unset. The toolbox declares
    # which tools are verdicts / journal-backed history; the budget just applies that.
    budget = _ContextBudget(
        context_window=context_window,
        max_tokens=max_tokens,
        system=system,
        tools=tools,
        verdict_tools=toolbox.VERDICT_TOOLS,
        history_tools=toolbox.STRATEGY_HISTORY_TOOLS,
    )

    kickoff = (
        "Run one research session now, driving the protocol to at least one explicit "
        "verdict. Start by inspecting the library, champions, and data inventory."
    )
    if mandate is not None:
        kickoff += f" Honor the OPERATOR MANDATE block (summary: {mandate.summary})."
    messages: list[dict] = [{"role": "user", "content": kickoff}]

    usage_totals = dict.fromkeys(_USAGE_FIELDS, 0)

    while True:
        if stop_event.is_set():
            summary.stopped_reason = "stop_event"
            break
        if (now() - start).total_seconds() >= budget_seconds:
            summary.stopped_reason = "time_budget"
            break
        if summary.iterations >= max_iterations:
            summary.stopped_reason = "max_iterations"
            break

        # Evict on the persistent list, BEFORE the per-round breakpoint copy is taken.
        budget.evict_to_fit(messages)
        # Heartbeat the blocking model call — but only when tokens aren't already streaming
        # (on_delta is None): at -vv the live token stream is the life sign; at -v the spinner
        # stands in for it. The stop/join in activity() ends the thread even if complete() raises.
        model_hb: AbstractContextManager[None] = nullcontext()
        if activity is not None and on_delta is None:
            model_hb = activity(_model_label(client))
        try:
            with model_hb:
                turn = client.complete(
                    system=system_prompt,
                    tools=tools,
                    messages=_with_moving_breakpoint(messages, cache=caps.prompt_cache),
                    max_tokens=max_tokens,
                    on_delta=on_delta,
                )
        except Exception as exc:  # noqa: BLE001 — research must never crash the runtime
            stumble = classify_completion_error(exc)
            if stumble is not None:
                # A model stumble surfacing as an exception, not an outage. The failed
                # completion still burns an iteration, so the ordinary max_iterations
                # budget bounds this retry like every other misfire.
                summary.iterations += 1
                emit(f"[misfire] {stumble.note}")
                messages = messages + [{"role": "user", "content": stumble.retry}]
                continue
            logger.warning("agent research call failed (%s); ending session", exc)
            summary.stopped_reason = "api_error"
            break

        summary.iterations += 1
        _accumulate_usage(usage_totals, turn.usage)
        # Calibrate the context budget's size estimate against the prompt size the backend
        # actually reported for this request (messages is still exactly what was sent).
        budget.observe(messages, turn.usage)

        # Tee what the completion already returned — never a new request. Reasoning + usage are
        # level-2 (the -vv / --show-reasoning firehose); narration (turn.text) is emitted per
        # branch below so the final conclusion can stay level-1 without a duplicate. None of
        # this is ever written to memory or the journal (reasoning must not reach a decision).
        if turn.reasoning.strip():
            emit(Event("think", turn.reasoning.strip(), level=2))
        emit(Event("usage", _usage_line(turn.usage), meta=dict(turn.usage or {}), level=2))

        if turn.stop_reason == "pause_turn":
            # Server-tool loop (web search) paused mid-turn; resume it verbatim.
            messages = messages + [turn.assistant_message]
            continue

        tool_calls = [tc for tc in turn.tool_calls if tc.name in tool_names]
        if not tool_calls:
            # A misfired turn — markup where a native call belongs, a reply truncated by the
            # output limit, or a thinking-only stall — is an attempted move, not a conclusion:
            # correct and retry. The misfired assistant turn is NOT appended (nothing parseable
            # to replay), and every retried round already burned an iteration, so the ordinary
            # max_iterations budget bounds a persistent misfirer to a legitimate stop.
            stumble = classify_turn(turn)
            if stumble is not None:
                emit(f"[misfire] {stumble.note}")
                messages = messages + [{"role": "user", "content": stumble.retry}]
                continue
            summary.stopped_reason = "agent_done"
            # The agent's deliberate final conclusion — level-1 so it shows at -v like today; the
            # renderer wraps to width, so no more 500-char truncation special-case.
            emit(Event("say", turn.text.strip(), level=1))
            break

        messages = messages + [turn.assistant_message]
        # Narration that rides alongside an action is inter-round context — level-2 (the -vv /
        # --show-reasoning firehose), so -v keeps a clean tool feed.
        if turn.text.strip():
            emit(Event("say", turn.text.strip(), level=2))
        results = []
        for tc in tool_calls:
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            # Heartbeat the blocking tool sweep (an optimize sweep runs 8 workers for minutes);
            # the spinner erases and the level-1 result Event below prints in its place.
            tool_hb: AbstractContextManager[None] = nullcontext()
            if activity is not None:
                tool_hb = activity(_tool_label(tc.name, args))
            with tool_hb:
                result = toolbox.dispatch(tc.name, args)
            emit(_tool_event(tc.name, args, result, toolbox.result_brief(result)))
            results.append(budget.tool_result(tc.id, tc.name, args, result, messages))
        messages = messages + results

    summary.promotions = toolbox.promotions
    summary.rejections = toolbox.rejections
    summary.candidates = list(toolbox.strategies_touched)
    summary.author_calls = toolbox.author_calls
    logger.info(
        "agent research session finished: %d rounds, %d backtests, %d coder calls, %d promotions, "
        "%d rejections, strategies=%s (%s)",
        summary.iterations,
        toolbox.backtests_run,
        toolbox.author_calls,
        summary.promotions,
        summary.rejections,
        ",".join(summary.candidates) or "-",
        summary.stopped_reason,
    )
    logger.info(
        "agent research usage: %d rounds, input=%d output=%d cache_write=%d cache_read=%d, "
        "cache_hit_ratio=%.3f",
        summary.iterations,
        usage_totals["input_tokens"],
        usage_totals["output_tokens"],
        usage_totals["cache_creation_input_tokens"],
        usage_totals["cache_read_input_tokens"],
        _cache_hit_ratio(usage_totals),
    )
    return summary


def _usage_line(usage: dict | None) -> str:
    """A compact per-round token line from a turn's neutral usage dict (0 for any field a
    provider omits, so a no-usage backend still yields a clean line)."""
    u = usage or {}

    def g(k: str) -> int:
        return int(u.get(k, 0) or 0)

    return (
        f"tokens in={g('input_tokens')} out={g('output_tokens')} "
        f"cache_w={g('cache_creation_input_tokens')} cache_r={g('cache_read_input_tokens')}"
    )


def _tool_event(name: str, args: dict, result: dict, brief: dict) -> Event:
    """One :class:`Event` per tool call for logs / the CLI feed (level 1). ``text`` is the call
    plus a compact outcome; ``brief`` — the toolbox's own gate-facing slice of the result
    (:meth:`~noctis.research.tools.ResearchToolbox.result_brief`) — carries the numbers, both
    printed and structured into ``meta``, plus an ``ok`` flag the renderer colors on (green
    success / red error)."""
    brief_args = {
        k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 80 else v)
        for k, v in (args or {}).items()
    }
    meta: dict = {}
    if isinstance(result, dict) and "error" in result:
        meta["ok"] = False
        outcome = f"ERROR: {result['error']}"
    else:
        meta["ok"] = True
        meta.update(brief)
        outcome = ", ".join(f"{k}={v}" for k, v in brief.items()) or "ok"
    text = f"{name}({json.dumps(brief_args, default=str)}) -> {outcome}"
    return Event("tool", text, meta=meta, level=1)


def _tool_label(name: str, args: dict) -> str:
    """A compact ``<tool> <salient-arg>`` for the heartbeat's live line — enough to tell which
    strategy/symbol a long sweep is on, without the full arg dump the result Event carries."""
    for key in ("strategy", "name", "symbol", "family", "spec"):
        value = (args or {}).get(key)
        if isinstance(value, str) and value:
            return f"{name} {value}"
    return name


def _model_label(client: object) -> str:
    """``thinking (<model>)`` when the client names its model, else a bare ``thinking``."""
    model = getattr(client, "model", None)
    return f"thinking ({model})" if isinstance(model, str) and model else "thinking"
