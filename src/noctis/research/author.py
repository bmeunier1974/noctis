"""The strategy-authoring engine — a structured brief in, a validated strategy file out.

:class:`StrategyAuthor` is the deep module behind the coder-model split (the epic in
``.claude/plan/coder-model-split.md``): a cheap session driver commits its research judgment
as a :class:`StrategyBrief` (thesis, entry/exit rules, parameter space, scenario sketch), and
a strong hosted coder model turns that brief — and only that brief — into one complete
:class:`~noctis.strategies.base.TraderStrategy` file. The driver never writes source; the
coder never invents edge. That division of labor is the whole point: the brief is the
research, the coder is the typist.

The engine owns no toolbox state, so it is exercised in isolation with a fake LLM client:

1. Compose a **stateless** prompt from the strategy contract (``TEMPLATE.py``, the API
   contract sheet, one complete worked-example seed, and the header/scenario rules) plus the
   brief.
2. One completion against the injected coder client — a bare, single, tool-free codegen call,
   streamed where the client can so the transport timeout bounds silence between chunks, never
   the whole thinking+generation wall clock (#98). Thinking rides on the client the composition
   root builds (``coder_thinking``, default on — authoring is the reasoning-heavy sub-task, #17;
   a thinking client's output ceiling grows by a thinking allowance so the file's budget
   survives, #98), and the byte-stable system prompt carries a cache breakpoint (gated on the
   client's ``prompt_cache``) so private retries re-read, never re-pay, the enlarged
   contract/worked-example prefix.
3. Extract the fenced code block; a reply carrying none is rejected and counts as an attempt.
4. Validate through the existing library write path
   (:func:`noctis.strategies.library.write_strategy`) — the same fresh-subprocess gate every
   write passes today. Validation is the sole arbiter of what lands; the engine never loosens it.
5. On a validation error, re-prompt the coder privately with the error context, up to
   :data:`_CODER_RETRIES` retries. The caller sees only the final outcome.
6. When the retry budget is exhausted, raise :class:`AuthoringError` carrying the final
   validation error.

Two composition modes ride the same loop, gate, and retry budget as plain authoring (#7):

* **Reference adaptation** — when the brief names an existing library strategy in
  ``reference``, that strategy's full source is folded into the prompt with a
  translate-don't-invent instruction, so the coder adapts proven structure instead of
  inventing from scratch. A ``reference`` that does not resolve in the library is rejected
  *before* any coder completion is spent (a bad brief, not a coder failure).
* **Revision** — when the target ``name`` already exists, its current source is folded in and
  the brief becomes the change request. The validated result replaces the file through the
  normal :func:`write_strategy` path; a revision that never validates leaves the previous
  version untouched (the library's own guarantee).

**The composition is ablatable, from the outside only (#223).** "Is the contract sheet earning its
tokens?" is only a runnable experiment if each piece the prompt is built from can be taken away one
at a time, so the five pieces the benchmark layer names — contract sheet, template, worked example,
feasibility rules, retry-hint enrichment — are plain constructor parameters here, defaulting to the
composition production ships (``AS_SHIPPED`` for the three-valued worked-example selection). They
are parameters and not config on purpose: production has no word for them, the mandate overlay
cannot reach them, and the engine never imports the eval layer that maps a benchmark spec onto them.

**Every completion can be bounded (#223).** ``attempt_timeout`` (seconds, ``None`` ⇒ today's
unbounded behavior) caps EACH completion, so a transport that accepted the request and never
answered costs one attempt — a retained :class:`CoderTimeout`, reported through the same
per-attempt seam as any coder error — instead of wedging the authoring job and the research phase
behind it.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from noctis.research.contract_sheet import CONTRACT_SHEET, hint_for_gate_error
from noctis.research.llm import LLMClient, Turn, cached_system
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.scenario_spec import SpecSuite, describe_spec

# The coder's output-token ceiling — the FILE's budget: sized so a full ~200-line strategy file
# never truncates mid-source under the current tokenizer. It deliberately diverges from — sits
# above — the driver loop's smaller ceiling (agent._MAX_TOKENS); output tokens are billed only as
# generated, so the slack costs nothing. A client that runs provider thinking gets
# _THINKING_ALLOWANCE added ON TOP, so this stays a pure file budget either way.
_MAX_TOKENS = 16000
# Extra output-token headroom for a coder client that runs provider thinking (Anthropic adaptive):
# on those models thinking and text share max_tokens, and the first field exercise of escalation
# (#98) showed sonnet-5's adaptive thinking eating most of a 32k ceiling — the file never fit in
# what remained, three attempts straight. Sized to cover that observed depth in full and added ON
# TOP of the configured ceiling, so the ceiling keeps a whole strategy file of headroom AFTER
# thinking by construction. Unused headroom is never billed.
_THINKING_ALLOWANCE = 32000
# Private re-prompts after the first attempt: initial + _CODER_RETRIES ≤ 3 coder completions
# per authoring job, whether an attempt failed as a non-code reply or a validation error.
_CODER_RETRIES = 2

# The body of the first fenced code block in a reply (```python … ``` or a bare ``` … ```).
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# The committed seed folded into EVERY coder system prompt as one complete worked example.
# In the failure-census session the coder's one success was the single brief that carried a
# reference — the one time it saw a whole working file built on the real APIs. This grounds
# every prompt the same way, referenced or not. donchian_breakout is the richest stateful
# entry/exit example the seed tier ships: two rolling deques plus a position latch, a breakout
# entry read against strictly-prior history (no lookahead) and a breakdown exit, with its own
# known-outcome scenarios. It is read-only input — never mutated, only shown to the coder.
WORKED_EXAMPLE_NAME = "donchian_breakout"


class AsShipped(Enum):
    """The 'no ablation' value of the worked-example selection: whatever production composes.

    A deliberate twin of the eval layer's sentinel rather than a shared import: the engine may
    never import :mod:`noctis.eval` (the boundary guard in ``tests/test_eval_boundary.py``), so the
    benchmark layer maps its spec onto this one when it drives an ablation.
    """

    TOKEN = "as_shipped"

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return "AS_SHIPPED"


AS_SHIPPED = AsShipped.TOKEN

# The worked-example dial is three-valued: the seed production ships (``AS_SHIPPED``), another
# seed by name (does *which* example matter?), or none at all (does an example matter?).
WorkedExample = str | None | AsShipped

_ROLE_RULES = (
    "You are a strategy-authoring coder for the Noctis paper-trading research system. You "
    "translate a fully-specified research brief into exactly one complete, valid Noctis "
    "strategy file. The brief IS the research judgment: never invent a thesis, an edge, or "
    "symbols the brief did not give you — turn the brief into code, nothing more."
)

_CONTRACT_RULES = (
    "The file defines exactly one TraderStrategy subclass whose `name` attribute equals the "
    "strategy/file name. It carries a module-docstring header (a one-paragraph thesis then "
    "`status:`/`style:` lines), a frozen `Params` dataclass, an `on_bar` that ends every bar "
    "with ctx.set_target(+1 long / -1 short / 0 flat) and stays O(lookback) with no I/O, "
    "globals, or randomness, a `param_space()` classmethod, and a `scenarios()` classmethod "
    "returning 2-8 known-outcome tapes (built from the noctis.strategies.scenarios DSL) — "
    "including at least one tape that demands a directional entry and one always_flat() "
    "no-trade tape, with windows derived from the Params defaults. Code that violates its own "
    "declared scenarios is rejected by the validation gate."
)

_FEASIBILITY_RULES = (
    "You own tape construction. The brief's scenario sketch states INTENT — each tape's shape "
    "and the behavior it must prove — not a literal tape to transcribe; when the sketch asks for "
    "something a tape cannot honor, build the tape that proves the same intent. Apply these "
    "feasibility rules BEFORE you place any expectation window:\n"
    "  - Derive warmup from the Params defaults: find the longest lookback the on_bar path needs "
    "to produce a non-None indicator, and start every directional expectation window strictly "
    "AFTER that many bars — an indicator is None until it has seen enough history.\n"
    "  - A higher-timeframe strategy (one that aggregates N base bars into each decision bar) "
    "multiplies warmup by N: budget warmup in base bars, not decision bars.\n"
    "  - A percentile-rank condition over an indicator's OWN rolling window is scale-free: the "
    "rank depends only on ordering inside the window, so no amplitude — however small — silences "
    "it. You CANNOT keep such a rule flat by shrinking chop amplitude; do not try.\n"
    "  - Build the reliable no-trade tape by FALSIFYING the level condition, never by hoping "
    "quiet chop stays below a threshold: e.g. under a long-only rule, a steady selloff drives the "
    "condition the wrong way and provably never triggers an entry."
)

_OUTPUT_RULES = (
    "Reply with EXACTLY ONE fenced ```python code block containing the complete strategy "
    "file, and no prose outside that block. Do not omit or abbreviate any part of the file."
)


@dataclass(frozen=True)
class StrategyBrief:
    """The research judgment a driver commits for the coder to translate — never invent.

    The four required fields are the division-of-labor guard: a brief that could degenerate
    to "write me something profitable" would mean research silently moved to the coder. The
    optional slots steer provenance (``style``/``symbols``) and name a library strategy to
    adapt (``reference``) — its source is composed into the prompt as proven structure to
    translate, and an unknown reference is rejected before any coder completion.
    """

    thesis: str
    entry_exit: str
    param_space: str
    scenarios: str
    reference: str | None = None
    style: str | None = None
    symbols: tuple[str, ...] = ()


class AuthoringError(Exception):
    """The coder could not turn the brief into a file that survives validation in budget.

    Carries the final validation error both as ``__cause__`` and on ``validation_error`` so
    the caller can surface the exact gate message — the driver refines the brief, never the
    gate.
    """

    def __init__(self, message: str, *, validation_error: Exception | None = None) -> None:
        super().__init__(message)
        self.validation_error = validation_error


class CoderTimeout(library.StrategyValidationError):
    """One coder completion outlived the per-attempt timeout, so the attempt failed unanswered.

    A :class:`~noctis.strategies.library.StrategyValidationError` by inheritance for the same
    reason a prose-only reply is one: the engine's failure vocabulary is "this attempt did not
    produce a file the gate accepted", and every consumer downstream — the per-attempt hook, the
    session event, the capped ``failed/`` record, the final :class:`AuthoringError` — already
    speaks it. Nothing was validated; nothing arrived to validate.
    """


def _complete_within(call: Callable[[], Turn], timeout: float) -> Turn:
    """Run ``call`` on a worker thread and give up on it after ``timeout`` seconds.

    The client seam is a plain synchronous ``complete()`` with no cancellation and no timeout
    parameter of its own (:class:`~noctis.research.llm.LLMClient`), so bounding it means bounding
    the *wait*, not the call: a wedged request cannot be cancelled by anyone but its transport.
    The worker is a **daemon** thread — one per bounded attempt, at most ``retries + 1`` per job —
    so an abandoned one can neither block interpreter exit nor be joined at shutdown by a pool,
    and it ends by itself when the transport's own request timeout fires
    (:mod:`noctis.research.llm` sends an explicit one on every completion). A completion that
    answers in time leaves nothing behind: the worker has already finished when the join returns.

    Whatever the call raised is re-raised here, unchanged, so a bounded attempt fails exactly the
    way an unbounded one does.
    """
    answer: list[Turn] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            answer.append(call())
        except BaseException as exc:  # noqa: BLE001 — re-raised verbatim in the calling thread
            failure.append(exc)

    worker = threading.Thread(target=run, name="noctis-coder-attempt", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise CoderTimeout(
            f"the coder completion did not answer within the per-attempt timeout of {timeout}s; "
            "the attempt was abandoned unanswered"
        )
    if failure:
        raise failure[0]
    return answer[0]


def _fixed_oracle_block(spec: SpecSuite) -> str:
    """The coder-facing presentation of the FIXED scenario oracle on the spec path (#85).

    Renders the oracle faithfully from the :class:`SpecSuite` (tape shapes, behaviors, target
    legs — via :func:`describe_spec`) and states the two structural facts that flip the coder's
    job from authoring tapes to satisfying a fixed one: the gate stamps ``scenarios()`` (so the
    coder must NOT author one), and each tape is preceded by a flat setup pad sized to the
    strategy's own declared warmup (so the Params defaults' warmup must stay modest — a warmup too
    large for the fixed tape is rejected; shrink the lookback defaults, never enlarge the tape)."""
    return (
        "FIXED SCENARIO ORACLE — do NOT author a scenarios() method: the write gate stamps one "
        "from the fixed spec below and REJECTS any scenarios() you write. The tape shape and the "
        "one behavior each tape must prove are FIXED; change only the trading logic "
        "(on_start / on_bar / param_space / Params) to satisfy them. Each tape is preceded by a "
        "flat setup pad sized to your strategy's declared warmup — warmup_bars(Params defaults) — "
        "so keep the Params defaults' warmup modest; a warmup too large for the fixed tape is "
        "rejected, so shrink the lookback defaults rather than trying to enlarge the tape. The "
        "fixed oracle (leg kind(bars) in order — the compiler derives every bar window from the "
        "leg lengths and your warmup, so you never write a bar index):\n"
        f"{describe_spec(spec)}"
    )


def _extract_code_block(text: str) -> str | None:
    """The first fenced code block's body, or ``None`` for a reply carrying no code."""
    match = _FENCE_RE.search(text or "")
    return match.group(1) if match else None


def _discard_delta(kind: str, text: str) -> None:
    """Discard one streaming delta — passed on every coder completion as a transport measure, not
    a rendering path (#98). A streaming-capable client then streams the completion, so its per-read
    timeout bounds *silence between chunks* instead of the whole thinking+generation wall clock —
    which a thinking hosted coder on a full authoring brief can legitimately exceed. The engine
    renders nothing; a non-streaming client ignores the hook and completes exactly as before."""


def _extraction_error(stop_reason: str) -> library.StrategyValidationError:
    """The retry correction for a reply whose code block would not extract — keyed on why.

    Two causes wear the same symptom (no closing fence, so :func:`_extract_code_block` returns
    ``None``) but need opposite corrections. When the transport truncated the completion at the
    output-token limit (``stop_reason == "length"``), the closing fence never arrived: telling
    the coder it "returned no code block" would send it to repeat the exact attempt at the same
    cap. Name the real cause and ask for a *terser* complete file instead — the only actionable
    move. Otherwise the reply genuinely carried no code (prose only), and the
    return-one-code-block correction stands byte-for-byte. Classification is keyed ONLY on the
    stop reason and reached ONLY after extraction has already failed."""
    if stop_reason == "length":
        return library.StrategyValidationError(
            "the reply was cut off by the output-token limit before the closing ``` fence "
            "arrived, so no complete code block landed; regenerate the ENTIRE strategy file more "
            "tersely — a shorter docstring, fewer inline comments, and no prose outside the code "
            "block — so the whole file fits under the limit"
        )
    return library.StrategyValidationError(
        "the reply carried no ```python code block; return the complete strategy "
        "file as one fenced code block and nothing else"
    )


def _seed_template(strategies_dir: library.LibrarySpec) -> str:
    """The committed ``TEMPLATE.py`` seed source, or ``""`` when it is not on disk.

    Best-effort: the template grounds the prompt in production (the seed tier ships it), but
    a caller pointed at a bare/empty library still authors — the coder gets the rules text.
    """
    seeds = library.LibraryPaths.coerce(strategies_dir).seeds
    try:
        return (seeds / library.TEMPLATE_NAME).read_text(encoding="utf-8")
    except OSError:
        return ""


def _seed_example(strategies_dir: library.LibrarySpec, selection: WorkedExample) -> str:
    """The selected worked-example seed's source, or ``""`` when there is none to compose.

    ``AS_SHIPPED`` reads the seed production folds in (:data:`WORKED_EXAMPLE_NAME`), a name reads
    that seed instead, and ``None`` composes no example at all. Best-effort, mirroring
    :func:`_seed_template`: a complete real-API strategy grounds the coder, but a degraded install
    with no seed on disk still authors — the coder gets the rules text and the API contract sheet.
    Read directly from the read-only ``seeds`` tier so a working copy never shadows the committed
    example (and the seed is never mutated).
    """
    if selection is None:
        return ""
    name = WORKED_EXAMPLE_NAME if selection is AS_SHIPPED else str(selection)
    seeds = library.LibraryPaths.coerce(strategies_dir).seeds
    try:
        return (seeds / f"{name}.py").read_text(encoding="utf-8")
    except OSError:
        return ""


class StrategyAuthor:
    """Brief in, validated strategy source out — the coder-model authoring engine.

    Deep module, small surface: construct with a coder client + the library write target,
    call :meth:`author`. Holds no session/toolbox state, so each job is independent and each
    coder completion is a fresh, self-contained prompt.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        strategies_dir: library.LibrarySpec,
        families: FamilyRegistry,
        max_tokens: int = _MAX_TOKENS,
        retries: int = _CODER_RETRIES,
        template_source: str | None = None,
        include_contract_sheet: bool = True,
        include_template: bool = True,
        worked_example: WorkedExample = AS_SHIPPED,
        include_feasibility_rules: bool = True,
        include_retry_hints: bool = True,
        attempt_timeout: float | None = None,
    ) -> None:
        self._client = client
        self._strategies_dir = strategies_dir
        self._families = families
        # Whether an emit-failure re-prompt carries the contract sheet's true-signature hint for a
        # known helper-API mistake (the fifth ablation dial). Default on = production.
        self._retry_hints = include_retry_hints
        # The per-completion wall-clock bound, in seconds. ``None`` — the default and every
        # production construction — leaves the completion unbounded here, exactly as before, with
        # only the transport's own timeouts under it.
        self._attempt_timeout = attempt_timeout
        # The per-call output ceiling. ``max_tokens`` (the config's coder_max_tokens, or the
        # built-in default) is the FILE's budget; a client that runs provider thinking — its
        # ``thinking_enabled`` is duck-typed, absent ⇒ False — shares that ceiling with its
        # thinking on Anthropic models, so the allowance rides on top and a full file of headroom
        # survives the thinking by construction (#98).
        self._max_tokens = max_tokens + (
            _THINKING_ALLOWANCE if getattr(client, "thinking_enabled", False) else 0
        )
        self._max_attempts = 1 + max(0, retries)
        # The template dial decides whether a template block is composed at all; the pre-existing
        # ``template_source`` override decides WHICH source that block carries. Dialled off, no
        # override can put one back — the ablation is "the coder never saw a template".
        template = ""
        if include_template:
            template = (
                template_source if template_source is not None else _seed_template(strategies_dir)
            )
        example = _seed_example(strategies_dir, worked_example)
        self._system_prompt = self._build_system_prompt(
            template,
            example,
            contract_sheet=include_contract_sheet,
            feasibility_rules=include_feasibility_rules,
        )
        # The coder's system prompt is byte-stable for the engine's life (contract sheet +
        # feasibility rules + one worked example — enlarged by #14-#16). Wrap it in one cache
        # breakpoint here, once, gated on the client's prompt_cache capability, and pass it by
        # identity on every completion: on a caching provider the enlarged prefix is written to
        # cache on the first attempt and re-read — never re-billed — by every private retry within
        # a job. ``cached_system`` returns the plain string on a non-caching/auto-caching provider,
        # so the breakpoint is a clean no-op there (the coder loop is otherwise unchanged).
        self._system = cached_system(self._system_prompt, cache=client.capabilities.prompt_cache)

    def author(
        self,
        name: str,
        brief: StrategyBrief,
        *,
        on_attempt: Callable[[int, Exception | None, str], None] | None = None,
        spec: SpecSuite | None = None,
    ) -> dict:
        """Turn ``brief`` into a validated ``name`` strategy file in the working tier.

        Returns the :func:`noctis.strategies.library.write_strategy` result (name/path/header)
        on success. Raises :class:`AuthoringError` when the coder cannot produce a file that
        passes the write gate within the retry budget. Every private retry is invisible to
        the caller.

        ``spec`` (#84) is the compiled scenario oracle: when supplied it is forwarded to the write
        gate, which replays it at the candidate's own declared warmup and machine-stamps a
        ``scenarios()`` block into the file. Default ``None`` keeps today's behavior — the coder's
        own declared scenarios validate as before.

        When ``brief.reference`` names a library strategy, its source is composed into the
        prompt as structure to adapt; an unknown reference raises
        :class:`~noctis.strategies.library.StrategyValidationError` *before* any coder
        completion. When ``name`` already exists, its current source is composed in and the
        brief becomes a revision request.

        ``on_attempt`` — the observability seam — is called exactly once per coder completion,
        *after* that attempt's validation resolves: ``on_attempt(attempt, None, source)`` on the
        completion that lands, ``on_attempt(attempt, error, source)`` (a
        :class:`~noctis.strategies.library.StrategyValidationError`) when a gate rejection, a
        non-code reply, an output-limit truncation, or a per-attempt timeout fails an attempt.
        ``attempt`` is 1-based (1 = first, 2… = a private retry); ``source`` is the attempt's
        material — the extracted code block on a gate rejection or success, the raw reply text on
        a non-code reply or a truncated completion, and ``""`` for a timeout that produced no bytes
        at all — carried so a toolbox-side sink can persist the exact bytes the
        coder produced. The engine holds no session state and keeps none of the source; the
        caller (the toolbox) adapts this into a session event carrying the coder model and
        strategy name, and into the capped ``failed/`` record.
        """
        reference_source = self._reference_source(brief)  # unknown ref → raise, no completion
        current_source = self._current_source(name)  # non-None ⇒ this is a revision
        prior: tuple[str, str] | None = None
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            content = self._user_prompt(
                name,
                brief,
                prior,
                reference_source=reference_source,
                current_source=current_source,
                spec=spec,
            )
            try:
                turn = self._completion(content)
            except CoderTimeout as exc:
                # The transport never answered inside the per-attempt bound. That is this
                # attempt's failure, reported like any other coder error — with no material,
                # because no bytes arrived — and the loop moves on to the next attempt. ``prior``
                # stays as it was: there is no previous source to correct, so the next attempt is
                # the same brief asked again rather than a fix-your-attempt re-prompt.
                last_error = exc
                self._report(on_attempt, attempt, exc, "")
                continue
            source = _extract_code_block(turn.text)
            if source is None:
                # No closing fence to extract. Classify why on the turn's own stop reason: an
                # output-limit truncation ("length") gets a name-the-cause, regenerate-terser
                # correction; a genuine no-code reply keeps the prose-only correction unchanged.
                last_error = _extraction_error(turn.stop_reason)
                prior = (turn.text or "", str(last_error))
                self._report(on_attempt, attempt, last_error, turn.text or "")
                continue
            try:
                result = library.write_strategy(
                    self._strategies_dir, name, source, self._families, spec=spec
                )
            except library.StrategyValidationError as exc:
                last_error = exc
                prior = (source, str(exc))
                self._report(on_attempt, attempt, exc, source)
                continue
            self._report(on_attempt, attempt, None, source)
            return result
        raise AuthoringError(
            f"the coder could not author a valid {name!r} strategy in {self._max_attempts} "
            f"attempts; last gate error: {last_error}",
            validation_error=last_error,
        ) from last_error

    def _completion(self, content: str) -> Turn:
        """One coder completion for ``content`` — bounded by the per-attempt timeout when set.

        Unbounded (the default and every production construction) the client is called inline on
        the caller's own thread, exactly as before. With a timeout the same call is made through
        :func:`_complete_within`, which turns a completion that never answers into a
        :class:`CoderTimeout` for this attempt.
        """

        def call() -> Turn:
            return self._client.complete(
                system=self._system,
                tools=[],
                messages=[{"role": "user", "content": content}],
                max_tokens=self._max_tokens,
                on_delta=_discard_delta,  # stream where the client can — a transport bound (#98)
            )

        if self._attempt_timeout is None:
            return call()
        return _complete_within(call, self._attempt_timeout)

    @staticmethod
    def _report(
        on_attempt: Callable[[int, Exception | None, str], None] | None,
        attempt: int,
        error: Exception | None,
        source: str,
    ) -> None:
        """Report one resolved attempt to the observability hook (a no-op when unset).

        ``source`` is the attempt's material — the extracted code block on a gate rejection or
        success, the raw reply text on a non-code reply or a truncated completion — passed so a
        toolbox-side sink can persist the exact bytes the coder produced. The engine itself keeps
        none of it.
        """
        if on_attempt is not None:
            on_attempt(attempt, error, source)

    # ── reference / revision resolution ──────────────────────────────────────
    def _reference_source(self, brief: StrategyBrief) -> str | None:
        """The named reference strategy's source to adapt, or ``None`` when none was named.

        An unknown reference is a bad brief, not a coder failure: it raises
        :class:`~noctis.strategies.library.StrategyValidationError` here — *before* the loop —
        so no completion is spent and the caller surfaces a repairable, actionable error.
        """
        if not brief.reference:
            return None
        try:
            return library.strategy_source(self._strategies_dir, brief.reference)
        except FileNotFoundError:
            raise library.StrategyValidationError(
                f"brief.reference names {brief.reference!r}, which is not in the strategy "
                f"library; reference an existing strategy (list_strategies shows what exists) "
                f"or drop the reference to author from scratch"
            ) from None

    def _current_source(self, name: str) -> str | None:
        """The current source of ``name`` when it already exists, else ``None``.

        A non-``None`` result makes this a *revision*: the brief is a change request against
        the returned source, and the validated result replaces it through the normal write
        path (a failed revision leaves it untouched — the library's own guarantee).
        """
        try:
            return library.strategy_source(self._strategies_dir, name)
        except FileNotFoundError:
            return None

    # ── prompt composition ───────────────────────────────────────────────────
    def _build_system_prompt(
        self,
        template: str,
        example: str,
        *,
        contract_sheet: bool = True,
        feasibility_rules: bool = True,
    ) -> str:
        # The contract sheet grounds the coder in the exact helper signatures the write gate
        # executes — the surface TEMPLATE.py deliberately elides — so it never hallucinates an
        # API. The feasibility rules make the coder own tape construction and kill the
        # unsatisfiable-brief retry loop from the coder's end. Both are independent of the
        # template, so a bare library still ships the full API surface and the discipline.
        # Each is composed unless its ablation dial took it away, and taking one away removes
        # exactly that block — the surrounding assets and their order never move.
        parts = [_ROLE_RULES, _CONTRACT_RULES]
        if contract_sheet:
            parts.append(CONTRACT_SHEET)
        if feasibility_rules:
            parts.append(_FEASIBILITY_RULES)
        if template:
            parts.append(
                "Here is TEMPLATE.py — the canonical shape every strategy file follows. "
                "Mirror its structure:\n\n```python\n" + template + "```"
            )
        # One complete worked example, UNCONDITIONALLY — a real shipped strategy that wires the
        # exact APIs above end to end (incremental state, no-lookahead indicator reads, a
        # directional tape and a no-trade tape). In the failure census the coder converged only
        # when it saw a full working file; every prompt now carries one, referenced or not.
        # Best-effort like the template: a degraded install with no seed still authors rules-only.
        if example:
            parts.append(
                "Here is a COMPLETE WORKED EXAMPLE — a real strategy file that uses the APIs "
                "above end to end: it resets incremental state in on_start, reads its indicator "
                "against strictly-prior history (no lookahead), ends every bar with a target, and "
                "declares its own directional and no-trade scenarios. Study how the pieces wire "
                "together and author your file the same way — but implement the BRIEF's edge, not "
                "this one:\n\n```python\n" + example + "```"
            )
        parts.append(_OUTPUT_RULES)
        return "\n\n".join(parts)

    def _user_prompt(
        self,
        name: str,
        brief: StrategyBrief,
        prior: tuple[str, str] | None,
        *,
        reference_source: str | None = None,
        current_source: str | None = None,
        spec: SpecSuite | None = None,
    ) -> str:
        lines = [
            f"Author the strategy file for name: {name}",
            "",
            "BRIEF",
            f"Thesis: {brief.thesis}",
            f"Entry/exit rules: {brief.entry_exit}",
            f"Parameter space: {brief.param_space}",
        ]
        # On the spec path (#85) the oracle is FIXED and machine-stamped by the gate: the coder is
        # briefed against it and must NOT author scenarios(). Without a spec the coder owns tape
        # construction, so it reads the brief's free-prose scenario sketch, exactly as before.
        if spec is not None:
            lines.append(_fixed_oracle_block(spec))
        else:
            lines.append(f"Scenario sketch: {brief.scenarios}")
        if brief.style:
            lines.append(f"Style: {brief.style}")
        if brief.symbols:
            lines.append(f"Researched symbols: {' '.join(brief.symbols)}")
        if brief.reference:
            lines.append(f"Reference strategy to adapt: {brief.reference}")
        prompt = "\n".join(lines)
        if reference_source:
            prompt += (
                f"\n\nREFERENCE STRATEGY ({brief.reference}) — proven structure to adapt, not "
                "invent past. Translate this file's structure into the BRIEF above; keep only "
                "the edge the brief specifies and add nothing the brief did not ask for:\n"
                "```python\n" + reference_source + "```"
            )
        if current_source:
            prompt += (
                f"\n\nCURRENT VERSION of {name} — this strategy already exists; the BRIEF is a "
                "change request against it. Revise this file to satisfy the brief, preserving "
                "everything the brief does not ask you to change, and return the complete "
                "revised file:\n```python\n" + current_source + "```"
            )
        if prior is not None:
            source, error = prior
            prompt += (
                "\n\nYour previous attempt did not pass validation. Fix it and return the "
                "complete corrected file.\n"
                f"Validation error: {error}\n"
            )
            # When the gate error names a helper the contract sheet declares (a State .update()
            # arity slip, an unknown ExitRules field, a scenario-builder kwarg typo), append the
            # true signature from that same table row — the coder is stateless across jobs, so the
            # retry must carry the real API to fix the actual mistake. Unmatched errors add nothing,
            # and neither does an engine whose retry-hint dial is off (the ablation, #223).
            hint = hint_for_gate_error(error) if self._retry_hints else None
            if hint:
                prompt += f"{hint}\n"
            prompt += "Previous source:\n```python\n" + source + "\n```"
        return prompt
