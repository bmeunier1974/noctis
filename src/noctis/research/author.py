"""The strategy-authoring engine — a structured brief in, a validated strategy file out.

:class:`StrategyAuthor` is the deep module behind the coder-model split (the epic in
``.claude/plan/coder-model-split.md``): a cheap session driver commits its research judgment
as a :class:`StrategyBrief` (thesis, entry/exit rules, parameter space, scenario sketch), and
a strong hosted coder model turns that brief — and only that brief — into one complete
:class:`~noctis.strategies.base.TraderStrategy` file. The driver never writes source; the
coder never invents edge. That division of labor is the whole point: the brief is the
research, the coder is the typist.

The engine owns no toolbox state, so it is exercised in isolation with a fake LLM client:

1. Compose a **stateless** prompt from the strategy contract (``TEMPLATE.py`` + the
   header/scenario rules) plus the brief.
2. One completion against the injected coder client — a bare, single, tool-free codegen call
   (thinking is pinned off where the client is built, ``client_for(..., thinking="off")``).
3. Extract the fenced code block; a reply carrying none is rejected and counts as an attempt.
4. Validate through the existing library write path
   (:func:`noctis.strategies.library.write_strategy`) — the same fresh-subprocess gate every
   write passes today. Validation is the sole arbiter of what lands; the engine never loosens it.
5. On a validation error, re-prompt the coder privately with the error context, up to
   :data:`_CODER_RETRIES` retries. The caller sees only the final outcome.
6. When the retry budget is exhausted, raise :class:`AuthoringError` carrying the final
   validation error.

Reference-adaptation and revision briefs (adapting a named library strategy, revising an
existing file) are a follow-up (#7); :class:`StrategyBrief` carries a ``reference`` slot so
the interface does not preclude them, but this engine does not compose them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from noctis.research.llm import LLMClient
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry

# The coder's output-token ceiling — the same default the agent loop sizes completions at
# (agent._MAX_TOKENS), chosen so a full ~200-line strategy file never truncates mid-source.
_MAX_TOKENS = 8000
# Private re-prompts after the first attempt: initial + _CODER_RETRIES ≤ 3 coder completions
# per authoring job, whether an attempt failed as a non-code reply or a validation error.
_CODER_RETRIES = 2

# The body of the first fenced code block in a reply (```python … ``` or a bare ``` … ```).
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

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

_OUTPUT_RULES = (
    "Reply with EXACTLY ONE fenced ```python code block containing the complete strategy "
    "file, and no prose outside that block. Do not omit or abbreviate any part of the file."
)


@dataclass(frozen=True)
class StrategyBrief:
    """The research judgment a driver commits for the coder to translate — never invent.

    The four required fields are the division-of-labor guard: a brief that could degenerate
    to "write me something profitable" would mean research silently moved to the coder. The
    optional slots steer provenance (``style``/``symbols``) and leave room for the follow-up
    reference-adaptation path (``reference``) without this engine composing it.
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


def _extract_code_block(text: str) -> str | None:
    """The first fenced code block's body, or ``None`` for a reply carrying no code."""
    match = _FENCE_RE.search(text or "")
    return match.group(1) if match else None


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
    ) -> None:
        self._client = client
        self._strategies_dir = strategies_dir
        self._families = families
        self._max_tokens = max_tokens
        self._max_attempts = 1 + max(0, retries)
        template = (
            template_source if template_source is not None else _seed_template(strategies_dir)
        )
        self._system_prompt = self._build_system_prompt(template)

    def author(self, name: str, brief: StrategyBrief) -> dict:
        """Turn ``brief`` into a validated ``name`` strategy file in the working tier.

        Returns the :func:`noctis.strategies.library.write_strategy` result (name/path/header)
        on success. Raises :class:`AuthoringError` when the coder cannot produce a file that
        passes the write gate within the retry budget. Every private retry is invisible to
        the caller.
        """
        prior: tuple[str, str] | None = None
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            turn = self._client.complete(
                system=self._system_prompt,
                tools=[],
                messages=[{"role": "user", "content": self._user_prompt(name, brief, prior)}],
                max_tokens=self._max_tokens,
            )
            source = _extract_code_block(turn.text)
            if source is None:
                last_error = library.StrategyValidationError(
                    "the reply carried no ```python code block; return the complete strategy "
                    "file as one fenced code block and nothing else"
                )
                prior = (turn.text or "", str(last_error))
                continue
            try:
                return library.write_strategy(self._strategies_dir, name, source, self._families)
            except library.StrategyValidationError as exc:
                last_error = exc
                prior = (source, str(exc))
        raise AuthoringError(
            f"the coder could not author a valid {name!r} strategy in {self._max_attempts} "
            f"attempts; last gate error: {last_error}",
            validation_error=last_error,
        ) from last_error

    # ── prompt composition ───────────────────────────────────────────────────
    def _build_system_prompt(self, template: str) -> str:
        parts = [_ROLE_RULES, _CONTRACT_RULES]
        if template:
            parts.append(
                "Here is TEMPLATE.py — the canonical shape every strategy file follows. "
                "Mirror its structure:\n\n```python\n" + template + "```"
            )
        parts.append(_OUTPUT_RULES)
        return "\n\n".join(parts)

    def _user_prompt(self, name: str, brief: StrategyBrief, prior: tuple[str, str] | None) -> str:
        lines = [
            f"Author the strategy file for name: {name}",
            "",
            "BRIEF",
            f"Thesis: {brief.thesis}",
            f"Entry/exit rules: {brief.entry_exit}",
            f"Parameter space: {brief.param_space}",
            f"Scenario sketch: {brief.scenarios}",
        ]
        if brief.style:
            lines.append(f"Style: {brief.style}")
        if brief.symbols:
            lines.append(f"Researched symbols: {' '.join(brief.symbols)}")
        if brief.reference:
            lines.append(f"Reference strategy to adapt: {brief.reference}")
        prompt = "\n".join(lines)
        if prior is not None:
            source, error = prior
            prompt += (
                "\n\nYour previous attempt did not pass validation. Fix it and return the "
                "complete corrected file.\n"
                f"Validation error: {error}\n"
                "Previous source:\n```python\n" + source + "\n```"
            )
        return prompt
