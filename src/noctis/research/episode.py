"""The episode runner — one forced structured-emit LLM call, misfires retried, JSON-in-text
fallback, all behind a briefing-in / typed-record-out surface.

The episodic research driver (epic #62) invokes the LLM only at narrow judgment points it calls
*episodes*: a single stateless call whose whole prompt — a system prompt plus one briefing the
driver rebuilt fresh from disk — is answered through one forced structured-emit tool. There is no
accumulated transcript across episodes, so a session can never overflow its context window
mid-run; each episode is a pure function of the prompt the caller hands in.

:class:`EpisodeRunner` is that call, made reusable:

* **One forced emit.** The runner offers exactly one function tool (the caller's
  :class:`EmitContract`) and forces it with the proven ``tool_choice`` idiom, so a compliant
  backend answers with a single structured tool call and nothing else.
* **One validation path, both transports.** A small local server sometimes mishandles the forced
  call and answers in prose; the runner then extracts a JSON object from the text and validates it
  against the SAME typed ``parse`` as a native tool call. Tool-call args and JSON-in-text meet at
  one parser — never two schemas that could drift.
* **Misfires classified and retried, bounded.** A turn that produced no usable emit (markup
  instead of a native call, an output-limit truncation, a thinking-only stall), a ``complete()``
  exception the backend raised on a truncated call, or a payload that fails the schema is a
  *misfire*, classified by :mod:`noctis.research.misfire` and re-prompted with its corrective, up
  to ``retries`` times. When the budget is spent the episode returns a typed failure — never an
  exception cascade — and the caller decides what that means. A genuine transport outage (a
  non-misfire exception) is a typed failure too, not a raise.
* **Counted once, in one place.** Every episode (retries folded in) increments
  :attr:`EpisodeRunner.episodes`, so the driver enforces ``max_episodes`` off one counter without
  double-counting a retried episode.

The result is an :class:`EpisodeResult`: the typed value on success, plus the model, token total,
misfire count, and outcome a caller writes to the session ledger's ``episode`` line
(:class:`~noctis.research.ledger.Episode`) before acting on it. The runner never writes the ledger
itself — persistence is the driver's; "suitable for ledger persistence" means the fields are here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from noctis.research.llm import LLMClient, Turn
from noctis.research.misfire import (
    classify_completion_error,
    classify_emit_failure,
    classify_turn,
)

T = TypeVar("T")

# Episodes emit a small structured payload; the default output ceiling is modest so it leaves
# room for the prompt on the small-context local model this epic targets, while still giving a
# reasoning backend headroom to think before it emits. Output is billed as generated, so unused
# headroom costs nothing; the driver overrides this from research.agent.max_tokens when it needs.
_DEFAULT_MAX_TOKENS = 2048

# The neutral four-field usage dict on a Turn; every field read defensively so a fake/no-usage
# client contributes 0 rather than raising (the token total is a measurement, never a gate).
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# The body of the first fenced code block in a reply (```json … ``` or a bare ``` … ```).
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Episode outcomes — the string a caller writes to the ledger's ``episode`` line.
OK = "ok"
MISFIRES_EXHAUSTED = "misfires_exhausted"
API_ERROR = "api_error"


@dataclass(frozen=True)
class EmitContract(Generic[T]):
    """The structured-emit contract for one kind of episode.

    ``name``/``description``/``schema`` build the single forced function tool; ``parse`` is the
    ONE typed validation both transports meet at — it turns the emitted payload dict into the
    typed record ``T`` and raises on anything the schema does not admit. A caller defines one
    contract per episode kind (a match verdict, a decide verdict, …) and reuses it across calls.
    """

    name: str
    description: str
    schema: dict[str, Any]
    parse: Callable[[dict[str, Any]], T]


@dataclass(frozen=True)
class EpisodeResult(Generic[T]):
    """One episode's typed outcome — the record a caller writes to the ledger before acting.

    ``value`` is the parsed record on success and ``None`` on any failure; ``outcome`` is one of
    :data:`OK` / :data:`MISFIRES_EXHAUSTED` / :data:`API_ERROR`. ``model``, ``tokens`` (summed
    across the episode's completions, retries included), and ``misfires`` are exactly the fields
    :meth:`~noctis.research.ledger.SessionLedger.record_episode` needs; ``note`` carries the last
    misfire/error text for observability (never required by the ledger).
    """

    outcome: str
    value: T | None
    model: str
    tokens: int
    misfires: int
    note: str = ""

    @property
    def ok(self) -> bool:
        """True iff the episode produced a validated typed record."""
        return self.value is not None


def _turn_tokens(usage: dict | None) -> int:
    """Total tokens one completion reported, 0 for any field a provider omits."""
    if not usage:
        return 0
    return sum(int(usage.get(field, 0) or 0) for field in _USAGE_FIELDS)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull one JSON object out of a text reply — the transport fallback for a backend that
    mishandles the forced tool call and answers in prose.

    Tries the whole reply, then a fenced ```json/``` block's body, then the outermost ``{…}``
    span; returns the first candidate that parses to a JSON object, or ``None`` when none do (a
    truncated or object-free reply). Only an object is accepted — a bare array/scalar is not a
    payload — so a non-emit reply falls through to misfire classification rather than mis-parsing.
    """
    if not text:
        return None
    candidates = [text.strip()]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _payload_from_turn(turn: Turn, tool_name: str) -> dict[str, Any] | None:
    """The emitted payload dict — from the forced tool call if the backend produced one, else the
    JSON-in-text fallback. ``None`` when neither transport yielded an object to validate."""
    for call in turn.tool_calls:
        if call.name == tool_name:
            return call.arguments if isinstance(call.arguments, dict) else None
    return _extract_json_object(turn.text)


class EpisodeRunner:
    """Runs forced-emit episodes against one injected :class:`~noctis.research.llm.LLMClient`.

    Construct with the client and the retry bound (the composition root wires
    ``research.agent.episode_retries`` here — the runner never reads Settings), then call
    :meth:`run` once per judgment point. The runner holds no per-episode state beyond the
    :attr:`episodes` counter, so each call is a fresh, self-contained round trip.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        retries: int,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._retries = max(0, retries)
        self._max_tokens = max_tokens
        # Completed episodes, retries folded in — one per run() call regardless of internal
        # retries or outcome, so the driver enforces max_episodes off this single counter.
        self.episodes = 0

    def run(
        self,
        *,
        contract: EmitContract[T],
        system: str | list[dict[str, Any]],
        briefing: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> EpisodeResult[T]:
        """Run one episode: force the ``contract`` emit tool against ``system`` + ``briefing``,
        validate the payload through ``contract.parse``, and retry a misfire with its corrective
        up to the constructed retry bound. Returns a typed :class:`EpisodeResult` — never raises.

        The prompt is rebuilt fresh here from the caller's ``system`` and ``briefing``; a within-
        episode retry appends only the classifier's corrective user turn (the misfired assistant
        turn is not replayed — nothing parseable), so the transcript stays tiny and bounded.
        ``model`` labels the ledger line (defaults to the client's own ``model``); ``max_tokens``
        overrides the runner default for this call (a small-context backend compatibility lever).
        """
        self.episodes += 1
        resolved_model = model if model is not None else str(getattr(self._client, "model", ""))
        tokens = 0
        misfires = 0
        note = ""
        value: T | None = None

        tools = [
            {
                "name": contract.name,
                "description": contract.description,
                "input_schema": contract.schema,
            }
        ]
        tool_choice = {"type": "function", "function": {"name": contract.name}}
        messages: list[dict[str, Any]] = [{"role": "user", "content": briefing}]

        for _ in range(self._retries + 1):
            try:
                turn = self._client.complete(
                    system=system,
                    tools=tools,
                    messages=messages,
                    max_tokens=max_tokens or self._max_tokens,
                    tool_choice=tool_choice,
                )
            except Exception as exc:  # noqa: BLE001 — an episode never crashes the driver
                stumble = classify_completion_error(exc)
                if stumble is None:
                    # A genuine transport/availability outage — a typed failure, not a retry.
                    return EpisodeResult(
                        outcome=API_ERROR,
                        value=value,
                        model=resolved_model,
                        tokens=tokens,
                        misfires=misfires,
                        note=_reason(exc),
                    )
                misfires += 1
                note = stumble.note
                messages = messages + [{"role": "user", "content": stumble.retry}]
                continue

            tokens += _turn_tokens(turn.usage)
            payload = _payload_from_turn(turn, contract.name)
            if payload is not None:
                try:
                    value = contract.parse(payload)
                except Exception as exc:  # noqa: BLE001 — a schema-invalid payload is a misfire
                    stumble = classify_emit_failure(_reason(exc))
                else:
                    return EpisodeResult(
                        outcome=OK,
                        value=value,
                        model=resolved_model,
                        tokens=tokens,
                        misfires=misfires,
                    )
            else:
                # No emit on either transport: the truncation/markup/thinking-only stumbles land
                # here, and a prose reply carrying no JSON object is the episode-only "no emit".
                stumble = classify_turn(turn) or classify_emit_failure(
                    "no JSON object in the reply"
                )

            misfires += 1
            note = stumble.note
            messages = messages + [{"role": "user", "content": stumble.retry}]

        return EpisodeResult(
            outcome=MISFIRES_EXHAUSTED,
            value=value,
            model=resolved_model,
            tokens=tokens,
            misfires=misfires,
            note=note,
        )


def _reason(exc: Exception) -> str:
    """A compact, bounded reason string for a corrective/note — the message, or the type name."""
    text = str(exc).strip()
    reason = text or type(exc).__name__
    return reason[:300]
