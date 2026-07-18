"""Misfire classification — the model stumbles the loop corrects and retries.

A *misfire* is an attempted move that never became an executable tool call, as opposed to
a deliberate plain-text conclusion (``agent_done``) or a genuine transport failure
(``api_error``). Small local backends produce four faces of the same stumble:

* **text-form markup** — the "tool call" is written as literal Hermes/Qwen-style
  ``<tool_call>``/``<function=`` markup in the thinking or text channel, where no template
  parses it, so the turn arrives with zero native tool calls;
* **truncation** — the completion ran out of output room (``finish_reason="length"``),
  usually mid-thinking, before any tool call;
* **thinking-only** — the model plans in its reasoning channel and just stops: empty text,
  no tool calls, no markup. Reasoning is invisible to the session, so an empty turn is
  never a valid protocol move (act via a tool call, or conclude in text);
* **invalid call** — the stumble surfaces as an *exception*: a backend that parses tool
  calls itself (llama-server) rejects a call whose JSON arguments were cut off by the
  output limit, and the whole completion raises.

Each classifies to one :class:`Misfire` carrying the operator-facing feed note and the
corrective user message the loop appends before re-completing. Every retried round still
burns an iteration, so a persistent misfirer ends the session through the ordinary
``max_iterations`` budget — retries are bounded by construction, never infinite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noctis.research.llm import Turn


@dataclass(frozen=True)
class Misfire:
    """One classified stumble: ``note`` is the one-line feed entry (emitted as
    ``[misfire] {note}``), ``retry`` the corrective user message sent back to the model."""

    note: str
    retry: str


_TOOL_MARKUP_TOKENS = ("<tool_call>", "<function=")
# Only this exception class is retried — a genuinely unreachable backend is not a misfire.
_MALFORMED_CALL_MARKERS = ("invalid tool call arguments", "unexpected end of json")

_TEXT_MARKUP = Misfire(
    note=(
        "tool call written as text markup instead of a native tool call — asking for a "
        "native re-issue"
    ),
    retry=(
        "Your tool call was written as literal text/markup (e.g. <tool_call>) inside your "
        "reasoning or reply, so it was never executed — only native tool calls run. Re-issue "
        "the same call through the native tool-call mechanism now, or state your final "
        "conclusion as plain text."
    ),
)
_TRUNCATED = Misfire(
    note=(
        "completion truncated by the output limit before any tool call — asking for a shorter turn"
    ),
    retry=(
        "Your last reply was cut off by the output limit before it produced a tool call. "
        "Keep your reasoning brief this time, then either issue one native tool call or "
        "state your final conclusion as plain text."
    ),
)
_THINKING_ONLY = Misfire(
    note="thinking-only turn (no text, no tool call) — asking for an action or a conclusion",
    retry=(
        "Your last turn had no visible text and no tool call — reasoning alone is invisible "
        "to the session and executes nothing. Either issue your next action as a native tool "
        "call now, or state your final conclusion as plain text."
    ),
)
_INVALID_CALL = Misfire(
    note=("backend rejected a truncated/invalid tool call — asking for a smaller, valid re-issue"),
    retry=(
        "Your last tool call was rejected before execution: its JSON arguments were invalid "
        "— usually cut off by the output limit. Keep your reasoning brief and re-issue a "
        "smaller, valid call (for a long file, a shorter source), or state your final "
        "conclusion as plain text."
    ),
)


def classify_turn(turn: Turn) -> Misfire | None:
    """Classify a turn that produced zero usable native tool calls: a :class:`Misfire` to
    correct and retry, or ``None`` — the turn carries plain text and is the agent's
    deliberate final conclusion (``agent_done``)."""
    if turn.stop_reason == "length":
        return _TRUNCATED
    blob = f"{turn.reasoning}\n{turn.text}"
    if any(token in blob for token in _TOOL_MARKUP_TOKENS):
        return _TEXT_MARKUP
    if not turn.text.strip():
        return _THINKING_ONLY
    return None


def classify_completion_error(exc: Exception) -> Misfire | None:
    """Classify a ``complete()`` exception: the backend rejecting a truncated/invalid tool
    call is a model stumble worth retrying; anything else — a transport/availability
    failure — returns ``None`` and ends the session as ``api_error``."""
    text = str(exc).lower()
    if any(marker in text for marker in _MALFORMED_CALL_MARKERS):
        return _INVALID_CALL
    return None
