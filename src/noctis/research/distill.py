"""Stage-2 memory distillation — periodic LLM map-reduce over findings (context plan P3).

Every ``research.memory_distill_every`` completed research sessions (0 = off, the default),
one summarization call folds the full findings history into a ~15-line distilled-lessons
block persisted in ``MEMORY.md``'s machine-owned section; every session after embeds that
block plus the newest raw entries instead of a raw tail. Summarize once, reuse every session:
the call runs at CLOSE — the phase that already owns memory upkeep — never inside a research
session's own loop, and never on the session backend's context window.

Provider-neutral by construction: the one LLM call goes through the same
:class:`~noctis.research.llm.LLMClient` seam research uses (no SDK import here), and every
failure path — no knob, no client, refusal, transport error — degrades to stage-1 behavior.
Distillation reads *memory findings only*: never scorecards, never bars, so nothing
gate-adjacent can leak into or out of it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noctis.memory.base import Memory

logger = logging.getLogger("noctis.research.distill")

# Session counter lives in the state dir, never in MEMORY.md (which the agent owns/edits).
_COUNTER_FILE = "memory_distill.json"
_DISTILL_MAX_LINES = 15
_DISTILL_MAX_TOKENS = 1200
# Below this many raw findings a distillation would only re-state the tail it replaces.
_MIN_FINDINGS_TO_DISTILL = 10

_PROMPT = """\
You maintain the long-term memory of an autonomous trading-research system. Distill the
research findings below into at most {max_lines} class-level lessons.

Rules:
- One lesson per line, each starting with "- ".
- Preserve EVERY distinct dead-end class (things proven not to work) — merging duplicates
  is the goal, dropping the only record of a class is not acceptable.
- State each lesson at the class level (strategy family / style / horizon), not as a
  session anecdote; keep concrete numbers only where they carry the lesson.
- No dates, no new claims, no speculation beyond what the findings state.

FINDINGS (oldest first):
{findings}
"""


def _counter_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / _COUNTER_FILE


def _read_counter(state_dir: str | Path) -> int:
    path = _counter_path(state_dir)
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["sessions_since_distill"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def _write_counter(state_dir: str | Path, value: int) -> None:
    path = _counter_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sessions_since_distill": value}), encoding="utf-8")


def bump_research_session(state_dir: str | Path) -> int:
    """Count one completed research session toward the next distillation. Never raises —
    the counter is bookkeeping and must not break the research phase."""
    try:
        count = _read_counter(state_dir) + 1
        _write_counter(state_dir, count)
        return count
    except OSError:  # pragma: no cover - a read-only state dir
        logger.warning("memory distill counter unwritable under %s", state_dir)
        return 0


def distill_findings(memory: Memory, client, *, max_lines: int = _DISTILL_MAX_LINES) -> bool:
    """One map-reduce call: full findings history in, ≤ ``max_lines`` lessons persisted via
    ``memory.set_distilled``. Returns True only when a block was actually written."""
    findings = memory.findings()
    if len(findings) < _MIN_FINDINGS_TO_DISTILL:
        return False
    prompt = _PROMPT.format(max_lines=max_lines, findings="\n".join(findings))
    try:
        turn = client.complete(
            system="You compact research memory faithfully.",
            tools=[],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_DISTILL_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 — memory upkeep must never crash the close phase
        logger.warning("memory distillation call failed (%s); keeping stage-1 view", exc)
        return False
    bullets = [ln.strip() for ln in (turn.text or "").splitlines() if ln.strip().startswith("- ")]
    if not bullets:
        logger.warning("memory distillation returned no lessons; keeping stage-1 view")
        return False
    memory.set_distilled(bullets[:max_lines])
    return True


def _distill_client(settings):
    """Pick the LLM client memory distillation should use (context plan P3 / story #72).

    Routes to the PAID coder-fallback model when one is configured AND its provider key resolves:
    distillation is a small, once-per-cadence map-reduce, so when the operator already pays for a
    strong coder it is the natural model for it. When ``coder_fallback_model`` is unset, or its
    provider key/extra is missing (``client_for`` returns ``None``), it degrades to the existing
    local/default research client — byte-identical to before this story, so conversation-loop
    behavior is unchanged when the knob is not configured."""
    from noctis.research import build_llm_client

    fallback_model = getattr(settings.research.agent, "coder_fallback_model", None)
    if fallback_model:
        from noctis.research.llm import client_for

        paid = client_for(settings, fallback_model)
        if paid is not None:
            return paid
    return build_llm_client(settings)


def maybe_distill(settings, memory: Memory, *, client=None) -> bool:
    """The periodic trigger: distill when ≥ ``research.memory_distill_every`` sessions have
    completed since the last distillation. Off (0/None knob) and no-client both degrade to
    stage-1 behavior; the counter resets only on a successful write, so a transient failure
    retries at the next close instead of silently skipping a cycle. When no ``client`` is passed,
    the model is chosen by :func:`_distill_client` — the paid coder-fallback when a key exists,
    the local/default client otherwise (story #72)."""
    every = int(getattr(settings.research, "memory_distill_every", 0) or 0)
    if every <= 0:
        return False
    state_dir = settings.state_dir
    if _read_counter(state_dir) < every:
        return False
    if client is None:
        client = _distill_client(settings)
    if client is None:
        logger.info("memory distillation due but no LLM client; keeping stage-1 view")
        return False
    if distill_findings(memory, client):
        _write_counter(state_dir, 0)
        logger.info("memory distilled into MEMORY.md (every %d sessions)", every)
        return True
    return False
