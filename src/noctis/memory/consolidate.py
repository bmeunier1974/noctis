"""Stage-1 memory consolidation — deterministic, code-side, always on (context plan P3).

Pure functions over the ``Memory`` protocol's ``findings()`` / ``rejected_ideas()`` output:
near-identical entries collapse to one line per *lesson class* so a fixed prompt tail spans
deep history instead of the newest N raw events. Load-time views only — the underlying
``MEMORY.md`` (or fake) stays the append-only durable record, never rewritten here.

Determinism is a contract: same corpus in ⇒ byte-identical digest out (no randomness, no
timestamps), so tests and cross-session prompt caching can rely on the bytes.
"""

from __future__ import annotations

import re

# Structured finding kinds written by the research loops (tools.py / engine/research.py /
# ideation.py). Their lesson class is (kind, subject): a strategy re-rejected across sessions
# is one class-level lesson, not N events. Free text falls back to exact-normalized identity —
# digits are NOT folded, so distinct free-text notes never merge by accident.
_KIND_RE = re.compile(r"^(PROMOTED|REJECTED strategy|DEAD END|MINTED spec family)\s+(\S+)")
_DATE_PREFIX_RE = re.compile(r"^-?\s*\d{4}-\d{2}-\d{2}\s*—\s*")
_WS_RE = re.compile(r"\s+")


def _join_entries(lines: list[str]) -> list[str]:
    """Rejoin wrapped bullets: only an *indented* non-bullet line continues the entry above.

    ``MemoryStore.findings()`` returns physical file lines where hand-edited or LLM-compacted
    entries wrap as indented continuations; fakes (``InMemoryMemory``) return whole unprefixed
    notes, which must stay separate entries."""
    entries: list[str] = []
    for line in lines:
        is_continuation = line[:1].isspace() and not line.lstrip().startswith("- ")
        if is_continuation and entries:
            entries[-1] = f"{entries[-1]} {line.strip()}"
        else:
            entries.append(line.strip())
    return entries


def _class_key(entry: str) -> str:
    """The lesson-class identity of one finding entry."""
    text = _DATE_PREFIX_RE.sub("", entry.lstrip("- ").strip())
    m = _KIND_RE.match(text)
    if m:
        # Candidate keys look like ``family{json params}`` — the class is the family.
        subject = m.group(2).split("{", 1)[0]
        return f"{m.group(1)}:{subject}"
    return _WS_RE.sub(" ", text).lower()


def consolidate_findings(
    lines: list[str], *, limit: int, char_budget: int | None = None
) -> list[str]:
    """One line per lesson class, newest phrasing kept, ``(×N)`` marking merged repeats.

    Groups are ordered by the recency of their latest event and the last ``limit`` are
    returned, so a fixed tail spans as many *distinct* lessons as the corpus holds instead
    of the newest ``limit`` raw events. ``char_budget`` then drops the oldest survivors
    until the total fits (the newest line always stays) — the prompt's hard size bound.
    """
    grouped: dict[str, tuple[str, int]] = {}  # key -> (latest text, count); insertion order
    for entry in _join_entries([ln for ln in lines if ln.strip()]):
        key = _class_key(entry)
        count = grouped.pop(key, ("", 0))[1]
        grouped[key] = (entry, count + 1)  # re-insert: order = recency of latest event

    out = [text if n == 1 else f"{text} (×{n})" for text, n in grouped.values()][-limit:]
    if char_budget is not None:
        while len(out) > 1 and sum(len(ln) for ln in out) > char_budget:
            out.pop(0)
    return out


def consolidate_rejected(ideas: list[dict], *, limit: int) -> list[dict]:
    """One dead-end record per family — the guard may merge repeats but never drop the only
    record of a class. Keeps each family's most recent dict (latest params/reason) plus a
    ``times`` count when the family was rejected more than once."""
    grouped: dict[str, tuple[dict, int]] = {}
    for idea in ideas:
        family = str(idea.get("family", ""))
        count = grouped.pop(family, ({}, 0))[1]
        grouped[family] = (idea, count + 1)
    out: list[dict] = []
    for idea, n in grouped.values():
        out.append(dict(idea) if n == 1 else {**idea, "times": n})
    return out[-limit:]
