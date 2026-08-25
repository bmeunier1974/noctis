"""Usage accounting — the four token fields a completion reports, and what the engine does with
them, declared once for the whole research package (story #346).

Every backend reports its spend as the same neutral four-field split, and the split is kept apart
rather than summed because the four **bill at four rates**: a cache-heavy session priced at the
input rate would look several times more expensive than it was. Three things follow the split
around, and all three live here so a renamed field breaks in one place instead of silently
pricing part of a session at zero:

* :data:`USAGE_FIELDS` — the field list itself, in billing order.
* :func:`accumulate_usage` — folding one completion's split into a running total. Defensive by
  design: a fake client, or a provider that omits a field, contributes 0 rather than raising.
  This is the measurement floor; it must never break the loop it measures.
* :func:`estimate_tokens` — the provider-neutral ~4-chars/token size estimate, deliberately
  *independent* of any usage report, because a context budget still has to hold on a backend that
  reports nothing at all. The conversation loop sizes its history with it and the briefing
  builders size their rendered prompts with it, so there is one token accounting, not two.

The consumers: :mod:`noctis.research.agent` (the per-session rollup and its context budget),
:mod:`noctis.research.episode` (the per-episode split), :mod:`noctis.research.ledger` (which
persists it) and :mod:`noctis.research.briefings` (the build-time fit assertion).

**Two mirrors stay, by design, and are pinned equal by test rather than by an import**
(``tests/test_usage.py``):

* :mod:`noctis.research.pricing` keys its field→rate mapping by these four names — the keys are
  this list, and the values are a second fact (which rate bills which field) that belongs there.
* :mod:`noctis.reporting.run_record` spells them as the record's own field names, because the
  record writer must stay light enough never to import the research package. That weight is what
  the run-tree boundary exists to keep out, so the record buys its independence with a mirror and
  pays for it with a test.

This module is **pure**: numbers and strings in, numbers out. No config, no clock, no I/O.
"""

from __future__ import annotations

import json

# The four neutral usage fields a completion reports (``Turn.usage``), in the order they are
# billed. Cache fields read 0 on a backend that does no caching; reading every field defensively
# is what keeps a fake or no-usage client from raising.
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# The provider-neutral size heuristic: roughly four characters to a token. One ratio, because the
# loop's context budget and the briefing builders' fit assertion must agree about what fits.
APPROX_CHARS_PER_TOKEN = 4


def accumulate_usage(totals: dict[str, int], usage: dict | None) -> None:
    """Fold one completion's token ``usage`` (the neutral four-field dict on a ``Turn``) into a
    running total — a per-session rollup, or one episode's split.

    ``totals`` is mutated in place and is expected to already carry the four fields (start it at
    ``dict.fromkeys(USAGE_FIELDS, 0)``). A completion that reports nothing, a field it omits, and
    a field it reports as ``None`` all contribute 0; a field the engine does not bill is ignored.
    """
    if not usage:
        return
    for field in USAGE_FIELDS:
        totals[field] += int(usage.get(field, 0) or 0)


def estimate_tokens(base_chars: int, messages: list[dict]) -> int:
    """Provider-neutral size estimate of a request: prefix chars + serialized history, at
    ~4 chars/token. Deliberately independent of provider usage reports."""
    chars = base_chars + sum(len(json.dumps(m, default=str)) for m in messages)
    return chars // APPROX_CHARS_PER_TOKEN
