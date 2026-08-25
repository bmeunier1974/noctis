"""LLM spend, **estimated** from a versioned ``$/Mtok`` table (story #140, epic #126).

Tokens are what the engine measures; dollars are what a run is compared on. This module is the
one place that turns the first into the second, and it is deliberately the smallest possible
thing: pure arithmetic over a table of prices, with no I/O, no clock and no configuration of its
own. The record builder and the research loops all price through here, so a number that reaches an
operator can always be traced to one table entry and one table version.

Three rules make an estimate honest enough to publish:

* **Unknown means ``None``, never zero.** A model the table does not carry contributes ``None``
  and poisons every total it belongs to (:meth:`PriceTable.total_usd`), because a partial sum
  presented as a total is exactly the confidently false number this module exists to prevent. A
  ``$0``/token local backend costs nothing, but that is a *stated* price (an explicit table entry),
  not an inference from the absence of one.
* **The table says which table it is.** :data:`PRICING_TABLE_VERSION` travels into the record
  beside every number it produced, so a later reader knows which prices were in force. The label is
  ``<month>[.<revision>]``: a new month means the prices were re-surveyed, a ``.<revision>`` means
  the same month's prices with corrected **coverage** — because gaining a prefix changes the
  published number (``null`` → a real one) for the same model, and two records disagreeing about one
  run under one label is the thing the label exists to prevent. A config override cannot borrow that
  identity either: :func:`table_from_config` stamps a derived ``<version>+custom.<digest>`` label,
  which is stable for the same override and different for a different one.
* **Every field is billed at its own rate.** Input, output, cache-write and cache-read are four
  different prices at every provider, so usage that never recorded the split is *unpriceable*
  (``None``) rather than billed at a blended guess.

Prices are list prices per million tokens as published around the table version, and they are an
**estimate**: they ignore volume discounts, batch tiers and mid-month price changes. Everything
downstream names them ``*_usd_estimate`` for that reason.

Not to be confused with :mod:`noctis.research.cost`, which is about *resource ceilings* (the
``cost_profile`` knob that scales a session's Class-B budgets). That module bounds what a session
may spend in rounds, backtests and author calls; this one reports what a session did spend, in
dollars. Neither reads the other, and neither is ever a gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "BUILTIN_PRICES",
    "LOCAL_PREFIXES",
    "PRICING_TABLE_VERSION",
    "USAGE_FIELDS",
    "ModelPrice",
    "PriceTable",
    "default_table",
    "table_from_config",
]

# The shipped table's version — the label that travels into the record beside every number it
# priced. It is a ``<month>[.<revision>]`` label, and it is revised in the same edit as the change
# it describes: a new ``<month>`` says the prices below were re-surveyed, a ``.<revision>`` says the
# same month's prices with **corrected coverage** — a prefix the table should always have carried
# (story #146 added ``ollama_chat/``). A coverage change earns a label of its own because it changes
# the published output (``null`` → a real number) for the same model, and two records that disagree
# about one run under one label are exactly what this label rules out. A record written under an
# older label keeps it: it says which prices produced its numbers, which is the whole job.
PRICING_TABLE_VERSION = "2026-07.1"


@dataclass(frozen=True)
class ModelPrice:
    """One model's four rates, in US dollars per **million** tokens.

    Four rates, not one: every provider bills cached input differently from fresh input, and a
    cache-heavy session priced at the input rate would look several times more expensive than it
    was. The field names carry their unit, like every other number in the record.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_write_usd_per_mtok: float
    cache_read_usd_per_mtok: float

    def usd_for(self, usage: Mapping[str, Any] | None) -> float | None:
        """This price applied to one usage dict, or ``None`` when the usage carries no split.

        The four neutral usage fields the LLM seam reports are billed field by field. A usage
        mapping missing any of them is *unknown*, not zero: tokens without their split cannot be
        priced, and guessing the shape of the spend is the failure mode this module refuses.
        """
        if usage is None:
            return None
        total = 0.0
        for field, rate in USAGE_FIELDS.items():
            if field not in usage:
                return None
            total += _tokens(usage[field]) / 1_000_000.0 * getattr(self, rate)
        return round(total, 6)


# The neutral four-field usage dict every completion reports (``Turn.usage``), mapped onto the rate
# that bills it. The keys are :data:`noctis.research.usage.USAGE_FIELDS` and the values are this
# module's own fact — which rate bills which field — so the mapping is spelled here rather than
# built from the list. That it still *is* the list is pinned by ``tests/test_usage.py``, because a
# renamed field that slipped past this mapping would price part of a session at zero in silence.
USAGE_FIELDS: Mapping[str, str] = {
    "input_tokens": "input_usd_per_mtok",
    "output_tokens": "output_usd_per_mtok",
    "cache_creation_input_tokens": "cache_write_usd_per_mtok",
    "cache_read_input_tokens": "cache_read_usd_per_mtok",
}


# The shipped table, keyed by **model prefix** so a point release of a known family prices itself
# (``anthropic/claude-opus-4-8`` matches ``anthropic/claude-opus-4``) and the longest matching
# prefix wins. Both the bare and the ``provider/`` spellings are carried, because the engine
# accepts both (``research.agent.model`` defaults to a bare ``claude-opus-4-8``).
#
# The local/self-hosted prefixes are priced at an explicit zero: a ``$0``/token backend really does
# cost nothing, and saying so as a table entry keeps "free" distinguishable from "unknown".
_ANTHROPIC = {
    "claude-opus-4": ModelPrice(15.0, 75.0, 18.75, 1.5),
    "claude-sonnet-4": ModelPrice(3.0, 15.0, 3.75, 0.3),
    "claude-sonnet-5": ModelPrice(3.0, 15.0, 3.75, 0.3),
    "claude-haiku-4": ModelPrice(1.0, 5.0, 1.25, 0.1),
}
_OPENAI = {
    "gpt-5-mini": ModelPrice(0.25, 2.0, 0.25, 0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.4, 0.05, 0.005),
    "gpt-5": ModelPrice(1.25, 10.0, 1.25, 0.125),
    "gpt-4o-mini": ModelPrice(0.15, 0.6, 0.15, 0.075),
    "gpt-4o": ModelPrice(2.5, 10.0, 2.5, 1.25),
    "o3-mini": ModelPrice(1.1, 4.4, 1.1, 0.55),
    "o3": ModelPrice(2.0, 8.0, 2.0, 0.5),
}
_FREE_LOCAL = ModelPrice(0.0, 0.0, 0.0, 0.0)

# Every prefix that names a **local / self-hosted** backend, stated once so a future driver is
# priced by adding one line here rather than one line here and a forgotten one there.
# ``ollama_chat/`` and ``ollama/`` are the same free Ollama server — the suffix only selects
# LiteLLM's chat-completions driver — and ``ollama_chat/`` is the spelling the system itself writes
# (``noctis setup``) and recommends (the ``--model`` help, the example config, the docs), so it is
# the one an operator who took every default is actually running.
#
# This list is deliberately an **allowlist**, and deliberately *not* shared with
# :func:`noctis.research.cost.is_free_local`, which defines free-local negatively (anything that is
# not ``anthropic`` or ``openai``). That definition is right where it lives — being wrong about a
# *resource ceiling* costs a budget — and wrong here: it would price any future paid third-party
# cloud (``gemini/``, ``mistralai/``, a paid OpenAI-compatible gateway) at a confident ``$0``, which
# is inferring a price from the absence of one, the first rule of this module. Being wrong here
# publishes a false dollar figure, so billing stays deny-by-default: a prefix earns its zero by
# being named on this list.
LOCAL_PREFIXES: tuple[str, ...] = (
    "ollama/",
    "ollama_chat/",
    "vllm/",
    "lm_studio/",
    "local/",
)

BUILTIN_PRICES: Mapping[str, ModelPrice] = {
    **{name: price for name, price in _ANTHROPIC.items()},
    **{f"anthropic/{name}": price for name, price in _ANTHROPIC.items()},
    **{name: price for name, price in _OPENAI.items()},
    **{f"openai/{name}": price for name, price in _OPENAI.items()},
    # Local / self-hosted backends: a stated zero, not an absent price.
    **dict.fromkeys(LOCAL_PREFIXES, _FREE_LOCAL),
}


@dataclass(frozen=True)
class PriceTable:
    """A versioned set of per-prefix prices, and the arithmetic over it.

    ``version`` is carried into the record beside every number this table produced — the shipped
    label for the built-in table, a derived ``+custom.<digest>`` one for an overridden table.
    """

    version: str
    prices: Mapping[str, ModelPrice]

    def price_for(self, model: str) -> ModelPrice | None:
        """The price for one model, matched on the **longest** prefix, or ``None`` if unknown.

        Longest-wins so a specific entry is never shadowed by the family it belongs to. Matching is
        case-insensitive because the model string is operator input.
        """
        name = (model or "").strip().lower()
        if not name:
            return None
        matches = [prefix for prefix in self.prices if name.startswith(prefix.lower())]
        if not matches:
            return None
        return self.prices[max(matches, key=len)]

    def estimate_usd(self, model: str, usage: Mapping[str, Any] | None) -> float | None:
        """What one model's usage cost, or ``None`` when the model or the split is unknown."""
        price = self.price_for(model)
        return None if price is None else price.usd_for(usage)

    def total_usd(self, items: Iterable[tuple[str, Mapping[str, Any] | None]]) -> float | None:
        """The bill for many ``(model, usage)`` pairs — ``None`` if **any** of them is unpriceable.

        Deliberately all-or-nothing: a total that quietly dropped the items it could not price
        would read as a complete bill while understating it, which is worse than admitting the
        table does not know. The per-model breakdown beside it still shows what *is* known.
        """
        total = 0.0
        for model, usage in items:
            one = self.estimate_usd(model, usage)
            if one is None:
                return None
            total += one
        return round(total, 6)


def default_table() -> PriceTable:
    """The shipped table under :data:`PRICING_TABLE_VERSION`."""
    return PriceTable(version=PRICING_TABLE_VERSION, prices=dict(BUILTIN_PRICES))


def table_from_config(overrides: Mapping[str, Mapping[str, Any]] | None) -> PriceTable:
    """The shipped table with an operator's ``research.pricing`` entries layered over it.

    An override adds a prefix the table never carried or replaces one it did (all four rates at
    once — a half-stated price is not a price). The result **never reuses the shipped version
    label**: it identifies itself as ``<version>+custom.<digest>``, derived from the override
    itself, so it is stable across writes of the same configuration and different for any other
    one. A reader of a record can therefore always tell whether the numbers came from the table
    this engine ships.
    """
    if not overrides:
        return default_table()
    prices = dict(BUILTIN_PRICES)
    for prefix, rates in overrides.items():
        prices[prefix] = _price_from(rates)
    return PriceTable(version=f"{PRICING_TABLE_VERSION}+custom.{_digest(overrides)}", prices=prices)


def _price_from(rates: Mapping[str, Any]) -> ModelPrice:
    """One override entry as a :class:`ModelPrice` (it carries all four rates by construction)."""
    if isinstance(rates, ModelPrice):
        return rates
    return ModelPrice(
        input_usd_per_mtok=float(rates["input_usd_per_mtok"]),
        output_usd_per_mtok=float(rates["output_usd_per_mtok"]),
        cache_write_usd_per_mtok=float(rates["cache_write_usd_per_mtok"]),
        cache_read_usd_per_mtok=float(rates["cache_read_usd_per_mtok"]),
    )


def _digest(overrides: Mapping[str, Mapping[str, Any]]) -> str:
    """A short, stable fingerprint of an override set — the custom table's identity."""
    payload = {prefix: asdict(_price_from(rates)) for prefix, rates in sorted(overrides.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def _tokens(value: Any) -> int:
    """A tolerant token count: anything that is not a plain number contributes nothing."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)
