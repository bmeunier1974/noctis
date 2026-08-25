"""Which symbols does this session touch — the roster it trades and the focus it researches.

Two different questions, two answers, one home. The **trading roster** is the growing universe
a session may hold a position in; the **research focus** is the capped enumeration the prompts
and the symbol-holdout pool are drawn from. They live here, in the data layer, because both are
answered by reading the lake: the config seed plus what the ``MarketData`` seam says it already
covers. Every consumer — the engine runtime, the CLI, the composition root, the research
toolbox — imports them from here, so "what does this session look at" has exactly one answer.

The lake is asked through its own :meth:`~noctis.data.seam.MarketData.coverage_records` surface,
never probed for a coverage attribute: a seam that cannot list its coverage is a broken lake and
says so, rather than having an empty roster invented on its behalf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noctis.config.settings import Settings
    from noctis.data.seam import MarketData


def trading_roster(settings: Settings, lake: MarketData) -> list[str]:
    """The growing trading universe: the config seed plus every lake-tracked ready symbol.

    The config list comes first, order preserved, so the research fit set (the first
    ``fit_set_size`` ready names) stays stable as the agent's discoveries accumulate;
    discovered symbols follow, sorted. The lake IS the persistent store — any symbol the
    research agent ever fetched via ``ensure_data`` is tracked in the coverage registry,
    so it joins the roster with no extra state. A lake tracking nothing yields the config
    list; a lake that cannot be asked at all is an error, not a short answer.

    This feeds the TRADING phase (``_load_bars``) and inventory views. It must never
    shrink under a live champion — champions trade discovered symbols. The *prompt-facing*
    enumeration is the separate, capped :func:`research_focus`.
    """
    seed = list(settings.universe)
    seen = {s.upper() for s in seed}
    extras = sorted(
        {
            rec.symbol
            for rec in lake.coverage_records()
            if rec.symbol.upper() not in seen and rec.status == "idle" and rec.row_count > 0
        }
    )
    return seed + extras


def research_focus(settings: Settings, lake: MarketData, mandate=None) -> list[str]:
    """What this session *intends* to research: fit set + symbol-holdout names +
    mandate-declared symbols, capped at ``research.focus_size``.

    Feeds the prompt-facing enumerations only (the MARKET REALITY digest and the
    symbol-holdout candidate pool) — never the trading roster. Without a cap, every
    ``ensure_data`` in every session grows every future prompt; discovered-but-unfocused
    symbols stay tradeable (roster) and re-fetchable (``preview_bars``/``list_symbols``).

    The first ``fit_set_size + symbol_holdout_size`` ready roster names come first —
    exactly the runtime's fit-set/holdout window, so the digest describes the symbols
    research actually tunes and gates on. Mandate-declared symbols follow (they may be
    unready — consumers already filter on readiness), then the cap applies.
    """
    ready = [s for s in trading_roster(settings, lake) if lake.check_symbol_ready(s)]
    cfg = settings.research
    focus = ready[: cfg.fit_set_size + cfg.symbol_holdout_size]
    seen = {s.upper() for s in focus}
    for sym in getattr(mandate, "symbols", None) or []:
        if sym.upper() not in seen:
            focus.append(sym)
            seen.add(sym.upper())
    return focus[: cfg.focus_size]
