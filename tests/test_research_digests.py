"""Byte-identity guard for the conversation loop's state/market prompt tail (epic #62 / #63).

The conversation loop renders four state facts into its system prompt — the MARKET REALITY
digest, the strategy library index (rejected entries stubbed), the champion board rows, and
the advisory memory block (findings + known dead ends). Story #63 extracts those four
builders into :mod:`noctis.research.digests` so the episodic research driver that lands later
renders the same facts *by construction* and the frozen conversation baseline cannot drift.

These tests pin the rendered tail so the extraction is provably byte-identical: an independent
reconstruction of the tail from the same live data sources must equal the tail
:func:`build_system_prompt` emits, both before and after the extraction. A second test proves
the prompt tail is assembled *from the shared builders*, not an inline copy that could drift.
"""

from __future__ import annotations

import json

from noctis.memory.consolidate import consolidate_findings, consolidate_rejected
from noctis.research import build_system_prompt, digests
from noctis.research.prompt import _MARKET_REALITY_BLOCK
from noctis.strategies import library
from noctis.strategies.library import set_header, write_strategy
from tests.test_champions import make_scorecard
from tests.test_research_tools import LENIENT, PROBE, _make_toolbox

# The prompt's own hard bound on the embedded findings block (mirrors prompt.py).
_FINDINGS_CHAR_BUDGET = 8_000
_MARKET_MARKER = "\nMARKET REALITY (do your cost arithmetic"


def _populated_toolbox(tmp_path):
    """A deterministic toolbox exercising every one of the four digests: a tiny sorted-keys
    market digest, a live + a rejected library entry, a seated champion, and a memory tail."""
    box = _make_toolbox(tmp_path)
    # A tiny, byte-stable market digest (no bar-stat noise): sorted-keys serialization.
    box.market_context = lambda: {"zeta": 1, "alpha": 2, "middle": 3}
    box.memory.append_finding("PROMOTED alpha_mom once")
    box.memory.record_rejected("dead_family", {"lookback": 3}, reason="cost-bound")
    corpse = PROBE.replace('name = "probe"', 'name = "corpse"').replace(
        "Toy probe: long above its own moving average.", "Corpse-only thesis marker."
    )
    write_strategy(box.strategies_dir, "corpse", corpse, box.families)
    set_header(box.strategies_dir, "corpse", families=box.families, status="rejected")
    box.registry.consider(
        make_scorecard("sma_crossover", test_metric=1.5, train_metric=1.6),
        LENIENT,
        mandate_source="profile:aggressive",
    )
    return box


def _tail_of(box, *, prefix_trim: bool = False) -> str:
    prompt = build_system_prompt(
        box, budget_minutes=60.0, max_iterations=10, prefix_trim=prefix_trim
    )
    return prompt[prompt.index(_MARKET_MARKER) :]


def _expected_tail(box, *, prefix_trim: bool = False) -> str:
    """Reconstruct the market + CURRENT STATE tail directly from the live data sources,
    independently of prompt.py — the golden the extraction must not perturb."""
    digest = box.market_context()
    market = _MARKET_REALITY_BLOCK.format(digest=json.dumps(digest, sort_keys=True))
    index = [
        entry
        if entry.get("status") != "rejected"
        else {"name": entry["name"], "status": entry["status"]}
        for entry in library.list_strategies(box.strategies_dir)
    ]
    champions = [
        {
            "family": e.family,
            "params": e.params,
            "test_metric": round(e.test_metric, 4),
            "sharpe": round(e.scorecard.avg_test_named("sharpe"), 4),
            "mandate_source": e.mandate_source,
            "fit_symbols": e.fit_symbols,
        }
        for e in box.registry.list()
    ]
    limit = 5 if prefix_trim else 20
    raw = box.memory.findings() if hasattr(box.memory, "findings") else []
    distilled = box.memory.distilled() if hasattr(box.memory, "distilled") else []
    if distilled:
        findings = distilled + raw[-3:]
    else:
        findings = consolidate_findings(raw, limit=limit, char_budget=_FINDINGS_CHAR_BUDGET)
    rejected = consolidate_rejected(box.memory.rejected_ideas(), limit=limit)
    state = (
        f"\nCURRENT STATE\n"
        f"Strategy library (rejected entries stubbed; list_strategies/get_strategy show any "
        f"in full): {json.dumps(index, default=str)}\n"
        f"Champion board ({box.registry.capacity} slots): {json.dumps(champions)}\n"
        f"Memory — findings: {json.dumps(findings)}\n"
        f"Memory — known dead ends (do not re-mine): {json.dumps(rejected)}\n"
    )
    return market + state


def test_prompt_state_tail_is_byte_identical_to_independent_render(tmp_path):
    """The rendered market + state tail equals an independent reconstruction from the same
    live sources — the extraction may not change a single byte of it."""
    box = _populated_toolbox(tmp_path)
    assert _tail_of(box) == _expected_tail(box)


def test_prompt_state_tail_byte_identical_under_prefix_trim(tmp_path):
    """The economy ``prefix_trim`` lever's tail is byte-identical too (advisory memory capped
    to the last 5), so the extraction preserves the cost-lever path as well as the default."""
    box = _populated_toolbox(tmp_path)
    for i in range(8):
        box.memory.append_finding(f"finding-{i}")
    assert _tail_of(box, prefix_trim=True) == _expected_tail(box, prefix_trim=True)


def test_prompt_tail_is_assembled_from_the_shared_digest_builders(tmp_path):
    """The prompt tail is rendered *from* :mod:`noctis.research.digests`, not an inline copy:
    the four shared builders' outputs, framed the documented way, reproduce the tail byte-for-
    byte — so the episodic driver that reuses them renders the same facts by construction."""
    box = _populated_toolbox(tmp_path)
    market = _MARKET_REALITY_BLOCK.format(digest=digests.market_digest(box))
    index = digests.library_index(box.strategies_dir)
    champions = digests.champion_digest(box.registry)
    findings, rejected = digests.memory_block(box.memory)
    state = (
        f"\nCURRENT STATE\n"
        f"Strategy library (rejected entries stubbed; list_strategies/get_strategy show any "
        f"in full): {json.dumps(index, default=str)}\n"
        f"Champion board ({box.registry.capacity} slots): {json.dumps(champions)}\n"
        f"Memory — findings: {json.dumps(findings)}\n"
        f"Memory — known dead ends (do not re-mine): {json.dumps(rejected)}\n"
    )
    assert _tail_of(box) == market + state
