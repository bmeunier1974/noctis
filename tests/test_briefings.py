"""Build-time fit assertion for the v1 episodic briefing builders (epic #62 / story #67).

The formulate and decide briefings are rebuilt fresh from disk on every call out of the shared
digest builders plus the session-ledger tail, and each asserts the rendered prompt fits the
configured context window — trimming only advisory blocks in a fixed priority order (memory tail
→ library stubs → digest breadth), never the gate-facing numbers, and failing loudly when even a
fully-trimmed briefing still does not fit. These tests lock:

* statelessness — two calls straddling a disk change reflect the change and share no state;
* the fit assertion at an 8k window, with the trim order exercised block-by-block;
* gate-facing numbers surviving every trim level (silent truncation is structurally impossible);
* a loud failure when the un-trimmable core alone overflows the window;
* the ~1.5-3k token target band on realistic fixture state;
* which store each briefing reads (epic #326 / story #330) — the session ledger owns the
  session narrative the ALREADY-TRIED tail renders, the experiment journal owns the durable
  per-strategy evidence a DECIDE verdict is earned against — so each test primes ONLY the
  store its briefing reads, and two pins state that boundary from the reading side.
"""

from __future__ import annotations

import pytest

from noctis.champions import PromotionRules, decide
from noctis.data.preflight import CostPreflight
from noctis.research import Mandate
from noctis.research.briefings import (
    BriefingTooLargeError,
    decide_briefing,
    discover_briefing,
    formulate_briefing,
)
from noctis.research.ledger import SessionLedger
from noctis.research.surface import ChampionBoard, ResearchFacts, ResearchLimits
from noctis.research.usage import estimate_tokens
from noctis.strategies.library import set_header, write_strategy
from tests.test_champions import make_scorecard
from tests.test_research_tools import LENIENT, PROBE, _make_toolbox

# Sentinels planted in each advisory block so trim decisions are observable in the rendered text.
_MEMORY_SENTINEL = "MEMSENTINEL-finding-do-not-repeat"
_LIBRARY_SENTINEL = "libsentinel"
_BREADTH_KEY = "trend_efficiency"  # a per-symbol character field, only in the digest-breadth block
_LEDGER_SENTINEL = "LEDGERSENTINEL"
# The ownership pins' sentinels (epic #326): a thesis text written to exactly ONE store, so a
# block that renders it is naming the store it read.
_JOURNAL_ONLY_THESIS = "JOURNALONLYSENTINEL journaled for the candidate, never told the ledger"
_LEDGER_ONLY_THESIS = "LEDGERONLYSENTINEL told to the ledger, never journaled"
_EXHAUSTED_LABEL = "minute rsi mean reversion"

# The one-slot-per-family steering (story #163): the rule itself and the crowned names it
# names, rendered beside the board so no session spends trials re-tuning a family that can
# never land. Both are gate-facing — they say what promotion will refuse.
_CROWNED_RULE = "one slot per family"
_CROWNED_NAMES = '"crowned_families": ["alpha_mom", "gamma_break"]'

# The vocabulary the family_slot rejection message speaks; the steering must not invent its own.
_FAMILY_SLOT_PHRASES = (
    "already holds a champion slot",
    "a champion file is immutable",
    "author an improvement under a new name",
    "one slot per family",
)

# Gate-facing markers that must survive every trim level of the FORMULATE briefing.
_FORMULATE_GATE_MARKERS = (
    "round_trip_cost_bp",  # market cost arithmetic
    _EXHAUSTED_LABEL,  # exhausted-class hygiene guard
    "test_metric",  # champion board (beat-the-weakest bar)
    _CROWNED_RULE,  # the one-slot-per-family rule
    _CROWNED_NAMES,  # the families that already hold a slot
    _LEDGER_SENTINEL,  # the session narrative — never dropped
)

_HUGE = 10_000_000


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """These tests exercise briefing assembly, not subprocess write-gate isolation."""


def _tokens(text: str) -> int:
    return estimate_tokens(len(text), [])


def _bloat_memory(box) -> None:
    """Push the advisory memory tail well past an 8k window (distinct dead-end families with
    long reasons — the consolidated rejected view keeps the latest 20, uncapped in chars)."""
    for i in range(20):
        box.memory.record_rejected(f"bloatfam{i}", {"lookback": i}, reason="y" * 1600)


def _named(source_name: str, new_name: str, marker: str) -> str:
    return PROBE.replace('name = "probe"', f'name = "{new_name}"').replace(
        "Toy probe: long above its own moving average.", marker
    )


def _session(tmp_path):
    """One session's shape with NEITHER research store primed: a populated market digest, several
    champions, a memory file with findings and dead ends, a handful of library strategies (one
    rejected) and an exhausted class — beside an empty experiment journal and an empty session
    ledger.

    The two stores are primed separately, by :func:`_prime_journal` and :func:`_prime_ledger`, so
    each test states which store its briefing actually reads (epic #326)."""
    box = _make_toolbox(tmp_path)  # universe AAA..DDD with bars; ships a 'probe' strategy

    # Library: two live strategies + one rejected corpse (collapsed to a stub by the index).
    for name in ("alpha_mom", _LIBRARY_SENTINEL):
        write_strategy(
            box.strategies_dir, name, _named("probe", name, f"{name} thesis marker."), box.families
        )
    write_strategy(
        box.strategies_dir, "corpse", _named("probe", "corpse", "corpse thesis."), box.families
    )
    set_header(box.strategies_dir, "corpse", families=box.families, status="rejected")

    # Champions (the beat-the-weakest bar).
    box.registry.consider(
        make_scorecard("alpha_mom", test_metric=1.5, train_metric=1.6),
        LENIENT,
        mandate_source="profile:aggressive",
    )
    box.registry.consider(
        make_scorecard("gamma_break", test_metric=1.2, train_metric=1.3),
        LENIENT,
        mandate_source="profile:balanced",
    )

    # Memory: advisory findings + a rejected dead end.
    box.memory.append_finding(f"PROMOTED alpha_mom once — {_MEMORY_SENTINEL}")
    box.memory.append_finding("DEAD END minute RSI mean reversion nets negative after 4bp round")
    box.memory.record_rejected("rsi_scalp", {"lookback": 3}, reason="gross edge below cost")

    # A cross-session exhausted class (research-hygiene guard).
    box.exhausted.record(
        _EXHAUSTED_LABEL,
        "gross edge/trade below the 4bp round trip on every symbol tried",
        example="corpse",
    )

    mandate = Mandate(
        text="Pursue liquid-name momentum that clears cost at 1h.",
        source="profile:aggressive",
        summary="aggressive: liquid-name momentum, tune on Sharpe",
        references=[],
        config_overrides={},
        symbols=["AAA", "BBB"],
    )
    return box, SessionLedger(box.state_dir, session_id="sess-1"), mandate


def _prime_journal(box):
    """The gate-facing evidence the DECIDE briefing reads for ``probe``: its thesis, its class tag
    and its ranked trials, all in the experiment journal.

    None of it is written to the session ledger — the journal owns the durable per-strategy facts,
    so this is what a decide-evidence test primes, and all it primes."""
    box.journal.record_thesis("probe", "Long above own moving average while the trend is up.")
    box.journal.record_class_tag("probe", "intraday momentum")
    box.journal.record_trial(
        "probe",
        source="backtest",
        symbols=["AAA", "BBB"],
        params={"lookback": 12, "edge": 1.0},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=1.41, train_metric=1.55, lookback=12),
    )
    box.journal.record_trial(
        "probe",
        source="sweep",
        symbols=["AAA"],
        params={"lookback": 20, "edge": 1.1},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=0.92, train_metric=1.03, lookback=20),
    )
    box.journal.record_trial(
        "probe",
        source="sweep",
        symbols=["BBB"],
        params={"lookback": 30, "edge": 0.9},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=0.55, train_metric=0.80, lookback=30),
    )
    return box


def _prime_ledger(ledger):
    """The session narrative the ALREADY-TRIED tail reads: the theses this session spent and the
    verdict one of them earned, all in the session ledger.

    None of it is journaled — the ledger owns the narrative, so this is what a formulate-tail (or
    discover) test primes, and all it primes."""
    ledger.record_session_start(mandate="profile:aggressive", budgets={}, models={})
    ledger.record_thesis("probe", f"{_LEDGER_SENTINEL} momentum long above own MA at 1h")
    ledger.record_thesis("corpse", "Minute RSI mean reversion buys oversold dips.")
    ledger.record_verdict(
        "corpse",
        verdict="reject",
        lesson="minute RSI mean reversion nets negative after the 4bp round trip",
        promoted=False,
    )
    return ledger


def _populate(tmp_path):
    """A realistic mid-session state with BOTH stores primed — the whole-session fixture the
    cross-module readers (the prompt goldens, the digest, surface and episodic-site suites) render
    off. The briefing tests below compose the pieces instead, one store at a time."""
    box, ledger, mandate = _session(tmp_path)
    _prime_journal(box)
    _prime_ledger(ledger)
    return box, ledger, mandate


# Reading ONE labeled block is how a test says which store it means: the ledger's narrative tail
# and the journal's evidence are different sections of the same rendered briefing.
_TAIL_LABEL = "ALREADY TRIED THIS SESSION"
_EVIDENCE_LABEL = "EVIDENCE FOR probe (gate-facing)"


def _block(brief: str, label: str) -> str:
    """The body rendered under ``label`` — every body a test reads here is single-line JSON, so
    the block ends at the blank line before the next label."""
    start = brief.index(f"{label}:\n") + len(label) + 2
    end = brief.find("\n\n", start)
    return brief[start:] if end < 0 else brief[start:end]


# ── statelessness: rebuilt fresh from disk, no state carried between calls ──────────────────
def test_formulate_briefing_is_deterministic_and_rebuilt_fresh_from_disk(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)  # a formulate tail reads the session ledger, and only that

    before = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)
    # Same inputs, same bytes — no hidden per-call state.
    assert before == formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)

    # Mutate the disk sources the two shared inputs read from (the ledger JSONL + the library).
    ledger.record_thesis("newidea", "NEWDISKSENTINEL breakout on opening gaps")
    write_strategy(
        box.strategies_dir,
        "newlib",
        _named("probe", "newlib", "newlib thesis marker."),
        box.families,
    )
    after = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)

    assert "NEWDISKSENTINEL" not in before and "NEWDISKSENTINEL" in after
    assert "newlib" not in before and "newlib" in after


def test_decide_briefing_rebuilt_fresh_from_disk(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_journal(box)  # a decide evidence block reads the journal, and only that
    before = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)

    box.journal.record_trial(
        "probe",
        source="sweep",
        symbols=["CCC"],
        params={"lookback": 44, "edge": 1.3},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=1.77, train_metric=1.80, lookback=44),
    )
    after = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)

    assert '"lookback": 44' not in before and '"lookback": 44' in after


# ── the trim order, exercised advisory-block by advisory-block ──────────────────────────────
def test_formulate_trim_order_drops_memory_then_library_then_breadth(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)  # a formulate tail reads the session ledger, and only that

    full = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)
    assert _MEMORY_SENTINEL in full and _LIBRARY_SENTINEL in full and _BREADTH_KEY in full

    # A window one token below each successive render forces exactly the next advisory drop.
    b1 = formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(full) - 1)
    assert _MEMORY_SENTINEL not in b1  # memory tail dropped first
    assert _LIBRARY_SENTINEL in b1 and _BREADTH_KEY in b1
    for marker in _FORMULATE_GATE_MARKERS:
        assert marker in b1

    b2 = formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(b1) - 1)
    assert _MEMORY_SENTINEL not in b2 and _LIBRARY_SENTINEL not in b2  # library stubs dropped next
    assert _BREADTH_KEY in b2
    for marker in _FORMULATE_GATE_MARKERS:
        assert marker in b2

    b3 = formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(b2) - 1)
    assert _BREADTH_KEY not in b3  # digest breadth dropped last
    for marker in _FORMULATE_GATE_MARKERS:
        assert marker in b3

    # Every advisory block already trimmed and the un-trimmable core still overflows → loud fail.
    with pytest.raises(BriefingTooLargeError):
        formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(b3) - 1)


def test_formulate_fit_assertion_at_8k_window_trims_and_keeps_gate_numbers(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)  # a formulate tail reads the session ledger, and only that
    _bloat_memory(box)  # push the advisory memory tail past an 8k window
    assert _tokens(formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)) > 8000

    fitted = formulate_briefing(box, ledger, mandate=mandate, context_window=8000)
    assert _tokens(fitted) <= 8000
    assert _MEMORY_SENTINEL not in fitted  # advisory memory trimmed to fit
    for marker in _FORMULATE_GATE_MARKERS:  # gate-facing numbers never trimmed
        assert marker in fitted


def test_formulate_raises_loudly_when_core_exceeds_window(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)  # a formulate tail reads the session ledger, and only that
    # A window smaller than the un-trimmable core is a loud failure, never a silent truncation.
    with pytest.raises(BriefingTooLargeError):
        formulate_briefing(box, ledger, mandate=mandate, context_window=10)


# ── decide briefing: gate-facing candidate evidence, never trimmed ─────────────────────────
def test_decide_briefing_carries_ranked_journal_evidence(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_journal(box)  # a decide evidence block reads the journal, and only that
    # A verdict already spent on this candidate is a durable per-strategy fact, so it is
    # journaled: the session ledger's verdicts are narrative and render in the tail, not here.
    box.journal.record_rejection(
        "probe",
        reason="an earlier cut of this space never cleared the round trip",
        best_params={"lookback": 30},
    )
    evidence = _block(
        decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE),
        _EVIDENCE_LABEL,
    )

    assert "min_trials_gate" in evidence  # the exhaustion floor the verdict is judged against
    assert '"n_distinct_params": 3' in evidence
    assert "top_trials" in evidence and '"lookback": 12' in evidence  # ranked trials + params
    assert '"verdict": "reject"' in evidence  # journaled verdicts surfaced


def test_decide_gate_numbers_survive_trim_at_8k(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_journal(box)  # a decide evidence block reads the journal, and only that
    _bloat_memory(box)

    fitted = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=8000)
    assert _tokens(fitted) <= 8000
    assert _MEMORY_SENTINEL not in fitted  # advisory memory trimmed
    # The candidate's gate-facing evidence survives the trim.
    assert "min_trials_gate" in fitted
    assert '"n_distinct_params": 3' in fitted
    assert '"lookback": 12' in fitted


# ── one owner per fact: the ledger owns the narrative, the journal owns the evidence (#330) ──
def test_a_journal_only_thesis_never_reaches_the_formulate_tail(tmp_path):
    """ALREADY TRIED THIS SESSION is the *session ledger's* story. A thesis only the experiment
    journal knows about is not part of it — the tail stays empty and says so — and the very same
    text does reach the tail once the ledger is what knows it."""
    box, ledger, mandate = _session(tmp_path)
    box.journal.record_thesis("probe", _JOURNAL_ONLY_THESIS)

    brief = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)
    assert _block(brief, _TAIL_LABEL) == "[]"
    assert _JOURNAL_ONLY_THESIS not in brief

    ledger.record_thesis("probe", _JOURNAL_ONLY_THESIS)
    tail = _block(
        formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE), _TAIL_LABEL
    )
    assert _JOURNAL_ONLY_THESIS in tail


def test_a_ledger_only_thesis_never_reaches_the_decide_evidence(tmp_path):
    """EVIDENCE FOR <name> is the *experiment journal's* case for one candidate. A thesis only the
    session ledger knows about is narrative: the evidence answers ``null`` for it (while the
    narrative still reaches the briefing through its own tail), and the block carries a thesis
    only once the journal is what knows it."""
    box, ledger, mandate = _session(tmp_path)
    ledger.record_thesis("probe", _LEDGER_ONLY_THESIS)

    brief = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)
    evidence = _block(brief, _EVIDENCE_LABEL)
    assert _LEDGER_ONLY_THESIS not in evidence
    assert '"thesis": null' in evidence  # the journal knows of no thesis for probe
    assert _LEDGER_ONLY_THESIS in _block(brief, _TAIL_LABEL)  # ...the narrative tail still has it

    box.journal.record_thesis("probe", _JOURNAL_ONLY_THESIS)
    after = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)
    assert _JOURNAL_ONLY_THESIS in _block(after, _EVIDENCE_LABEL)


# ── one slot per family: the board steers, it does not merely score (story #163) ────────────
def _family_slot_rationale() -> str:
    """The live ``family_slot`` rejection message — the one dialect the steering must speak."""
    card = make_scorecard("crowned_fam", test_metric=1.0, train_metric=1.1)
    rules = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)
    return decide(card, [card], rules).rationale


def test_briefings_name_the_crowned_families_and_state_the_one_slot_rule(tmp_path):
    # The steering comes off the champion board — neither research store primed, because
    # neither is what says a family already holds a slot.
    box, ledger, mandate = _session(tmp_path)
    for brief in (
        formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE),
        decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE),
    ):
        assert _CROWNED_NAMES in brief  # exactly which families are off the table
        assert _CROWNED_RULE in brief  # ...and why re-tuning one cannot land
        assert "full funnel" in brief  # the honest path: a new name through the whole funnel


def test_one_slot_steering_speaks_the_family_slot_gate_dialect(tmp_path):
    box, ledger, mandate = _session(tmp_path)  # board steering again: neither store
    brief = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)
    rationale = _family_slot_rationale()
    for phrase in _FAMILY_SLOT_PHRASES:
        assert phrase in rationale, f"the gate no longer says {phrase!r}"
        assert phrase in brief, f"the briefing invented a dialect for {phrase!r}"


# ── the DISCOVER briefing: the no-lake-match symbol ask (story #112) ────────────────────────
# Same machinery as the other two — core (gate-facing / decision-facing) sections plus the shared
# advisory blocks in the fixed trim order — so a small-context backend can answer the one question
# it is asked: which real tickers express the character the lake could not match.
_PROFILE = {"trend": "low", "volatility": "high", "liquidity": "low"}
_DISCOVER_WINDOW = {"start": "2026-02-08", "end": "2026-03-09", "history_days": 30}
_DISCOVER_THESIS = "DISCOVERTHESIS fade panic in thin names once volatility clears cost"
_DISCOVER_CHARACTER = "illiquid volatile small-caps that mean-revert"

# What the discover episode must be able to see no matter how hard the briefing is trimmed.
_DISCOVER_CORE_MARKERS = (
    _DISCOVER_THESIS,  # the thesis the symbols are for
    _DISCOVER_CHARACTER,  # the requested symbol character
    '"volatility": "high"',  # the exact band profile that found no lake match
    "CCC",  # the lake inventory — names the validator would drop
    "2026-03-09",  # the fetch window the spend would cover
    _LEDGER_SENTINEL,  # the session narrative
)


def _discover(box, ledger, mandate, *, context_window):
    return discover_briefing(
        box,
        ledger,
        thesis=_DISCOVER_THESIS,
        symbol_character=_DISCOVER_CHARACTER,
        profile=_PROFILE,
        window=_DISCOVER_WINDOW,
        mandate=mandate,
        context_window=context_window,
    )


def test_discover_briefing_carries_the_mandate_thesis_profile_inventory_and_budget(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)  # discover tails the same session narrative formulate does
    brief = _discover(box, ledger, mandate, context_window=_HUGE)

    assert "aggressive: liquid-name momentum" in brief  # the mandate body...
    assert "declared symbols: AAA, BBB" in brief  # ...declared symbols included
    for marker in _DISCOVER_CORE_MARKERS:
        assert marker in brief
    assert "1-6" in brief and "rationale" in brief  # the emit contract it must answer with
    assert brief == _discover(box, ledger, mandate, context_window=_HUGE)  # no per-call state


def test_discover_briefing_surfaces_the_configured_data_budget_when_the_seam_exposes_it(tmp_path):
    # The spend context states the budget the toolbox answers with: present ⇒ the number a
    # refusal would be judged against; absent (a fake lake with no cost preflight) ⇒ omitted,
    # never invented. The preflight is the real one a lake with a vendor is built with, so the
    # briefing is shown the same object production shows it.
    box, ledger, mandate = _session(tmp_path)  # the spend context reads neither store
    assert "budget_usd" not in _discover(box, ledger, mandate, context_window=_HUGE)

    box.lake.preflight = CostPreflight(125.0)
    assert "125.0" in _discover(box, ledger, mandate, context_window=_HUGE)


def test_discover_trim_order_drops_advisory_blocks_and_keeps_the_ask_intact(tmp_path):
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)  # the narrative is core here too — it survives every trim

    full = _discover(box, ledger, mandate, context_window=_HUGE)
    assert _MEMORY_SENTINEL in full and _BREADTH_KEY in full

    b1 = _discover(box, ledger, mandate, context_window=_tokens(full) - 1)
    assert _MEMORY_SENTINEL not in b1  # the advisory memory tail goes first
    for marker in _DISCOVER_CORE_MARKERS:
        assert marker in b1

    b2 = _discover(box, ledger, mandate, context_window=_tokens(b1) - 1)
    assert _BREADTH_KEY not in b2  # then the per-symbol digest breadth
    for marker in _DISCOVER_CORE_MARKERS:
        assert marker in b2

    # The un-trimmable ask still overflowing is a loud failure, never a silent truncation.
    with pytest.raises(BriefingTooLargeError):
        _discover(box, ledger, mandate, context_window=_tokens(b2) - 1)


def test_discover_briefing_fits_a_small_window_on_realistic_state(tmp_path):
    # The whole point of the episode: a small-context backend must be able to answer it.
    box, ledger, mandate = _session(tmp_path)
    _prime_ledger(ledger)
    _bloat_memory(box)
    fitted = _discover(box, ledger, mandate, context_window=8000)
    assert _tokens(fitted) <= 8000
    for marker in _DISCOVER_CORE_MARKERS:
        assert marker in fitted


# ── the ~1.5-3k token target band on realistic (un-bloated) state ───────────────────────────
def test_briefings_land_in_target_token_band_on_realistic_state(tmp_path):
    # The one test that primes BOTH stores, because the size of a real mid-session briefing
    # is the sum of two reads: the ledger's ALREADY-TRIED tail (in all three briefings) and
    # the journal's evidence block (in decide). Either alone understates the band.
    box, ledger, mandate = _populate(tmp_path)
    formulate = _tokens(formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE))
    decide = _tokens(decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE))
    discover = _tokens(_discover(box, ledger, mandate, context_window=_HUGE))
    assert 800 < formulate < 3500, formulate
    assert 800 < decide < 3500, decide
    assert 500 < discover < 3500, discover  # the smallest of the three asks


# ── the seam: a briefing reads the facts, never the collaborators behind them (#258) ────────
# What a renderer is handed is a ResearchFacts (noctis.research.surface): the derived facts, and
# nothing that could change the session. The stand-in below implements exactly that Protocol and
# owns no journal, registry, memory, lake, library path or scalar ceiling — so a builder that
# still reaches THROUGH the toolbox into a collaborator fails here by attribute error, rather
# than by quietly rendering a session other than the one it was handed.
_FACTS_MARKET = "FACTSMARKET-round-trip"
_FACTS_EXHAUSTED = "factsclass exhausted here"
_FACTS_BREADTH = "FACTSBREADTH"
_FACTS_CROWNED = "facts_fam"
_FACTS_LIBRARY = "facts_probe"
_FACTS_FINDING = "FACTSFINDING kept advisory"
_FACTS_DEAD_END = "FACTSDEADEND do not re-mine"
_FACTS_INVENTORY = "FACTSHELD"
_FACTS_EVIDENCE = "FACTSEVIDENCE the journaled case for it"
_FACTS_BUDGET = 42.5


class _FactsOnly:
    """One session's derived facts, and nothing else — the whole surface a renderer may read."""

    limits = ResearchLimits(min_trials=4, max_backtests=9, sweep_trials=3, max_author_calls=2)

    def market_context(self) -> dict:
        return {
            "round_trip_cost_bp": _FACTS_MARKET,
            "symbols": {"ZZZ": {"character": _FACTS_BREADTH}},
            "exhausted_classes": [_FACTS_EXHAUSTED],
        }

    def journal_evidence(self, name: str) -> dict:
        return {"strategy": name, "thesis": _FACTS_EVIDENCE, "min_trials_gate": 4}

    def champion_board(self) -> ChampionBoard:
        return ChampionBoard(
            rows=({"family": _FACTS_CROWNED, "test_metric": 1.25},),
            crowned_families=(_FACTS_CROWNED,),
            capacity=3,
        )

    def library_index(self) -> list[dict]:
        return [{"name": _FACTS_LIBRARY, "status": "draft"}]

    def template_text(self) -> str:
        return "# the shipped template, stated by the facts"

    def memory_tail(self, *, prefix_trim: bool = False) -> tuple[list, list]:
        return [_FACTS_FINDING], [_FACTS_DEAD_END]

    def lake_inventory(self, *, limit: int = 60) -> list[str]:
        return [_FACTS_INVENTORY]

    def data_budget(self) -> float | None:
        return _FACTS_BUDGET


def _facts_ledger(tmp_path) -> SessionLedger:
    ledger = SessionLedger(tmp_path / "state", session_id="facts-1")
    ledger.record_thesis("facts_probe", f"{_LEDGER_SENTINEL} the session narrative")
    return ledger


def test_the_facts_stand_in_holds_no_collaborator_a_briefing_could_reach_through():
    facts = _FactsOnly()

    assert isinstance(facts, ResearchFacts)
    for collaborator in ("journal", "registry", "memory", "lake", "strategies_dir", "min_trials"):
        assert not hasattr(facts, collaborator), collaborator


def test_formulate_renders_every_block_from_the_facts_alone(tmp_path):
    brief = formulate_briefing(_FactsOnly(), _facts_ledger(tmp_path), context_window=_HUGE)

    for marker in (
        _FACTS_MARKET,
        _FACTS_BREADTH,
        _FACTS_EXHAUSTED,
        _FACTS_CROWNED,
        _FACTS_LIBRARY,
        _FACTS_FINDING,
        _FACTS_DEAD_END,
        _LEDGER_SENTINEL,
    ):
        assert marker in brief


def test_decide_renders_the_journaled_evidence_the_facts_state(tmp_path):
    brief = decide_briefing(
        _FactsOnly(), _facts_ledger(tmp_path), "facts_probe", context_window=_HUGE
    )

    assert "EVIDENCE FOR facts_probe (gate-facing)" in brief
    assert _FACTS_EVIDENCE in brief
    assert '"min_trials_gate": 4' in brief
    for marker in (_FACTS_MARKET, _FACTS_CROWNED, _FACTS_LIBRARY, _LEDGER_SENTINEL):
        assert marker in brief


def test_discover_renders_the_inventory_and_the_budget_the_facts_state(tmp_path):
    brief = discover_briefing(
        _FactsOnly(),
        _facts_ledger(tmp_path),
        thesis=_DISCOVER_THESIS,
        symbol_character=_DISCOVER_CHARACTER,
        profile=_PROFILE,
        window=_DISCOVER_WINDOW,
        context_window=_HUGE,
    )

    assert _FACTS_INVENTORY in brief
    assert f'"budget_usd": {_FACTS_BUDGET}' in brief
    assert _DISCOVER_THESIS in brief and "2026-03-09" in brief
