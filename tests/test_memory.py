"""The MEMORY.md store: template, append, dedup, size budget, champion refresh — plus the
stage-1 consolidation views and the machine-owned distilled section (context plan P3)."""

from __future__ import annotations

from noctis.backtest.scorecard import Metrics, Scorecard, SplitScore, SymbolScore
from noctis.champions import ChampionRegistry, PromotionRules
from noctis.memory import MemoryStore
from noctis.memory.consolidate import consolidate_findings, consolidate_rejected

RULES = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)


def _sc(family, test, train, **params):
    def m(x):
        return Metrics(x, x, 0.0, 0.0, 0.0, 0.0, 0.0)

    return Scorecard(
        family=family,
        params=params,
        stage="validated",
        symbols={"FIT": SymbolScore(splits=[SplitScore(0, m(train), m(test))])},
    )


def test_template_created_when_missing(tmp_path):
    path = tmp_path / "MEMORY.md"
    assert not path.exists()
    store = MemoryStore(path)
    assert path.exists()
    text = store.read()
    for header in ("## Champions", "## Learnings", "## Rejected ideas", "## Index / changelog"):
        assert header in text


def test_append_finding_dates_changelog(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.append_finding("promoted a donchian breakout")
    text = store.read()
    assert "promoted a donchian breakout" in text
    # Dated (ISO date appears in the changelog line).
    import re

    assert re.search(r"- \d{4}-\d{2}-\d{2} — promoted a donchian breakout", text)


def test_rejected_ideas_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.record_rejected("sma_crossover", {"fast": 5, "slow": 20}, reason="overfit")
    ideas = store.rejected_ideas()
    assert ideas == [{"family": "sma_crossover", "params": {"fast": 5, "slow": 20}}]
    # Survives a reload from disk.
    reloaded = MemoryStore(tmp_path / "MEMORY.md")
    assert reloaded.rejected_ideas() == ideas


def test_reorganize_dedups_learnings_and_keeps_sections(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.add_learning("momentum works in the morning")
    store.add_learning("momentum works in the morning")  # duplicate
    store.add_learning("mean reversion fails in trends")
    store.reorganize()
    learnings = store.sections["Learnings"]
    assert learnings.count("- momentum works in the morning") == 1
    assert any("mean reversion fails" in ln for ln in learnings)
    text = store.read()
    for header in ("## Champions", "## Learnings", "## Rejected ideas", "## Index / changelog"):
        assert header in text


def test_reorganize_refreshes_champions_from_registry(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    # Seed a stale champion note that should be replaced.
    store.sections["Champions"] = ["- STALE champion that no longer holds"]
    registry = ChampionRegistry(tmp_path / "champs.json", capacity=3)
    registry.consider(_sc("donchian_breakout", 1.5, 1.6, channel=20), RULES)

    store.reorganize(registry)
    champs_text = "\n".join(store.sections["Champions"])
    assert "STALE" not in champs_text
    assert "donchian_breakout" in champs_text
    assert "test=1.5000" in champs_text


def test_reorganize_enforces_size_budget(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md", max_lines=30)
    for i in range(50):
        store.append_finding(f"finding number {i}")
    store.reorganize()
    assert store._total_lines() <= 30
    assert any("older changelog entries pruned" in ln for ln in store.sections["Index / changelog"])


def test_reorganize_caps_learnings_and_rejected_overflow(tmp_path):
    # The budget must hold even when the overflow lives outside the changelog.
    store = MemoryStore(tmp_path / "MEMORY.md", max_lines=30)
    for i in range(40):
        store.add_learning(f"learning number {i}")
    for i in range(40):
        store.record_rejected("sma_crossover", {"fast": i, "slow": i * 2}, reason="weak OOS")
    store.reorganize()
    assert store._total_lines() <= 30
    # Newest entries survive; the oldest were pruned first.
    assert any("learning number 39" in ln for ln in store.sections["Learnings"])


def test_store_is_pure_file_io_no_provider_sdk(tmp_path):
    """The store never talks to a provider: LLM memory upkeep lives behind the client seam
    (research.distill), so no SDK name may appear in this module (provider-agnostic rule)."""
    import inspect

    import noctis.memory.store as store_mod

    source = inspect.getsource(store_mod)
    for sdk in ("anthropic", "openai", "litellm"):
        assert sdk not in source
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.add_learning("keep this insight")
    store.reorganize()  # rule-based only — must not need network
    assert any("keep this insight" in ln for ln in store.sections["Learnings"])
    text = store.read()
    for header in ("## Champions", "## Learnings", "## Rejected ideas", "## Index / changelog"):
        assert header in text


# ── stage-1 consolidation: one line per lesson class, deterministic (context plan P3) ──────
_CORPUS = [
    "- 2026-06-01 — REJECTED strategy alpha — CLASS LESSON: 1m RSI mean reversion nets negative",
    "- 2026-06-02 — operator note: reset champions after the",
    "  noise-inflated regime",  # wrapped continuation of the previous bullet
    "- 2026-06-03 — REJECTED strategy beta — costs eat the edge",
    "- 2026-06-04 — REJECTED strategy beta — costs eat the edge on every symbol tried",
    '- 2026-06-05 — PROMOTED beta {"lookback": 20} — cleared all gates',
    "- 2026-06-06 — REJECTED strategy beta — still cost-bound after the short leg",
]


def test_consolidate_findings_is_deterministic_and_groups_classes():
    once = consolidate_findings(_CORPUS, limit=10)
    again = consolidate_findings(list(_CORPUS), limit=10)
    assert once == again  # byte-identical on the same corpus
    joined = "\n".join(once)
    # The three beta rejections are one class-level line: newest phrasing, ×3 marker.
    assert joined.count("REJECTED strategy beta") == 1
    assert "still cost-bound after the short leg (×3)" in joined
    # Distinct classes never merge: alpha's lesson and beta's promotion are separate lines.
    assert "REJECTED strategy alpha" in joined
    assert "PROMOTED beta" in joined
    # The wrapped operator note re-joined with its bullet (one entry, not two).
    assert any("reset champions after the noise-inflated regime" in ln for ln in once)


def test_consolidate_findings_keeps_old_class_lesson_past_raw_tail():
    corpus = ["- 2026-05-01 — REJECTED strategy old_class — the class-level lesson"] + [
        f"- 2026-06-0{i} — REJECTED strategy hot_class — attempt details {i}" for i in range(1, 7)
    ]
    tail = consolidate_findings(corpus, limit=5)
    # Raw [-5:] would be five hot_class events and the old lesson would be gone; consolidated,
    # both classes survive in two lines.
    assert len(tail) == 2
    assert any("old_class" in ln for ln in tail)
    assert any("hot_class" in ln and "(×6)" in ln for ln in tail)


def test_consolidate_findings_enforces_char_budget_keeping_newest():
    corpus = [f"- note {c} " + "x" * 100 for c in "abcde"]
    out = consolidate_findings(corpus, limit=5, char_budget=250)
    assert sum(len(ln) for ln in out) <= 250
    assert "note e" in out[-1]  # newest always survives
    assert not any("note a" in ln for ln in out)  # oldest dropped first


def test_consolidate_rejected_merges_per_family_never_drops_a_class():
    ideas = [
        {"family": "sma_x", "params": {"fast": 5}},
        {"family": "rsi_mr", "params": {"period": 7}},
        {"family": "sma_x", "params": {"fast": 9}},
        {"family": "sma_x", "params": {"fast": 12}},
    ]
    out = consolidate_rejected(ideas, limit=10)
    assert len(out) == 2  # one record per family (class), none dropped
    by_family = {d["family"]: d for d in out}
    assert by_family["sma_x"]["params"] == {"fast": 12}  # latest params kept
    assert by_family["sma_x"]["times"] == 3
    assert "times" not in by_family["rsi_mr"]  # single rejection stays unannotated


# ── the machine-owned distilled section (stage 2 persistence) ──────────────────────────────
def test_distilled_section_replaced_in_place_without_touching_others(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.add_learning("hand-written learning stays put")
    store.append_finding("a finding")
    store.record_rejected("sma_x", {"fast": 5})
    before = {name: list(lines) for name, lines in store.sections.items()}

    store.set_distilled(["first lesson", "- second lesson"])
    assert store.distilled() == ["- first lesson", "- second lesson"]
    # Regeneration REPLACES the block (no duplication), still one fenced section.
    store.set_distilled(["- rewritten lesson"])
    assert store.distilled() == ["- rewritten lesson"]
    assert store.read().count("## Distilled lessons") == 1
    assert "machine-owned" in store.read()  # the ownership marker fences the block
    # Nothing outside the fence moved.
    for name, lines in before.items():
        assert store.sections[name] == lines

    # Survives a reload; a store that never distilled has no section at all.
    assert MemoryStore(tmp_path / "MEMORY.md").distilled() == ["- rewritten lesson"]
    fresh = MemoryStore(tmp_path / "fresh.md")
    assert fresh.distilled() == []
    assert "Distilled lessons" not in fresh.read()
