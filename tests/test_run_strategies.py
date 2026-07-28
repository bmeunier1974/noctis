"""The record's strategies section — everything considered, and how each one died (story #141).

The rejections **are** the product. A results page that shows an equity curve is one claim; a
results page that shows "47 of 66 candidates died at the symbol-holdout gate" is evidence, and it
is computable only if every candidate — not just the champions — carries its structured gate
results. This file pins that section end to end:

* the **pure builder** (``run_record.build``) renders what it is handed, caps the list honestly,
  and never drops a champion to make room;
* the **store** (``run_store.read_strategies``) reads the run's own champion board, experiment
  journals and strategy tiers, embeds a champion's source in full, and references every other
  candidate by a run-relative path plus a content hash;
* ``--embed-all-sources`` — frozen at run creation, like every other knob that says what a run
  *is* — fills them all in for an experiment worth archiving whole;
* and a synthetic two-week run stays inside the stated size budget, which is the whole reason the
  source policy exists.

Everything asserted is external: what the record contains, what it weighs, what is on disk.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from noctis.reporting import schema
from noctis.reporting.run_record import (
    EngineIdentity,
    RunArtifacts,
    SegmentArtifact,
    StrategyArtifact,
    build,
)
from noctis.reporting.run_store import RUN_RECORD_NAME, open_run, read_strategies

START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)
HOUR = 3600.0

ENGINE = EngineIdentity(
    engine_version=1,
    fingerprint={"gates": "f63d47b7b9604ab1", "backtest": "3ba3e0bf1c97134f"},
    comparable_key="1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe",
    noctis_version="0.1.0",
)

# One plausible one-file strategy — the artifact a candidate *is*. Sized like a real one (the
# seeds in ``strategies/`` run 3–8 KB) so the size budget below is measured against something
# honest rather than against a stub.
SOURCE = (
    '"""Volatility breakout.\n\nthesis: range expansion after a quiet session persists for a few '
    'bars.\nstatus: candidate\n"""\n\n'
    + "".join(f"# one line of a one-file strategy, no {i:04d}\n" for i in range(68))
)
assert 2_400 <= len(SOURCE) <= 3_100, "size the fixture like the shipped seeds, or measure nothing"


class FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def _gate(gate: str, passed: bool, observed=None, threshold=None, note=None) -> dict:
    return {
        "gate": gate,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "note": note,
    }


def _strategy(name: str, *, outcome: str = "rejected", died_at: str | None = None, **overrides):
    gates = [_gate("validated", True), _gate("activity_floor", True, 0.9, 0.5)]
    if died_at is not None:
        gates.append(_gate(died_at, False, -0.4, 0.0))
    base = dict(
        name=name,
        outcome=outcome,
        tier="__tmp",
        decided_utc="2026-07-28T02:10:04.002Z",
        trials=42,
        gates=tuple(gates),
        rationale=f"rejected: {name} died at {died_at}",
        source_path=f"strategies/__tmp/{name}.py",
        source_sha256="a" * 64,
        source=None,
    )
    base.update(overrides)
    return StrategyArtifact(**base)  # type: ignore[arg-type]


def _artifacts(**overrides) -> RunArtifacts:
    base = dict(
        run_id="20260727T142233Z-a1b2c3",
        created_utc="2026-07-27T14:22:33.418Z",
        last_active_utc="2026-07-28T02:10:04.002Z",
        engine=ENGINE,
        segments=(
            SegmentArtifact(
                index=0,
                started_utc="2026-07-27T14:22:33.418Z",
                stopped_utc="2026-07-28T02:10:04.002Z",
                stopped_reason="time_limit",
                status="stopped",
                engine=ENGINE,
            ),
        ),
        complete=True,
    )
    base.update(overrides)
    return RunArtifacts(**base)  # type: ignore[arg-type]


# ── the section: every candidate, and the gate that stopped it ─────────────────────────────


def test_the_record_carries_a_strategies_section_even_when_nothing_was_considered():
    record = build(_artifacts())

    assert record["strategies"] == []
    assert schema.validate(record) == []


def test_the_death_counts_that_make_a_results_page_credible_are_computable():
    """The sentence this whole story exists for: "N of M candidates died at gate X", read off the
    record with no prose parsing at all."""
    considered = [
        *(_strategy(f"a{i}", died_at="symbol_holdout") for i in range(5)),
        *(_strategy(f"b{i}", died_at="overfit_gap") for i in range(2)),
        _strategy("champ", outcome="promoted", tier="champions", source=SOURCE),
    ]

    record = build(_artifacts(strategies=tuple(considered)))

    deaths: dict[str, int] = {}
    for entry in record["strategies"]:
        for gate in entry["gates"]:
            if not gate["passed"]:
                deaths[gate["gate"]] = deaths.get(gate["gate"], 0) + 1
    assert deaths == {"symbol_holdout": 5, "overfit_gap": 2}
    assert len(record["strategies"]) == 8  # the denominator is the section itself
    assert schema.validate(record) == []


def test_every_candidate_entry_states_its_outcome_trials_and_the_gates_it_faced():
    record = build(_artifacts(strategies=(_strategy("momo", died_at="forward_holdout"),)))

    entry = record["strategies"][0]
    assert entry["name"] == "momo"
    assert entry["outcome"] == "rejected"
    assert entry["trials"] == 42
    assert entry["tier"] == "__tmp"
    assert [gate["gate"] for gate in entry["gates"]] == [
        "validated",
        "activity_floor",
        "forward_holdout",
    ]
    assert entry["gates"][-1] == {
        "gate": "forward_holdout",
        "passed": False,
        "observed": -0.4,
        "threshold": 0.0,
        "note": None,
    }


def test_a_candidate_that_never_reached_the_gates_says_so_rather_than_pretending():
    record = build(
        _artifacts(
            strategies=(
                StrategyArtifact(name="abandoned", outcome="undecided", tier="__tmp", trials=3),
            )
        )
    )

    entry = record["strategies"][0]
    assert (entry["outcome"], entry["gates"], entry["decided_utc"]) == ("undecided", [], None)
    assert schema.validate(record) == []


def test_the_champions_come_first_so_a_reader_never_has_to_sort_the_product():
    considered = (
        _strategy("zeta", died_at="overfit_gap"),
        _strategy("alpha", died_at="overfit_gap"),
        _strategy("winner", outcome="promoted", tier="champions", source=SOURCE),
    )

    record = build(_artifacts(strategies=considered))

    assert [entry["name"] for entry in record["strategies"]] == ["winner", "alpha", "zeta"]


# ── the source policy: champions in full, everyone else by path + hash ─────────────────────


def test_a_champion_embeds_its_source_and_every_other_candidate_references_one():
    record = build(
        _artifacts(
            strategies=(
                _strategy("champ", outcome="promoted", tier="champions", source=SOURCE),
                _strategy("reject", died_at="symbol_holdout"),
            )
        )
    )

    champion, rejected = record["strategies"]
    assert champion["source"] == SOURCE
    assert rejected["source"] is None
    assert rejected["source_path"] == "strategies/__tmp/reject.py"
    assert rejected["source_sha256"] == "a" * 64
    assert schema.validate(record) == []


def test_the_validator_refuses_an_embedded_source_with_no_hash_beside_it():
    record = build(
        _artifacts(
            strategies=(_strategy("champ", outcome="promoted", source=SOURCE, source_sha256=None),)
        )
    )

    problems = schema.validate(record)
    assert any("source_sha256" in problem for problem in problems)


def test_the_validator_names_an_unknown_outcome_and_a_malformed_gate():
    record = build(_artifacts(strategies=(_strategy("odd", outcome="maybe"),)))
    record["strategies"][0]["gates"][0].pop("threshold")

    problems = schema.validate(record)
    assert any("outcome" in problem for problem in problems)
    assert any("threshold" in problem for problem in problems)


# ── the cap: it bites honestly, and never on a champion ────────────────────────────────────


def test_the_strategy_cap_writes_a_truncation_note_with_kept_and_total_counts():
    considered = tuple(
        _strategy(f"c{i:04d}", died_at="overfit_gap") for i in range(schema.STRATEGY_CAP + 7)
    )

    record = build(_artifacts(strategies=considered))

    assert len(record["strategies"]) == schema.STRATEGY_CAP
    assert record["run"]["truncated"]["strategies"] == {
        "kept": schema.STRATEGY_CAP,
        "total": schema.STRATEGY_CAP + 7,
    }
    assert schema.validate(record) == []


def test_the_cap_never_drops_a_champion_the_run_actually_produced():
    considered = (
        *(_strategy(f"c{i:04d}", died_at="overfit_gap") for i in range(schema.STRATEGY_CAP + 7)),
        _strategy("zzz_champion", outcome="promoted", tier="champions", source=SOURCE),
    )

    record = build(_artifacts(strategies=considered))

    names = [entry["name"] for entry in record["strategies"]]
    assert names[0] == "zzz_champion"  # last alphabetically, first in the record
    assert record["run"]["truncated"]["strategies"]["total"] == schema.STRATEGY_CAP + 8


# ── the store: reading the run's own board, journals and tiers ─────────────────────────────


def _run_tree(run_dir: Path, *, candidates: int = 2, trials: int = 3) -> None:
    """One run's worth of research on disk: judged candidates, their files, their journals."""
    from noctis.champions import ChampionRegistry, PromotionRules

    from ._promotion_cases import card

    registry = ChampionRegistry(run_dir / "state" / "champions.json", capacity=3)
    rules = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)
    registry.consider(card("winner", test=1.5), rules)
    _write_strategy(run_dir, "champions", "winner")
    for index in range(candidates):
        name = f"reject_{index}"
        registry.consider(card(name, test=1.0, symbol_holdout=-0.4), rules)
        _write_strategy(run_dir, "__tmp", name)
    for name in ["winner", *(f"reject_{i}" for i in range(candidates))]:
        _journal(run_dir, name, trials=trials)


def _write_strategy(run_dir: Path, tier: str, name: str) -> Path:
    path = run_dir / "strategies" / tier / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SOURCE, encoding="utf-8")
    return path


def _journal(run_dir: Path, name: str, *, trials: int) -> None:
    path = run_dir / "state" / "experiments" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"event": "trial", "params": {"lookback": i}}) + "\n" for i in range(trials)
        ),
        encoding="utf-8",
    )


def _by_name(strategies) -> dict:
    return {entry.name: entry for entry in strategies}


def test_the_store_reads_the_gates_the_champion_board_journaled(tmp_path):
    """End to end from ``decide()`` to the record: a rejection's structured evidence survives the
    board it was written to, which is the only durable trace a rejected candidate leaves."""
    _run_tree(tmp_path, candidates=1)

    found = _by_name(read_strategies(tmp_path))

    assert found["reject_0"].outcome == "rejected"
    assert [gate["gate"] for gate in found["reject_0"].gates][-1] == "symbol_holdout"
    assert found["reject_0"].gates[-1]["passed"] is False
    assert found["reject_0"].gates[-1]["observed"] == -0.4
    assert found["winner"].outcome == "promoted"
    assert all(gate["passed"] for gate in found["winner"].gates)


def test_the_store_embeds_a_champions_source_and_hashes_every_other_candidate(tmp_path):
    _run_tree(tmp_path, candidates=1)

    found = _by_name(read_strategies(tmp_path))

    digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    assert found["winner"].source == SOURCE
    assert found["winner"].tier == "champions"
    assert found["winner"].source_path == "strategies/champions/winner.py"
    assert found["reject_0"].source is None
    assert found["reject_0"].source_sha256 == digest
    assert found["reject_0"].source_path == "strategies/__tmp/reject_0.py"


def test_a_referenced_source_path_resolves_against_the_run_directory(tmp_path):
    """Referenced *by path* has to mean something: the path is run-relative — portable, and
    resolvable beside the record that quotes it."""
    _run_tree(tmp_path, candidates=1)

    found = _by_name(read_strategies(tmp_path))

    resolved = tmp_path / found["reject_0"].source_path
    assert resolved.is_file()
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == found["reject_0"].source_sha256


def test_embed_all_sources_fills_in_every_candidates_source(tmp_path):
    _run_tree(tmp_path, candidates=2)

    archived = _by_name(read_strategies(tmp_path, _frozen(embed_all_sources=True)))

    assert [entry.source for entry in archived.values()] == [SOURCE] * 3
    assert all(entry.source_sha256 for entry in archived.values())


def _frozen(**resolved) -> dict:
    """A record's frozen inputs — the only place a run's own knobs are ever read from."""
    return {
        "config_epoch": 1,
        "config_changes": [],
        "frozen_at_utc": "2026-07-27T14:22:33.418Z",
        "execution_mode": "paper",
        "mandate": None,
        "models": None,
        "data": None,
        "settings": {
            "digest": "sha256:0f1e2d",
            "resolved": dict(resolved),
            "frozen_keys": [],
            "live_keys": [],
            "refused_keys": [],
        },
    }


def test_the_trial_count_beside_each_candidate_comes_from_its_own_journal(tmp_path):
    _run_tree(tmp_path, candidates=1, trials=5)

    found = _by_name(read_strategies(tmp_path))

    assert found["reject_0"].trials == 5


def test_a_candidate_with_a_file_but_no_verdict_is_undecided(tmp_path):
    _write_strategy(tmp_path, "__tmp", "half_done")

    found = _by_name(read_strategies(tmp_path))

    assert found["half_done"].outcome == "undecided"
    assert found["half_done"].gates == ()
    assert found["half_done"].trials is None


def test_a_run_that_researched_nothing_reports_no_candidates(tmp_path):
    assert read_strategies(tmp_path) == ()


def test_an_unreadable_board_costs_the_record_its_verdicts_never_its_candidates(tmp_path):
    """Partial evidence is still evidence — and a record that dropped a whole section because one
    file was corrupt would be least informative exactly when someone needs to look."""
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "champions.json").write_text("{not json", encoding="utf-8")
    _write_strategy(tmp_path, "__tmp", "orphan")

    found = _by_name(read_strategies(tmp_path))

    assert found["orphan"].outcome == "undecided"
    assert found["orphan"].source_sha256 is not None


def test_the_tiers_the_store_reads_are_the_tiers_the_library_writes(tmp_path):
    """One layout, two readers: the record names the run's strategy tiers by hand (it must not
    import the library — that would drag pandas into every record write), so this pins the two
    spellings together."""
    from types import SimpleNamespace

    from noctis.reporting.run_store import (
        CHAMPIONS_TIER,
        STRATEGIES_SUBDIR,
        STRATEGY_TIER_SUBDIRS,
        TMP_TIER,
    )
    from noctis.strategies.library import LibraryPaths

    paths = LibraryPaths.from_settings(
        SimpleNamespace(run_dir=str(tmp_path), strategies_dir="strategies/")
    )

    assert paths.champions == tmp_path / STRATEGIES_SUBDIR / CHAMPIONS_TIER
    assert paths.tmp == tmp_path / STRATEGIES_SUBDIR / TMP_TIER
    assert set(STRATEGY_TIER_SUBDIRS) == set(schema.STRATEGY_TIERS)


# ── the whole way through: a run that researched, written to disk ──────────────────────────


def _open(runs: Path, clock: FakeClock, **kwargs):
    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs, clock=clock, **kwargs)


def _on_disk(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def test_a_night_of_research_lands_in_the_record_it_wrote(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    _run_tree(store.run_dir, candidates=2)
    clock.advance(HOUR)
    store.close(reason="stop_requested")

    record = _on_disk(store.run_dir)

    assert schema.validate(record) == []
    assert [entry["name"] for entry in record["strategies"]] == [
        "winner",
        "reject_0",
        "reject_1",
    ]
    assert record["strategies"][0]["source"] == SOURCE
    assert record["strategies"][1]["source"] is None


def test_the_section_is_derived_at_every_write_never_carried_forward(tmp_path):
    """Epic D4: a second segment re-reads the run's own board and tiers, so a candidate judged
    tonight appears without anyone having incremented anything."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    clock.advance(HOUR)
    first.close(reason="stop_requested")
    assert _on_disk(first.run_dir)["strategies"] == []

    _run_tree(first.run_dir, candidates=1)
    second = _open(runs, clock, run_id=first.run_id, resume=True)
    clock.advance(HOUR)
    second.close(reason="stop_requested")

    assert len(_on_disk(second.run_dir)["strategies"]) == 2


def test_the_embed_choice_is_the_runs_own_frozen_one_not_this_processes(tmp_path):
    """``--embed-all-sources`` is frozen at creation like every other knob that says what a run
    *is*: a resumed segment must not silently drop (or add) the sources an earlier one embedded,
    or the record's content would depend on how it was last written."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, inputs=_frozen(embed_all_sources=True))
    _run_tree(first.run_dir, candidates=1)
    clock.advance(HOUR)
    first.close(reason="stop_requested")
    assert all(entry["source"] for entry in _on_disk(first.run_dir)["strategies"])

    second = _open(runs, clock, run_id=first.run_id, resume=True, inputs=_frozen())
    clock.advance(HOUR)
    second.close(reason="stop_requested")

    assert all(entry["source"] for entry in _on_disk(second.run_dir)["strategies"])


# ── the size budget the source policy exists to hold ───────────────────────────────────────


def _two_week_run(tmp_path: Path, *, embed_all: bool = False) -> Path:
    """Fourteen nights: 66 candidates considered, 3 crowned, ~3 000 trials journaled.

    The epic's own worked figures (§6, §11), so the number this weighs is the number an operator
    would actually get — not a stub that would make any budget look comfortable.
    """
    from noctis.champions import ChampionRegistry, PromotionRules

    from ._promotion_cases import card

    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, inputs=_frozen(embed_all_sources=embed_all))
    run_dir = store.run_dir
    registry = ChampionRegistry(run_dir / "state" / "champions.json", capacity=3)
    rules = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)
    for index in range(66):
        name = f"candidate_{index:03d}"
        promoted = index % 22 == 0
        challenger = card(name, test=1.0 + index) if promoted else card(name, symbol_holdout=-0.3)
        registry.consider(challenger, rules)
        _write_strategy(run_dir, "champions" if promoted else "__tmp", name)
        _journal(run_dir, name, trials=45)
    clock.advance(HOUR)
    store.close(reason="time_limit")
    for _ in range(13):
        night = _open(runs, clock, run_id=store.run_id, resume=True)
        clock.advance(8 * HOUR)
        night.close(reason="time_limit", counters={"cycles": 1, "research_iterations": 40})
    return run_dir


def test_a_synthetic_two_week_run_stays_inside_the_records_size_budget(tmp_path):
    run_dir = _two_week_run(tmp_path)

    record = _on_disk(run_dir)
    size = (run_dir / RUN_RECORD_NAME).stat().st_size

    assert schema.validate(record) == []
    assert len(record["segments"]) == 14
    assert len(record["strategies"]) == 66
    assert size <= schema.RECORD_SIZE_BUDGET_BYTES, f"{size} bytes"
    assert record["run"]["truncated"] == {}  # nothing bit, so nothing is claimed


def test_archiving_the_same_run_whole_is_what_the_source_policy_saves(tmp_path):
    """Why champions-only is the default, in one number: the same fourteen nights with every
    candidate's source embedded — the bill ``--embed-all-sources`` pays deliberately."""
    lean = (_two_week_run(tmp_path / "lean") / RUN_RECORD_NAME).stat().st_size
    whole = (_two_week_run(tmp_path / "whole", embed_all=True) / RUN_RECORD_NAME).stat().st_size

    assert whole > 2 * lean


# ── the operator's switch ──────────────────────────────────────────────────────────────────


def test_the_flag_freezes_the_choice_onto_the_run_at_creation():
    from noctis.bootstrap import resolve_session

    session = resolve_session(config_path="does-not-exist.yaml", embed_all_sources=True)

    assert session.settings.embed_all_sources is True


def test_the_setting_defaults_to_champions_only():
    from noctis.config.settings import load_settings

    assert load_settings(config_path="does-not-exist.yaml").embed_all_sources is False


def test_a_resume_refuses_the_flag_because_the_choice_belongs_to_the_run(tmp_path):
    from noctis.bootstrap import UsageError, resolve_session

    with pytest.raises(UsageError) as refusal:
        resolve_session(resume="latest", embed_all_sources=True)

    assert "frozen at creation" in str(refusal.value)


def test_the_freezing_tier_keeps_the_choice_with_the_run():
    from noctis.config.rehydrate import classify

    assert classify("embed_all_sources") == "frozen"


def test_a_mandate_may_not_decide_how_much_of_the_run_the_record_embeds():
    """The overlay is deny-by-default (AGENTS.md rule 5): steering says what to look for, never
    what the artifact describing it contains."""
    from noctis.config.overlay import REFUSED

    assert "embed_all_sources" in REFUSED
