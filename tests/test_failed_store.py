"""The capped failed/ store — rejected coder attempts persisted, oldest evicted.

External behavior only: what files land on disk, what they carry, and that the cap holds.
The store is the on-disk half of #18 (a bad authoring session inspectable from disk); the
toolbox drives it, but the cap and file shape are proven directly here.
"""

from __future__ import annotations

from noctis.research.failed_store import DEFAULT_CAP, FailedAttemptStore


def test_records_source_and_error_in_one_file(tmp_path):
    # One file per attempt carries BOTH halves a human needs: the attempted source and the
    # gate error, plus the strategy name and attempt number.
    store = FailedAttemptStore(tmp_path / "failed")
    path = store.record("rsi_meanrev", 2, "source-body-here", "gate error: boom")

    assert path.parent == tmp_path / "failed"
    body = path.read_text(encoding="utf-8")
    assert "source-body-here" in body
    assert "gate error: boom" in body
    assert "rsi_meanrev" in body
    assert "attempt 2" in body


def test_records_the_fixed_oracle_when_supplied(tmp_path):
    # On the spec path the fixed oracle is the target the code missed: persisted in the header
    # alongside the attempted source and the gate error, so a post-mortem shows both what the code
    # did (the observed-behavior diagnostics riding the error) and the target it was gated against.
    store = FailedAttemptStore(tmp_path / "failed")
    oracle = "rally: trend(60) — enter long during leg 0; grind: flat(60) — never trade"
    path = store.record("spec_probe", 1, "source-body", "gate error: boom", oracle=oracle)

    body = path.read_text(encoding="utf-8")
    assert "source-body" in body
    assert "gate error: boom" in body
    assert "fixed oracle" in body.lower()  # the section header
    assert "enter long during leg 0" in body  # the oracle identity itself


def test_omits_the_oracle_section_when_absent(tmp_path):
    # Spec-less writes carry no oracle: the record is unchanged (no fixed-oracle header).
    store = FailedAttemptStore(tmp_path / "failed")
    path = store.record("hand_written", 1, "src", "err")

    assert "fixed oracle" not in path.read_text(encoding="utf-8").lower()


def test_first_record_creates_the_area_lazily(tmp_path):
    # No failures ⇒ no failed/ dir; the first record materializes it.
    store = FailedAttemptStore(tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    store.record("probe", 1, "src", "err")
    assert (tmp_path / "failed").is_dir()


def test_cap_keeps_only_the_most_recent_attempts(tmp_path):
    store = FailedAttemptStore(tmp_path / "failed", cap=50)
    for i in range(60):
        store.record("probe", 1, f"source {i}", "gate error")

    files = list((tmp_path / "failed").glob("*.py"))
    assert len(files) == 50  # the oldest 10 were evicted
    survived = {f.read_text(encoding="utf-8").splitlines()[-1] for f in files}
    assert survived == {f"source {i}" for i in range(10, 60)}


def test_cap_is_global_across_strategy_names(tmp_path):
    # The cap spans the whole failed/ area, not per-strategy.
    store = FailedAttemptStore(tmp_path / "failed", cap=50)
    for i in range(30):
        store.record("alpha", 1, f"a{i}", "e")
    for i in range(30):
        store.record("beta", 1, f"b{i}", "e")

    assert len(list((tmp_path / "failed").glob("*.py"))) == 50


def test_default_store_caps_at_the_default(tmp_path):
    store = FailedAttemptStore(tmp_path / "failed")  # default cap
    for i in range(DEFAULT_CAP + 5):
        store.record("probe", 1, f"s{i}", "e")

    assert len(list((tmp_path / "failed").glob("*.py"))) == DEFAULT_CAP


def test_sequence_stays_monotonic_after_eviction(tmp_path):
    # Eviction removes the lowest sequence; the next record keeps climbing, so ordering never
    # collides after the store rolls over its cap.
    store = FailedAttemptStore(tmp_path / "failed", cap=3)
    for i in range(5):
        store.record("probe", 1, f"s{i}", "e")

    seqs = sorted(int(p.name.split("-", 1)[0]) for p in (tmp_path / "failed").glob("*.py"))
    assert seqs == [3, 4, 5]  # 1 and 2 evicted, sequence never reused
