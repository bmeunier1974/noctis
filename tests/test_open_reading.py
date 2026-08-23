"""The reading band — one entry reads a run, or the reserved one (story #293, epic #292).

``bootstrap.open_reading`` is to a read-only verb what ``resolve_session`` is to ``noctis run``:
the one place the precedence chain is resolved, and the one place a run's **frozen** inputs come
back out of its record. Unaddressed it is today's chain (settings → gate when asked → mandate
overlay) over the reserved ``legacy`` tree; addressed it resolves the operator's address, reads
the record without a lock, binds the run's tree and rehydrates what the run was steered with.

Everything asserted here is external: what the returned ``Reading`` reports, and what the run
tree looks like afterwards (byte-identical ``run.json``, no ``run.lock``). The band is driven
directly against a ``tmp_path`` runs directory — no Typer command is invoked to read a run — and
runs are minted through the real composition root (``resolve_session`` → ``open_segment``), so
what a reading rehydrates is what a session actually froze.

Two contracts get particular attention, because they are the ones that would otherwise be written
twice and drift:

* **A reading of a run sees what a resume of that run would run under, minus the lock** — pinned
  directly by reading and resuming the same record and comparing the frozen tier.
* **A reading acts on nothing** — a completed run and a live-locked run are both readable, and
  the tree is untouched afterwards.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from noctis.bootstrap import (
    RunPrunedError,
    open_reading,
    open_segment,
    resolve_session,
    resume_session,
)
from noctis.config import SafetyGateError, load_settings
from noctis.reporting.run_tree import (
    RUN_LOCK_NAME,
    RunAmbiguousError,
    RunNotFoundError,
    finish_run,
    prune_run_state,
)

from ._run_tree_helpers import hold_lock, write_run

# The mandate every fixture below steers with: it binds the one ``promotion.*`` key an overlay may
# bind, which is exactly the key four read-only verbs used to read pre-overlay.
SPICY = "---\nsummary: go fast\nconfig:\n  promotion:\n    metric: sortino\n---\nGo fast.\n"


def _mandate_dir(tmp_path: Path, profile: str = "spicy", body: str = SPICY) -> Path:
    path = tmp_path / "mandate" / "profiles" / f"{profile}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return tmp_path / "mandate"


def _config(tmp_path: Path, lines: list[str], name: str = "config.yaml") -> str:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _steered_config(tmp_path: Path, name: str = "config.yaml") -> str:
    """``config.yaml`` saying ``sharpe``, under a mandate that overlays ``sortino`` onto it."""
    mandate_dir = _mandate_dir(tmp_path)
    return _config(
        tmp_path,
        [
            "mode: paper",
            "promotion:",
            "  metric: sharpe",
            f"mandate_dir: {mandate_dir}",
            "research:",
            "  mandate: spicy",
        ],
        name,
    )


def _bare_config(tmp_path: Path, name: str = "bare.yaml") -> str:
    """The same file after the operator changed the metric back and dropped the mandate."""
    return _config(
        tmp_path,
        ["mode: paper", "promotion:", "  metric: sharpe", f"mandate_dir: {tmp_path}/no-mandate"],
        name,
    )


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _mint(cfg: str, *, label: str | None = None) -> Path:
    """One real run segment, opened and closed through the composition root. Returns its tree."""
    inputs = resolve_session(cfg)
    with open_segment(inputs, command="run", argv=["run"], label=label) as segment:
        run_dir = Path(segment.store.run_dir)
        segment.finish("time_limit")
    return run_dir


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _now() -> datetime:
    return datetime.now(UTC)


# ── unaddressed: the chain resolve_session runs, minus the shaping flags ───────────────────


def test_an_unaddressed_reading_sees_the_post_overlay_metric(tmp_path):
    """The bug the band exists for: a reader that stopped at ``load_settings`` saw ``sharpe``
    while the run that crowned the champions was steered onto ``sortino``."""
    reading = open_reading(_steered_config(tmp_path))

    assert reading.settings.promotion.metric == "sortino"
    assert reading.inputs.overrides == ["promotion.metric=sortino"]
    assert reading.inputs.mandate is not None and reading.inputs.mandate.source == "profile:spicy"
    assert reading.addressed is False
    assert reading.address is None
    assert reading.record is None
    assert reading.run_id == "legacy"
    assert reading.run_dir == _runs_dir(tmp_path) / "legacy"


def test_a_reading_that_skips_the_overlay_reads_the_file_and_no_mandate(tmp_path):
    """``status``'s degrade (D6): a refused mandate becomes a report line, not a second prelude."""
    reading = open_reading(_steered_config(tmp_path), mandate_overlay=False)

    assert reading.settings.promotion.metric == "sharpe"
    assert reading.inputs.mandate is None
    assert reading.inputs.overrides == []


def test_a_reading_resolves_no_mode_unless_the_verb_narrates_one(tmp_path):
    """D1: eleven readers gain no gate refusal for no safety gain — a reading places no order."""
    cfg = _steered_config(tmp_path)

    assert open_reading(cfg).mode is None
    assert open_reading(cfg, require_gate=True).mode == "paper"


def test_a_gated_reading_still_refuses_a_live_mode_without_the_env_gate(tmp_path):
    """Rule 1's two gates are re-resolved fresh, exactly as at any other start."""
    cfg = _config(tmp_path, ["mode: live"], "live.yaml")

    assert open_reading(cfg).mode is None  # ungated: nothing measured, nothing refused
    with pytest.raises(SafetyGateError):
        open_reading(cfg, require_gate=True)


# ── addressed: what the run was steered with, out of its own record ───────────────────────


def test_an_addressed_reading_reads_the_runs_frozen_metric_after_the_config_moved(tmp_path):
    """Frozen wins: the run was crowned under ``sortino``, and it still reads ``sortino`` after
    ``config.yaml`` went back to ``sharpe`` and the mandate was deleted."""
    run_dir = _mint(_steered_config(tmp_path))
    bare = _bare_config(tmp_path)

    reading = open_reading(bare, run_dir.name)

    assert reading.settings.promotion.metric == "sortino"
    assert reading.addressed is True
    assert reading.address == run_dir.name
    assert reading.run_id == run_dir.name
    # The control: the same files, read unaddressed, are the current ones.
    assert open_reading(bare).settings.promotion.metric == "sharpe"


def test_a_run_that_froze_nothing_is_read_under_the_current_config_with_no_overlay(
    tmp_path, caplog
):
    """Adopted history (``noctis migrate``) froze no configuration, so there is nothing to
    rehydrate — exactly ``resume_session``'s branch, warning included."""
    write_run(_runs_dir(tmp_path), "20260101-000000-abcdef")

    with caplog.at_level(logging.WARNING, logger="noctis.bootstrap"):
        reading = open_reading(_steered_config(tmp_path), "20260101-000000-abcdef")

    assert reading.settings.promotion.metric == "sharpe"
    assert reading.inputs.mandate is None
    assert reading.inputs.overrides == []
    assert "froze no configuration" in caplog.text


def test_an_addressed_reading_binds_the_four_run_scoped_paths_and_not_the_lake(tmp_path):
    """Opening a run binds its state; reading one binds the same four paths and nothing else —
    the data lake is expensive, reproducible and run-neutral, so it stays shared."""
    cfg = _config(tmp_path, ["mode: paper", "data:", f"  lake_dir: {tmp_path}/lake"], "lake.yaml")
    run_dir = _mint(cfg)

    reading = open_reading(cfg, run_dir.name)

    assert Path(reading.settings.run_dir) == run_dir
    assert Path(reading.settings.state_dir) == run_dir / "state"
    assert Path(reading.settings.reports_dir) == run_dir / "reports"
    assert Path(reading.settings.qa_dir) == run_dir / "qa"
    assert Path(reading.settings.memory_path) == run_dir / "memory" / "MEMORY.md"
    assert Path(reading.settings.data.lake_dir) == tmp_path / "lake"


def test_the_frozen_mandate_round_trips_out_of_the_record(tmp_path):
    """A run freezes its mandate as resolved *text*, so a reading quotes the bytes that steered
    it rather than whatever the profile says today."""
    cfg = _steered_config(tmp_path)
    minted = resolve_session(cfg)
    run_dir = _mint(cfg)
    _mandate_dir(tmp_path, body="---\nconfig:\n  promotion:\n    metric: total_return\n---\nNew.\n")

    reading = open_reading(cfg, run_dir.name)

    assert minted.mandate is not None and reading.inputs.mandate is not None
    assert reading.inputs.mandate.text == minted.mandate.text
    assert reading.inputs.mandate.source == "profile:spicy"
    assert reading.inputs.overrides == ["promotion.metric=sortino"]
    assert reading.settings.promotion.metric == "sortino"  # never the rewritten profile's value


def test_a_reading_carries_the_record_it_read(tmp_path):
    run_dir = _mint(_steered_config(tmp_path))

    reading = open_reading(_bare_config(tmp_path), run_dir.name)

    assert reading.record == _record(run_dir)
    assert reading.pruned is False


# ── the four address forms, resolved by the one resolver ──────────────────────────────────


def test_every_address_form_resolves_to_the_run_it_names(tmp_path):
    cfg = _steered_config(tmp_path)
    _mint(cfg)  # a second run, so `latest` and `@label` have something to choose against
    momo = _mint(cfg, label="nightly-momo")

    for address in (momo.name, "@nightly-momo", "latest", str(momo / "run.json")):
        reading = open_reading(cfg, address)
        assert reading.run_dir == momo, address
        assert reading.address == address, address


def test_an_ambiguous_label_refuses_naming_both_runs(tmp_path):
    cfg = _steered_config(tmp_path)
    first = _mint(cfg, label="nightly-momo")
    second = _mint(cfg, label="nightly-momo")

    with pytest.raises(RunAmbiguousError) as refusal:
        open_reading(cfg, "@nightly-momo")

    assert first.name in str(refusal.value)
    assert second.name in str(refusal.value)


def test_an_address_nobody_answers_raises_run_not_found(tmp_path):
    with pytest.raises(RunNotFoundError):
        open_reading(_steered_config(tmp_path), "20200101-000000-nosuch")


# ── a reading acts on nothing ─────────────────────────────────────────────────────────────


def test_a_completed_run_is_readable(tmp_path):
    """A completed run is terminal for *working* a run, never for reading one."""
    cfg = _steered_config(tmp_path)
    run_dir = _mint(cfg)
    finish_run(_runs_dir(tmp_path), run_dir.name, clock=_now, election_metric="sortino")

    reading = open_reading(cfg, run_dir.name)

    assert reading.run_dir == run_dir
    assert reading.record is not None and reading.record["run"]["status"] == "completed"
    assert reading.settings.promotion.metric == "sortino"


def test_a_run_another_engine_is_live_on_is_readable(tmp_path):
    """Reading a run beside a running engine never blocks and never corrupts: no lock is taken."""
    cfg = _steered_config(tmp_path)
    run_dir = _mint(cfg)
    hold_lock(run_dir, run_id=run_dir.name)

    reading = open_reading(cfg, run_dir.name)

    assert reading.run_dir == run_dir
    assert (run_dir / RUN_LOCK_NAME).is_file()  # the other engine's lock, untouched


def test_a_reading_writes_no_byte_of_the_tree(tmp_path):
    cfg = _steered_config(tmp_path)
    run_dir = _mint(cfg)
    before = (run_dir / "run.json").read_bytes()
    entries = sorted(p.name for p in run_dir.iterdir())

    open_reading(cfg, run_dir.name)

    assert (run_dir / "run.json").read_bytes() == before
    assert not (run_dir / RUN_LOCK_NAME).exists()
    assert sorted(p.name for p in run_dir.iterdir()) == entries


# ── a pruned run: the band refuses, the verb says whether it can read one ──────────────────


def test_a_pruned_run_is_refused_by_default_and_pointed_at_its_record(tmp_path):
    cfg = _steered_config(tmp_path)
    run_dir = _mint(cfg)
    finish_run(_runs_dir(tmp_path), run_dir.name, clock=_now, election_metric="sortino")
    prune_run_state(_runs_dir(tmp_path), run_dir.name, clock=_now, election_metric="sortino")

    with pytest.raises(RunPrunedError) as refusal:
        open_reading(cfg, run_dir.name)

    assert run_dir.name in str(refusal.value)
    assert "run-prune" in str(refusal.value)
    assert f"noctis run-record {run_dir.name}" in str(refusal.value)


def test_a_pruned_run_reads_when_the_verb_says_it_can(tmp_path):
    """``run-record``, ``--finish`` and ``run-prune`` still reach it: the record *is* the history,
    and sealing must never depend on the heavy directories."""
    cfg = _steered_config(tmp_path)
    run_dir = _mint(cfg)
    finish_run(_runs_dir(tmp_path), run_dir.name, clock=_now, election_metric="sortino")
    prune_run_state(_runs_dir(tmp_path), run_dir.name, clock=_now, election_metric="sortino")

    reading = open_reading(cfg, run_dir.name, readable_pruned=True)

    assert reading.pruned is True
    assert reading.settings.promotion.metric == "sortino"  # the run's own election metric


# ── the shared recipe: a reading and a resume of one record agree ─────────────────────────


def test_a_reading_and_a_resume_of_one_record_agree_on_the_frozen_tier(tmp_path):
    """The pin on the extracted recipe (D2): "what the run saw" has one definition, so a reader
    and a resumed segment can never be told two different things about the same run."""
    run_dir = _mint(_steered_config(tmp_path))
    bare = _bare_config(tmp_path)

    reading = open_reading(bare, run_dir.name)
    resumed = resume_session(bare, run_id=run_dir.name)

    assert reading.settings.promotion.metric == resumed.settings.promotion.metric == "sortino"
    assert reading.settings.universe == resumed.settings.universe
    assert reading.settings.research.model_dump() == resumed.settings.research.model_dump()
    assert reading.settings.promotion.model_dump() == resumed.settings.promotion.model_dump()
    assert reading.inputs.mandate == resumed.mandate
    assert reading.inputs.overrides == resumed.overrides


def test_a_reading_runs_none_of_the_resume_policy(tmp_path):
    """A reader acts on nothing: no ``assert_resumable``, no rebase, no engine-change note — and
    therefore nothing on the reading that a segment would then have to record."""
    cfg = _steered_config(tmp_path)
    run_dir = _mint(cfg)
    finish_run(_runs_dir(tmp_path), run_dir.name, clock=_now, election_metric="sortino")

    reading = open_reading(cfg, run_dir.name)

    assert reading.inputs.resume is None
    assert reading.inputs.engine_notes == []
    assert reading.inputs.engine_upgrade is None
    assert reading.inputs.rebase is None


def test_the_reserved_run_needs_no_tree_on_disk(tmp_path):
    """An unaddressed reading resolves and binds nothing: the reserved run is what the settings
    already point at, whether or not anything has ever written there."""
    settings = load_settings(config_path=_bare_config(tmp_path))
    assert not Path(settings.run_dir).exists()

    reading = open_reading(_bare_config(tmp_path))

    assert reading.run_dir == Path(settings.run_dir)
    assert not reading.run_dir.exists()
