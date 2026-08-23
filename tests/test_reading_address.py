"""Every read-only verb takes a run address (story #295, epic #292).

``champions``, ``account``, ``backtest`` and ``strategies`` used to read whatever the current
``config.yaml`` pointed at — always the reserved ``legacy`` run — so the run ``noctis run`` had
just minted could not be inspected at all. They now open the same reading ``report`` does
(``bootstrap.open_reading`` through ``cli._reading_or_exit``) and take the same optional trailing
address, in the same four forms, resolved by the same resolver.

Everything asserted here is external: what a command prints, what it exits with, and which run's
tree the answer came out of. Runs are minted by the real CLI rather than faked on disk, so the
addresses under test are the ones an operator would actually type — and each run's tree is seeded
with a mark only that run carries, so "which run did this read?" is answered by the output alone.

The reader table is the point: one row per verb on the band, so a verb that stops at
``load_settings`` again fails its own row. ``status`` joined it in #296 — the one reader that
narrates the resolved mode, so the one that arms the safety gate and warns where the others
refuse. The pruned-run-readable trio ``run-record`` / ``run --finish`` / ``run-prune`` (#297)
joins as it moves.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app

runner = CliRunner()


# ── the workspace an operator would have ───────────────────────────────────────────────────


def _config(tmp_path: Path) -> str:
    """A paper config whose lake and seed tier are empty tmp directories.

    Nothing run-scoped is pinned on purpose: ``state_dir`` and friends must stay *derived*, or
    ``bind_run_dir`` would honour the explicit path and no address could ever rebind them.
    """
    path = tmp_path / "config.yaml"
    path.write_text(
        f"mode: paper\nuniverse: [AAPL]\ndata:\n  lake_dir: {tmp_path}/lake\n"
        f"  dataset: EQUS.MINI\nstrategies_dir: {tmp_path}/seeds\n"
    )
    (tmp_path / "seeds").mkdir(exist_ok=True)
    return str(path)


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _mint_run(tmp_path: Path, cfg: str, *, label: str | None = None) -> Path:
    """One real ``noctis run`` — the run tree an operator would then address."""
    argv = ["run", "--config", cfg] + (["--label", label] if label else [])
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, result.output
    minted = sorted(
        (p for p in _runs_dir(tmp_path).iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return minted[-1]


def _steered_config(tmp_path: Path) -> Path:
    """The same config, plus a mandate that binds ``promotion.metric: sortino``.

    Rewriting it with :func:`_config` afterwards is the whole point of the two tests below: the
    steering is gone from the files, and a run minted under it must still be read under it.
    """
    profile = tmp_path / "mandate" / "profiles" / "sortino_hunter.md"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        "---\nsummary: A steering personality.\nconfig:\n  promotion:\n"
        "    metric: sortino\n---\nSteer this session.\n"
    )
    cfg = Path(_config(tmp_path))
    steering = f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: sortino_hunter\n"
    cfg.write_text(cfg.read_text() + steering)
    return cfg


def _prune(tmp_path: Path, cfg: str) -> Path:
    """A minted run whose heavy tree retention has deleted — record kept, state gone."""
    run_dir = _mint_run(tmp_path, cfg)
    sealed = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name, "--finish"])
    assert sealed.exit_code == 0, sealed.output
    pruned = runner.invoke(app, ["run-prune", run_dir.name, "--config", cfg])
    assert pruned.exit_code == 0, pruned.output
    assert json.loads((run_dir / "run.json").read_text())["run"]["state_pruned"] is True
    return run_dir


# ── what one run's tree can be marked with ─────────────────────────────────────────────────


def _strategy_source(name: str) -> str:
    """One importable one-file strategy — what a run's working tier actually holds."""
    return f'''"""A probe authored inside one run's working tier.

status: draft
style: momentum
"""

from dataclasses import dataclass

from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class Probe(TraderStrategy):
    name = "{name}"

    @dataclass(frozen=True)
    class Params:
        lookback: int = 12

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        self._seen = 0

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._seen += 1
        ctx.set_target(0)

    @classmethod
    def param_space(cls):
        return [ParamSpec("lookback", "int", 5, 40, 1)]
'''


def _seed_champion(run_dir: Path, mark: int) -> str:
    """Crown one champion on this run's own board; return the family only this run has."""
    from noctis.backtest.scorecard import Scorecard
    from noctis.champions.registry import ChampionEntry, ChampionRegistry

    family = f"champion_of_run_{mark}"
    registry = ChampionRegistry(run_dir / "state" / "champions.json", 3)
    params = {"fast": 3, "slow": 8}
    registry.champions.append(
        ChampionEntry(
            family=family,
            params=params,
            scorecard=Scorecard(family=family, params=params),
            crowned_at="2026-01-01",
            rationale="seed",
        )
    )
    registry.save()
    return family


def _run_cash(mark: int) -> float:
    """The starting cash one run — and only that run — opened its paper account at."""
    return 100_000.0 + mark


def _seed_account(run_dir: Path, mark: int) -> str:
    """Open this run's own paper account at a starting cash only this run has."""
    from noctis.broker.paper import PaperBroker
    from noctis.broker.persistence import AccountStore

    store = AccountStore(run_dir / "state" / "paper_account.json")
    store.save(PaperBroker(starting_cash=_run_cash(mark)), date(2026, 7, 6))
    return f"starting cash:    {_run_cash(mark):,.2f}"


def _seed_account_equity(run_dir: Path, mark: int) -> str:
    """The same per-run account, read back the way ``status`` narrates it: one equity line."""
    _seed_account(run_dir, mark)
    return f"equity {_run_cash(mark):,.2f}"


def _seed_draft(run_dir: Path, mark: int) -> str:
    """Author one draft into this run's working tier; return the name only this run has."""
    name = f"probe_of_run_{mark}"
    tmp = run_dir / "strategies" / "__tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"{name}.py").write_text(_strategy_source(name))
    return name


# ── the reader table ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Reader:
    """One read-only verb on the reading band: how to mark a run, and how to read the mark back."""

    id: str
    # Give one run's tree something no other run has; returns the text that proves it was read.
    seed: Callable[[Path, int], str]
    # The whole argv for this verb, given the address (``None`` is the bare, reserved-run form).
    argv: Callable[[str | None], list[str]]
    # What a reading that *worked* exits with — not always 0: ``backtest`` below is asked for a
    # family nobody has, precisely so that it prints the addressed run's library index and stops.
    code: int = 0
    # What the *bare* form exits with beside an un-migrated pre-workspace layout. The guard
    # refuses (2) for every reader but ``status``, whose standing exception is to warn and carry
    # on (0) — it is the diagnostic an operator runs *because* the layout is wrong.
    guard_code: int = 2


READERS = (
    Reader(
        id="champions",
        seed=_seed_champion,
        argv=lambda address: ["champions", *([address] if address else [])],
    ),
    Reader(
        id="account",
        seed=_seed_account,
        argv=lambda address: ["account", *([address] if address else [])],
    ),
    Reader(
        id="strategies",
        seed=_seed_draft,
        argv=lambda address: ["strategies", *([address] if address else [])],
    ),
    # ``backtest`` reads the run's *library tiers* to resolve the family it is asked for, so the
    # "known" list an unknown name prints is exactly the index of the addressed run's strategies.
    Reader(
        id="backtest",
        seed=_seed_draft,
        argv=lambda address: ["backtest", "no_such_family", *([address] if address else [])],
        code=1,
    ),
    # ``status`` reports the addressed run's own tree — its account, its board — under that
    # run's frozen inputs, and is the only reader whose bare form warns rather than refuses
    # beside an un-migrated layout.
    Reader(
        id="status",
        seed=_seed_account_equity,
        argv=lambda address: ["status", *([address] if address else [])],
        guard_code=0,
    ),
)

EVERY_READER = [pytest.param(reader, id=reader.id) for reader in READERS]


# ── the four address forms ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reader", EVERY_READER)
def test_every_reader_takes_the_same_four_address_forms(tmp_path, reader):
    """One resolver, one set of rules: an id, a ``run.json`` path, ``@LABEL`` and ``latest`` all
    land on the same run's tree, and never on the run beside it."""
    cfg = _config(tmp_path)
    other = _mint_run(tmp_path, cfg)
    momo = _mint_run(tmp_path, cfg, label="nightly-momo")
    mine = reader.seed(momo, 1)
    theirs = reader.seed(other, 2)

    for address in (momo.name, "@nightly-momo", "latest", str(momo / "run.json")):
        result = runner.invoke(app, [*reader.argv(address), "--config", cfg])
        assert result.exit_code == reader.code, f"{address}: {result.output}"
        assert mine in result.output, address
        assert theirs not in result.output, address


@pytest.mark.parametrize("reader", EVERY_READER)
def test_the_bare_form_still_reads_the_reserved_legacy_run(tmp_path, reader):
    """No address keeps today's behaviour exactly: the run every unaddressed verb reads."""
    cfg = _config(tmp_path)
    minted = _mint_run(tmp_path, cfg)
    theirs = reader.seed(minted, 2)
    mine = reader.seed(_runs_dir(tmp_path) / "legacy", 1)

    result = runner.invoke(app, [*reader.argv(None), "--config", cfg])

    assert result.exit_code == reader.code, result.output
    assert mine in result.output
    assert theirs not in result.output


@pytest.mark.parametrize("reader", EVERY_READER)
def test_an_address_is_authoritative_beside_an_un_migrated_legacy_layout(
    tmp_path, monkeypatch, reader
):
    """The legacy guard answers "which tree does an *unaddressed* command read?" — and an address
    answers it instead, so a named run still reads beside a pre-workspace layout that refuses the
    bare form."""
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    mine = reader.seed(run_dir, 1)
    (tmp_path / "state").mkdir()  # pre-workspace artifacts beside config.yaml
    (tmp_path / "reports").mkdir()

    bare = runner.invoke(app, [*reader.argv(None), "--config", cfg])
    addressed = runner.invoke(app, [*reader.argv(run_dir.name), "--config", cfg])

    assert bare.exit_code == reader.guard_code, bare.output  # refused, or warned for `status`
    assert "noctis migrate" in bare.output  # unaddressed: the guard asks its question
    assert addressed.exit_code == reader.code, addressed.output
    assert "noctis migrate" not in addressed.output  # addressed: the address answered it
    assert mine in addressed.output


# ── refusals: the resolver's contract, and retention's ─────────────────────────────────────


@pytest.mark.parametrize("reader", EVERY_READER)
def test_an_ambiguous_label_refuses_naming_both_runs(tmp_path, reader):
    """A label may be reassigned; the id is the identity. Two answers is a refusal, not a pick."""
    cfg = _config(tmp_path)
    first = _mint_run(tmp_path, cfg, label="nightly-momo")
    second = _mint_run(tmp_path, cfg, label="nightly-momo")

    result = runner.invoke(app, [*reader.argv("@nightly-momo"), "--config", cfg])

    assert result.exit_code == 1
    assert first.name in result.output and second.name in result.output


@pytest.mark.parametrize("reader", EVERY_READER)
def test_an_unknown_address_refuses_naming_the_address(tmp_path, reader):
    cfg = _config(tmp_path)
    _mint_run(tmp_path, cfg)

    result = runner.invoke(app, [*reader.argv("20260101T000000Z-nope00"), "--config", cfg])

    assert result.exit_code == 1
    assert "20260101T000000Z-nope00" in result.output


@pytest.mark.parametrize("reader", EVERY_READER)
def test_a_pruned_run_is_refused_with_the_line_that_points_at_the_record(tmp_path, reader):
    """Retention deleted this run's state, strategies and reports on purpose, and the record says
    so. Every verb whose answer would be assembled from that tree is refused with one line — the
    reading band's own, said under no prefix — that names what deleted the tree and what survived
    it."""
    cfg = _config(tmp_path)
    run_dir = _prune(tmp_path, cfg)

    result = runner.invoke(app, [*reader.argv(run_dir.name), "--config", cfg])

    assert result.exit_code == 1
    line = result.output.strip()
    assert line.startswith(f"Run {run_dir.name} was pruned:")
    assert "`noctis run-prune` deleted its reports/ and state/" in line
    assert line.endswith(f"still in its record — `noctis run-record {run_dir.name}`.")
    assert not (run_dir / "state").exists()  # nothing was resurrected


@pytest.mark.parametrize("reader", EVERY_READER)
def test_the_address_argument_is_documented_in_the_house_wording(reader):
    """Every reader describes the address in one wording — the same four forms, named the same
    way, so no verb's help can quietly grow a fifth meaning for the same string."""
    result = runner.invoke(app, [*reader.argv(None)[:1], "--help"])

    # Typer boxes and wraps help text at the terminal width, and where it breaks a line depends
    # on the verb's name; flatten the box back into one sentence before reading it.
    flowed = " ".join(result.output.replace("│", " ").split())
    assert result.exit_code == 0, result.output
    assert "Run address: an id as `noctis runs` lists it" in flowed
    assert "The same four forms `run --resume` takes" in flowed
    assert "Omitted, this reads the reserved `legacy` run" in flowed


# ── reading a run costs the run nothing ────────────────────────────────────────────────────


@pytest.mark.parametrize("reader", EVERY_READER)
def test_reading_a_run_writes_nothing_into_it(tmp_path, reader):
    """A reading takes no lock and writes no byte: an operator may read a run an engine is
    working on, and the record they read is the record that was there."""
    cfg = _config(tmp_path)
    run_dir = _mint_run(tmp_path, cfg)
    reader.seed(run_dir, 1)
    before = (run_dir / "run.json").read_bytes()

    result = runner.invoke(app, [*reader.argv(run_dir.name), "--config", cfg])

    assert result.exit_code == reader.code, result.output
    assert (run_dir / "run.json").read_bytes() == before
    assert not (run_dir / "run.lock").exists()


# ── what an addressed run means, not just where it is (bug fix 2) ──────────────────────────


def test_backtest_defaults_its_symbol_from_the_addressed_runs_frozen_universe(tmp_path):
    """A run's universe is frozen at creation, so replaying one of its strategies picks the symbol
    the run itself was researching — not whatever ``config.yaml`` says today."""
    cfg = Path(_config(tmp_path))
    run_dir = _mint_run(tmp_path, str(cfg))
    cfg.write_text(cfg.read_text().replace("universe: [AAPL]", "universe: [MSFT]"))

    addressed = runner.invoke(app, ["backtest", "sma_crossover", run_dir.name, "-c", str(cfg)])
    bare = runner.invoke(app, ["backtest", "sma_crossover", "-c", str(cfg)])

    assert addressed.exit_code == 0, addressed.output
    assert "No catalog data for AAPL" in addressed.output  # the run's own universe
    assert bare.exit_code == 0, bare.output
    assert "No catalog data for MSFT" in bare.output  # today's file, for the reserved run


def test_backtest_scores_an_addressed_run_on_the_metric_that_run_was_steered_with(tmp_path):
    """The scorecard ``backtest`` prints is the one that promoted the champion: a run steered onto
    ``sortino`` replays on ``sortino`` even after the mandate is gone and ``config.yaml`` reads
    ``sharpe`` again. Frozen wins — the same rule ``--resume`` runs under."""
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns

    from ._data_helpers import MockVendor

    cfg = _steered_config(tmp_path)
    run_dir = _mint_run(tmp_path, str(cfg))
    # The config changes back under the run's feet; the lake it replays from is shared and live.
    cfg.write_text(Path(_config(tmp_path)).read_text())
    lake = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL"], to_ns("2026-01-01"), to_ns("2026-12-31")
    )

    addressed = runner.invoke(app, ["backtest", "sma_crossover", run_dir.name, "-c", str(cfg)])
    bare = runner.invoke(app, ["backtest", "sma_crossover", "-c", str(cfg)])

    assert addressed.exit_code == 0, addressed.output
    assert "metric:           sortino" in addressed.output
    assert bare.exit_code == 0, bare.output
    assert "metric:           sharpe" in bare.output


def test_champions_labels_an_addressed_runs_board_with_that_runs_own_metric(tmp_path):
    """The board is labelled against the metric the run was crowned under, so an addressed run's
    champions never read ``(stale)`` because the current file scores differently."""
    from noctis.backtest.scorecard import Scorecard
    from noctis.champions.registry import ChampionEntry, ChampionRegistry

    cfg = _steered_config(tmp_path)
    run_dir = _mint_run(tmp_path, str(cfg))
    registry = ChampionRegistry(run_dir / "state" / "champions.json", 3)
    registry.champions.append(
        ChampionEntry(
            family="sortino_winner",
            params={},
            scorecard=Scorecard(family="sortino_winner", params={}, metric_name="sortino"),
            crowned_at="2026-01-01",
            rationale="seed",
        )
    )
    registry.save()
    cfg.write_text(Path(_config(tmp_path)).read_text())  # the mandate is gone; the file says sharpe

    result = runner.invoke(app, ["champions", run_dir.name, "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert "sortino_winner" in result.output
    assert "sortino(stale)" not in result.output


# ── the one reader that narrates the mode (story #296) ─────────────────────────────────────


def test_status_reports_an_addressed_runs_frozen_universe(tmp_path):
    """``status`` addressed reports the run, not the file: its universe is the one frozen at
    creation, so the summary describes the experiment that ran rather than the config edited
    since."""
    cfg = Path(_config(tmp_path))
    run_dir = _mint_run(tmp_path, str(cfg))
    cfg.write_text(cfg.read_text().replace("universe: [AAPL]", "universe: [MSFT]"))

    addressed = runner.invoke(app, ["status", run_dir.name, "--config", str(cfg)])
    bare = runner.invoke(app, ["status", "--config", str(cfg)])

    assert addressed.exit_code == 0, addressed.output
    assert "universe:          AAPL" in addressed.output  # the run's own, frozen
    assert bare.exit_code == 0, bare.output
    assert "universe:          MSFT" in bare.output  # today's file, for the reserved run


def test_status_still_refuses_under_a_misconfigured_safety_gate(tmp_path, monkeypatch):
    """``status`` is the one reader that resolves the safety gate (D1), because it is the one that
    prints the resolved mode — and that gate is never degraded to a report line the way an
    unusable mandate is, addressed or bare: a mode line status had to guess at would be worse than
    no mode line at all."""
    monkeypatch.delenv("ALLOW_LIVE", raising=False)
    cfg = Path(_config(tmp_path))
    run_dir = _mint_run(tmp_path, str(cfg))
    cfg.write_text(cfg.read_text().replace("mode: paper", "mode: live"))

    for argv in (["status"], ["status", run_dir.name]):
        result = runner.invoke(app, [*argv, "--config", str(cfg)])
        assert result.exit_code == 1, result.output
        assert "SAFETY GATE" in result.output
        assert "mode (resolved)" not in result.output
