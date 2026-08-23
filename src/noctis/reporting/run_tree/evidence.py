"""The run's evidence — the six reads, and every heavy import in the package (story #288).

A record's derived fields are **read, never counted**: how many trials the run journaled, what its
judgments cost, how many champions it holds, which candidates it considered, which sessions its
paper account closed and how an equal-weight hold of the same names would have done. Every one of
them comes off the run's own durable artifacts (its journals, its ledgers, its board, its strategy
tiers) or off the shared lake, so the numbers are cumulative across every segment by construction
and nothing has to survive a restart in memory.

That makes this the module with the *expensive* dependencies — the research package, the champion
registry, the broker's ledgers, the data types, the settings model, pandas — and it is the **only**
one allowed to name them. Every such import is deferred into the body that needs it, so the core
install imports the package (and writes a record) without the research extras;
``tests/test_run_tree_boundary.py`` refuses a heavy import anywhere else in the package, at any
nesting level, which is what turns "a record write stays cheap" from a comment into a shape.

:func:`derive_evidence` runs the six reads **once**, in one place, under the run's own frozen
inputs — spend is priced and sources are embedded the way the run was created, never the way this
process is configured. :func:`~noctis.reporting.run_tree.store.read_artifacts` is its other half
(the record, parsed); the store puts the two together at write time.

Nothing here raises: unreadable evidence is evidence we do not have, and a reporting artifact must
never be what fails a run's write.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from noctis.reporting.metrics import Benchmark, DailySession, TradeFill
from noctis.reporting.run_record import (
    EMBED_ALL_SOURCES_SETTING,
    EngineIdentity,
    SpendEntry,
    StrategyArtifact,
    utc_iso,
)
from noctis.reporting.run_tree.record import optional_str
from noctis.reporting.schema import (
    PROMOTED_OUTCOME,
    REJECTED_OUTCOME,
    UNDECIDED_OUTCOME,
)

__all__ = [
    "CHAMPIONS_NAME",
    "CHAMPIONS_TIER",
    "STRATEGIES_SUBDIR",
    "STRATEGY_TIER_SUBDIRS",
    "TMP_TIER",
    "Evidence",
    "derive_evidence",
    "read_benchmark",
    "read_champions",
    "read_engine_identity",
    "read_sessions",
    "read_spend",
    "read_strategies",
    "read_trials",
]


# The run's champion board, inside its own state dir — the denominator of the record's
# per-champion numbers. One of the two run-tree names this module owns (the strategy tiers below
# are the other); the record's, the lock's and the index's are in the modules that own those
# (``record.RUN_RECORD_NAME``, ``lock.RUN_LOCK_NAME``, ``index.RUN_INDEX_NAME``) — one place each,
# so nothing spells one by hand. The registry owns this file's schema.
CHAMPIONS_NAME = "champions.json"

# The run's own strategy library, and the two writable tiers inside it — the layout
# ``strategies.library.LibraryPaths.from_settings`` writes and this reads back for the record's
# strategies section (story #141). Spelled here rather than imported: the library module pulls
# pandas and numpy in behind it at *module* scope, and importing the package must stay as cheap as
# writing a record (the subprocess probe in ``tests/test_run_tree_store.py`` measures exactly
# that). One test asserts the two spellings still describe the same directories, which is where a
# drift would surface. The committed ``strategies/`` seeds are deliberately absent: they are
# read-only input every run starts from, so they are nobody's candidate.
STRATEGIES_SUBDIR = "strategies"
TMP_TIER = "__tmp"
CHAMPIONS_TIER = "champions"
# Lowest precedence first, the order the library itself discovers them in: a champion of the same
# name overrides a working-tier draft.
STRATEGY_TIER_SUBDIRS = (TMP_TIER, CHAMPIONS_TIER)


@dataclass(frozen=True)
class Evidence:
    """The seven derived fields of a record, read off the run's own artifacts — once.

    Frozen and shallow: it carries the values the collectors returned, in the shapes
    :class:`~noctis.reporting.run_record.RunArtifacts` holds them in, and knows how to say which
    fields it changes. It is not a second record shape — the builder still owns that.
    """

    trials: int | None
    spend: tuple[SpendEntry, ...] | None
    pricing_table_version: str | None
    champions: int | None
    strategies: tuple[StrategyArtifact, ...]
    sessions: tuple[DailySession, ...]
    benchmark: Benchmark | None

    def changes(self) -> dict[str, object]:
        """``{field name: value}`` — the changes :func:`store.with_evidence` applies.

        Deliberately **not** ``dataclasses.asdict``, which recurses: it would turn every
        ``SpendEntry``, ``StrategyArtifact`` and ``DailySession`` into a plain dict and hand the
        record builder something it cannot read.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


def derive_evidence(run_dir: Path | str, inputs: Mapping[str, object] | None) -> Evidence:
    """Read the run's own artifacts once, under the configuration the run froze at creation.

    ``inputs`` is the record's frozen ``inputs`` block (``None`` for a run that never froze one),
    and it is the reason this takes an argument at all: spend is priced under the run's **own**
    price table and sources are embedded under its own choice, so resuming tomorrow under a
    different ``config.yaml`` cannot restate what last night cost or drop what an earlier segment
    archived.

    One pass. Every caller that writes a record calls this exactly once for that write —
    ``RunStore._flush`` before its write, ``finish_run`` before it seals, ``prune_run_state``
    *before* it removes anything (the trial count is counted off the very journals it is about to
    delete). Opening a run does not call it at all: an open parses the record and writes what the
    flush derives.
    """
    trials = read_trials(run_dir)
    # Spend is derived beside the trial count and for the same reason, and it is priced under the
    # run's **own** frozen configuration — so resuming tomorrow under different prices cannot
    # restate what last night cost.
    spend, table_version = read_spend(run_dir, inputs)
    champions = read_champions(run_dir)
    # Candidates are read under the run's own frozen inputs too: whether a source is embedded is
    # the run's choice, made once at creation, not this process's.
    strategies = read_strategies(run_dir, inputs)
    # The realised record (story #142), derived like every cumulative fact beside it: the sessions
    # off the run's own account ledger, and the benchmark priced from the shared lake under the
    # run's own frozen data settings — a read, never a fetch.
    sessions = read_sessions(run_dir)
    return Evidence(
        trials=trials,
        spend=spend,
        pricing_table_version=table_version,
        champions=champions,
        strategies=strategies,
        sessions=sessions,
        benchmark=read_benchmark(sessions, inputs),
    )


def read_trials(run_dir: Path | str) -> int | None:
    """How many trials this run has journaled, or ``None`` when it has journaled nothing.

    **Read, never counted.** The number comes from the run's own experiment journals — the very
    lines the exhaustion gate counts (``<run>/state/experiments/<name>.jsonl``) — so the record and
    the research discipline can never disagree about how much searching a run did, and no counter
    has to survive a restart to be right. It is therefore cumulative across every segment by
    construction, including the research-only ones ``noctis research --resume`` appends (story
    #137): the journals are the run's, not the process's.

    Both imports are deferred, as every heavy import in this module is: the run tree is written on
    the core install alone and must stay importable without pulling the research package (or the
    settings model) in behind it. The
    journal owns the record schema end-to-end, so nothing here parses an ``event`` string, and the
    state directory is derived by the one function that owns that derivation.

    Never raises: an unreadable journal is missing evidence, not a reason to fail a run's write.
    """
    from noctis.config.settings import run_scoped_paths
    from noctis.research.journal import ExperimentJournal

    try:
        state_dir = run_scoped_paths(Path(run_dir))["state_dir"]
        totals = ExperimentJournal(state_dir).totals()
    except Exception:  # pragma: no cover - a journal we cannot read is evidence we do not have
        return None
    return None if totals is None else totals.n_trials


def read_spend(
    run_dir: Path | str, inputs: Mapping[str, object] | None = None
) -> tuple[tuple[SpendEntry, ...] | None, str | None]:
    """What this run has spent on model judgments, read off its own session ledgers (story #140).

    **Read, never counted** — the exact twin of :func:`read_trials`, and for the same reason: the
    ledgers under ``<run>/state/sessions/*.jsonl`` are the run's, not the process's, so a total
    summed from them is cumulative across every segment by construction and a rewrite after a crash
    cannot double-count it. One :class:`~noctis.reporting.run_record.SpendEntry` per journaled
    episode, priced here (pricing needs the table, and the table comes from the run's frozen
    configuration — both of which are this side of the I/O boundary), leaving the record builder
    with nothing but arithmetic.

    Returns ``(entries, table_version)``: ``(None, None)`` when the run journaled no ledger at all —
    the shape of a run with no LLM key, which must report an unknown bill rather than a free one —
    and an *empty* tuple when a session ran and spent nothing, which is a real zero.

    ``inputs`` is the run's frozen inputs, the only source of a price override: a run prices under
    the table it was created with, so resuming it tomorrow with a different ``research.pricing`` in
    ``config.yaml`` cannot restate what last night cost.

    Never raises: unreadable evidence is missing evidence, not a reason to fail a run's write.
    """
    from noctis.config.settings import run_scoped_paths
    from noctis.research.ledger import SESSIONS_DIRNAME, SessionLedger, episode_usage
    from noctis.research.pricing import table_from_config

    try:
        table = table_from_config(_price_overrides(inputs))
        state_dir = run_scoped_paths(Path(run_dir))["state_dir"]
        ledgers = sorted((Path(state_dir) / SESSIONS_DIRNAME).glob("*.jsonl"))
        if not ledgers:
            return None, None
        entries: list[SpendEntry] = []
        for path in ledgers:
            for episode in SessionLedger.from_path(path).episodes():
                usage = episode_usage(episode)
                entries.append(
                    SpendEntry(
                        at=episode.at or None,
                        stage=episode.stage,
                        model=episode.model,
                        tokens=episode.tokens,
                        usage=usage,
                        usd_estimate=table.estimate_usd(episode.model, usage),
                    )
                )
        return tuple(entries), table.version
    except Exception:  # pragma: no cover - a ledger we cannot read is evidence we do not have
        return None, None


def _price_overrides(inputs: Mapping[str, object] | None) -> Mapping[str, Mapping[str, object]]:
    """The run's own ``research.pricing`` block, read out of its frozen settings (or nothing).

    A plain read, deliberately tolerant: a record that carries no inputs, or carries something
    else in that slot, prices under the shipped table rather than failing a write.
    """
    node: object = inputs
    for key in ("settings", "resolved", "research", "pricing"):
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, Mapping) else {}  # type: ignore[return-value]


def read_champions(run_dir: Path | str) -> int | None:
    """How many champions this run currently holds, off its own board — or ``None`` if unreadable.

    The denominator of the record's two per-champion numbers, and read at write time like every
    other cumulative fact (epic D4). The *board*, not the promotion history: a champion that was
    displaced is no longer something this run has, and the board is the run's actual product.

    ``None`` — never ``0`` — when there is no board to read (a fresh run, a pruned one), because
    "no champions yet" and "nobody could look" are different claims and only one of them is a
    number. Capacity is irrelevant to a count, so the registry is opened with none.
    """
    from noctis.champions.registry import ChampionRegistry
    from noctis.config.settings import run_scoped_paths

    try:
        state_dir = run_scoped_paths(Path(run_dir))["state_dir"]
        board = Path(state_dir) / CHAMPIONS_NAME
        if not board.is_file():
            return None
        return len(ChampionRegistry(board, capacity=0).list())
    except Exception:  # pragma: no cover - an unreadable board is evidence we do not have
        return None


def read_strategies(
    run_dir: Path | str, inputs: Mapping[str, object] | None = None
) -> tuple[StrategyArtifact, ...]:
    """Every candidate this run considered, off its own board, journals and strategy tiers (#141).

    **Read, never counted** — the third of the same family as :func:`read_trials` and
    :func:`read_spend`, and for the same reason: all three sources are the *run's*, not the
    process's, so the section is cumulative across every segment by construction and a rewrite
    after a crash cannot double-count it. Three reads, joined on the candidate's name:

    * ``state/champions.json``'s decision history — the only durable trace a *rejected* candidate
      leaves, and since story #141 it carries each decision's structured gate evidence beside its
      prose rationale. The last decision journaled for a name is the decision of record;
    * ``state/experiments/<name>.jsonl`` — how many trials that candidate cost;
    * ``strategies/__tmp/`` and ``strategies/champions/`` — the file itself, for the path, the
      content hash, and (for a champion) the text.

    A name known to any one of them is a candidate. A file with no verdict is ``undecided``, which
    is a real state and not a gap: it is the width of the funnel above the gates.

    **The source policy lives here**, because it is a decision about what to open: a champion's
    file is embedded in full, every other candidate is referenced by a run-relative path plus its
    sha256, and a run created with ``embed_all_sources`` embeds them all. That is what holds a
    fortnight's record to :data:`~noctis.reporting.schema.RECORD_SIZE_BUDGET_BYTES` rather than a
    megabyte; the cost — a rejected candidate's code is readable only while the run's tree survives
    — is stated on the record itself by ``run.state_pruned``.

    ``inputs`` is the run's frozen inputs, the only source of that choice: it is fixed at creation
    like the compute cap beside it, so a resumed segment can never quietly drop the sources an
    earlier one embedded.

    Never raises, and the three reads degrade **independently**: an unparseable board costs the
    record its verdicts, not the candidates whose files are sitting right there.
    """
    run = Path(run_dir)
    decisions = _last_decisions(run)
    files = _strategy_files(run)
    trials = _journaled_trials(run)
    embed_all = _embed_all_sources(inputs)
    return tuple(
        _strategy_artifact(
            run,
            name,
            decision=decisions.get(name),
            located=files.get(name),
            trials=trials.get(name),
            embed_all=embed_all,
        )
        for name in sorted(set(decisions) | set(files) | set(trials))
    )


def _last_decisions(run_dir: Path) -> dict[str, Mapping[str, object]]:
    """The decision of record for each candidate: the **last** one the board journaled for it.

    A candidate may be judged more than once (tuned, re-run, re-considered), and a champion may
    later be dropped by a reset. The latest entry is what the run currently says about that name,
    and it carries its own rationale — so a reader is never left reconciling two verdicts.

    Each of the three reads behind the section degrades **on its own**: a board nobody can parse
    costs the record its verdicts, never the candidates whose files and journals are right there.
    Partial evidence is still evidence, and a record that dropped it all because one file was
    corrupt would be the least informative exactly when an operator most needs to look.
    """
    from noctis.champions.registry import ChampionRegistry

    latest: dict[str, Mapping[str, object]] = {}
    try:
        board = _state_dir(run_dir) / CHAMPIONS_NAME
        if not board.is_file():
            return {}
        for entry in ChampionRegistry(board, capacity=0).history:
            name = entry.get("family")
            if isinstance(name, str) and name:
                latest[name] = entry
    except Exception:  # a board we cannot read is a verdict we do not have
        return {}
    return latest


def _strategy_files(run_dir: Path) -> dict[str, tuple[str, Path]]:
    """Each candidate's file and the tier it sits in, later tiers overriding earlier ones."""
    located: dict[str, tuple[str, Path]] = {}
    for tier in STRATEGY_TIER_SUBDIRS:
        directory = run_dir / STRATEGIES_SUBDIR / tier
        if not directory.is_dir():
            continue
        try:
            paths = sorted(directory.glob("*.py"))
        except OSError:  # pragma: no cover - a tier we cannot list
            continue
        for path in paths:
            located[path.stem] = (tier, path)
    return located


def _journaled_trials(run_dir: Path) -> dict[str, int]:
    """How many trials each candidate cost, off its own experiment journal."""
    from noctis.research.journal import ExperimentJournal

    try:
        journal = ExperimentJournal(_state_dir(run_dir))
        return {name: journal.stats(name).n_trials for name in journal.strategies()}
    except Exception:  # pragma: no cover - an unreadable journal is evidence we do not have
        return {}


def _strategy_artifact(
    run_dir: Path,
    name: str,
    *,
    decision: Mapping[str, object] | None,
    located: tuple[str, Path] | None,
    trials: int | None,
    embed_all: bool,
) -> StrategyArtifact:
    """One candidate, assembled from whichever of the three sources knew about it."""
    outcome = UNDECIDED_OUTCOME
    gates: tuple[Mapping[str, object], ...] = ()
    rationale = None
    decided_utc = None
    if decision is not None:
        outcome = PROMOTED_OUTCOME if decision.get("promoted") else REJECTED_OUTCOME
        journaled = decision.get("gates")
        gates = (
            tuple(gate for gate in journaled if isinstance(gate, Mapping))
            if isinstance(journaled, Sequence) and not isinstance(journaled, str | bytes)
            else ()
        )
        rationale = optional_str(decision.get("rationale"))
        decided_utc = _record_stamp(decision.get("at"))
    tier, path = located if located is not None else (None, None)
    # A champion by *either* measure — the tier its file sits in, or the verdict the board
    # journaled — because the one source worth never losing is the run's own product.
    embed = embed_all or tier == CHAMPIONS_TIER or outcome == PROMOTED_OUTCOME
    return StrategyArtifact(
        name=name,
        outcome=outcome,
        tier=tier,
        decided_utc=decided_utc,
        trials=trials,
        gates=gates,
        rationale=rationale,
        source_path=path.relative_to(run_dir).as_posix() if path is not None else None,
        source_sha256=_file_sha256(path) if path is not None else None,
        source=_source_text(path) if path is not None and embed else None,
    )


def _file_sha256(path: Path) -> str | None:
    """The content hash of the file as stored — what a path reference is checkable against."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - a file that vanished between the glob and the read
        return None


def _source_text(path: Path) -> str | None:
    """A strategy file's text, or ``None`` when it cannot be read as the UTF-8 source it is.

    Deliberately strict: a lossy decode would embed text whose hash does not match the reference
    beside it, and a record that quotes something the file does not say is worse than one that
    quotes nothing and points at the path.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # pragma: no cover - not source after all
        return None


def _record_stamp(value: object) -> str | None:
    """One foreign ISO stamp in the record's own shape, or ``None`` if it is not a stamp.

    The champion board stamps its history with ``datetime.now(UTC).isoformat()``; the record's
    contract is UTC ISO-8601 with a ``Z``. Converting here — rather than teaching the board a
    second format — keeps one timestamp shape in the record without moving a byte in the arbiter.
    """
    if not isinstance(value, str):
        return None
    try:
        return utc_iso(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _embed_all_sources(inputs: Mapping[str, object] | None) -> bool:
    """Whether this run archives every candidate's source, off its own frozen settings.

    The same tolerant read as the price overrides beside it: a run that froze no configuration, or
    carries something else in that slot, gets the default — champions only.
    """
    node: object = inputs
    for key in ("settings", "resolved", EMBED_ALL_SOURCES_SETTING):
        node = node.get(key) if isinstance(node, Mapping) else None
    return bool(node)


def read_sessions(run_dir: Path | str) -> tuple[DailySession, ...]:
    """Every session this run's paper account closed, off its own daily ledger (story #142).

    **Read, never counted** — the same family as :func:`read_trials`, :func:`read_spend` and
    :func:`read_strategies`, and for the same reason: ``<run>/state/equity_curve.jsonl`` is the
    *run's*, not the process's, so the curve derived from it is cumulative across every segment by
    construction. Nothing about the equity curve is carried in memory across a restart; a resumed
    segment re-reads the whole ledger at every write, which is what makes "three short nights equal
    one long night" true of the curve rather than merely intended.

    One :class:`~noctis.reporting.metrics.DailySession` per session date (the ledger deduplicates,
    last write wins), oldest first. Never raises: an unreadable ledger is a curve we do not have,
    not a reason to fail a run's write.
    """
    from noctis.broker.persistence import EQUITY_CURVE_NAME, EquityLedger

    try:
        state_dir = _state_dir(Path(run_dir))
        marks = EquityLedger(state_dir / EQUITY_CURVE_NAME).marks()
    except Exception:  # pragma: no cover - an unreadable ledger is evidence we do not have
        return ()
    return tuple(_daily_session(mark) for mark in marks)


def _daily_session(mark: Mapping[str, object]) -> DailySession:
    """One journaled mark as the value the metrics module consumes."""
    positions = mark.get("positions_end")
    trades = mark.get("trades")
    return DailySession(
        date=str(mark.get("date", "")),
        equity=_number(mark.get("equity")) or 0.0,
        start_equity=_number(mark.get("start_equity")),
        end_equity=_number(mark.get("end_equity")),
        realized_pnl=_number(mark.get("realized_pnl")),
        orders_submitted=int(_number(mark.get("orders_submitted")) or 0),
        fills=tuple(
            _trade_fill(trade)
            for trade in (trades if isinstance(trades, list) else [])
            if isinstance(trade, Mapping)
        ),
        positions_end={
            str(symbol): float(quantity)
            for symbol, quantity in (positions if isinstance(positions, Mapping) else {}).items()
            if isinstance(quantity, int | float) and not isinstance(quantity, bool)
        },
    )


def _trade_fill(trade: Mapping[str, object]) -> TradeFill:
    """One journaled fill. The ledger stores the *report's* field names, which is deliberate — one
    trade shape is written at CLOSE and read back here, rather than two that could drift."""
    return TradeFill(
        ts=optional_str(trade.get("ts")),
        symbol=str(trade.get("symbol", "")),
        side=str(trade.get("side", "")),
        quantity=_number(trade.get("quantity")) or 0.0,
        price=_number(trade.get("price")) or 0.0,
        fees_usd=_number(trade.get("fees")) or 0.0,
        slippage_bps=_number(trade.get("slippage_bps")),
        champion=optional_str(trade.get("champion")),
        rationale=optional_str(trade.get("rationale")),
    )


# The bar schema the engine trades and researches on, and therefore the one the benchmark is priced
# from. Stated here rather than read from the frozen settings because it is not one of the keys the
# provenance block froze (``inputs.data`` carries provider/dataset/lake_dir); a symbol the lake
# holds under another schema simply yields no bars, and the benchmark degrades to a note.
BAR_SCHEMA = "ohlcv-1m"


def read_benchmark(
    sessions: Sequence[DailySession], inputs: Mapping[str, object] | None
) -> Benchmark:
    """Equal-weight buy-and-hold over the names this run traded, priced from the **shared lake**.

    The fair question a results page has to answer — did the strategy beat simply holding the names
    it traded? — computed with **no vendor call and no new spend**: the bars either are already in
    the workspace lake or they are not, and a symbol that is not is left out with a note rather
    than fetched. That is why this is a read and not an ``ensure_coverage``.

    The roster is derived from the run's own fills, the window from its own session dates, and the
    weights are set at the first session mark and never rebalanced (the convention is stated on the
    record so the comparison is reproducible). Daily levels are the last close of each UTC date; a
    symbol with no bar on a date carries its previous close forward, so one missing session cannot
    silently re-weight the basket.

    Never raises, and never reads a bar outside the run's own session window: a benchmark is
    evidence, and a record write must not fail on a parquet file it could not open.
    """
    symbols = sorted({fill.symbol for session in sessions for fill in session.fills if fill.symbol})
    dates = [session.date for session in sessions if session.date]
    if not symbols or not dates:
        return Benchmark(symbols=(), points=(), note="this run has traded nothing to benchmark")
    window = f"{dates[0]}…{dates[-1]}"
    try:
        closes = _lake_closes(symbols, dates, inputs)
    except Exception:  # pragma: no cover - an unreadable lake is a benchmark we do not have
        closes = {}
    points = _equal_weight_levels(closes, dates)
    if len(points) < 2:
        return Benchmark(
            symbols=tuple(symbols),
            points=(),
            note=(
                f"the shared lake holds no usable bars for {', '.join(symbols)} over {window}, so "
                "this run is not benchmarked — a benchmark is never worth a vendor fetch"
            ),
        )
    return Benchmark(symbols=tuple(symbols), points=tuple(points))


def _lake_closes(
    symbols: Sequence[str], dates: Sequence[str], inputs: Mapping[str, object] | None
) -> dict[str, dict[str, float]]:
    """``{symbol: {date: last close}}`` for the run's window, read straight off the catalog.

    Deferred imports, as everywhere in this module: the run tree is written on the core install
    and must stay importable without pulling pandas in behind it. Nothing here writes, fetches or
    creates a directory — a lake that is not there yields nothing.
    """
    import pandas as pd

    from noctis.data.types import SeriesKey

    data = inputs.get("data") if isinstance(inputs, Mapping) else None
    section: Mapping[str, object] = data if isinstance(data, Mapping) else {}
    lake_dir = Path(str(section.get("lake_dir") or "data_lake"))
    dataset = str(section.get("dataset") or "")
    if not dataset or not lake_dir.is_dir():
        return {}
    wanted = set(dates)
    closes: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        path = lake_dir / SeriesKey(dataset, BAR_SCHEMA, symbol).rel_path
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["ts_event", "close"])
        if frame.empty:
            continue
        stamps = pd.to_datetime(frame["ts_event"], unit="ns", utc=True)
        frame = frame.assign(session=stamps.dt.strftime("%Y-%m-%d"))
        # The run's own window and nothing else: bars from before the account opened or after its
        # last mark are never read into the comparison.
        frame = frame[frame["session"].isin(wanted)]
        if frame.empty:
            continue
        last = frame.groupby("session")["close"].last()
        closes[symbol] = {str(day): float(value) for day, value in last.items()}
    return closes


def _equal_weight_levels(
    closes: Mapping[str, Mapping[str, float]], dates: Sequence[str]
) -> list[tuple[str, float]]:
    """The equal-weight buy-and-hold level per session date, starting at 1.0.

    Weights are set on the first date the lake can price at all, and never rebalanced: each
    symbol's contribution is its own close over its base close, and the level is their mean. A
    symbol with no bar on a later date carries its last known close forward rather than dropping
    out, which would re-weight the basket without saying so.
    """
    priced = [day for day in dates if any(day in series for series in closes.values())]
    if not priced:
        return []
    base_day = priced[0]
    basis = {
        symbol: series[base_day] for symbol, series in closes.items() if series.get(base_day, 0) > 0
    }
    if not basis:
        return []
    carried = dict(basis)
    levels: list[tuple[str, float]] = []
    for day in priced:
        for symbol in basis:
            value = closes[symbol].get(day)
            if value is not None:
                carried[symbol] = value
        levels.append((day, sum(carried[s] / basis[s] for s in basis) / len(basis)))
    return levels


def read_engine_identity(election_metric: str, root: Path | None = None) -> EngineIdentity:
    """This engine's identity: the declared version, the per-component digests, the bucket key.

    Computed at every open (it reads source files, so it belongs on this side of the boundary),
    and stamped onto the segment as well as the run — a run resumed after a code change ran two
    engines and the record must be able to say so.
    """
    from noctis.observability.engine_id import comparable_key, fingerprint

    fp = fingerprint(root)
    return EngineIdentity(
        engine_version=fp.engine_version,
        fingerprint=fp.digests(),
        comparable_key=str(comparable_key(election_metric, fp)),
        noctis_version=_noctis_version(),
    )


def _state_dir(run_dir: Path) -> Path:
    from noctis.config.settings import run_scoped_paths

    return Path(run_scoped_paths(run_dir)["state_dir"])


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _noctis_version() -> str:
    """The package literal — informational beside the engine version, never a comparison key."""
    from importlib import metadata

    try:
        return metadata.version("noctis")
    except Exception:  # not pip-installed (editable/source tree)
        from noctis import __version__

        return __version__
