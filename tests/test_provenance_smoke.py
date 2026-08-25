"""End-to-end provenance smoke (story #187, closing epic #168) — the capture trail COMPOSES.

Each story in the epic proved its own slice: the store (#180), the served model (#181), the
gate-time scorecard (#182), the coder-site brief/spec capture (#184), the episodic briefing/knob
capture (#185), the ideation/distill prompt sidecars (#186). This module asks the only question
none of them can answer alone: after one ordinary stubbed research session, is the whole trail
**walkable from the records**?

That phrase is the contract under test. Every assertion here starts from an artifact a
post-mortem actually has — the session ledger file, the experiment journal, the run's ``qa/``
capture tree, the champion board on disk — reopened *by path*, never from a variable the test
kept. A hash is read out of a ledger row and handed to
:meth:`~noctis.observability.capture.CaptureStore.read`; the body that comes back is compared with
what the fake transport was really sent. If a future refactor rooted two stores differently, or
stopped stamping a hash onto its row, the walk would break here rather than in a stale post-mortem
six months later.

The other half of the contract is that none of it is load-bearing. Capture is observability, so a
latched (write-failing) store must leave the session itself byte-for-byte unharmed with the hash
fields simply *omitted*, and a ledger/journal written before the epic — carrying none of the new
fields — must still read cleanly through the typed views, reporting "not carried" (``None``) rather
than raising or inventing a value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from noctis.backtest.scorecard import Scorecard
from noctis.champions import ChampionRegistry
from noctis.observability.capture import CAPTURE_DIRNAME, CaptureStore
from noctis.research.distill import PROMPT_CAPTURE_KIND as DISTILL_PROMPT_KIND
from noctis.research.distill import bump_research_session, maybe_distill
from noctis.research.driver import (
    DECIDE_CONTRACT,
    FORMULATE_CONTRACT,
    make_episodes,
    run_episodic_research,
)
from noctis.research.episode import EpisodeRunner
from noctis.research.ideation import PROMPT_CAPTURE_KIND as IDEATION_PROMPT_KIND
from noctis.research.ideation import build_ideator
from noctis.research.journal import ExperimentJournal
from noctis.research.ledger import SESSIONS_DIRNAME, SessionLedger
from noctis.research.tools import (
    BRIEF_CAPTURE_KIND,
    BRIEFING_CAPTURE_KIND,
    KNOBS_CAPTURE_KIND,
    SPEC_CAPTURE_KIND,
)
from noctis.strategies import CandidateProposer
from noctis.strategies.scenario_spec import spec_from_json
from tests._capture_helpers import failing_capture_store
from tests.test_distill import FakeDistillClient
from tests.test_episodic_driver import (
    _FORMULATE_PAYLOAD,
    _REJECT_PAYLOAD,
    FakeCoder,
    FakeEpisodeClient,
    _emit,
)
from tests.test_ideation import FakeClient as FakeIdeationClient
from tests.test_ideation import _valid_spec
from tests.test_research_tools import _make_toolbox

# The candidate the shared FORMULATE payload derives (class tag "intraday momentum", first name).
CANDIDATE = "intraday_momentum_1"

# The verdict that carries the session all the way through a real gate arbitration: an approve
# routes DECIDE into tool_evaluate_vs_champion, which is the site that journals the scorecard
# (#182). The toolbox fixture's LENIENT rules make the promotion itself deterministic.
_APPROVE_PAYLOAD = dict(
    _REJECT_PAYLOAD, verdict="approve", reason="gross edge clears cost on the fit panel"
)

# The served ids the stub provider reports per episode — distinct on purpose, so a row that lost
# its served model would read as the other episode's rather than merely as empty.
_SERVED_FORMULATE = "stub-model-2026-04-01"
_SERVED_DECIDE = "stub-model-2026-06-14"


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """The smoke drives real gate *outcomes*; subprocess isolation is proven elsewhere."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _drive_session(tmp_path, session_id: str, *, capture=None):
    """One whole stubbed research session — formulate → match → author → optimize → decide —
    through the production ``make_episodes`` wiring, a real :class:`EpisodeRunner`, a real
    :class:`~noctis.research.tools.ResearchToolbox` and the real write/promotion gates.

    Only the two transports are fakes: the episode client replays scripted judgments and the coder
    returns a valid file. Everything the provenance trail is made of — the ledger, the journal, the
    capture store, the champion board — is the production code path.

    Returns the toolbox, the episode client (the external record of what was actually *sent*), and
    the summary.
    """
    box = _make_toolbox(tmp_path, coder_client=FakeCoder(), capture=capture)
    ledger = SessionLedger(box.state_dir, session_id=session_id)
    client = FakeEpisodeClient(
        [
            _emit(FORMULATE_CONTRACT.name, _FORMULATE_PAYLOAD, served_model=_SERVED_FORMULATE),
            _emit(DECIDE_CONTRACT.name, _APPROVE_PAYLOAD, served_model=_SERVED_DECIDE),
        ]
    )
    runner = EpisodeRunner(client=client, retries=2)
    formulate, decide, _discover = make_episodes(
        runner=runner, toolbox=box, ledger=ledger, mandate=None, context_window=10_000_000
    )
    summary = run_episodic_research(
        toolbox=box,
        ledger=ledger,
        formulate=formulate,
        decide=decide,
        fallback_panel_source=lambda: ["AAA", "BBB", "CCC"],
        budget_minutes=60.0,
        max_episodes=2,
        completions=lambda: runner.completions,
        sweep_trials=3,
    )
    return box, client, summary


def _artifacts(tmp_path, session_id: str):
    """Reopen the run's durable artifacts **by path alone** — the post-mortem's own view.

    Nothing here is threaded out of the session that wrote it: the ledger is reopened from its
    file, the journal from the state dir, the capture store from the run's ``qa/`` area and the
    champion board from ``champions.json``. If the trail is only walkable through live objects, it
    is not walkable.
    """
    state_dir = tmp_path / "state"
    return (
        SessionLedger.from_path(state_dir / SESSIONS_DIRNAME / f"{session_id}.jsonl"),
        ExperimentJournal(state_dir),
        CaptureStore(tmp_path / "qa" / CAPTURE_DIRNAME),
        ChampionRegistry(state_dir / "champions.json", capacity=3),
    )


# ── 1. the full trail: every hash-bearing row resolves to the body it names ──────────────────
def test_a_driven_session_leaves_a_capture_trail_walkable_from_the_records(tmp_path):
    """The epic's promise in one assertion set: after an ordinary session, an operator holding
    nothing but the run's files can start at a ledger row, read a hash out of it, and fetch the
    exact bytes that judgment was made on — the briefing each episode was sent, the brief the
    coder was given, the fixed oracle it was gated against — with the model that answered named
    beside each."""
    _box, client, summary = _drive_session(tmp_path, "smoke-trail")
    ledger, _journal, capture, _board = _artifacts(tmp_path, "smoke-trail")

    # The session really happened: a candidate was authored, optimized and gated.
    assert summary.candidates == [CANDIDATE] and summary.undecided == []
    assert [s.stage for s in ledger.stages()] == [
        "formulate",
        "match",
        "author",
        "optimize",
        "decide",
    ]

    # ── the episodes: each row names the question it was asked, and who answered it ──
    rows = ledger.episodes()
    assert [(r.stage, r.contract) for r in rows] == [
        ("formulate", FORMULATE_CONTRACT.name),
        ("decide", DECIDE_CONTRACT.name),
    ]
    assert [r.served_model for r in rows] == [_SERVED_FORMULATE, _SERVED_DECIDE]
    # Walk: hash off the row → body out of the capture tree → the exact string the transport got.
    sent = [messages[0]["content"] for messages in client.calls]
    assert [capture.read(BRIEFING_CAPTURE_KIND, r.input_sha256) for r in rows] == sent
    assert CANDIDATE in sent[1]  # the decide briefing really is this candidate's evidence
    # The knobs behind each call resolve too, and identical knobs name one shared body.
    knob_bodies = [json.loads(capture.read(KNOBS_CAPTURE_KIND, r.knobs_sha256)) for r in rows]
    assert knob_bodies[0] == knob_bodies[1]
    assert knob_bodies[0]["model"] == "fake/model"  # the alias the episodes asked for

    # ── the AUTHOR row: the brief the coder was given and the oracle it was gated against ──
    author = next(s for s in ledger.stages() if s.stage == "author")
    brief = json.loads(capture.read(BRIEF_CAPTURE_KIND, author.brief_sha256))
    assert brief["thesis"] == _FORMULATE_PAYLOAD["thesis"]
    assert brief["param_space"] == _FORMULATE_PAYLOAD["param_space_sketch"]
    spec = spec_from_json(capture.read(SPEC_CAPTURE_KIND, author.spec_sha256))
    # Two records, one fact: the captured spec's scenarios ARE the oracle the row names.
    assert [s.name for s in spec.scenarios] == author.detail["oracle"] == ["rally", "grind"]

    # Every sidecar the session wrote sits in ONE tree, under the four wired kinds and no other.
    assert sorted(p.name for p in capture.root.iterdir()) == [
        BRIEF_CAPTURE_KIND,  # author-brief
        SPEC_CAPTURE_KIND,  # author-spec
        BRIEFING_CAPTURE_KIND,  # episode-briefing
        KNOBS_CAPTURE_KIND,  # episode-knobs
    ]


def test_the_arbitrated_scorecard_is_journaled_beside_the_verdict_it_produced(tmp_path):
    """The evidence half of the trail (#182): the session's gate arbitration leaves the whole
    compact card in the journal, and it is the *same* card the champion board crowned — so the
    decision stays gradeable from two independent files long after the session ended."""
    _box, _client, summary = _drive_session(tmp_path, "smoke-scorecard")
    ledger, journal, _capture, board = _artifacts(tmp_path, "smoke-scorecard")

    assert summary.promotions == 1
    (verdict,) = ledger.verdicts()
    assert (verdict.strategy, verdict.verdict, verdict.promoted) == (CANDIDATE, "approve", True)

    records = [r for r in journal.records(CANDIDATE) if r.get("event") == "scorecard"]
    assert len(records) == 1  # exactly one per arbitration
    (entry,) = board.list()
    assert records[0]["scorecard"] == json.loads(json.dumps(entry.to_dict()["scorecard"]))
    # The journaled card reproduces the numbers the gates read, not a summary of them.
    card = Scorecard.from_dict(records[0]["scorecard"])
    assert card.family == CANDIDATE and card.params == entry.params
    assert card.symbol_holdout_metric is not None  # both holdout axes survived into the record


def test_ideation_and_distill_sidecars_land_in_the_runs_one_capture_tree(tmp_path):
    """The two sidecar-only sites (#186) write no ledger row, so their composition claim is about
    *place*: wired the way the composition root wires them, they default to the very tree the
    toolbox's own store writes to — one area under the run's ``qa/``, one cap over the lot — and
    each round's prompt is fetchable there by its content hash."""
    box = _make_toolbox(tmp_path)
    # A toolbox flow captures first, so the tree already belongs to this run's research session.
    hashes = box.capture_brief({"thesis": "buy strength", "param_space": "lookback 5-40"})

    # Ideation, built by the same helper the composition root calls (its capture root comes from
    # settings); only the provider transport is stubbed.
    ideator = build_ideator(
        settings=box.settings,
        registry=box.registry,
        families=box.families,
        proposer=CandidateProposer(box.families, seed=0),
        memory=box.memory,
        state_dir=box.state_dir,
    )
    ideator.client = FakeIdeationClient([_valid_spec("minted_in_the_smoke")])
    assert ideator.run(0) == ["minted_in_the_smoke"]

    # Distillation at CLOSE, through the periodic trigger's own default capture root.
    for i in range(12):
        box.memory.append_finding(f"REJECTED strategy s{i} — lesson {i}")
    box.settings.research.memory_distill_every = 1
    bump_research_session(box.settings.state_dir)
    distiller = FakeDistillClient("- a lesson")
    assert maybe_distill(box.settings, box.memory, client=distiller) is True

    # One tree, three kinds — reopened by path, exactly like a post-mortem would.
    capture = CaptureStore(tmp_path / "qa" / CAPTURE_DIRNAME)
    assert capture.root == box.capture.root
    assert sorted(p.name for p in capture.root.iterdir()) == [
        BRIEF_CAPTURE_KIND,
        DISTILL_PROMPT_KIND,
        IDEATION_PROMPT_KIND,
    ]
    ideation_sent = ideator.client.last_kwargs["messages"][0]["content"]
    assert capture.read(IDEATION_PROMPT_KIND, _sha256(ideation_sent)) == ideation_sent
    distill_sent = distiller.calls[0]["messages"][0]["content"]
    assert capture.read(DISTILL_PROMPT_KIND, _sha256(distill_sent)) == distill_sent
    assert "lesson 11" in distill_sent  # the history that was actually folded
    assert capture.read(BRIEF_CAPTURE_KIND, hashes["brief_sha256"]) is not None


# ── 2. graceful degradation: a latched store, and records written before the epic ────────────
def test_a_latched_capture_store_leaves_the_whole_session_unharmed(tmp_path):
    """Capture is strictly secondary at every site at once: with the store latched off by a disk
    failure, the same session still authors, optimizes and reaches a *promoted* verdict, the
    journal's evidence record is untouched (it is not capture), and every row simply OMITS its
    hash fields — never a name for a body that was never written, and never an exception."""
    capture_root = tmp_path / "qa" / CAPTURE_DIRNAME
    box, _client, summary = _drive_session(
        tmp_path, "smoke-latched", capture=failing_capture_store(capture_root)
    )
    ledger, journal, _capture, board = _artifacts(tmp_path, "smoke-latched")

    assert box.capture.root == capture_root and box.capture.disabled
    # The session is whole: a real candidate cleared the real gates onto the real board.
    assert summary.promotions == 1 and summary.undecided == []
    assert [e.family for e in board.list()] == [CANDIDATE]
    assert [r.get("event") for r in journal.records(CANDIDATE)].count("scorecard") == 1

    # Every hash field is absent from the raw records — "not carried", not empty.
    written = ledger.records()
    for record in [r for r in written if r.get("event") == "episode"]:
        assert not {"input_sha256", "knobs_sha256"} & set(record)
    for record in [r for r in written if r.get("stage") == "author"]:
        assert not {"brief_sha256", "spec_sha256"} & set(record)
    author = next(s for s in ledger.stages() if s.stage == "author")
    assert author.brief_sha256 is None and author.spec_sha256 is None
    rows = ledger.episodes()
    assert all(r.input_sha256 is None and r.knobs_sha256 is None for r in rows)
    # What never depended on capture still rides: the served model and the emit contract name
    # neither of them a body on disk.
    assert [r.served_model for r in rows] == [_SERVED_FORMULATE, _SERVED_DECIDE]
    assert [r.contract for r in rows] == [FORMULATE_CONTRACT.name, DECIDE_CONTRACT.name]


# One session's records as pre-#168 code would have written them: no served model, no capture
# hashes, no scorecard evidence — the shape every existing run on disk still has.
_PRE_EPIC_LEDGER: tuple[dict, ...] = (
    {
        "event": "session_start",
        "at": "2026-01-02T09:00:00+00:00",
        "mandate": "tune-first",
        "budgets": {"minutes": 60},
        "models": {"research": "old/model"},
    },
    {
        "event": "thesis",
        "at": "2026-01-02T09:00:01+00:00",
        "strategy": "legacy_probe",
        "thesis": "Fade the overnight gap when it clears the spread.",
    },
    {
        "event": "stage",
        "at": "2026-01-02T09:00:02+00:00",
        "stage": "author",
        "strategy": "legacy_probe",
        "detail": {"oracle": ["rally", "grind"]},
    },
    {
        "event": "episode",
        "at": "2026-01-02T09:00:03+00:00",
        "stage": "formulate",
        "model": "old/model",
        "tokens": 120,
        "misfires": 0,
        "outcome": "ok",
        "escalated": False,
    },
    {
        "event": "verdict",
        "at": "2026-01-02T09:00:04+00:00",
        "strategy": "legacy_probe",
        "verdict": "reject",
        "lesson": "no edge after cost",
        "promoted": False,
    },
    {
        "event": "session_end",
        "at": "2026-01-02T09:00:05+00:00",
        "formulated": 1,
        "promoted": 0,
        "rejected": 1,
    },
)

_PRE_EPIC_JOURNAL: tuple[dict, ...] = (
    {"event": "class_tag", "at": "2026-01-02T09:00:02+00:00", "class_tag": "overnight gap"},
    {
        "event": "thesis",
        "at": "2026-01-02T09:00:02+00:00",
        "thesis": "Fade the overnight gap when it clears the spread.",
    },
    {
        "event": "trial",
        "at": "2026-01-02T09:00:03+00:00",
        "source": "backtest",
        "strategy": "legacy_probe",
        "symbols": ["AAA"],
        "params": {"lookback": 10},
        "window": {},
        "metrics": {"stage": "validated", "metric_name": "sharpe", "test": 0.4},
    },
    {
        "event": "verdict",
        "at": "2026-01-02T09:00:04+00:00",
        "verdict": "reject",
        "reason": "no edge after cost",
        "best_params": {"lookback": 10},
    },
)


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8"
    )


def test_pre_epic_records_still_read_with_every_new_field_not_carried(tmp_path):
    """A run recorded before this epic keeps loading, and says so honestly: the typed views parse
    every line, and each field the epic added reads ``None`` — *not carried* — rather than an
    invented empty value or an exception. Additive-only, proven against fixture lines written the
    way the old code wrote them."""
    state_dir = tmp_path / "state"
    _write_jsonl(state_dir / SESSIONS_DIRNAME / "pre-epic.jsonl", _PRE_EPIC_LEDGER)
    _write_jsonl(state_dir / "experiments" / "legacy_probe.jsonl", _PRE_EPIC_JOURNAL)

    ledger = SessionLedger.from_path(state_dir / SESSIONS_DIRNAME / "pre-epic.jsonl")
    journal = ExperimentJournal(state_dir)

    # The ledger reads whole — every kind, and the derived views the CLOSE report renders.
    assert ledger.session_start() is not None and ledger.session_end() is not None
    assert [t.strategy for t in ledger.theses()] == ["legacy_probe"]
    rollup = ledger.rollup()
    assert rollup.theses == 1 and rollup.verdicts == {"reject": 1}
    assert ledger.candidate_trails()[0].oracle == ("rally", "grind")
    assert ledger.report_view() is not None

    # …with every field the epic added reading "not carried".
    (episode,) = ledger.episodes()
    assert episode.stage == "formulate" and episode.tokens == 120  # the old fields are intact
    assert episode.served_model is None  # #181
    assert (episode.input_sha256, episode.contract, episode.knobs_sha256) == (None, None, None)
    author = next(s for s in ledger.stages() if s.stage == "author")
    assert author.brief_sha256 is None and author.spec_sha256 is None  # #184

    # The journal likewise: its old records read, and the evidence record simply was never written.
    assert journal.strategies() == ["legacy_probe"]
    stats = journal.stats("legacy_probe")
    assert stats.n_trials == 1 and stats.n_distinct_params == 1
    assert journal.thesis("legacy_probe").text.startswith("Fade the overnight gap")
    assert journal.class_tag("legacy_probe") == "overnight gap"
    assert [r for r in journal.records("legacy_probe") if r.get("event") == "scorecard"] == []
