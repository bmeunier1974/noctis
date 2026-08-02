"""The parity harness's metric computation (story #75) — the deterministic half.

Everything here runs on hand-built :class:`ResearchSummary` objects and real
:class:`SessionLedger` rollups: no LLM, no network, no paid runs. The paid dual-loop run is the
operator's explicit action in ``scripts/parity_harness.py``; these tests lock the pure metric math
the script prints — verdicts/session, tokens/verdict, the two gate-pass rates, undecided counts,
the side-by-side rendering (including every ``n/a`` path and the zero-verdict division safety), and
the flip-criterion assessment.
"""

from __future__ import annotations

import pytest

from noctis.engine.research import PROSE_STALL, ResearchSummary
from noctis.research.ledger import SessionLedger
from noctis.research.parity import (
    CONVERSATION,
    EPISODIC,
    MATERIAL_TOKEN_REDUCTION,
    LoopMetrics,
    assess_flip,
    compute_loop_metrics,
    render_comparison,
    rollup_for,
)


def _summary(
    *,
    promotions: int = 0,
    rejections: int = 0,
    tokens_total: int = 0,
    candidates: tuple[str, ...] = (),
    undecided: tuple[str, ...] = (),
    ledger_path: str | None = None,
    stopped_reason: str = "",
) -> ResearchSummary:
    return ResearchSummary(
        promotions=promotions,
        rejections=rejections,
        tokens_total=tokens_total,
        candidates=list(candidates),
        undecided=list(undecided),
        ledger_path=ledger_path,
        stopped_reason=stopped_reason,
    )


def _episodic_ledger(tmp_path, sid: str) -> SessionLedger:
    """A real ledger: two strategies authored (reached OPTIMIZE), one author that failed the write
    gate (never optimized ⇒ a validation failure), plus two judgment episodes (12 + 8 tokens)."""
    led = SessionLedger(tmp_path, sid)
    led.record_session_start(mandate="m", budgets={}, models={})
    for name in ("a_1", "b_2"):
        led.record_thesis(name, "t")
        led.record_stage("author", strategy=name)
        led.record_stage("optimize", strategy=name, detail={"trials": 5})
    led.record_thesis("c_3", "t")
    led.record_stage("author", strategy="c_3")  # failed the write gate — no optimize
    led.record_episode(stage="formulate", model="drv", tokens=12, outcome="ok")
    led.record_episode(stage="decide", model="drv", tokens=8, outcome="ok")
    return led


# ── verdicts/session ─────────────────────────────────────────────────────────────────────────
def test_verdicts_per_session_counts_promotions_plus_rejections():
    sessions = [
        (_summary(promotions=1, rejections=1), None),
        (_summary(promotions=0, rejections=2), None),
    ]
    m = compute_loop_metrics(CONVERSATION, sessions)
    assert m.loop == CONVERSATION
    assert m.sessions == 2
    assert m.verdicts == 4  # (1+1) + (0+2)
    assert m.verdicts_per_session == 2.0


def test_verdicts_per_session_is_zero_for_no_sessions():
    m = compute_loop_metrics(EPISODIC, [])
    assert m.sessions == 0
    assert m.verdicts == 0
    assert m.verdicts_per_session == 0.0
    assert m.tokens_per_verdict is None


# ── tokens/verdict ───────────────────────────────────────────────────────────────────────────
def test_tokens_per_verdict_divides_total_tokens_by_verdicts():
    sessions = [
        (_summary(promotions=1, rejections=1, tokens_total=600), None),
        (_summary(promotions=0, rejections=2, tokens_total=200), None),
    ]
    m = compute_loop_metrics(EPISODIC, sessions)
    assert m.tokens_total == 800
    assert m.verdicts == 4
    assert m.tokens_per_verdict == 200.0


def test_tokens_per_verdict_is_na_when_no_verdicts():
    """Zero-verdict division safety: tokens spent but nothing decided ⇒ n/a, never a raise."""
    sessions = [(_summary(promotions=0, rejections=0, tokens_total=500), None)]
    m = compute_loop_metrics(CONVERSATION, sessions)
    assert m.tokens_total == 500
    assert m.verdicts == 0
    assert m.tokens_per_verdict is None


# ── validator job-pass % (episodic: from the ledger; conversation: n/a) ────────────────────────
def test_validator_job_pass_pct_from_real_ledger_rollup(tmp_path):
    led = _episodic_ledger(tmp_path, "s1")
    rollup = led.rollup().to_dict()
    assert rollup["authored"] == 2 and rollup["validation_failures"] == 1
    m = compute_loop_metrics(EPISODIC, [(_summary(promotions=1, rejections=1), rollup)])
    # 2 of 3 authoring JOBS ended with a file the write gate accepted — retries included.
    assert m.validator_job_pass_pct == 200.0 / 3.0


def test_validator_job_pass_pct_is_na_without_a_ledger():
    """The conversation loop writes no ledger, so the validator pass-rate is honestly unavailable —
    n/a, never invented."""
    m = compute_loop_metrics(CONVERSATION, [(_summary(promotions=1, rejections=1), None)])
    assert m.validator_job_pass_pct is None


def test_validator_job_pass_pct_is_na_with_no_author_attempts(tmp_path):
    led = SessionLedger(tmp_path, "empty")
    led.record_session_start(mandate=None, budgets={}, models={})
    m = compute_loop_metrics(EPISODIC, [(_summary(), led.rollup().to_dict())])
    assert m.validator_job_pass_pct is None


def test_the_validator_rate_is_named_for_the_job_level_thing_it_measures(tmp_path):
    """The metric row an operator reads says *job*, because a private retry still counts here."""
    rollup = _episodic_ledger(tmp_path, "named").rollup().to_dict()

    text = render_comparison(_conversation_metrics(), _episodic_metrics(rollup))

    assert "Validator job-pass %" in text
    assert "1st-attempt" not in text


# ── one definition, two layers: the parity row and the coder bench's job_pass rate ─────────────
def test_the_parity_validator_rate_and_the_coder_benchs_job_pass_rate_are_one_definition(tmp_path):
    """The same attempt-sequence facts, expressed in each layer's own shape, yield the same rate.

    Three authoring jobs: one landed on its opening ask, one landed after a private validator
    retry, one never landed. The parity row reads that off a ledger rollup (``authored`` vs
    ``validation_failures``); the coder benchmark reads it off retained job records. Both are
    job-level — a retry that landed is a pass on either side — so the two numbers must agree, and
    this test is what stops them drifting apart.
    """
    from noctis.eval.coder_scorer import score_coder_jobs
    from tests.test_eval_coder_scorer import _job

    attempts = {"a": [True], "b": [False, True], "c": [False, False]}
    landed = sum(1 for outcomes in attempts.values() if any(outcomes))
    rollup = {"authored": landed, "validation_failures": len(attempts) - landed}

    parity = compute_loop_metrics(EPISODIC, [(_summary(), rollup)])
    bench = score_coder_jobs([_job(case, outcomes) for case, outcomes in attempts.items()])

    assert bench.rates.job_pass_rate == 2.0 / 3.0
    assert parity.validator_job_pass_pct == pytest.approx(100.0 * bench.rates.job_pass_rate)


# ── promotion-gate reach % ─────────────────────────────────────────────────────────────────────
def test_promotion_gate_reach_pct_is_verdicts_over_candidates():
    sessions = [
        (_summary(promotions=1, rejections=1, candidates=("a", "b", "c", "d")), None),
    ]
    m = compute_loop_metrics(EPISODIC, sessions)
    # 2 gated verdicts out of 4 strategies worked on.
    assert m.promotion_gate_reach_pct == 50.0


def test_promotion_gate_reach_pct_is_na_without_candidates():
    m = compute_loop_metrics(CONVERSATION, [(_summary(), None)])
    assert m.promotion_gate_reach_pct is None


# ── undecided ────────────────────────────────────────────────────────────────────────────────
def test_undecided_total_sums_the_undecided_lists():
    sessions = [
        (_summary(undecided=("x", "y")), None),
        (_summary(undecided=("z",)), None),
    ]
    m = compute_loop_metrics(EPISODIC, sessions)
    assert m.undecided == 3


# ── prose stalls (#100) ──────────────────────────────────────────────────────────────────────
def test_prose_stalls_counts_only_stalled_sessions():
    """Sessions ended ``prose_stall`` (the conversation loop's zero-verdict prose ending past the
    nudge cap) are counted per loop; every other stop reason — including a deliberate post-verdict
    ``agent_done`` — is not."""
    sessions = [
        (_summary(tokens_total=500, stopped_reason=PROSE_STALL), None),
        (_summary(promotions=1, tokens_total=800, stopped_reason="agent_done"), None),
        (_summary(tokens_total=300, stopped_reason="time_budget"), None),
        (_summary(tokens_total=400, stopped_reason=PROSE_STALL), None),
    ]
    m = compute_loop_metrics(CONVERSATION, sessions)
    assert m.prose_stalls == 2


# ── rollup_for: load the episodic rollup from the summary's ledger path ────────────────────────
def test_rollup_for_reads_episodic_ledger_and_is_none_for_conversation(tmp_path):
    led = _episodic_ledger(tmp_path, "load-me")
    episodic_summary = _summary(ledger_path=str(led.path))
    rollup = rollup_for(episodic_summary)
    assert rollup is not None
    assert rollup["authored"] == 2 and rollup["validation_failures"] == 1
    # The conversation loop leaves ledger_path None ⇒ no rollup.
    assert rollup_for(_summary(ledger_path=None)) is None


# ── side-by-side rendering ─────────────────────────────────────────────────────────────────────
def _conversation_metrics() -> LoopMetrics:
    return compute_loop_metrics(
        CONVERSATION,
        [(_summary(promotions=1, rejections=1, tokens_total=4000, candidates=("a", "b")), None)],
    )


def _episodic_metrics(rollup) -> LoopMetrics:
    return compute_loop_metrics(
        EPISODIC,
        [
            (
                _summary(
                    promotions=1,
                    rejections=1,
                    tokens_total=1000,
                    candidates=("a", "b"),
                ),
                rollup,
            )
        ],
    )


def test_render_comparison_is_side_by_side_with_na_paths(tmp_path):
    rollup = _episodic_ledger(tmp_path, "r").rollup().to_dict()
    text = render_comparison(_conversation_metrics(), _episodic_metrics(rollup))
    # Both loops are columns.
    assert CONVERSATION in text and EPISODIC in text
    # The decision rows are named.
    assert "Verdicts / session" in text
    assert "Tokens / verdict" in text
    # The conversation loop cannot supply the validator pass-rate ⇒ n/a in its column.
    assert "n/a" in text
    # The stall count (#100) is a row, so a stalled model is visible on the same page.
    assert "Prose stalls (sessions)" in text
    # The flip criterion is stated in the output.
    assert "flip" in text.lower()


# ── flip criterion assessment ──────────────────────────────────────────────────────────────────
def test_flip_criterion_passes_when_verdicts_hold_and_tokens_materially_lower():
    conv = _summary(promotions=1, rejections=1, tokens_total=4000, candidates=("a", "b"))
    epi = _summary(promotions=1, rejections=1, tokens_total=1000, candidates=("a", "b"))
    a = assess_flip(
        compute_loop_metrics(CONVERSATION, [(conv, None)]),
        compute_loop_metrics(EPISODIC, [(epi, None)]),
    )
    assert a.verdicts_ok is True
    assert a.tokens_materially_lower is True
    assert a.meets_flip_criterion is True


def test_flip_criterion_fails_when_tokens_not_materially_lower():
    # Same verdicts/session, but only a ~10% token reduction — below the material threshold.
    conv = _summary(promotions=2, rejections=0, tokens_total=1000)
    epi = _summary(promotions=2, rejections=0, tokens_total=900)
    a = assess_flip(
        compute_loop_metrics(CONVERSATION, [(conv, None)]),
        compute_loop_metrics(EPISODIC, [(epi, None)]),
    )
    assert a.verdicts_ok is True
    assert a.tokens_materially_lower is False
    assert a.meets_flip_criterion is False


def test_flip_criterion_fails_when_episodic_has_fewer_verdicts():
    conv = _summary(promotions=2, rejections=2, tokens_total=4000)
    epi = _summary(promotions=1, rejections=0, tokens_total=500)
    a = assess_flip(
        compute_loop_metrics(CONVERSATION, [(conv, None)]),
        compute_loop_metrics(EPISODIC, [(epi, None)]),
    )
    assert a.verdicts_ok is False
    assert a.meets_flip_criterion is False


def test_flip_criterion_is_inconclusive_when_a_tokens_per_verdict_is_na():
    conv = _summary(promotions=1, rejections=1, tokens_total=4000)
    epi = _summary(promotions=0, rejections=0, tokens_total=500)  # no verdicts ⇒ tokens/verdict n/a
    a = assess_flip(
        compute_loop_metrics(CONVERSATION, [(conv, None)]),
        compute_loop_metrics(EPISODIC, [(epi, None)]),
    )
    assert a.tokens_materially_lower is None
    assert a.meets_flip_criterion is False


def test_inconclusive_names_the_stall_when_every_conversation_session_stalled():
    """The #100 signature — every conversation session prose-stalls while episodic keeps reaching
    verdicts — still cannot compute the table, but the verdict line says WHY and what the honest
    move is (prefer episodic on this model), instead of advising a re-run that will stall again."""
    conv_sessions = [
        (_summary(tokens_total=55_000, stopped_reason=PROSE_STALL), None),
        (_summary(tokens_total=34_000, stopped_reason=PROSE_STALL), None),
    ]
    epi = _summary(promotions=1, rejections=1, tokens_total=9_000, stopped_reason="max_episodes")
    a = assess_flip(
        compute_loop_metrics(CONVERSATION, conv_sessions),
        compute_loop_metrics(EPISODIC, [(epi, None)]),
    )
    assert a.meets_flip_criterion is False
    assert a.tokens_materially_lower is None
    assert "prose stall" in a.summary and "#100" in a.summary
    assert "episodic" in a.summary


def test_inconclusive_stays_generic_when_the_stall_is_not_universal():
    """One stalled session among working ones — or an episodic side with no verdicts either — is
    not the #100 signature: the generic re-run advice stands."""
    partial = [
        (_summary(tokens_total=1_000, stopped_reason=PROSE_STALL), None),
        (_summary(tokens_total=2_000, stopped_reason="time_budget"), None),
    ]
    epi = _summary(promotions=1, tokens_total=900)
    a = assess_flip(
        compute_loop_metrics(CONVERSATION, partial),
        compute_loop_metrics(EPISODIC, [(epi, None)]),
    )
    assert "Re-run" in a.summary

    # All-stalled conversation side, but zero episodic verdicts: "episodic is the only loop
    # reaching verdicts" would be false, so the generic message stands there too.
    stalled = [(_summary(tokens_total=1_000, stopped_reason=PROSE_STALL), None)]
    no_verdicts = _summary(promotions=0, rejections=0, tokens_total=500)
    b = assess_flip(
        compute_loop_metrics(CONVERSATION, stalled),
        compute_loop_metrics(EPISODIC, [(no_verdicts, None)]),
    )
    assert "Re-run" in b.summary


def test_material_token_reduction_threshold_is_a_stated_fraction():
    assert 0.0 < MATERIAL_TOKEN_REDUCTION < 1.0
