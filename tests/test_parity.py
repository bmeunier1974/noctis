"""The parity harness's metric computation (story #75) — the deterministic half.

Everything here runs on hand-built :class:`ResearchSummary` objects and real
:class:`SessionLedger` rollups: no LLM, no network, no paid runs. The paid dual-loop run is the
operator's explicit action in ``scripts/parity_harness.py``; these tests lock the pure metric math
the script prints — verdicts/session, tokens/verdict, the two gate-pass rates, undecided counts,
the side-by-side rendering (including every ``n/a`` path and the zero-verdict division safety), and
the flip-criterion assessment.
"""

from __future__ import annotations

from noctis.engine.research import ResearchSummary
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
) -> ResearchSummary:
    return ResearchSummary(
        promotions=promotions,
        rejections=rejections,
        tokens_total=tokens_total,
        candidates=list(candidates),
        undecided=list(undecided),
        ledger_path=ledger_path,
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


# ── validator first-attempt % (episodic: from the ledger; conversation: n/a) ───────────────────
def test_validator_first_attempt_pct_from_real_ledger_rollup(tmp_path):
    led = _episodic_ledger(tmp_path, "s1")
    rollup = led.rollup().to_dict()
    assert rollup["authored"] == 2 and rollup["validation_failures"] == 1
    m = compute_loop_metrics(EPISODIC, [(_summary(promotions=1, rejections=1), rollup)])
    # 2 of 3 author attempts passed the write gate on the first try.
    assert m.validator_first_attempt_pct == 200.0 / 3.0


def test_validator_first_attempt_pct_is_na_without_a_ledger():
    """The conversation loop writes no ledger, so the validator pass-rate is honestly unavailable —
    n/a, never invented."""
    m = compute_loop_metrics(CONVERSATION, [(_summary(promotions=1, rejections=1), None)])
    assert m.validator_first_attempt_pct is None


def test_validator_first_attempt_pct_is_na_with_no_author_attempts(tmp_path):
    led = SessionLedger(tmp_path, "empty")
    led.record_session_start(mandate=None, budgets={}, models={})
    m = compute_loop_metrics(EPISODIC, [(_summary(), led.rollup().to_dict())])
    assert m.validator_first_attempt_pct is None


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


def test_material_token_reduction_threshold_is_a_stated_fraction():
    assert 0.0 < MATERIAL_TOKEN_REDUCTION < 1.0
