"""The coder site's runner integration (#225): throwaway tiers, the write gate as the scorer.

Every test here drives the *real* authoring engine over a scripted coder client, so what is
asserted is what a benchmark job really does: a file lands in a library nobody else can see, or an
:class:`~noctis.research.author.AuthoringError` falls out carrying the gate's own words. Nothing is
stubbed except the model — the write gate is never mocked, because the gate **is** this site's
score.

Four properties carry the story:

* **The gate is the verdict.** A known-good source passes; each known-bad class fails with the gate
  error retained on the job record, and no bad file is ever on disk under a library name.
* **A job is confined to the workdir the runner handed it.** The committed seeds are read in place
  and stay byte-identical (a job whose name collides with a seed still writes into its own working
  tier), and nothing outside the workdir is touched — no run tree, no workspace library.
* **Two levels of attempt, both visible.** One runner attempt is one authoring *job*; the engine's
  private retries (and an escalation to the paid fallback) land as per-attempt records in the job
  record the runner retains, which is what lets a later story compute first-attempt pass rates.
* **The site is live-askable.** ``bench run --site coder --dry-run`` preflights a coder corpus
  through the ask row, and a stubbed live run writes a bench record that validates.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

import noctis.research as research_mod
from noctis.config.settings import Settings
from noctis.eval.bootstrap import (
    BenchSeams,
    LiveModelUnavailable,
    cases_root,
    live_attempt,
    site_ask,
)
from noctis.eval.case import Case, parse_case
from noctis.eval.cli import run_bench
from noctis.eval.coder_distill_sites import CODER_SITE, NO_PROMPT, AuthoringJob, CoderKnobs
from noctis.eval.coder_site import (
    ESCALATED_PROMPT_NAME,
    JOB_RECORD_NAME,
    PROMPT_NAME,
    coder_attempt,
    coder_clients,
    job_library,
    job_record,
    resolve_coder_knobs,
    strategy_name,
)
from noctis.eval.harness import HarnessSpec
from noctis.eval.record import ModelConfig, validate
from noctis.eval.runner import BENCH_RECORD_NAME, Attempt, AttemptRequest, bench_root
from noctis.research.contract_sheet import CONTRACT_SHEET
from noctis.research.failed_store import FailedAttemptStore
from noctis.research.llm import Capabilities, Turn
from noctis.strategies import library
from tests.test_research_tools import PROBE

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = REPO_ROOT / "strategies"

CASE_ID = "canary-probe-case"
NAME = "canary_probe_case"

#: The fixed oracle a spec-path case carries, in the FORMULATE emit's own vocabulary.
FIXED_ORACLE: Mapping[str, Any] = {
    "scenarios": [
        {
            "name": "rally_runs",
            "legs": [{"kind": "flat", "bars": 20}, {"kind": "trend", "bars": 60, "pct": 0.10}],
            "behavior": "enter_long_during_leg",
            "leg": 1,
        },
        {
            "name": "steady_selloff_never_longs",
            "legs": [{"kind": "selloff", "bars": 80, "pct": 0.20}],
            "behavior": "never_trade",
        },
    ]
}


# ── the coder, scripted ───────────────────────────────────────────────────────────────────


class ScriptedCoder:
    """A coder client playing a fixed script of replies, reporting usage like a real backend."""

    def __init__(self, replies: Sequence[str], *, served_model: str = "bench/coder-2026") -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.capabilities = Capabilities()
        self.served_model = served_model

    def complete(
        self,
        *,
        system: Any,
        tools: list[dict],
        messages: list[dict],
        max_tokens: int,
        tool_choice: dict | None = None,
        on_delta: Any = None,
    ) -> Turn:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        if not self._replies:
            raise AssertionError("coder script exhausted — the job should have stopped")
        return Turn(
            text=self._replies.pop(0),
            tool_calls=[],
            stop_reason="end_turn",
            usage={
                "input_tokens": 900,
                "output_tokens": 120,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            served_model=self.served_model,
        )


def fenced(source: str) -> str:
    """Strategy source wrapped in a python code fence, as a coder reply carries it."""
    return f"Here is the file:\n```python\n{source}```\n"


def named(source_name: str) -> str:
    """PROBE re-pointed at a file name (its ``name`` attribute must match the file)."""
    return PROBE.replace('name = "probe"', f'name = "{source_name}"')


GOOD = fenced(named(NAME))
#: A class whose ``name`` disagrees with the file — the gate's most deterministic rejection.
MISNAMED = fenced(PROBE)
#: A reply that carries no code block at all.
PROSE = "I would rather describe the strategy than write it."
#: Source whose entry rule is inverted, so it fails replay of its own declared scenarios.
INVERTED = fenced(named(NAME).replace("int(bar.close > mean", "int(bar.close < mean"))


# ── the ask ───────────────────────────────────────────────────────────────────────────────


def _document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "site_id": "coder",
        "payload": {
            "thesis": "Long above a short moving average; the drift persists intraday.",
            "entry_exit": "Long when close > SMA(lookback); flat otherwise.",
            "param_space": "lookback int 5..40",
            "scenarios": "A rally pulls long; a steady decline stays flat.",
            "style": "momentum",
            "symbols": ["AAA", "BBB"],
        },
        "provenance": "authored:2026-08-02",
        "tags": ["bucket:canary"],
        "difficulty": {
            "composition_mode": "scratch",
            "oracle_mode": "authored",
            "warmup_arithmetic": "single",
            "state_complexity": "rolling",
            "no_trade_tape": "falsified",
            "param_space_breadth": "narrow",
            "api_surface": "indicators",
        },
        "split": "tuning",
    }
    for key, value in overrides.items():
        document[key] = value
    return document


def _case(case_id: str = CASE_ID, **overrides: Any) -> Case:
    """One coder case, validated by the eval core exactly as a corpus file is."""
    return parse_case(_document(**overrides), case_id=case_id, source=f"{case_id}.yaml")


def _settings(tmp_path: Path, **agent: Any) -> Settings:
    settings = Settings()
    settings.workspace_dir = str(tmp_path / "workspace")
    for knob, value in agent.items():
        setattr(settings.research.agent, knob, value)
    return settings


def _request(
    tmp_path: Path, case: Case | None = None, *, harness: HarnessSpec | None = None
) -> AttemptRequest:
    """One attempt request of the shape the bench runner hands the attempt callable."""
    return AttemptRequest(
        prompt="(the site's rendered preview)",
        case=case if case is not None else _case(),
        config=ModelConfig(config_id="default"),
        harness=harness if harness is not None else HarnessSpec(),
        rep=1,
        attempt=1,
        workdir=tmp_path / "bench" / "cases" / CASE_ID / "default" / "rep-01" / "work",
    )


def _run(
    tmp_path: Path,
    client: ScriptedCoder,
    *,
    case: Case | None = None,
    harness: HarnessSpec | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> tuple[Attempt, AttemptRequest]:
    """One whole authoring job, through the attempt callable the bench runner would call."""
    request = _request(tmp_path, case, harness=harness)
    attempt = coder_attempt(
        settings if settings is not None else _settings(tmp_path),
        client=client,
        model="bench/coder",
        seeds=SEEDS,
        **kwargs,
    )
    return attempt(request), request


def _one_shot(tmp_path: Path) -> Settings:
    """Settings whose coder gets exactly one attempt — a failure class, asked once."""
    return _settings(tmp_path, coder_retries=0)


def _record(answer: Attempt) -> dict[str, Any]:
    """The job record the attempt retained, as a later scorer reads it back."""
    return json.loads(answer.output or "{}")


def _tree(root: Path) -> dict[str, str]:
    """Every file under ``root``, keyed by relative path, with the sha256 of its bytes."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ── the write gate is the score ───────────────────────────────────────────────────────────


def test_a_source_the_write_gate_accepts_lands_a_file_and_passes_the_job(tmp_path, fast_gate):
    answer, request = _run(tmp_path, ScriptedCoder([GOOD]))

    assert answer.outcome.passed is True
    assert (request.workdir / "strategies" / library.TMP_SUBDIR / f"{NAME}.py").is_file()


def test_a_class_whose_name_disagrees_with_the_file_fails_the_job_with_the_gates_own_words(
    tmp_path, fast_gate
):
    answer, _ = _run(tmp_path, ScriptedCoder([MISNAMED]), settings=_one_shot(tmp_path))

    assert answer.outcome.passed is False
    assert "name" in (answer.outcome.error or "")
    assert _record(answer)["error"]


def test_a_reply_carrying_no_code_block_fails_the_job_and_retains_why(tmp_path, fast_gate):
    answer, _ = _run(tmp_path, ScriptedCoder([PROSE]), settings=_one_shot(tmp_path))

    assert answer.outcome.passed is False
    assert "code block" in (answer.outcome.error or "")


def test_source_that_violates_its_own_declared_scenarios_fails_the_job(tmp_path, fast_gate):
    answer, _ = _run(tmp_path, ScriptedCoder([INVERTED]), settings=_one_shot(tmp_path))

    assert answer.outcome.passed is False
    assert _record(answer)["error"]


def test_a_rejected_job_leaves_no_file_on_disk_under_the_library_name(tmp_path, fast_gate):
    answer, request = _run(tmp_path, ScriptedCoder([MISNAMED]), settings=_one_shot(tmp_path))

    assert answer.outcome.passed is False
    assert not (request.workdir / "strategies" / library.TMP_SUBDIR / f"{NAME}.py").exists()


def test_a_case_carrying_a_fixed_oracle_is_gated_against_the_stamped_spec(tmp_path, fast_gate):
    """The spec is forwarded to the gate, which stamps the oracle and refuses authored tapes."""
    case = _case(
        payload={**_document()["payload"], "scenario_spec": FIXED_ORACLE},
        difficulty={**_document()["difficulty"], "oracle_mode": "fixed_spec"},
    )

    answer, _ = _run(tmp_path, ScriptedCoder([GOOD]), case=case, settings=_one_shot(tmp_path))

    assert answer.outcome.passed is False
    assert "machine-stamped by the gate" in (answer.outcome.error or "")


def test_a_brief_naming_a_reference_the_library_lacks_fails_without_spending_a_completion(
    tmp_path, fast_gate
):
    """A bad brief is not a coder failure, so nobody is asked — the paid coder least of all."""
    local = ScriptedCoder([GOOD])
    paid = ScriptedCoder([GOOD])
    case = _case(
        payload={**_document()["payload"], "reference": "kama_adaptive_trend"},
        difficulty={**_document()["difficulty"], "composition_mode": "reference"},
    )

    answer, _ = _run(
        tmp_path,
        local,
        case=case,
        settings=_settings(tmp_path, max_escalations=1),
        fallback_client=paid,
        fallback_model="bench/paid-coder",
    )

    assert answer.outcome.passed is False
    assert local.calls == [] and paid.calls == []
    assert "not in the strategy library" in (answer.outcome.error or "")
    assert _record(answer)["attempts"] == []


def test_a_brief_the_engine_refuses_retains_the_refusal_where_its_prompt_would_have_been(
    tmp_path, fast_gate
):
    """There are no bytes to keep, so the artifact says so rather than inventing a prompt."""
    case = _case(
        payload={**_document()["payload"], "reference": "kama_adaptive_trend"},
        difficulty={**_document()["difficulty"], "composition_mode": "reference"},
    )

    _, request = _run(tmp_path, ScriptedCoder([GOOD]), case=case)

    kept = (request.workdir / PROMPT_NAME).read_text(encoding="utf-8")
    assert kept.startswith(NO_PROMPT)
    assert "kama_adaptive_trend" in kept


# ── isolation: the job owns its workdir and nothing else ──────────────────────────────────


def test_a_job_writes_only_inside_the_workdir_the_runner_handed_it(tmp_path, fast_gate):
    workspace = tmp_path / "workspace"
    (workspace / "runs" / "legacy" / "state").mkdir(parents=True)
    (workspace / "runs" / "legacy" / "run.json").write_text("{}", encoding="utf-8")
    before = _tree(workspace)

    _, request = _run(tmp_path, ScriptedCoder([GOOD]))

    assert _tree(workspace) == before
    assert any(request.workdir.rglob("*.py"))


def test_the_committed_seed_library_is_byte_identical_after_a_job(tmp_path, fast_gate):
    before = _tree(SEEDS)

    _run(tmp_path, ScriptedCoder([GOOD]))

    assert _tree(SEEDS) == before


def test_a_job_authoring_under_a_committed_seeds_own_name_writes_into_its_working_tier(
    tmp_path, fast_gate
):
    """The tier design, exercised: a seed is read-only input, so a rewrite lands in ``__tmp``."""
    before = _tree(SEEDS)
    seed = "sma_crossover"

    answer, request = _run(
        tmp_path, ScriptedCoder([fenced(named(seed))]), case=_case("sma-crossover")
    )

    assert answer.outcome.passed is True
    assert _tree(SEEDS) == before
    assert (request.workdir / "strategies" / library.TMP_SUBDIR / f"{seed}.py").is_file()


def test_two_jobs_in_one_bench_cannot_see_each_others_working_tiers(tmp_path, fast_gate):
    first = job_library(tmp_path / "one", seeds=SEEDS)
    second = job_library(tmp_path / "two", seeds=SEEDS)

    assert first.tmp != second.tmp and first.champions != second.champions
    assert first.seeds == second.seeds == SEEDS
    assert first.tmp.is_dir() and first.champions.is_dir()


def test_the_strategy_a_case_is_authored_under_is_named_by_its_own_case_id():
    assert strategy_name("canary-probe-case") == "canary_probe_case"
    assert library.NAME_RE.match(strategy_name("2-odd.case id"))


# ── the per-attempt records ───────────────────────────────────────────────────────────────


def test_every_internal_attempt_lands_one_record_carrying_its_outcome(tmp_path, fast_gate):
    answer, _ = _run(tmp_path, ScriptedCoder([MISNAMED, GOOD]))

    attempts = _record(answer)["attempts"]
    assert [one["attempt"] for one in attempts] == [1, 2]
    assert [one["passed"] for one in attempts] == [False, True]
    assert attempts[0]["error"] and attempts[1]["error"] is None


def test_each_attempt_record_carries_the_usage_and_the_model_that_served_it(tmp_path, fast_gate):
    answer, _ = _run(tmp_path, ScriptedCoder([GOOD]))

    (attempt,) = _record(answer)["attempts"]
    assert attempt["usage"]["input_tokens"] == 900
    assert attempt["served_model"] == "bench/coder-2026"
    assert attempt["model"] == "bench/coder"
    assert attempt["escalated"] is False


def test_the_job_record_the_runner_retains_is_written_beside_the_job_as_well(tmp_path, fast_gate):
    answer, request = _run(tmp_path, ScriptedCoder([MISNAMED, GOOD]))

    beside = json.loads((request.workdir / JOB_RECORD_NAME).read_text(encoding="utf-8"))
    assert beside == _record(answer)


def test_the_job_spends_the_usage_of_every_internal_attempt_it_made(tmp_path, fast_gate):
    answer, _ = _run(tmp_path, ScriptedCoder([MISNAMED, GOOD]))

    assert answer.outcome.input_tokens == 1800
    assert answer.served_model == "bench/coder-2026"


def test_every_rejected_attempt_is_retained_with_its_source_and_gate_error(tmp_path, fast_gate):
    answer, request = _run(tmp_path, ScriptedCoder([MISNAMED, GOOD]))

    store = FailedAttemptStore(request.workdir / "strategies" / library.TMP_SUBDIR / "failed")
    (kept,) = sorted(store.root.glob("*.py"))
    assert "class Probe" in kept.read_text(encoding="utf-8")
    assert _record(answer)["attempts"][0]["artifact"] == str(kept.relative_to(request.workdir))


def test_the_prompt_the_engine_actually_sent_is_retained_beside_the_job(tmp_path, fast_gate):
    client = ScriptedCoder([GOOD])

    _, request = _run(tmp_path, client)

    kept = (request.workdir / PROMPT_NAME).read_text(encoding="utf-8")
    sent = client.calls[0]
    assert kept == f"{sent['system']}\n\n{sent['messages'][0]['content']}"


# ── the retry budget, and escalation ──────────────────────────────────────────────────────


def test_the_configured_retry_budget_is_the_number_of_attempts_a_job_may_make(tmp_path, fast_gate):
    client = ScriptedCoder([MISNAMED, MISNAMED, MISNAMED])

    answer, _ = _run(tmp_path, client, settings=_settings(tmp_path, coder_retries=0))

    assert answer.outcome.passed is False
    assert len(client.calls) == 1


def test_a_bench_override_of_the_retry_budget_wins_over_the_configured_one(tmp_path, fast_gate):
    client = ScriptedCoder([MISNAMED, MISNAMED, GOOD])

    answer, _ = _run(
        tmp_path,
        client,
        settings=_settings(tmp_path, coder_retries=0),
        harness=HarnessSpec(knob_overrides={"coder_retries": 2}),
    )

    assert answer.outcome.passed is True
    assert len(client.calls) == 3


def test_the_coder_knob_layering_puts_a_bench_override_above_the_configured_value(tmp_path):
    settings = _settings(tmp_path, coder_retries=1, coder_max_tokens=4096)

    knobs = resolve_coder_knobs(settings, HarnessSpec(knob_overrides={"coder_retries": 4}))

    assert knobs.coder_retries == 4
    assert knobs.coder_max_tokens == 4096


def test_a_spent_local_author_escalates_to_the_paid_fallback_when_the_cap_allows_it(
    tmp_path, fast_gate
):
    local = ScriptedCoder([MISNAMED], served_model="local/coder-2026")
    paid = ScriptedCoder([GOOD], served_model="paid/coder-2026")

    answer, _ = _run(
        tmp_path,
        local,
        settings=_settings(tmp_path, coder_retries=0, max_escalations=1),
        fallback_client=paid,
        fallback_model="bench/paid-coder",
    )

    attempts = _record(answer)["attempts"]
    assert answer.outcome.passed is True
    assert [one["escalated"] for one in attempts] == [False, True]
    assert attempts[1]["model"] == "bench/paid-coder"
    assert _record(answer)["escalated"] is True


def test_an_escalated_job_retains_the_prompt_the_paid_coder_was_sent(tmp_path, fast_gate):
    paid = ScriptedCoder([GOOD])

    _, request = _run(
        tmp_path,
        ScriptedCoder([MISNAMED]),
        settings=_settings(tmp_path, coder_retries=0, max_escalations=1),
        fallback_client=paid,
        fallback_model="bench/paid-coder",
    )

    kept = (request.workdir / ESCALATED_PROMPT_NAME).read_text(encoding="utf-8")
    assert kept == f"{paid.calls[0]['system']}\n\n{paid.calls[0]['messages'][0]['content']}"


def test_a_spent_escalation_cap_leaves_the_paid_coder_untouched(tmp_path, fast_gate):
    paid = ScriptedCoder([GOOD])

    answer, _ = _run(
        tmp_path,
        ScriptedCoder([MISNAMED]),
        settings=_settings(tmp_path, coder_retries=0, max_escalations=0),
        fallback_client=paid,
        fallback_model="bench/paid-coder",
    )

    assert answer.outcome.passed is False
    assert paid.calls == []
    assert _record(answer)["escalated"] is False


# ── the ablation dials reach the engine that authors ──────────────────────────────────────


def test_the_harness_dials_take_their_pieces_out_of_the_prompt_the_job_sends(tmp_path, fast_gate):
    client = ScriptedCoder([GOOD])

    _run(
        tmp_path,
        client,
        harness=HarnessSpec(
            contract_sheet=False, template=False, worked_example=None, feasibility_rules=False
        ),
    )

    system = client.calls[0]["system"]
    assert CONTRACT_SHEET not in system
    assert "COMPLETE WORKED EXAMPLE" not in system and "TEMPLATE.py" not in system


def test_the_production_harness_composes_the_prompt_a_research_session_sends(tmp_path, fast_gate):
    client = ScriptedCoder([GOOD])

    _run(tmp_path, client)

    system = client.calls[0]["system"]
    assert CONTRACT_SHEET in system and "COMPLETE WORKED EXAMPLE" in system


# ── the site's input adapter, and the live ask row ────────────────────────────────────────


def test_the_coder_ask_row_adapts_a_case_into_the_authoring_job_its_renderer_takes(tmp_path):
    settings = _settings(tmp_path)
    settings.strategies_dir = str(SEEDS)

    adapt = site_ask("coder").site_input(settings)
    job = adapt(_case())

    assert isinstance(job, AuthoringJob)
    assert job.name == NAME
    rendered = CODER_SITE.render(job, HarnessSpec())
    assert "COMPLETE WORKED EXAMPLE" in rendered and "drift persists intraday" in rendered


def test_the_coder_ask_row_quotes_the_role_framing_production_ships():
    ask = site_ask("coder")

    assert ask.system and "strategy-authoring coder" in ask.system
    assert ask.contract is None


def test_a_live_coder_bench_is_asked_through_the_sites_own_maker_not_a_forced_emit(tmp_path):
    """The refusal names the *coder* client, so the dispatch really went through the ask row."""
    with pytest.raises(LiveModelUnavailable) as refusal:
        live_attempt(_settings(tmp_path), site_id="coder", model="anthropic/claude-sonnet-4")

    assert "coder client" in str(refusal.value)
    assert "anthropic/claude-sonnet-4" in str(refusal.value)


def test_a_live_coder_bench_authors_through_the_client_it_was_handed(tmp_path, fast_gate):
    settings = _settings(tmp_path)
    settings.strategies_dir = str(SEEDS)
    attempt = live_attempt(
        settings, site_id="coder", model="bench/coder", client=ScriptedCoder([GOOD])
    )

    answer = attempt(_request(tmp_path))

    assert answer.outcome.passed is True
    assert _record(answer)["strategy"] == NAME


def test_an_injected_client_is_the_coder_a_bench_asks_and_no_fallback_is_invented(tmp_path):
    injected = ScriptedCoder([GOOD])

    clients = coder_clients(_settings(tmp_path), model="bench/coder", client=injected)

    assert clients.client is injected and clients.model == "bench/coder"
    assert clients.fallback is None and clients.fallback_model is None


def test_a_bench_asks_the_coder_production_builds_dials_and_all(tmp_path, monkeypatch):
    """A live coder bench runs on the engine's own coder clients: the configured model on the
    coder's dials (thinking, deliberate, sampling) and the paid fallback its budget allows —
    a bench that asked the same model without them would measure a configuration nobody ships."""
    asked: dict[str, dict] = {}

    def fake_client_for(settings, model, **kwargs):
        asked[model] = kwargs
        return ScriptedCoder([GOOD])

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _settings(
        tmp_path,
        coder_model="ollama/qwen3-coder",
        coder_fallback_model="anthropic/claude-opus-4-8",
        max_escalations=1,
    )

    clients = coder_clients(settings)

    assert clients.model == "ollama/qwen3-coder" and clients.client is not None
    assert clients.fallback is not None
    assert clients.fallback_model == "anthropic/claude-opus-4-8"
    assert asked["ollama/qwen3-coder"]["thinking"] == "on"
    assert asked["ollama/qwen3-coder"]["deliberate"] is True
    assert asked["anthropic/claude-opus-4-8"]["thinking"] == "off"


def test_a_bench_with_no_escalation_budget_builds_no_paid_fallback(tmp_path, monkeypatch):
    """``max_escalations: 0`` (the shipped default) is a bench that will never escalate, so the
    paid provider is never consulted — the escalation cap gates the client, not just the call."""
    asked: dict[str, dict] = {}

    def fake_client_for(settings, model, **kwargs):
        asked[model] = kwargs
        return ScriptedCoder([GOOD])

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _settings(
        tmp_path,
        coder_model="ollama/qwen3-coder",
        coder_fallback_model="anthropic/claude-opus-4-8",
    )

    clients = coder_clients(settings)

    assert clients.fallback is None and clients.fallback_model is None
    assert "anthropic/claude-opus-4-8" not in asked


def test_a_bench_override_of_the_escalation_cap_buys_back_the_fallback(tmp_path, monkeypatch):
    """The knob set a bench resolved (site defaults → config → harness override) is what the
    clients are built from, so an override of the cap reaches the fallback client."""
    monkeypatch.setattr(research_mod, "client_for", lambda *a, **k: ScriptedCoder([GOOD]))
    settings = _settings(tmp_path, coder_model="ollama/qwen3-coder")
    knobs = resolve_coder_knobs(
        settings,
        HarnessSpec(
            knob_overrides={
                "coder_fallback_model": "anthropic/claude-opus-4-8",
                "max_escalations": 2,
            }
        ),
    )

    clients = coder_clients(settings, knobs=knobs)

    assert clients.fallback is not None
    assert clients.fallback_model == "anthropic/claude-opus-4-8"


def test_a_coder_the_engine_cannot_build_refuses_instead_of_degrading(tmp_path, monkeypatch):
    """The bench's stricter contract on top of the engine's graceful degradation: a session would
    assemble in driver-authored mode, but a coder bench has nothing to measure without a model."""
    monkeypatch.setattr(research_mod, "client_for", lambda *a, **k: None)

    with pytest.raises(LiveModelUnavailable) as refusal:
        coder_clients(_settings(tmp_path, coder_model="ollama/qwen3-coder"))

    assert "no coder client for 'ollama/qwen3-coder'" in str(refusal.value)


def test_a_job_record_reads_back_from_the_document_it_was_written_as(tmp_path, fast_gate):
    """The retained format has exactly one reader, so a round trip is what keeps it honest."""
    answer, _ = _run(tmp_path, ScriptedCoder([MISNAMED, GOOD]))

    read = job_record(_record(answer))

    assert read.document() == _record(answer)
    assert read.first_attempt_passed is False and read.passed is True


# ── the verb, over a corpus ───────────────────────────────────────────────────────────────


def _corpus(tmp_path: Path, *, cases: int = 2) -> Path:
    """A small coder corpus in the workspace's cases root, in the bucket layout the site reads."""
    directory = cases_root(tmp_path / "workspace") / "coder" / "canary"
    directory.mkdir(parents=True)
    for index in range(cases):
        (directory / f"canary-probe-{index}.yaml").write_text(
            yaml.safe_dump(_document(), sort_keys=True), encoding="utf-8"
        )
    return directory


def test_a_dry_run_over_the_coder_site_preflights_its_whole_corpus(tmp_path, capsys):
    _corpus(tmp_path)

    run_bench("coder", dry_run=True)

    printed = capsys.readouterr().out
    assert "coder/all: 2 case(s)" in printed
    assert not bench_root(tmp_path / "workspace").exists()


def test_a_dry_run_over_the_committed_corpus_preflights_every_case_it_ships(tmp_path, capsys):
    shutil.copytree(REPO_ROOT / "cases" / "coder", cases_root(tmp_path / "workspace") / "coder")

    run_bench("coder", dry_run=True)

    assert "coder/all: 20 case(s)" in capsys.readouterr().out


def test_a_stubbed_live_run_over_the_coder_site_writes_a_valid_bench_record(tmp_path, fast_gate):
    _corpus(tmp_path)
    settings = _settings(tmp_path)
    replies = [fenced(named(strategy_name(f"canary-probe-{index}"))) for index in range(2)]
    attempt = coder_attempt(
        settings, client=ScriptedCoder(replies), model="bench/coder", seeds=SEEDS
    )

    run_bench("coder", seams=BenchSeams(attempt=attempt))

    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    record = json.loads((directory / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert validate(record) == []
    assert record["metrics"]["job_pass_rate"] == 1.0


def test_a_case_the_engine_refuses_fails_alone_rather_than_ending_the_bench(tmp_path, fast_gate):
    """The corpus ships a brief naming an absent reference; its honest outcome is one red case."""
    directory = _corpus(tmp_path, cases=1)
    (directory / "canary-absent-reference.yaml").write_text(
        yaml.safe_dump(
            _document(
                payload={**_document()["payload"], "reference": "kama_adaptive_trend"},
                difficulty={**_document()["difficulty"], "composition_mode": "reference"},
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    attempt = coder_attempt(
        _settings(tmp_path),
        client=ScriptedCoder([fenced(named(strategy_name("canary-probe-0")))]),
        model="bench/coder",
        seeds=SEEDS,
    )

    run_bench("coder", seams=BenchSeams(attempt=attempt))

    (bench,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    record = json.loads((bench / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert validate(record) == []
    assert record["metrics"]["job_pass_rate"] == 0.5


def test_a_bench_job_keeps_its_authored_file_inside_the_bench_directory(tmp_path, fast_gate):
    _corpus(tmp_path, cases=1)
    attempt = coder_attempt(
        _settings(tmp_path),
        client=ScriptedCoder([fenced(named(strategy_name("canary-probe-0")))]),
        model="bench/coder",
        seeds=SEEDS,
    )

    run_bench("coder", seams=BenchSeams(attempt=attempt))

    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    authored = sorted(directory.rglob(f"{strategy_name('canary-probe-0')}.py"))
    assert [path.parent.name for path in authored] == [library.TMP_SUBDIR]


def test_a_coder_bench_override_naming_a_knob_the_site_does_not_declare_is_still_refused(tmp_path):
    with pytest.raises(Exception) as refusal:
        HarnessSpec(knob_overrides={"temperature": 0.7}).validate_knobs(CoderKnobs)

    assert "temperature" in str(refusal.value)
