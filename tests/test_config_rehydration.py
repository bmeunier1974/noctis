"""Config freezing and rehydration — the three tiers (story #132, epic #126).

``noctis.config.rehydrate`` is a statement about *settings*, exactly like its neighbour
``noctis.config.overlay``: it classifies every leaf dotted path in
:class:`~noctis.config.settings.Settings` into one of three freezing tiers and turns a run record
plus the live process's settings back into the settings that run was started with. It owns no run,
no record writer and no clock, so everything here is exercised in memory with hand-built values.

The tiers, and why each exists:

* **frozen** — everything that decides what the accumulated results *mean*. Restored from the
  record; the current ``config.yaml`` and ``mandate/`` are ignored for these keys, because an edit
  tomorrow must not retroactively change what a running experiment was told to do.
* **live** — secrets (redacted out of the record, so they can only come from the live ``.env``),
  every path/workspace knob (a run may resume on a machine with different absolute paths), and the
  per-process budgets.
* **refused** — ``mode`` and ``allow_live``. The safety gate re-resolves fresh at every process
  start (AGENTS.md rule 1); a record can never resurrect a mode, and a frozen mode that disagrees
  with the freshly resolved one is a hard error rather than a silent downgrade.

The **ratchets** come first: the classification is total over the live settings model, and the two
sub-tiers a second hand-written list would eventually get wrong (the live-money pair and the
path knobs) are *derived* from ``overlay``'s own refusal table rather than re-listed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from noctis.config import load_settings, overlay
from noctis.config.rehydrate import (
    FROZEN,
    LIVE,
    REFUSED,
    RehydrationError,
    assert_mode_unchanged,
    classify,
    config_drift,
    freeze_inputs,
    frozen_digest,
    rebase_inputs,
    rehydrate,
)
from noctis.config.settings import SECRET_FIELDS, Settings

REHYDRATE_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/config/rehydrate.py"

FROZEN_AT = "2026-07-27T14:22:33.418Z"


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    """The live model's leaves, walked here independently of the module under test."""
    leaves: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            leaves |= _leaf_paths(annotation, f"{prefix}{name}.")
        else:
            leaves.add(f"{prefix}{name}")
    return leaves


SETTINGS_LEAVES = _leaf_paths(Settings)


def _flatten_keys(data: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        keys.add(f"{prefix}{key}")
        if isinstance(value, dict):
            keys |= _flatten_keys(value, f"{prefix}{key}.")
    return keys


def _settings(tmp_path: Path, body: str = "mode: paper\n"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return load_settings(config_path=path)


def _record(settings, *, mandate=None, overrides=(), mode: str = "paper") -> dict:
    """A minimal record carrying only what rehydration reads: its frozen ``inputs``."""
    return {
        "run": {"status": "stopped"},
        "inputs": freeze_inputs(
            settings,
            mandate=mandate,
            overrides=list(overrides),
            execution_mode=mode,
            frozen_at=FROZEN_AT,
        ),
    }


# ── ratchet 1: the classification is total over the live settings model ────────────────────


def test_every_settings_leaf_is_classified_into_exactly_one_freezing_tier():
    classified = [FROZEN, LIVE, REFUSED]
    union = FROZEN | LIVE | REFUSED

    assert union == SETTINGS_LEAVES, sorted(union.symmetric_difference(SETTINGS_LEAVES))
    for one, other in ((0, 1), (0, 2), (1, 2)):
        assert not classified[one] & classified[other]


def test_classify_is_total_over_any_string():
    assert classify("promotion.metric") == "frozen"
    assert classify("workspace_dir") == "live"
    assert classify("mode") == "refused"
    assert classify("promotion.no_such_knob") == "unknown"


# ── ratchet 2: the sub-tiers are derived from overlay's table, never re-listed ──────────────


def test_the_refused_tier_is_derived_from_the_overlays_live_money_refusals():
    """One list, one place: the pair the overlay refuses because of the live-money double gate is
    exactly the pair a record may never restore."""
    assert REFUSED == overlay.refused_paths(overlay.LIVE_MONEY)
    assert REFUSED == {"mode", "allow_live"}


def test_the_live_path_knobs_are_derived_from_the_overlays_state_io_refusals():
    """A run resumes on a machine with different absolute paths, so every path knob is live —
    and which paths those are is already declared once, in the overlay's refusal table."""
    paths = overlay.refused_paths(overlay.STATE_IO)

    assert paths <= LIVE
    assert "workspace_dir" in paths and "data.lake_dir" in paths


def test_the_secret_fields_are_the_one_set_the_config_digest_already_excludes():
    from noctis.bootstrap import _DIGEST_SECRET_FIELDS

    assert SECRET_FIELDS == _DIGEST_SECRET_FIELDS
    assert SECRET_FIELDS == overlay.refused_paths(overlay.SECRETS)
    assert SECRET_FIELDS <= LIVE


@pytest.mark.parametrize(
    "path",
    ["time_limit_hours", "qa.keep_last_runs", "observability.heartbeat_polls", "data.budget_usd"],
)
def test_the_per_process_budgets_are_live(path):
    assert classify(path) == "live"


@pytest.mark.parametrize(
    "path",
    [
        "promotion.metric",
        "promotion.max_gap",
        "research.model",
        "research.min_trials",
        "research_time_budget_minutes",
        "backtest.fee_bps",
        "trading.execution",
        "risk.max_daily_loss_pct",
        "universe",
        "session.calendar",
        "champion_count",
        "data.provider",
        "data.dataset",
    ],
)
def test_what_the_results_mean_is_frozen(path):
    assert classify(path) == "frozen"


# ── the frozen tier: restored from the record, current files ignored ───────────────────────


def test_the_frozen_tier_is_restored_and_the_current_config_is_ignored(tmp_path):
    frozen = _settings(
        tmp_path / "a",
        "mode: paper\npromotion:\n  metric: sortino\n  max_gap: 0.4\n"
        "research:\n  min_trials: 20\nchampion_count: 5\n",
    )
    record = _record(frozen)
    live = _settings(tmp_path / "b", "mode: paper\npromotion:\n  metric: sharpe\n")

    resumed = rehydrate(record, live)

    assert resumed.promotion.metric == "sortino"
    assert resumed.promotion.max_gap == 0.4
    assert resumed.research.min_trials == 20
    assert resumed.champion_count == 5


def test_the_live_tier_always_comes_from_the_current_process(tmp_path, monkeypatch):
    monkeypatch.setenv("NOCTIS_WORKSPACE", f"{tmp_path}/old")
    frozen = _settings(
        tmp_path / "a",
        "mode: paper\ntime_limit_hours: 1.0\nqa:\n  keep_last_runs: 2\n"
        "data:\n  budget_usd: 500.0\n",
    )
    record = _record(frozen)
    monkeypatch.setenv("NOCTIS_WORKSPACE", f"{tmp_path}/new")
    live = _settings(
        tmp_path / "b",
        "mode: paper\ntime_limit_hours: 8.0\nqa:\n  keep_last_runs: 40\ndata:\n  budget_usd: 7.5\n",
    )
    live.databento_api_key = "sk-from-the-live-env"

    resumed = rehydrate(record, live)

    assert Path(resumed.workspace_dir) == tmp_path / "new"
    assert Path(resumed.state_dir).is_relative_to(tmp_path / "new")
    assert Path(resumed.data.lake_dir).is_relative_to(tmp_path / "new")
    assert resumed.time_limit_hours == 8.0
    assert resumed.qa.keep_last_runs == 40
    assert resumed.data.budget_usd == 7.5
    assert resumed.databento_api_key == "sk-from-the-live-env"


def test_a_frozen_and_a_live_knob_in_one_section_both_land(tmp_path):
    """``data.provider`` is frozen and ``data.budget_usd`` is live — the section rebuild has to
    take one from each side rather than restoring or keeping the section wholesale."""
    frozen = _settings(
        tmp_path / "a", "mode: paper\ndata:\n  provider: yfinance\n  budget_usd: 500.0\n"
    )
    live = _settings(tmp_path / "b", "mode: paper\ndata:\n  budget_usd: 7.5\n")

    resumed = rehydrate(_record(frozen), live)

    assert resumed.data.provider == "yfinance"
    assert resumed.data.budget_usd == 7.5


def test_rehydration_returns_a_new_object_and_leaves_the_live_settings_untouched(tmp_path):
    frozen = _settings(tmp_path / "a", "mode: paper\npromotion:\n  metric: sortino\n")
    live = _settings(tmp_path / "b", "mode: paper\n")

    resumed = rehydrate(_record(frozen), live)

    assert resumed is not live
    assert live.promotion.metric == "sharpe"


def test_a_record_with_no_frozen_inputs_leaves_the_live_settings_alone(tmp_path):
    """An adopted (pre-run-scoped) run has a record but never froze a config. Resuming it is a
    normal resume against the current files, not a crash."""
    live = _settings(tmp_path, "mode: paper\npromotion:\n  metric: sortino\n")

    resumed = rehydrate({"run": {"status": "stopped"}, "inputs": None}, live)

    assert resumed.promotion.metric == "sortino"


# ── the refused tier: the safety gate is never rehydrated ──────────────────────────────────


def test_mode_and_allow_live_are_never_written_to_the_record(tmp_path):
    settings = _settings(tmp_path, "mode: paper\n")
    settings.allow_live = True

    resolved = _record(settings)["inputs"]["settings"]["resolved"]

    assert "mode" not in resolved
    assert "allow_live" not in resolved
    assert not REFUSED & _flatten_keys(resolved)
    # The pair survives in the record only as *names*, in the list that documents the tier — a
    # reader can see what was refused without the record carrying either value.
    assert _record(settings)["inputs"]["settings"]["refused_keys"] == ["allow_live", "mode"]


def test_mode_and_allow_live_are_never_restored_from_a_record(tmp_path):
    """Even a hand-edited record that smuggles the pair back in cannot move them: the gate's
    verdict comes from this process, never from a file a record could carry."""
    frozen = _settings(tmp_path / "a", "mode: paper\n")
    record = _record(frozen)
    record["inputs"]["settings"]["resolved"]["mode"] = "live"
    record["inputs"]["settings"]["resolved"]["allow_live"] = True
    live = _settings(tmp_path / "b", "mode: paper\n")

    resumed = rehydrate(record, live)

    assert resumed.mode == "paper"
    assert resumed.allow_live is False


def test_a_frozen_mode_that_disagrees_with_the_resolved_mode_is_a_hard_error(tmp_path):
    record = _record(_settings(tmp_path, "mode: paper\n"), mode="paper")

    with pytest.raises(RehydrationError) as excinfo:
        assert_mode_unchanged(record, "live")

    assert "paper" in str(excinfo.value) and "live" in str(excinfo.value)


def test_a_frozen_mode_that_matches_the_resolved_mode_is_accepted(tmp_path):
    record = _record(_settings(tmp_path, "mode: paper\n"), mode="paper")

    assert assert_mode_unchanged(record, "paper") is None


def test_a_record_that_froze_no_mode_at_all_never_blocks_a_resume():
    assert assert_mode_unchanged({"run": {}, "inputs": None}, "paper") is None


def test_no_secret_value_reaches_the_frozen_inputs(tmp_path):
    settings = _settings(tmp_path, "mode: paper\n")
    for field in SECRET_FIELDS:
        setattr(settings, field, f"sk-{field}-do-not-leak")

    frozen = json.dumps(_record(settings))

    assert "do-not-leak" not in frozen
    for field in SECRET_FIELDS:
        assert field not in _record(settings)["inputs"]["settings"]["resolved"]


# ── the mandate is frozen as resolved text, not as a selector ──────────────────────────────


class _Reference:
    def __init__(self, path: str, text: str) -> None:
        self.path = path
        self.text = text


class _Mandate:
    """The duck-typed shape ``freeze_inputs`` reads — the research package's ``Mandate``."""

    def __init__(self) -> None:
        self.text = "Trade only the most volatile names. Risk appetite: high."
        self.source = "profile:aggressive"
        self.summary = "aggressive momentum"
        self.symbols = ["NVDA", "TSLA"]
        self.config_overrides = {"promotion.metric": "sortino"}
        self.references = [_Reference("references/watchlist.md", "NVDA\nTSLA\n")]


def test_the_mandate_is_frozen_as_resolved_text_plus_its_applied_overlay(tmp_path):
    settings = _settings(tmp_path, "mode: paper\n")

    frozen = _record(settings, mandate=_Mandate(), overrides=["promotion.metric=sortino"])[
        "inputs"
    ]["mandate"]

    assert frozen["text"] == "Trade only the most volatile names. Risk appetite: high."
    assert frozen["source"] == "profile:aggressive"
    assert frozen["summary"] == "aggressive momentum"
    assert frozen["symbols"] == ["NVDA", "TSLA"]
    assert frozen["config_overrides"] == {"promotion.metric": "sortino"}
    assert frozen["overrides_applied"] == ["promotion.metric=sortino"]
    assert frozen["references"][0]["path"] == "references/watchlist.md"
    assert frozen["references"][0]["text"] == "NVDA\nTSLA\n"
    assert len(frozen["text_sha256"]) == 64  # the text is pinned, not just carried


def test_a_run_with_no_mandate_freezes_an_explicit_null(tmp_path):
    assert _record(_settings(tmp_path))["inputs"]["mandate"] is None


# ── the rest of the provenance block: which models, and which data (story #139) ────────────


def test_the_frozen_inputs_carry_the_resolved_models(tmp_path):
    """Which model authored, judged and ideated is what a run's research trail *means*. It is
    already frozen key-by-key in ``resolved``; this states it once, resolved, in one block a
    website can render beside the mandate."""
    settings = _settings(
        tmp_path,
        "mode: paper\nresearch:\n  model: openai/gpt-5.4\n  cost_profile: economy\n"
        "  agent:\n    coder_model: anthropic/claude-sonnet-5\n"
        "    coder_fallback_model: anthropic/claude-opus-4-8\n    context_window: 32768\n"
        "ideation:\n  model: claude-opus-4-8\n",
    )

    models = freeze_inputs(
        settings, research_loop="episodic", frozen_at=FROZEN_AT, execution_mode="paper"
    )["models"]

    assert models["research"] == "openai/gpt-5.4"
    assert models["coder"] == "anthropic/claude-sonnet-5"
    assert models["coder_fallback"] == "anthropic/claude-opus-4-8"
    assert models["ideation"] == "claude-opus-4-8"
    assert models["research_loop"] == "episodic"
    assert models["context_window"] == 32768
    assert models["cost_profile"] == "economy"


def test_the_research_model_falls_back_to_the_agents_own_model(tmp_path):
    """``research.model: null`` means "use ``research.agent.model``" — the record states the model
    that will actually run, not the null that stood in for it."""
    settings = _settings(tmp_path, "mode: paper\nresearch:\n  model: null\n")

    models = freeze_inputs(settings, frozen_at=FROZEN_AT)["models"]

    assert models["research"] == settings.research.agent.model


def test_an_unset_model_is_an_explicit_null_never_an_omitted_key(tmp_path):
    models = freeze_inputs(_settings(tmp_path), frozen_at=FROZEN_AT)["models"]

    assert models["coder"] is None
    assert models["coder_fallback"] is None
    assert models["context_window"] is None
    assert "research_loop" in models and models["research_loop"] is None


def test_the_frozen_inputs_carry_the_data_provider_and_dataset(tmp_path):
    """Which vendor, which dataset, and where the lake is — the run's data provenance, stated
    beside its configuration rather than hunted for inside the resolved dump."""
    settings = _settings(
        tmp_path, "mode: paper\ndata:\n  provider: databento\n  dataset: EQUS.MINI\n"
    )

    data = freeze_inputs(settings, frozen_at=FROZEN_AT)["data"]

    assert data["provider"] == "databento"
    assert data["dataset"] == "EQUS.MINI"
    # The lake is workspace-level and SHARED across runs by design — stated so a reader never
    # mistakes it for something this run owns.
    assert data["lake_dir"] == settings.data.lake_dir


def test_no_secret_reaches_the_models_or_the_data_block(tmp_path):
    """A model name is public; an API key is not, and the two live one settings section apart."""
    settings = _settings(tmp_path, "mode: paper\n")
    for field in SECRET_FIELDS:
        setattr(settings, field, f"sk-{field}-do-not-leak")

    frozen = freeze_inputs(settings, research_loop="conversation", frozen_at=FROZEN_AT)

    assert "do-not-leak" not in json.dumps(frozen["models"])
    assert "do-not-leak" not in json.dumps(frozen["data"])
    assert "do-not-leak" not in json.dumps(frozen)


# ── the frozen digest: a label for "these runs mean the same thing" ────────────────────────


def test_the_frozen_digest_moves_on_a_frozen_key_and_not_on_a_live_one(tmp_path):
    base = _settings(tmp_path / "a", "mode: paper\n")
    moved_live = _settings(
        tmp_path / "b", f"mode: paper\nworkspace_dir: {tmp_path}/elsewhere\ntime_limit_hours: 3\n"
    )
    moved_frozen = _settings(tmp_path / "c", "mode: paper\npromotion:\n  metric: sortino\n")

    assert frozen_digest(moved_live) == frozen_digest(base)
    assert frozen_digest(moved_frozen) != frozen_digest(base)


def test_the_records_digest_is_the_digest_of_what_rehydration_restores(tmp_path):
    frozen = _settings(tmp_path / "a", "mode: paper\npromotion:\n  metric: sortino\n")
    record = _record(frozen)
    live = _settings(tmp_path / "b", f"mode: paper\nworkspace_dir: {tmp_path}/new\n")

    resumed = rehydrate(record, live)

    assert frozen_digest(resumed) == record["inputs"]["settings"]["digest"]


def test_the_runs_own_tree_is_not_quoted_back_at_it(tmp_path):
    """Settings are assembled before a run exists, so the run-scoped paths they carry at freeze
    time name the reserved default rather than this run — recording them would be quoting a
    directory that was never the run's. They are live tier anyway, so nothing is lost."""
    from noctis.config.rehydrate import RUN_IDENTITY

    resolved = _record(_settings(tmp_path))["inputs"]["settings"]["resolved"]

    assert not RUN_IDENTITY & _flatten_keys(resolved)
    assert "workspace_dir" in resolved  # the workspace-level paths stay, as evidence


def test_the_config_digest_and_the_record_exclude_the_same_run_identity_paths():
    from noctis.bootstrap import _digest_excluded_fields
    from noctis.config.rehydrate import RUN_IDENTITY

    assert _digest_excluded_fields() == set(SECRET_FIELDS) | set(RUN_IDENTITY)


def test_the_frozen_inputs_name_the_keys_of_all_three_tiers(tmp_path):
    settings = json.loads(json.dumps(_record(_settings(tmp_path))["inputs"]["settings"]))

    assert set(settings["frozen_keys"]) == FROZEN
    assert set(settings["live_keys"]) == LIVE
    assert set(settings["refused_keys"]) == REFUSED


# ── drift: what the current files would change, if a resume let them (story #134) ──────────


def test_config_drift_names_a_changed_frozen_key_with_both_values(tmp_path):
    """The whole point of the flag: an operator sees what they *would* be adopting, per key."""
    record = _record(_settings(tmp_path / "a", "mode: paper\npromotion:\n  metric: sortino\n"))
    current = _settings(tmp_path / "b", "mode: paper\npromotion:\n  metric: total_return\n")

    drift = config_drift(record, current)

    assert bool(drift) is True
    assert [(change.path, change.frozen, change.current) for change in drift.settings] == [
        ("promotion.metric", "sortino", "total_return")
    ]


def test_a_run_whose_files_have_not_moved_reports_no_drift(tmp_path):
    settings = _settings(tmp_path, "mode: paper\npromotion:\n  metric: sortino\n")

    drift = config_drift(_record(settings), settings)

    assert bool(drift) is False
    assert drift.settings == ()
    assert drift.mandate is None


def test_a_moved_live_tier_key_is_never_drift(tmp_path, monkeypatch):
    """The live tier is live *by design* — paths, secrets and per-process budgets are always this
    process's — so reporting one as drift would invite an operator to "adopt" a non-difference."""
    monkeypatch.setenv("NOCTIS_WORKSPACE", f"{tmp_path}/old")
    record = _record(_settings(tmp_path / "a", "mode: paper\ntime_limit_hours: 1.0\n"))
    monkeypatch.setenv("NOCTIS_WORKSPACE", f"{tmp_path}/new")
    current = _settings(tmp_path / "b", "mode: paper\ntime_limit_hours: 9.0\n")
    current.databento_api_key = "sk-from-the-live-env"

    drift = config_drift(record, current)

    assert bool(drift) is False


def test_the_live_money_gates_are_never_drift_even_if_a_record_smuggles_them_in(tmp_path):
    """The refused tier is absolute: it is not recorded, not restored, and not rebasable — so it
    is not drift either, whatever a hand-edited record claims."""
    record = _record(_settings(tmp_path / "a", "mode: paper\n"))
    record["inputs"]["settings"]["resolved"]["mode"] = "live"
    record["inputs"]["settings"]["resolved"]["allow_live"] = True

    drift = config_drift(record, _settings(tmp_path / "b", "mode: paper\n"))

    assert [change.path for change in drift.settings] == []


def test_mandate_drift_is_drift_in_the_resolved_text_not_in_the_selector(tmp_path):
    settings = _settings(tmp_path, "mode: paper\n")
    record = _record(settings, mandate=_Mandate())
    rewritten = _Mandate()
    rewritten.text = "Buy and hold index funds."

    drift = config_drift(record, settings, mandate=rewritten)

    assert drift.mandate is not None
    assert drift.mandate.frozen_sha256 != drift.mandate.current_sha256
    assert drift.mandate.frozen_text == "Trade only the most volatile names. Risk appetite: high."
    assert drift.mandate.current_text == "Buy and hold index funds."


def test_the_same_text_reached_through_a_different_selector_is_not_drift(tmp_path):
    """A run freezes what it was *told*, not which file told it — so renaming the profile behind
    identical text changes nothing about what the accumulated results mean."""
    settings = _settings(tmp_path, "mode: paper\n")
    record = _record(settings, mandate=_Mandate())
    renamed = _Mandate()
    renamed.source = "profile:volatility"

    drift = config_drift(record, settings, mandate=renamed)

    assert drift.mandate is None
    assert bool(drift) is False


def test_dropping_the_mandate_entirely_is_drift(tmp_path):
    settings = _settings(tmp_path, "mode: paper\n")

    drift = config_drift(_record(settings, mandate=_Mandate()), settings, mandate=None)

    assert drift.mandate is not None
    assert drift.mandate.current_sha256 is None


def test_a_record_that_froze_no_config_has_nothing_to_drift_from(tmp_path):
    """An adopted history (story #131) never froze a configuration, so there is no difference to
    show — and nothing to rebase either."""
    drift = config_drift({"run": {"status": "stopped"}, "inputs": None}, _settings(tmp_path))

    assert bool(drift) is False
    assert drift.frozen is False


# ── rebasing: adopting the current config deliberately, never silently ─────────────────────


def test_rebasing_bumps_the_epoch_and_appends_a_before_after_entry(tmp_path):
    record = _record(_settings(tmp_path / "a", "mode: paper\npromotion:\n  metric: sortino\n"))
    current = _settings(tmp_path / "b", "mode: paper\npromotion:\n  metric: total_return\n")

    rebased = rebase_inputs(record, current, at=FROZEN_AT, segment=3)

    assert rebased is not None
    assert rebased["config_epoch"] == 2
    (change,) = rebased["config_changes"]
    assert change["at"] == FROZEN_AT
    assert change["segment"] == 3
    assert change["from_epoch"] == 1 and change["to_epoch"] == 2
    assert change["settings"] == [
        {"path": "promotion.metric", "from": "sortino", "to": "total_return"}
    ]
    assert change["digest_before"] == record["inputs"]["settings"]["digest"]
    assert change["digest_after"] == frozen_digest(current)
    assert rebased["settings"]["resolved"]["promotion"]["metric"] == "total_return"


def test_a_rebased_block_carries_the_whole_provenance_section_again(tmp_path):
    """A rebase re-freezes the *whole* inputs block, so the models and the data provenance are
    re-stated for the new epoch rather than silently going null on the way through."""
    record = _record(_settings(tmp_path / "a", "mode: paper\ndata:\n  provider: databento\n"))
    current = _settings(tmp_path / "b", "mode: paper\ndata:\n  provider: yfinance\n")

    rebased = rebase_inputs(record, current, research_loop="episodic", at=FROZEN_AT, segment=1)

    assert rebased is not None
    assert rebased["data"]["provider"] == "yfinance"
    assert rebased["models"]["research_loop"] == "episodic"
    assert rebased["models"]["research"] == current.research.model


def test_rebasing_a_drift_free_run_is_a_no_op_that_does_not_bump_the_epoch(tmp_path):
    settings = _settings(tmp_path, "mode: paper\npromotion:\n  metric: sortino\n")

    assert rebase_inputs(_record(settings), settings, at=FROZEN_AT, segment=1) is None


def test_rebasing_keeps_every_earlier_change_entry(tmp_path):
    """The change log is append-only: a run that changed config twice must say so twice, or the
    second rebase would erase the evidence of the first."""
    record = _record(_settings(tmp_path / "a", "mode: paper\npromotion:\n  metric: sortino\n"))
    once = rebase_inputs(
        record,
        _settings(tmp_path / "b", "mode: paper\npromotion:\n  metric: total_return\n"),
        at=FROZEN_AT,
        segment=1,
    )
    assert once is not None

    twice = rebase_inputs(
        {"run": {"status": "stopped"}, "inputs": once},
        _settings(tmp_path / "c", "mode: paper\npromotion:\n  metric: sharpe\n"),
        at=FROZEN_AT,
        segment=2,
    )

    assert twice is not None
    assert twice["config_epoch"] == 3
    assert [change["to_epoch"] for change in twice["config_changes"]] == [2, 3]
    assert [change["segment"] for change in twice["config_changes"]] == [1, 2]


def test_a_rebased_block_never_carries_the_live_money_gates(tmp_path):
    settings = _settings(tmp_path / "b", "mode: paper\npromotion:\n  metric: total_return\n")
    settings.allow_live = True
    settings.databento_api_key = "sk-do-not-leak"

    rebased = rebase_inputs(
        _record(_settings(tmp_path / "a", "mode: paper\npromotion:\n  metric: sortino\n")),
        settings,
        at=FROZEN_AT,
        segment=1,
    )

    assert rebased is not None
    assert not REFUSED & _flatten_keys(rebased["settings"]["resolved"])
    assert "do-not-leak" not in json.dumps(rebased)


def test_a_rebase_carries_the_current_mandate_text_forward(tmp_path):
    settings = _settings(tmp_path, "mode: paper\n")
    record = _record(settings, mandate=_Mandate())
    rewritten = _Mandate()
    rewritten.text = "Buy and hold index funds."

    rebased = rebase_inputs(record, settings, mandate=rewritten, at=FROZEN_AT, segment=1)

    assert rebased is not None
    assert rebased["mandate"]["text"] == "Buy and hold index funds."
    assert (
        rebased["config_changes"][0]["mandate"]["to"]["text_sha256"]
        == (rebased["mandate"]["text_sha256"])
    )


def test_a_fresh_run_freezes_an_empty_change_log_at_epoch_one(tmp_path):
    inputs = _record(_settings(tmp_path))["inputs"]

    assert inputs["config_epoch"] == 1
    assert inputs["config_changes"] == []


# ── purity, structurally ───────────────────────────────────────────────────────────────────


def test_rehydration_reaches_no_io_no_clock_and_no_settings_source():
    """``(record, live_settings) -> Settings``, and nothing else. Rebuilding a ``Settings`` from
    its sources would re-read the environment, ``.env`` and the YAML file — exactly the files a
    resume must ignore — so this module never constructs one. The drift diff and the rebase
    (story #134) are held to the same rule: both are ``(record, settings) -> value``, so the CLI
    owns every read and every render."""
    text = REHYDRATE_SOURCE.read_text()
    for forbidden in (
        "datetime.now",
        "utcnow",
        "open(",
        "read_text",
        "write_text",
        "load_settings",
        "Settings(",
        "os.environ",
        "Path(",
    ):
        assert forbidden not in text, forbidden
