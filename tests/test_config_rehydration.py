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
    freeze_inputs,
    frozen_digest,
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


# ── purity, structurally ───────────────────────────────────────────────────────────────────


def test_rehydration_reaches_no_io_no_clock_and_no_settings_source():
    """``(record, live_settings) -> Settings``, and nothing else. Rebuilding a ``Settings`` from
    its sources would re-read the environment, ``.env`` and the YAML file — exactly the files a
    resume must ignore — so this module never constructs one."""
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
