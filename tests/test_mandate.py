"""The operator mandate loader, overlay, and seam (Phase 1).

Pure, client-free coverage of noctis.research.mandate: selector precedence, name lookup,
the empty-MANDATE rule, front-matter parsing (incl. the malformed-fence degrade), reference
inclusion/caps/confinement, summary extraction, and the config-overlay *adapter* — the thin
seam onto noctis.config.overlay that keeps apply_overrides' signature and exception type and
makes a refused/unknown/invalid key fatal at startup, naming the mandate that carried it.
(The classifier and the applier themselves are covered in tests/test_settings_overlay.py.)
Plus the CLI mutual-exclusion of --directive/--mandate, the exit-1-on-a-bad-overlay path,
and the "empty input == today" parity.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.config import load_settings
from noctis.research import (
    Mandate,
    MandateError,
    apply_overrides,
    profiles_catalog,
    resolve_mandate,
)
from noctis.research.prompt import _MANDATE_BLOCK

runner = CliRunner()


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _settings(tmp_path, mandate_dir, selector=None):
    """Isolated Settings (repo config.yaml is bypassed via a missing config path)."""
    s = load_settings(config_path=tmp_path / "missing.yaml", mandate_dir=str(mandate_dir))
    s.research.mandate = selector
    return s


def _profile(summary: str, extra: str = "") -> str:
    return f"---\nsummary: {summary}\n{extra}---\n{summary} — full body prose.\n"


# ── selector precedence (all four rows) ──────────────────────────────────────────────────
def test_precedence_cli_directive_wins(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "aggressive.md", _profile("Aggressive."))
    settings = _settings(tmp_path, mandate_dir, selector="aggressive")
    m = resolve_mandate(settings, cli_directive="inline steer", cli_mandate="aggressive")
    assert m is not None
    assert m.source == "cli"
    assert m.text == "inline steer"


def test_precedence_cli_mandate_beats_config(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "aggressive.md", _profile("Aggressive."))
    _write(mandate_dir / "profiles" / "conservative.md", _profile("Conservative."))
    settings = _settings(tmp_path, mandate_dir, selector="conservative")
    m = resolve_mandate(settings, cli_mandate="aggressive")
    assert m.source == "profile:aggressive"


def test_precedence_config_selector_used(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "aggressive.md", _profile("Aggressive."))
    settings = _settings(tmp_path, mandate_dir, selector="aggressive")
    m = resolve_mandate(settings)
    assert m.source == "profile:aggressive"


def test_precedence_none_when_no_selector(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate", selector=None)
    assert resolve_mandate(settings) is None


# ── name lookup order + MandateError on a bad selector ───────────────────────────────────
def test_lookup_prefers_profiles_dir(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "foo.md", _profile("Profile foo."))
    _write(mandate_dir / "foo.md", _profile("Top-level foo."))
    settings = _settings(tmp_path, mandate_dir, selector="foo")
    m = resolve_mandate(settings)
    assert m.source == "profile:foo"
    assert "Profile foo." in m.text


def test_lookup_falls_back_to_top_level(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "bar.md", _profile("Top-level bar."))
    settings = _settings(tmp_path, mandate_dir, selector="bar")
    m = resolve_mandate(settings)
    assert m.source == "mandate/bar.md"
    assert "Top-level bar." in m.text


def test_missing_selector_raises(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate", selector="nope")
    with pytest.raises(MandateError):
        resolve_mandate(settings)


def test_path_separator_selector_rejected(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate", selector="profiles/aggressive")
    with pytest.raises(MandateError):
        resolve_mandate(settings)


def test_missing_mandate_md_raises(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate", selector="MANDATE")
    with pytest.raises(MandateError):
        resolve_mandate(settings)


# ── the empty-MANDATE rule ───────────────────────────────────────────────────────────────
def test_empty_mandate_md_resolves_to_none(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(
        mandate_dir / "MANDATE.md",
        "<!-- This is the shipped template. Write your mandate here. -->\n\n   \n",
    )
    settings = _settings(tmp_path, mandate_dir, selector="MANDATE")
    assert resolve_mandate(settings) is None


def test_nonempty_mandate_md_resolves(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "MANDATE.md", "Trade volatile US names, high risk appetite.\n")
    settings = _settings(tmp_path, mandate_dir, selector="MANDATE")
    m = resolve_mandate(settings)
    assert m is not None
    assert m.source == "mandate/MANDATE.md"
    assert "volatile US names" in m.text


# ── front-matter parsing incl. the malformed-fence degrade ───────────────────────────────
def test_front_matter_parsed(tmp_path):
    mandate_dir = tmp_path / "mandate"
    body = (
        "---\n"
        "summary: High risk appetite, volatile names.\n"
        "config:\n"
        "  promotion:\n"
        "    metric: total_return\n"
        "---\n"
        "Go find something spicy.\n"
    )
    _write(mandate_dir / "profiles" / "spicy.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="spicy")
    m = resolve_mandate(settings)
    assert m.summary == "High risk appetite, volatile names."
    assert m.config_overrides == {"promotion.metric": "total_return"}
    assert m.text == "Go find something spicy."


def test_front_matter_symbols_parsed_normalized_and_deduped(tmp_path):
    mandate_dir = tmp_path / "mandate"
    body = (
        "---\n"
        "summary: Uranium names.\n"
        "symbols:\n"
        "  - smr\n"
        "  - CCJ\n"
        "  - smr\n"
        "  - 7\n"
        "---\n"
        "Research the uranium complex.\n"
    )
    _write(mandate_dir / "profiles" / "uranium.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="uranium")
    m = resolve_mandate(settings)
    # Upper-cased, deduped, non-strings dropped (warn, never fatal); order preserved.
    assert m.symbols == ["SMR", "CCJ"]

    # A malformed block (not a list) drops to no declared symbols.
    _write(mandate_dir / "profiles" / "badsyms.md", "---\nsymbols: SMR\n---\nBody.\n")
    settings = _settings(tmp_path, mandate_dir, selector="badsyms")
    assert resolve_mandate(settings).symbols == []


def test_malformed_front_matter_degrades_to_prose(tmp_path):
    mandate_dir = tmp_path / "mandate"
    # A proper fence, but the YAML inside is broken (unterminated quote).
    body = '---\nkey: "unterminated\nconfig: broken\n---\nReal body here.\n'
    _write(mandate_dir / "profiles" / "broken.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="broken")
    m = resolve_mandate(settings)
    # Whole file is treated as prose: no overrides, and the fence survives in the body.
    assert m.config_overrides == {}
    assert m.text.startswith("---")
    assert "Real body here." in m.text


# ── reference inclusion + caps + bad-path/../absolute rejection ───────────────────────────
def test_references_front_matter_and_inline_merge_and_dedupe(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "references" / "a.md", "alpha notes")
    _write(mandate_dir / "references" / "b.md", "beta notes")
    body = (
        "---\n"
        "references:\n"
        "  - references/a.md\n"
        "---\n"
        "See [[references/a]] and also [[references/b]].\n"
    )
    _write(mandate_dir / "profiles" / "refs.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="refs")
    m = resolve_mandate(settings)
    assert [r.path for r in m.references] == ["references/a.md", "references/b.md"]
    assert m.references[0].text == "alpha notes"
    # The wikilink text stays in the body prose as-is.
    assert "[[references/a]]" in m.text


def test_reference_over_file_cap_dropped(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "references" / "big.md", "x" * 3000)  # > 2 KB per-file cap
    body = "---\nreferences:\n  - references/big.md\n---\nBody.\n"
    _write(mandate_dir / "profiles" / "big.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="big")
    m = resolve_mandate(settings)
    assert m.references == []


def test_references_over_total_budget_dropped(tmp_path):
    mandate_dir = tmp_path / "mandate"
    for name in ("a", "b", "c", "d"):
        _write(mandate_dir / "references" / f"{name}.md", "y" * 2000)  # each under file cap
    body = (
        "---\nreferences:\n"
        "  - references/a.md\n  - references/b.md\n"
        "  - references/c.md\n  - references/d.md\n---\nBody.\n"
    )
    _write(mandate_dir / "profiles" / "many.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="many")
    m = resolve_mandate(settings)
    # a+b+c = 6000 bytes fits under the ~6 KB total budget; d overflows and is dropped.
    assert [r.path for r in m.references] == [
        "references/a.md",
        "references/b.md",
        "references/c.md",
    ]


def test_absolute_reference_rejected(tmp_path):
    mandate_dir = tmp_path / "mandate"
    body = "---\nreferences:\n  - /etc/passwd\n---\nBody.\n"
    _write(mandate_dir / "profiles" / "abs.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="abs")
    m = resolve_mandate(settings)
    assert m.references == []


def test_dotdot_escape_reference_rejected(tmp_path):
    mandate_dir = tmp_path / "mandate"
    # A real file OUTSIDE the mandate tree — dropped by confinement, not by being missing.
    _write(tmp_path / "outside.md", "secret")
    body = "---\nreferences:\n  - ../outside.md\n---\nBody.\n"
    _write(mandate_dir / "profiles" / "esc.md", body)
    settings = _settings(tmp_path, mandate_dir, selector="esc")
    m = resolve_mandate(settings)
    assert m.references == []


# ── summary extraction ───────────────────────────────────────────────────────────────────
def test_summary_from_front_matter(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(
        mandate_dir / "profiles" / "s.md",
        "---\nsummary: The stated summary.\n---\nBody line one.\n",
    )
    settings = _settings(tmp_path, mandate_dir, selector="s")
    assert resolve_mandate(settings).summary == "The stated summary."


def test_summary_falls_back_to_first_prose_line(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "s.md", "\n\nFirst real line.\nSecond line.\n")
    settings = _settings(tmp_path, mandate_dir, selector="s")
    assert resolve_mandate(settings).summary == "First real line."


def test_html_comments_stripped_from_agent_text(tmp_path):
    """Author/operator HTML comments (the MANDATE.md how-to header) are not steering — they
    must not reach the agent-facing ``text`` or the summary, only the file itself."""
    mandate_dir = tmp_path / "mandate"
    _write(
        mandate_dir / "MANDATE.md",
        "---\nsummary: Real summary.\n---\n"
        "<!-- THIS FILE IS YOURS: operator how-to that the agent must never see. -->\n\n"
        "Steer toward volatile names.\n",
    )
    settings = _settings(tmp_path, mandate_dir, selector="MANDATE")
    mandate = resolve_mandate(settings)
    assert "<!--" not in mandate.text
    assert "THIS FILE IS YOURS" not in mandate.text
    assert mandate.text == "Steer toward volatile names."
    assert mandate.summary == "Real summary."


# ── the config overlay adapter (§3.4) ────────────────────────────────────────────────────
# apply_overrides keeps its signature and its exception type; the classification and the
# validation live in noctis.config.overlay (tests/test_settings_overlay.py). What is asserted
# here is the adapter's own contract: the mandate source is named, and a refused, unknown, or
# invalid key is FATAL at startup rather than a buried warning.
def _mandate_with(overrides: dict, source: str = "profile:test") -> Mandate:
    return Mandate(text="x", source=source, summary="x", references=[], config_overrides=overrides)


def test_overlay_applies_the_scoring_metric(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate")
    assert settings.promotion.metric == "sharpe"  # base default
    lines = apply_overrides(settings, _mandate_with({"promotion.metric": "total_return"}))
    assert settings.promotion.metric == "total_return"
    assert lines == ["promotion.metric=total_return"]


def test_a_mandate_binds_the_run_shaping_settings_from_nested_front_matter(tmp_path):
    """A mandate configures the whole run (#117) — which model thinks, what it may spend, how
    big the prompt gets — straight out of the nested ``config:`` block, flattened here and
    applied by the overlay."""
    mandate_dir = tmp_path / "mandate"
    _write(
        mandate_dir / "profiles" / "local.md",
        "---\n"
        "summary: A local-model personality.\n"
        "config:\n"
        "  research:\n"
        "    model: ollama/qwen3-coder-30b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    cost_profile: economy\n"
        "    focus_size: 8\n"
        "    agent:\n"
        "      loop: episodic\n"
        "      context_window: 32768\n"
        "      max_tokens: 4096\n"
        "  data:\n"
        "    history_days: 180\n"
        "---\n"
        "Hunt mean reversion on a small local model.\n",
    )
    settings = _settings(tmp_path, mandate_dir, selector="local")
    mandate = resolve_mandate(settings)
    assert mandate is not None

    lines = apply_overrides(settings, mandate)

    assert settings.research.model == "ollama/qwen3-coder-30b"
    assert settings.research.base_url == "http://localhost:11434/v1"
    assert settings.research.cost_profile == "economy"
    assert settings.research.focus_size == 8
    assert settings.research.agent.loop == "episodic"
    assert settings.research.agent.context_window == 32768
    assert settings.research.agent.max_tokens == 4096
    assert settings.data.history_days == 180
    assert lines == sorted(lines)  # the echo is stable, one line per applied knob
    assert "research.model=ollama/qwen3-coder-30b" in lines
    assert len(lines) == 8


def test_a_tune_first_mandate_may_demand_more_trials_and_less_spend(tmp_path):
    """The tier-B direction clamps through real front matter (#118): a conduct mandate that
    wants more evidence per verdict and a smaller vendor bill is steering *towards* discipline,
    so both apply and both are echoed."""
    mandate_dir = tmp_path / "mandate"
    _write(
        mandate_dir / "tune-first.md",
        "---\n"
        "summary: Tune the existing library first.\n"
        "config:\n"
        "  research:\n"
        "    min_trials: 20\n"
        "  data:\n"
        "    budget_usd: 40.0\n"
        "---\n"
        "Drive one existing strategy to a verdict before authoring anything.\n",
    )
    settings = _settings(tmp_path, mandate_dir, selector="tune-first")
    mandate = resolve_mandate(settings)

    lines = apply_overrides(settings, mandate)

    assert settings.research.min_trials == 20
    assert settings.data.budget_usd == 40.0
    assert lines == ["data.budget_usd=40.0", "research.min_trials=20"]


def test_a_mandate_cannot_loosen_the_exhaustion_floor(tmp_path):
    """The wrong direction is fatal at startup, naming both the steering file to fix and the
    config value it tried to cross."""
    settings = _settings(tmp_path, tmp_path / "mandate")

    with pytest.raises(MandateError) as exc:
        apply_overrides(settings, _mandate_with({"research.min_trials": 3}))

    message = str(exc.value)
    assert "profile:test" in message
    assert "the configured 8" in message
    assert settings.research.min_trials == 8


def test_a_mandate_setting_every_allowed_knob_leaves_the_gates_byte_identical(tmp_path):
    """The maximal legal mandate — every knob the surface allows, at once — cannot move one
    byte of the refused subtree (the safety mode, the fill costs, the promotion thresholds,
    the holdout geometry, the paths, the secrets)."""
    import json

    from noctis.config.overlay import gate_snapshot
    from tests.test_settings_overlay import SAMPLE_VALUES  # ratcheted total over ALLOWED

    settings = _settings(tmp_path, tmp_path / "mandate")
    before = json.dumps(gate_snapshot(settings), sort_keys=True, default=str)

    lines = apply_overrides(settings, _mandate_with(dict(SAMPLE_VALUES)))

    assert len(lines) == len(SAMPLE_VALUES)
    assert json.dumps(gate_snapshot(settings), sort_keys=True, default=str) == before


def test_overlay_refuses_the_arena_and_applies_nothing(tmp_path):
    """One error names the mandate, lists every refused key *and* every wrong-direction clamp,
    and applies nothing — not even the one legal key beside them."""
    settings = _settings(tmp_path, tmp_path / "mandate")
    mandate = _mandate_with(
        {
            "mode": "live",
            "risk.max_daily_loss_pct": 99.0,
            "data.budget_usd": 9999.0,  # a clamp violation, reported in the same error
            "promotion.max_gap": 5.0,
            "promotion.min_test_metric": -5.0,
            "promotion.min_holdout_metric": -5.0,
            "promotion.metric": "total_return",  # the sole allowed key
        }
    )
    with pytest.raises(MandateError) as exc:
        apply_overrides(settings, mandate)
    message = str(exc.value)
    assert "profile:test" in message  # the refusal names its source
    for refused in (
        "mode",
        "risk.max_daily_loss_pct",
        "data.budget_usd",
        "promotion.max_gap",
        "promotion.min_test_metric",
        "promotion.min_holdout_metric",
    ):
        assert refused in message
    # Nothing moved, including the legal key that travelled with them.
    assert settings.mode == "paper"
    assert settings.risk.max_daily_loss_pct == 3.0
    assert settings.data.budget_usd == 125.0
    assert settings.promotion.max_gap == 1.0
    assert settings.promotion.min_test_metric == 0.0
    assert settings.promotion.min_holdout_metric == 0.0
    assert settings.promotion.metric == "sharpe"


def test_overlay_refuses_invalid_metric_value(tmp_path):
    """A bad value is fatal too, carrying the one unknown-metric diagnosis.

    Raw attribute assignment on these models still does NOT validate (no
    validate_assignment) — the overlay's section re-validation is what protects the knob.
    """
    settings = _settings(tmp_path, tmp_path / "mandate")
    settings.promotion.metric = "bogus"
    assert settings.promotion.metric == "bogus"  # proves assignment is unchecked
    settings.promotion.metric = "sharpe"

    with pytest.raises(MandateError, match="unknown metric 'not_a_metric'") as exc:
        apply_overrides(settings, _mandate_with({"promotion.metric": "not_a_metric"}))
    assert "profile:test" in str(exc.value)
    assert settings.promotion.metric == "sharpe"  # invalid value refused, left unchanged


def test_overlay_refuses_an_unknown_key(tmp_path):
    """A typo'd path never silently un-steers a multi-day run."""
    settings = _settings(tmp_path, tmp_path / "mandate")
    with pytest.raises(MandateError, match="promotion.metrik"):
        apply_overrides(settings, _mandate_with({"promotion.metrik": "sortino"}))
    assert settings.promotion.metric == "sharpe"


def test_overlay_cannot_touch_backtest_costs(tmp_path):
    """However wide the run-shaping surface gets, the fill costs stay out of it — a mandate
    steers WHAT to look for, never how forgiving the arena is (#23). A backtest cost override
    is fatal and the section is left on its floored default."""
    settings = _settings(tmp_path, tmp_path / "mandate")
    assert settings.backtest.fee_bps == 1.0
    assert settings.backtest.slippage_bps == 1.0
    mandate = _mandate_with(
        {
            "backtest.fee_bps": 0.1,  # a cost-cheapening attempt via the overlay
            "backtest.slippage_bps": 0.1,
        }
    )
    with pytest.raises(MandateError) as exc:
        apply_overrides(settings, mandate)
    assert "backtest.fee_bps" in str(exc.value)
    assert "backtest.slippage_bps" in str(exc.value)
    assert settings.backtest.fee_bps == 1.0  # untouched
    assert settings.backtest.slippage_bps == 1.0


def test_apply_overrides_none_is_noop(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate")
    assert apply_overrides(settings, None) == []
    assert settings.promotion.metric == "sharpe"


def test_metric_flag_beats_overlay_via_apply_order(tmp_path):
    """The CLI applies --metric AFTER the overlay, so a one-off flag still wins."""
    settings = _settings(tmp_path, tmp_path / "mandate")
    apply_overrides(settings, _mandate_with({"promotion.metric": "total_return"}))
    assert settings.promotion.metric == "total_return"
    settings.promotion.metric = "sortino"  # what --metric does, after the overlay
    assert settings.promotion.metric == "sortino"


# ── auto selector + profiles catalog ─────────────────────────────────────────────────────
def test_auto_returns_selection_instruction(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "aggressive.md", _profile("Aggressive, volatile names."))
    _write(mandate_dir / "profiles" / "conservative.md", _profile("Conservative, steady names."))
    settings = _settings(tmp_path, mandate_dir, selector="auto")
    m = resolve_mandate(settings)
    assert m.source == "auto"
    assert m.config_overrides == {}  # structurally inert under auto
    assert "aggressive" in m.text and "conservative" in m.text
    assert "declare" in m.text.lower()


def test_auto_instruction_selects_on_sharpe(tmp_path):
    """The auto text must instruct selection on the neutral Sharpe basis (§7)."""
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "aggressive.md", _profile("Aggressive, volatile names."))
    settings = _settings(tmp_path, mandate_dir, selector="auto")
    m = resolve_mandate(settings)
    assert "Sharpe" in m.text  # the neutral yardstick is named
    assert "sharpe" in m.text  # references the champion-board field
    assert "mandate_source" in m.text  # attribution evidence
    # It must be explicit that a total_return-tuning profile can't win on its own metric.
    assert "total_return" in m.text


def test_profiles_catalog(tmp_path):
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "aggressive.md", _profile("Aggressive."))
    _write(mandate_dir / "profiles" / "conservative.md", _profile("Conservative."))
    catalog = profiles_catalog(mandate_dir)
    assert catalog == [
        {"name": "aggressive", "summary": "Aggressive."},
        {"name": "conservative", "summary": "Conservative."},
    ]


def test_profiles_catalog_missing_dir_is_empty(tmp_path):
    assert profiles_catalog(tmp_path / "nope") == []


# ── the seam: block, guardrail, parity ───────────────────────────────────────────────────
def test_mandate_block_keeps_guardrail_verbatim():
    assert "OPERATOR MANDATE" in _MANDATE_BLOCK
    assert "search prior, not a suspension of arithmetic" in _MANDATE_BLOCK
    assert "never overrides the gates, the protocol, or the honesty rules" in _MANDATE_BLOCK


def test_empty_input_is_today_parity(tmp_path):
    """mandate=None injects no block — identical to the historic no-directive path."""
    from noctis.research import build_system_prompt
    from tests.test_research_tools import _make_toolbox

    settings = _settings(tmp_path, tmp_path / "mandate", selector=None)
    assert resolve_mandate(settings) is None

    toolbox = _make_toolbox(tmp_path)
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10, mandate=None)
    assert "OPERATOR MANDATE" not in prompt


def test_kickoff_carries_summary_not_body(tmp_path):
    from noctis.research import run_agent_research
    from tests.test_agent_research import NO_CAPS, FakeLLM, text_turn
    from tests.test_research_tools import _make_toolbox

    toolbox = _make_toolbox(tmp_path)
    mandate = Mandate(
        text="DEEP volatile-names detail that must not be embedded twice",
        source="cli",
        summary="short one-line summary",
        references=[],
        config_overrides={},
    )
    client = FakeLLM([text_turn()], capabilities=NO_CAPS)
    run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, mandate=mandate)
    kickoff = client.calls[0]["messages"][0]["content"]
    assert "short one-line summary" in kickoff
    assert "DEEP volatile-names detail" not in kickoff


# ── CLI: --directive and --mandate are mutually exclusive ────────────────────────────────
def test_directive_and_mandate_together_errors(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\nstate_dir: {tmp_path}/state/\n"
    )
    result = runner.invoke(
        app, ["research", "--directive", "x", "--mandate", "aggressive", "-c", str(cfg)]
    )
    assert result.exit_code != 0
    assert "not both" in result.output


@pytest.mark.parametrize(
    ("block", "diagnosis"),
    [
        # refused: a promotion threshold is a gate, by design
        ("promotion:\n    max_gap: 5.0\n    metric: sortino", "promotion gates"),
        # unknown: a typo'd path
        ("promotion:\n    metrik: sortino", "promotion.metrik"),
        # invalid: the value fails the section's own validator
        ("promotion:\n    metric: alpha", "unknown metric 'alpha'"),
    ],
)
def test_bad_mandate_overlay_exits_one_and_starts_no_loop(tmp_path, block, diagnosis):
    """Loud at startup: a refused, unknown, or invalid key stops `noctis run` before the
    loop, with the reason on stderr — never a buried warning under a multi-day run."""
    mandate_dir = tmp_path / "mandate"
    _write(mandate_dir / "profiles" / "bad.md", f"---\nconfig:\n  {block}\n---\nGo.\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: paper\nmandate_dir: {mandate_dir}\ndata:\n  lake_dir: {tmp_path}/lake\n"
        f"state_dir: {tmp_path}/state/\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", "--mandate", "bad", "-c", str(cfg)])
    assert result.exit_code == 1
    assert "MANDATE:" in result.stderr
    assert diagnosis in result.stderr
    assert "profile:bad" in result.stderr  # which mandate caused it
    assert "Noctis starting" not in result.stdout  # no loop was entered


# ── the SHIPPED mandate/ folder (Phase 2): reads the real committed files ─────────────────
# These guard that the checked-in mandate/ folder always resolves and stays allowlist-clean.
# They deliberately point at the REAL repo mandate/ + config.yaml, not tmp fixtures.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_MANDATE = _REPO_ROOT / "mandate"
_REPO_CONFIG = _REPO_ROOT / "config.yaml"
_SHIPPED_PROFILES = (
    "aggressive",
    "conservative",
    "long-term",
    "short-term",
    "sector-specialist",
)


def _repo_settings(selector):
    """Real repo config.yaml, mandate_dir pinned to the committed mandate/ (cwd-independent)."""
    s = load_settings(config_path=_REPO_CONFIG, mandate_dir=str(_REPO_MANDATE))
    s.research.mandate = selector
    return s


@pytest.mark.parametrize("name", _SHIPPED_PROFILES)
def test_shipped_profile_resolves_and_declares_promotion_metric_only(name):
    """Every shipped profile declares exactly one key — ``promotion.metric``.

    Tighter than "within the allowlist": a profile the shipped ``auto`` selector may pick
    must never declare a knob that would be refused (now fatal) or silently inert.
    """
    settings = _repo_settings(name)
    mandate = resolve_mandate(settings)  # loads without raising
    assert mandate is not None
    assert mandate.source == f"profile:{name}"
    assert set(mandate.config_overrides) == {"promotion.metric"}
    echoes = apply_overrides(settings, mandate)
    assert len(echoes) == 1
    assert echoes[0].startswith("promotion.metric=")


def test_shipped_profile_metrics_are_expected():
    """Each shipped profile binds the risk dial the folder documents."""
    expected = {
        "aggressive": "total_return",
        "conservative": "sharpe",
        "long-term": "sharpe",
        "short-term": "sortino",
        "sector-specialist": "sharpe",
    }
    for name, metric in expected.items():
        settings = _repo_settings(name)
        mandate = resolve_mandate(settings)
        assert mandate.config_overrides == {"promotion.metric": metric}
        apply_overrides(settings, mandate)
        assert settings.promotion.metric == metric


def test_shipped_profiles_catalog_has_all_five_with_summaries():
    settings = _repo_settings(None)
    catalog = profiles_catalog(settings.mandate_dir)
    assert {c["name"] for c in catalog} == set(_SHIPPED_PROFILES)
    assert all(c["summary"].strip() for c in catalog)


def test_shipped_mandate_md_resolves_to_sortino(tmp_path):
    """The committed MANDATE.md.example — exactly what `noctis init` installs as the local
    MANDATE.md — resolves and binds sortino. Hermetic on purpose: seeded into a tmp
    mandate_dir with default settings, so the test depends on neither the developer's
    gitignored mandate/MANDATE.md nor their local config.yaml (the old form failed on any
    fresh checkout until `noctis init` had run)."""
    mandate_dir = tmp_path / "mandate"
    mandate_dir.mkdir()
    example = _REPO_MANDATE / "MANDATE.md.example"
    (mandate_dir / "MANDATE.md").write_bytes(example.read_bytes())
    settings = load_settings(config_path=tmp_path / "missing.yaml", mandate_dir=str(mandate_dir))
    settings.research.mandate = "MANDATE"
    mandate = resolve_mandate(settings)
    assert mandate is not None
    assert mandate.source == "mandate/MANDATE.md"
    # The balanced Sortino brief survived into the body.
    assert "liquid us large- and mid-cap" in mandate.text.lower()
    assert settings.promotion.metric == "sharpe"  # the base dial, before the overlay
    apply_overrides(settings, mandate)
    # The shipped mandate binds sortino ("Score on Sortino: ... downside deviation is what to
    # minimise").
    assert settings.promotion.metric == "sortino"
