"""The operator mandate loader, overlay, and seam (Phase 1).

Pure, client-free coverage of noctis.research.mandate: selector precedence, name lookup,
the empty-MANDATE rule, front-matter parsing (incl. the malformed-fence degrade), reference
inclusion/caps/confinement, summary extraction, and the metric-only config overlay allowlist
(with the assignment-validation subtlety it exists to guard). Plus the CLI mutual-exclusion
of --directive/--mandate and the "empty input == today" parity.
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


# ── the config overlay allowlist (§3.4) ──────────────────────────────────────────────────
def test_overlay_applies_metric_only(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate")
    assert settings.promotion.metric == "sharpe"  # base default
    mandate = Mandate(
        text="x",
        source="profile:test",
        summary="x",
        references=[],
        config_overrides={"promotion.metric": "total_return"},
    )
    lines = apply_overrides(settings, mandate)
    assert settings.promotion.metric == "total_return"
    assert lines == ["promotion.metric=total_return"]


def test_overlay_refuses_everything_but_metric(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate")
    base_mode = settings.mode
    mandate = Mandate(
        text="x",
        source="profile:test",
        summary="x",
        references=[],
        config_overrides={
            "mode": "live",
            "risk.max_daily_loss_pct": 99.0,
            "data.budget_usd": 9999.0,
            "promotion.max_gap": 5.0,
            "promotion.min_test_metric": -5.0,
            "promotion.min_holdout_metric": -5.0,
            "promotion.metric": "total_return",  # the sole allowed key
        },
    )
    lines = apply_overrides(settings, mandate)
    # Every refused key is untouched; only the metric moved.
    assert settings.mode == base_mode
    assert settings.risk.max_daily_loss_pct == 3.0
    assert settings.data.budget_usd == 125.0
    assert settings.promotion.max_gap == 1.0
    assert settings.promotion.min_test_metric == 0.0
    assert settings.promotion.min_holdout_metric == 0.0
    assert settings.promotion.metric == "total_return"
    assert lines == ["promotion.metric=total_return"]


def test_overlay_refuses_invalid_metric_value(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate")
    # The subtlety apply_overrides guards: raw assignment does NOT validate (no
    # validate_assignment on these models), so a hand-check is the only protection.
    settings.promotion.metric = "bogus"
    assert settings.promotion.metric == "bogus"  # proves assignment is unchecked
    settings.promotion.metric = "sharpe"

    mandate = Mandate(
        text="x",
        source="profile:test",
        summary="x",
        references=[],
        config_overrides={"promotion.metric": "not_a_metric"},
    )
    lines = apply_overrides(settings, mandate)
    assert settings.promotion.metric == "sharpe"  # invalid value refused, left unchanged
    assert lines == []


def test_overlay_cannot_touch_backtest_costs(tmp_path):
    """The mandate config: overlay stays promotion.metric-only — it may steer WHAT to look
    for, never how forgiving the arena is (#23). A backtest cost override is refused and the
    section is left on its floored default."""
    settings = _settings(tmp_path, tmp_path / "mandate")
    assert settings.backtest.fee_bps == 1.0
    assert settings.backtest.slippage_bps == 1.0
    mandate = Mandate(
        text="x",
        source="profile:test",
        summary="x",
        references=[],
        config_overrides={
            "backtest.fee_bps": 0.1,  # a cost-cheapening attempt via the overlay
            "backtest.slippage_bps": 0.1,
        },
    )
    lines = apply_overrides(settings, mandate)
    assert settings.backtest.fee_bps == 1.0  # untouched
    assert settings.backtest.slippage_bps == 1.0
    assert lines == []  # nothing applied


def test_apply_overrides_none_is_noop(tmp_path):
    settings = _settings(tmp_path, tmp_path / "mandate")
    assert apply_overrides(settings, None) == []
    assert settings.promotion.metric == "sharpe"


def test_metric_flag_beats_overlay_via_apply_order(tmp_path):
    """The CLI applies --metric AFTER the overlay, so a one-off flag still wins."""
    settings = _settings(tmp_path, tmp_path / "mandate")
    mandate = Mandate(
        text="x",
        source="profile:test",
        summary="x",
        references=[],
        config_overrides={"promotion.metric": "total_return"},
    )
    apply_overrides(settings, mandate)
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
def test_shipped_profile_resolves_and_overlay_is_allowlist_clean(name):
    from noctis.research.mandate import _OVERRIDE_ALLOWLIST

    settings = _repo_settings(name)
    mandate = resolve_mandate(settings)  # loads without raising
    assert mandate is not None
    assert mandate.source == f"profile:{name}"
    # No profile puts anything outside the allowlist in its config: block.
    assert set(mandate.config_overrides) <= set(_OVERRIDE_ALLOWLIST)
    # apply_overrides only ever echoes a promotion.metric change (or nothing).
    echoes = apply_overrides(settings, mandate)
    assert all(line.startswith("promotion.metric=") for line in echoes)
    assert len(echoes) <= 1


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
    # The migrated high-risk brief survived into the body.
    assert "volatile" in mandate.text.lower()
    assert settings.promotion.metric == "sharpe"  # the base dial, before the overlay
    apply_overrides(settings, mandate)
    # The shipped mandate binds sortino (f2a3a60: "risk management matters more than hit-rate").
    assert settings.promotion.metric == "sortino"
