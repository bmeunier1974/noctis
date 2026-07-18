"""`noctis setup` — the guided first-run wizard and its surgical file editors.

Every network/subprocess seam (extras probe, uv sync, DataBento verify, LLM verify, the
Ollama probe) is a module-level function stubbed here, so the wizard runs end-to-end with
no network and no installs — and the suite stays green on the bare core install.
"""

from __future__ import annotations

import io
from pathlib import Path

import yaml
from typer.testing import CliRunner

from noctis.cli import app
from noctis.onboarding import probe_ollama, set_config_value, set_env_key

runner = CliRunner()

REPO_TEMPLATE = Path(__file__).resolve().parent.parent / "config.example.yaml"


def _project(tmp_path, monkeypatch) -> Path:
    """A project root the CLI runs in (chdir'd; config + .env resolve here)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOCTIS_CONFIG", str(tmp_path / "config.yaml"))
    (tmp_path / "config.example.yaml").write_text(REPO_TEMPLATE.read_text())
    (tmp_path / ".env.example").write_text("# secrets\nDATABENTO_API_KEY=\nALLOW_LIVE=\n")
    (tmp_path / "mandate").mkdir()
    (tmp_path / "mandate" / "MANDATE.md.example").write_text("# my mandate\n")
    return tmp_path


def _stub_probes(monkeypatch, **overrides):
    """Give the wizard offline answers for every probe; override per-test as needed."""
    import noctis.onboarding as onboarding

    defaults = {
        "missing_extras": lambda: [],
        "verify_databento": lambda key: (True, "verified (stub)"),
        "verify_llm": lambda settings: (True, "stub answered in 0.0s"),
        "probe_ollama": lambda base_url=None: None,
        "run_uv_sync": lambda root: (_ for _ in ()).throw(AssertionError("uv sync ran")),
    }
    for name, fn in {**defaults, **overrides}.items():
        monkeypatch.setattr(onboarding, name, fn)


def _stub_client_ok(monkeypatch, model: str) -> None:
    """Make client_status report a buildable client, independent of installed extras."""
    from noctis.research.llm import ClientStatus

    monkeypatch.setattr(
        "noctis.research.client_status",
        lambda settings: ClientStatus(ok=True, model=model, provider="ollama_chat", reason=None),
    )


def _stub_client_ok_only_for(monkeypatch, ok_model: str) -> None:
    """client_status that is ok *only* once ``ok_model`` is configured.

    The wizard shows the interactive LLM menu only while ``client_status`` is not ok
    (the template default has no key), then re-checks after writing the choice — so a
    blanket-ok stub would skip the menu entirely.
    """
    from noctis.research.llm import ClientStatus

    def status(settings):
        model = settings.research.model
        ok = model == ok_model
        return ClientStatus(
            ok=ok, model=model, provider="ollama_chat", reason=None if ok else "no API key"
        )

    monkeypatch.setattr("noctis.research.client_status", status)


# ── the dotenv editor ─────────────────────────────────────────────────────────────────────
def test_set_env_key_fills_the_template_slot_preserving_everything_else(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# secrets live here\nDATABENTO_API_KEY=\n\nALLOW_LIVE=\n")
    set_env_key(env, "DATABENTO_API_KEY", "db-123")
    assert env.read_text() == "# secrets live here\nDATABENTO_API_KEY=db-123\n\nALLOW_LIVE=\n"


def test_set_env_key_appends_when_absent_and_ignores_commented_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# OPENAI_API_KEY=old-comment\nALLOW_LIVE=\n")
    set_env_key(env, "OPENAI_API_KEY", "sk-1")
    assert env.read_text() == "# OPENAI_API_KEY=old-comment\nALLOW_LIVE=\nOPENAI_API_KEY=sk-1\n"


def test_set_env_key_creates_a_missing_file(tmp_path):
    env = tmp_path / "sub" / ".env"
    set_env_key(env, "DATABENTO_API_KEY", "db-9")
    assert env.read_text() == "DATABENTO_API_KEY=db-9\n"


# ── the config editor ─────────────────────────────────────────────────────────────────────
def test_set_config_value_replaces_in_place_keeping_the_inline_comment():
    text = "mode: paper # paper or live\n\nchampion_count: 3\n"
    out = set_config_value(text, "champion_count", "5")
    assert out == "mode: paper # paper or live\n\nchampion_count: 5\n"


def test_set_config_value_replaces_a_nested_value_and_skips_commented_twins():
    text = "research:\n  # model: commented-out-example\n  model: openai/gpt-5.4 # seam\n"
    out = set_config_value(text, "research.model", "ollama_chat/x:1b")
    assert "  model: ollama_chat/x:1b # seam" in out
    assert "# model: commented-out-example" in out


def test_set_config_value_appends_inside_the_right_section_not_under_the_next():
    text = "research:\n  model: a\n\n# promotion banner\npromotion:\n  metric: sharpe\n"
    out = set_config_value(text, "research.mandate", "aggressive")
    parsed = yaml.safe_load(out)
    assert parsed == {
        "research": {"model": "a", "mandate": "aggressive"},
        "promotion": {"metric": "sharpe"},
    }
    # appended before the next section's banner, not after it
    assert out.index("mandate: aggressive") < out.index("# promotion banner")


def test_set_config_value_builds_missing_sections_from_empty_text():
    out = set_config_value("", "research.agent.max_tokens", "4096")
    assert yaml.safe_load(out) == {"research": {"agent": {"max_tokens": 4096}}}


def test_the_shipped_template_stays_wizard_editable():
    """The regression guard the whole feature hangs on: the wizard must be able to rewire
    the real template (local-backend model + output cap) without disturbing anything else."""
    text = REPO_TEMPLATE.read_text()
    text = set_config_value(text, "research.model", "ollama_chat/noctis-qwen3:14b")
    text = set_config_value(text, "research.agent.max_tokens", "4096")
    parsed = yaml.safe_load(text)
    assert parsed["research"]["model"] == "ollama_chat/noctis-qwen3:14b"
    assert parsed["research"]["agent"]["max_tokens"] == 4096
    # untouched neighbours survive with their shipped values
    assert parsed["mode"] == "paper"
    assert parsed["data"]["auto_backfill"] is True
    assert parsed["promotion"]["min_test_activity"] == 0.05


# ── probes ────────────────────────────────────────────────────────────────────────────────
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_probe_ollama_lists_tags_and_returns_none_when_nothing_answers(monkeypatch):
    body = b'{"models": [{"name": "noctis-qwen3:14b"}, {"name": "llama3:8b"}]}'
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: _Resp(body))
    assert probe_ollama() == ["noctis-qwen3:14b", "llama3:8b"]

    def _refuse(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _refuse)
    assert probe_ollama() is None


# ── the wizard, end to end ────────────────────────────────────────────────────────────────
def test_check_mode_reports_gaps_read_only_and_exits_nonzero(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch)
    result = runner.invoke(app, ["setup", "--check"])
    assert result.exit_code == 1, result.output
    assert "MISSING" in result.output
    assert "issue(s):" in result.output
    assert not (tmp_path / "config.yaml").exists()  # read-only: nothing scaffolded
    assert not (tmp_path / ".env").exists()


def test_flag_driven_setup_scaffolds_writes_key_and_model_and_verifies(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch)
    _stub_client_ok(monkeypatch, "ollama_chat/fake:1b")
    result = runner.invoke(
        app,
        ["setup", "--yes", "--databento-key", "db-42", "--model", "ollama_chat/fake:1b"],
    )
    assert result.exit_code == 0, result.output
    assert "DATABENTO_API_KEY=db-42" in (root / ".env").read_text()
    parsed = yaml.safe_load((root / "config.yaml").read_text())
    assert parsed["research"]["model"] == "ollama_chat/fake:1b"
    assert parsed["research"]["agent"]["max_tokens"] == 4096  # the local-backend output cap
    assert "verified — stub answered" in result.output
    # the scaffolded template's own knobs survived the surgical edit
    assert parsed["data"]["auto_backfill"] is True


def test_setup_then_check_round_trip_is_healthy(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch)
    _stub_client_ok(monkeypatch, "ollama_chat/fake:1b")
    assert (
        runner.invoke(
            app, ["setup", "--yes", "--databento-key", "db-42", "--model", "ollama_chat/fake:1b"]
        ).exit_code
        == 0
    )
    result = runner.invoke(app, ["setup", "--check"])
    assert result.exit_code == 0, result.output
    assert "Everything checks out." in result.output


def test_interactive_skip_everything_lands_on_the_legacy_loop(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch)
    # Enter at the DataBento prompt (skip), then "4" at the LLM menu (skip).
    result = runner.invoke(app, ["setup"], input="\n4\n")
    assert result.exit_code == 0, result.output
    assert "legacy" in result.output
    assert "no DATABENTO_API_KEY" in result.output


def test_interactive_key_paste_is_acknowledged_masked(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch)
    # Paste at the hidden-input DataBento prompt (nothing echoes), then "4" skips the LLM
    # menu. The masked receipt is the only proof the paste landed.
    result = runner.invoke(app, ["setup"], input="db-SECRETMIDDLE1234\n4\n")
    assert result.exit_code == 0, result.output
    assert "received db-S…1234 (19 chars)" in result.output
    assert "SECRETMIDDLE" not in result.output  # the secret itself is never echoed
    assert "DATABENTO_API_KEY=db-SECRETMIDDLE1234" in (root / ".env").read_text()


def test_interactive_ollama_menu_lists_detected_tags_noctis_first(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _stub_probes(
        monkeypatch,
        probe_ollama=lambda base_url=None: ["llama3:8b", "noctis-qwen3:14b", "qwen2.5:7b"],
    )
    _stub_client_ok_only_for(monkeypatch, "ollama_chat/qwen2.5:7b")
    # Enter skips DataBento; Enter takes the LLM-menu default (3 — a server was detected);
    # "3" picks the third *listed* tag, which is qwen2.5:7b because noctis-* sorts first.
    result = runner.invoke(app, ["setup"], input="\n\n3\n")
    assert result.exit_code == 0, result.output
    assert "1. noctis-qwen3:14b  (agent-ready)" in result.output
    assert "2. llama3:8b" in result.output
    assert "3. qwen2.5:7b" in result.output
    parsed = yaml.safe_load((root / "config.yaml").read_text())
    assert parsed["research"]["model"] == "ollama_chat/qwen2.5:7b"


def test_interactive_ollama_menu_default_is_the_first_noctis_tag(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _stub_probes(
        monkeypatch,
        probe_ollama=lambda base_url=None: ["llama3:8b", "noctis-qwen3:14b"],
    )
    _stub_client_ok_only_for(monkeypatch, "ollama_chat/noctis-qwen3:14b")
    # Enter, Enter, Enter: skip DataBento, accept menu choice 3, accept tag choice 1.
    result = runner.invoke(app, ["setup"], input="\n\n\n")
    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load((root / "config.yaml").read_text())
    assert parsed["research"]["model"] == "ollama_chat/noctis-qwen3:14b"


def test_interactive_ollama_menu_accepts_a_typed_tag(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch, probe_ollama=lambda base_url=None: ["noctis-qwen3:14b"])
    _stub_client_ok_only_for(monkeypatch, "ollama_chat/mistral:7b")
    result = runner.invoke(app, ["setup"], input="\n\nmistral:7b\n")
    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load((root / "config.yaml").read_text())
    assert parsed["research"]["model"] == "ollama_chat/mistral:7b"


def test_declined_install_prints_the_command_and_runs_nothing(tmp_path, monkeypatch):
    import noctis.onboarding as onboarding

    _project(tmp_path, monkeypatch)
    _stub_probes(monkeypatch, missing_extras=lambda: ["llm", "data"])
    monkeypatch.setattr(onboarding, "_can_uv_sync", lambda root: True)
    # "n" declines the install, Enter skips DataBento, "4" skips the LLM menu.
    result = runner.invoke(app, ["setup"], input="n\n\n4\n")
    assert result.exit_code == 0, result.output
    assert "missing: llm, data" in result.output
    assert "uv sync --all-extras" in result.output  # the stubbed runner would raise if called


def test_setup_never_overwrites_operator_edits(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    (root / "config.yaml").write_text(
        "# my notes\nmode: paper\nchampion_count: 5 # my pick\nresearch:\n  model: openai/gpt-5.4\n"
    )
    (root / ".env").write_text("DATABENTO_API_KEY=already-mine\n")
    _stub_probes(monkeypatch)
    _stub_client_ok(monkeypatch, "ollama_chat/fake:1b")
    result = runner.invoke(app, ["setup", "--yes", "--model", "ollama_chat/fake:1b"])
    assert result.exit_code == 0, result.output
    text = (root / "config.yaml").read_text()
    assert "# my notes" in text and "champion_count: 5 # my pick" in text
    assert yaml.safe_load(text)["research"]["model"] == "ollama_chat/fake:1b"
    assert (root / ".env").read_text() == "DATABENTO_API_KEY=already-mine\n"  # kept, not re-asked
