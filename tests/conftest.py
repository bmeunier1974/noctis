"""Shared test fixtures."""

from __future__ import annotations

import pytest

from noctis.config.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate each test from the developer's real environment, ``.env``, and config.

    Clears the safety-gate and secret env vars plus NOCTIS_CONFIG, and disables reading a
    developer ``.env`` file, so tests see defaults rather than whatever is on the machine.
    """
    # The safety gate + secrets are PINNED EMPTY, not deleted. The research seam lazily
    # ``import litellm``, and litellm calls python-dotenv's ``load_dotenv(override=False)`` on
    # import — which would otherwise repopulate these from the developer's real ``.env`` mid-test,
    # leaking secrets and (worse) letting a paid provider client build and make real API calls.
    # A present-but-empty value is falsy for every check here AND, because it already exists in the
    # environment, ``load_dotenv(override=False)`` will not reintroduce the real value.
    for var in (
        "ALLOW_LIVE",
        "DATABENTO_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(var, "")
    # Config selectors are safe to remove outright (not in .env; empty values would misparse).
    for var in ("MODE", "CHAMPION_COUNT"):
        monkeypatch.delenv(var, raising=False)
    # The workspace root is PINNED to this test's tmp_path (never deleted, never empty — an
    # empty string would misparse as a path). With it unset, any test that reaches a derived
    # output path (memory, reports, state) would write ``workspace/…`` relative to the CWD —
    # i.e. into the developer's repo checkout. Pinning makes ambient writes structurally
    # impossible; tests that assert the *default* derivation delenv it locally.
    monkeypatch.setenv("NOCTIS_WORKSPACE", str(tmp_path / "workspace"))
    # NOCTIS_CONFIG is *redirected*, not deleted: with it unset, Settings() falls back to
    # ./config.yaml — the repo file — so any operator edit there (cost_profile, mandate,
    # universe) would leak into tests that assert on shipped defaults. Pointing at a
    # nonexistent path makes Settings() resolve to pure defaults; tests that want a real
    # file still pass config_path explicitly (which re-sets this variable themselves).
    monkeypatch.setenv("NOCTIS_CONFIG", "tests-do-not-read-config.yaml/does-not-exist.yaml")
    # Don't let a real .env (with the developer's secrets) leak into tests.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    yield


@pytest.fixture
def fast_gate(monkeypatch):
    """Run the write gate in-process via the library's validator seam.

    Same checks and error contract as the production subprocess runner, minus an
    interpreter spawn per write/rewrite/promotion — for tests that assert gate
    *outcomes*. Tests proving subprocess isolation itself stay on the default.
    """
    from noctis.strategies import library

    monkeypatch.setattr(library, "validator", library.validate_in_process)
