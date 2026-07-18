"""Tests for the paper/live double-gate."""

from __future__ import annotations

import pytest

from noctis.config import SafetyGateError, load_settings, resolve_execution_mode


def _settings(mode: str, allow_live: bool):
    return load_settings(mode=mode, allow_live=allow_live)


def test_paper_unset_resolves_paper():
    assert resolve_execution_mode(_settings("paper", False)) == "paper"


def test_paper_with_allow_live_still_paper():
    """Config wins toward safety: paper stays paper even with ALLOW_LIVE=true."""
    assert resolve_execution_mode(_settings("paper", True)) == "paper"


def test_live_without_allow_live_raises():
    """mode=live without the env gate refuses to start (no silent downgrade)."""
    with pytest.raises(SafetyGateError):
        resolve_execution_mode(_settings("live", False))


def test_live_with_allow_live_resolves_live():
    assert resolve_execution_mode(_settings("live", True)) == "live"


@pytest.mark.parametrize(
    ("mode", "allow_live", "expected"),
    [
        ("paper", False, "paper"),
        ("paper", True, "paper"),
        ("live", True, "live"),
    ],
)
def test_gate_matrix(mode, allow_live, expected):
    assert resolve_execution_mode(_settings(mode, allow_live)) == expected
