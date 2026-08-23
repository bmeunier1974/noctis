"""Per-segment environment capture — ``observability/environment.py`` (story #139, epic #126).

Every test here builds an :class:`EnvironmentProbes` **by hand**. That is the whole design
constraint: the module never reads a machine, never shells out to ``git`` and never imports an
optional package, so a test needs no hardware, no repository and no subprocess. A test that
shelled out to ``git`` to check the git block would have missed the point — the real probes are
wired once, at the composition root, and are exercised end-to-end there.

The other contract asserted here is **honest degradation**. A machine with no ``psutil``, no git
and none of the optional extras is a completely ordinary Noctis install, so it must produce a
complete environment block whose absent values are explicit ``null`` and whose ``degraded_seams``
names every capability that was missing. Silence would be indistinguishable from "this schema
version had no such field".
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from noctis.observability.environment import (
    GIT_SEAM,
    HARDWARE_SEAM,
    LOCKFILE_SEAM,
    EnvironmentProbes,
    capture,
    hostname_hash,
)

ENVIRONMENT_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/observability/environment.py"

# One representative extra set, spelled the way the installer spells it (``noctis.onboarding``).
EXTRAS_PRESENT = {
    "llm": "0.120.0",
    "data": "0.82.0",
    "research": "1.0.0",
    "engine": None,
    HARDWARE_SEAM: "7.2.2",
}

GIT = {
    "commit": "a380d3a4f1c0b9e2d5a6f7c8b9a0d1e2f3a4b5c6",
    "branch": "main",
    "dirty": False,
    "describe": "v0.1.0-42-ga380d3a",
}

LOCKFILE = b"# a uv.lock, byte for byte\n"


def _probes(**overrides) -> EnvironmentProbes:
    """A fully-populated probe set — one machine, stated by hand."""
    base = dict(
        hostname=lambda: "quant-box-01",
        os_facts=lambda: {
            "system": "Linux",
            "release": "7.0.0-14-generic",
            "arch": "x86_64",
            "container": True,
        },
        hardware=lambda: {
            "model": "AMD Ryzen 9 7950X",
            "cores_physical": 16,
            "cores_logical": 32,
            "freq_max_mhz": 5881.0,
            "memory_total_bytes": 67351248896,
            "disk_free_bytes": 412000000000,
        },
        versions=lambda: {"python": "3.11.9", "noctis": "0.1.0"},
        git=lambda: dict(GIT),
        lockfile=lambda: LOCKFILE,
        extras=lambda: dict(EXTRAS_PRESENT),
    )
    base.update(overrides)
    return EnvironmentProbes(**base)  # type: ignore[arg-type]


def _bare_probes() -> EnvironmentProbes:
    """The core install on a machine with no git checkout: everything degrades, nothing raises."""
    return _probes(
        os_facts=lambda: {"system": "Linux", "release": "7.0.0-14-generic", "arch": "x86_64"},
        hardware=lambda: {"cores_logical": 8},
        git=lambda: None,
        lockfile=lambda: None,
        extras=lambda: dict.fromkeys(EXTRAS_PRESENT, None),
    )


# ── what one machine's environment says ────────────────────────────────────────────────────


def test_the_environment_captures_hardware_os_python_git_lockfile_digest_and_extras():
    """Everything a run's throughput has to be attributed to, in one block."""
    block = capture(_probes()).as_dict()

    assert block["os"] == {"system": "Linux", "release": "7.0.0-14-generic", "arch": "x86_64"}
    assert block["container"] is True
    assert block["cpu"] == {
        "model": "AMD Ryzen 9 7950X",
        "cores_physical": 16,
        "cores_logical": 32,
        "freq_max_mhz": 5881.0,
    }
    assert block["memory_total_bytes"] == 67351248896
    assert block["disk_free_bytes"] == 412000000000
    assert block["python"] == "3.11.9"
    assert block["noctis_version"] == "0.1.0"
    assert block["git"] == GIT
    assert block["lockfile_digest"] == "sha256:" + hashlib.sha256(LOCKFILE).hexdigest()
    assert block["extras_present"] == EXTRAS_PRESENT


def test_capturing_the_same_probes_twice_returns_identical_blocks():
    assert capture(_probes()).as_dict() == capture(_probes()).as_dict()


def test_the_lockfile_digest_is_taken_over_the_bytes_handed_in():
    """The resolution the run ran under, pinned: two different locks are two environments."""
    other = capture(_probes(lockfile=lambda: b"a different resolution\n")).as_dict()

    assert other["lockfile_digest"] != capture(_probes()).as_dict()["lockfile_digest"]


# ── the hostname is hashed, never stored (coherent with the run lock, story #129) ───────────


def test_the_hostname_is_hashed_and_the_raw_name_never_appears():
    block = capture(_probes()).as_dict()

    assert block["hostname_hash"] == hostname_hash("quant-box-01")
    assert "quant-box-01" not in str(block)


def test_the_hostname_hash_is_the_shape_the_run_lock_already_writes():
    """#129 hashes the hostname into ``run.lock`` for privacy and portability: ``sha256[:12]``.
    The record keeps that choice coherent by reading this one function — a run store test then
    pins that the lock and the record agree on a real machine."""
    assert hostname_hash("quant-box-01") == hashlib.sha256(b"quant-box-01").hexdigest()[:12]


def test_a_machine_that_will_not_name_itself_reports_a_null_hash():
    assert capture(_probes(hostname=lambda: None)).as_dict()["hostname_hash"] is None


# ── degradation is explicit: null values, named seams ──────────────────────────────────────


def test_no_git_no_psutil_no_extras_still_yields_a_complete_block_of_explicit_nulls():
    """The acceptance criterion, in one assertion set: a bare core install on a machine with no
    checkout produces every key, with ``null`` where a fact was unavailable."""
    block = capture(_bare_probes()).as_dict()

    assert block["git"] is None
    assert block["lockfile_digest"] is None
    assert block["cpu"] == {
        "model": None,
        "cores_physical": None,
        "cores_logical": 8,
        "freq_max_mhz": None,
    }
    assert block["memory_total_bytes"] is None
    assert block["disk_free_bytes"] is None
    assert block["container"] is None
    assert set(block["extras_present"]) == set(EXTRAS_PRESENT)
    assert set(block["extras_present"].values()) == {None}


def test_every_absent_capability_is_named_in_the_degraded_seams():
    seams = capture(_bare_probes()).as_dict()["degraded_seams"]

    assert GIT_SEAM in seams
    assert LOCKFILE_SEAM in seams
    assert HARDWARE_SEAM in seams  # the psutil extra — the stdlib subset was used instead
    assert set(EXTRAS_PRESENT) <= set(seams)
    assert seams == sorted(seams)  # a stable order, so two records diff cleanly


def test_a_present_extra_is_not_a_degraded_seam():
    seams = capture(_probes()).as_dict()["degraded_seams"]

    assert seams == ["engine"]  # the only one this machine is missing


def test_a_probe_that_raises_degrades_to_null_and_names_its_seam():
    """A reporting artifact must never take down a run: a probe that blows up on an exotic
    platform costs one null and one named seam, never an exception into the engine."""

    def boom():
        raise OSError("no /proc on this platform")

    block = capture(_probes(hardware=boom, git=boom)).as_dict()

    assert block["cpu"] == dict.fromkeys(
        ("model", "cores_physical", "cores_logical", "freq_max_mhz")
    )
    assert block["memory_total_bytes"] is None
    assert block["git"] is None
    assert HARDWARE_SEAM in block["degraded_seams"]
    assert GIT_SEAM in block["degraded_seams"]


def test_the_extra_names_are_the_ones_the_installer_already_knows():
    """One list of optional components, not two: the environment block reports exactly the
    extras ``noctis setup`` probes for, so "degraded seam" and "missing extra" are one notion."""
    from noctis.onboarding import EXTRA_MODULES

    assert set(EXTRAS_PRESENT) == set(EXTRA_MODULES)
    assert HARDWARE_SEAM in EXTRA_MODULES


def test_psutil_is_an_optional_extra_and_never_a_core_dependency():
    """AGENTS.md's seam discipline: the full suite and bare paper mode run on the core install
    alone, so a hardware nicety may not become a core dependency."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]

    assert not any("psutil" in dep for dep in project["dependencies"])
    assert any("psutil" in dep for dep in project["optional-dependencies"][HARDWARE_SEAM])


def test_a_captured_block_is_exactly_what_the_record_schema_requires():
    """The ratchet between this module and the record contract: a fully-populated capture and a
    fully-degraded one both satisfy ``schema.validate`` on a segment, key for key."""
    from noctis.reporting import schema
    from noctis.reporting.run_record import RunArtifacts, SegmentArtifact, build
    from noctis.reporting.run_tree.store import read_engine_identity

    for probes in (_probes(), _bare_probes()):
        record = build(
            RunArtifacts(
                run_id="20260727T142233Z-a1b2c3",
                created_utc="2026-07-27T14:22:33.418Z",
                last_active_utc="2026-07-27T14:22:33.418Z",
                engine=read_engine_identity("sharpe"),
                segments=(
                    SegmentArtifact(
                        index=0,
                        started_utc="2026-07-27T14:22:33.418Z",
                        environment=capture(probes).as_dict(),
                    ),
                ),
            )
        )

        assert schema.validate(record) == []
        assert record["environment_latest"] == capture(probes).as_dict()


# ── purity, structurally ───────────────────────────────────────────────────────────────────


def test_the_module_reaches_no_machine_no_subprocess_and_no_config():
    """All probes are injected. Nothing here may import ``platform``, ``os``, ``subprocess`` or
    the settings — the real probes live at the composition root, and this module only shapes
    what they hand back."""
    tree = ast.parse(ENVIRONMENT_SOURCE.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "collections", "dataclasses", "hashlib", "typing"}
    # The allowlist above rules out ``os``, ``platform``, ``socket``, ``subprocess`` and
    # ``pathlib``; the one remaining way to reach the world without an import is a builtin.
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"open", "input", "exec", "eval", "compile", "__import__"}
