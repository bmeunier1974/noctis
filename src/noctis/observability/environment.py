"""What machine, and what inputs — the run record's per-segment environment (story #139).

A run outlives the process that started it, and it may outlive the *machine*: an experiment
stopped each morning and resumed each night can migrate boxes mid-flight, or be resumed after a
``git pull``. So the environment is recorded **per segment**, never per run. Research throughput
is CPU-bound — the sweep fork pool, the walk-forward splits — which makes trials-per-hour and
USD-per-champion comparable across runs only when the machine that produced them is on the
record. A run whose second segment ran on twice the cores must not have its first segment's
throughput attributed to that hardware.

**Every probe is injected, and that is the whole design.** This module reads no ``/proc``,
imports no ``platform``, shells out to no ``git`` and touches no optional package: it takes an
:class:`EnvironmentProbes` — seven callables — and shapes what they hand back into the record's
block. The real probes are wired once, at the composition root (:mod:`noctis.bootstrap`), so a
test needs no hardware, no repository and no subprocess, and a structural test pins that by AST.

**Degradation is explicit, never silent.** ``psutil`` stays an optional extra (the ``hardware``
one) rather than becoming a core dependency, git capture degrades to ``null`` outside a
repository, and an install without the optional stacks is a completely ordinary Noctis install.
So every absent value is an explicit ``null`` **and** the capability that was missing is named in
:attr:`Environment.degraded_seams` — a reader can then tell "this machine had no ``psutil``" from
"this schema version had no such field". A probe that *raises* (an exotic platform, a container
with no ``/proc``) costs exactly the same: one null and one named seam, never an exception into
the engine.

**The seams are the ones the installer already knows.** ``extras_present`` is keyed by the
optional-extra names ``noctis setup`` probes for (:data:`noctis.onboarding.EXTRA_MODULES`), so
"missing extra" and "degraded seam" are one notion with one list behind it, and the remedy the
record implies (``uv sync --extra <name>``) is the one the operator can actually type.

**The hostname is hashed, never stored.** Story #129 deliberately writes ``sha256(hostname)[:12]``
into ``run.lock`` — comparable across segments, not doxxing — and the record keeps that choice
coherent by reading the same :func:`hostname_hash` here. The raw name arrives from a probe and
leaves as a digest; it never reaches the record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "CPU_KEYS",
    "ENVIRONMENT_KEYS",
    "EXTRAS_SEAM",
    "GIT_KEYS",
    "GIT_SEAM",
    "HARDWARE_SEAM",
    "HOSTNAME_SEAM",
    "LOCKFILE_SEAM",
    "OS_KEYS",
    "OS_SEAM",
    "VERSIONS_SEAM",
    "Environment",
    "EnvironmentProbes",
    "capture",
    "hostname_hash",
]

# The seam names a degraded capture reports. The first is also an **extra** name
# (``pyproject.toml``'s ``hardware`` = ``psutil``), deliberately: whether the richer CPU/memory
# facts are missing because the extra is not installed or because the probe failed, the operator's
# remedy is the same one, so it is the same word.
HARDWARE_SEAM = "hardware"
GIT_SEAM = "git"
LOCKFILE_SEAM = "lockfile"
HOSTNAME_SEAM = "hostname"
VERSIONS_SEAM = "versions"
EXTRAS_SEAM = "extras"
OS_SEAM = "os"

# The keys each sub-block always carries — presence is the contract, ``null`` is a value. Stated
# here rather than inferred from whatever a probe returned, so a probe that learns a new fact
# cannot silently widen the record and a probe that loses one cannot silently narrow it.
OS_KEYS = ("system", "release", "arch")
CPU_KEYS = ("model", "cores_physical", "cores_logical", "freq_max_mhz")
GIT_KEYS = ("commit", "branch", "dirty", "describe")

# The block's own top-level keys, in the order the record writes them.
ENVIRONMENT_KEYS = (
    "hostname_hash",
    "os",
    "container",
    "cpu",
    "memory_total_bytes",
    "disk_free_bytes",
    "python",
    "noctis_version",
    "git",
    "lockfile_digest",
    "extras_present",
    "degraded_seams",
)

# ``sha256(hostname)[:12]`` — the exact shape ``run.lock`` has written since story #129.
_HOST_DIGEST_CHARS = 12


def hostname_hash(name: str) -> str:
    """A stable, non-identifying host id: ``sha256(hostname)[:12]``.

    The **one** implementation, shared by the run lock and the record. Two segments on one machine
    are provably the same host without a machine name ever being published, and because both
    readers call this function they can never drift into two different answers.
    """
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:_HOST_DIGEST_CHARS]


@dataclass(frozen=True)
class EnvironmentProbes:
    """The seven facts this module cannot compute for itself, each as an injected callable.

    Deliberately **without defaults**: a probe nobody supplied would be a fact nobody noticed was
    missing, and the composition root is the one place that knows how to read this machine. Each
    returns plain data (or ``None``); each is allowed to raise, which :func:`capture` turns into
    a null value and a named degraded seam rather than an exception into the engine.
    """

    # The raw machine name. Hashed on the way in — see :func:`hostname_hash`.
    hostname: Callable[[], str | None]
    # ``{"system", "release", "arch", "container"}``; a partial mapping is fine (missing keys
    # become explicit nulls), which is what lets a platform that cannot answer one of them answer
    # the rest.
    os_facts: Callable[[], Mapping[str, object]]
    # ``{"model", "cores_physical", "cores_logical", "freq_max_mhz", "memory_total_bytes",
    # "disk_free_bytes"}``. The seam ``psutil`` enriches: without the extra the stdlib subset
    # answers what it can and the rest is null.
    hardware: Callable[[], Mapping[str, object]]
    # ``{"python", "noctis"}``.
    versions: Callable[[], Mapping[str, object]]
    # ``{"commit", "branch", "dirty", "describe"}``, or ``None`` outside a repository.
    git: Callable[[], Mapping[str, object] | None]
    # The lockfile's bytes (``uv.lock``), digested here so the whole dependency resolution is
    # pinned by one value; ``None`` when there is no lockfile beside the install.
    lockfile: Callable[[], bytes | None]
    # ``{extra name: version or None}`` over the optional extras the installer knows.
    extras: Callable[[], Mapping[str, str | None]]


@dataclass(frozen=True)
class Environment:
    """One machine, as one segment of one run saw it.

    Built only by :func:`capture`, and rendered by :meth:`as_dict` into the block the record
    carries on every segment (and, for the newest one, at ``environment_latest``).
    """

    hostname_hash: str | None
    os: Mapping[str, object]
    container: bool | None
    cpu: Mapping[str, object]
    memory_total_bytes: int | None
    disk_free_bytes: int | None
    python: str | None
    noctis_version: str | None
    git: Mapping[str, object] | None
    lockfile_digest: str | None
    extras_present: Mapping[str, str | None]
    degraded_seams: Sequence[str]

    def as_dict(self) -> dict:
        """The record's environment block: every key present, absent values explicit ``null``."""
        return {
            "hostname_hash": self.hostname_hash,
            "os": dict(self.os),
            "container": self.container,
            "cpu": dict(self.cpu),
            "memory_total_bytes": self.memory_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "python": self.python,
            "noctis_version": self.noctis_version,
            "git": dict(self.git) if self.git is not None else None,
            "lockfile_digest": self.lockfile_digest,
            "extras_present": dict(self.extras_present),
            "degraded_seams": list(self.degraded_seams),
        }


def capture(probes: EnvironmentProbes) -> Environment:
    """Shape one machine's facts into an :class:`Environment`. Pure over what the probes return.

    Each probe is read exactly once, defensively: a raised exception is *evidence* (the seam is
    named degraded) rather than a failure, because the environment block is reporting and
    reporting must never take a multi-week run down. Nothing here reads a clock, a file or the
    configuration, so the same probes always yield the same block.
    """
    degraded: set[str] = set()
    os_facts = _mapping(probes.os_facts, OS_SEAM, degraded)
    hardware = _mapping(probes.hardware, HARDWARE_SEAM, degraded)
    versions = _mapping(probes.versions, VERSIONS_SEAM, degraded)
    git = _optional_mapping(probes.git, GIT_SEAM, degraded)
    extras = _extras(probes.extras, degraded)

    raw_host = _value(probes.hostname, HOSTNAME_SEAM, degraded)
    lockfile = _value(probes.lockfile, LOCKFILE_SEAM, degraded)
    if git is None:
        degraded.add(GIT_SEAM)
    if not isinstance(lockfile, bytes | bytearray):
        degraded.add(LOCKFILE_SEAM)
    # A missing extra IS a degraded seam — one notion, one list (see the module docstring).
    degraded |= {name for name, version in extras.items() if version is None}

    return Environment(
        hostname_hash=hostname_hash(raw_host) if isinstance(raw_host, str) and raw_host else None,
        os={key: _scalar(os_facts.get(key)) for key in OS_KEYS},
        container=_bool(os_facts.get("container")),
        cpu={key: _scalar(hardware.get(key)) for key in CPU_KEYS},
        memory_total_bytes=_int(hardware.get("memory_total_bytes")),
        disk_free_bytes=_int(hardware.get("disk_free_bytes")),
        python=_str(versions.get("python")),
        noctis_version=_str(versions.get("noctis")),
        git={key: _scalar(git.get(key)) for key in GIT_KEYS} if git is not None else None,
        lockfile_digest=_lockfile_digest(lockfile),
        extras_present=extras,
        degraded_seams=sorted(degraded),
    )


def _lockfile_digest(raw: object) -> str | None:
    """``sha256:<hex>`` over the lockfile's bytes — the whole resolution, pinned by one value."""
    if not isinstance(raw, bytes | bytearray):
        return None
    return "sha256:" + hashlib.sha256(bytes(raw)).hexdigest()


def _value(probe: Callable[[], object], seam: str, degraded: set[str]) -> object:
    """Read one probe, or name its seam degraded. The only place a probe is ever called."""
    try:
        return probe()
    except Exception:
        degraded.add(seam)
        return None


def _mapping(probe: Callable[[], object], seam: str, degraded: set[str]) -> Mapping[str, object]:
    value = _value(probe, seam, degraded)
    return value if isinstance(value, Mapping) else {}


def _optional_mapping(
    probe: Callable[[], object], seam: str, degraded: set[str]
) -> Mapping[str, object] | None:
    value = _value(probe, seam, degraded)
    return value if isinstance(value, Mapping) else None


def _extras(probe: Callable[[], object], degraded: set[str]) -> dict[str, str | None]:
    value = _value(probe, EXTRAS_SEAM, degraded)
    if not isinstance(value, Mapping):
        return {}
    return {str(name): _str(version) for name, version in value.items()}


def _scalar(value: object) -> object:
    """A record-safe leaf: strings, numbers and bools pass; anything else is an honest null."""
    if isinstance(value, bool | int | float | str):
        return value
    return None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
