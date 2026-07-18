"""The strategy library — authored one-file Python strategies in ``strategies/``.

Each ``*.py`` in the library defines exactly one :class:`~noctis.strategies.base.TraderStrategy`
subclass and carries its research record in a structured module-docstring header::

    \"\"\"Buy shallow oversold dips inside an uptrend; exit when momentum snaps back.

    status: candidate            # draft | candidate | champion | rejected
    style: mean-reversion
    symbols: AAPL MSFT NVDA      # the panel it was researched/tuned on
    tuned: 2026-07-04            # date the current Params defaults were fitted
    \"\"\"

The header is convention plus the tiny parser here, not a new format. The loader mirrors
``noctis/strategies/spec/strategy.py``'s ``load_and_register``; :func:`write_strategy` is the
validation gate — import in a **fresh interpreter** (via the swappable :data:`validator`
seam), a smoke replay on the synthetic fixture, and a replay of the file's declared
known-outcome scenarios (``noctis.strategies.scenarios``) — folded into the write so an
invalid strategy can never exist on disk (grid-mng's ``validate_spec``, made structural).
:func:`set_header` is the mechanical header stamp (rejections); the approval-time hand-off
is :func:`plan_promotion` → :meth:`PromotionPlan.commit`: the winning parameters become the
``Params`` defaults and the header is re-stamped ``status: champion``, rendered and
gate-validated as one unit *before* the champion registry crowns — so ``noctis backtest
<name>`` with no arguments replays exactly what the agent shipped, and a failed write-back
can never strand a crowned champion.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd

from noctis.strategies import scenarios as scenarios_mod
from noctis.strategies.base import TraderStrategy, params_to_dict, replay_targets
from noctis.strategies.families import FamilyRegistry

logger = logging.getLogger("noctis.library")

TEMPLATE_NAME = "TEMPLATE.py"
# The library is three tiers: committed seeds + TEMPLATE (read-only input), then two output
# folders. ``__tmp/`` is the agent's scratch area (drafts, candidates, rejects — local only);
# ``champions/`` holds locally-promoted champions (never via the public repo).
# Discovery scans seeds → __tmp → champions with later tiers overriding earlier ones,
# so a champion always beats a seed and a live working copy shadows the seed it derives from.
# The tier roots live in a :class:`LibraryPaths`; a bare path still means the historical
# sibling layout (both output tiers under the seeds root).
TMP_SUBDIR = "__tmp"
CHAMPIONS_SUBDIR = "champions"
HEADER_FIELDS = ("status", "style", "symbols", "tuned")
VALID_STATUSES = ("draft", "candidate", "champion", "rejected")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_NS_PER_MINUTE = 60 * 1_000_000_000
_VALIDATE_TIMEOUT_S = 120
_module_counter = itertools.count()


@dataclass(frozen=True)
class LibraryPaths:
    """The three library tier roots, decoupled so committed input and engine output split.

    ``seeds`` is the committed library (the repo's ``strategies/``, read-only to the
    engine); ``tmp`` and ``champions`` are output tiers (under the workspace in the
    standard layout). ``from_single_root`` reproduces the historical sibling layout and
    is how a bare path coerces, so legacy callers and tests stay valid unchanged. A
    frozen dataclass of paths, it pickles through ``ProcessPoolExecutor`` initargs into
    the sweep workers.
    """

    seeds: Path
    tmp: Path
    champions: Path

    @classmethod
    def from_single_root(cls, root: str | Path) -> LibraryPaths:
        base = Path(root)
        return cls(seeds=base, tmp=base / TMP_SUBDIR, champions=base / CHAMPIONS_SUBDIR)

    @classmethod
    def from_settings(cls, settings) -> LibraryPaths:
        work = Path(settings.workspace_dir) / "strategies"
        return cls(
            seeds=Path(settings.strategies_dir),
            tmp=work / TMP_SUBDIR,
            champions=work / CHAMPIONS_SUBDIR,
        )

    @classmethod
    def coerce(cls, value: LibraryPaths | str | Path) -> LibraryPaths:
        """A bare path means the historical single-root layout."""
        return value if isinstance(value, LibraryPaths) else cls.from_single_root(value)


# What every public library function accepts where it used to take a bare directory.
LibrarySpec = LibraryPaths | str | Path


class StrategyValidationError(Exception):
    """The submitted strategy source failed the import/smoke-backtest gate."""


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixture — the smoke-backtest bars (also the ideation parity fixture)
# ─────────────────────────────────────────────────────────────────────────────
def fixture_frame(n: int = 180, seed: int = 7) -> pd.DataFrame:
    """A deterministic OHLCV path that exercises entries and exits (spread-bracketed)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, n).cumsum() + 5.0 * np.sin(np.linspace(0, 6 * np.pi, n))
    return pd.DataFrame(
        {
            "ts_event": [i * _NS_PER_MINUTE for i in range(n)],
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1000] * n,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header parse / write-back
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StrategyHeader:
    thesis: str = ""
    status: str = "draft"
    style: str = ""
    symbols: list[str] = field(default_factory=list)
    tuned: str | None = None

    def to_dict(self) -> dict:
        return {
            "thesis": self.thesis,
            "status": self.status,
            "style": self.style,
            "symbols": list(self.symbols),
            "tuned": self.tuned,
        }


_FIELD_RE = re.compile(rf"^({'|'.join(HEADER_FIELDS)})\s*:\s*(.*)$")


def parse_header(source: str) -> StrategyHeader:
    """Parse the docstring header (thesis first paragraph + ``field: value`` lines)."""
    header = StrategyHeader()
    try:
        doc = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        return header
    thesis_lines: list[str] = []
    in_thesis = True
    for line in doc.splitlines():
        stripped = line.strip()
        match = _FIELD_RE.match(stripped)
        if match:
            in_thesis = False
            value = match.group(2).split("#", 1)[0].strip()
            if match.group(1) == "symbols":
                header.symbols = [
                    s.strip().upper() for s in re.split(r"[,\s]+", value) if s.strip()
                ]
            elif match.group(1) == "tuned":
                header.tuned = value or None
            else:
                setattr(header, match.group(1), value)
            continue
        if in_thesis:
            if not stripped and thesis_lines:
                in_thesis = False
                continue
            if stripped:
                thesis_lines.append(stripped)
    header.thesis = " ".join(thesis_lines)
    return header


def _docstring_span(source: str) -> tuple[int, int]:
    """(start, end) line indexes (0-based, end exclusive) of the module docstring."""
    tree = ast.parse(source)
    node = tree.body[0] if tree.body else None
    if (
        node is None
        or not isinstance(node, ast.Expr)
        or not isinstance(node.value, ast.Constant)
        or not isinstance(node.value.value, str)
    ):
        raise StrategyValidationError("strategy file has no module docstring header")
    return node.lineno - 1, cast(int, node.end_lineno)


def _render_header_fields(source: str, **fields) -> str:
    """Return ``source`` with docstring header fields updated (or inserted before close)."""
    start, end = _docstring_span(source)
    lines = source.splitlines(keepends=True)
    doc_lines = lines[start:end]
    pending = {k: v for k, v in fields.items() if v is not None and k in HEADER_FIELDS}

    def render(name: str, value) -> str:
        if name == "symbols" and isinstance(value, (list, tuple)):
            value = " ".join(value)
        return f"{name}: {value}\n"

    for i, line in enumerate(doc_lines):
        match = _FIELD_RE.match(line.strip())
        if match and match.group(1) in pending:
            name = match.group(1)
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            doc_lines[i] = f"{indent}{render(name, pending.pop(name)).rstrip()}{newline}"

    if pending:
        inserted = [render(name, pending[name]) for name in HEADER_FIELDS if name in pending]
        if len(doc_lines) == 1:
            # Single-line docstring: split it open so the fields live inside it.
            match = re.match(r"^(\s*)(\"\"\"|''')(.*?)(\2)\s*$", doc_lines[0].rstrip("\n"))
            if match is None:
                raise StrategyValidationError("cannot rewrite docstring header (unusual quoting)")
            indent, quote, body = match.group(1), match.group(2), match.group(3)
            doc_lines = [f"{indent}{quote}{body}\n", "\n", *inserted, f"{indent}{quote}\n"]
        else:
            # Insert just before the closing quotes, blank-separated from a thesis
            # paragraph above (but packed together with existing header fields).
            closing = len(doc_lines) - 1
            above = doc_lines[closing - 1].strip()
            if above and not _FIELD_RE.match(above):
                inserted = ["\n", *inserted]
            doc_lines = doc_lines[:closing] + inserted + doc_lines[closing:]

    return "".join(lines[:start] + doc_lines + lines[end:])


def _render_param_defaults(source: str, name: str, params: dict) -> str:
    """Return ``source`` with the ``Params`` dataclass defaults replaced by ``params``."""
    lines = source.splitlines(keepends=True)
    class_idx = None
    class_indent = 0
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)class\s+Params\b", line)
        if match:
            class_idx, class_indent = i, len(match.group(1))
            break
    if class_idx is None:
        raise StrategyValidationError(f"{name}: no `class Params` block found for write-back")

    remaining = dict(params)
    for i in range(class_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if stripped and (len(line) - len(line.lstrip())) <= class_indent:
            break  # left the Params block
        match = re.match(r"^(\s*)(\w+)(\s*:\s*[^=#\n]+=\s*)([^#\n]*?)([ \t]*)(#.*)?(\n?)$", line)
        if match and match.group(2) in remaining:
            value = remaining.pop(match.group(2))
            comment = f"{match.group(5)}{match.group(6)}" if match.group(6) else ""
            lines[i] = (
                f"{match.group(1)}{match.group(2)}{match.group(3)}{value!r}{comment}{match.group(7)}"
            )
    if remaining:
        raise StrategyValidationError(
            f"{name}: params {sorted(remaining)} not found as Params fields for write-back"
        )
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Import / discovery
# ─────────────────────────────────────────────────────────────────────────────
def _load_module(path: Path):
    """Import a strategy file under a unique module name (fresh on every call)."""
    # Drop cached bytecode first: pyc staleness is judged by (mtime-second, size), and a
    # rewrite of the same-named candidate file within the same second with a same-length
    # value (e.g. 1.0 -> 0.9) collides — the gate would silently validate the OLD code.
    Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
    mod_name = f"noctis_authored_{path.stem}_{next(_module_counter)}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise StrategyValidationError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _find_strategy_class(module) -> type[TraderStrategy]:
    """The single TraderStrategy subclass defined (not just imported) in the module."""
    found = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, TraderStrategy)
        and obj is not TraderStrategy
        and obj.__module__ == module.__name__
    ]
    if len(found) != 1:
        raise StrategyValidationError(
            f"expected exactly one TraderStrategy subclass, found {len(found)}"
        )
    return found[0]


def _root(strategies_dir: LibrarySpec) -> Path:
    return LibraryPaths.coerce(strategies_dir).seeds


def _tmp_dir(strategies_dir: LibrarySpec) -> Path:
    return LibraryPaths.coerce(strategies_dir).tmp


def _champions_dir(strategies_dir: LibrarySpec) -> Path:
    return LibraryPaths.coerce(strategies_dir).champions


def _is_library_file(p: Path) -> bool:
    """A discoverable strategy file — not the template, not a hidden/scratch sibling."""
    return p.name != TEMPLATE_NAME and not p.name.startswith((".", "_"))


def _discard_scratch(path: Path) -> None:
    """Remove a scratch file AND the bytecode its validation import cached beside it —
    a lingering ``__pycache__`` entry for a dead scratch name is pure residue (and pyc
    staleness has bitten ``_load_module`` before)."""
    path.unlink(missing_ok=True)
    Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)


def _resolved_files(strategies_dir: LibrarySpec) -> dict[str, Path]:
    """Map ``name -> winning file`` across the three tiers; a later tier overrides an earlier one.

    The scan order (root → __tmp → champions) is the precedence order: seeds are the floor, a
    working copy in __tmp/ shadows a seed during research, and a promoted champion wins over both.
    ``__tmp/`` and ``champions/`` never carry the same name at once (write_strategy refuses to
    author over a champion; promotion moves the file out of __tmp/), so this stays unambiguous.
    """
    resolved: dict[str, Path] = {}
    tiers = (_root(strategies_dir), _tmp_dir(strategies_dir), _champions_dir(strategies_dir))
    for directory in tiers:
        if not directory.is_dir():
            continue
        for p in sorted(directory.glob("*.py")):
            if _is_library_file(p):
                resolved[p.stem] = p
    return resolved


def _strategy_files(strategies_dir: LibrarySpec) -> list[Path]:
    """The one winning file per strategy name, path-sorted (drives load + listing)."""
    return sorted(_resolved_files(strategies_dir).values())


def _locate(strategies_dir: LibrarySpec, name: str) -> Path | None:
    """The authoritative on-disk file for ``name`` (champions > __tmp > seed), or None."""
    for directory in (
        _champions_dir(strategies_dir),
        _tmp_dir(strategies_dir),
        _root(strategies_dir),
    ):
        p = directory / f"{name}.py"
        if p.is_file():
            return p
    return None


def strategy_path(strategies_dir: LibrarySpec, name: str) -> Path | None:
    """Public locator: where ``name`` currently lives, or None if it is not in the library."""
    return _locate(strategies_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_and_register(strategies_dir: LibrarySpec, families: FamilyRegistry) -> list[str]:
    """Import every library file and register its family. Returns the registered names.

    A broken file (only possible via hand-editing — the write gate forbids it) is skipped
    with a warning rather than crashing startup.
    """
    names: list[str] = []
    for path in _strategy_files(strategies_dir):
        try:
            module = _load_module(path)
            cls = _find_strategy_class(module)
            if cls.name != path.stem:
                raise StrategyValidationError(
                    f"class name attribute {cls.name!r} != file name {path.stem!r}"
                )
            families.register(cls)
            names.append(cls.name)
        except Exception as exc:  # noqa: BLE001 — startup must survive one bad file
            logger.warning("library: skipping %s (%s)", path.name, exc)
    return names


def list_strategies(strategies_dir: LibrarySpec) -> list[dict]:
    """Library index: header fields + current Params defaults + param space per file."""
    out: list[dict] = []
    for path in _strategy_files(strategies_dir):
        source = path.read_text(encoding="utf-8")
        info: dict = {"name": path.stem, **parse_header(source).to_dict()}
        try:
            module = _load_module(path)
            cls = _find_strategy_class(module)
            info["timeframe"] = cls.timeframe
            info["params"] = params_to_dict(cls.params_cls())
            info["param_space"] = [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "low": s.low,
                    "high": s.high,
                    "step": s.step,
                    **({"choices": list(s.choices)} if s.choices else {}),
                }
                for s in cls.param_space()
            ]
        except Exception as exc:  # noqa: BLE001 — a broken file is listed, not fatal
            info["error"] = str(exc)
        out.append(info)
    return out


def strategy_source(strategies_dir: LibrarySpec, name: str) -> str:
    path = _locate(strategies_dir, name)
    if path is None:
        raise FileNotFoundError(f"no strategy named {name!r} in {strategies_dir}")
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# The validator seam — how the write gate runs :func:`_validate_file`
# ─────────────────────────────────────────────────────────────────────────────
class Validator(Protocol):
    def __call__(self, path: Path, name: str, *, require_scenarios: bool = True) -> None: ...


def validate_in_subprocess(path: Path, name: str, *, require_scenarios: bool = True) -> None:
    """Run the import + smoke gate in an isolated interpreter; raise on any failure.

    The production default: a fresh subprocess is the only honest proof the file stands
    alone — no help from this process's import cache or already-registered siblings.

    Popen + group kill rather than ``subprocess.run(timeout=...)``: on timeout, ``run()``
    kills only the direct child and then blocks in a second unbounded ``communicate()`` —
    if the (agent-authored) file spawned anything that inherited the pipes, the research
    loop hangs there forever. The child gets its own process group (``start_new_session``)
    so a timeout kills the whole tree, and the drain after the kill is itself bounded.
    """
    argv = [sys.executable, "-m", "noctis.strategies.library", str(path), name]
    if require_scenarios:
        argv.append("--require-scenarios")
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=_VALIDATE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_validation_tree(proc)
        raise StrategyValidationError(
            f"validation timed out after {_VALIDATE_TIMEOUT_S}s — on_bar must be O(lookback) "
            f"per bar with no I/O, subprocesses, or unbounded loops"
        ) from None
    if proc.returncode != 0:
        detail = (err or out or "").strip()
        raise StrategyValidationError(detail.splitlines()[-1] if detail else "validation failed")


def _kill_validation_tree(proc: subprocess.Popen) -> None:
    """Kill a timed-out validation child *and everything it spawned*, without blocking.

    The whole process group goes down at once (the child is a session leader via
    ``start_new_session``), so no orphaned grandchild can keep the pipes open; the final
    drain is bounded, and if something un-killable somehow survives, the pipes are closed
    and abandoned rather than waited on.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):  # no killpg / already gone
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - an unkillable pipe holder
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def validate_in_process(path: Path, name: str, *, require_scenarios: bool = True) -> None:
    """Run the same gate checks in THIS interpreter — the test runner for the seam.

    Identical checks to :func:`validate_in_subprocess` (both funnel into
    :func:`_validate_file`) minus the per-call interpreter spawn, for tests that assert
    gate *outcomes* rather than subprocess isolation. Non-gate exceptions (a syntax
    error, a crash in ``on_bar``) are normalized to :class:`StrategyValidationError`
    carrying the same one-line message the subprocess entry point prints.
    """
    try:
        _validate_file(path, name, require_scenarios=require_scenarios)
    except StrategyValidationError:
        raise
    except Exception as exc:
        raise StrategyValidationError(f"{type(exc).__name__}: {exc}") from exc


# The seam itself: every write-path caller (write_strategy, _rewrite, plan_promotion)
# validates through this module attribute, so swapping the runner never touches callers.
validator: Validator = validate_in_subprocess


def _install(path: Path, families: FamilyRegistry) -> type[TraderStrategy]:
    """Import a validated file in-process and register its family."""
    cls = _find_strategy_class(_load_module(path))
    families.register(cls)
    return cls


def write_strategy(
    strategies_dir: LibrarySpec, name: str, source: str, families: FamilyRegistry
) -> dict:
    """Atomically write ``name``.py after the validation gate; register on success.

    The candidate source is written to a hidden sibling file, validated in a subprocess
    (clean import + smoke replay over the synthetic fixture + parity + header checks +
    replay of the file's declared known-outcome scenarios), and only then moved into
    place — so a broken file is never on disk under a library name, and a failed rewrite
    of an existing strategy leaves the old version untouched.
    """
    if not _NAME_RE.match(name):
        raise StrategyValidationError(
            f"invalid strategy name {name!r} (want lower_snake_case, e.g. rsi_meanrev)"
        )
    if (_champions_dir(strategies_dir) / f"{name}.py").is_file():
        raise StrategyValidationError(
            f"{name!r} is already a promoted champion; a champion file is immutable — author a "
            f"new name for a variant rather than overwriting the crown"
        )
    work = _tmp_dir(strategies_dir)
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / f".candidate-{name}.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        validator(tmp, name)
        final = work / f"{name}.py"
        tmp.replace(final)
    except BaseException:
        _discard_scratch(tmp)
        raise
    cls = _install(final, families)
    header = parse_header(source)
    return {"name": cls.name, "path": str(final), "header": header.to_dict()}


def _rewrite(
    strategies_dir: LibrarySpec, name: str, new_source: str, families: FamilyRegistry
) -> None:
    """Validate + atomically install a mechanical rewrite of an existing strategy file.

    Declared scenarios are still replayed, but their *presence* is not required — header
    stamps (e.g. ``status: rejected``) must keep working on legacy scenario-less files.

    A committed seed at the library root is never mutated in place: a mechanical rewrite of a
    seed (a status stamp, tuned defaults) is redirected into the gitignored ``__tmp/`` working
    area, so the public repo's seed files stay pristine.
    """
    target = _locate(strategies_dir, name)
    if target is None:
        raise FileNotFoundError(f"no strategy named {name!r} in {strategies_dir}")
    if target.parent == _root(strategies_dir):
        target = _tmp_dir(strategies_dir) / f"{name}.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".candidate-{name}.py"
    tmp.write_text(new_source, encoding="utf-8")
    try:
        validator(tmp, name, require_scenarios=False)
        tmp.replace(target)
    except BaseException:
        _discard_scratch(tmp)
        raise
    _install(target, families)


def set_header(
    strategies_dir: LibrarySpec, name: str, *, families: FamilyRegistry, **fields
) -> None:
    """Update docstring header fields (status/style/symbols/tuned) in place."""
    bad = set(fields) - set(HEADER_FIELDS)
    if bad:
        raise ValueError(f"unknown header fields: {sorted(bad)}")
    status = fields.get("status")
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; want one of {VALID_STATUSES}")
    source = strategy_source(strategies_dir, name)
    _rewrite(strategies_dir, name, _render_header_fields(source, **fields), families)


# ─────────────────────────────────────────────────────────────────────────────
# Promotion — the whole approval-time hand-off, validated before the crown
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PromotionPlan:
    """A fully rendered, gate-validated champion write-back, ready to install.

    Built by :func:`plan_promotion` *before* the champion registry crowns anything;
    :meth:`commit` afterwards only moves the pre-validated bytes, so nothing on the
    file side can fail between the crown and the champion file landing.
    """

    paths: LibraryPaths
    name: str
    source: str  # the finished champion file: tuned defaults + re-stamped header
    origin: Path  # the file it was rendered from (decides the move-vs-copy semantics)

    def commit(self, families: FamilyRegistry) -> Path:
        """Install the champion and register the final class. Returns the champion path.

        ``champions/<name>.py`` is written atomically; a ``__tmp/`` origin is retired
        (the crown leaves the scratch area) while a committed seed origin stays pristine.
        """
        dest = self.paths.champions / f"{self.name}.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".promote-{self.name}.py"
        tmp.write_text(self.source, encoding="utf-8")
        tmp.replace(dest)
        if self.origin != dest and self.origin.parent == self.paths.tmp:
            self.origin.unlink(missing_ok=True)
        _install(dest, families)
        return dest


def plan_promotion(
    strategies_dir: LibrarySpec,
    name: str,
    params: dict,
    *,
    symbols: list[str],
    tuned: str,
) -> PromotionPlan:
    """The promotion write-back as one artifact, gate-checked up front, library untouched.

    Renders the champion source in one pass — ``params`` become the ``Params`` dataclass
    defaults (so ``noctis backtest <name>`` with no arguments replays exactly what
    shipped) and the header is re-stamped ``status: champion`` with the fit ``symbols``
    and ``tuned`` date — then runs the write gate ONCE on the result. Raises
    :class:`StrategyValidationError` (typically: tuned params that violate the file's
    declared known-outcome scenarios) while the caller can still refuse the verdict;
    only a plan that survives here reaches :meth:`PromotionPlan.commit`, which the
    caller invokes after the registry crowns.
    """
    paths = LibraryPaths.coerce(strategies_dir)
    origin = _locate(paths, name)
    if origin is None:
        raise FileNotFoundError(f"no strategy named {name!r} in {strategies_dir}")
    source = origin.read_text(encoding="utf-8")
    rendered = _render_header_fields(
        _render_param_defaults(source, name, params),
        status="champion",
        symbols=symbols,
        tuned=tuned,
    )
    paths.tmp.mkdir(parents=True, exist_ok=True)
    probe = paths.tmp / f".promote-{name}.py"  # dot-prefixed: invisible to discovery
    probe.write_text(rendered, encoding="utf-8")
    try:
        validator(probe, name, require_scenarios=False)
    finally:
        _discard_scratch(probe)
    return PromotionPlan(paths=paths, name=name, source=rendered, origin=origin)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess validation entry point (``python -m noctis.strategies.library``)
# ─────────────────────────────────────────────────────────────────────────────
def _validate_file(path: Path, expected_name: str, require_scenarios: bool = True) -> None:
    module = _load_module(path)
    cls = _find_strategy_class(module)
    if cls.name != expected_name:
        raise StrategyValidationError(
            f"class sets name={cls.name!r} but the strategy/file name is {expected_name!r}"
        )
    if not (module.__doc__ or "").strip():
        raise StrategyValidationError("missing module docstring (thesis + header)")
    from noctis.data.aggregate import TIMEFRAMES

    if cls.timeframe not in TIMEFRAMES:
        raise StrategyValidationError(
            f"timeframe {cls.timeframe!r} unsupported; want one of {sorted(TIMEFRAMES)}"
        )
    header = parse_header(Path(path).read_text(encoding="utf-8"))
    if header.status not in VALID_STATUSES:
        raise StrategyValidationError(
            f"header status {header.status!r} invalid; want one of {VALID_STATUSES}"
        )
    space = cls.param_space()
    if not isinstance(space, list):
        raise StrategyValidationError("param_space() must return a list of ParamSpec")
    params = cls.params_cls()
    frame = fixture_frame()
    targets = replay_targets(cls(params), frame)
    if len(targets) != len(frame):
        raise StrategyValidationError("on_bar replay produced a short target series")
    vectorised = [int(x) for x in cls.signals(frame, params)]
    if vectorised != targets:
        raise StrategyValidationError(
            "signals() disagrees with the on_bar replay on the fixture (parity violation); "
            "drop the signals() override or fix it"
        )
    try:
        scenarios_mod.check_scenario_contract(cls, require=require_scenarios)
    except scenarios_mod.ScenarioError as exc:
        raise StrategyValidationError(str(exc)) from exc


def _main(argv: list[str]) -> int:
    require_scenarios = "--require-scenarios" in argv
    argv = [a for a in argv if a != "--require-scenarios"]
    if len(argv) != 2:
        print(
            "usage: python -m noctis.strategies.library <file.py> <name> [--require-scenarios]",
            file=sys.stderr,
        )
        return 2
    try:
        _validate_file(Path(argv[0]), argv[1], require_scenarios=require_scenarios)
    except Exception as exc:  # noqa: BLE001 — report the reason on one line, exit nonzero
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
