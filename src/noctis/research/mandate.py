"""The operator mandate — a human-ownable input surface for the research agent.

One selector resolves to one :class:`Mandate`: the free-form prose the human wants the
agent to pursue this session, its supporting reference files, and the single config knob a
personality may consciously bind (``promotion.metric``). This module is pure and
unit-testable — no agent, no network — so the loader, the overlay, and the seam are all
exercised without a client.

Design constraints (see ``docs/operator-mandate.md``):

* **Errors are loud at startup, quiet mid-session.** An unresolvable selector (typo'd
  profile, missing file/dir, unreadable file) raises :class:`MandateError` so an entrypoint
  can exit non-zero before a multi-day run starts un-steered. A reference problem inside an
  otherwise-valid mandate degrades with a warning instead — the same fatal-config vs.
  degradable-runtime split the rest of the app uses.
* **References are a lean steering prior, not a knowledge base.** Per-file and total byte
  caps (module constants, deliberately small) keep the operator's supporting material from
  crowding out the agent's own reasoning; a reference that wants to be bigger is a signal it
  should be a link the agent follows with web_search, not an embed.
* **The overlay may set exactly one knob.** ``_OVERRIDE_ALLOWLIST`` is ``promotion.metric``
  and nothing else; every gate, safety, budget, and structural knob is refused (with a
  warning, never a crash). Widening it is a deliberate, owner-gated edit to that constant.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from noctis.backtest.scorecard import Metric

logger = logging.getLogger("noctis.research.mandate")

# Context-hygiene guardrails — module constants beside the agent's _MAX_TOKENS /
# _RESULT_CHAR_CAP, NOT operator config (exposing them just invites prompt bloat, §3.3/§7).
_REFERENCE_FILE_CAP_BYTES = 2048
_REFERENCE_TOTAL_CAP_BYTES = 6144
_MANDATE_BODY_WARN_BYTES = 6144

# The single overridable knob (§3.4). Widening this past ``promotion.metric`` is a
# deliberate, owner-gated change to this constant — never something a mandate author reaches.
_OVERRIDE_ALLOWLIST = ("promotion.metric",)

# A leading ``---`` / ``---`` YAML front-matter fence, then the prose body.
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)
# Inline ``[[references/name]]`` wikilink includes (the operator's memory convention).
_INCLUDE_RE = re.compile(r"\[\[([^\]\n]+)\]\]")
# HTML comments, stripped only for the empty-body check.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class MandateError(Exception):
    """A selector that doesn't resolve (typo'd profile, missing file). Fatal at startup."""


@dataclass(frozen=True)
class Reference:
    """One supporting file pulled into the prompt, capped and confined per §3.3."""

    path: str  # mandate-relative, e.g. "references/watchlist.md"
    text: str  # file contents, already capped


@dataclass(frozen=True)
class Mandate:
    """The resolved operator input for one session."""

    text: str  # the resolved mandate body (a file, or an inline --directive line)
    source: str  # provenance: "mandate/MANDATE.md" | "profile:aggressive" | "cli" | "auto"
    summary: str  # front-matter `summary:`, else first prose line (kickoff echo + catalog)
    references: list[Reference]  # the files the mandate asked to include
    config_overrides: dict  # flattened {"promotion.metric": ...} from the front-matter config:
    # Front-matter ``symbols:`` — tickers the mandate declares it wants researched. A search
    # prior only (rule 5): they join the session's research *focus set* (prompt digest +
    # holdout candidate pool), never a gate or the trading roster.
    symbols: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_mandate(settings, *, cli_directive=None, cli_mandate=None) -> Mandate | None:
    """Resolve the active mandate by first-match-wins precedence (§3.2).

    ``cli_directive`` (inline one-liner) → ``cli_mandate`` (name) → ``settings.research.mandate``
    (config selector) → ``None`` (unconstrained). Raises :class:`MandateError` on a selector
    that doesn't resolve; reference problems inside a valid mandate only warn and drop.
    """
    mandate_dir = Path(getattr(settings, "mandate_dir", "mandate/"))

    # 1. Inline --directive: a one-off mandate with no file, for quick one-liners.
    if cli_directive is not None:
        text = cli_directive.strip()
        return Mandate(
            text=text,
            source="cli",
            summary=_extract_summary(None, text) or text,
            references=[],
            config_overrides={},
        )

    # 2. --mandate <name>, else 3. research.mandate (config). 4. None → unconstrained.
    selector = cli_mandate
    if selector is None:
        selector = getattr(settings.research, "mandate", None)
    if not selector:
        return None
    return _resolve_selector(mandate_dir, str(selector))


def _resolve_selector(mandate_dir: Path, selector: str) -> Mandate | None:
    """Map a selector name to a resolved :class:`Mandate` (or ``None`` for empty MANDATE)."""
    if selector == "auto":
        return _auto_mandate(mandate_dir)

    if selector == "MANDATE":
        path = mandate_dir / "MANDATE.md"
        if not path.is_file():
            raise MandateError(
                f"research.mandate=MANDATE but {path} does not exist — create it or set a profile."
            )
        return _build_mandate_from_file(path, "mandate/MANDATE.md", mandate_dir, is_mandate_md=True)

    if any(sep in selector for sep in ("/", "\\")):
        raise MandateError(
            f"mandate selector {selector!r} contains a path separator — profiles are flat names."
        )

    stem = selector[:-3] if selector.endswith(".md") else selector
    filename = f"{stem}.md"
    profile_path = mandate_dir / "profiles" / filename
    if profile_path.is_file():
        return _build_mandate_from_file(profile_path, f"profile:{stem}", mandate_dir)
    top_path = mandate_dir / filename
    if top_path.is_file():
        return _build_mandate_from_file(top_path, f"mandate/{filename}", mandate_dir)
    raise MandateError(f"mandate {selector!r} not found: looked for {profile_path} and {top_path}.")


def _build_mandate_from_file(
    path: Path, source: str, mandate_dir: Path, *, is_mandate_md: bool = False
) -> Mandate | None:
    """Read a mandate file into a :class:`Mandate` (front-matter + references + overlay)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # a selector that points at an unreadable file is fatal config.
        raise MandateError(f"mandate file {path} is unreadable: {exc}") from exc

    front_matter, body = _split_front_matter(raw, path)

    # The empty-MANDATE rule: the shipped template is all comments, so a bare repo injects
    # no block. Only MANDATE.md degrades to None — profiles ship with real content.
    if is_mandate_md and _is_effectively_empty(body):
        logger.warning(
            "MANDATE selected but %s is empty; running unconstrained this session.", path
        )
        return None

    if len(body.encode("utf-8")) > _MANDATE_BODY_WARN_BYTES:
        logger.warning(
            "mandate body %s is over %d bytes — move detail into links so the agent keeps "
            "context for its own reasoning (§3.3).",
            path,
            _MANDATE_BODY_WARN_BYTES,
        )

    # Author/operator HTML comments (e.g. the MANDATE.md how-to header) are not agent-facing
    # steering — strip them so they never reach the prompt, mirroring the empty-body check.
    clean_body = _HTML_COMMENT_RE.sub("", body)
    return Mandate(
        text=clean_body.strip(),
        source=source,
        summary=_extract_summary(front_matter, clean_body),
        references=_load_references(front_matter, clean_body, mandate_dir),
        config_overrides=_extract_overrides(front_matter),
        symbols=_extract_symbols(front_matter),
    )


def _auto_mandate(mandate_dir: Path) -> Mandate:
    """``auto`` → a selection *instruction* plus the profiles catalog (§3.2/§3.5).

    ``config_overrides`` is necessarily empty: the agent picks the profile inside the session,
    after the overlay already ran, so an auto-selected profile's ``config:`` block is inert.
    """
    catalog = profiles_catalog(mandate_dir)
    if catalog:
        listing = "\n".join(f"- {c['name']}: {c['summary']}" for c in catalog)
    else:
        listing = "(no profiles available)"
    text = (
        "No fixed operator mandate is set for this session. Choose ONE trader profile from "
        "the catalog below to govern this session, and declare your choice — with a one-line "
        "reason — at the start of FORMULATE.\n\n"
        "Available profiles:\n"
        f"{listing}\n\n"
        "SELECTION RULE — pick on Sharpe. Choose the profile whose recent champions are "
        "strongest on Sharpe, the neutral risk-adjusted yardstick shown per champion in the "
        "CURRENT STATE champion board (its `sharpe` field), REGARDLESS of the metric each "
        "profile tunes on. Use each champion's `mandate_source` field to attribute it to the "
        "profile that produced it. Judging on the common Sharpe basis is deliberate: a profile "
        "that tunes total_return must not win selection just because its own favorable metric "
        "flatters it. While no champion is attributable to a profile yet, choose by judgment "
        "against memory and the market digest, and say why."
    )
    return Mandate(
        text=text,
        source="auto",
        summary="auto: the agent selects a profile per session",
        references=[],
        config_overrides={},
    )


# ─────────────────────────────────────────────────────────────────────────────
# The config overlay
# ─────────────────────────────────────────────────────────────────────────────
def apply_overrides(settings, mandate) -> list[str]:
    """Apply a mandate's ``config:`` overlay to ``settings`` in place; return "k=v" echoes.

    Only ``promotion.metric`` may be set; every other key is refused with a warning and
    skipped. The metric value is pre-validated through :meth:`Metric.parse` **here** —
    these pydantic models do not enable ``validate_assignment``, so the field validator does
    NOT run on attribute assignment (§3.4). Nothing here crashes: bad input warns and skips.
    """
    if mandate is None:
        return []
    echoes: list[str] = []
    for dotted, value in mandate.config_overrides.items():
        if dotted not in _OVERRIDE_ALLOWLIST:
            logger.warning(
                "mandate %s tried to override %s=%r — refused: only %s may be set by a mandate; "
                "skipping.",
                mandate.source,
                dotted,
                value,
                ", ".join(_OVERRIDE_ALLOWLIST),
            )
            continue
        if dotted == "promotion.metric":
            try:
                value = Metric.parse(value).value
            except ValueError as exc:
                logger.warning(
                    "mandate %s set promotion.metric — %s; skipping.", mandate.source, exc
                )
                continue
            settings.promotion.metric = value
            echoes.append(f"promotion.metric={value}")
    return echoes


# ─────────────────────────────────────────────────────────────────────────────
# The profiles catalog (the `auto` menu)
# ─────────────────────────────────────────────────────────────────────────────
def profiles_catalog(mandate_dir) -> list[dict]:
    """``[{"name", "summary"}, ...]`` for ``mandate/profiles/*.md`` (empty if the dir is absent)."""
    base = Path(mandate_dir) / "profiles"
    if not base.is_dir():
        return []
    catalog: list[dict] = []
    for path in sorted(base.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:  # noqa: PERF203 — one bad profile must not kill the catalog.
            logger.warning("profile %s unreadable (%s); skipping catalog entry.", path, exc)
            continue
        front_matter, body = _split_front_matter(raw, path)
        catalog.append({"name": path.stem, "summary": _extract_summary(front_matter, body)})
    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _split_front_matter(raw: str, path: Path) -> tuple[dict, str]:
    """Split a leading ``---`` fence from the prose. Malformed fence → warn, all prose."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning(
            "front-matter in %s failed to parse (%s); treating the whole file as prose.", path, exc
        )
        return {}, raw
    if data is None:
        return {}, match.group(2)
    if not isinstance(data, dict):
        logger.warning(
            "front-matter in %s is not a mapping; treating the whole file as prose.", path
        )
        return {}, raw
    return data, match.group(2)


def _is_effectively_empty(body: str) -> bool:
    """True if the body is only HTML comments and whitespace (the shipped template)."""
    return _HTML_COMMENT_RE.sub("", body).strip() == ""


def _extract_summary(front_matter, body: str) -> str:
    """Front-matter ``summary:`` if present and non-empty, else the first non-empty prose line."""
    if front_matter:
        summary = front_matter.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _extract_overrides(front_matter) -> dict:
    """Flatten the front-matter ``config:`` block to dotted paths (allowlist enforced later)."""
    if not front_matter:
        return {}
    config = front_matter.get("config")
    if not isinstance(config, dict):
        return {}
    return _flatten(config)


def _extract_symbols(front_matter) -> list[str]:
    """Front-matter ``symbols:`` — a list of ticker strings, normalized upper-case, deduped.

    A malformed block (not a list, non-string items) warns and drops, mirroring the
    reference-loading policy: a valid mandate never becomes fatal over a steering hint.
    """
    if not front_matter:
        return []
    raw = front_matter.get("symbols")
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning("mandate symbols: is not a list; dropping.")
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            logger.warning("mandate symbols: entry %r is not a ticker string; dropping.", item)
            continue
        sym = item.strip().upper()
        if sym not in out:
            out.append(sym)
    return out


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _with_md_suffix(name: str) -> str:
    return name if Path(name).suffix else f"{name}.md"


def _load_references(front_matter, body: str, mandate_dir: Path) -> list[Reference]:
    """Load front-matter + inline ``[[…]]`` references, capped and confined to ``mandate_dir``."""
    names: list[str] = []
    if front_matter:
        fm_refs = front_matter.get("references")
        if isinstance(fm_refs, list):
            names.extend(item.strip() for item in fm_refs if isinstance(item, str) and item.strip())
    names.extend(m.group(1).strip() for m in _INCLUDE_RE.finditer(body))

    # De-dupe, front-matter order first then first-mention order.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = _with_md_suffix(name)
        if key not in seen:
            seen.add(key)
            ordered.append(name)

    base = mandate_dir.resolve()
    refs: list[Reference] = []
    total = 0
    for name in ordered:
        rel = _with_md_suffix(name)
        if Path(rel).is_absolute():
            logger.warning(
                "reference %r is an absolute path; dropping (references stay in-tree).", name
            )
            continue
        candidate = (mandate_dir / rel).resolve()
        if not candidate.is_relative_to(base):
            logger.warning("reference %r escapes %s; dropping.", name, mandate_dir)
            continue
        if not candidate.is_file():
            logger.warning("reference %r not found under %s; dropping.", name, mandate_dir)
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("reference %r unreadable (%s); dropping.", name, exc)
            continue
        size = len(text.encode("utf-8"))
        if size > _REFERENCE_FILE_CAP_BYTES:
            logger.warning(
                "reference %r is %d bytes (> %d cap); dropping — link it instead of embedding it.",
                name,
                size,
                _REFERENCE_FILE_CAP_BYTES,
            )
            continue
        if total + size > _REFERENCE_TOTAL_CAP_BYTES:
            logger.warning(
                "reference %r would exceed the %d-byte total budget; dropping.",
                name,
                _REFERENCE_TOTAL_CAP_BYTES,
            )
            continue
        total += size
        refs.append(Reference(path=rel, text=text))
    return refs
