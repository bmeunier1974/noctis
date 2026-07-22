"""Shared state/digest builders — the facts both research loops render, in one place.

The conversation loop (:mod:`noctis.research.prompt`) and the episodic research driver that
lands later (epic #62) both have to show the model the same four state facts before it acts:
the MARKET REALITY economics digest, the strategy library index (rejected entries collapsed
to stubs), the champion board rows, and the advisory memory block (findings + known dead
ends). Rendering them in one shared module means the two loops present the *same facts by
construction* — the frozen conversation baseline cannot silently drift from the episodic path.

Each builder takes explicit collaborators (toolbox / registry / memory / library paths) and
returns plain, JSON-serializable data (or the serialized digest string); the callers own the
surrounding prose framing. Serialization is deterministic — same inputs ⇒ byte-identical
output — so a session's system prefix stays cache-stable and both loops agree byte-for-byte.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from noctis.memory.consolidate import consolidate_findings, consolidate_rejected
from noctis.strategies import library

logger = logging.getLogger("noctis.research.digests")

# Hard byte bound on the findings block a prefix embeds (context plan P3): the consolidated
# tail drops its oldest lines past this. Sized so today's economy tail always fits untouched.
_MEMORY_FINDINGS_CHAR_BUDGET = 8_000


def market_digest(toolbox: Any) -> str:
    """The MARKET REALITY economics digest, serialized with sorted keys for a byte-stable
    prefix (insertion order can't perturb the bytes, so a future cross-session cache hits).

    A lake hiccup while building the per-symbol digest must not kill the session, so a failure
    degrades to a cost-facts-only note rather than propagating.
    """
    try:
        digest = toolbox.market_context()
    except Exception as exc:  # noqa: BLE001 — a lake hiccup must not kill the session
        logger.warning("market context digest failed (%s); degrading to cost facts only", exc)
        digest = {"note": "per-symbol digest unavailable this session"}
    return json.dumps(digest, sort_keys=True)


def library_index(strategies_dir: Any) -> list[dict]:
    """The strategy library index with every ``rejected`` entry collapsed to a ``{name,
    status}`` stub.

    A rejected strategy's class-level lesson already reaches the model via memory's
    ``rejected_ideas`` and the ``exhausted_classes`` digest, so re-shipping each corpse's
    thesis/params/param_space is pure duplication that grows with every rejection. The files
    stay on disk untouched and the ``list_strategies`` tool keeps returning everything in full.
    """
    return [
        entry
        if entry.get("status") != "rejected"
        else {"name": entry["name"], "status": entry["status"]}
        for entry in library.list_strategies(strategies_dir)
    ]


def champion_digest(registry: Any) -> list[dict]:
    """The champion board rows: family, params, out-of-sample test metric, the neutral
    cross-profile Sharpe yardstick, mandate provenance, and fit symbols.

    ``sharpe`` is read on a common basis for every champion regardless of the metric it was
    elected on (the ``auto`` rule), so cross-profile boards stay comparable.
    """
    return [
        {
            "family": e.family,
            "params": e.params,
            "test_metric": round(e.test_metric, 4),
            "sharpe": round(e.scorecard.avg_test_named("sharpe"), 4),
            "mandate_source": e.mandate_source,
            "fit_symbols": e.fit_symbols,
        }
        for e in registry.list()
    ]


def memory_block(memory: Any, *, prefix_trim: bool = False) -> tuple[list, list]:
    """The advisory memory tail: ``(findings, rejected_dead_ends)``.

    ``prefix_trim`` (the ``economy`` cost lever) caps the tail to the last 5 lesson classes
    instead of 20, shrinking the cache write + per-round reads — cost, not capability: the
    dead-end guard and the gates are unaffected. The tail counts consolidated *lesson classes*
    (P3 stage 1), not raw events, so the same size spans deeper history; once a distilled block
    exists (P3 stage 2), it is embedded plus the 3 newest raw entries.
    """
    limit = 5 if prefix_trim else 20
    raw_findings = memory.findings() if hasattr(memory, "findings") else []
    distilled = memory.distilled() if hasattr(memory, "distilled") else []
    if distilled:
        findings = distilled + raw_findings[-3:]
    else:
        findings = consolidate_findings(
            raw_findings, limit=limit, char_budget=_MEMORY_FINDINGS_CHAR_BUDGET
        )
    rejected = consolidate_rejected(memory.rejected_ideas(), limit=limit)
    return findings, rejected
