"""The run-record contract page is held to the code it documents (story #144, epic #126).

A documentation story's quality bar is *accuracy*, and a confident, wrong sentence in a contract
page is worse than no page: a website author reads ``docs/run-record.md`` instead of the source, so
a field the schema grew and the page never mentioned would be invisible to exactly the reader the
page exists for.

Three cheap checks keep it honest, and each fails on a *code* change rather than needing a human to
remember:

* the committed worked example (``examples/run_record.json``) is a real record and still validates
  against :func:`noctis.reporting.schema.validate`;
* the page **names every key the schema names** — every ``*_KEYS`` tuple in ``schema.py``, plus the
  enumerations beside them — so a section added tomorrow lands here as a red test;
* the numbers the page pins (the schema version, the engine version, the caps, the size budget, the
  freezing-tier counts, the stale-lock horizon) are the constants, not a snapshot of them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from noctis.config import rehydrate
from noctis.observability import engine_id
from noctis.reporting import run_store, schema

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "run-record.md"
EXAMPLE = ROOT / "examples" / "run_record.json"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _schema_key_names() -> set[str]:
    """Every field name the schema itself names — the contract, read off the module.

    Deliberately discovered rather than listed: the point is that a key added to ``schema.py``
    without a line in the page is a failing test, and a hand-kept list here would need the same
    edit the page does.
    """
    names: set[str] = set()
    for attribute, value in vars(schema).items():
        if not attribute.endswith("_KEYS") and attribute != "REQUIRED_SECTIONS":
            continue
        if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            names |= set(value)
    return names


def _enumerations() -> Iterable[str]:
    for enumeration in (
        schema.RUN_STATUSES,
        schema.SEGMENT_STATUSES,
        schema.SEGMENT_COMMANDS,
        schema.STRATEGY_OUTCOMES,
        schema.STRATEGY_TIERS,
        schema.UNRECORDABLE_SETTINGS,
    ):
        yield from enumeration


# ── the worked example is a real record ────────────────────────────────────────────────────


def test_worked_example_validates(example: dict) -> None:
    """The example a website author copies from is schema-valid, not schema-plausible."""
    assert schema.validate(example) == []


def test_worked_example_carries_every_section(example: dict) -> None:
    """Every required section is present, so the example shows the whole shape and not a subset."""
    assert set(example) == {"schema_version", "kind", *schema.REQUIRED_SECTIONS}


def test_worked_example_is_a_populated_run(example: dict) -> None:
    """It exercises the shapes a page has to render: a funnel, a traded account, a priced bill."""
    outcomes = {entry["outcome"] for entry in example["strategies"]}
    assert outcomes == set(schema.STRATEGY_OUTCOMES)
    assert {segment["command"] for segment in example["segments"]} == set(schema.SEGMENT_COMMANDS)
    assert example["run"]["traded"] is True
    assert example["performance"] is not None
    assert example["spend"]["llm_usd_estimate"] is not None


def test_worked_example_carries_no_live_money_gate(example: dict) -> None:
    """The published example is a redaction test too: neither gate is a *key* anywhere in it.

    Naming the pair is fine and is the point of ``inputs.settings.refused_keys`` — what the record
    may never do is carry a **value** for either, which is what a key would be (AGENTS.md rule 1).
    """
    assert not sorted(_keys(example) & set(schema.UNRECORDABLE_SETTINGS))


def _keys(node: object) -> set[str]:
    """Every mapping key anywhere in one document."""
    if isinstance(node, dict):
        return set(node) | {name for value in node.values() for name in _keys(value)}
    if isinstance(node, list):
        return {name for item in node for name in _keys(item)}
    return set()


# ── the page documents what the schema declares ────────────────────────────────────────────


def test_page_names_every_schema_key(page: str) -> None:
    """A key the schema names and the page does not is a field a consumer cannot discover."""
    missing = sorted(name for name in _schema_key_names() if name not in page)
    assert not missing, f"docs/run-record.md does not mention: {', '.join(missing)}"


def test_page_names_every_enumerated_value(page: str) -> None:
    """The statuses, the segment commands, the outcomes and the tiers are all spelled out."""
    missing = sorted({value for value in _enumerations() if value not in page})
    assert not missing, f"docs/run-record.md does not mention: {', '.join(missing)}"


def test_page_names_every_engine_component(page: str) -> None:
    """The component table is complete, and both arbiter components are marked as such."""
    for component in engine_id.COMPONENT_PATHS:
        assert f"`{component}`" in page, f"the component table omits {component}"
    for component in engine_id.ARBITER_COMPONENTS:
        assert re.search(rf"`{component}`[^\n]*\*\*arbiter\*\*", page), component


# ── the numbers the page pins are the constants ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("`SCHEMA_VERSION` is **{}**", schema.SCHEMA_VERSION),
        ("`TRADE_CAP` | {:,}", schema.TRADE_CAP),
        ("`EVENT_CAP` | {:,}", schema.EVENT_CAP),
        ("`STRATEGY_CAP` | {}", schema.STRATEGY_CAP),
        ("`RECORD_SIZE_BUDGET_BYTES` | {} KiB", schema.RECORD_SIZE_BUDGET_BYTES // 1024),
        ("**{}** today", engine_id.ENGINE_VERSION),
        ("(**{} days**", int(run_store.STALE_HEARTBEAT_S // 86400)),
        (
            "**{} frozen, {} live, {} refused**",
            (len(rehydrate.FROZEN), len(rehydrate.LIVE), len(rehydrate.REFUSED)),
        ),
    ],
)
def test_page_pins_the_constants(page: str, claim: str, value: object) -> None:
    """Every number the page states can be pinned to a constant, and is."""
    expected = claim.format(*value) if isinstance(value, tuple) else claim.format(value)
    # ``{:,}`` renders 5000 as ``5,000``; the page uses a thin space, like the source comments do.
    assert expected.replace(",", " ") in page or expected in page, f"stale claim: {expected!r}"
