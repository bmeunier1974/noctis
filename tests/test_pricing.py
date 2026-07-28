"""The versioned LLM price table — ``noctis.research.pricing`` (story #140, epic #126).

Every number this module produces is an **estimate**, and the one rule that makes it a usable
estimate rather than a liability is stated in the tests below: a model the table does not carry
contributes ``None``, never a zero and never an inferred number. An operator comparing two runs on
cost has to be able to tell "this cost $4.10" from "nobody knows what this cost".

Pure arithmetic over data handed in — no I/O, no clock, no config — so it is snapshot-testable and
the record builder can lean on it without importing the world.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from noctis.research.pricing import (
    LOCAL_PREFIXES,
    PRICING_TABLE_VERSION,
    ModelPrice,
    PriceTable,
    default_table,
    table_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/noctis/research/pricing.py"
# The two files that decide which model a stock install ends up running.
ONBOARDING = ROOT / "src/noctis/onboarding.py"
CLI = ROOT / "src/noctis/cli.py"

# One completion's worth of the four neutral usage fields every provider seam reports.
USAGE = {
    "input_tokens": 1_000_000,
    "output_tokens": 1_000_000,
    "cache_creation_input_tokens": 1_000_000,
    "cache_read_input_tokens": 1_000_000,
}


def _table() -> PriceTable:
    return PriceTable(
        version="test-1",
        prices={
            "anthropic/claude-opus-4": ModelPrice(15.0, 75.0, 18.75, 1.5),
            "anthropic/claude-opus-4-8-mini": ModelPrice(1.0, 2.0, 3.0, 4.0),
            "ollama/": ModelPrice(0.0, 0.0, 0.0, 0.0),
        },
    )


# ── pricing one model ──────────────────────────────────────────────────────────────────────


def test_a_known_model_is_priced_per_million_tokens_field_by_field():
    """Each of the four fields is billed at its own rate — they are not one blended number."""
    assert _table().estimate_usd("anthropic/claude-opus-4-8", USAGE) == 15.0 + 75.0 + 18.75 + 1.5


def test_the_table_is_keyed_by_model_prefix_so_a_point_release_prices_itself():
    table = _table()

    assert table.price_for("anthropic/claude-opus-4-8") is not None
    assert table.price_for("anthropic/claude-opus-4-9-preview") is not None


def test_the_longest_matching_prefix_wins():
    """A more specific entry must not be shadowed by the family it belongs to."""
    table = _table()

    assert table.price_for("anthropic/claude-opus-4-8-mini-2026") == ModelPrice(1.0, 2.0, 3.0, 4.0)


def test_an_unknown_model_contributes_null_never_an_inferred_or_zero_number():
    """The whole point of the module: a confidently false number is worse than no number."""
    table = _table()

    assert table.price_for("openai/gpt-9-turbo") is None
    assert table.estimate_usd("openai/gpt-9-turbo", USAGE) is None


def test_a_free_local_backend_is_priced_at_zero_by_a_table_entry_not_by_inference():
    """A ``$0``/token local backend genuinely costs nothing — but that is a stated price."""
    assert _table().estimate_usd("ollama/qwen3-coder-30b", USAGE) == 0.0


def test_usage_that_never_recorded_the_split_is_unpriceable():
    """Tokens without their split cannot be priced: the four fields bill at four rates."""
    table = _table()

    assert table.estimate_usd("anthropic/claude-opus-4-8", None) is None
    assert table.estimate_usd("anthropic/claude-opus-4-8", {"input_tokens": 10}) is None


def test_a_zero_usage_session_on_a_known_model_costs_a_real_zero():
    zero = dict.fromkeys(USAGE, 0)

    assert _table().estimate_usd("anthropic/claude-opus-4-8", zero) == 0.0


# ── totalling across models ────────────────────────────────────────────────────────────────


def test_a_total_sums_every_priced_item():
    table = _table()
    items = [("anthropic/claude-opus-4-8", USAGE), ("ollama/qwen3", USAGE)]

    assert table.total_usd(items) == 15.0 + 75.0 + 18.75 + 1.5


def test_a_total_containing_one_unpriceable_item_is_null_not_a_partial_sum():
    """A partial sum presented as a total is exactly the confidently false number this forbids."""
    table = _table()
    items = [("anthropic/claude-opus-4-8", USAGE), ("openai/gpt-9-turbo", USAGE)]

    assert table.total_usd(items) is None


def test_a_total_over_nothing_is_a_real_zero():
    assert _table().total_usd([]) == 0.0


# ── versioning ─────────────────────────────────────────────────────────────────────────────


def test_the_builtin_table_declares_the_version_that_produced_its_numbers():
    assert default_table().version == PRICING_TABLE_VERSION
    assert default_table().prices  # the shipped table is not empty


def test_the_shipped_label_is_a_month_and_an_optional_revision_of_that_month():
    """``<month>[.<revision>]``: a re-survey of the prices is a new month, a corrected **coverage**
    is a revision of the same one (story #146). Either way the label changes, because both change
    the number a record publishes — and nothing may parse the label beyond the ``+custom`` split,
    which is why a dot inside it stays safe."""
    assert re.fullmatch(r"\d{4}-\d{2}(\.\d+)?", PRICING_TABLE_VERSION)
    assert "+" not in PRICING_TABLE_VERSION  # ``+custom.`` is the one separator anything splits on


def test_the_shipped_table_prices_the_default_research_and_agent_models():
    """The two models a stock install actually runs must not report a null bill."""
    table = default_table()

    assert table.price_for("openai/gpt-5.4") is not None
    assert table.price_for("claude-opus-4-8") is not None


# ── the shipped local driver's coverage (story #146) ────────────────────────────────────────

LOCAL_DRIVER = "ollama_chat/noctis-qwen3:14b"  # exactly what `noctis setup` writes


def test_the_shipped_local_driver_prefix_is_a_stated_zero_not_an_unknown_model():
    """``ollama_chat/`` is the same free Ollama server ``ollama/`` is — the suffix only selects
    the chat-completions driver — so the two must price identically. Leaving it uncarried spent the
    ``null`` signal ("nobody knows what this cost") on the one run whose cost is known exactly, and
    because a total is all-or-nothing it nulled the whole run's estimate."""
    table = default_table()

    assert table.price_for(LOCAL_DRIVER) == ModelPrice(0.0, 0.0, 0.0, 0.0)
    assert table.price_for(LOCAL_DRIVER) == table.price_for("ollama/qwen3:14b")
    assert table.estimate_usd(LOCAL_DRIVER, USAGE) == 0.0


def test_every_local_prefix_is_priced_from_the_one_stated_allowlist():
    """The list of ``$0`` backends is stated once, so adding a driver is one line rather than a
    line here and a forgotten one there — which is how ``ollama_chat/`` came to be missing."""
    table = default_table()

    assert LOCAL_PREFIXES  # deny-by-default is not the same as an empty table
    for prefix in LOCAL_PREFIXES:
        assert table.estimate_usd(f"{prefix}any-tag:14b", USAGE) == 0.0, prefix


def _wizard_models() -> list[str]:
    """Every model string the shipped onboarding path itself emits or offers, read out of the source
    that emits it — the same technique ``tests/test_engine_id.py`` uses on the fingerprint's files.

    Derived, not transcribed: a wizard that starts writing a different prefix must fail the test
    below rather than quietly configure an operator into an unpriceable run.
    """
    wizard = ONBOARDING.read_text(encoding="utf-8")
    hosted = re.search(r"_HOSTED_DEFAULT_MODEL = (\{.*?\})", wizard)
    assert hosted, "the wizard's hosted defaults moved — re-point this reader, not its point"
    # The local branch interpolates the tag the operator picked off their own Ollama server.
    local = re.findall(r'f"([a-z0-9_]+/)\{tag\}"', wizard)
    offered = re.findall(r"([a-z0-9_]+/[a-z0-9][\w.:-]*) \(skips the LLM menu\)", CLI.read_text())
    assert local and offered, "the local-driver literals moved — re-point this reader"
    models = [
        *ast.literal_eval(hosted.group(1)).values(),
        *(f"{prefix}noctis-qwen3:14b" for prefix in local),
        *offered,
    ]
    return models


def test_every_model_the_setup_wizard_configures_is_priced_by_the_shipped_table():
    """The shipped path cannot silently become unpriced: an operator who took every default must
    never be told the price table cannot bill the model it chose for them."""
    table = default_table()

    for model in _wizard_models():
        assert table.estimate_usd(model, USAGE) is not None, (
            f"`noctis setup` configures {model!r}, which the shipped price table cannot bill, so "
            "the operator's whole run publishes a null cost (a total is all-or-nothing). Add its "
            "prefix to LOCAL_PREFIXES if the backend is a $0 local one, or its four rates to "
            f"BUILTIN_PRICES if it is paid — then revise PRICING_TABLE_VERSION "
            f"({PRICING_TABLE_VERSION}), because coverage changes the published number."
        )


def test_a_paid_cloud_the_table_never_surveyed_is_still_null_never_a_free_local_guess():
    """The invariant the coverage fix must not erode. The ``$0`` list is an allowlist, so "not one
    of the clouds we priced" still means *unknown*: pricing a paid third-party gateway at a
    confident zero would publish a false bill, which is worse than the gap it closed."""
    table = default_table()

    assert table.price_for("mistralai/mistral-large-3") is None
    assert table.price_for("gemini/gemini-3-pro") is None
    assert table.estimate_usd("acme/oracle-1", USAGE) is None
    assert table.total_usd([(LOCAL_DRIVER, USAGE), ("gemini/gemini-3-pro", USAGE)]) is None


def _override(input_rate: float = 99.0) -> dict[str, dict[str, float]]:
    return {
        "openai/gpt-5": {
            "input_usd_per_mtok": input_rate,
            "output_usd_per_mtok": 0.0,
            "cache_write_usd_per_mtok": 0.0,
            "cache_read_usd_per_mtok": 0.0,
        }
    }


def test_a_config_override_replaces_one_price_and_leaves_the_rest_of_the_table_alone():
    table = table_from_config(_override())

    assert table.estimate_usd("openai/gpt-5.4", USAGE) == 99.0
    assert table.price_for("claude-opus-4-8") == default_table().price_for("claude-opus-4-8")


def test_an_overridden_table_never_passes_itself_off_as_the_shipped_one():
    """A record must never say ``2026-07`` about numbers the shipped table did not produce."""
    overridden = table_from_config(_override())

    assert overridden.version != PRICING_TABLE_VERSION
    assert overridden.version.startswith(PRICING_TABLE_VERSION)


def test_the_same_override_always_identifies_itself_the_same_way():
    assert table_from_config(_override()).version == table_from_config(_override()).version
    assert table_from_config(_override()).version != table_from_config(_override(98.0)).version


def test_no_override_is_the_shipped_table_under_its_own_version():
    assert table_from_config(None).version == PRICING_TABLE_VERSION
    assert table_from_config({}).version == PRICING_TABLE_VERSION


def test_an_override_may_add_a_model_the_shipped_table_never_heard_of():
    table = table_from_config(
        {
            "acme/oracle": {
                "input_usd_per_mtok": 1.0,
                "output_usd_per_mtok": 2.0,
                "cache_write_usd_per_mtok": 3.0,
                "cache_read_usd_per_mtok": 4.0,
            }
        }
    )

    assert table.estimate_usd("acme/oracle-v2", USAGE) == 10.0


# ── purity ─────────────────────────────────────────────────────────────────────────────────


def test_the_pricing_module_reaches_no_io_no_clock_and_no_config():
    """It is arithmetic over data handed in, so the pure record builder can rely on it."""
    tree = ast.parse(SOURCE.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "collections", "dataclasses", "hashlib", "json", "typing"}
    text = SOURCE.read_text()
    for forbidden in ("datetime.now", "open(", "Path(", "os.environ", "Settings"):
        assert forbidden not in text, forbidden
