"""The structural lint (#159) — AST coherence checks the write gate runs before import.

Behavioural proof, not call structure: a source string in, a one-line refusal (or ``None``)
out, plus the end-to-end wiring — ``write_strategy`` refusing a duplicate ``on_bar`` through
the REAL subprocess gate, which is the only proof the check stands between an authored file
and the library.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noctis.strategies.families import FamilyRegistry
from noctis.strategies.library import (
    StrategyValidationError,
    strategy_path,
    validate_in_process,
    write_strategy,
)
from noctis.strategies.structure import check_structure
from tests.test_library import GOOD_SOURCE

SEED_DIR = Path("strategies")  # the repo's committed library (read-only in tests)


@pytest.fixture
def families():
    return FamilyRegistry()


# ── duplicate definitions ─────────────────────────────────────────────────────────────────
DUPLICATE_METHOD = '''"""A class that defines on_bar twice — the second silently wins."""


class Probe:
    def on_start(self, ctx):
        self._n = 0

    def on_bar(self, ctx, bar):
        ctx.set_target(1)

    def on_bar(self, ctx, bar):
        ctx.set_target(0)
'''

DUPLICATE_FUNCTION = '''"""Two module-level helpers under one name."""


def helper(x):
    return x + 1


def helper(x):
    return x - 1
'''

PROPERTY_PAIR = '''"""A property getter/setter pair — a decorated redefinition, which is legal."""


class Probe:
    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
'''


def test_duplicate_method_in_a_class_body_is_refused():
    msg = check_structure(DUPLICATE_METHOD)
    assert msg is not None
    assert "on_bar" in msg
    assert "class body" in msg
    assert "\n" not in msg  # one line: the REPAIR loop reads it as a single instruction


def test_duplicate_function_at_module_level_is_refused():
    msg = check_structure(DUPLICATE_FUNCTION)
    assert msg is not None
    assert "helper" in msg
    assert "module level" in msg


def test_a_decorated_redefinition_passes():
    assert check_structure(PROPERTY_PAIR) is None


def test_a_duplicate_in_a_nested_class_body_is_refused():
    # The strategy contract nests ``class Params`` inside the strategy class, so nested bodies
    # are exactly as load-bearing as the outer one.
    source = '''"""Nested duplication."""


class Probe:
    class Params:
        def scale(self):
            return 1.0

        def scale(self):
            return 2.0
'''
    msg = check_structure(source)
    assert msg is not None
    assert "scale" in msg


def test_the_first_violation_in_source_order_is_reported():
    source = '''"""Two violations; the earlier one is the one to fix first."""


def helper(x):
    return x


def helper(x):
    return x


class Probe:
    def on_bar(self, ctx, bar):
        ...

    def on_bar(self, ctx, bar):
        ...
'''
    assert "helper" in (check_structure(source) or "")


def test_an_overload_style_async_duplicate_is_refused():
    source = '''"""Async defs override each other exactly the same way."""


class Probe:
    async def fetch(self):
        ...

    async def fetch(self):
        ...
'''
    assert "fetch" in (check_structure(source) or "")


def test_distinct_names_and_same_name_in_sibling_scopes_pass():
    source = '''"""Same name, different bodies — no override anywhere."""


class A:
    def on_bar(self, ctx, bar):
        ...


class B:
    def on_bar(self, ctx, bar):
        ...


def on_bar(ctx, bar):
    ...
'''
    assert check_structure(source) is None


def test_unparsable_source_is_not_a_structure_violation():
    # A syntax error is the import step's to report, in its canonical wording; the lint stays
    # silent rather than inventing a second dialect for it.
    assert check_structure("def broken(:\n    pass\n") is None


@pytest.mark.parametrize(
    "name",
    ["TEMPLATE.py", "sma_crossover.py", "rsi_meanrev.py", "donchian_breakout.py"],
)
def test_the_committed_template_and_seeds_are_structurally_clean(name):
    assert check_structure((SEED_DIR / name).read_text(encoding="utf-8")) is None


def test_the_good_gate_fixture_is_structurally_clean():
    assert check_structure(GOOD_SOURCE) is None


# ── dead state ────────────────────────────────────────────────────────────────────────────
DEAD_STATE = '''"""on_bar branches on a flag that on_start pins to 0 and nothing ever moves."""


class Probe:
    def on_start(self, ctx):
        self._pos = 0

    def on_bar(self, ctx, bar):
        if self._pos == 1:
            ctx.set_target(1)
        ctx.set_target(0)
'''


def test_state_only_ever_assigned_a_literal_in_on_start_is_refused():
    msg = check_structure(DEAD_STATE)
    assert msg is not None
    assert "_pos" in msg  # the REPAIR loop needs the offending attribute by name
    assert "\n" not in msg


def test_a_params_derived_read_only_attribute_passes():
    # Derived config is legitimately assigned once and read forever — only a *literal* pin
    # makes the branches that read it unreachable.
    source = '''"""A read-only attribute derived from Params."""


class Probe:
    def on_start(self, ctx):
        self._floor = self.params.threshold * 2

    def on_bar(self, ctx, bar):
        ctx.set_target(1 if bar.close > self._floor else 0)
'''
    assert check_structure(source) is None


def test_a_collection_mutated_by_a_method_call_passes():
    source = '''"""A deque fed by .append every bar — mutation the assignment can't show."""

from collections import deque


class Probe:
    def on_start(self, ctx):
        self._closes = deque(maxlen=20)

    def on_bar(self, ctx, bar):
        self._closes.append(bar.close)
        ctx.set_target(1 if len(self._closes) == 20 else 0)
'''
    assert check_structure(source) is None


def test_an_attribute_reassigned_in_on_bar_passes():
    # The seed hysteresis pattern: pinned to 0 in on_start, moved by on_bar itself.
    source = '''"""Hysteresis on a position flag that on_bar drives."""


class Probe:
    def on_start(self, ctx):
        self._pos = 0

    def on_bar(self, ctx, bar):
        if self._pos == 0 and bar.close > 100:
            self._pos = 1
        elif self._pos == 1 and bar.close < 90:
            self._pos = 0
        ctx.set_target(self._pos)
'''
    assert check_structure(source) is None


def test_an_aug_assigned_counter_passes():
    source = '''"""A bar counter that moves by += only."""


class Probe:
    def on_start(self, ctx):
        self._n = 0

    def on_bar(self, ctx, bar):
        self._n += 1
        ctx.set_target(1 if self._n > 10 else 0)
'''
    assert check_structure(source) is None


def test_an_attribute_assigned_outside_on_start_passes():
    source = '''"""A flag reset by a helper the strategy calls elsewhere."""


class Probe:
    def on_start(self, ctx):
        self._pos = 0

    def reset(self):
        self._pos = 0

    def on_bar(self, ctx, bar):
        ctx.set_target(self._pos)
'''
    assert check_structure(source) is None


def test_an_attribute_this_class_never_assigns_passes():
    # Assigned by a base class, or by the engine: nothing in this file says it is dead.
    source = '''"""Reads an attribute the base class owns."""


class Probe:
    def on_bar(self, ctx, bar):
        ctx.set_target(1 if bar.close > self.entry_price else 0)
'''
    assert check_structure(source) is None


def test_a_literal_attribute_on_bar_never_reads_passes():
    # Unread state is inert, not a lie about a branch; the lint speaks only to what on_bar
    # actually decides on.
    source = '''"""A pinned attribute nothing reads."""


class Probe:
    def on_start(self, ctx):
        self._pos = 0

    def on_bar(self, ctx, bar):
        ctx.set_target(0)
'''
    assert check_structure(source) is None


def test_duplicate_definitions_are_reported_before_dead_state():
    # Fixed order: dead-state analysis over a class whose defs shadow each other reports
    # findings about code that never runs.
    source = '''"""Both defects at once."""


class Probe:
    def on_start(self, ctx):
        self._pos = 0

    def on_bar(self, ctx, bar):
        ctx.set_target(self._pos)

    def on_bar(self, ctx, bar):
        ctx.set_target(0)
'''
    msg = check_structure(source)
    assert msg is not None
    assert "duplicate definition of 'on_bar'" in msg


# ── the wiring: both validation paths refuse, and nothing lands on disk ────────────────────
DUPLICATE_ON_BAR_SOURCE = GOOD_SOURCE.replace(
    "    @classmethod\n    def param_space(cls):",
    "    def on_bar(self, ctx: Context, bar: Bar) -> None:\n"
    "        ctx.set_target(0)\n\n"
    "    @classmethod\n    def param_space(cls):",
    1,
)


def test_write_strategy_refuses_a_duplicate_on_bar_through_the_subprocess_gate(tmp_path, families):
    # The REAL subprocess validator (the default seam), so this proves the lint runs inside the
    # gate the authoring paths funnel through and its message survives the process boundary.
    with pytest.raises(StrategyValidationError, match="duplicate definition of 'on_bar'"):
        write_strategy(tmp_path, "probe", DUPLICATE_ON_BAR_SOURCE, families)
    assert strategy_path(tmp_path, "probe") is None
    assert "probe" not in families


def test_the_in_process_gate_refuses_the_same_source(tmp_path):
    path = tmp_path / "probe.py"
    path.write_text(DUPLICATE_ON_BAR_SOURCE, encoding="utf-8")
    with pytest.raises(StrategyValidationError, match="duplicate definition of 'on_bar'"):
        validate_in_process(path, "probe")


DEAD_STATE_SOURCE = GOOD_SOURCE.replace(
    "        self._closes = deque(maxlen=self.params.lookback)",
    "        self._closes = deque(maxlen=self.params.lookback)\n        self._pos = 0",
    1,
).replace(
    "        ctx.set_target(0 if mean is None else int(bar.close > mean * self.params.edge))",
    "        if self._pos == 1:\n            ctx.set_target(1)\n            return\n"
    "        ctx.set_target(0 if mean is None else int(bar.close > mean * self.params.edge))",
    1,
)


def test_write_strategy_refuses_dead_state_through_the_subprocess_gate(tmp_path, families):
    # This file imports, replays and passes its own scenarios — the dead branch only ever
    # fails to fire — so the subprocess gate is the only thing standing between it and the
    # library (#158).
    with pytest.raises(StrategyValidationError, match="_pos"):
        write_strategy(tmp_path, "probe", DEAD_STATE_SOURCE, families)
    assert strategy_path(tmp_path, "probe") is None
    assert "probe" not in families


def test_the_lint_refuses_before_the_module_is_imported(tmp_path):
    # A file that would blow up on import (a bad module-level statement) AND duplicates a method:
    # the structural message is what comes back, which can only happen if the lint reads the raw
    # source first.
    source = DUPLICATE_ON_BAR_SOURCE.replace(
        "class Probe(TraderStrategy):",
        'raise RuntimeError("import side effect")\n\n\nclass Probe(TraderStrategy):',
        1,
    )
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(StrategyValidationError, match="duplicate definition of 'on_bar'"):
        validate_in_process(path, "probe")
