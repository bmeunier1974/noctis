"""The write gate owns the oracle (#84) — spec-path validation and machine-stamped scenarios().

When ``write_strategy`` is handed a compiled scenario spec (a :class:`SpecSuite`), the gate
resolves ``warm`` from the candidate's own declared warmup, replays the compiled oracle against
the candidate, rejects any coder-authored ``scenarios()`` block, and — on success — machine-stamps
a warmup-parametric ``scenarios()`` into the installed file so the file stays the whole artifact.

The spec-less path is strictly unchanged: no spec, no stamping, byte-identical behavior.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.library import (
    StrategyValidationError,
    fixture_frame,
    load_and_register,
    strategy_path,
    strategy_source,
    write_strategy,
)
from noctis.strategies.scenario_spec import (
    Behavior,
    LegSpec,
    ScenarioSpec,
    SpecSuite,
)
from noctis.strategies.scenarios import check_scenario_contract

# ── candidate sources (no scenarios() — the gate owns the oracle on the spec path) ──────────
# Long above its own SMA, else flat: scale-free (ratio), deterministic (state reset in
# on_start), and it declares an honest warmup equal to its lookback. It carries NO scenarios()
# — on the spec path the gate stamps one in.
SPEC_CANDIDATE = '''"""Toy probe: long above its own moving average.

status: draft
style: momentum
"""

from collections import deque
from dataclasses import dataclass

from noctis.strategies import indicators as ind
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class Probe(TraderStrategy):
    name = "probe"

    @dataclass(frozen=True)
    class Params:
        lookback: int = 10

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        self._closes = deque(maxlen=self.params.lookback)

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._closes.append(bar.close)
        mean = ind.sma(self._closes, self.params.lookback)
        if mean is None or bar.close == mean:
            ctx.set_target(0)
        else:
            ctx.set_target(1 if bar.close > mean else 0)

    @classmethod
    def warmup_bars(cls, params) -> int:
        return params.lookback

    @classmethod
    def param_space(cls):
        return [ParamSpec("lookback", "int", 5, 40, 1)]
'''

# The same candidate but declaring an absurd warmup — larger than the fixed oracle's tape can
# hold, so the compiled tape overruns the 2000-bar maximum at validation time.
HUGE_WARMUP_CANDIDATE = SPEC_CANDIDATE.replace(
    "    def warmup_bars(cls, params) -> int:\n        return params.lookback",
    "    def warmup_bars(cls, params) -> int:\n        return 3000",
)

# The candidate with a coder-authored scenarios() block bolted on — forbidden on the spec path.
CODER_SCENARIOS = """
    @classmethod
    def scenarios(cls):
        from noctis.strategies import scenarios as sc

        return [
            sc.Scenario("up", segments=[sc.flat(20), sc.trend(60, 0.2)],
                        expect=[sc.long_within(21, 79)]),
            sc.Scenario("flat", segments=[sc.flat(80)], expect=[sc.always_flat()]),
        ]
"""
CANDIDATE_WITH_CODER_SCENARIOS = SPEC_CANDIDATE + CODER_SCENARIOS


def _suite() -> SpecSuite:
    """A minimal contract-satisfying oracle: one directional entry + one no-trade tape."""
    return SpecSuite(
        [
            ScenarioSpec("rally", [LegSpec("trend", 60, pct=0.15)], Behavior.ENTER_LONG, leg=0),
            ScenarioSpec("grind", [LegSpec("flat", 60)], Behavior.NEVER_TRADE),
        ]
    )


@pytest.fixture
def families() -> FamilyRegistry:
    return FamilyRegistry()


# ── 1. a spec-supplied write validates against the compiled oracle (both runners) ───────────
def test_spec_write_validates_and_stamps_through_the_in_process_gate(tmp_path, families, fast_gate):
    result = write_strategy(tmp_path, "probe", SPEC_CANDIDATE, families, spec=_suite())
    assert result["name"] == "probe"
    installed = strategy_source(tmp_path, "probe")
    # The gate machine-stamped a warmup-parametric scenarios() that re-derives from the spec.
    assert "def scenarios(cls):" in installed
    assert "compile_spec(" in installed
    assert "spec_from_json(" in installed
    assert "cls.warmup_bars(cls.params_cls())" in installed
    # And the candidate carried none of its own — the coder never wrote a scenarios() here.
    assert installed.count("def scenarios(cls):") == 1


def test_spec_write_validates_and_stamps_through_the_subprocess_gate(tmp_path, families):
    # The DEFAULT fresh-interpreter validator: the spec crosses the process boundary and the
    # stamped file lands, proving the JSON carrier + subprocess entry point wiring end to end.
    assert library.validator is library.validate_in_subprocess
    result = write_strategy(tmp_path, "probe", SPEC_CANDIDATE, families, spec=_suite())
    assert result["name"] == "probe"
    installed = strategy_source(tmp_path, "probe")
    assert "compile_spec(" in installed and "spec_from_json(" in installed


# ── 2. warmup-too-large: a precise, actionable failure that points at shrinking lookback ─────
def test_huge_warmup_rejected_with_actionable_message_in_process(tmp_path, families, fast_gate):
    with pytest.raises(StrategyValidationError) as exc:
        write_strategy(tmp_path, "probe", HUGE_WARMUP_CANDIDATE, families, spec=_suite())
    msg = str(exc.value)
    assert "3000" in msg  # names the declared warmup
    assert "shrink" in msg.lower() and "lookback" in msg.lower()  # the honest fix
    assert strategy_path(tmp_path, "probe") is None  # nothing lands on disk


def test_huge_warmup_message_survives_the_subprocess_boundary(tmp_path, families):
    with pytest.raises(StrategyValidationError) as exc:
        write_strategy(tmp_path, "probe", HUGE_WARMUP_CANDIDATE, families, spec=_suite())
    msg = str(exc.value)
    assert "3000" in msg
    assert "shrink" in msg.lower() and "lookback" in msg.lower()
    assert strategy_path(tmp_path, "probe") is None


# ── 3. a coder-authored scenarios() block is rejected — the coder cannot re-fit the oracle ───
def test_coder_authored_scenarios_are_rejected_on_the_spec_path(tmp_path, families, fast_gate):
    with pytest.raises(StrategyValidationError) as exc:
        write_strategy(tmp_path, "probe", CANDIDATE_WITH_CODER_SCENARIOS, families, spec=_suite())
    msg = str(exc.value).lower()
    assert "oracle is fixed" in msg or "scenarios()" in msg
    assert "trading logic" in msg or "on_bar" in msg  # tells the coder what MAY change
    assert strategy_path(tmp_path, "probe") is None


def test_coder_authored_scenarios_rejected_through_the_subprocess_gate(tmp_path, families):
    with pytest.raises(StrategyValidationError) as exc:
        write_strategy(tmp_path, "probe", CANDIDATE_WITH_CODER_SCENARIOS, families, spec=_suite())
    assert "scenarios" in str(exc.value).lower()
    assert strategy_path(tmp_path, "probe") is None


# ── 4. the stamped file is the whole artifact: reload + scenarios + a backtest replay ────────
def test_stamped_file_reloads_and_replays_standalone(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", SPEC_CANDIDATE, families, spec=_suite())

    # Reload the installed file into a *fresh* registry — nothing from the write session survives.
    fresh = FamilyRegistry()
    load_and_register(tmp_path, fresh)
    cls = fresh.get_class("probe")
    # Its stamped scenarios() re-derive the oracle at the declared warmup and pass the contract.
    check_scenario_contract(cls)
    # And it replays like any shipped strategy on its own defaults (the `noctis backtest` seam).
    card = evaluate(
        Candidate("probe", {}),
        {"AAA": fixture_frame(n=320)},
        config=PipelineConfig(prefilter_min_score=None),
        families=fresh,
    )
    assert card.stage == "validated"
    assert card.symbols["AAA"].splits


def test_stamped_scenarios_are_warmup_parametric(tmp_path, families, fast_gate):
    # The stamped block re-derives the oracle from the embedded spec at the strategy's declared
    # warmup — the setup leg tracks warmup_bars, so no bar index is frozen into the file.
    from noctis.strategies.scenario_spec import compile_spec

    write_strategy(tmp_path, "probe", SPEC_CANDIDATE, families, spec=_suite())
    cls = families.get_class("probe")
    stamped = cls.scenarios()
    warm = cls.warmup_bars(cls.params_cls())
    # The stamped scenarios equal the spec compiled at the DECLARED warmup...
    assert stamped == list(compile_spec(_suite(), warm))
    # ...and that setup-leg size genuinely tracks warmup (a different warm ⇒ a different leg).
    near = compile_spec(_suite(), warm)[0].segments[0].n
    far = compile_spec(_suite(), warm + 15)[0].segments[0].n
    assert near != far


# ── 5. spec-less path is byte-identical apart from the richer messages already shipped ───────
def test_spec_less_write_is_unchanged_and_does_not_stamp(tmp_path, families, fast_gate):
    # A hand-declared-scenarios source authored WITHOUT a spec installs byte-for-byte — no stamp,
    # no oracle rewrite. (Uses the library test's canonical GOOD_SOURCE, which declares its own.)
    from tests.test_library import GOOD_SOURCE

    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)  # no spec
    assert strategy_source(tmp_path, "probe") == GOOD_SOURCE  # installed bytes == input bytes


def test_stamped_file_is_ruff_format_stable(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", SPEC_CANDIDATE, families, spec=_suite())
    installed = Path(strategy_path(tmp_path, "probe"))
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", str(installed)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"stamped file is not ruff-format-stable:\n{proc.stdout}\n{proc.stderr}"
    )
