"""The promotion **decision corpus** — one frozen case per branch of ``decide()``.

The gates are the arbiter of quality by design (AGENTS.md rule 2), so a change to
``champions/promotion.py`` has to be provable rather than plausible: every case here is fed to
``decide()`` and its ``(promote, rationale, demote_index)`` outcome is snapshotted into
``tests/fixtures/promotion_decisions_golden.json``. The golden was generated from the promotion
module **as it stood before** structured gate evidence landed (story #141), so
``tests/test_gate_evidence.py`` proves bit-identical outcomes rather than asserting them.

Two rules for this file:

* **Every branch, and both sides of every boundary.** A gate that trips only on ``>`` must have a
  case sitting exactly *on* the threshold, or a ``>=`` slipping in tomorrow would go unseen.
* **Regenerating the golden is a decision, not a fix.** If a case's outcome moves, a *judgment*
  moved: that is an engine change, and it belongs in a PR that bumps ``ENGINE_VERSION``. Never
  regenerate to make a red test green.

Nothing here reads a clock, a file or the configuration — the cards are stated by hand, so the
corpus is as deterministic as the function it pins.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from noctis.backtest.scorecard import Metrics, Scorecard, SplitScore, SymbolScore
from noctis.champions import PromotionRules

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "promotion_decisions_golden.json"


def _metrics(value: float, *, exposure: float = 1.0) -> Metrics:
    """One split's metrics: the election metric's value, and whether the split traded at all."""
    return Metrics(
        total_return=0.0,
        sharpe=value,
        sortino=value,
        max_drawdown=0.0,
        win_rate=0.0,
        turnover=0.0,
        exposure=exposure,
    )


def card(
    family: str = "vol_breakout",
    *,
    test: float = 1.0,
    train: float | None = None,
    per_symbol: dict[str, float] | None = None,
    exposures: tuple[float, ...] = (1.0,),
    holdout: float | None = None,
    symbol_holdout: float | None = None,
    metric_name: str = "sharpe",
    stage: str = "validated",
    symbols: bool = True,
) -> Scorecard:
    """A validated panel card whose aggregates are exactly the numbers asked for.

    ``per_symbol`` builds a real panel (each symbol one split at its own test metric); otherwise
    the card is a panel of one with ``len(exposures)`` splits, so the activity floor has real
    per-split exposure to read.
    """
    train_metric = test if train is None else train
    if per_symbol is not None:
        panel = {
            symbol: SymbolScore(splits=[SplitScore(0, _metrics(train_metric), _metrics(value))])
            for symbol, value in per_symbol.items()
        }
    else:
        panel = {
            "FIT": SymbolScore(
                splits=[
                    SplitScore(index, _metrics(train_metric), _metrics(test, exposure=exposure))
                    for index, exposure in enumerate(exposures)
                ]
            )
        }
    return Scorecard(
        family=family,
        params={"lookback": 40},
        metric_name=metric_name,
        stage=stage,
        holdout_metric=holdout,
        symbol_holdout_metric=symbol_holdout,
        symbols=panel if symbols else {},
    )


def rules(**overrides) -> PromotionRules:
    """The default board, with exactly the gates a case re-arms."""
    base = dict(champion_count=3, max_gap=1.0, min_test_metric=0.0)
    return PromotionRules(**(base | overrides))


@dataclass(frozen=True)
class Case:
    """One frozen decision: a challenger, the board it faces, and the rules that judge it."""

    name: str
    challenger: Scorecard
    champions: list[Scorecard]
    rules: PromotionRules


def cases() -> Iterator[Case]:
    """Every branch of ``decide()``, and both sides of every threshold it compares on."""
    full = [card("champ_a", test=1.0), card("champ_b", test=0.5), card("champ_c", test=2.0)]

    yield Case("not_validated", card(stage="prefilter_rejected"), [], rules())
    yield Case("no_panel", card(symbols=False), [], rules())

    # 1) activity floor
    yield Case(
        "activity_below_floor",
        card(exposures=(1.0, 0.0, 0.0, 0.0)),
        [],
        rules(min_test_activity=0.5),
    )
    yield Case(
        "activity_exactly_on_floor",
        card(exposures=(1.0, 1.0, 0.0, 0.0)),
        [],
        rules(min_test_activity=0.5),
    )
    yield Case(
        "activity_floor_disabled",
        card(exposures=(0.0, 0.0)),
        [],
        rules(min_test_activity=0.0),
    )

    # 2) overfit gap guard, and its two degeneracy mirrors
    yield Case("gap_over_max", card(test=1.0, train=2.5), [], rules())
    yield Case("gap_exactly_max", card(test=1.0, train=2.0), [], rules())
    yield Case("reverse_gap_over_max", card(test=3.0, train=1.0), [], rules(max_reverse_gap=1.0))
    yield Case("reverse_gap_exactly_max", card(test=2.0, train=1.0), [], rules(max_reverse_gap=1.0))
    yield Case("reverse_gap_disabled", card(test=9.0, train=1.0), [], rules())
    capped = rules(max_test_metric=50.0)
    yield Case("magnitude_over_cap", card(test=74601.0, train=74601.5), [], capped)
    yield Case("magnitude_exactly_cap", card(test=50.0, train=50.0), [], capped)
    yield Case("magnitude_cap_disabled", card(test=74601.0, train=74601.5), [], rules())

    # 3) forward holdout
    yield Case("holdout_below_bar", card(holdout=-0.5), [], rules())
    yield Case("holdout_exactly_bar", card(holdout=0.0), [], rules())
    yield Case("holdout_absent", card(holdout=None), [], rules(min_holdout_metric=5.0))

    # 4) symbol holdout
    yield Case("symbol_holdout_below_bar", card(symbol_holdout=-0.2), [], rules())
    yield Case("symbol_holdout_exactly_bar", card(symbol_holdout=0.0), [], rules())
    yield Case(
        "symbol_holdout_absent", card(symbol_holdout=None), [], rules(min_symbol_holdout_metric=5.0)
    )

    # 5) symbol consistency
    yield Case(
        "consistency_below_bar",
        card(per_symbol={"AAA": 1.5, "BBB": -0.4, "CCC": -0.1}),
        [],
        rules(min_symbol_consistency=0.6),
    )
    yield Case(
        "consistency_exactly_bar",
        card(per_symbol={"AAA": 1.5, "BBB": 1.2, "CCC": -0.1}),
        [],
        rules(min_symbol_consistency=0.66),
    )
    yield Case(
        "consistency_disabled",
        card(per_symbol={"AAA": 1.5, "BBB": -0.4, "CCC": -0.1}),
        [],
        rules(),
    )

    # 6) free capacity
    yield Case("free_slot_above_bar", card(test=1.5), [], rules())
    yield Case("free_slot_exactly_on_bar", card(test=0.0), [], rules())
    yield Case("free_slot_below_bar", card(test=-0.2), [], rules())
    yield Case("free_slot_beside_one_champion", card(test=0.2), full[:1], rules())

    # 7) stale champions lose first
    stale_board = [card("champ_a", test=1.0, metric_name="sortino"), card("champ_b", test=9.0)]
    yield Case("stale_champion_displaced", card(test=0.3), stale_board, rules(champion_count=2))
    yield Case(
        "stale_champion_but_below_bar", card(test=-0.1), stale_board, rules(champion_count=2)
    )

    # 8) beat the weakest
    yield Case("beats_weakest", card(test=0.9), full, rules())
    yield Case("ties_weakest", card(test=0.5), full, rules())
    yield Case("does_not_beat_weakest", card(test=0.4), full, rules())


def outcomes() -> dict[str, dict[str, object]]:
    """Every case's decision, reduced to the three fields that *are* the decision."""
    from noctis.champions import decide

    snapshot: dict[str, dict[str, object]] = {}
    for case in cases():
        decision = decide(case.challenger, case.champions, case.rules)
        snapshot[case.name] = {
            "promote": decision.promote,
            "rationale": decision.rationale,
            "demote_index": decision.demote_index,
        }
    return snapshot


def write_golden() -> Path:
    """Regenerate the golden. **A deliberate act** — see this module's docstring."""
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(outcomes(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return GOLDEN


def read_golden() -> dict[str, dict[str, object]]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - the regeneration entry point
    print(write_golden())
