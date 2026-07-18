"""Challenger-vs-champion promotion rules — a pure decision function.

Fixed rules, in order:

1. **Activity floor** — reject if fewer than ``min_test_activity`` of the test splits had
   any market exposure. A strategy that almost never trades can post a positive average
   metric on a handful of lucky windows (noise, not edge) and then sit unbeatable at the
   top of the registry. ``0.0`` disables the gate.
2. **Gap guard** — reject if the challenger's train − test metric gap exceeds ``max_gap``
   (an overfit signal), no matter how good the test metric looks.
3. **Forward-holdout gate** — reject if a holdout metric is present and falls below
   ``min_holdout_metric``. The holdout is the most-recent slice the search never touched, so
   this is the backstop against selection/lookahead bias (and web-grounded ideation). Inert
   when the scorecard carries no ``holdout_metric``.
4. **Symbol-holdout gate** — reject if a symbol-holdout metric is present and falls below
   ``min_symbol_holdout_metric``. The held-out symbols were never used in tuning or
   selection, so this is the cross-sectional twin of the temporal gate. Inert when the
   scorecard carries no ``symbol_holdout_metric`` (no symbol-holdout set was reserved).
5. **Consistency gate (optional)** — reject if fewer than ``min_symbol_consistency`` of the
   fit symbols have a positive per-symbol test metric. ``0.0`` disables it (the default —
   a legitimately specialized strategy should not be punished for breadth).
6. **Free slot** — if the registry is below capacity and the challenger clears the minimum
   out-of-sample bar, promote.
7. **Stale champions lose first** — a champion whose scorecard was scored under a different
   metric than the challenger's carries a number in different units; comparing across
   metrics is meaningless. A stale slot behaves like a free slot: any challenger clearing
   the minimum bar displaces the stale champion, and one below the bar is rejected.
8. **Beat the weakest** — otherwise promote iff the challenger's out-of-sample test metric
   beats the weakest champion's; that champion is demoted. On panel scorecards the test
   metric is the panel mean, so champions with different fit sets compare on the same
   scale-free, risk-adjusted footing.

Every outcome carries a one-line rationale for memory and the close report.
"""

from __future__ import annotations

from dataclasses import dataclass

from noctis.backtest.scorecard import Scorecard


@dataclass(frozen=True)
class PromotionRules:
    champion_count: int = 3
    max_gap: float = 1.0
    min_test_metric: float = 0.0
    # Minimum metric on the forward-holdout window; only enforced when the scorecard has one.
    min_holdout_metric: float = 0.0
    # Minimum metric on the held-out symbols; only enforced when the scorecard has one.
    min_symbol_holdout_metric: float = 0.0
    # Minimum fraction of fit symbols with a positive test metric; 0.0 = gate off.
    min_symbol_consistency: float = 0.0
    # Minimum fraction of test splits with market exposure; 0.0 = gate off.
    min_test_activity: float = 0.0
    # Reverse-gap guard: reject when test exceeds train by more than this (a large negative
    # train−test gap is a degeneracy signal, the mirror of the overfit guard); 0.0 = gate off.
    max_reverse_gap: float = 0.0
    # Magnitude sanity cap: reject when |test metric| exceeds this ceiling; 0.0 = gate off.
    max_test_metric: float = 0.0

    @classmethod
    def from_settings(cls, settings) -> PromotionRules:
        """The one mapping from typed config to promotion rules (every entrypoint uses it).

        ``settings`` is duck-typed (anything with ``champion_count`` + ``promotion.*``) so
        this module stays pure — no config import, unit-testable with a stub.
        """
        promotion = settings.promotion
        return cls(
            champion_count=settings.champion_count,
            max_gap=promotion.max_gap,
            min_test_metric=promotion.min_test_metric,
            min_holdout_metric=promotion.min_holdout_metric,
            min_symbol_holdout_metric=promotion.min_symbol_holdout_metric,
            min_symbol_consistency=promotion.min_symbol_consistency,
            min_test_activity=promotion.min_test_activity,
            max_reverse_gap=promotion.max_reverse_gap,
            max_test_metric=promotion.max_test_metric,
        )


@dataclass(frozen=True)
class Decision:
    promote: bool
    rationale: str
    demote_index: int | None = None  # index into the champions list to demote, if any


def decide(challenger: Scorecard, champions: list[Scorecard], rules: PromotionRules) -> Decision:
    """Judge a challenger against the current champions. Pure: scorecards in, decision out."""
    if challenger.stage != "validated" or not challenger.symbols:
        return Decision(False, "rejected: challenger was not validated (no out-of-sample splits)")

    metric = challenger.avg_test_metric
    gap = challenger.gap

    # 1) activity floor — a strategy that almost never trades can post a positive average
    # on a handful of lucky test windows and then sit unbeatable at the top of the registry
    if rules.min_test_activity > 0.0:
        activity = challenger.test_activity
        if activity < rules.min_test_activity:
            return Decision(
                False,
                f"rejected: only {activity:.4f} of test splits traded, below "
                f"{rules.min_test_activity:.2f} (activity floor)",
            )

    # 2) overfit guard
    if gap > rules.max_gap:
        return Decision(
            False,
            f"rejected: train−test gap {gap:.4f} exceeds max {rules.max_gap:.4f} (overfit)",
        )

    # 2b) degeneracy guards — the mirror of overfit and a magnitude sanity cap. A test metric
    # that wildly EXCEEDS train (large negative gap), or an implausibly large test metric, is a
    # noise signal (e.g. a near-zero-downside split annualizing into the thousands) — not edge.
    # Without these a noise scorecard is crowned and, being unbeatable, freezes the registry.
    if rules.max_reverse_gap > 0.0 and -gap > rules.max_reverse_gap:
        return Decision(
            False,
            f"rejected: test exceeds train by {-gap:.4f}, over max {rules.max_reverse_gap:.4f} "
            f"(degenerate — likely noise, not a robust edge)",
        )
    if rules.max_test_metric > 0.0 and abs(metric) > rules.max_test_metric:
        return Decision(
            False,
            f"rejected: |test metric| {abs(metric):.4f} exceeds sane ceiling "
            f"{rules.max_test_metric:.4f} (degenerate metric)",
        )

    # 3) forward-holdout gate — must survive on data the search never touched
    holdout = challenger.holdout_metric
    if holdout is not None and holdout < rules.min_holdout_metric:
        return Decision(
            False,
            f"rejected: holdout metric {holdout:.4f} below bar "
            f"{rules.min_holdout_metric:.4f} (forward-holdout gate)",
        )

    # 4) symbol-holdout gate — must survive on symbols the search never touched
    symbol_holdout = challenger.symbol_holdout_metric
    if symbol_holdout is not None and symbol_holdout < rules.min_symbol_holdout_metric:
        return Decision(
            False,
            f"rejected: symbol-holdout metric {symbol_holdout:.4f} below bar "
            f"{rules.min_symbol_holdout_metric:.4f} (symbol-holdout gate)",
        )

    # 5) optional breadth gate — fraction of fit symbols with a positive test metric
    if rules.min_symbol_consistency > 0.0 and challenger.symbols:
        per_symbol = challenger.symbol_test_metrics()
        consistency = sum(1 for v in per_symbol.values() if v > 0.0) / len(per_symbol)
        if consistency < rules.min_symbol_consistency:
            return Decision(
                False,
                f"rejected: only {consistency:.2f} of fit symbols positive, below "
                f"{rules.min_symbol_consistency:.2f} (symbol-consistency gate)",
            )

    # 6) free capacity
    if len(champions) < rules.champion_count:
        if metric > rules.min_test_metric:
            return Decision(
                True,
                f"promoted: test metric {metric:.4f} clears bar {rules.min_test_metric:.4f} "
                f"into a free slot (gap {gap:.4f})",
            )
        return Decision(
            False,
            f"rejected: test metric {metric:.4f} below minimum bar {rules.min_test_metric:.4f}",
        )

    # 7) stale champions lose first — a champion scored under a different metric carries a
    # number in different units, so its stored value cannot be compared. A stale slot
    # behaves like a free slot: the minimum bar applies, nothing more.
    stale = [i for i, c in enumerate(champions) if c.metric_name != challenger.metric_name]
    if stale:
        if metric > rules.min_test_metric:
            idx = stale[0]
            return Decision(
                True,
                f"promoted: test metric {metric:.4f} clears bar {rules.min_test_metric:.4f}; "
                f"displacing stale champion scored on '{champions[idx].metric_name}' "
                f"(current metric '{challenger.metric_name}')",
                demote_index=idx,
            )
        return Decision(
            False,
            f"rejected: test metric {metric:.4f} below minimum bar "
            f"{rules.min_test_metric:.4f} (stale champion present, but the bar still applies)",
        )

    # 8) beat the weakest champion
    weakest_index = min(range(len(champions)), key=lambda i: champions[i].avg_test_metric)
    weakest_metric = champions[weakest_index].avg_test_metric
    if metric > weakest_metric:
        return Decision(
            True,
            f"promoted: test metric {metric:.4f} beats weakest champion {weakest_metric:.4f}; "
            f"demoting the weakest (gap {gap:.4f})",
            demote_index=weakest_index,
        )
    return Decision(
        False,
        f"rejected: test metric {metric:.4f} does not beat weakest champion {weakest_metric:.4f}",
    )
