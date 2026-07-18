"""The cost preflight — the single owner of the vendor cost call.

Vendor estimates drift, so the raw estimate is padded (+20 % by default). If the padded
cost fits the budget the ingest proceeds autonomously; otherwise it is refused and the
estimate is surfaced. ``dry_run`` prices a request without ever spending.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PAD = 0.20


@dataclass(frozen=True)
class CostDecision:
    allowed: bool
    raw_cost: float
    padded_cost: float
    budget_usd: float
    reason: str


class BudgetExceededError(RuntimeError):
    """Raised when a padded cost estimate exceeds the configured data budget."""

    def __init__(self, decision: CostDecision):
        self.decision = decision
        super().__init__(
            f"DataBento ingest refused: padded estimate ${decision.padded_cost:.4f} "
            f"exceeds budget ${decision.budget_usd:.2f} "
            f"(raw ${decision.raw_cost:.4f}, +{int(DEFAULT_PAD * 100)}% pad)."
        )


class CostPreflight:
    """Budget-gated cost check with a padded estimate."""

    def __init__(self, budget_usd: float, pad: float = DEFAULT_PAD):
        self.budget_usd = float(budget_usd)
        self.pad = float(pad)

    def decide(self, raw_cost: float) -> CostDecision:
        padded = float(raw_cost) * (1.0 + self.pad)
        allowed = padded <= self.budget_usd
        reason = (
            f"padded ${padded:.4f} <= budget ${self.budget_usd:.2f}"
            if allowed
            else f"padded ${padded:.4f} > budget ${self.budget_usd:.2f}"
        )
        return CostDecision(
            allowed=allowed,
            raw_cost=float(raw_cost),
            padded_cost=padded,
            budget_usd=self.budget_usd,
            reason=reason,
        )
