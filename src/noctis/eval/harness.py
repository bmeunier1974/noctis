"""``HarnessSpec`` — the eval-only overlay: prompt-composition ablations plus knob overrides.

"Is our prompt earning its tokens?" is only a runnable experiment if the pieces a prompt is
composed of can be taken away one at a time. This type is that dial set, and it names the five
pieces the engine's prompts are actually built from today: the authoring **contract sheet**, the
**template** shown as the canonical file shape, the complete **worked example**, the
**feasibility rules**, and the **retry-hint** enrichment an emit-failure re-prompt carries. Beside
them ride **knob overrides** — the bench layer of the knob-layering chain (site defaults → config
→ mandate overlay → bench override), validated against the target site's
:class:`~noctis.eval.knobs.SiteKnobs` type by :meth:`HarnessSpec.validate_knobs` before a run
spends anything.

**The defaults are production.** ``HarnessSpec()`` is the composition a real research session
uses — everything on, the worked example as shipped, no override — because the ablation is the
deviation and deviation is what a bench record must be able to state. :attr:`is_production` says
whether a spec deviated at all, in one boolean a record can carry.

**It lives here and only here.** A harness spec is never a field of a settings model and never a
path the mandate overlay can bind (the overlay's classification is total over ``Settings`` and
refuses anything unknown, so a harness path is refused structurally, not by an entry in a list).
That is platform invariant 3 in its sharpest form: nobody should be able to ship a research
session with the contract sheet off, and the way to guarantee that is for production config to
have no word for it. The eval runner constructs this type; the engine cannot even import it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from noctis.eval.knobs import SiteKnobs

__all__ = ["AS_SHIPPED", "AsShipped", "HarnessSpec", "WorkedExample"]


class AsShipped(Enum):
    """The 'no ablation' value of a selection-valued dial: whatever production composes."""

    TOKEN = "as_shipped"

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return "AS_SHIPPED"


AS_SHIPPED = AsShipped.TOKEN

# A worked-example selection is three-valued on purpose: the example production ships, a named
# alternative (to measure whether *which* example matters), or none at all (to measure whether an
# example matters). Collapsing "as shipped" into a name would hard-code one site's constant into a
# type every site shares.
WorkedExample = str | None | AsShipped


@dataclass(frozen=True)
class HarnessSpec:
    """One benchmark run's prompt composition: what to leave in, what to take out, what to override.

    The four boolean dials are ``True`` when the component is composed in, exactly as production
    composes it; ``worked_example`` is :data:`AS_SHIPPED`, a strategy name, or ``None``;
    ``knob_overrides`` is a read-only mapping of knob name to value, empty by default.
    """

    contract_sheet: bool = True
    template: bool = True
    worked_example: WorkedExample = AS_SHIPPED
    feasibility_rules: bool = True
    retry_hints: bool = True
    knob_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Take the caller's overrides by value, read-only: a spec is a record of what was run."""
        object.__setattr__(self, "knob_overrides", MappingProxyType(dict(self.knob_overrides)))

    @property
    def is_production(self) -> bool:
        """Whether this spec composes the prompt exactly as a real research session does."""
        return (
            self.contract_sheet
            and self.template
            and self.feasibility_rules
            and self.retry_hints
            and self.worked_example is AS_SHIPPED
            and not self.knob_overrides
        )

    def validate_knobs(self, knobs: type[SiteKnobs]) -> None:
        """Refuse this spec's overrides against a site's knob set — before a run spends a dollar.

        Raises :class:`~noctis.eval.knobs.UnknownKnob` naming every knob the site does not declare.
        """
        knobs.validate_overrides(self.knob_overrides)
