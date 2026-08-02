"""``SiteKnobs`` — the typed per-site knob set an override is checked against before it is spent.

A benchmark run is money: an override that names a knob its target site does not have ("set the
coder's ``temperature``") must fail while it is still a dict, not after a hundred completions have
been billed against a setting nobody read. So every declared site names a knob *type*, and the type
is the check — :meth:`SiteKnobs.validate_overrides` refuses an override the site cannot accept and
names every offending knob in one error, the same one-pass posture as the settings overlay's
refusals (fix a bad spec in one edit, never in a fix-one-rerun loop).

**A knob set references knobs that already exist.** A concrete subclass (the site declarations in
later stories) is a frozen dataclass whose fields mirror the production settings that site reads —
the coder's model family, an episodic site's token and retry levers, distillation's cadence. It
promotes nothing: a site whose only levers are module constants declares an honestly empty knob set
rather than inventing config.

**Names, not values, at this layer.** Validation is membership: an override may only name knobs the
site declares. Type coercion belongs to whoever applies the override to a real knob set (the eval
runner), because that is where the value meets the field it is going to bind to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

__all__ = ["SiteKnobs", "UnknownKnob"]


class UnknownKnob(ValueError):
    """An override naming knobs its target site does not declare. Callers catch this one type."""


@dataclass(frozen=True)
class SiteKnobs:
    """The base shape of one site's knob set: a frozen record whose fields are the site's levers.

    Subclasses add the fields; this base adds only the two things a harness needs to reason about
    an override — the names a site accepts, and the refusal when an override strays outside them.
    """

    @classmethod
    def knob_names(cls) -> frozenset[str]:
        """Every knob this site accepts. Empty for a site whose levers are module constants."""
        return frozenset(field.name for field in fields(cls))

    @classmethod
    def validate_overrides(cls, overrides: Mapping[str, Any]) -> None:
        """Refuse an override that names a knob this site does not declare; return otherwise.

        Every unknown name is reported in one :class:`UnknownKnob`, beside the knob set it was
        measured against and the names that set does accept, so a bad harness spec is one fix.
        """
        unknown = sorted(set(overrides) - cls.knob_names())
        if not unknown:
            return
        declared = ", ".join(sorted(cls.knob_names())) or "no knobs"
        raise UnknownKnob(
            f"{cls.__name__} does not declare {', '.join(unknown)} — it accepts {declared}"
        )
