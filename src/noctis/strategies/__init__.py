"""Noctis strategies — the abstraction shared by research and live.

Each strategy exposes a vectorised ``signals()`` (for the fast pre-filter) and an
incremental ``on_bar()`` (for event-driven validation and live); a golden test proves the
two paths agree. A :class:`FamilyRegistry` (built in ``noctis.bootstrap``) instantiates
families by name + params.
"""

from __future__ import annotations

from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy, params_to_dict
from noctis.strategies.candidate import Candidate
from noctis.strategies.donchian_breakout import DonchianBreakout, DonchianParams
from noctis.strategies.families import SEED_FAMILIES, FamilyRegistry
from noctis.strategies.proposer import CandidateProposer, signature
from noctis.strategies.rsi_meanrev import RsiMeanReversion, RsiParams
from noctis.strategies.sma_crossover import SmaCrossover, SmaParams

__all__ = [
    "Bar",
    "Context",
    "ParamSpec",
    "TraderStrategy",
    "params_to_dict",
    "Candidate",
    "CandidateProposer",
    "signature",
    "SmaCrossover",
    "SmaParams",
    "RsiMeanReversion",
    "RsiParams",
    "DonchianBreakout",
    "DonchianParams",
    "FamilyRegistry",
    "SEED_FAMILIES",
]
