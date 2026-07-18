"""The vendor client seam.

Every metered vendor call (cost estimate, data fetch) goes through a :class:`VendorClient`.
Keeping this a thin, single-symbol interface lets the fetch-once tests inject a mock that
**counts calls** — the whole "no byte bought twice" contract is verified by asserting that
the fetch count matches only the missing slices.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class VendorClient(Protocol):
    """A metered historical market-data vendor (e.g. DataBento), one symbol at a time."""

    def get_cost(self, *, dataset: str, schema: str, symbol: str, start: int, end: int) -> float:
        """Estimated USD cost of fetching ``[start, end]`` (metadata call, no data)."""
        ...

    def fetch_bars(
        self, *, dataset: str, schema: str, symbol: str, start: int, end: int
    ) -> pd.DataFrame:
        """Fetch bars for ``[start, end]`` (inclusive ns). Returns canonical bar columns.

        This is the only method that actually **spends** and downloads data.
        """
        ...
