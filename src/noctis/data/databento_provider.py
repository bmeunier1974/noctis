"""DataBento historical provider (behind the ``VendorClient`` seam).

Uses the official ``databento`` client — an optional dependency (the ``data`` extra). The
package is imported lazily so the module stays importable without it; the client only spends
when :meth:`fetch_bars` runs and only after the cost preflight has cleared the request.

Default dataset is ``EQUS.MINI`` (the cheapest consolidated view); venue-precise datasets
are used only by explicit override. The one legitimate full-history re-fetch is
split/dividend adjustment factors — a new split re-scales all prior history.
"""

from __future__ import annotations

import pandas as pd

from noctis.data.types import BAR_COLUMNS, ns_to_timestamp


class DataBentoVendorClient:
    """Metered historical vendor client backed by ``databento.Historical``."""

    def __init__(self, api_key: str, default_schema: str = "ohlcv-1m"):
        if not api_key:
            raise ValueError("DATABENTO_API_KEY is required for the DataBento provider")
        self.api_key = api_key
        self.default_schema = default_schema
        self._client = None  # lazily constructed

    def _historical(self):
        if self._client is None:
            try:
                import databento as db
            except ImportError as exc:  # pragma: no cover - exercised only with the extra
                raise ImportError(
                    "The 'databento' package is required for the DataBento provider. "
                    "Install the data extra: pip install -e '.[data]'"
                ) from exc
            self._client = db.Historical(self.api_key)
        return self._client

    def get_cost(self, *, dataset: str, schema: str, symbol: str, start: int, end: int) -> float:
        client = self._historical()
        return float(
            client.metadata.get_cost(
                dataset=dataset,
                symbols=[symbol],
                schema=schema or self.default_schema,
                start=ns_to_timestamp(start),
                end=ns_to_timestamp(end),
            )
        )

    def fetch_bars(
        self, *, dataset: str, schema: str, symbol: str, start: int, end: int
    ) -> pd.DataFrame:
        client = self._historical()
        store = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema=schema or self.default_schema,
            start=ns_to_timestamp(start),
            end=ns_to_timestamp(end),
        )
        return self._to_bars(store.to_df())

    @staticmethod
    def _to_bars(df: pd.DataFrame) -> pd.DataFrame:
        """Map a DataBento OHLCV DataFrame to the canonical bar columns."""
        out = df.reset_index()
        # DataBento to_df() indexes on ts_event; ensure it is int64 UTC ns.
        ts = pd.DatetimeIndex(pd.to_datetime(out["ts_event"], utc=True))
        result = pd.DataFrame(
            {
                "ts_event": ts.asi8,
                "open": out["open"].astype("float64"),
                "high": out["high"].astype("float64"),
                "low": out["low"].astype("float64"),
                "close": out["close"].astype("float64"),
                "volume": out["volume"].astype("int64"),
            }
        )
        return result.loc[:, list(BAR_COLUMNS)]

    def refresh_adjustment_factors(self, *, dataset: str, symbols: list[str]):  # pragma: no cover
        """The one legitimate full-history re-fetch: corporate-action adjustment factors.

        A new split re-scales all prior history, so these refresh full-history rather than
        tail-only. Returned as a DataFrame of adjustment events for the caller to apply.
        """
        client = self._historical()
        return client.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            schema="adjustment",
            start="1970-01-01",
        ).to_df()
