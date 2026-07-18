"""yfinance live-data feed — the free, delayed day-loop feed (behind the live-feed seam).

Pulls closed intraday OHLCV candles from Yahoo Finance and emits them as
:class:`~noctis.strategies.base.Bar` objects for the streaming driver. No credentials, no
tick assembly. Execution never goes through here — this is data only; the live execution
adapter stays a gated stub.
"""

from __future__ import annotations

from noctis.data.yfinance.feed import YFinanceBarFeed, build_yfinance_feed

__all__ = [
    "YFinanceBarFeed",
    "build_yfinance_feed",
]
