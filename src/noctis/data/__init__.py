"""Noctis data subpackage — the fetch-once market-data lake.

The ``MarketData`` seam serves bars from a Parquet catalog fronted by a single lake owner,
a coverage registry, coverage-diffed ingest, an incremental updater, an integrity check, and
a padded cost preflight. No byte is ever bought twice.
"""

from __future__ import annotations

from noctis.data.coverage import CoverageRecord, CoverageRegistry
from noctis.data.ingest import Ingestor, IngestResult, missing_slices
from noctis.data.integrity import IntegrityChecker, IntegrityReport
from noctis.data.lake import Lake, ManifestEntry
from noctis.data.preflight import BudgetExceededError, CostDecision, CostPreflight
from noctis.data.seam import MarketData, MarketDataLake
from noctis.data.types import SeriesKey
from noctis.data.updater import SyncResult, Updater
from noctis.data.vendor import VendorClient

__all__ = [
    "CoverageRecord",
    "CoverageRegistry",
    "IngestResult",
    "Ingestor",
    "missing_slices",
    "IntegrityChecker",
    "IntegrityReport",
    "Lake",
    "ManifestEntry",
    "BudgetExceededError",
    "CostDecision",
    "CostPreflight",
    "MarketData",
    "MarketDataLake",
    "SeriesKey",
    "SyncResult",
    "Updater",
    "VendorClient",
]
