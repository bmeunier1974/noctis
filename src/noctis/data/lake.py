"""The lake owner — the *only* module that writes to the Parquet catalog.

Every write appends (deduped, sorted), rewrites the series' Parquet file, and restamps
``manifest.json`` with the row count, first/last timestamp, and a content checksum. Nothing
else touches the files, so the manifest is always an accurate fingerprint of what is on
disk — which is exactly what the integrity check compares against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from noctis.data.types import SeriesKey, empty_bars, normalize_bars


@dataclass(frozen=True)
class ManifestEntry:
    slug: str
    rel_path: str
    row_count: int
    first_ts: int | None
    last_ts: int | None
    checksum: str
    updated_at: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Lake:
    """Sole writer/reader of the Parquet catalog and its manifest."""

    def __init__(self, lake_dir: str | Path):
        self.root = Path(lake_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    # --- paths ---
    def path_for(self, key: SeriesKey) -> Path:
        return self.root / key.rel_path

    # --- reads ---
    def read(self, key: SeriesKey) -> pd.DataFrame:
        """Read a series from the catalog. Missing series → empty canonical frame."""
        path = self.path_for(key)
        if not path.is_file():
            return empty_bars()
        return normalize_bars(pd.read_parquet(path))

    # --- writes (the single write path) ---
    def write(self, key: SeriesKey, bars: pd.DataFrame) -> ManifestEntry:
        """Append ``bars`` to the series (dedup on ts_event), rewrite, restamp manifest.

        Passing an empty frame compacts the existing file (dedup + resort) without adding
        rows — used by integrity repair.
        """
        path = self.path_for(key)
        existing = self.read(key)
        combined = normalize_bars(pd.concat([existing, normalize_bars(bars)], ignore_index=True))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: land the new Parquet in a temp file, then replace in one step so a
        # crash mid-write can never leave a torn file that breaks a later read. Only once the
        # file is safely in place do we restamp the manifest (also atomic, tmp+replace).
        tmp = path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(path)
        entry = self._stamp(key, combined, path)
        return entry

    # --- manifest ---
    def load_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text())

    def _stamp(self, key: SeriesKey, df: pd.DataFrame, path: Path) -> ManifestEntry:
        first_ts = int(df["ts_event"].iloc[0]) if len(df) else None
        last_ts = int(df["ts_event"].iloc[-1]) if len(df) else None
        entry = ManifestEntry(
            slug=key.slug,
            rel_path=key.rel_path,
            row_count=int(len(df)),
            first_ts=first_ts,
            last_ts=last_ts,
            checksum=_sha256(path),
            updated_at=datetime.now(UTC).isoformat(),
        )
        manifest = self.load_manifest()
        manifest[key.slug] = asdict(entry)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        tmp.replace(self.manifest_path)
        return entry

    def manifest_matches_disk(self, key: SeriesKey, *, frame: pd.DataFrame | None = None) -> bool:
        """True iff the manifest checksum + row count match the file on disk.

        ``frame`` lets a caller that already read the Parquet (e.g. the integrity check)
        supply it for the row count, avoiding a second full read of the same file.
        """
        manifest = self.load_manifest()
        rec = manifest.get(key.slug)
        path = self.path_for(key)
        if rec is None or not path.is_file():
            return False
        if _sha256(path) != rec.get("checksum"):
            return False
        actual_rows = len(frame) if frame is not None else int(len(pd.read_parquet(path)))
        return actual_rows == rec.get("row_count")
