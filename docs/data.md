# Data

Two sources, two roles: **DataBento** backs the historical research lake (pay-as-you-go,
fetch-once), and **Yahoo Finance** (via `yfinance`) optionally feeds the live trading day for
free. No vendor data is ever committed to git — the lake is reproducible from the coverage
registry + manifests.

## The fetch-once lake

Historical data is cached to a local Parquet catalog (`workspace/data_lake/`) with a coverage
registry:

- 🔁 Every ingest is **diffed against the coverage registry** — re-requesting a covered range
  is a `$0` no-op; the nightly sync fetches **only the missing tail**.
- 🧮 A **cost preflight** pads the vendor estimate +20% and **refuses** ingests over
  `data.budget_usd` (default `$125`, the signup credit). `--dry-run` prices without spending.
- 🩺 An **integrity check** flags gaps/duplicates/manifest drift and repairs only what it
  flags.

```bash
python -m noctis data status               # tracked series in the coverage registry
python -m noctis data ingest AAPL --start 2024-01-01 --end 2024-12-31 [--dry-run]
python -m noctis data sync                 # tail-only incremental catalog sync
```

## Auto-backfill

When `data.auto_backfill` is on, `run` fetches missing history for any not-yet-ready universe
symbol over a `data.history_days` lookback window before entering the loop. The code default is
**off** (a bare run fetches nothing and reports "ingest history first" on an empty lake), but
the **shipped config template enables it** with `history_days: 720` — set it back to `false` if
you don't want `run` to ingest. It is coverage-diffed (already-cached ranges are `$0` no-ops)
and respects `data.budget_usd` — an over-budget backfill is refused cleanly (no fetch, no
crash) and the run continues. This initial backfill is the big one-time DataBento spend, so it
needs `DATABENTO_API_KEY`; without the key, `run` warns and skips the backfill.

## The live feed (opt-in)

The live day-loop feed is **free** Yahoo Finance data via
[yfinance](https://github.com/ranaroussi/yfinance) — no credentials, no per-request cost. Opt
in with `data.provider: yfinance` (and the `data` extra); TRADING then pulls the recent closed
1-min bars each poll and runs champions on them, still emitting **paper** orders only.

- Yahoo intraday is delayed ~15 min — fine for paper trading.
- The feed **self-throttles** its fetches, so a tight `live_feed.poll_interval_s` never hammers
  Yahoo.
- If a fetch fails or the data stops advancing, it **halts order emission** (rather than trade
  on stale prices) and resumes on recovery.
- At close, the live-built bars are **reconciled against the authoritative catalog** (after the
  T+1 sync) and drift over threshold is flagged in the report.

The default (`data.provider` unset to a live source) keeps TRADING on offline **catalog
replay** — see [architecture.md](architecture.md) for how replay forms a rolling live-holdout.

## Market-data policy

Vendor market data is **never redistributed through this repository**. No `.parquet`, database,
or other data blob is committed — only the coverage registry and manifests, which record *what*
was fetched, not the data itself (the lake lives inside the gitignored `workspace/`). Every user rebuilds the
lake **locally**, from their own vendor account, under that vendor's terms of use:

- **DataBento** (historical research lake) is pay-as-you-go under your own DataBento agreement;
  ingests run through the cost preflight above and never leave your machine.
- **Yahoo Finance** (optional live feed via `yfinance`) is subject to Yahoo's terms; it is used
  only to build paper bars locally and is likewise never redistributed here.

In short: the repository ships the *recipe* for the lake (registry + manifests), never the
*ingredients* (the vendor bars). This keeps the project free of any market-data licensing
encumbrance and reproducible by anyone with their own vendor access.
