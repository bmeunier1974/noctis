# Changelog

All notable changes to Noctis are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The run is now a real, always-on entity with its own record.** Every `noctis run` mints a
  fresh run id (identity is *minted*, never derived from the config — two byte-identical configs
  are two runs) and gets its own tree, `workspace/runs/<run_id>/`, holding one self-describing
  `run.json` at `schema_version: 1`, `kind: "noctis.run"`, plus a `run.lock`. The record carries
  the run's identity/lifecycle, an append-only `segments[]` (one per process invocation, with
  start/stop stamps, duration, stop reason, argv and its own counters), the **engine identity**
  that produced it (declared version + per-component fingerprint + comparable key), and the
  events/errors streams. Three new modules keep the I/O boundary sharp:
  `reporting/run_store.py` (the only module that touches the run tree), `reporting/run_record.py`
  (a **pure** `build(artifacts) -> dict` — no I/O, no clock, no config; snapshot-tested against a
  committed golden record) and `reporting/schema.py` (the versioned, additive-only contract plus a
  pure `validate()`). The `--debug` QA tree now rides the run's own id, so one run has one
  identity.
  - **Durability**: written at each CLOSE and at segment close, atomically (temp file +
    `os.replace`), synchronously, on an injected clock. A writer failure logs exactly one warning,
    latches the writer off, and leaves the record honestly marked `complete: false` — a reporting
    artifact can never take down a multi-week run. Caps (`events` 2 000, `trades` 5 000) always
    write a truncation note with kept/total counts; segments are uncapped.
  - **Liveness**: a live `run.lock` (pid, hashed hostname, started, heartbeat) is a **hard
    refusal** — two engines writing one run would corrupt it. A stale lock (a dead pid on this
    host, or a heartbeat colder than a week) is stolen with a warning and a recorded event.
  - **Honesty**: a run killed mid-segment is marked `interrupted` on the **next open**, never
    guessed at write time.
- **Runs are findable: `noctis runs` / `noctis run-record`, and a derived `index.json`.**
  `runs [--all]` is the experiment board — id, label, status, segment count, cumulative runtime
  and the run's **comparable key** on one line, newest first — so experiments are found and
  compared without opening files. The default listing hides *noise* (a finished run under 60 s of
  cumulative runtime: a startup failure or a typo), always prints how many it hid, and never
  hides a still-`running` run or one whose record could not be read. `run-record <id>` prints one
  run's whole `run.json` on stdout (`| jq`), exiting non-zero when no run answers the id or that
  record is unreadable. Address resolution lives in the run store, shared with the `--resume` the
  next story adds.
  - `workspace/runs/index.json` is a **derived** roll-up of one entry per run — a listing page in
    one more `fetch()`. Refreshed after every record write (re-read from the file just written,
    so it can never advertise a record that is not on disk), regenerated from scratch by
    `noctis runs`, written atomically, and pinned by a test that a rebuild from the records alone
    reproduces the incrementally-maintained file **byte for byte**. It is derived, never
    authoritative: delete it whenever you like.
  - Every index entry carries `comparable_key` (`null` when unknown) beside the record's own, so
    a leaderboard partitions structurally instead of trusting a human to remember which runs are
    poolable. A run with no `run.json` yet, or an unreadable/foreign one, is *listed as such* with
    the reason where its key would be — one broken file can never take a listing down.
- **Engine identity** (`src/noctis/observability/engine_id.py`) and the `noctis engine` verb.
  A declared `ENGINE_VERSION` (a plain incrementing integer, decoupled from the package
  version) plus a **per-component** fingerprint over the committed files that decide behaviour
  — `gates`, `backtest`, `research`, `prompts`, `profiles`, `seeds`, `memory_seed`, `schema` —
  and the comparable key `(engine_version, gates_digest, backtest_digest, election_metric)`
  two runs must match on before their numbers may be pooled or ranked together. Only committed
  scaffold is hashed, so an operator's gitignored mandate never moves a digest and fingerprints
  stay portable between machines.
- **The engine fingerprint ratchet** — behavioural drift can no longer land silently. The
  committed `engine_fingerprint.json` records the current per-component digests (and a digest
  per allowlisted file) beside `ENGINE_VERSION`; `scripts/engine_fingerprint.py` recomputes and
  compares it in CI and pre-commit. Drift in an **arbiter** component (`gates`, `backtest` — what
  passes, what a number means) without an `ENGINE_VERSION` bump **fails**, naming the component
  and the files that moved; **searcher**-tier drift (`research`, `prompts`, `profiles`, `seeds`,
  `memory_seed`, `schema`) **warns and passes**, because improving the searcher must not
  invalidate an experiment whose arbiter held still. A stale record fails with the regeneration
  command in the message: `uv run python scripts/engine_fingerprint.py --write`. The check reads
  the one `ARBITER_COMPONENTS` constant the resume policy will read — two copies of that set
  would eventually disagree, silently.

### Changed

- **The mandate is the sole run input.** A mandate's `config:` block now overlays the whole
  run-shaping tier — the model seam, the spend ceilings, the search shape, the data window,
  housekeeping, and the seed `universe` — rather than the single `promotion.metric` knob of
  0.1.0. Every setting is classified exactly once by a deny-by-default classifier
  (`src/noctis/config/overlay.py`), with a completeness ratchet in the suite: the arena (safety
  mode, fill costs, promotion thresholds, holdout geometry, output paths, secrets) is refused
  **by name**, a refused/unknown/invalid key is now **fatal at startup** with its reason printed
  (it used to warn and be silently skipped), and two knobs are clamped to the disciplined
  direction only — `research.min_trials` may only be raised, `data.budget_usd` only lowered.
  For the overlaid subset the mandate applies *above* the environment; `noctis status` and the
  `run`/`research` kickoff now echo the active mandate and every applied override.

## [0.1.0] - 2026-07-13

First public release. Noctis is an autonomous, **paper-only** quantitative research system: it
researches strategies while the market is closed, trades champions on (live or replayed) bars
while it is open, and reports at the close — looping day after day. This release captures the
assembled state rather than promising it.

### Added

- **Phase-loop engine** — a market-clock-driven state machine (RESEARCH ↔ TRADING → CLOSE →
  RESEARCH, plus STOPPED) with a global time limit and a clean between-phases shutdown.
- **Agent research sessions** — with an LLM configured, RESEARCH is an agent session that
  authors one-file Python strategies and drives formulate → match → optimize → decide through a
  curated tool registry; a legacy proposer/Optuna loop runs without a key.
- **Strategy library + validation-on-write** — one reviewable `.py` per strategy;
  `write_strategy` validates each in a fresh subprocess (import + smoke replay + known-outcome
  scenario replay + signals/on_bar parity), so a broken file can never land.
- **Backtest pipeline** — a vectorbt-style pre-filter → walk-forward validation → a panel
  Scorecard, with no lookahead (decide on bar *t*, fill at bar *t+1*'s open).
- **Promotion gates with two-axis out-of-sample validation** — activity floor → overfit-gap
  guard → forward temporal holdout → symbol holdout → consistency → beat-the-weakest, over a
  pure decision function. Every champion is reproducible via `noctis backtest <name>`.
- **Paper-only two-gate safety** — real-money order paths are reachable only with both
  `mode: live` (config) and `ALLOW_LIVE=true` (env), and even then the live adapter is a
  refusing stub.
- **Operator mandates** — a committed `mandate/` steering surface; a mandate may bind exactly
  one knob (`promotion.metric`) and never loosens a gate.
- **Continuous paper account + forward record** — equity and open positions carry across
  sessions; catalog replay forms a rolling live-holdout of unseen bars.
- **Reproducible tooling** — uv-locked environments (`uv.lock`, `.python-version`, a PEP 735
  dev group) and a GitHub Actions CI pipeline (`uv sync --locked` → pytest / ruff / mypy /
  build) across Python 3.11 and 3.12.
- **Governance** — `SECURITY.md` (private disclosure; the paper-only gate as a security
  boundary), `CODE_OF_CONDUCT.md` (Contributor Covenant), an extended `CONTRIBUTING.md`,
  issue/PR templates, and Dependabot.
- **Documentation** — architecture, research, configuration, data, CLI, safety, development,
  and the validation methodology (`docs/validation.md`), plus runnable `examples/`.

[Unreleased]: https://github.com/bmeunier1974/agent-trader/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bmeunier1974/agent-trader/releases/tag/v0.1.0
