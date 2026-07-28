# Changelog

All notable changes to Noctis are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Config drift: `noctis run --resume … --show-config-drift` and `--rebase-config`.** A run's
  configuration is frozen at creation and drift is normal — frozen wins, silently. These are the
  two things an operator needs on top of that: *see* what the current `config.yaml` and `mandate/`
  would change, and *adopt* them deliberately.
  - **`--show-config-drift`** prints the diff and exits. It is an inspection: it opens no segment,
    takes no lock and rewrites nothing, because looking first must never itself be a decision.
    It compares the **frozen** keys and the **resolved mandate text** (so a rewritten profile behind
    an unchanged selector shows up, and the same bytes behind a renamed one do not). The **live**
    tier — paths, secrets, per-process budgets — is never reported: it is this process's by design.
  - **`--rebase-config`** adopts the current files for the rest of the run: it re-freezes them,
    bumps `inputs.config_epoch`, and appends a before/after entry to the new `inputs.config_changes`
    list naming the **segment** it happened in. A mid-run config change is never silent — a record
    whose config changed must say so *and say where*. After a rebase the new values are the run's
    own, so the next resume restores those.
  - **With no drift, `--rebase-config` is a documented no-op**: the epoch does not move and no
    entry is written. A cosmetic bump would mark the run mixed-config forever.
  - **`mode` and `allow_live` are never rebasable under any flag.** `mode` is not in the frozen
    settings at all (the record keeps only the gate's verdict), so the concrete attempt — edit
    `mode`, open `ALLOW_LIVE`, ask for the current files — is refused by the mode check that runs
    before any rebase, with a message stating that no flag lifts it (AGENTS.md rule 1).
  - `config_drift()` and `rebase_inputs()` are **pure** functions beside the freezing tiers in
    `config/rehydrate.py` — `(record, settings) -> value`, no I/O and no clock — so the diff an
    operator is shown and the entry the record stores are built from the one value.
- **Resume addressing: `--resume latest`, a `run.json` path, `--resume @label`, and `--label`.**
  `--resume` now takes four address forms, resolved in one place (shared with `run-record`) in a
  fixed order, so one string always names one run whatever a workspace happens to contain: a
  **path** (anything with a separator, or `run.json` — the record file you are looking at, wherever
  it lives), **`@label`** (the label first, then the same name as an id, so an id pasted with a
  leading `@` still resolves), the reserved word **`latest`**, and a **run id**. A bare address is
  *always* the id — a run merely labelled like one can never shadow it — and `latest` is a reserved
  word rather than a lookup, so a run named or labelled `latest` never captures it (address those
  by path, or as `@latest`). "Most recently active" is read off the record's own stamps, never a
  filesystem mtime, ties break on the id, and `latest` skips `completed` runs (they refuse resume
  anyway) and unreadable ones; with nothing resumable left it fails saying what it found.
  - **`--label nightly-momo`** attaches a human alias, stored in the **record** and derived from
    there into `index.json`, listed by `noctis runs`. Also accepted with `--resume`, where it
    renames the run it addressed — a nickname decides nothing, unlike the frozen config a resume
    refuses to move.
  - Labels are **convenience only: the id is the identity.** A label may be reassigned; both runs
    then keep their own ids, records and history, and `@label` **refuses, naming both candidate
    ids**, rather than silently picking one — an alias that chose between two runs would eventually
    append a night's work to the wrong record. Everything an aliased open writes (the lock, the
    record, the echoed `Resumed run:` line, every refusal message) names the run's **id**, never
    the address it was reached by.
- **A run is stoppable and resumable: `noctis run --resume <run_id>`.** It opens an existing run,
  rehydrates that run's settings from its record, appends a segment, and keeps accumulating
  research hours, trials, champions and P&L into the same `run.json`. The **run**, not the
  process, is now the unit progress is tracked on — stop the engine each morning and resume it
  each night without losing a multi-week experiment. A resumed run reads its own run-scoped state
  (champions, paper account, memory, strategy tiers, reports) and shares the workspace-level data
  lake, exactly as its earlier segments did.
  - **Config is frozen at run creation**, in the record's new `inputs` section, and restored on
    every later segment by a new **pure** module, `config/rehydrate.py`
    (`(record, live_settings) -> Settings`, no I/O). Three tiers, each leaf classified exactly
    once and ratcheted by the suite: **frozen** (69 — everything that decides what the accumulated
    results *mean*: `research.*`, `promotion.*`, `backtest.*`, `trading.*`, `risk.*`, `universe`,
    `session.*`, `champion_count`, the data provider/dataset, the research time budget), **live**
    (17 — the three secrets, every path/workspace knob so a run can resume on a machine with
    different absolute paths, and the per-process budgets `time_limit_hours`, `data.budget_usd`,
    `qa.keep_last_runs`, `observability.heartbeat_polls`) and **refused** (2). Editing
    `config.yaml` between segments does not move a frozen key or the run's frozen digest; drift is
    normal and silently fine, and frozen wins. Two of the three tiers are *derived* from
    `config/overlay.py`'s own refusal table rather than re-listed, so classifying a new knob there
    puts it in the right freezing tier with no second edit — and frozen, being the complement,
    is what a knob added tomorrow defaults to.
  - **The mandate is frozen as resolved text** — body, digest, summary, symbols, references and
    the overlay it applied — not as a selector, so editing `mandate/profiles/aggressive.md`
    tonight cannot retroactively change what a running experiment was told to do. Consequently
    `--mandate` / `--directive` / `--metric` are **refused with a reason** on a resume rather than
    silently ignored.
  - **The safety gate is never rehydrated** (AGENTS.md rule 1). `mode` and `allow_live` are never
    written to a record (the schema validator refuses one that carries either) and never restored;
    `resolve_execution_mode` re-resolves from `config.yaml` + `ALLOW_LIVE` at every process start,
    so `mode: live` without `ALLOW_LIVE` is the same hard startup error on a resume as on a first
    start. The record carries the gate's *verdict* for the run as evidence, and a resume whose
    freshly resolved mode disagrees with it is a **hard error** — a paper run's results can never
    acquire live segments.
  - **Derived, never incremented**, proved by test: one three-hour segment and three one-hour
    segments over the same work leave records with **identical** derived totals. A crash between
    phases marks the segment `interrupted` on the next open, contributes no runtime (an unclosed
    segment has no honest duration), never double-counts, and a third segment resumes cleanly.
  - Refusals, all before any work starts: an id that names no run, a `completed` run (terminal by
    design — a published result never quietly gains segments), and a mode that disagrees.
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

- **A run owns its state; the data lake stays shared.** `workspace/state/` used to be shared by
  every invocation in the workspace, so two runs with different mandates crowned champions onto
  one board and traded one paper account and neither run's numbers meant anything. Champions, the
  paper account, the forward ledger, specs, sessions, experiment journals, the strategy
  `__tmp`/`champions` tiers, agent memory and the per-day close reports now live under
  `workspace/runs/<run_id>/`, beside that run's `run.json`. `workspace/data_lake/` is deliberately
  **not** run-scoped — vendor data is expensive, reproducible and run-neutral, so every run reads
  and writes the one lake.
  - The mechanism is the one that already existed: `state_dir` / `reports_dir` / `qa_dir` /
    `memory_path` derive from a new **`run_dir`** knob (refused by the mandate overlay, like every
    other path) instead of from `workspace_dir`, and `bootstrap.open_run_store` rebinds `run_dir`
    to the run it just minted — one settings change plus one composition-root change, no path
    arithmetic in any command body. An explicitly configured path still wins over the derivation.
  - `run_dir` defaults to the reserved `workspace/runs/legacy/` run: what an invocation that never
    opened a run reads (`status`, `champions`, `account`, `report`, `backtest`, a bare `research`),
    and the run `noctis migrate` adopts existing state into — so an upgrading operator finds their
    history exactly where those commands already look.
  - **Per-run agent memory**: each run's `memory/MEMORY.md` is seeded from the committed
    `MEMORY.seed.md` at run creation (before the store constructs, as before), so one run's
    lessons never leak into another's trajectory and the seed itself is never mutated. The
    committed `strategies/` seeds stay read-only input for every run and the three-tier discovery
    contract is unchanged; only the two writable tiers moved under the run.
- **`noctis migrate` adopts pre-run-scoped state into the reserved `legacy` run.** One command
  still covers both legacy generations — the pre-workspace artifacts beside `config.yaml` and now
  a pre-run-scoped `workspace/state|reports|memory|qa|strategies` — with one plan, one `--dry-run`
  and one conflict refusal (a destination that already exists, or that two legacy copies claim, is
  refused with the reason rather than resolved by guessing). The adopted run gets a real `run.json`
  with **zero segments** (no process ever worked it: inventing a segment would fabricate a night)
  and an event recording where its contents came from, so it lists in `noctis runs` and is
  resumable later. Running it twice is a no-op. The startup guard now covers both generations in
  one message with one instruction: abandoned pre-workspace data still **refuses** (`status`
  warns), while un-adopted workspace state **warns** — that state is not abandoned, only unclaimed,
  and the run starting beside it is a new run with its own board, which is the point.
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
