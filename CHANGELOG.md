# Changelog

All notable changes to Noctis are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`docs/run-record.md` — the run record's contract, so a website can be built from the docs
  alone.** The epic ends with the artifact it exists to publish being documented rather than
  inferred: every section field by field with **when each field is `null`**, the additive-only
  versioning promise and the in-place upgrade path, the four conventions `validate()` enforces
  (self-describing `kind`, units in the name, UTC `Z`, explicit `null`), the caps and the measured
  384 KiB size budget, and the load-bearing rules stated plainly — evidence is never a gate, the
  safety gate is never rehydrated, the writer is never fatal except for lock contention, cumulative
  fields are derived, numbers are never pooled across `comparable_key`, retention never breaks
  resumability, output is workspace-only.
  - **The resume model in one place**: mint vs. resume, the four statuses (and why `completed` is
    the only one that refuses a resume *and* the only one that may be pruned — the same constant),
    `interrupted` observed on the next open rather than guessed, the liveness lock and its 7-day
    stale horizon, and everything a resume refuses.
  - **Engine identity, whole**: the eight-component table with its allowlisted files, the
    arbiter/searcher split declared once and read by both enforcers, and `comparable_key` as the
    tuple two runs must match on before their numbers may be pooled.
  - **New: `examples/run_record.json`** — a complete, schema-valid record built by the real builder
    (two nights plus a research-only segment, a promoted champion, a rejection that died at the
    symbol-holdout gate, an undecided draft, ten traded sessions, a priced spend block).
    `tests/test_docs_run_record.py` validates it and holds the page to the code: every key the
    schema names must appear on the page, and every number the page pins (schema version, engine
    version, the three caps, the size budget, the freezing-tier counts, the stale-lock horizon) is
    compared against the constant.
  - **What the record does *not* know is documented too**: the conversation research loop writes no
    session ledger (so its spend is `null`, not `$0`), coder-authoring completions are not
    token-metered, `assumptions.symbol_holdout.symbols` is always `null`, and there is no
    `strategies[].scorecard` — a candidate's backtest numbers reach the record only as each gate's
    `observed` value.
  - Docs swept for staleness against the delivered code: the frozen-key count in the drift
    paragraph (70 → **72**) and the mandate-overlay refusal count (52 → **53**) now match
    `rehydrate.py` and `overlay.py`; `noctis engine` sample output shows `ENGINE_VERSION` **2** with
    a real `schema` digest instead of version 1 and a missing input; `docs/architecture.md` and
    `docs/validation.md` no longer point at a `strategies[].scorecard` field that does not exist;
    four broken doc anchors fixed. `AGENTS.md`'s "Where state lives" now names `runs/index.json`
    and the `state_scope` the record publishes.

- **The run record's `assumptions` block, a versioned schema with an upgrade path, and
  `noctis run-record --validate`.** The arena a run's numbers were produced in, published as data
  so a website renders it as a table and two runs' blocks subtract.
  - **New: `reporting/assumptions.py` (pure).** The fill model (`next_bar_open`) and the
    no-lookahead rule, the fee and slippage the run was charged **per side** plus the round-trip
    cost, the walk-forward geometry and both holdouts, the exhaustion gate's `min_trials`, every
    promotion threshold, and the benchmark's rebalancing convention (none — weights are set at the
    first session mark and drift thereafter). Everything the run *configured* is read out of its
    own frozen `inputs`, so a promotion knob added tomorrow appears with no edit; the three values
    that are engine constants rather than settings are each held to the code that implements them
    by a test (a real fill lands on bar *t+1*'s open, `PipelineConfig.auto` sizes every split
    inside the stated bounds at both ends, the equal-weight basket really does drift).
  - **`paper_only` is a measurement.** It and `live_gate.real_orders_reachable` are derived from
    the safety gate's own resolved verdict (frozen as `inputs.execution_mode`), never written as
    constants: a test opens both live-money gates and watches the block flip. A run that froze no
    verdict reports `null` rather than claiming paper, and the validator refuses a record carrying
    `mode` or `allow_live` anywhere — the pair keeps exactly two independent sources.
  - **The schema is a promise, not an accident.** `schema.upgrade(record)` walks a record up one
    version at a time through a reviewable registry of steps, restamps it, and returns the sentence
    the run files as an event; the run store applies it on every open, so a record written under an
    older schema is **upgraded in place** by the next ordinary write and says so. Exercised against
    a synthetic version 2 — the real `SCHEMA_VERSION` stays 1, because a bump nothing needed would
    be a version number spent on a test.
  - **Two contract-wide conventions are now checked structurally**, over the whole document rather
    than section by section, so a section added later inherits both: every timestamp key
    (`*_utc`, `t`, `at`, `ts`) must be UTC ISO-8601 with a `Z`, and every dimensioned number must
    spell its unit the one canonical way (`_s`, not `_seconds`; `_bps`, not `_bp`). The verbatim
    quotations inside `inputs` — the operator's own resolved settings and mandate front-matter —
    are exempt by name: renaming a key there would make the record disagree with the file it quotes.
  - **`noctis run-record <address> --validate`** schema-checks a record instead of printing it,
    naming every problem at once and exiting non-zero when there is one.
  - Redaction and no-lookahead are re-pinned over the sections the epic has grown since: no
    credential's *value or name* survives anywhere in a record once `inputs.settings`' tier lists
    (where naming one is the point) are removed, and the fullest record this engine builds carries
    holdout **metrics** and holdout geometry while carrying nothing a bar could be reconstructed
    from.

- **The run record's realised performance: trades, the equity curve, a metrics module, a benchmark
  and the Deflated Sharpe Ratio.** The profitability picture, published beside the funnel — and
  kept structurally apart from the backtest numbers that decide promotions, so no website can
  present one as the other.
  - **Trades are enriched, and the per-day report is byte-identical.** `reporting/report.Trade`
    gains `ts`, `fees`, `slippage_bps` and `champion` — all optional and **absent** (not `null`)
    when unset, so every `<run>/reports/<date>.json` an operator already has is reproduced byte for
    byte and the Markdown report renders exactly the five columns it always did. The champion
    attribution comes from the same `assign_indices` call the forward P&L attribution makes, so
    "which champion made this trade" and "which champion earned this P&L" are one answer.
  - **New: `<run>/state/equity_curve.jsonl` and `broker.persistence.EquityLedger`.** At each CLOSE
    the engine appends one dated mark — the *account's* mark-to-market, plus that session's fills,
    orders and closing positions. Append-only, one mark per date with the last write winning (a
    re-run CLOSE supersedes rather than doubles a day), and never fatal: an unreadable line costs
    that mark and nothing else.
  - **New: `sessions[]` and `performance` on the run record.** The curve is **re-derived from the
    ledger at every write** and never carried in memory, so a run stopped and resumed three times
    publishes exactly the curve one long night would — `tests/test_run_resume.py`'s segmentation
    equivalence now compares the whole realised record, not just the counters. `performance` names
    itself `source: "paper_account"`; scorecards stay under `strategies[].scorecard`.
  - **New: `reporting/metrics.py` — a pure module, deliberately not an extension of
    `scorecard.py`.** CAGR, annualised volatility, Sortino, Calmar, drawdown depth *and* duration,
    recovery factor, profit factor, expectancy, payoff ratio, win/loss rates, exposure, turnover,
    monthly returns, skew, kurtosis, PSR and DSR — plus per-champion trade statistics over **closed
    round trips** (flat → position → flat, P&L as the cash the account gained, fees included, so no
    second copy of the broker's accounting can drift from it). No I/O, no clock, no configuration,
    enforced by AST; `scorecard.py` is untouched, and the promotion path still imports nothing from
    `noctis.reporting`.
  - **The Deflated Sharpe Ratio, deflated by the run's own cumulative trial count**, published
    beside `n_trials_used` (and `deflation_basis`, which names the variance assumption) so the
    correction is auditable from the record alone. PSR and DSR are pinned in the tests against
    their published closed forms (Bailey & López de Prado 2012/2014) *and* cross-checked against
    independent published results — Lo (2002) on the Sharpe ratio's standard error, and the exact
    expected maximum of two standard normals. More trials always deflate further.
  - **New benchmark: `equal_weight_universe_bh`** — equal-weight buy-and-hold over the symbols the
    run actually traded, priced from bars **already in the shared lake** over the run's own session
    window, with alpha, beta, information ratio, tracking error and correlation. **No vendor call
    and no new spend**: a name the lake does not hold leaves the block `null` with a note saying
    why. Only statistics are published — the price series never reaches the record.
  - **`traded: false` ⇒ `performance: null`**, still, and now enforced with the block populated: a
    research-only run (and a run whose account journaled no mark) reports nothing rather than
    zeros, so "researching" can never render as a flat 0% curve.
  - **`RECORD_SIZE_BUDGET_BYTES` is 384 KiB** (was 256 KiB): the same synthetic fortnight — 14
    segments, 66 candidates, ~3 000 trials — now also carries 14 traded sessions at 30 fills each
    and measures **285 899 bytes**, held by a test. The budget states measured reality; the caps
    beside it (`TRADE_CAP`) bound the pathological run.

- **Structured gate evidence, and the run record's strategies section: the rejections are now
  computable.** "47 of 66 candidates died at the symbol-holdout gate" is the sentence that makes
  these results credible where an equity curve does not, and until now it could not be computed
  from anything stored — a promotion rationale was prose.
  - **`decide()` returns `gates: tuple[GateResult, ...]`** — `(gate, passed, observed, threshold,
    note)` per gate, appended as each one evaluates. **No decision moved.** Same early returns,
    same order (activity floor → overfit gap → forward holdout → symbol holdout → consistency →
    beat the weakest), same outcomes; every existing caller ignores the new field. That is
    *proved*, not asserted: `tests/fixtures/promotion_decisions_golden.json` snapshots
    `(promote, rationale, demote_index)` for a 30-case corpus covering every branch and both sides
    of every threshold, captured from the promotion module **before** the change, and
    `tests/test_gate_evidence.py` replays it. Regenerating that golden means a judgment moved —
    which is an engine change, not a test fix.
  - **A rejection short-circuits, and the record says so.** `gates` holds the gates *reached* plus
    the one that failed; an absent gate means "never reached", never "passed". That shape is part
    of the record contract (`reporting/schema.py`), because a consumer counting deaths per gate
    would otherwise silently mis-attribute them. A gate that could not bite — switched off by a
    zero threshold, or handed a metric the scorecard never carried — is still recorded, with a
    note saying which: a funnel's denominators are honest only when every gate in the path is on
    the record.
  - **New: `strategies[]` on the run record — every candidate, not just the champions.** Name,
    outcome (`promoted` / `rejected` / `undecided`), tier, decision stamp, trials journaled, the
    prose rationale and the structured gate results behind it. Derived at every write from the
    run's **own** champion board, experiment journals and strategy tiers (epic D4), so three short
    segments report exactly what one long one does. The champion board now journals each
    decision's gates beside its rationale, which is the only durable trace a *rejected* candidate
    leaves.
  - **Champion sources are embedded in full; every other candidate is a path plus a content
    hash** (`source_path` relative to the run directory, `source_sha256`). A synthetic two-week
    run — 14 segments, 66 candidates, 3 champions, ~3 000 trials — weighs ~140 KB, against a
    stated `RECORD_SIZE_BUDGET_BYTES` of 256 KB that a test holds; embedding everything is 2.4×
    that. (The epic's ~40 KB planning estimate predates per-candidate gate evidence, which is the
    largest thing in the section and also the point of it.) The cost is stated rather than hidden:
    a rejected candidate's code is readable while the run's tree survives, and `run.state_pruned`
    says when it no longer does — what the record *embeds* survives a prune, as the retention
    suite now asserts directly.
  - **New: `noctis run --embed-all-sources`** (setting `embed_all_sources`, default off) archives
    a run whole. Frozen at creation like the compute cap beside it and refused on `--resume`: the
    record is rewritten whole at every write, so a flag passed on some nights and not others would
    make what the record contains depend on how it was last invoked. Refused to the mandate
    overlay by name — steering says what to look for, never what the evidence describing it
    contains.
  - **New cap: `STRATEGY_CAP` (500 candidates)**, with the usual honest note
    (`truncated.strategies` = kept/total). Champions are ordered first and are therefore never the
    entries a cap drops.
  - **Evidence, never a gate** (AGENTS.md rule 2). A test imports the whole promotion path in a
    fresh subprocess and asserts that **nothing under `noctis.reporting`** is reachable from it —
    deliberately the package, not one module, so it is meaningful today and strictly stronger the
    moment `reporting/metrics.py` (story #142) exists.

- **The run record's spend roll-up: attributable tokens, a versioned price table, efficiency
  numbers.** Token usage was already computed per session and then only *logged*, so cost per
  champion — the single most useful cross-run number this system can publish — was not recoverable
  from any artifact. It is now.
  - **New: `spend` on the run record.** The cumulative token split (input / output / cache-write /
    cache-read, plus the total behind it) broken out **by model, by stage and by segment**, so
    spend is attributable rather than one opaque figure. The axes are read straight off what the
    session ledgers already journal per episode; the record invents no taxonomy of its own.
  - **New: `research/pricing.py` — a pure, versioned `$/Mtok` table** keyed by model prefix
    (longest match wins), with all four rates stated separately because all four bill separately.
    `spend.pricing_table_version` travels into the record beside the numbers it produced.
    Overridable per model prefix through the new `research.pricing` config knob; an overridden
    table **cannot borrow the shipped version label** — it identifies itself as
    `<version>+custom.<digest>`, derived from the override, so a reader always knows whose prices
    these are.
  - **`null`, never zero, wherever a number is unknown.** A model the table does not carry costs
    `null`, and so does any total it belongs to (a partial sum presented as a total understates the
    bill while looking complete). A run with **no LLM key** journals no ledger and reports `null`
    spend rather than a `$0` bill it never earned. A ledger written before the split existed
    reports its token *total* with `null` fields. Every efficiency ratio with a zero or unknown
    denominator is `null` — a run that has crowned no champion has no cost per champion.
  - **New: `spend.efficiency`** — `usd_per_champion_estimate`, `usd_per_trial_estimate`,
    `trials_per_hour`, `research_hours_per_champion`: the currency two runs are compared on, over
    the run's own champion board, its own journaled trials and its own measured research hours.
  - **Every cost field says `estimate`**, in the record's field names and in the CLI's session
    line, and `schema.validate` now *enforces* it: a dollar-bearing key anywhere in the block that
    does not name itself an estimate is a schema violation. These are list prices, not receipts.
  - **`ResearchSummary` gained `usage` and `usd_estimate`, on both research paths.** The
    conversation loop fills the split from the per-round usage it already accumulates; the episodic
    driver **sums the ledger it already wrote** rather than re-instrumenting — its `episode` lines
    now carry the four-field split beside the token total they always had (a tolerant, additive
    ledger extension).
  - **Derived at write time, never incremented.** Spend is re-read from the run's own session
    ledgers (and champions from its own board) at every write, exactly as the trial count is read
    from its experiment journals — so three short segments produce the same totals as one long one,
    a rewrite after a crash cannot double-count, and the segmentation-equivalence test now covers
    the whole spend block. Pricing uses the run's **frozen** config, so editing prices tomorrow
    cannot restate what last night cost.
  - **Evidence, never a gate** (AGENTS.md rule 2): nothing here is read by a promotion gate, a
    research budget or the exhaustion floor, and the new knob is refused by the mandate overlay by
    name — an experiment that could restate its own bill would make every cross-run cost comparison
    a claim it made about itself.
- **Per-segment environment capture, and the run record's frozen-inputs provenance block
  completed.** A run record now says *what machine* produced each night's numbers, and *what
  inputs* the run was started with.
  - **New: `segments[].environment`, and a derived `environment_latest`.** Hardware (CPU model,
    physical/logical cores, max frequency, total RAM, free disk), OS (system, release, arch,
    container), python and noctis versions, git state (commit, branch, dirty, describe), the
    `uv.lock` digest, the optional extras present, and the seams that degraded. It is recorded
    **per segment, never per run**: a stopped-and-resumed experiment may migrate machines, and
    research throughput is CPU-bound (the sweep fork pool, the walk-forward splits) — so one
    night's trials-per-hour and USD-per-champion must never be attributed to another night's
    cores. `environment_latest` is derived from the segments, so it can never disagree with them.
  - **`psutil` is an optional extra (`hardware`), never a core dependency.** Without it the
    stdlib subset answers what it can (logical cores, free disk, `/proc/meminfo`) and the rest is
    explicit `null` with `hardware` named in `degraded_seams` — the same seam discipline
    `vectorbt` and `databento` already follow. Git capture degrades to `null` outside a
    repository, and so does the lockfile digest. **Every absent value is an explicit `null` with
    the missing capability named**, never a silently dropped key.
  - **One notion of "degraded seam".** `extras_present` is keyed by the optional-extra names
    `noctis setup` already probes for (`llm`, `data`, `research`, `engine`, `hardware`), so a
    missing extra and a degraded seam are one thing with one list behind them — and the remedy the
    record implies (`uv sync --extra <name>`) is one an operator can type.
  - **The hostname is hashed, never stored.** `sha256(hostname)[:12]`, the same digest `run.lock`
    has written since the run store landed and through the same function, so two segments on one
    machine are provably the same host without a machine name ever being published.
  - **New: `inputs.models` and `inputs.data`.** Which model researches, authors, escalates and
    ideates, which research loop was resolved, the declared context window and cost profile; and
    the data provider, dataset and (shared, workspace-level) lake directory. Both are derived
    *views* over values `inputs.settings.resolved` already froze — stated once, resolved, so a
    consumer never reconstructs a fallback chain. A model name is public; the API key that
    authenticates it is secret tier and reaches no record (AGENTS.md rule 6).
  - All probes are **injected**, wired once in the composition root, so the capture module reads
    no hardware, shells out to no `git` and imports no optional package — and the test suite needs
    none of them.
- **`noctis run-prune <address> [--dry-run]` — opt-in retention for completed runs.** A run's
  `state/`, `strategies/` and `reports/` directories are the megabytes; its `run.json` is
  kilobytes and *is* the long-term progress history. This reclaims the first three and never the
  record — so a pruned run still lists in `noctis runs`, still prints in full through
  `run-record`, and keeps every number it accumulated (including its trial count, which is read
  off the journals *before* anything is deleted).
  - **Completed runs only, and the refusal is the point.** The pruned directories are exactly what
    a resume reads back, so pruning a `stopped`, `interrupted` or `running` run would silently
    destroy its resumability — the one thing the run record promises — and all three are refused
    with that reason. "Prunable" is not a second rule but the same constant as "terminal"
    (`PRUNABLE_STATUSES is TERMINAL_STATUSES`), so a pruned-then-resumed run is unreachable rather
    than merely unlikely. A run another engine is live on is refused too, whatever its record says.
  - **Opt-in, never a schedule.** No config knob, no startup sweep, no automatic retention: a byte
    is removed only when an operator names one run. The default keeps everything, forever.
  - **`--dry-run` reports the directories and the bytes and removes nothing** — same measurement,
    one step short of the removal.
  - **New: `run.state_pruned`** (`false` on every run until one is pruned), so a reader knows the
    record's path-plus-hash references into those directories no longer resolve. Everything the
    record *carries* is untouched: pruning removes three directories and rewrites one flag.
  - The blast radius is a reviewable constant — three fixed child names of one run directory,
    each removed only when it really is a directory and never through a symlink, and never at all
    unless the address resolved to a tree whose own `run.json` says `completed`.
- **`noctis research --resume <address>` — a research-only night belongs to the same run.** A
  standalone session is no longer an unrecorded write into the reserved `legacy` run: a bare
  `noctis research` now **mints its own run** (record, tree and lock, like `noctis run`), and
  `--resume` appends a **research-only segment** to an existing one — same lock, same frozen
  config, same run-scoped state, strategy tiers and per-run memory, same record.
  - **One resume, two verbs.** The address forms (`<id>`, `latest`, a `run.json` path, `@label`),
    the refusals (unknown address, ambiguous label, live lock, `completed` run) and the frozen-tier
    rules (`--mandate`/`--directive`/`--metric` refused with a reason) all come from the same
    resolver and the same composition root as `run --resume`, so the two cannot drift apart.
  - **"Research-only" is derived, not a second flag.** A segment already records the `command` it
    was invoked as, and `research` is a verb that cannot trade — so the record answers "which
    nights were research-only?" with what it already had to carry. The segment keeps its own
    start/stop stamps, duration, `stopped_reason` (the session's own), argv and counters
    (`sessions` / `research_iterations` / `research_promotions`).
  - **New: `run.cumulative_trials`, read from the journals.** The run's trial count is counted at
    write time off its own `state/experiments/*.jsonl` — the very lines the exhaustion gate counts
    — never a counter carried across a restart, so it is cumulative across every segment including
    the research-only ones, and it is the multiple-testing count a deflated Sharpe will want.
    `null` when a run has journaled nothing.
  - **New: `run.traded` and an explicit `performance: null`.** A run may research for weeks and
    never trade; that is a **first-class** shape, so it reports `traded: false` and a `null`
    performance block rather than zeros, and a consumer renders "researching" instead of a fake
    flat 0% equity curve. The schema validator enforces the pairing (`traded: false` ⇒
    `performance: null`), so the degenerate zeros cannot arrive by accident when the performance
    block itself lands.
- **A run-level compute cap (`noctis run --run-limit-hours`) and an explicit finish
  (`--finish`).** `time_limit_hours` bounds one *process*; this bounds the whole *run*, so
  "research this mandate for 100 hours, then stop" is expressible — and two runs become comparable
  on **equal compute**, which matters as much as equal config.
  - **Frozen at creation.** `run_limit_hours` is an ordinary frozen setting (flag, `config.yaml`,
    or a mandate's `config:` block — it joins the overlay's spend-ceiling tier), pinned into
    `inputs.settings.resolved` when the run is minted. Editing `config.yaml` between segments does
    not move it, and `--run-limit-hours` **with** `--resume` is refused with a reason: a cap that
    could be raised each morning would bound nothing at all.
  - **It stops through the shutdown path that already exists.** The state machine gained the cap
    beside `time_limit_hours` and one `limit_hit()` both are asked through, so the run stops
    cleanly *between phases* exactly like the per-process limit and `SIGINT` — no second shutdown
    route — and the between-phase waits clamp to whichever deadline is earlier. The segment closes
    with `stopped_reason: run_limit`, and the CLI says the run is now completed.
  - **The breach is derived, not latched.** `run.status` reads `completed` whenever the record's
    own `cumulative_runtime_s` (summed from `segments[]`) crosses the frozen cap, stamped
    `completed_utc` at the segment close that crossed it. So a run killed at the instant it
    crossed is still terminal when it is next read, and every later resume is refused — with a
    message naming the cap and the runtime spent. A segment ending *below* the cap leaves the run
    `stopped` and resumable, unchanged.
  - **`--finish` seals a run deliberately**, and runs **no segment**: no engine starts, no segment
    is opened, and the liveness lock is read only far enough to refuse a run another process is
    working. On an already-`completed` run it is a documented no-op that keeps the original seal
    stamp — terminal means terminal.
  - **New: cumulative research / trading seconds, derived like everything else.** Each segment now
    records the seconds its process spent *working* in each phase (`segments[].phase_seconds`;
    waiting out a weekend is not research), and the record sums them across `segments[]` into
    `run.cumulative_research_s` / `cumulative_trading_s` at every write. Never incremented in
    memory: three short nights total exactly what one long one would. A segment that measured
    nothing reports `null` rather than a `0` it never observed.
  - Visible where runs are compared: `run_limit_hours` on the record and on every `index.json`
    entry, and `noctis runs` shows the budget beside the runtime spent (`2d12h/100h`).
- **Engine-change resume policy, and `noctis run --resume … --allow-engine-upgrade`.** A run
  resumed after a `git pull` may find a different engine, and the policy splits on **who changed:
  the judge, or the searcher** — the same arbiter/searcher line the CI ratchet enforces, read
  through the one classifier over the one `ARBITER_COMPONENTS` constant (a test binds both
  enforcers to it, and to each other, component by component).
  - **Arbiter drift (`gates`, `backtest`) refuses the resume**, naming the component and both
    digests, before a segment is opened or a lock is taken. Champions crowned under two sets of
    gates must never accumulate inside one experiment — inside a single run that is worse than
    across two (AGENTS.md rule 2).
  - **`--allow-engine-upgrade` overrides that refusal, and is never invisible**: it bumps
    `engine.engine_epoch`, appends an `engine.engine_changes` entry naming every component that
    moved with both digests and the **segment** it happened in, re-freezes the run onto the new
    engine (so its comparable key honestly follows it), and flags the run `mixed_engine` for good.
    With no arbiter drift it is a documented **no-op** — the epoch never moves for nothing.
  - **Searcher-tier drift (`research`, `prompts`, `profiles`, `seeds`, `memory_seed`, `schema`)
    warns, records and proceeds**: an event on the record naming the component, both digests and
    the files to go and look at. Improving how candidates are *found* must not invalidate an
    experiment whose arbiter held still.
  - **No drift is silent** — nothing printed, nothing recorded. A policy that says something every
    time is one operators learn to skip.
  - `mixed_engine` is visible in the record, in `index.json` and in the `noctis runs` board (beside
    the comparable key, which alone would over-promise for a run that ran two engines).
  - **The engine identity is now frozen at run creation** and carried forward verbatim, exactly
    like the frozen config: the run-level `engine` section is what the run was created under (and
    what every resume is compared against), while each **segment** records the engine that actually
    produced it. Previously every write restamped the run with whatever engine was running, so
    there was nothing stable to compare a resume against.
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

### Fixed

- **The engine ratchet's promise now actually holds: `--write` refuses to record an undeclared
  arbiter move.** The ratchet compares the committed record against the freshly computed tree, so
  the only state it could see was *disagreement between the two* — and regenerating erased it. "Edit
  `promotion.py`, run `--write`, commit" therefore passed CI with no `ENGINE_VERSION` bump, which is
  exactly the one change that invalidates every stored champion comparison. Worse, `--write` is the
  one-command fix the failure message itself recommends and it rewrites *every* component at once, so
  a PR that also moved a searcher component (the common case) was told to run the command that
  laundered the arbiter move. `--write` now evaluates the check first and, on arbiter drift while the
  recorded and computed `ENGINE_VERSION` agree, **writes nothing and exits 1**, printing the
  bump-or-restore guidance plus its refusal — leaving the tree checkable, and failing. An arbiter
  move must arrive *declared*: bump, then regenerate.
  - Every legitimate regeneration stays a single command: searcher-only drift, an arbiter move whose
    bump *is* in the tree (the record had simply not caught up), no drift at all, and a missing or
    unreadable record — there is nothing to compare against, and that is how the baseline is created.
  - The refusal is a decision in the `--write` I/O path (`regenerate`, reading one new derived
    property off the existing verdict), so `compare_records` stays pure, `build_record` stays
    independent of the previous record, and `--check`'s four-case rule and report text are unchanged.
    The arbiter/searcher line is still the one `ARBITER_COMPONENTS` constant, read through `tier_of`.

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
