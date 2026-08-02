# Configuration

Configuration lives in a **local, gitignored `config.yaml`** — created (and edited) by
`noctis setup`, or copied from the committed template `config.example.yaml` by `noctis
init`. A missing `config.yaml` is fine: every knob has a safe built-in default, which is
why the template stays short — it lists only the knobs operators actually touch, and this
page is the full reference. Secrets live in `.env` (same pattern: `.env.example` → `.env`).
Both resolve into typed settings (`src/noctis/config/settings.py`). **Environment
variables override `config.yaml`**, and `NOCTIS_CONFIG=/path/to/config.yaml` points at an
alternate file. The active mandate then overlays the run-shaping subset on top of all of it —
see [The mandate overlay](#the-mandate-overlay) for the full precedence chain.

## The knobs

| Key | What it controls |
|---|---|
| `mode` | `paper` (default) or `live` — see [safety.md](safety.md) for the double gate |
| `universe` | Seed symbol list; the *effective* universe grows as the agent discovers symbols ([research.md](research.md)) |
| `session` | Exchange calendar + timezone |
| `research_time_budget_minutes` | Wall-clock cap on a research session |
| `research.mode` | `agent` (default) or `legacy` |
| `research.model`, `research.base_url` | LiteLLM `provider/model` string; base URL for local/self-hosted backends |
| `research.mandate` | Mandate selector: a profile name, `MANDATE`, `auto`, or `null` — code default `null` (unconstrained); **shipped config ships `auto`** (agent picks a profile per session) |
| `research.min_trials` | Exhaustion floor — verdict tools refuse before this many journaled trials |
| `research.draft_ttl_hours` | Working-tier housekeeping: undecided (`draft`/`candidate`) `__tmp/` drafts older than this many hours are swept into `__tmp/archive/` on each research-session start (default `48.0`; `null`/`0` disables). Moves bytes verbatim — never fabricates a verdict or touches a gate ([research.md](research.md)) |
| `research.max_iterations`, `max_backtests`, `sweep_trials`, `web_search` | Agent session budgets |
| `research.agent.coder_model`, `max_author_calls` | Dedicated authoring model + its Class-B budget: coder completions/session (retries included); exhausted → brief authoring refused, hand-written `source` stays open |
| `research.agent.coder_thinking` | `on` (default) / `off` — the coder reasons through scenario-window/warmup arithmetic (deliberate, budgeted by `max_author_calls`); separate from the driver `thinking` dial |
| `research.agent.coder_fallback_model`, `max_escalations`, `coder_fallback_thinking` | Paid escalation coder (#72): a local authoring job that spends its validator retries escalates the same brief, bounded per session by `max_escalations` (`0` = default = never). The escalated coder's thinking dial defaults `off` (#98) |
| `research.agent.coder_max_tokens` | The coder's output-token ceiling — the *file's* budget: `null` (default) defers to the built-in `16000`; a number resizes it for a different coder backend. A thinking coder client gets a thinking allowance added on top (#98). A compat/sizing lever, **not** a cost budget (unused headroom is never billed); inert without a `coder_model` |
| `research.agent.coder_temperature`, `coder_seed` | The coder's sampling dials: `null` (default) sends nothing (today's request, byte for byte); a value is forwarded **only where the provider supports it** — the local/OpenAI-compatible seam — and is a clean no-op elsewhere (Anthropic has no `seed` parameter and rejects a temperature beside the pinned thinking dial). Neither buys determinism; reps + paired stats are the variance defence |
| `research.agent.coder_retries` | Private validator re-prompts per authoring job: `null` (default) defers to the engine's built-in `2` (initial + 2 ≤ 3 completions); a number pins it. Every attempt is a coder completion billed against `max_author_calls` |
| `research.cost_profile` | `full` / `balanced` / `economy` — resource ceilings only, never quality gates |
| `research.pricing` | `$/Mtok` price overrides for the run record's **spend estimate**, keyed by model prefix (see **Pricing the spend estimate** below). Pure accounting: it changes what a run is *reported* to have cost, never what it does |
| `research.agent.thinking` | `off` (default) / `on` — opt a **watch** session into provider-native reasoning; costs output tokens (see below) |
| `research.agent.max_tokens`, `context_window` | Small-context-backend compatibility levers (see **Local backends** below) — not cost budgets. A declared `context_window` ≤ 32,768 also flips `research.agent.loop: auto` to the episodic driver (#76) |
| `research.agent.sweep_workers` | Parallel workers for sweep trials + panel symbols (`1` = sequential) |
| `research.fit_set_size`, `symbol_holdout_size` | Panel geometry: fit set + symbol holdout sizes |
| `research.focus_size` | Cap on symbols enumerated into each session's prompt — a prompt-size lever, never the trading roster |
| `research.tuning_dispersion_penalty` | Penalizes parameter sets whose panel scores are dispersed |
| `risk` | Trading-loop risk limits (incl. the daily loss limit) |
| `trading.max_catchup_sessions` | Cap on missed replay sessions a restart catches up (newest kept) |
| `trading.min_order_notional`, `rebalance_band_pct` | Rebalance dead-band: skip immaterial same-direction re-trues; opens/exits/flips always execute |
| `trading.execution` | `auto` (derive from `data.provider`) / `replay` / `live` — selects the TRADING driver only, never real-order reachability |
| `data.provider` | Historical source *and* TRADING-phase feed — `yfinance` opts in to the live feed ([data.md](data.md)) |
| `data.budget_usd` | Data spend cap enforced by the cost preflight (default `$125`) |
| `data.dataset`, `data.lake_dir` | Vendor dataset; lake location |
| `data.auto_backfill`, `data.history_days` | Pre-loop backfill of missing history (code default off; **shipped config enables it**, `history_days: 720`). `history_days` is also the lookback the episodic driver's mandate-symbol preflight fetches over ([research.md](research.md)) |
| `live_feed.poll_interval_s` | Live-feed poll pacing (the feed self-throttles regardless) |
| `observability.heartbeat_polls` | `-vv` live-trading heartbeat cadence in polls (0 disables; default `60` ≈ every ~2 min) |
| `qa.keep_last_runs` | `--debug` QA-report retention: prune-on-start keeps the newest N run folders under `qa_dir` (default `20`; `0` keeps none). Housekeeping only — no decision path reads it. See [development.md → Reading a QA report](development.md#reading-a-qa-report) |
| `promotion.metric` | `sharpe` \| `sortino` \| `total_return` — the election metric every candidate is scored, ranked and promoted on; changing it reinterprets every threshold below in the new metric's units. **Normally set in the active mandate's `config:` block** ([The mandate overlay](#the-mandate-overlay)) — it is the one `promotion.*` knob a mandate may bind. The value in `config.yaml` stays the **base**: what an un-mandated run scores on, and what an `auto` session scores on by contract ([below](#researchmandate-auto-makes-a-profiles-config-block-inert)) |
| `promotion.max_gap`, `min_test_metric`, `min_test_activity` | Overfit-gap guard, OOS bar, and the almost-never-trades activity floor |
| `promotion.min_holdout_metric`, `min_symbol_holdout_metric`, `min_symbol_consistency` | The out-of-sample promotion gates |
| `promotion.annualization_cap`, `max_period_ratio`, `max_reverse_gap`, `max_test_metric` | Metric-robustness caps + degeneracy backstops (sub-daily Sharpe inflation, too-good-to-be-true OOS) |
| `backtest.fee_bps`, `backtest.slippage_bps` | Simulated fill costs **per side** (default `1.0`/`1.0` — a 4bp round trip). One value threaded to the pre-filter, validation, the agent's cost hint, and paper fills. Enforced minimum `1.0` each — see **Fill costs** below |
| `ideation` | The legacy StrategySpec path |
| `champion_count` | Champion board size — and since the `family_slot` gate allows **one slot per strategy family**, it is also the number of distinct families the board can hold ([validation.md](validation.md)) |
| `time_limit_hours` | Global stop from any phase — bounds **one process** (how long tonight lasts); the run stays resumable |
| `run_limit_hours` | Compute cap on the **whole run**, in hours of cumulative runtime across every stop/resume (`--run-limit-hours`, frozen at creation). At the cap the loop stops between phases and the run is marked `completed` — terminal, so it refuses resume. `null` = uncapped. See [cli.md](cli.md#bounding-a-run----run-limit-hours-and---finish) |
| `embed_all_sources` | Embed **every** candidate's strategy source in the run record, not just the champions' (`--embed-all-sources`, frozen at creation). Default `false`: champions are embedded in full, every other candidate is a run-relative path plus a content hash. See [cli.md](cli.md#archiving-a-run-whole----embed-all-sources) |
| `workspace_dir` | **The one output root** (default `workspace/`; env `NOCTIS_WORKSPACE`) — every path below derives from it when not set |
| `runs_dir`, `data.lake_dir` | The workspace-level pair: the run tree (`workspace/runs`) and the data lake (`workspace/data_lake`), which is **shared by every run** |
| `run_dir` | **The one run root** (default `workspace/runs/legacy`, the reserved run an invocation that never opened a run reads). `noctis run` / `noctis research` rebind it to the run they mint or resume |
| `state_dir`, `reports_dir`, `memory_path`, `qa_dir` | Per-artifact overrides; each defaults to its **run**-derived location (`<run_dir>/state`, `<run_dir>/reports`, `<run_dir>/memory/MEMORY.md`, `<run_dir>/qa`) |
| `strategies_dir`, `mandate_dir` | The committed input surfaces: the seed strategy library and the mandate scaffold |

## The workspace

Everything the engine writes lands under the single gitignored `workspace_dir` — one gitignore
line, one thing to back up or wipe — and inside it **a run owns its state**:

```text
workspace/
  data_lake/            # SHARED by every run: vendor data is expensive and run-neutral
  runs/<run_id>/        # run.json + run.lock, and everything that run produced:
    state/  strategies/{__tmp,champions}/  memory/MEMORY.md  reports/  qa/
```

So two runs in one workspace cannot crown champions onto one board or trade one paper account.
`state_dir`, `reports_dir`, `qa_dir` and `memory_path` derive from `run_dir`; `noctis run`
rebinds `run_dir` to the run it mints, and unset it is the reserved `runs/legacy/` run that the
read-only commands (`status`, `champions`, `account`, `report`, `backtest`) and a bare
`research` read. Setting the env var `NOCTIS_WORKSPACE` relocates all of it at once (useful when
running the CLI from outside the repo); an explicit per-artifact knob is an absolute override and
survives the rebinding.

`noctis init` creates the workspace alongside the local config; `noctis migrate` moves a
pre-workspace layout (`state/`, `data_lake/`, `reports/`, root `MEMORY.md`,
`strategies/__tmp|champions`) into it **and** adopts a pre-run-scoped `workspace/state|reports|
memory|qa|strategies` into the reserved `legacy` run, which then has a real `run.json` and is
resumable. Every state-touching command refuses to run beside abandoned pre-workspace data until
it has (`status` only warns), so a stale layout can never silently present an empty champion
board; un-adopted workspace state only **warns**, because that state is not abandoned — it is
sitting in the same workspace waiting to be claimed by a run.

## Config freezing — what a resumed run reads

A run outlives the process that started it: `noctis run --resume <run_id>`
([cli.md](cli.md#resuming-a-run----resume-address)) appends a segment and keeps accumulating into
the same record. That only *means* something if the configuration held still in between — so a
run's config is **frozen at creation**, stored in its own `run.json`, and restored on every later
segment. Editing `config.yaml` or a mandate profile tomorrow cannot retroactively change what a
running experiment was told to do. Drift is normal and silently fine: frozen wins — until an
operator deliberately adopts it ([below](#seeing-the-drift-and-adopting-it)).

Every leaf setting belongs to exactly one of three tiers, classified in
`src/noctis/config/rehydrate.py` and ratcheted by the test suite the same way the overlay's table
is. Today: **72 frozen, 17 live, 2 refused**. The record publishes the three lists it froze under
`inputs.settings` ([run-record.md](run-record.md#inputs--the-frozen-configuration)), so a consumer
never has to guess which tier a key is in.

| Tier | Count | What | Where it comes from on a resume |
|---|---|---|---|
| **Frozen** | 72 | Everything that decides what the accumulated results *mean*: `research.*`, `promotion.*`, `backtest.*`, `trading.*`, `risk.*`, `ideation.*`, `universe`, `session.*`, `champion_count`, `data.provider` / `dataset` / `history_days` / `auto_backfill`, `research_time_budget_minutes`, `run_limit_hours`, `embed_all_sources`, `live_feed.*` — **plus the whole mandate** | the record |
| **Live** | 17 | The three API keys; every path knob (`workspace_dir`, `runs_dir`, `run_dir`, `state_dir`, `reports_dir`, `memory_path`, `qa_dir`, `strategies_dir`, `mandate_dir`, `data.lake_dir`); the per-process budgets `time_limit_hours`, `data.budget_usd`, `qa.keep_last_runs`, `observability.heartbeat_polls` | the current process |
| **Refused** | 2 | `mode`, `allow_live` | neither — see below |

**Frozen includes the mandate, as resolved text.** The record stores the mandate's body verbatim
(with a digest), its summary, symbols, references and the overlay it applied — not the selector.
Freezing `profile:aggressive` would freeze nothing at all, because the file behind that name is
free to change tonight.

**Live is not a gap, it is the design.** Secrets are redacted out of the record, so a record is
shareable and resuming it needs *your own* keys from `.env`. Paths are live so a run can resume on
a machine whose absolute paths differ. And the per-process budgets bound one *night*, not one
experiment — `--time-limit-hours` is how you decide how long tonight lasts, weeks after the run
started.

**The two wall-clock ceilings sit in different tiers on purpose.** `time_limit_hours` is **live**:
it bounds this process, and how long tonight lasts is your call every night. `run_limit_hours` is
**frozen**: it bounds the whole run across every stop/resume, so it is part of what the experiment
*is* — 100 research hours and 30 are not the same experiment, and a cap that could be raised each
morning would bound nothing at all. Editing `run_limit_hours` in `config.yaml` therefore has no
effect on a run already under way; it applies to the next run you mint
([cli.md](cli.md#bounding-a-run----run-limit-hours-and---finish)).

**Refused means never recorded and never restored.** The safety gate re-resolves from
`config.yaml` + `ALLOW_LIVE` at every process start (see [safety.md](safety.md)); a record can
never resurrect a mode, so `mode: live` without `ALLOW_LIVE` is the same hard startup error on a
resume as on a first start. The record does carry the gate's *verdict* for the run
(`inputs.execution_mode`) as evidence — and a resume whose freshly resolved mode disagrees with it
is a **hard error**, so a paper run's results can never acquire live segments.

Two of the three tiers are **derived, not re-listed**: the refused pair is exactly the overlay's
live-money refusals, the path knobs are exactly its state/IO refusals, and the secrets are the one
set `Settings` names — so classifying a new knob in the overlay's table (which you must do anyway)
puts it in the right freezing tier with no second edit. Frozen is then the complement, which means
a knob added tomorrow freezes by default: the safe direction, because it keeps meaning attached to
results.

Because the mandate and the metric are frozen, `--mandate` / `--directive` / `--metric` are
**refused with a reason** on a resume rather than silently ignored. Start a new run to research
something else — identity is minted, never derived, so a fresh run under any configuration is one
command away.

### Seeing the drift, and adopting it

Frozen winning silently is right for the common case and wrong as the *only* option: an operator
who really did mean to change the run's configuration needs a way to say so. Two flags on
`noctis run --resume` (details and output in
[cli.md](cli.md#config-drift-seeing-it-and-adopting-it)):

- `--show-config-drift` prints how the current `config.yaml` and `mandate/` differ from what the
  run froze, then exits. Inspection only — it opens no segment, takes no lock, writes nothing.
  It compares the **72 frozen keys** and the resolved **mandate text**; the 17 live keys are never
  reported (they are this process's by design) and the 2 refused ones never appear at all.
- `--rebase-config` adopts the current files for the rest of the run: it re-freezes them, bumps
  `inputs.config_epoch`, and appends a before/after entry to `inputs.config_changes` naming the
  segment. **Never silent** — a run whose config changed mid-flight says so and says where. With
  no drift it is a no-op: the epoch never moves for a change that did not happen.

The refused tier is absolute here too. `mode` is not in the frozen settings at all (the record
carries only the gate's verdict), so the one way to *attempt* rebasing it — editing `mode`, opening
`ALLOW_LIVE`, and asking for the current files — is refused by the mode check that runs before any
rebase, with a message saying that no flag lifts it.

## The mandate overlay

The active mandate's front-matter `config:` block overlays the **run-shaping** knobs of this
page — and only those. The split is a statement about who owns what: a mandate configures the
*run* (which model thinks, what it may spend, how big its prompt gets, which names it starts
from, what "good" is scored as), while `config.yaml` keeps the *arena* (the safety mode, the
fill-cost floor, the promotion thresholds, the two-axis holdout geometry, the output paths, the
secrets). Every leaf setting is classified **exactly once** in `src/noctis/config/overlay.py` —
the authoritative table, with a justification comment per group — and a completeness ratchet in
the test suite fails until a newly added knob is classified deliberately, so nothing is allowed
by accident of omission. Today: **39 allowed, 2 clamped, 53 refused**. The whole surface also
ships commented-out in `mandate/MANDATE.md.example`, so it is discoverable without reading
source.

**Allowed (39), in six groups.** None of them is read by the promotion gates
(`champions/promotion.py`), the split geometry (`backtest/splits.py`), or the safety gate
(`config/gate.py`) — that is the property that makes them settable at all.

| Group | Knobs |
|---|---|
| Model seam | `research.model`, `research.base_url`, `research.agent.model` / `coder_model` / `coder_fallback_model`, the three thinking dials (`thinking`, `coder_thinking`, `coder_fallback_thinking`), the coder's sampling dials (`coder_temperature`, `coder_seed`), `research.agent.loop` |
| Spend ceilings | `research.cost_profile`, `research.agent.max_iterations` / `max_backtests` / `sweep_trials` / `max_author_calls` / `max_escalations` / `max_tokens` / `coder_max_tokens` / `context_window` / `episode_retries` / `coder_retries` / `web_search` / `max_web_searches` / `sweep_workers` / `worker_bar_budget`, `research_time_budget_minutes`, `time_limit_hours`, `run_limit_hours` |
| Search shape | `promotion.metric`, `research.focus_size`, `research.tuning_dispersion_penalty`, `research.draft_ttl_hours`, `research.memory_distill_every` |
| Data acquisition | `data.history_days`, `data.auto_backfill` |
| Housekeeping | `observability.heartbeat_polls`, `qa.keep_last_runs` |
| Seed universe | `universe` |

**Clamped (2) — legal only in the direction that adds discipline.**

| Knob | Direction | Why |
|---|---|---|
| `research.min_trials` | **raise only** | The exhaustion floor. A tune-first personality demanding 20 journaled param sets per verdict is legitimate steering; demanding fewer is the loosening [safety.md](safety.md) forbids. |
| `data.budget_usd` | **lower only** | The vendor spend cap. A mandate may spend less of your money; the ceiling `config.yaml` set stays the ceiling. |

The bound is whatever config resolved for that path (equal to it is fine), so the clamp
constrains the *overlay* and nothing else — you still set the number you want in `config.yaml`.
A wrong-direction value is **fatal**, and the message names the configured value it tried to
cross; nothing is silently clipped, because a silent clamp would let a mandate say one thing
while the run does another. `null` — "no bound": an unlimited budget, no exhaustion floor at all
— ranks as the least-disciplined end of *either* scale, so an overlay may replace it with a
number and never the reverse.

**Refused (53) — fatal at startup, with the reason printed.** A refused, unknown, or invalid
key stops the process before any work starts, listing **every** problem in one error so a bad
mandate is one fix rather than a fix-one-rerun loop. (Refusals used to be warned about and
silently skipped, which meant discovering three days into a run that a knob never applied.) The
refused set is the arena: the live-money double gate (`mode`, `allow_live`), the enforced
fill-cost floor (`backtest.fee_bps`, `backtest.slippage_bps`), every `promotion.*` except
`metric`, the holdout geometry (`research.fit_set_size`, `research.symbol_holdout_size`),
`champion_count`, every state/IO path (`workspace_dir`, `runs_dir`, `run_dir`, `state_dir`,
`reports_dir`, `memory_path`, `qa_dir`, `strategies_dir`, `mandate_dir`, `data.lake_dir`), the
three API keys, cost accounting (`research.pricing`), record content (`embed_all_sources`),
self-selection (`research.mandate`, `research.mode`), `risk.*` / `trading.*` /
`live_feed.poll_interval_s`, `data.provider` / `data.dataset`, `session.calendar` /
`session.timezone`, and `ideation.*`.

Two of those refusals look like they could have been clamps, and are worth spelling out:

- **The promotion thresholds are refused, not "tighten-only" clamped.** They are read in the
  units of `promotion.metric`, which the same mandate may change — so a `max_gap: 0.5` is not
  tighter than a Sharpe-units `1.0`, it is a different number in a different scale. It is the
  same incomparability that makes a champion scored under a different metric **stale** rather
  than beatable ([validation.md](validation.md)).
- **The state paths are the sharpest escape hatch there is.** Redirecting `state_dir` moves the
  experiment journal the exhaustion gate counts trials from — a fresh directory is a *reset
  exhaustion gate*, having touched no gate knob at all. Redirecting `strategies_dir` moves the
  champions tier, and with it champion immutability.

**The seed universe carries a starvation guard.** A mandate-set `universe` is upper-cased and
de-duped (exactly like a mandate's `symbols:` list), and must then name at least
`research.fit_set_size + research.symbol_holdout_size` symbols — both refused, so a mandate
cannot shrink the requirement to fit its roster. Fewer is fatal, and the message names the
**symbol-holdout gate** that would otherwise go inert rather than be cleared: a gate can be
defeated by starving it as well as by lowering it. This is an overlay-only check by design — a
`config.yaml` universe below the panel size is the operator's own arena and is left alone.

### Precedence

```
CLI flags  >  mandate overlay  >  environment  >  .env  >  config.yaml  >  built-in defaults
```

The overlay applies to a **fully-constructed** settings object — pydantic has already resolved
env > `.env` > YAML by the time a mandate is read — so for the overlaid subset this **inverts**
the "environment always wins over the YAML file" rule stated at the top of this page. That is
deliberate: a mandate is a per-run *selection* an operator makes on purpose, not ambient
environment, and the whole point of pinning one is that the run is configured by it. Secrets and
`ALLOW_LIVE` are unaffected — they are refused, so the environment stays their only source.
`--metric` and `--time-limit-hours` are applied after the overlay and still win; for one session
`--mandate <name>` or `--directive "<text>"` (mutually exclusive) override the
`research.mandate` selector. The whole chain resolves in one place — `resolve_session` in
`src/noctis/bootstrap.py`, the composition root — so the ordering can never drift between
commands, and every overlay it performs is bracketed by a gate-unmoved assertion over the
refused subtree.

### `research.mandate: auto` makes a profile's `config:` block inert

Under the shipped `auto` selector the *agent* picks its profile partway through the session,
long after settings were assembled — so no profile's `config:` block ever reaches the overlay.
Nothing fails; the run simply isn't steered by it. Startup therefore logs one warning naming
every profile whose keys will not apply, plus the remedy (pin the mandate). `promotion.metric`
is deliberately **suppressed** from that warning: it is inert *by contract* under `auto` (the
selection instruction is explicitly metric-neutral, and the shipped `config.yaml` says an auto
session is scored on the base metric), so warning about it on every stock install would be the
noise that gets the interesting warning ignored. That is exactly why the five shipped profiles
stay **metric-only** — a shipped profile that declared a second key would make every default
install warn, and a test holds them there.

Details: [research.md](research.md) and `mandate/README.md`.

## Fill costs (the enforced floor)

`backtest.fee_bps` and `backtest.slippage_bps` set the simulated trading cost **per side**
(enter and exit each pay), so the round trip the research agent reasons about is `2 ×
(fee_bps + slippage_bps)`. The default is the shipped baseline — `1.0`/`1.0`, a 4bp round
trip — and an unset `backtest:` section behaves bit-for-bit as before. The single value is
threaded from the composition root into the coarse pre-filter, walk-forward validation, the
agent's cost hint (the market digest's `round_trip_cost_bp`), and the paper-fill broker, so
those four can never disagree on what a trade costs.

The baseline is **also the enforced minimum**: a value below `1.0` per side is a hard
startup error (like `mode: live` without `ALLOW_LIVE`), never a silent clamp. The cost model
is the system's main difficulty knob — dialing it below the baseline is the cheapest way to
manufacture champions that would die on real fills — so the knob may only be raised toward
per-venue realism, never lowered. For the same reason the whole `backtest:` section is
**refused by name** in the mandate overlay, however wide that surface grows: a research
personality steers *what* to look for, never how forgiving the arena is. A mandate that tries
it does not start, and the error says so.

## Pricing the spend estimate

The run record publishes what a run cost — `spend.llm_usd_estimate`, `usd_per_champion_estimate`,
`usd_per_trial_estimate` — and every one of those numbers is an **estimate**, priced from a
versioned `$/Mtok` table in `src/noctis/research/pricing.py` and labelled as such in the record and
in any CLI output. Tokens are measured; dollars are inferred from list prices that ignore volume
discounts, batch tiers and mid-month changes.

Three rules make the estimate safe to publish, and they are worth knowing before you read one:

- **An unknown model contributes `null`, never zero.** A model the table does not carry has no
  price, and any total it belongs to is `null` too — a partial sum presented as a total would read
  as complete while understating the bill. A `$0`/token local backend (`ollama/…`, `ollama_chat/…`,
  `vllm/…`, `lm_studio/…`, `local/…`) is priced at an explicit zero, because that zero is a *stated
  price*. That list is an **allowlist**, deliberately: a provider that merely isn't one of the paid
  clouds the table surveyed is *unknown*, not free — otherwise a paid third-party gateway would one
  day publish a confident `$0`.
- **The table version travels with the numbers.** `spend.pricing_table_version` names the table
  that produced them, so a record read next year is still interpretable. The label is
  `<month>[.<revision>]` — `2026-07.1` is July 2026's prices with corrected coverage (`ollama_chat/`
  was missing, so the shipped local driver priced as an unknown model). Coverage earns a revision
  because it changes the published number for the same model, and a record keeps whichever label was
  in force when it was written: nothing is migrated, and that is the point of having one.
- **The estimate is never a gate.** Nothing here is read by a promotion gate, a budget, or the
  exhaustion floor. `data.budget_usd` (the vendor-data preflight) and `research.cost_profile` (the
  session ceilings) are the knobs that actually bound spend; this one only reports it.

`research.pricing` overrides or extends the table, keyed by **model prefix** (longest match wins),
with all four rates required — input, output, cache-write and cache-read bill separately, and a
half-stated price would silently value the rest at nothing:

```yaml
research:
  pricing:
    "anthropic/claude-opus-4":
      input_usd_per_mtok: 15.0
      output_usd_per_mtok: 75.0
      cache_write_usd_per_mtok: 18.75
      cache_read_usd_per_mtok: 1.5
```

An overridden table **cannot borrow the shipped version label**: it identifies itself as
`<version>+custom.<digest>` (e.g. `2026-07.1+custom.a1b2c3d4`), derived from the override itself —
stable for the same prices, different for any other. So a reader can always tell whether the
numbers came from this engine's own table.

Pricing is **frozen with the rest of the run's config**: a run prices under the table it was
created with, so editing `research.pricing` tomorrow cannot restate what last night cost. And it is
**refused by the mandate overlay** by name, for the reason the fill-cost floor is: an experiment
that could restate its own bill would make every cross-run cost comparison a claim it made about
itself.

## Local backends (noctis-ollama)

Any OpenAI-compatible or Ollama endpoint can serve `research.model` at $0/token —
[noctis-ollama](https://github.com/bmeunier1974/noctis-ollama) turns a bare GPU box into a
verified, agent-ready one with a single `./setup.sh`. `noctis setup` detects a running
server and writes the wiring for you; by hand, it is one block in `config.yaml`:

```yaml
research:
  model: ollama_chat/noctis-qwen3:14b # any tag the server carries; `ollama_chat/` prefix
  agent:
    max_tokens: 4096 # output cap — small-context backends bound prompt+output together
    context_window: 32768 # the model's num_ctx — activates the prompt-trimming levers
```

No API key is needed. The three knobs usually travel together: `max_tokens` keeps a
completion inside the window (a thinking model needs room to reason *and* emit a tool
call), and `context_window` bounds the whole request — per-result caps tier down, the
oldest tool results evict to pointer lines, and a decided strategy's history collapses at
its verdict. Both are compatibility levers, not cost budgets: the on-disk experiment
journal stays the ground truth, so no gate or holdout is affected. Declaring a
`context_window` of at most 32,768 also flips the default `research.agent.loop: auto` to
the **episodic** driver — the loop built for exactly this class of backend (see
[research.md](research.md) and the parity evidence in [parity.md](parity.md); set
`loop: conversation` to opt out). A non-Ollama endpoint
(vLLM, a proxy) uses `research.base_url` plus its own model id. On a free/local provider
the `cost_profile` automatically resolves to `full`.

## The coder-model split

`research.model` runs the whole session — the thesis, the tool orchestration, the judgment —
but `write_strategy` also demands a complete, validation-passing ~200-line strategy file in one
shot, the one thing a cheap or local driver thrashes on. `research.agent.coder_model` splits
that role out: the driver keeps the session, a dedicated **coder** model does nothing but turn a
structured brief into a validated file (the mechanics are in [research.md](research.md)).

```yaml
research:
  model: ollama_chat/noctis-qwen3:14b # cheap driver runs the session…
  agent:
    coder_model: anthropic/claude-sonnet-5 # …a real coder authors the strategy files
    max_author_calls: 12 # cap coder completions/session (null = cost_profile's 20/12/6)
```

- **`coder_model`** takes the same LiteLLM `provider/model` grammar as `research.model` — the
  provider prefix resolves the API key from `.env` (`anthropic/claude-sonnet-5` reads
  `ANTHROPIC_API_KEY`; a local driver still needs none). It defaults to `null` =
  **driver-authored mode**: the driver writes full source itself, today's behavior bit for bit.
  A configured coder whose provider key or `[llm]` extra is missing degrades *loudly* back to
  that mode at startup — a warning, never a silent mid-session downgrade.
- **`max_author_calls`** is the coder's Class-B budget: coder completions per session, private
  validation retries included (one authored or revised file is nominally one call; a file that
  needs a retry spends more). Like the other agent budgets it defaults to `null` = the active
  `cost_profile`'s value (`20` / `12` / `6` for `full` / `balanced` / `economy`); a number here
  **pins** it regardless of profile. Once spent, further brief authoring is refused — the driver
  is told to revise by hand or proceed to a verdict — while the hand-written `source` path,
  which spends no coder completion, always stays open. Inert without a `coder_model`: source
  writes never touch this budget.
- **`coder_thinking`** is the coder's own thinking dial, **on by default**. Authoring — the
  scenario-window and warmup arithmetic — is the reasoning-heavy sub-task, so the coder reasons
  through it instead of repeating an error it was just shown. It is a *deliberate*, budgeted
  decision made where the coder client is built, so it turns thinking on even for a Sonnet coder
  (whose driver-side thinking stays the cheap-path pin below); the extra spend is already bounded
  by `max_author_calls`. Set it `off` to run a cheaper coder. The coder's (enlarged) system prompt
  is prompt-cached, so private validation retries within a job re-read it rather than re-paying it.
  Inert without a `coder_model`. Separate from the driver's `thinking` watch dial (next section).
- **`coder_max_tokens`** is the coder's output-token ceiling — the **file's** budget. It defaults
  to `null` = the author engine's built-in ceiling (`16000`), sized so a full ~200-line strategy
  file never truncates mid-source; a number **pins** it. A coder client that runs provider
  thinking gets the engine's thinking allowance (`32000`) added *on top* — on Anthropic models
  thinking and text share `max_tokens`, and the first field run of escalation (#98) showed
  adaptive thinking eating a 32k ceiling until no file fit — so this ceiling stays all text by
  construction. This is a compatibility/sizing lever — resize it for a coder backend whose output
  window differs — **not** a cost budget: output tokens are billed only as they are generated, so
  unused headroom costs nothing (the coder's spend is bounded by `max_author_calls`, not by this
  ceiling). Like the other `coder_*` knobs it is inert without a `coder_model`.
- **`coder_fallback_model`** + **`max_escalations`** are the paid escalation path (#72): when the
  local coder spends its whole validator-retry budget on a brief, the *same* brief escalates once
  to this (typically stronger, hosted) model with the full retry budget. `max_escalations`
  defaults to `0` = never escalate, so the paid coder is strictly opt-in bounded spend; each
  escalation counts whether or not the file then lands. **`coder_fallback_thinking`** is the
  escalated coder's own thinking dial, **off by default** (#98): the fallback is the strong model,
  and the field run that first exercised escalation showed a thinking sonnet-5 fallback timing out
  and thinking-truncating every file — with it off, the escalated call spends its whole ceiling on
  the file. Set it `on` to opt the fallback into the same deliberate thinking as the local coder
  (authoring completions stream, and the thinking allowance above applies, so it is survivable —
  just slower and costlier). All three are inert without a `coder_model`.

## Watching the model reason

`research.agent.thinking` is a binary, provider-neutral watch switch, **off by default**. Turned
`on`, it opts the session into provider-native reasoning where it exists: for an Anthropic
non-Sonnet model (the fallback `claude-opus-4-8`) it sends adaptive thinking with a summarized
display, so the research loop surfaces `think` events (see [research.md](research.md) on
observability). It is a no-op on OpenAI and local backends (no thinking dial) and leaves Sonnet's
thinking pinned off (the deliberate cheap path). This is the only observability knob that changes a
request parameter and the only one that spends more — turning it on to watch the model think costs
adaptive-thinking output tokens, so **leave it `off` for unattended runs**. Adaptive thinking has
no token budget to tune (the model chooses depth), which is why the knob is a switch, not a dial.

## Live feed opt-in

`data.provider` also selects the TRADING-phase live data source. The default keeps TRADING on
offline **catalog replay**; `data.provider: yfinance` opts in to the free, ~15-min-delayed
Yahoo Finance feed (no credentials; needs the `data` extra). A bare `noctis run` never contacts
a live feed unless you explicitly ask for it. Feed behavior: [data.md](data.md).

## Secrets

Secrets live in `.env` (gitignored — copy `.env.example` and fill in). **Nothing is required to
run in paper mode.**

| Variable | Purpose |
|---|---|
| `DATABENTO_API_KEY` | Historical market-data ingests (research/backtest) |
| `<PROVIDER>_API_KEY` | Optional LLM key for hosted models — the variable name matches your `research.model` provider prefix; local backends need none |
| `ALLOW_LIVE` | The live-execution env gate (leave blank/unset for paper) |
