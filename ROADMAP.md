# Roadmap

Noctis follows [Semantic Versioning](https://semver.org/). The milestone ladder below is the
project's direction; see [CHANGELOG.md](CHANGELOG.md) for what has actually shipped. The
codebase is already past "foundation", so v0.1.0 *captures* the current state rather than
promising it.

Two threads run through every milestone:

- **Close the real gaps.** The components a serious research system needs that Noctis does not
  have yet: a forward record that feeds back into who stays champion, portfolio-level risk,
  cost realism beyond flat basis points, and a research history that is corrected for how many
  things were tried.
- **Sharpen what sets it apart.** Most open trading projects compete on backtest features;
  Noctis competes on **how hard it is to fool**. Every differentiator below deepens that:
  research-wide multiple-testing honesty, champions that must keep earning their slot on live
  bars, and provenance an outsider can audit and replay. Items marked *(differentiator)* are
  the ones few comparable projects attempt.

## Non-goals

Some things are permanently off this roadmap, not merely unscheduled:

- **Live order execution.** Paper-only is a design invariant, not a maturity stage. The two-gate
  refusal and the stub live adapter stay ([docs/safety.md](docs/safety.md)).
- **Configurable gate weakening.** No milestone will add knobs that let an operator (or the
  agent) lower a holdout bar, widen a gap tolerance, or shrink a holdout set to make a
  candidate pass. A failing strategy is a signal ([docs/validation.md](docs/validation.md)).
- **Vendor lock-in.** The provider-neutral LLM seam and optional heavy dependencies are
  permanent structure; no feature may require a specific hosted model to function.

## v0.1.0 — First public release ✅

Already built; the release tag certifies it. Package structure, CLI, phase-loop engine, agent
research sessions, strategy library + validation-on-write gate, backtest pipeline, promotion
gates, two-axis holdout validation, uv-locked environments + CI, governance, and documentation.
See the [0.1.0 changelog entry](CHANGELOG.md).

## v0.2 — Research-engine deepening

*Theme: the research loop got its largest structural change — an episodic driver a small local
model can drive end-to-end — and next, every LLM judgment the loop makes becomes measurable,
while the experiment journal it already leans on is made to do more work.*

- **Episodic research: a loop a small local model can drive.** ✅ *Shipped (#62–#113): the
  episodic driver is live, and `auto` selects it on small context windows.*
  - **Why:** overnight research is *throughput* economics — most output honestly ends in
    `reject_strategy`, so frontier-API prices for tokens whose expected fate is rejection invert
    the value proposition. A weaker model is acceptable because quality control lives in the
    gates and the fresh-subprocess validator, not in the model's judgment: a dumber model can
    waste attempts but cannot corrupt the champion board.
  - **The shape — code owns the loop:** a deterministic driver runs formulate → match →
    optimize → decide; the LLM is a stateless decision function at narrow judgment points, and
    **every call is an episode rebuilt from disk** — no transcript. The session narrative lives
    in a structured ledger plus `thesis` journal lineage, so a killed session re-enters the
    stage machine from disk, something a transcript never could.
  - **Local-first, escalation-bounded:** the local model attempts everything; the paid model is
    a counted, validator-triggered fallback (`coder_fallback_model`, `max_escalations`), never a
    default — making the roadmap's own vendor-neutrality non-goal real for research.
  - **No invariant moved:** no gate weakened, the journal schema extended (not changed), both
    holdout axes live, paper-only intact; the conversation loop stays **frozen** as the parity
    baseline and big-window fallback, sharing the same toolbox.
  - **The flip was evidence-gated:** the parity harness ran both loops on a fixed lake fixture
    before `auto` switched to episodic for declared context windows ≤ 32k (#76).
- **Agent-evaluation platform: benchmark every LLM call site** *(differentiator)*. Noctis makes
  LLM calls at ~7 sites — the coder authors files, DECIDE converts candidates into verdicts,
  FORMULATE invents theses, DISCOVER proposes symbols, ideation seeds rounds, distillation
  compacts memory — and every one is steered by knobs tuned on **anecdote**. Today "is model A
  better than B, per dollar" is unanswerable without a live session. The episodic driver is
  already the right shape (stateless episodes, typed contracts, persisted before acted on); the
  platform closes the three gaps around it — inputs not persisted, sites not declared, prompts
  not versioned — as a dependency-ordered epic sequence:
  - **Capture and provenance first.** Every benchmarkable site's input is captured (content hash
    in the ledger row, body in capped sidecars under the run's `qa/`), a `scorecard` record is
    journaled at gate time, prompt assets are content-hashed behind a fingerprint-style ratchet,
    and the served model id is recorded per attempt — so every real research session becomes a
    free, continuously growing, in-distribution eval corpus.
  - **A thin in-house eval core**, import-isolated from the engine and enforced by a test:
    declared `AgentSite`s, a YAML case corpus with a frozen tuning/holdout split, pure metrics
    (pass rates, attempts, cost/latency, failure taxonomy, McNemar/bootstrap paired statistics),
    and a `bench.json` record keyed by `comparable_key` — records with different keys render
    side by side with a banner, never as a delta.
  - **Two sites prove the abstraction in v1.** **DECIDE** (site #1): approval-side agreement
    with the promotion gates, always paired with the approval rate, mined retrospectively from
    existing run records at zero new model spend. **The coder** (site #2): `write_strategy` as
    its executable oracle, a failure-taxonomy classifier, and a four-bucket case corpus.
  - **One CLI for every site:** `bench run` / `report` / `compare` / `sweep` / `history` /
    `mine`, with a `--dry-run` spend preflight; every tier is operator-run, nothing wired into
    CI.
  - **The standing rule:** the platform measures *agent capability*, never trading edge. A
    benchmark number is never grounds to touch a gate, ablation knobs are unreachable from a
    production run, and no cross-site composite score will ever exist.
- **Trial-aware honesty** *(differentiator)*. A scorecard number means less the more things were
  tried before it appeared. Use the experiment journal to report multiplicity-adjusted evidence:
  deflated performance metrics (Bailey/López de Prado-style) computed over the full trial
  history of a candidate's family, and a visible "survived N distinct trials" stamp on every
  promotion. This never *loosens* anything — it makes strong results harder to claim, which is
  exactly the project's brand. *Partially delivered: the run record already publishes a Deflated
  Sharpe Ratio deflated by the run's own cumulative trial count; this item extends the
  correction to per-candidate promotion evidence.*
- **Research bundles (reproducibility artifacts).** Every champion gains a self-contained
  provenance bundle: the strategy file, its journal slice, the scorecards that promoted it, the
  active mandate, and the data-coverage manifest it was evaluated on — enough for a stranger to
  re-run the promotion decision from scratch.
- **Research meta-analysis.** Cross-session mining of the journal and agent memory: which thesis
  styles, symbols, and parameter regions keep failing or keep working, distilled into the
  memory/digest so successive sessions search smarter instead of re-treading rejected ground.
- **Richer journaling and reporting.** Close the gap between what the loop knows and what the
  CLI report shows: per-trial lineage, verdict rationales, and research-session summaries in
  `noctis report`.
- **Growth of `examples/`** — worked mandates, replay scenarios, and an end-to-end "one research
  night, annotated" walkthrough.
- **Cost-profile hardening.** Keep the $0 local-backend path first-class: tighter token budgets
  and per-session spend accounting. The hard part — making an ~8k-context local model actually
  drive a session — is the episodic-research item above, not a separate line here.

## v0.3 — Validation framework: the forward record feeds back

*Theme: Noctis builds a genuine forward paper record, but today that record does not influence
who stays champion — displacement compares backtest scorecards only. Close the loop.*

- **Live-forward accountability** *(differentiator)*. Champions accumulate forward-ledger
  statistics on bars no tuning ever saw; a champion whose live performance decays materially
  below its holdout expectation is put on probation and demoted — not just displaced when a
  better backtest shows up. This is the last unexploited out-of-sample axis, and the one that
  cannot be overfit even in principle.
- **Benchmark suite.** Every scorecard gains context: buy-and-hold on the same panel, a
  random-entry baseline with matched turnover, and per-family reference strategies — so "the
  edge" is always measured against what zero skill would have earned, costs included.
- **Adversarial review** *(differentiator)*. A second, skeptic agent session that attacks a
  candidate before promotion: parameter-neighborhood stability (does the edge survive ±1 step
  on every knob?), regime slicing (is all the profit from one week?), and thesis-consistency
  probes (does it trade *for the stated reason*?). Findings attach to the scorecard as evidence
  for the verdict — they inform the decision, they don't silently gate it.
- **Cost realism.** The simulator charges flat fee/slippage basis points today. Deepen it:
  spread-aware slippage, volume-participation caps for 1-minute bars, short-borrow cost, and
  overnight-gap treatment — with the honest default being the *pessimistic* one.
- **Universe integrity audit.** Corporate actions, delistings, and symbol-selection bias:
  verify the lake's adjustment story end-to-end and make survivorship assumptions explicit in
  the data docs and the coverage registry.
- **Reproducibility reports.** `noctis report` grows a validation appendix: re-run drift checks
  on stored champions, benchmark deltas, and forward-vs-holdout tracking over time.

## v0.4 — Portfolio intelligence

*Theme: champions currently trade side by side with naive per-position sizing (a fixed fraction
of equity per symbol). Treat the board as a portfolio.*

- **Capital allocation across champions.** Replace equal-and-independent sizing with an
  explicit allocator: volatility-targeted position sizing, per-champion capital budgets, and a
  policy seam so allocation logic is swappable and testable like everything else.
- **Exposure and correlation limits.** Portfolio-level caps on gross/net exposure and on
  crowding — three champions long the same symbol is one bet, not three.
- **Portfolio-level risk policy.** Session and multi-day drawdown rules at the account level
  (extending the existing halt latch), with the same structural, non-configurable-away
  discipline as the promotion gates.
- **Regime awareness.** Market-state detection (trend/chop/vol level) in the market digest,
  regime tags on champion provenance, and regime-conditional research priors — so the mandate
  can say "find something for *this* tape" without any gate changing.

## v0.5 — Operator surface

*Theme: the system is observable through a CLI today; give it a face.*

- **Local dashboard.** A read-only local web UI over the workspace: equity curve, champion
  board with provenance, live research feed (the same event seam the CLI streams), journal
  browser, and data-lake coverage. No new write paths — observability only.

## v1.0 — Stable release

API stabilization, complete documentation, PyPI packaging, and production readiness: frozen
public seam contracts (strategy file, scorecard, journal, bundle formats), a documented
deprecation policy, and upgrade paths for workspace state.

## Beyond 1.0 — exploratory

Unscheduled directions worth recording: additional asset classes and calendars (crypto's 24/7
clock would exercise the phase machine hard), multi-timeframe strategies, additional data
vendors behind the existing seam, and multi-agent research tournaments (independent sessions
competing on the same mandate, with promotion arbitrating).
