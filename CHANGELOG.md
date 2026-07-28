# Changelog

All notable changes to Noctis are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Engine identity** (`src/noctis/observability/engine_id.py`) and the `noctis engine` verb.
  A declared `ENGINE_VERSION` (a plain incrementing integer, decoupled from the package
  version) plus a **per-component** fingerprint over the committed files that decide behaviour
  — `gates`, `backtest`, `research`, `prompts`, `profiles`, `seeds`, `memory_seed`, `schema` —
  and the comparable key `(engine_version, gates_digest, backtest_digest, election_metric)`
  two runs must match on before their numbers may be pooled or ranked together. Only committed
  scaffold is hashed, so an operator's gitignored mandate never moves a digest and fingerprints
  stay portable between machines.

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
