# Contributing to Noctis

Thank you for your interest in contributing to **Noctis**. This document defines the
standards, workflow, and governance model that apply to all contributions. It is a binding
process document: pull requests that do not follow it will be returned for revision before
review, regardless of the technical merit of the change.

Noctis is an autonomous, **paper-only** quantitative trading system released under the
[MIT License](LICENSE). We welcome well-scoped issues, disciplined pull requests, and rigorous
discussion. We equally value the willingness to have a contribution declined when it does not
fit the project's architecture or invariants.

All participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). To report a
security issue — including anything that weakens the paper-only gates — follow the
[Security Policy](SECURITY.md) and **do not** open a public issue.

---

## Table of Contents

1. [Project Invariants](#1-project-invariants)
2. [Development Environment](#2-development-environment)
3. [Code Quality Standards](#3-code-quality-standards)
4. [Contribution Workflow](#4-contribution-workflow)
5. [Testing Requirements](#5-testing-requirements)
6. [Pull Request Checklist](#6-pull-request-checklist)
7. [Maintainer Notice and Governance](#7-maintainer-notice-and-governance)
8. [Licensing of Contributions](#8-licensing-of-contributions)

---

## 1. Project Invariants

The following invariants are load-bearing and **non-negotiable**. Pull requests that weaken any
of them will be closed without merge, independent of code quality:

- **Paper-only, two-gate safety.** Real-money order paths are structurally unreachable unless
  both the `mode: live` configuration and the `ALLOW_LIVE=true` environment gate are set — and
  the live adapter remains a refusing stub. Changes may never weaken either gate, merge their
  sources, or wire a real order path into the stub.
- **Promotion gates are the arbiter of quality.** A strategy that fails the promotion gates is a
  signal, not a defect. Contributions must not loosen gates, holdout thresholds, or holdout set
  sizes to make a candidate pass.
- **No lookahead.** Decisions are made on bar *t* and filled at bar *t+1*; test windows sit
  strictly after training windows; holdout data never reaches previews or research context. Any
  change that lets future or holdout information reach a decision is a correctness bug.
- **Secrets and vendor data stay out of git.** Credentials live in `.env` (gitignored); market
  data lives in the reproducible data lake, never in the repository.

If a proposed change would touch one of these areas, open an issue for discussion **before**
writing code.

## 2. Development Environment

The project targets **Python 3.11+** and uses [`uv`](https://github.com/astral-sh/uv) for
environment management.

```bash
# Minimal environment (core + dev tooling; sufficient for the test suite and paper mode).
# Reads uv.lock and installs the exact locked versions into .venv.
uv sync

# Full development environment (adds all optional runtime seams)
uv sync --all-extras

# Install the pre-commit hooks (required — see Section 3)
uv run pre-commit install
```

Run commands in the synced environment with `uv run` (e.g. `uv run pytest`), or activate
`.venv` first. CI runs `uv sync --locked`, which fails if `pyproject.toml` and `uv.lock` drift.

Heavy dependencies (`nautilus_trader`, `vectorbt`, `databento`, `optuna`, `anthropic`, …) are
**optional extras** isolated behind swappable seams. Do not promote an extra into the core
dependency list; the full test suite must continue to pass on the minimal install.

## 3. Code Quality Standards

All code must pass the following gates **before a pull request will be reviewed**. Submissions
that fail any gate will be returned without technical review.

### 3.1 Linting and Formatting — Ruff

The project enforces Ruff for both linting and formatting (line length 100, target `py311`,
rule sets `E`, `F`, `I`, `W`, `UP`, `B` as configured in `pyproject.toml`):

```bash
ruff check .            # must report zero violations
ruff format --check .   # must report zero formatting diffs
```

Do not add `# noqa` suppressions or per-file ignores without prior maintainer approval in the
associated issue. Existing ignores in `pyproject.toml` are deliberate and documented inline.

### 3.2 Static Type Checking — mypy

The `src/noctis` package is statically type-checked and must pass cleanly:

```bash
mypy
```

New code must be fully type-annotated. Do not introduce `type: ignore` comments, `Any`-typed
public interfaces, or relaxations of the mypy configuration to achieve a passing run.

### 3.3 Consolidated Gate — pre-commit

All quality gates can be executed together, and must pass on every commit:

```bash
pre-commit run --all-files
```

### 3.4 AI-Assisted Code

Noctis itself is developed in close collaboration with AI coding agents (primarily
[Claude Code](https://claude.com/claude-code)), and AI-assisted contributions are welcome.
The standard is identical regardless of authorship: every gate in this section, the testing
requirements in Section 5, and the invariants in Section 1 apply in full. You are responsible
for understanding and standing behind everything you submit — a PR whose author cannot explain
its changes in review will be returned, exactly as a hand-written one would be.

## 4. Contribution Workflow

Contributions follow a strict issue-first, fork-and-branch pipeline:

1. **Open an issue first.** Every change — feature, fix, or refactor — begins with a GitHub
   issue describing the motivation, the proposed approach, and the affected components. For
   non-trivial changes, wait for maintainer sign-off on the approach before writing code. This
   protects your time as much as the project's.
2. **Fork the repository** to your own GitHub account and clone your fork locally.
3. **Create a feature branch** from the current `main`. Use a descriptive, scoped name:
   - `feature/<short-description>` for new functionality
   - `fix/<short-description>` for bug fixes
   - `docs/<short-description>` for documentation-only changes
4. **Develop in small, coherent commits.** Each commit should represent one logical change with
   a clear, imperative-mood message (e.g., "Add symbol-holdout coverage check"). Do not mix
   refactoring with behavioral changes in the same commit.
5. **Verify locally.** Run the full gate sequence in the uv environment before pushing:
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
   ```
6. **Submit a pull request** against `main`. The PR description must:
   - Reference the originating issue (`Closes #NN`);
   - Summarize *what* changed and *why*;
   - State how the change was tested, including any new test files;
   - Note any impact on the invariants in Section 1 (normally: "none").
7. **Respond to review.** Address review feedback through additional commits on the same
   branch. Keep the branch rebased on `main`; resolve conflicts on your side.

Pull requests should remain focused and reviewable. Large, multi-concern PRs will be asked to
be split. Draft PRs are welcome for early architectural feedback.

### Research contributions

A contribution that adds or changes a strategy, a promotion gate, or the validation pipeline is
a *research* contribution, and the bar is **evidence, not opinion**. Alongside the code, state:

- **Hypothesis** — the specific, falsifiable edge you claim (e.g. "post-gap momentum on
  high-ATR names persists for N bars").
- **Dataset** — the symbols and date range it was tuned and validated on, and which slices were
  held out (both the temporal and the symbol holdout).
- **Validation methodology** — how you tested it: walk-forward splits, the election metric, and
  the out-of-sample gates it cleared (see [Research methodology](docs/research.md)).
- **Expected outcome** — what the scorecard shows and why it is a genuine edge rather than an
  overfit. A candidate that only passes by loosening a gate is a rejection, not a contribution
  (Section 1).

Strategy files must additionally satisfy the full contract in
[`strategies/README.md`](strategies/README.md), including known-outcome `scenarios()`.

## 5. Testing Requirements

Testing is a condition of merge, not an afterthought.

- **New features require unit tests.** Every new feature or behavioral change must ship with
  `pytest` tests in `tests/` that exercise the new behavior, including relevant edge cases and
  failure paths. Property-based tests using `hypothesis` are encouraged where the input space
  warrants them.
- **Bug fixes require a regression test.** A fix must include a test that fails on the
  pre-fix code and passes on the fixed code.
- **Coverage must be adequate.** New code paths — particularly branching logic in gates,
  backtesting, and data-integrity code — must be meaningfully covered. A PR that adds logic
  without corresponding tests will not be reviewed.
- **The full suite must pass on the minimal install.** Tests must not require optional extras;
  heavy dependencies are exercised through their seams. Run:
  ```bash
  uv run pytest                                   # full suite
  uv run pytest tests/test_<module>.py::test_name # targeted run during development
  ```
- **Tests must be isolated and deterministic.** Follow the isolation conventions in
  `tests/conftest.py` (registry snapshot/restore, safety-gate and secret env vars cleared).
  Tests must not depend on network access, wall-clock market state, local `.env` contents, or
  ambient files in `state/` or `data_lake/`.
- **Strategy files carry their own oracle.** Contributions under `strategies/` must satisfy the
  full strategy contract (`strategies/README.md`), including known-outcome `scenarios()` that
  validate in a fresh subprocess.

## 6. Pull Request Checklist

Before requesting review, confirm every item:

- [ ] An issue exists and is referenced in the PR description.
- [ ] The branch is up to date with `main`.
- [ ] `uv sync` (or `uv sync --locked`) produces a clean environment.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass with zero findings.
- [ ] `uv run mypy` passes with zero errors.
- [ ] `uv run pytest` passes on a minimal (`uv sync`, core + dev) install.
- [ ] New behavior is covered by new tests; bug fixes include a regression test.
- [ ] Documentation is updated for any user-facing or behavioral change.
- [ ] Algorithm changes include benchmark/scorecard evidence (see *Research contributions*).
- [ ] No secrets, credentials, vendor market data, or files from `state/`, `data_lake/`, or
      `reports/` are included in the diff.
- [ ] The change does not weaken any invariant listed in Section 1.

## 7. Maintainer Notice and Governance

**The Noctis Core Team acts as the final authority** over all
architectural decisions, the project roadmap, the interpretation of the invariants in
Section 1, and the acceptance or rejection of every contribution.

In practical terms:

- All pull requests are reviewed and merged exclusively by the maintainer. No contribution is
  merged without explicit maintainer approval, and approval of an approach in an issue does not
  guarantee acceptance of its implementation.
- Architectural direction — including module boundaries, the seam-based dependency model, the
  promotion-gate design, and the safety architecture — is set by the maintainer. Proposals are
  welcome through issues; decisions rest with the maintainer.
- The maintainer may decline, defer, or request rework of any contribution at their sole
  discretion, including contributions that are technically sound but outside the project's
  scope or philosophy.
- Review turnaround is on a best-effort basis. This is a single-maintainer project; patience is
  appreciated.

## 8. Licensing of Contributions

Noctis is licensed under the [MIT License](LICENSE). By submitting a contribution, you
agree that your contribution is provided under the same license and that you have the right to
submit it. No separate contributor license agreement is required.

---

Thank you for helping to keep this project rigorous. Disciplined contributions — well-scoped,
well-tested, and honest about their limitations — are what make a quantitative codebase
trustworthy.
