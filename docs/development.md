# Development

Full installation, the optional extras, and the quality gates. For workflow, standards, and
governance see [CONTRIBUTING.md](../.github/CONTRIBUTING.md); for the strategy-file contract see
`strategies/README.md`.

## Installation

Everything — dependencies, groups, extras, and tool configuration — lives in `pyproject.toml`,
resolved to exact versions in `uv.lock`. [`uv`](https://docs.astral.sh/uv/) is the project
standard; a `.python-version` pins the interpreter (3.11). Requires Python ≥ 3.11.

**Running Noctis? Install everything** — this is the standard operator install, and what the
README's getting-started uses:

```bash
uv sync --all-extras               # every seam filled, reproducible from uv.lock
uv run python -m noctis setup      # then the guided first-run wizard
```

`uv sync` reads `uv.lock` and installs the exact locked versions into `.venv`. Run any command
in that environment with `uv run` (e.g. `uv run pytest`) or activate `.venv` first. Use
`uv sync --locked` to fail loudly if `pyproject.toml` and `uv.lock` have drifted (this is what
CI runs).

**Developing the core?** A minimal install exists as an engineering property, not a usage
recommendation: the heavy stacks live behind swappable seams, so a bare `uv sync` (core + dev
tooling) imports, runs bare paper mode, and passes the *entire* test suite with no optional
stack installed. That is what CI runs, and it is what keeps every seam honest — but it is not
a machine that can research or ingest data. Individual extras exist for working on one seam:

```bash
uv sync                                    # core + dev only (CI / seam work)
uv sync --extra data --extra llm           # databento + yfinance + the LLM seam
uv sync --extra engine --extra research    # nautilus / vectorbt / optuna
```

> [!NOTE]
> On a partial install, a feature whose seam is empty says so and continues — e.g.
> `The 'databento' package is required … continuing without it`. Add the extra named in the
> warning, or just `uv sync --all-extras`. `noctis setup --check` lists what's missing.

## Dependency groups

`dev` is a PEP 735 **dependency-group** — a bare `uv sync` installs it by default. The runtime
seams are **optional extras** (`uv sync --extra <name>`); the core install runs paper-only with
none of them.

| Group / extra | Kind | Packages |
|---|---|---|
| **core** | base | `pydantic`, `typer`, `numpy`, `pandas`, `pyarrow` |
| **dev** | group (default) | `pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit` |
| **engine** | extra | `nautilus_trader` |
| **research** | extra | `vectorbt`, `optuna`, `quantstats-lumi` |
| **data** | extra | `databento`, `yfinance`, `exchange-calendars`, `transitions`, `apscheduler` |
| **llm** | extra | `anthropic`, `litellm` |

## Quality gates

The source lives in a `src/` layout (`src/noctis/`); tests live in `tests/`. Ruff handles
linting and formatting, mypy handles static type-checking, and pre-commit runs all of it on
every commit — each tool is configured in `pyproject.toml`.

```bash
uv run pre-commit install         # one-time: lint/format/type-check on every commit

uv run pytest                     # full test suite
uv run ruff check .               # lint
uv run ruff format .              # format
uv run mypy                       # type-check src/noctis
uv run pre-commit run --all-files # all quality gates at once
```

## AI-assisted development

Noctis is developed in close collaboration with AI coding agents (primarily
[Claude Code](https://claude.com/claude-code)). That is a statement of process, not of
standards: every change — agent-written or human-written — passes the same gates before it
lands: the full test suite, mypy, ruff, and CI on the minimal install. It is the same
philosophy the system applies to its own strategies: provenance does not earn trust,
surviving the gates does.
