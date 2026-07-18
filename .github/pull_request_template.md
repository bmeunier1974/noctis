<!--
  Thanks for contributing to Noctis. Please read CONTRIBUTING.md first — PRs that don't
  follow the process are returned for revision before review, regardless of merit.
  Every PR starts from an issue: reference it below with `Closes #NN`.
-->

## Description

<!-- What changed and why. Link the originating issue: Closes #NN -->

## Type of Change

- [ ] Feature
- [ ] Bug fix
- [ ] Research improvement
- [ ] Documentation

## Validation

- **Tests:** <!-- new/updated tests and how you ran them -->
- **Benchmarks / scorecard:** <!-- evidence for any algorithm change (see CONTRIBUTING → Research contributions) -->
- **Research impact:** <!-- effect on strategies, promotion gates, or holdouts — normally "none" -->

## Checklist

- [ ] `uv run pytest` passes on a minimal (`uv sync`, core + dev) install
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] `uv run mypy` is clean
- [ ] Documentation is updated for any user-facing or behavioral change
- [ ] **No promotion-gate or safety-gate changes** — or explicitly justified and discussed in the linked issue first
- [ ] No secrets, credentials, vendor market data, or `state/` / `data_lake/` / `reports/` files in the diff

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full rules and the non-negotiable project
invariants.
