# MEMORY

> Noctis's own long-term memory. Loaded at start of every run, appended during
> research, reorganized and compacted at each market close. Kept small and useful — four
> sections, pruned regularly. Not application code; safe to hand-edit.
>
> This live file is local to your workspace and never lands in git. It was seeded on
> first run from the committed `MEMORY.seed.md` — curated starting lessons so a fresh
> install doesn't re-learn them the expensive way.

## Champions

_(none yet.)_

## Learnings

- **1-minute bars are hostile to strategy evaluation** — round-trip costs dominate, causing all long-only strategies to flatten near 0.0 regardless of thesis quality
- **Prior rejections are unreliable across timeframes** — ideas dismissed at 1-min bars should not be treated as genuinely dead
- **Choose timeframe from cost arithmetic first** — the per-trade move must clear the round-trip cost before a thesis is even testable
- **Re-test on the appropriate timeframe** — only conclude a strategy is viable/dead after testing on timeframes where the cost structure allows profitability
- **One continuous paper account** — TRADING carries equity AND open positions across sessions; the daily loss limit anchors to the day's carried equity; carried positions hold (not flatten) at the next open; a corrupt account file refuses to trade — recover with `noctis account --reset`; champion turnover never resets the account

## Rejected ideas

_(none yet.)_

## Index / changelog

- 2026-07-15 — Seed memory: curated starting lessons for a fresh workspace.
