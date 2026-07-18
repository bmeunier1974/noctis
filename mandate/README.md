# The `mandate/` folder — your input surface for the research agent

This folder is where a human steers the research agent. It is skill-shaped: an entry file
(`MANDATE.md`), a library of ready-made personalities (`profiles/`), and optional supporting
notes (`references/`). One config selector, `research.mandate`, chooses which mandate governs
a run. See `docs/operator-mandate.md` for the full design.

```
mandate/
├─ README.md              # this file                                    (committed)
├─ MANDATE.md.example     # preserved known-good brief; copy over MANDATE.md to use (committed)
├─ MANDATE.md             # YOUR own input — edit the prose to steer research    (LOCAL, gitignored)
├─ tune-first.md          # conduct mandate: tune/decide the existing library     (committed)
├─ profiles/              # five shipped personalities; pick one or copy one to start
│  ├─ aggressive.md       → promotion.metric: total_return               (committed)
│  ├─ conservative.md     → promotion.metric: sharpe                     (committed)
│  ├─ long-term.md        → promotion.metric: sharpe                     (committed)
│  ├─ short-term.md       → promotion.metric: sortino                    (committed)
│  ├─ sector-specialist.md→ promotion.metric: sharpe                     (committed)
│  └─ <your-name>.md      # custom personalities you author              (LOCAL, gitignored)
└─ references/            # small supporting notes a mandate can pull in
   ├─ example-watchlist.md                                               (committed)
   └─ <your-notes>.md     # your own reference notes                     (LOCAL, gitignored)
```

**Committed vs local.** The repo ships only the *scaffold* — this README, the `.example`
template, `tune-first.md`, the five shipped profiles, and one reference example. Your own
`MANDATE.md`, any custom personality (`mandate/<name>.md` or `profiles/<name>.md`), and your
personal `references/` are **gitignored**, so steering the agent never shows up as a repo
change. Start from the template: `cp mandate/MANDATE.md.example mandate/MANDATE.md`.

## Choosing what governs a run: `research.mandate`

Set `research.mandate` in `config.yaml` to one of:

| Value            | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `MANDATE`        | Use your own `mandate/MANDATE.md` (copy it from `MANDATE.md.example` first). |
| a profile name   | Use `mandate/profiles/<name>.md`, e.g. `aggressive`, `conservative`.    |
| `auto`           | Let the agent pick a profile each session (see the caveat below).       |
| `null`           | No mandate — the agent runs unconstrained.                              |

Names are flat (no path separators). A name is looked up under `profiles/` first, then at
the top level of `mandate/`. A selector that doesn't resolve (typo'd profile, missing file)
is fatal at startup — the run exits non-zero rather than silently un-steering.

## Authoring `MANDATE.md`

`MANDATE.md` is your first-person brief: tell the agent what kind of trader you want the
system to be — risk appetite, horizon, which names to favour, what to avoid. Write as much
or as little as you like. The prose is injected into the agent's OPERATOR MANDATE block.

Structure of a mandate file (this is also the shape of every profile):

```markdown
---
summary: One line describing this mandate (shown in the auto menu and kickoff echo).
config:
  promotion:
    metric: total_return      # the ONLY key a mandate may override
symbols:                      # optional: tickers this mandate wants researched
  - SMR                       # they join the session's research focus set (the prompt's
  - CCJ                       # market digest + holdout candidate pool) — a search prior,
references:                   # never a gate change; the focus cap is research.focus_size
  - references/example-watchlist.md
---
Your prose goes here.
```

The front-matter must be the very first bytes of the file (a `---` … `---` fence). Any
HTML comments or prose go *below* it. The shipped `MANDATE.md` keeps its how-to header as an
HTML comment at the top of the body for that reason.

**Empty MANDATE.md → unconstrained.** The loader strips HTML comments and whitespace before
its empty check, so a `MANDATE.md` that is only comments resolves to "no mandate." To hand
the wheel to the profiles instead, clear the prose *and* set `research.mandate: auto`.

To make your own personality, copy a profile into `mandate/<name>.md` (or `profiles/`), edit
it, and point `research.mandate` at its name. Custom personality files are gitignored (only the
five shipped profiles are committed), so they stay on your machine.

## `tune-first` — a conduct mandate, not a personality

`mandate/tune-first.md` steers a session's *conduct* rather than its taste: tune and decide
the EXISTING library first, author new files only after a completed tune-to-verdict cycle.
It exists for small local backends that fixate on authoring `write_strategy` submissions
which (correctly) fail the write gate and never reach a backtest — steering them to the
existing library is the mandate system doing its job as a search prior. It lives at the top
level rather than in `profiles/` on purpose, so the `auto` menu stays a catalog of trader
personalities. Its metric overlay mirrors `MANDATE.md` (sortino) so champion comparisons
stay like-for-like; keep the two in lockstep if you change one.

## The metric-only overlay rule

A mandate carries its risk dial with it: the `config:` block may set **`promotion.metric`
and nothing else** (`sharpe` | `sortino` | `total_return`). Every other key — gate
thresholds, `mode`, `risk.*`, budgets, `state_dir`, session budgets — is refused with a
warning and ignored. This is deliberate: a mandate *steers* what the agent looks for and how
risk is scored; it never loosens a gate, the exhaustion rule, or the honesty contract. Those
still bind. Widening the allowlist is an owner-gated change to a constant in
`noctis/research/mandate.py`, not something a mandate author can reach.

## Adding references (keep them small — links over embeds)

A mandate can pull in supporting notes from `references/`, two ways (they merge and
de-duplicate):

- **Front-matter list:**
  ```yaml
  references:
    - references/example-watchlist.md
  ```
- **Inline wikilink** in the prose: `[[references/example-watchlist.md]]` (the `.md` is
  optional).

References are confined to this folder (no `..` escapes, no absolute paths) and are **capped
small** (~2 KB per file, ~6 KB total). A reference that wants to be bigger is a signal it
should be a **link the agent follows with web_search**, not an embed — every kilobyte of
loaded reference prose is context the agent can't spend on its own reasoning. `references/`
ships with one worked example, `example-watchlist.md`, that no shipped mandate wires in.

## The `auto` caveat (why a profile's overlay can look inert)

Under `research.mandate: auto`, the *agent* picks a profile partway through the session —
after the config overlay has already been applied and the toolbox built. So an auto-selected
profile's `config:` overlay does **not** take effect: **auto sessions always score on the
base `config.yaml` metric.** This keeps every auto session comparable on one yardstick. If
you want a specific profile's metric overlay, **pin that profile** (`research.mandate:
aggressive`) instead of using `auto`.

## Precedence

From lowest to highest priority:

1. Base `config.yaml` / defaults.
2. `research.mandate` (config) — the persistent selector.
3. `--mandate <name>` or `--directive "<text>"` on the CLI — a one-session override.
   (`--directive` and `--mandate` together is a usage error.)
4. `--metric <m>` on `noctis research` — an explicit one-off flag applied **after** the
   overlay, so it always wins over a mandate's `promotion.metric`.
