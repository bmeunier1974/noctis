# The `mandate/` folder — your input surface for the research agent

This folder is where a human steers the research agent. It is skill-shaped: an entry file
(`MANDATE.md`), a library of ready-made personalities (`profiles/`), and optional supporting
notes (`references/`). One config selector, `research.mandate`, chooses which mandate governs
a run. See `docs/research.md` ("Mandates + a growing universe") for the full design.

```
mandate/
├─ README.md              # this file                                    (committed)
├─ MANDATE.md.example     # balanced Sortino swing brief + the whole overlay surface,
│                         # commented out; copy over MANDATE.md to use            (committed)
├─ MANDATE.md             # YOUR own input — edit the prose to steer research    (LOCAL, gitignored)
├─ tune-first.md          # conduct mandate: tune/decide the existing library     (committed)
├─ profiles/              # five shipped personalities, metric-only by design (see `auto` below)
│  ├─ aggressive.md       → promotion.metric: total_return               (committed)
│  ├─ conservative.md     → promotion.metric: sharpe                     (committed)
│  ├─ long-term.md        → promotion.metric: sharpe                     (committed)
│  ├─ short-term.md       → promotion.metric: sortino                    (committed)
│  ├─ sector-specialist.md→ promotion.metric: sharpe                     (committed)
│  └─ <your-name>.md      # custom personalities you author              (LOCAL, gitignored)
└─ references/            # small supporting notes a mandate can pull in
   ├─ example-watchlist.md                                               (committed)
   ├─ high-vol-momentum-brief.md                                         (committed)
   └─ <your-notes>.md     # your own reference notes                     (LOCAL, gitignored)
```

**Committed vs local.** The repo ships only the *scaffold* — this README, the `.example`
template, `tune-first.md`, the five shipped profiles, and two reference examples. Your own
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
config:                       # run-shaping settings this personality binds (see below)
  promotion:
    metric: total_return      # the risk dial every candidate is scored on
  research:
    model: ollama_chat/noctis-qwen3:14b   # …and anything else on the allowed list
symbols:                      # optional: tickers this mandate wants LOOKED AT
  - SMR                       # they join the session's research focus set (the prompt's
  - CCJ                       # market digest + holdout candidate pool) — a search prior,
references:                   # never a gate change; the focus cap is research.focus_size
  - references/example-watchlist.md
---
Your prose goes here.
```

`symbols:` and `config: universe:` are **two different things**, deliberately: `symbols:` says
what to *look at* (a search prior that joins the research focus set), `universe:` says what is
*traded* and what the research panel is drawn from. Both normalize identically (upper-case,
de-duped, first-mention order); only the roster must be long enough to fill the panel.

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

## The overlay rule: the run is yours, the arena is not

A mandate configures the whole **run** it steers — which model thinks, what one session may
spend, how big its prompt gets, which names it starts from, and `promotion.metric`, the risk
dial. It never touches the **arena**: the safety mode, the fill costs, the promotion
thresholds, the fit-set/symbol-holdout geometry, the output paths, the secrets stay
`config.yaml`'s alone.

- **Allowed (35 knobs, six groups):** the model seam, the spend ceilings, the search shape
  (`promotion.metric`, `focus_size`, the tuning penalty, the draft TTL, the distill cadence),
  data acquisition (`history_days`, `auto_backfill`), housekeeping, and the seed `universe`.
- **Clamped (2), one direction only:** `research.min_trials` may only be **raised** (a mandate
  may demand more evidence per verdict, never less) and `data.budget_usd` may only be
  **lowered** (spend less of your vendor money, never more). Equal to the configured value is
  fine; the wrong direction is fatal, and the message names the number it tried to cross.
  Nothing is silently clipped.
- **Refused (49): fatal at startup, with the reason printed.** A refused, unknown, or invalid
  key stops the run before it starts and lists *every* problem in one error. It is never a
  warning you find three days into a run.

Two more rules worth knowing: a mandate-set `universe` must name at least
`research.fit_set_size + research.symbol_holdout_size` symbols (8 as shipped) — too few and the
symbol-holdout gate would go inert instead of being cleared, so the run stops. And the whole
allowed/clamped surface ships **commented out** in `MANDATE.md.example`: uncomment what your
personality needs. The authoritative tables, with a justification per group, live in
`src/noctis/config/overlay.py`; the operator-facing reference is
[docs/configuration.md](../docs/configuration.md#the-mandate-overlay). Widening the surface is
an owner-gated edit to that module, not something a mandate author can reach.

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
ships two inert examples that no shipped mandate wires in: `example-watchlist.md` and
`high-vol-momentum-brief.md` — the high-volatility momentum brief this system once ran under,
kept here as documentation after the balanced Sortino brief became the scaffolded
`MANDATE.md.example`.

## The `auto` caveat (why a profile's overlay is inert)

Under `research.mandate: auto`, the *agent* picks a profile partway through the session —
after the config overlay has already been applied and the toolbox built. So an auto-selected
profile's `config:` block does **not** take effect, whatever it declares: **auto sessions
always run on the base `config.yaml` configuration**, and score on its metric. That keeps every
auto session comparable on one yardstick. Startup logs one warning naming any profile whose
keys would be lost; `promotion.metric` is suppressed from it, because metric-neutrality *is*
the `auto` contract and a warning that fires on every stock install is the noise that gets the
interesting warning ignored.

That is why the **five shipped profiles stay metric-only**: a profile `auto` may pick must
never declare a knob that would be silently inert on the default config — and a second key in
a shipped profile would make every default install warn. Your own profiles are free to bind the
whole allowed surface; just **pin them** (`research.mandate: my-profile`, or `--mandate
my-profile` for one session) so the overlay actually applies.

## Precedence

From lowest to highest priority:

1. Built-in defaults, then `config.yaml`, then `.env`, then the environment — the ordinary
   settings chain ([docs/configuration.md](../docs/configuration.md)).
2. The **mandate overlay**, applied on top of all of it. For the overlaid subset this inverts
   the usual "environment beats the YAML file" rule, deliberately: a mandate is a per-run
   selection, not ambient environment. Secrets and `ALLOW_LIVE` are refused, so the environment
   stays their only source.
3. `--mandate <name>` or `--directive "<text>"` on the CLI selects *which* mandate overlays,
   for one session. (`--directive` and `--mandate` together is a usage error.)
4. `--metric <m>` (on `noctis research`) and `--time-limit-hours` (on `noctis run`) — explicit
   one-off flags applied **after** the overlay, so they always win over a mandate.

`noctis status` prints the resolved end of that chain: the active mandate and every `k=v`
override it applied, with all the values above them post-overlay. `run` and `research` echo the
same lines at kickoff.
