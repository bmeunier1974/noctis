# Engine changelog

What moved in the engine's behavioural surface, and — for the moves that changed no behaviour —
why that claim is true. This is the human half of the engine fingerprint: `engine_fingerprint.json`
(repo root) holds one content digest per component, and a digest that moved without an
`ENGINE_VERSION` bump reads back to an entry here. See
[development.md → The engine fingerprint ratchet](development.md#the-engine-fingerprint-ratchet)
for the rule and the commands.

**Newest entry first.** One entry per declared no-op, heading first, `components:` naming every
component whose digest moved and `behaviour: unchanged` marking the claim — both clauses are read
by `noctis.observability.engine_ratchet`, so they have to be on the heading line and the components
spelled exactly as they are named in `engine_id.COMPONENT_PATHS`:

```text
## <YYYY-MM-DD> — components: <component>[, <component>…] — behaviour: unchanged

<what moved, and why no result can differ — one short paragraph or a few bullets>
```

An entry with `components:` but no `behaviour: unchanged` declares nothing to the ratchet. The page
is the arbiter's human history, so it may also narrate a version bump — but a bump is declared by
`ENGINE_VERSION` itself, and the check never demands an entry for one.

The components, the tier each sits on, and the files its digest covers:

| Component | Tier | What it decides | Files |
|---|---|---|---|
| `gates` | arbiter | what passes | `src/noctis/champions/promotion.py`, `src/noctis/backtest/scorecard.py`, `src/noctis/backtest/splits.py` |
| `backtest` | arbiter | what a number means | `src/noctis/backtest/pipeline.py`, `src/noctis/backtest/validate.py`, `src/noctis/backtest/candidate.py`, `src/noctis/backtest/prefilter.py`, `src/noctis/broker/seam.py`, `src/noctis/broker/paper.py` |
| `research` | searcher | how candidates are found | `src/noctis/research/agent.py`, `src/noctis/research/driver.py`, `src/noctis/research/tools.py`, `src/noctis/research/episode.py`, `src/noctis/research/sweep.py`, `src/noctis/research/usage.py` |
| `prompts` | searcher | what the model is told | `src/noctis/research/prompt.py`, `src/noctis/research/briefings.py`, `src/noctis/research/contract_sheet.py`, `src/noctis/research/digests.py`, `src/noctis/research/ideation.py` |
| `profiles` | searcher | the shipped steering personalities | `mandate/MANDATE.md.example`, `mandate/tune-first.md`, `mandate/profiles/aggressive.md`, `mandate/profiles/conservative.md`, `mandate/profiles/long-term.md`, `mandate/profiles/sector-specialist.md`, `mandate/profiles/short-term.md` |
| `seeds` | searcher | the read-only library every run starts from | `strategies/TEMPLATE.py`, `strategies/donchian_breakout.py`, `strategies/rsi_meanrev.py`, `strategies/sma_crossover.py` |
| `memory_seed` | searcher | the starting condition of every run's memory | `MEMORY.seed.md` |
| `schema` | searcher | what is recorded | `src/noctis/reporting/schema.py` |

Only the two **arbiter** components can be declared here, because they are the only ones a drift
blocks on: searcher drift already warns and passes, so an entry naming one of those lifts nothing
and the ratchet reads it as inert.

**What qualifies as a no-op.** Mechanical edits only — a change a reader can verify changed no
behaviour by reading the diff:

- a rename (a symbol, a file, a parameter kept in the same position);
- an import path, including deleting a pass-through module and repointing its importers;
- a docstring or a comment;
- a type annotation, a `TYPE_CHECKING` guard, an overload;
- formatting (line wrapping, quote style, a trailing comma).

**What never does.** Anything that reaches a decision. If the diff touches a branch, a constant, a
default, a formula, a threshold, an ordering, or an error a gate reads, it is a behaviour change
and the declaration is an `ENGINE_VERSION` bump — not an entry here. "It should not matter" is not
the bar; "no result can differ" is.

**The reviewer's bar.** The full local suite passes with its **goldens and scorecard fixtures
unchanged**. A claimed no-op that needs a golden regenerated or a fixture number edited is not
one: the fixture is the evidence, and editing it is how a behaviour change gets through review
looking like a rename. Declare the bump instead.

Note what a declaration does *not* do: the digest still moved, so the resume policy still refuses a
run frozen under the old one (until `--allow-engine-upgrade`) and `comparable_key` still buckets
this engine separately. Over-partitioning is the accepted cost — a wrong claim here can spend a
contributor's time, never pool two engines' numbers.

---

*No entries yet. The first declaration lands with the first mechanical arbiter move.*
