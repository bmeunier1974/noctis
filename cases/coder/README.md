# The coder corpus — one authoring job per file

This is the benchmark corpus for the **coder** site: the LLM call that turns a research brief into
one complete strategy file. A case here *is* a brief — the same fields
`src/noctis/research/author.py`'s `StrategyBrief` carries, no more and no less — because the moment
a case carries a field production does not, the benchmark stops measuring the coder and starts
measuring a translation layer nobody reviews.

Read [`../README.md`](../README.md) first for the generic case format; this file covers what the
coder site adds: the bucket directories, the brief payload, the seven difficulty axes, and how the
tuning/holdout split gets frozen into the files.

## Layout: one directory per bucket

```
cases/coder/
├─ README.md    this file                                                     (committed)
├─ edge/        deliberately hard briefs — one difficulty axis at its extreme  (committed)
├─ canary/      briefs so plain a failure indicts the harness, not the model   (committed)
├─ field/       harvested from an operator's own runs                (LOCAL, gitignored)
└─ replay/      an operator's own known gate rejections              (LOCAL, gitignored)
```

The generic provider (`src/noctis/eval/case_provider.py`) reads `<cases_root>/<site_id>/*.yaml`,
flat. The coder corpus is **not** flat, and `src/noctis/eval/coder_corpus.py` carries its own
loader (`CoderCaseProvider`) that walks `cases/coder/<bucket>/*.yaml` and hands the same generic
`Corpus` what it finds. The reason is `.gitignore`: ignore rules match *paths*, so with bucket
directories "committed vs. local" is four allowlist lines that never change again — the
deny-by-default shape `mandate/` already uses — where a flat layout would make it a per-file
decision a curator can typo into a leaked corpus. The bucket still rides in the case's own
`bucket:` tag, and the two must agree: a file in `canary/` tagged `bucket:field` is refused, because
a benchmark reports the tag.

Everything else the layout can get wrong is refused the same way, since all of it means "a case
nobody measures": a `.yml` near-miss, a case file sitting beside the buckets rather than in one, and
a sub-directory that is not one of the four declared buckets.

## The four buckets

| bucket | what it is for |
| --- | --- |
| `field` | harvested from a real run — production actually asked the coder this. Local. |
| `replay` | a curated re-run of a brief with a known gate rejection, kept so a fix stays fixed. Local. |
| `edge` | a deliberately hard brief that pushes one difficulty axis to its extreme. Committed. |
| `canary` | a brief so plain that a failure indicts the harness rather than the model. Committed. |

`field` and `replay` are mined from an operator's own runs — their briefs, their symbols, their run
ids — so they stay theirs. `edge` and `canary` are authored by hand and reviewed in a pull request,
which is what makes them safe to ship to every user.

## The file format

One case per file, `<case_id>.yaml`, the file stem being the case id:

```yaml
site_id: coder                       # required — always 'coder' under this directory
payload:                             # required — the brief, field for field
  thesis: "why this edge should exist"                    # required
  entry_exit: "when to be long/short/flat"                # required
  param_space: "what to tune, and over what ranges"       # required
  scenarios: "the known-outcome tapes the file must satisfy"   # required
  reference: sma_crossover           # optional — a library strategy to adapt
  style: momentum                    # optional
  symbols: [AAPL, MSFT]              # optional
  scenario_spec:                     # optional — the FIXED oracle, in FORMULATE's own vocabulary
    scenarios:
      - name: rally
        legs: [{kind: trend, bars: 60, pct: 0.05}]
        behavior: enter_long_during_leg
        leg: 0
provenance: authored:2026-08-02      # required — 'mined:<run_id>' or 'authored:<YYYY-MM-DD>'
tags: [bucket:canary, adapts:sma_crossover, rule:exponential-average]
difficulty: {...}                    # required — all seven axes, see below
split: tuning                        # stamped by the authoring tool, never edited by hand
```

The four required brief fields are the division-of-labor guard: a case missing one would ask the
coder to supply research judgment the brief owes it. `scenario_spec` is optional and is parsed by
the episodic driver's *own* FORMULATE emit parser, so a case's fixed oracle is one a real FORMULATE
could have emitted or it is refused here.

**A case carries no expected output, and the loader refuses one by name** — `expected_source`,
`reference_solution`, `model_answer` and their relatives, on top of the generic set. The
fresh-subprocess write gate is this site's oracle. A stored answer would be correct only until the
strategy contract improved.

**`provenance` has exactly two forms**: `mined:<run_id>` for a case harvested from a real run, or
`authored:<YYYY-MM-DD>` for one a person wrote on a date.

## The seven difficulty axes

Every case is labelled on **all seven**, each from a closed value set (a partly-labelled case would
land in a stratum of its own and quietly skew the split). Every level names a distinction the
production authoring path already makes:

| axis | levels, easiest first | what it measures |
| --- | --- | --- |
| `composition_mode` | `scratch`, `reference`, `revision` | author from nothing, adapt a named library strategy, or revise the file already carrying the target name |
| `oracle_mode` | `authored`, `fixed_spec` | the coder writes its own `scenarios()`, or the gate stamps a fixed spec and rejects any it writes |
| `warmup_arithmetic` | `single`, `composed`, `higher_timeframe` | one lookback, several composed into the longest, or a warmup multiplied by the bars per decision bar |
| `state_complexity` | `stateless`, `rolling`, `latched` | nothing kept between bars, an incremental rolling window, or rolling state plus a position latch |
| `no_trade_tape` | `trivial`, `falsified`, `scale_free` | a tape that cannot trigger, one that must falsify the level condition, or a scale-free percentile rule no amplitude can silence |
| `param_space_breadth` | `narrow`, `moderate`, `broad` | one or two tuned parameters, three or four, or five and up |
| `api_surface` | `bars_only`, `indicators`, `exits` | bars and targets alone, plus the indicator surface, plus protective `ExitRules` |

A label that disagrees with its payload is refused: `composition_mode: reference` with no
`reference` in the brief, `oracle_mode: fixed_spec` with no `scenario_spec`, `oracle_mode:
authored` that ships one anyway.

## The split is stamped into the files, and reading never re-deals it

The generic `Corpus` deals a stratified ~70/30 tuning/holdout split deterministically — but
*deterministic* is not *frozen*: the assignment depends on the whole stratum, so adding a case can
move an unassigned one, and a corpus that grows a case a week would keep re-dealing. So the deal
happens **once, at authoring time**, and the answer is written into each file's `split:` field:

```bash
uv run python -c "from pathlib import Path; \
  from noctis.eval.coder_corpus import stamp_splits; print(stamp_splits(Path('cases')))"
```

`stamp_splits` is idempotent — a file that already declares a `split:` is left byte for byte alone,
because frozen is frozen — so re-running it after adding cases stamps only the new ones. Every
loader then honours what the file says (`Case.assigned_to` refuses reassignment, and the corpus
counts frozen cases toward their stratum's quota), which is why a committed case's half is visible
in the diff and can only move when a reviewer moves it. **Never hand-edit a stamped `split:`.**

The stratum key is `(bucket, the seven difficulty axes)`: `deal_splits` partitions by bucket first
and runs the generic rule inside each partition — stratify on the difficulty mapping, order each
stratum by `sha256(case_id)`, cut so ~30% of it is holdout (rounded half-up; a stratum of one goes
to tuning). Two cases with identical axes in different buckets are not the same kind of case — a
canary and a field case share nothing but their labels — and folding a synthetic `bucket` axis into
the difficulty mapping was rejected because the seven axes are a closed set the schema validates.

Tuning is what a prompt or harness change may be justified on. Holdout is what that justification is
confirmed against, and is never looked at while iterating.

## The canary bucket

The six committed canaries are one-rule adaptations of the simplest committed seed,
`strategies/sma_crossover.py`: exponential instead of simple averages, a slower slow leg, the close
against a single average, the comparison inverted, a symmetric short leg, and averages over highs
rather than closes. Each states its whole rule in plain English (so `composition_mode` is honestly
`scratch` — the brief names no reference), and records its lineage in tags: `adapts:sma_crossover`
plus one `rule:<change>` tag naming the single thing that differs from the seed.

They sit at the easy end of every axis — no canary carries the hardest level of any of the seven —
because that is the whole point: if a canary fails, look at the harness, the extras, the prompt
assembly or the write gate before looking at the model.
