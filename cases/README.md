# The `cases/` folder — the benchmark corpus, one ask per file

This folder is the eval layer's **input**: the cases the benchmark measures the engine's LLM
judgment sites over. It follows the same input/output contract as everything else committed in
this repo (see AGENTS.md, "Where state lives") — cases are reviewable files a human curates, not
something the engine writes, so they live here rather than under `workspace/`. Everything a
benchmark *produces* lands in a run's workspace tree; nothing in here is ever machine-edited.

```
cases/
├─ README.md              # this file — the format, and what it may never say   (committed)
└─ <site_id>/             # one directory per declared site: coder, formulate,
   │                      # decide, discover, distill
   └─ <case_id>.yaml      # one case per file; the file stem IS the case id   (LOCAL, gitignored)
```

**Committed vs local.** The repo ships only this scaffold. Every case file under `cases/<site_id>/`
is **gitignored** by default (`cases/*` with this README re-included, the deny-by-default allowlist
shape `mandate/` uses), because a mined corpus is harvested from an operator's own runs and carries
their briefs, their symbols and their run ids. A corpus that has been reviewed and is meant to ship
to every user is re-included by name in `.gitignore`, one line per directory, the way a new shipped
mandate profile is — deliberately an explicit act, never a default.

## The file format

One case per file, `<case_id>.yaml`, in the directory of the site it exercises. The file stem is
the case id, so the directory itself keeps ids unique and a rename is visible in a diff.

```yaml
site_id: coder                      # required — must match the directory this file sits in
payload:                            # required — the site's input, any non-empty mapping
  brief: "a mean-reversion strategy on liquid large caps"
  symbols: [AAPL, MSFT]
provenance: authored:2026-07-20     # required — one of the two forms below
tags: [seed, reversion]             # optional — free-form labels, reported over
difficulty:                         # optional — the axes the corpus stratifies on
  reasoning: hard
  novelty: low
split: holdout                      # optional — pins this case; normally left out
```

Validation is strict and one-pass: an unknown key, a missing required key, a malformed provenance
and a bad tag list are all reported in a single refusal naming the file, so a broken case is one
edit rather than a fix-one-rerun loop.

**`provenance` has exactly two forms.** `mined:<run_id>` for a case harvested from a real run
(the run id that run minted, e.g. `mined:20260720T144233Z-a3f9c1`) or
`authored:<YYYY-MM-DD>` for one a person wrote on a date. Between them the corpus can always answer
where it came from and how old it is — the first question asked when a benchmark number surprises
somebody.

**A case carries no expected output, and the loader refuses one by name.** `expected`,
`expected_output`, `answer`, `gold`, `solution`, `ground_truth` and their relatives are rejected
rather than ignored. The expectation is the site's own **oracle** — the coder's fresh-subprocess
write gate, a scorer over a typed emit — never a reference answer parked in a corpus file. A stored
answer is correct only until production improves, and a corpus that has rotted into disagreement
with the thing it measures reports staleness as regression. If you find yourself wanting to write
the answer down, the missing piece is a scorer, not a field.

## Tuning and holdout

The split is **not** a curation decision and is normally absent from these files. A `Corpus`
(`src/noctis/eval/corpus.py`) computes it once, deterministically, from the cases themselves:
stratify by the `difficulty` axes, order each stratum by the hash of the case id, and cut so ~30%
of every stratum is holdout (rounded half-up; a stratum of one goes to tuning). The same cases
therefore always yield the same halves — in any file order, in any process, on any machine — and
a case that *does* declare `split:` keeps what it declares and counts toward its stratum's quota.

Tuning is what a prompt or harness change may be justified on. Holdout is what that justification
is confirmed against, and is never looked at while iterating. That is the promotion gates'
out-of-sample discipline applied to prompts, and it only means anything because nothing re-deals it.

A corpus also carries a content hash — `sha256` over a canonical rendering of every case, so
editing any case moves it and touching nothing reproduces it — which a benchmark record folds into
its comparable key. Two numbers computed over different corpora were never comparable, and the
hash is what says so out loud.
