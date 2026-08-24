# Prompt changelog

What each LLM call site's prompt has been told to do, and when it changed. This is the human half
of the prompt-asset fingerprint: `prompt_fingerprint.json` (repo root) holds one content hash per
site, and every hash in it reads back to an entry here. See
[development.md → The prompt fingerprint ratchet](development.md#the-prompt-fingerprint-ratchet)
for the rule and the commands.

**Newest entry first.** One entry per change, heading first, `sites:` naming every call site whose
assets moved — that list is read by `noctis.observability.prompt_ratchet`, so it has to be on the
heading line and spelled exactly as the site is named in `prompt_id.SITE_ASSETS`:

```text
## <YYYY-MM-DD> — sites: <site>[, <site>…]

<what changed, and why — one short paragraph or a few bullets>
```

The sites, and the assets each one's hash covers:

| Site | What it is | Assets |
|---|---|---|
| `author` | the coder site: the authoring brief and the contract sheet it must satisfy | `research/author.py`, `research/contract_sheet.py`, `research/digests.py` |
| `briefings` | the rendered briefings that are the episodic stages' user turns | `research/briefings.py`, `research/digests.py` |
| `conversation` | the conversation loop's system prompt | `research/prompt.py`, `research/digests.py` |
| `distill` | the memory distiller's one summarization prompt | `research/distill.py` |
| `episodic` | the episodic driver's per-stage system texts and emit contracts | `research/driver.py`, `strategies/scenario_spec.py`, `research/digests.py` |
| `ideation` | the seeded-idea prompt, web search included | `research/ideation.py` |

`research/digests.py` renders the facts (market digest, library index, champion board, memory
block) that four of those prompts embed, so it is listed under each of them: editing it moves all
four hashes at once. That over-partitions on purpose — a site whose assembled text changed must
never keep its old identity.

---

## 2026-08-24 — sites: episodic

Epic #326 story #329: bookkeeping refactor in the episodic driver — `summary.undecided` now derives from the session ledger (`undecided_names()`), and `_record_verdict` states the write order (journal fact first, ledger narrative second). No wording change, no emit-contract change. Story #331 adds one docstring paragraph to the same file: the driver states the epic's ownership rule — it is the session ledger's one writer, the toolbox is the experiment journal's, and the `thesis` double write is deliberate. Documentation only; no stage system text, briefing or emit contract moved.

## 2026-08-23 — sites: episodic

Structural move, **one named wording change**: epic #319 gives `noctis.strategies.scenario_spec`
every crossing of the FORMULATE scenario spec. The model-dialect parse — `spec_from_payload`, the
`PARSE_WARM` it is compiled at, and every refusal sentence a malformed spec earns (#321) — and the
JSON Schema the FORMULATE emit contract advertises for `scenario_spec` move out of
`research/driver.py` into the strategy-layer module that owns the vocabulary they describe. The
driver keeps the boundary and nothing else: read the emitted field, parse, compile, translate the
strategy layer's `SpecError` into the schema-misfire currency the episode runner already reads. The
schema the model receives is **dict-equal** across the move: the golden
`tests/fixtures/scenario_spec/formulate_scenario_spec_schema.json`, captured from the pre-epic
commit `262034f`, is the checked claim (#322), and the `kind`/`behavior` enums it advertises are
now read off the vocabulary itself (`LEG_KINDS`, taken from the compiler's own builders, and
`Behavior`) instead of a second list kept beside it in the driver. `scenario_spec.py` joins this
site's assets, so the description strings the model reads about legs and behaviors can never
drift un-ratcheted again.

The one wording change is D5, and it is a *unification*, not a rewrite: the five suite-shape rules
now have a single spelling in `scenarios.check_suite_shape`, which both the spec path and a
hand-authored `scenarios()` are judged by, so the two refusals whose parentheticals differed
between them now name both dialects — "at least one scenario must demand a directional entry
(enter/hold long/short — long_within/holds_long_through/short_within/holds_short_through)" and "at
least one scenario must be a no-trade tape (never_trade / always_flat())". Those sentences ride
into a FORMULATE corrective, which is why they are declared here rather than left as diagnostics.
Nothing else the model is told moves: the per-stage system texts, the emit contracts and their
field descriptions, the retry hint with its `_SCENARIO_SPEC_EXAMPLE`, and every other refusal
sentence are verbatim, pinned by the driver's own tests and by the schema golden. The hash moves
because the code that assembles those texts moved. The module's own docstring was rewritten
alongside (#324) to name the three crossings it owns — model dialect, carrier, compile — which
moves the hash a second time without changing a word the model reads. This one entry declares the
whole epic.

## 2026-08-22 — sites: author, briefings, conversation, episodic

Seam refactor, **no wording change**: the renderers that assemble these prompts were re-typed
against `noctis.research.surface.ResearchFacts` (epic #255, stories #257-#259) and now read the
session's *derived facts* — the champion board, the library index, the memory tail, the lake
inventory, the data budget, one candidate's journaled evidence — instead of reaching **through**
the research toolbox into whichever collaborator happened to hold the answer (`toolbox.journal`,
`toolbox.registry.capacity`, `toolbox.lake.preflight.budget_usd`), often behind a `getattr` probe
that quietly invented a fact when the reach missed. Three builders moved with that: the DECIDE
evidence block is now `journal.evidence_block` (one builder, beside the record schema it reads),
the lake-inventory builder moved out of `research/digests.py` onto the toolbox, and the one
tolerant read left in the codebase — a lake with no cost preflight — answers `None` once, from
`ResearchToolbox.data_budget()`, rather than being probed for at each call site.

Not one asset's text moved. `tests/test_prompt_goldens.py` pins the rendered FORMULATE, DECIDE and
DISCOVER briefings and the conversation system prompt (at both `prefix_trim` values) by length +
SHA-256, and all five fingerprints are unchanged across the epic — that is the proof of byte
identity, on the same principle as the previous entry's author golden. The hashes move because the
composition *code* moved: `research/briefings.py` (`briefings`), `research/prompt.py`
(`conversation`), and the shared fact-renderer `research/digests.py`, which is listed under
`author` and `episodic` as well — exactly the over-partition this ratchet is designed for.
`episodic`'s own asset `research/driver.py` carries the same move (#259): the episodic driver holds
a `noctis.research.surface.Toolbox` instead of an `Any`, and its ten `getattr(toolbox, …, default)`
probes are gone — readiness is `toolbox.symbol_ready`, the FORMULATE class check takes
`toolbox.class_exhausted` as a callable, sweep sizing reads `toolbox.limits`, and session end takes
one `toolbox.session_counters()` snapshot instead of six live reads. No wording change there
either: the per-stage system texts, emit contracts and every ledgered/emitted line are
byte-identical, pinned by the goldens and by the driver's own ledger assertions. This one entry
declares the whole epic's move.

Amended for #261, the epic's last story and the same kind of move: the eval layer's frozen DECIDE
case now reads the rendered trial cap straight from `research/journal.py`, the module that owns the
evidence block, so the `_TOP_TRIALS` alias `research/briefings.py` was keeping for it is deleted.
That is the only line of any asset this amendment moves, and it is a name nothing renders.

## 2026-08-02 — sites: author

Ablation seam widening, **no wording change**: the authoring engine's five prompt pieces (contract
sheet, `TEMPLATE.py`, worked example, feasibility rules, retry-hint enrichment) became constructor
parameters the benchmark layer can dial off one at a time (#223), and each completion can now be
bounded by an optional per-attempt timeout. Not one asset's text moved, and the default
composition — every dial at its shipped value, which is the only composition any production
construction site builds — is byte-identical to the previous entry's: `tests/test_strategy_author.py`
locks it with a golden length + SHA-256 of the system prompt a default author sends. The `author`
hash moves because the composition *code* in `research/author.py` moved, which is exactly the
over-partition this ratchet is designed for. The timeout rides the same file (a bounded wait around
the client seam) and tells the coder nothing at all; a completion that never answers now fails its
own attempt instead of the job.

## 2026-08-01 — sites: author, briefings, conversation, distill, episodic, ideation

Baseline. The first committed prompt-asset fingerprint (#183): every site's hash records the
prompt text as it stands today, with no change to any of them. From here on a prompt edit is a
visible event — the ratchet fails until the site is named in a new entry above this one, and
`--write` refuses to record a change the changelog does not declare.
