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
| `episodic` | the episodic driver's per-stage system texts and emit contracts | `research/driver.py`, `research/digests.py` |
| `ideation` | the seeded-idea prompt, web search included | `research/ideation.py` |

`research/digests.py` renders the facts (market digest, library index, champion board, memory
block) that four of those prompts embed, so it is listed under each of them: editing it moves all
four hashes at once. That over-partitions on purpose — a site whose assembled text changed must
never keep its old identity.

---

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
