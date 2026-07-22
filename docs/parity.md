# Parity harness: conversation vs episodic

The **evidence gate** for flipping `auto` to the episodic loop on small-context backends. Noctis
runs research two ways behind one seam — the **conversation** loop (one long tool-use transcript)
and the **episodic** driver (a deterministic state machine that calls the model only at narrow
judgment points and keeps the cross-strategy story in a session ledger, not a growing chat
context). Both return the same `ResearchSummary`. Before `auto` should prefer episodic (that flip is
a separate change), an operator needs legible proof that episodic is **at least as effective per
session at materially lower spend**. This harness produces that proof: it runs both loops on the
**same model, the same lake fixture, and the same mandate**, then prints a side-by-side comparison.

It is a **dev tool, not a CLI subcommand** — it runs *paid* model sessions, so it lives in
`scripts/` and is the operator's explicit action, never CI. The metric math is pure and tested
(`src/noctis/research/parity.py`, `tests/test_parity.py`); the script (`scripts/parity_harness.py`)
is a thin orchestrator that reuses the same composition-root builders `noctis research` uses.

## Prerequisites

- **A hosted API key and the `[llm]` extra.** The harness spends real tokens on a hosted model, so
  it refuses to start unless a client is buildable for the configured `research.model` (or the
  `--model` override). Configure the `[llm]` extra and the provider's key (`ANTHROPIC_API_KEY` for
  `anthropic/*`, `OPENAI_API_KEY` for `openai/*`) as in [configuration.md](configuration.md).
- **A synced lake fixture.** Both loops research the *same* fixed lake. Sync/ingest the symbols you
  want the run to see first (`noctis data sync` / `noctis data ingest`, see [data.md](data.md)); the
  harness reads whatever `data.lake_dir` points at.
- **One mandate for both loops.** Pass `--mandate <profile>` or an inline `--directive "…"` so both
  loops chase the same target; omit it to use the configured selector. A mandate steers *what* to
  look for — it never loosens a gate.

> [!NOTE]
> The lake fixture (market data) is read-only and identical for both loops. Mutable research state
> (champions, `__tmp/` drafts, experiment journals) is *not* reset between runs, so for a clean
> comparison point the harness at a fresh workspace, or reset champions
> (`noctis champions --reset`) before a run. The harness is a directional dev tool, not a
> statistical benchmark — run a few sessions per loop and read the trend.

## How to run

```bash
# See every flag first (fast, no spend):
uv run python scripts/parity_harness.py --help

# One session per loop (two paid sessions) on the configured model + mandate:
uv run python scripts/parity_harness.py --yes

# Three sessions per loop (six paid sessions) on a named hosted model + a profile mandate,
# streaming each session's tool feed:
uv run python scripts/parity_harness.py \
    --sessions 3 \
    --model anthropic/claude-3-5-haiku \
    --mandate momentum-hunter \
    -v
```

The harness **prints what it is about to spend** — `sessions × 2 loops` — and, on a TTY, asks for
confirmation before it runs (`--yes` skips the prompt for a non-interactive run). It then runs the
conversation loop N times, the episodic loop N times, and prints the comparison.

## Reading the output

Each row aggregates across the N sessions of a loop. A cell reads `n/a` when a loop **cannot
honestly supply** that metric — never a fabricated number.

| Row | What it means |
|---|---|
| **Verdicts / session** *(primary)* | `(promotions + rejections)` per session — the spent promote/reject decisions the gates arbitrated. The effectiveness axis: more real verdicts per session is more research done. |
| **Tokens / verdict** *(decision)* | `tokens_total ÷ verdicts` — judgment-model tokens spent per verdict reached. The spend axis. `n/a` when a loop reached zero verdicts. |
| Validator 1st-attempt % | Of the strategies a session tried to author, the fraction that passed the write gate on the first try. Episodic-only (from the ledger); the conversation loop keeps no ledger, so it reads `n/a`. |
| Promotion-gate reach % | `verdicts ÷ candidates` — the fraction of strategies worked on that reached a gated verdict. |
| Undecided (total) | Strategies authored but never carried to a verdict (archived after the TTL). |

### How tokens are counted (the honesty note)

`tokens_total` is one comparable number for both loops: the four neutral usage fields (input +
output + cache-creation + cache-read) summed across **every completion the loop's own judgment model
made**, retries included. The conversation loop totals its per-round usage; the episodic driver sums
its ledger's per-episode token counts. **Coder-authoring** completions run on a *separate* client
and are excluded from both, so the figure compares like with like. This is exactly why the episodic
driver is expected to win: the conversation loop re-sends (and cache-reads) a growing transcript
every round, while an episode is a fresh, bounded prompt — so the gap shows up squarely in
tokens/verdict.

Two metrics are `n/a` for the conversation loop by design: **validator first-attempt %** (it writes
no ledger to derive author-vs-optimize counts from), and any ratio whose denominator is zero. The
comparison still makes the decision legible, because the two decision rows — verdicts/session and
tokens/verdict — are computed identically for both loops.

## The flip criterion

Episodic **meets the flip criterion** on a fixture when **both** hold:

1. **Verdicts/session does not regress** — `episodic ≥ conversation`.
2. **Tokens/verdict is materially lower** — episodic spends at least **30% fewer** tokens per
   verdict than the conversation loop (`MATERIAL_TOKEN_REDUCTION = 0.30` in
   `src/noctis/research/parity.py`; a real small-context win is a large gap, not a rounding margin).

The harness prints the verdict on the last line:

- **PASS** — both hold: the evidence supports flipping `auto` to episodic on small windows.
- **FAIL** — verdicts/session regressed, or the token cut was under the threshold; the summary names
  which.
- **INCONCLUSIVE** — a tokens/verdict is `n/a` (a loop reached zero verdicts), so spend can't be
  compared; re-run with a fixture/mandate that produces verdicts on both loops.

A PASS here is the evidence an operator (or a follow-up change) leans on to prefer episodic when
`research.agent.loop: auto` meets a small context window. This harness only *reports* — it never
changes the loop selection itself.

See [research.md](research.md) for how each loop drives `formulate → match → optimize → decide`, and
[development.md](development.md) for the quality gates the parity module ships under.
