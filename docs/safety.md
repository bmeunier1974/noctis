# Safety

Noctis is **paper-only by design**, and the guarantees are structural — enforced by gates
and seams in the code, not by configuration discipline or prompting.

## The two-gate invariant

Real-money order paths are unreachable unless **two independent gates are both open**:

1. Config `mode: live` in `config.yaml`
2. Environment `ALLOW_LIVE=true`

The gates deliberately live in different sources (file vs environment), so no single edit can
open both. Either gate alone keeps the system in paper mode. `mode: live` without `ALLOW_LIVE`
is a **hard startup error** (`src/noctis/config/gate.py`) — the system refuses to start rather
than silently downgrading, so a misconfiguration is always visible.

And even with both gates open, the live execution adapter is a **stub that refuses** — no
real-order path exists in the codebase.

**The gate is never restored from a file that is not `config.yaml`.** A run freezes its
configuration so it can be resumed weeks later ([configuration.md](configuration.md#config-freezing-what-a-resumed-run-reads)),
and `mode`/`allow_live` are the one pair excluded from that: never written to a run record (the
record schema refuses one that carries either) and never rehydrated. `resolve_execution_mode` runs
fresh at **every** process start, so `noctis run --resume <run_id>` faces exactly the same hard
startup error as a first start. A record could otherwise have become a third source for a decision
that must have exactly two independent ones. The record does carry the gate's *verdict* for the
run, and a resume whose freshly resolved mode disagrees with it refuses to continue — a paper run's
results can never acquire live segments.

**`paper_only` on a run record is a measurement, not a claim.** The record's `assumptions` block
publishes `paper_only` and `live_gate.real_orders_reachable`, and both are derived from that frozen
*verdict* — the one `resolve_execution_mode` reached for this run — rather than written as
constants beside it. A run that froze no verdict (an adopted history) reports `null` on both:
"nobody measured" and "paper" are different facts, and only one of them is a claim the record is
entitled to make. The block never carries `mode` or `allow_live` themselves; the schema validator
refuses a record that does, wherever in the document they appear.

## No lookahead

- Both backtest stages execute a bar-*t* decision at bar *t+1*'s open — asserted by tests, not
  assumed.
- Walk-forward test windows sit strictly *after* their train windows
  (`src/noctis/backtest/splits.py`).
- Previews and market digests never expose holdout bars to the research agent.

## Honest promotion

Fixed seeds, versioned catalog snapshots, and out-of-sample metrics on **two axes** (a temporal
holdout and a symbol holdout the search never touched) before any promotion. The promotion
gates are the arbiter of quality: a failing candidate is answered with a better thesis or an
honest rejection — never a loosened gate. An operator mandate steers *what* to research and
configures the run around it (which model thinks, what it may spend, which names it starts
from, what "good" is scored as), but it can never loosen a gate, the exhaustion rule, or the
honesty contract. That is structural, not a convention: the overlay allowlist
(`src/noctis/config/overlay.py`) classifies **every** setting exactly once, the arena — the
safety mode, the fill-cost floor, every `promotion.*` threshold but `metric`, the
fit-set/symbol-holdout geometry, the output paths, the secrets — is refused **by name**, and a
refusal is **fatal at startup** with its reason printed rather than a warning nobody reads. The
one gate-adjacent knob a mandate may move is the exhaustion floor `research.min_trials`, and it
is clamped to **raise only**: more evidence per verdict is legitimate steering, less is the
loosening this rule forbids. The composition root proves the point after the fact — every
overlay is bracketed by a gate-unmoved assertion over the whole refused subtree.

## Spend safety

- Every data ingest passes a **cost preflight** that pads the vendor estimate +20% and refuses
  anything over `data.budget_usd` — cleanly, with the run continuing ([data.md](data.md)).
- A mandate may move that cap **down only**: steering can spend less of your vendor budget,
  never more than the ceiling `config.yaml` set ([configuration.md](configuration.md#the-mandate-overlay)).
- `research.cost_profile` scales resource ceilings only; it can never lower the `min_trials`
  exhaustion floor or touch a promotion gate.

## State integrity

- The continuous paper account **refuses to trade on a corrupt state file** rather than
  silently restarting at 100k.
- A trading day with no new lake data **skips trading and says so** in the report instead of
  replaying stale bars.

## Secrets and data hygiene

- All credentials come from `.env` / the environment (gitignored) — no secrets in the repo.
- No vendor market data is committed to git; the lake is reproducible from the coverage
  registry + manifests.

## Disclaimer

Noctis is **research and educational software**, **paper-only by construction**, and is **not
financial, investment, or trading advice**. It is provided without warranty of any kind and
**with no warranty of fitness for live trading**. Backtested and paper results are simulated;
**past simulated performance does not indicate future results**. Any decision to adapt this
code toward live trading is made entirely at your own risk. See the
[README disclaimer](../README.md#disclaimer).
