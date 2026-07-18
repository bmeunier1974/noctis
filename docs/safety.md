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
honest rejection — never a loosened gate. An operator mandate may steer *what* to research and
bind exactly one knob (`promotion.metric`); it can never loosen a gate, the exhaustion rule, or
the honesty contract.

## Spend safety

- Every data ingest passes a **cost preflight** that pads the vendor estimate +20% and refuses
  anything over `data.budget_usd` — cleanly, with the run continuing ([data.md](data.md)).
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
