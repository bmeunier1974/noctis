"""Symbol → champion assignment — the one resolver every consumer shares.

Three places must agree on which champion owns which symbol: the trading driver (who
decides the bars), realized P&L attribution at settle (who earned the session), and the
unrealized attribution on open positions (who holds them now). They all call
:func:`assign_indices`, and the two that start from registry entries derive its inputs
through :func:`slot_inputs` — one derivation, so trading and attribution cannot drift on
what a champion's eligible symbols or election score are.
"""

from __future__ import annotations


def slot_inputs(entries) -> tuple[list[set[str] | None], list[float]]:
    """Per-champion live-symbol sets and election scores from registry entries, in registry
    order (index ``i`` is the same champion in both lists and in ``entries``).

    ``None`` marks a legacy champion (crowned before panel research, no persisted symbols),
    eligible for the whole universe; scores are the out-of-sample test metrics the board
    elected on.
    """
    sets = [None if e.live_symbols is None else set(e.live_symbols) for e in entries]
    scores = [e.test_metric for e in entries]
    return sets, scores


def assign_indices(
    n: int,
    symbols: list[str],
    live_symbols: list[set[str] | None] | None = None,
    scores: list[float] | None = None,
) -> dict[str, int]:
    """Symbol → champion **index** (into the ``n``-long champion list): the assignment core.

    ``live_symbols[i]`` is champion *i*'s attached symbol set; ``None`` marks a legacy
    champion (crowned before panel research), eligible everywhere. Among eligible champions
    the highest ``scores[i]`` wins; ties fall back to round-robin rotation, so all-legacy
    champions with equal scores reproduce the original round-robin exactly. A symbol with no
    eligible champion is left unassigned. One champion per symbol always.
    """
    if n == 0:
        return {}
    live_symbols = live_symbols if live_symbols is not None else [None] * n
    scores = scores if scores is not None else [0.0] * n
    out: dict[str, int] = {}
    for i, sym in enumerate(symbols):
        eligible = [j for j, ls in enumerate(live_symbols) if ls is None or sym in ls]
        if not eligible:
            continue
        best = max(scores[j] for j in eligible)
        tied = {j for j in eligible if scores[j] == best}
        for k in range(n):  # tiebreak: next tied candidate in round-robin order
            j = (i + k) % n
            if j in tied:
                out[sym] = j
                break
    return out
