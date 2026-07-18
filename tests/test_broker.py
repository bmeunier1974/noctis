"""Paper broker fills / positions / P&L, and the gated live stub."""

from __future__ import annotations

import pytest

from noctis.broker import (
    FeeModel,
    LiveBrokerUnavailableError,
    Order,
    OrderType,
    PaperBroker,
    Side,
    SlippageModel,
)
from noctis.broker.live_stub import LiveBroker
from noctis.config import SafetyGateError, load_settings


def test_market_order_fills_and_moves_position_and_pnl():
    broker = PaperBroker(starting_cash=100_000.0)
    broker.set_price("AAPL", 100.0, ts_event=1)

    fill = broker.submit_order(Order("AAPL", Side.BUY, 10))
    assert fill.symbol == "AAPL"
    assert fill.quantity == 10
    assert fill.price >= 100.0  # buy slips up

    pos = broker.position("AAPL")
    assert pos.quantity == 10
    assert pos.avg_price == fill.price
    # cash dropped by notional + fee.
    assert broker.cash < 100_000.0

    equity_at_100 = broker.equity()
    broker.set_price("AAPL", 110.0)
    assert broker.equity() > equity_at_100  # unrealised gain as price rises


def test_round_trip_costs_are_a_loss():
    """Buy then sell at the same mark → a small loss equal to fees + slippage."""
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(2.0), slippage_model=SlippageModel(2.0)
    )
    broker.set_price("SPY", 400.0)
    broker.submit_order(Order("SPY", Side.BUY, 5))
    broker.submit_order(Order("SPY", Side.SELL, 5))
    assert broker.position("SPY").quantity == 0
    assert broker.realized_pnl < 0.0  # costs make a flat round-trip lose money
    assert broker.fees_paid > 0.0
    # equity ends below start by exactly the incurred costs.
    assert broker.equity() < 100_000.0


def test_selling_realizes_gain():
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )
    broker.set_price("NVDA", 100.0)
    broker.submit_order(Order("NVDA", Side.BUY, 10))
    broker.set_price("NVDA", 120.0)
    broker.submit_order(Order("NVDA", Side.SELL, 10))
    assert broker.position("NVDA").quantity == 0
    assert broker.realized_pnl == pytest.approx(200.0)  # 10 * (120 - 100)


# --- priced fills with provenance (protective-exits phase 1) ------------------------------


def test_priced_rebalance_fills_at_price_with_adverse_slippage_and_fee():
    """An exit-style close executes at the caller's price — never at the mark."""
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(2.0), slippage_model=SlippageModel(2.0)
    )
    broker.set_price("AAPL", 100.0, ts_event=7)
    broker.rebalance_to("AAPL", 10.0)

    # A stop fires at 95 while the mark still says 100: the SELL fills at 95 slipped
    # adversely (down), with the fee charged on notional at the fill price.
    fill = broker.rebalance_to("AAPL", 0.0, price=95.0)
    assert fill is not None
    expected_price = 95.0 * (1 - 2.0 / 10_000.0)
    assert fill.price == pytest.approx(expected_price)
    assert fill.fee == pytest.approx(expected_price * 10.0 * (2.0 / 10_000.0))
    assert broker.position("AAPL").quantity == 0.0


def test_fill_reason_propagates_and_defaults_to_target():
    """Reporting and the forward ledger tell exit fills apart by ``Fill.reason``."""
    broker = PaperBroker(starting_cash=100_000.0)
    broker.set_price("MSFT", 200.0)

    opened = broker.rebalance_to("MSFT", 5.0)
    assert opened is not None
    assert opened.reason == "target"

    stopped = broker.rebalance_to("MSFT", 0.0, price=190.0, reason="stop")
    assert stopped is not None
    assert stopped.reason == "stop"


def test_order_type_carries_stop_and_limit_provenance_labels():
    """STOP/LIMIT tag an order's provenance — there is no resting-order book behind them."""
    order = Order("AAPL", Side.SELL, 1, OrderType.STOP)
    assert order.order_type is OrderType.STOP
    assert OrderType.LIMIT.value == "LIMIT"


def test_submit_requires_mark_price():
    broker = PaperBroker()
    with pytest.raises(RuntimeError):
        broker.submit_order(Order("AAPL", Side.BUY, 1))


# --- serialization (the continuous account carried across sessions) -----------------------


def test_paper_broker_state_round_trips_through_dict():
    import json

    broker = PaperBroker(100_000.0, fee_model=FeeModel(2.0), slippage_model=SlippageModel(2.0))
    broker.set_price("AAPL", 100.0, ts_event=1)
    broker.set_price("TSLA", 200.0, ts_event=2)
    broker.submit_order(Order("AAPL", Side.BUY, 10))  # long
    broker.submit_order(Order("TSLA", Side.SELL, 5))  # short (negative qty)
    broker.submit_order(Order("AAPL", Side.SELL, 4))  # partial close → realized P&L

    data = json.loads(json.dumps(broker.to_dict()))  # survives an actual JSON round trip
    clone = PaperBroker.from_dict(data, fee_model=FeeModel(2.0), slippage_model=SlippageModel(2.0))

    assert clone.equity() == pytest.approx(broker.equity())
    assert clone.cash == pytest.approx(broker.cash)
    assert clone.starting_cash == broker.starting_cash
    assert set(clone.positions()) == set(broker.positions())
    for sym, pos in broker.positions().items():
        assert clone.position(sym).quantity == pytest.approx(pos.quantity)
        assert clone.position(sym).avg_price == pytest.approx(pos.avg_price)
    assert clone.position("TSLA").quantity == -5  # the short survived
    assert clone.realized_pnl == pytest.approx(broker.realized_pnl)
    assert clone.fees_paid == pytest.approx(broker.fees_paid)
    assert clone.slippage_cost == pytest.approx(broker.slippage_cost)
    assert clone.fills == []  # fills are per-session report material, not account state


def test_realized_pnl_by_symbol_accumulates_and_sums_to_global():
    # Per-symbol realized (plan 5) tracks close/flip P&L separately and must sum to the global.
    broker = PaperBroker(fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0))
    broker.set_price("AAPL", 100.0, ts_event=1)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    broker.set_price("AAPL", 120.0, ts_event=2)
    broker.submit_order(Order("AAPL", Side.SELL, 10))  # +200
    broker.set_price("TSLA", 200.0, ts_event=3)
    broker.submit_order(Order("TSLA", Side.SELL, 5))  # short
    broker.set_price("TSLA", 180.0, ts_event=4)
    broker.submit_order(Order("TSLA", Side.BUY, 5))  # cover → +100

    assert broker.realized_pnl_by_symbol["AAPL"] == pytest.approx(200.0)
    assert broker.realized_pnl_by_symbol["TSLA"] == pytest.approx(100.0)
    assert sum(broker.realized_pnl_by_symbol.values()) == pytest.approx(broker.realized_pnl)


def test_realized_pnl_by_symbol_round_trips_and_defaults_when_absent():
    broker = PaperBroker(fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0))
    broker.set_price("NVDA", 100.0, ts_event=1)
    broker.submit_order(Order("NVDA", Side.BUY, 10))
    broker.set_price("NVDA", 120.0, ts_event=2)
    broker.submit_order(Order("NVDA", Side.SELL, 10))  # +200

    data = broker.to_dict()
    assert data["realized_pnl_by_symbol"]["NVDA"] == pytest.approx(200.0)
    clone = PaperBroker.from_dict(data)
    assert clone.realized_pnl_by_symbol["NVDA"] == pytest.approx(200.0)

    # An older account file (pre plan 5) has no per-symbol key → loads with an empty split,
    # global realized unchanged.
    del data["realized_pnl_by_symbol"]
    legacy = PaperBroker.from_dict(data)
    assert legacy.realized_pnl_by_symbol == {}
    assert legacy.realized_pnl == pytest.approx(200.0)


def test_paper_broker_round_trip_after_flipping_through_zero():
    broker = PaperBroker(fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0))
    broker.set_price("NVDA", 100.0, ts_event=1)
    broker.submit_order(Order("NVDA", Side.BUY, 10))
    broker.set_price("NVDA", 120.0, ts_event=2)
    broker.submit_order(Order("NVDA", Side.SELL, 25))  # long 10 → short 15 through zero

    clone = PaperBroker.from_dict(
        broker.to_dict(), fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )
    assert clone.position("NVDA").quantity == -15
    assert clone.position("NVDA").avg_price == pytest.approx(120.0)
    assert clone.realized_pnl == pytest.approx(200.0)  # 10 * (120 - 100)
    # And the clone keeps trading correctly: covering the short at 110 realizes the gain.
    clone.set_price("NVDA", 110.0, ts_event=3)
    clone.submit_order(Order("NVDA", Side.BUY, 15))
    assert clone.realized_pnl == pytest.approx(200.0 + 15 * (120.0 - 110.0))
    assert clone.position("NVDA").is_flat


# --- gated live stub ---------------------------------------------------------------------


def test_live_stub_raises_in_paper_mode():
    with pytest.raises(LiveBrokerUnavailableError):
        LiveBroker(load_settings(mode="paper", allow_live=False))


def test_live_stub_raises_when_paper_but_allow_live():
    """Only one gate open (ALLOW_LIVE) but config paper → still unreachable."""
    with pytest.raises(LiveBrokerUnavailableError):
        LiveBroker(load_settings(mode="paper", allow_live=True))


def test_live_stub_raises_when_live_but_no_allow_live():
    """Config live but no ALLOW_LIVE → the gate itself refuses to start."""
    with pytest.raises(SafetyGateError):
        LiveBroker(load_settings(mode="live", allow_live=False))


def test_live_stub_raises_even_when_both_gates_open():
    """Both gates open → the adapter still refuses (no real-order path exists)."""
    with pytest.raises(LiveBrokerUnavailableError):
        LiveBroker(load_settings(mode="live", allow_live=True))
