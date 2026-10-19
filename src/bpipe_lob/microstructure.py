"""
High-Frequency Market Microstructure & Optimal Execution Analytics.

Implements:
- Order Book Imbalance (L1 and multi-level exponentially decayed OBI)
- Volume-weighted Micro-Price & Weighted Mid-Price
- Roll (1984) Effective Bid-Ask Spread estimator from serial trade autocovariance
- Kyle's Lambda (price impact coefficient via OLS regression)
- Realized Volatility (annualized log-return estimator)
- VPIN (Volume-Synchronized Probability of Informed Trading) flow toxicity
- Optimal execution schedules: TWAP, VWAP, and Almgren-Chriss (2000) trajectory
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

import numpy as np

from .orderbook import LimitOrderBook, Order, OrderBookSnapshot, OrderType, Side


def order_book_imbalance(bid_qty: float, ask_qty: float) -> float:
    """
    Compute Top-of-Book Order Book Imbalance (OBI) in [-1.0, +1.0]:
        OBI = (Q_bid - Q_ask) / (Q_bid + Q_ask)
    """
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.0
    return float((bid_qty - ask_qty) / total)


def multi_level_obi(
    snapshot: OrderBookSnapshot, levels: int = 5, decay_lambda: float = 0.5
) -> float:
    """
    Compute exponentially depth-weighted Order Book Imbalance across top K levels:
        w_k = exp(-decay_lambda * (k - 1)),  k = 1..K
        OBI_K = (sum w_k Q_bid,k - sum w_k Q_ask,k) / (sum w_k Q_bid,k + sum w_k Q_ask,k)
    """
    k_max = min(levels, len(snapshot.bids), len(snapshot.asks))
    if k_max == 0:
        return 0.0

    weighted_bid = 0.0
    weighted_ask = 0.0
    for k in range(k_max):
        weight = math.exp(-decay_lambda * k)
        weighted_bid += weight * snapshot.bids[k].visible_quantity
        weighted_ask += weight * snapshot.asks[k].visible_quantity

    denom = weighted_bid + weighted_ask
    if denom <= 0:
        return 0.0
    return float((weighted_bid - weighted_ask) / denom)


def micro_price(
    best_bid: float, best_ask: float, bid_qty: float, ask_qty: float
) -> float:
    """
    Compute Stoikov (2018) / volume-weighted micro-price:
        P_micro = (Q_bid * P_ask + Q_ask * P_bid) / (Q_bid + Q_ask)

    When bid_qty >> ask_qty, buying pressure pushes micro-price toward best_ask.
    """
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.5 * (best_bid + best_ask)
    return float((bid_qty * best_ask + ask_qty * best_bid) / total)


def weighted_mid_price(snapshot: OrderBookSnapshot) -> float:
    """Compute top-of-book micro-price directly from an OrderBookSnapshot."""
    if not snapshot.bids or not snapshot.asks:
        raise ValueError("Snapshot must contain at least one bid and one ask level")
    b0 = snapshot.bids[0]
    a0 = snapshot.asks[0]
    return micro_price(b0.price, a0.price, b0.visible_quantity, a0.visible_quantity)


def roll_effective_spread(trade_prices: Sequence[float]) -> float:
    """
    Compute Roll (1984) implied effective bid-ask spread from serial covariance
    of consecutive price changes:
        Delta P_t = P_t - P_{t-1}
        Cov(Delta P_t, Delta P_{t-1}) = -c^2  =>  Spread = 2 * sqrt(-Cov)
    If sample serial covariance is non-negative, returns 0.0.
    """
    prices = np.asarray(trade_prices, dtype=np.float64)
    if prices.size < 3:
        raise ValueError("At least 3 trade prices are required for Roll spread estimation")
    diffs = np.diff(prices)
    cov_matrix = np.cov(diffs[1:], diffs[:-1], ddof=1)
    serial_cov = float(cov_matrix[0, 1])
    if serial_cov >= 0.0:
        return 0.0
    return float(2.0 * math.sqrt(-serial_cov))


def kyle_lambda(price_changes: Sequence[float], net_order_flows: Sequence[float]) -> float:
    """
    Estimate Kyle's (1985) price impact coefficient lambda via OLS:
        Delta P_t = alpha + lambda * OFI_t + epsilon_t
    """
    dp = np.asarray(price_changes, dtype=np.float64)
    ofi = np.asarray(net_order_flows, dtype=np.float64)
    if dp.size != ofi.size or dp.size < 2:
        raise ValueError("price_changes and net_order_flows must have equal length >= 2")
    var_ofi = float(np.var(ofi, ddof=1))
    if var_ofi <= 1e-16:
        return 0.0
    cov = float(np.cov(ofi, dp, ddof=1)[0, 1])
    return cov / var_ofi


def realized_volatility(prices: Sequence[float], periods_per_year: float = 252.0) -> float:
    """Compute annualized realized volatility from log price returns."""
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size < 2 or np.any(arr <= 0):
        raise ValueError("Need at least 2 positive prices")
    log_rets = np.diff(np.log(arr))
    return float(np.std(log_rets, ddof=1) * math.sqrt(periods_per_year))


def vpin_toxicity(
    trade_prices: Sequence[float],
    trade_volumes: Sequence[float],
    bucket_volume: float,
    window_buckets: int = 10,
) -> float:
    """
    Compute Easley, Lopez de Prado, and O'Hara (2012) Volume-Synchronized
    Probability of Informed Trading (VPIN) in [0.0, 1.0].

    Uses tick-rule / bulk volume classification across equal-volume buckets V:
        VPIN = (1 / (n * V)) * sum_{tau=1..n} |V_tau^B - V_tau^S|
    """
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be positive")
    prices = np.asarray(trade_prices, dtype=np.float64)
    vols = np.asarray(trade_volumes, dtype=np.float64)
    if prices.size != vols.size or prices.size < 2:
        raise ValueError("trade_prices and trade_volumes must have matching length >= 2")

    # Classify trades using tick rule with carry-forward sign
    signs = np.ones(prices.size, dtype=np.float64)
    for i in range(1, prices.size):
        dp = prices[i] - prices[i - 1]
        if dp > 0:
            signs[i] = 1.0
        elif dp < 0:
            signs[i] = -1.0
        else:
            signs[i] = signs[i - 1]

    bucket_imbalances: List[float] = []
    cur_buy = 0.0
    cur_sell = 0.0
    cur_vol = 0.0

    for vol, sgn in zip(vols, signs):
        rem = float(vol)
        while rem > 0:
            space = bucket_volume - cur_vol
            take = min(rem, space)
            if sgn > 0:
                cur_buy += take
            else:
                cur_sell += take
            cur_vol += take
            rem -= take
            if abs(cur_vol - bucket_volume) <= 1e-9:
                bucket_imbalances.append(abs(cur_buy - cur_sell) / bucket_volume)
                cur_buy = 0.0
                cur_sell = 0.0
                cur_vol = 0.0

    if not bucket_imbalances:
        total_v = float(np.sum(vols))
        if total_v <= 0:
            return 0.0
        buy_v = float(np.sum(vols[signs > 0]))
        sell_v = float(np.sum(vols[signs < 0]))
        return abs(buy_v - sell_v) / total_v

    recent = bucket_imbalances[-window_buckets:]
    return float(np.mean(recent))


@dataclass(frozen=True)
class ExecutionSlice:
    """Single child order slice in an algorithmic execution schedule."""

    slice_index: int
    timestamp_ns: int
    quantity: int


@dataclass(frozen=True)
class ExecutionReport:
    """Summary of an algorithmic child-order schedule execution against the LOB."""

    total_quantity: int
    executed_quantity: int
    arrival_mid_price: float
    vwap_execution_price: float
    implementation_shortfall_bps: float
    slice_Count: int


def twap_schedule(
    total_quantity: int, num_slices: int, start_ts_ns: int, interval_ns: int
) -> List[ExecutionSlice]:
    """Partition `total_quantity` evenly across `num_slices` intervals (TWAP)."""
    if total_quantity <= 0 or num_slices <= 0:
        raise ValueError("total_quantity and num_slices must be positive")
    base = total_quantity // num_slices
    rem = total_quantity % num_slices
    slices: List[ExecutionSlice] = []
    for i in range(num_slices):
        qty = base + (1 if i < rem else 0)
        if qty > 0:
            slices.append(
                ExecutionSlice(
                    slice_index=i,
                    timestamp_ns=start_ts_ns + i * interval_ns,
                    quantity=qty,
                )
            )
    return slices


def vwap_schedule(
    total_quantity: int,
    volume_profileWeights: Sequence[float],
    start_ts_ns: int,
    interval_ns: int,
) -> List[ExecutionSlice]:
    """
    Partition `total_quantity` proportionally to historical intraday volume curve weights.
    Guarantees exact integer conservation summing to `total_quantity`.
    """
    weights = np.asarray(volume_profileWeights, dtype=np.float64)
    if total_quantity <= 0 or weights.size == 0 or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("Invalid total_quantity or volume_profileWeights")
    norm_w = weights / np.sum(weights)
    raw_alloc = norm_w * total_quantity
    int_alloc = np.floor(raw_alloc).astype(int)
    residual = int(total_quantity - np.sum(int_alloc))
    if residual > 0:
        frac_order = np.argsort(-(raw_alloc - int_alloc))
        for idx in frac_order[:residual]:
            int_alloc[idx] += 1

    return [
        ExecutionSlice(
            slice_index=i,
            timestamp_ns=start_ts_ns + i * interval_ns,
            quantity=int(q),
        )
        for i, q in enumerate(int_alloc)
        if q > 0
    ]


def almgren_chriss_trajectory(
    total_quantity: int,
    num_slices: int,
    risk_aversion: float,
    volatility: float,
    temporary_impact: float,
) -> List[int]:
    """
    Compute the Almgren-Chriss (2000) optimal liquidation trade list:
        kappa = sqrt(lambda_risk * sigma^2 / eta)
        x_j = X * sinh(kappa * (T - j)) / sinh(kappa * T)
    Higher risk aversion front-loads execution; risk-neutral (risk_aversion -> 0)
    converges to a uniform TWAP schedule.
    """
    if total_quantity <= 0 or num_slices <= 0:
        raise ValueError("total_quantity and num_slices must be positive")
    if risk_aversion <= 1e-12 or volatility <= 0 or temporary_impact <= 0:
        base = total_quantity // num_slices
        rem = total_quantity % num_slices
        return [base + (1 if i < rem else 0) for i in range(num_slices)]

    kappa = math.sqrt(risk_aversion * (volatility ** 2) / temporary_impact)
    t_steps = float(num_slices)
    denom = math.sinh(kappa * t_steps)
    inventory = [
        total_quantity * math.sinh(kappa * (t_steps - j)) / denom
        for j in range(num_slices + 1)
    ]
    raw_trades = np.diff(-np.asarray(inventory))
    int_trades = np.floor(raw_trades).astype(int)
    residual = int(total_quantity - np.sum(int_trades))
    if residual > 0:
        order = np.argsort(-(raw_trades - int_trades))
        for idx in order[:residual]:
            int_trades[idx] += 1
    return [int(x) for x in int_trades]


def simulate_schedule_execution(
    book: LimitOrderBook,
    side: Side,
    schedule: Sequence[ExecutionSlice],
) -> ExecutionReport:
    """
    Execute a child-order schedule against a `LimitOrderBook` and compute
    volume-weighted average execution price (VWAP) and Implementation Shortfall (bps).
    """
    arrival_mid = book.mid_price
    if arrival_mid is None:
        raise ValueError("Order book must have both bid and ask liquidity to benchmark arrival mid")

    total_target = sum(s.quantity for s in schedule)
    exec_qty = 0
    notional = 0.0

    for sl in schedule:
        mkt_order = Order(
            order_id=f"EXEC-SLICE-{sl.slice_index}-{sl.timestamp_ns}",
            symbol=book.symbol,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=sl.quantity,
            timestamp_ns=sl.timestamp_ns,
        )
        fills = book.submit_order(mkt_order)
        for f in fills:
            exec_qty += f.quantity
            notional += f.quantity * f.price

    if exec_qty == 0:
        raise RuntimeError("No liquidity filled during schedule simulation")

    vwap_px = notional / exec_qty
    if side == Side.BUY:
        shortfall_bps = ((vwap_px - arrival_mid) / arrival_mid) * 10_000.0
    else:
        shortfall_bps = ((arrival_mid - vwap_px) / arrival_mid) * 10_000.0

    return ExecutionReport(
        total_quantity=total_target,
        executed_quantity=exec_qty,
        arrival_mid_price=arrival_mid,
        vwap_execution_price=vwap_px,
        implementation_shortfall_bps=float(shortfall_bps),
        slice_Count=len(schedule),
    )
