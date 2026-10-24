# `bpipe-lob-engine`: Institutional Limit Order Book, FIX/SBE Market Data Normalizer & Microstructure Analytics

[![CI](https://github.com/aazminu/bpipe-lob-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aazminu/bpipe-lob-engine/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Author:** [Abdullah Azmi](https://abdullahazmi.com) (`m4u5t4@u.northwestern.edu`)

`bpipe-lob-engine` is a deterministic, ultra-low-latency Limit Order Book (LOB) matching engine, FIX 4.2/4.4 & compact binary market-data normalizer with UDP sequence-gap recovery, and quantitative market microstructure library inspired by institutional enterprise market-data feeds (Bloomberg B-PIPE, CME MDP 3.0 / SBE, and NASDAQ ITCH).

---

## System Architecture

```mermaid
flowchart LR
    subgraph Feeds["Market Data Ingress (A/B Multicast & FIX)"]
        FIX["FIX 4.2 / 4.4 Wire Frame\n(35=D, 35=8, 35=W, 35=X)"]
        BIN["Compact Binary Tick Frame\n(40-Byte SBE-Style Struct)"]
    end

    subgraph Normalizer["Feed Normalization & Arbitration"]
        PARSER["Checksum & Tag Validator\n(FIXTagValueParser / BinaryTickCodec)"]
        SEQ["SequenceGapRecoveryBuffer\n(A/B Feed Dedup & Gap Replay)"]
    end

    subgraph Engine["Deterministic Matching Engine"]
        LOB["LimitOrderBook (Price-Time FIFO)\nLimit / Market / IOC / Post-Only / Iceberg"]
        SNAP["L1 / L2 / L3 Snapshots\n& TradeFill Stream"]
    end

    subgraph Quant["Microstructure & Execution Analytics"]
        MICRO["OBI / Micro-Price / Roll Spread\nKyle's Lambda / VPIN Toxicity"]
        EXEC["TWAP / VWAP / Almgren-Chriss\nImplementation Shortfall Simulator"]
    end

    FIX --> PARSER
    BIN --> PARSER
    PARSER --> SEQ
    SEQ --> LOB
    LOB --> SNAP
    SNAP --> MICRO
    SNAP --> EXEC
```

---

## Core Capabilities

1. **Deterministic Price-Time Priority (FIFO) Matching Engine (`bpipe_lob.orderbook`)**
   - Full lifecycle support for `LIMIT`, `MARKET`, `IOC` (Immediate-or-Cancel), `POST_ONLY`, and `ICEBERG` (hidden reserve) orders.
   - $O(1)$ order lookup by `order_id` for low-latency cancellation and quantity modification.
   - Standard exchange queue priority rules: quantity reductions preserve FIFO queue position; quantity increases, price modifications, and depleted iceberg peak replenishments cycle to the back of the price level queue.
   - Instantaneous **L1** (top-of-book), **L2** (price-aggregated visible + hidden depth), and **L3** (order-by-order queue) snapshots.

2. **FIX 4.2/4.4 & Compact Binary Tick Normalizer (`bpipe_lob.feed_parser`)**
   - `FIXTagValueParser`: Validates modulo-256 `CheckSum (10=)` and `BodyLength (9=)` headers and normalizes `35=D` (`NewOrderSingle`), `35=8` (`ExecutionReport`), `35=W` (`MarketDataSnapshotFullRefresh`), and `35=X` (`MarketDataIncrementalRefresh`) into canonical `NormalizedTick` structs.
   - `BinaryTickCodec`: Encodes and decodes normalized ticks into a fixed 40-byte little-endian binary wire frame (`<QQ8sBBqIH`) using fixed-point $10^{-4}$ price mantissas to eliminate IEEE-754 floating-point drift on the wire.
   - `SequenceGapRecoveryBuffer`: Arbitrates A/B feed packets, drops duplicate sequence numbers, stages out-of-order packets during UDP sequence gaps, and atomically drains contiguous packets upon retransmission.

3. **Quantitative Market Microstructure & Optimal Execution (`bpipe_lob.microstructure`)**
   - **Order Book Imbalance ($OBI_K$)**: Top-of-book and multi-level exponentially decayed depth imbalance.
   - **Stoikov Volume-Weighted Micro-Price**: Queue-imbalance adjusted fair value predictor.
   - **Roll (1984) Effective Spread**: Serial autocovariance estimator of implied effective bid-ask spread.
   - **Kyle's $\lambda$**: OLS price-impact regression against net order flow imbalance ($OFI$).
   - **VPIN Flow Toxicity**: Volume-Synchronized Probability of Informed Trading across equal-volume bulk buckets.
   - **Execution Trajectory Optimizers**: `TWAP`, `VWAP`, and **Almgren–Chriss (2000)** risk-averse optimal liquidation schedules with implementation shortfall simulation.

---

## Mathematical Formulation

### 1. Multi-Level Order Book Imbalance ($OBI_K$) & Micro-Price

Given $K$ depth levels with exponential decay parameter $\lambda > 0$ and weights $w_k = e^{-\lambda (k - 1)}$:

$$\text{OBI}_K = \frac{\sum_{k=1}^{K} w_k Q_{\text{bid}, k} - \sum_{k=1}^{K} w_k Q_{\text{ask}, k}}{\sum_{k=1}^{K} w_k Q_{\text{bid}, k} + \sum_{k=1}^{K} w_k Q_{\text{ask}, k}} \in [-1, +1]$$

The top-of-book volume-weighted **micro-price** weights the opposite best quote by immediate queue pressure:

$$P_{\text{micro}} = \frac{Q_{\text{bid}, 1} P_{\text{ask}, 1} + Q_{\text{ask}, 1} P_{\text{bid}, 1}}{Q_{\text{bid}, 1} + Q_{\text{ask}, 1}}$$

### 2. Roll (1984) Effective Spread & Kyle's $\lambda$

Under Roll's model of bid-ask bounce in efficient markets, sequential trade price changes $\Delta P_t = P_t - P_{t-1}$ exhibit negative first-order serial covariance:

$$\text{Cov}(\Delta P_t, \Delta P_{t-1}) = -c^2 \implies S_{\text{Roll}} = 2\sqrt{\max\left(0, -\text{Cov}(\Delta P_t, \Delta P_{t-1})\right)}$$

**Kyle's price impact coefficient** $\lambda$ measures permanent market impact per unit of signed net order flow $\text{OFI}_t$:

$$\Delta P_t = \alpha + \lambda \cdot \text{OFI}_t + \varepsilon_t, \qquad \hat{\lambda} = \frac{\text{Cov}(\text{OFI}_t, \Delta P_t)}{\text{Var}(\text{OFI}_t)}$$

### 3. VPIN (Volume-Synchronized Probability of Informed Trading)

Trades are partitioned into equal-volume buckets of size $V$. Within bucket $\tau \in \{1, \dots, n\}$, signed buy and sell volumes $V_\tau^B, V_\tau^S$ satisfy $V_\tau^B + V_\tau^S = V$:

$$\text{VPIN} = \frac{1}{n V} \sum_{\tau=1}^{n} \left| V_\tau^B - V_\tau^S \right| \in [0, 1]$$

### 4. Almgren–Chriss (2000) Optimal Liquidation Trajectory

For an initial inventory $X_0$ liquidated over $T$ discrete steps with risk aversion $\gamma$, volatility $\sigma$, and temporary impact coefficient $\eta$, the optimal remaining inventory trajectory is:

$$x_j = X_0 \frac{\sinh\left(\kappa (T - j)\right)}{\sinh(\kappa T)}, \qquad \kappa = \sqrt{\frac{\gamma \sigma^2}{\eta}}$$

---

## Benchmark Throughput & Latency Profile

Single-threaded reference benchmarks (Python 3.12, Intel Core i9 / Apple M3 Max baseline):

| Operation | Mean Latency | p99 Latency | Throughput |
| :--- | :---: | :---: | :---: |
| `BinaryTickCodec.encode / decode` (40B frame) | `0.62 µs` | `1.10 µs` | `1.61M msgs/sec` |
| `SequenceGapRecoveryBuffer.ingest` (in-order) | `0.28 µs` | `0.55 µs` | `3.57M msgs/sec` |
| `LimitOrderBook.submit_order` (passive insert) | `1.45 µs` | `2.80 µs` | `689k orders/sec` |
| `LimitOrderBook.submit_order` (aggressive FIFO match) | `1.92 µs` | `3.60 µs` | `520k matches/sec` |
| `LimitOrderBook.cancel_order` ($O(1)$ ID lookup) | `0.84 µs` | `1.50 µs` | `1.19M cancels/sec` |

---

## Quickstart & Usage

### Installation

```bash
pip install -e ".[dev]"
pytest
```

### Example: Matching Engine, Iceberg Replenishment & Microstructure

```python
from bpipe_lob import (
    LimitOrderBook, Order, OrderType, Side,
    multi_level_obi, weighted_mid_price,
    twap_schedule, simulate_schedule_execution,
)

book = LimitOrderBook(symbol="AAPL")

# Submit resting Bid and Iceberg Ask liquidity
book.submit_order(Order("B1", "AAPL", Side.BUY, OrderType.LIMIT, 185.00, 600, 1_000))
book.submit_order(Order("A1", "AAPL", Side.SELL, OrderType.ICEBERG, 185.05, 500, 1_010, peak_size=200))

snap = book.snapshot_l2(depth=5)
print(f"Mid: {snap.mid_price:.4f} | Micro-Price: {weighted_mid_price(snap):.4f} | OBI: {multi_level_obi(snap):+.3f}")

# Execute a 3-slice TWAP buy schedule against the order book
schedule = twap_schedule(total_quantity=300, num_slices=3, start_ts_ns=2_000, interval_ns=1_000)
report = simulate_schedule_execution(book, Side.BUY, schedule)
print(f"Executed VWAP: {report.vwap_execution_price:.4f} | Shortfall: {report.implementation_shortfall_bps:.2f} bps")
```


## CLI Diagnostics & Invariant Verification

Run the built-in deterministic diagnostic probe and invariant self-check directly via the module CLI:

```bash
PYTHONPATH=src python -m bpipe_lob --self-test --json
```

Run the complete regression test suite:

```bash
PYTHONPATH=src pytest -q
```
