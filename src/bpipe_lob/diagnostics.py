"""Deterministic runtime diagnostics and invariant verification for `bpipe_lob`."""

from __future__ import annotations

import time
from typing import Any

from .orderbook import LimitOrderBook, Order, Side, OrderType
from .microstructure import order_book_imbalance, micro_price

def _domain_probe(seed: int) -> dict:
    lob = LimitOrderBook(symbol="UST10Y")
    lob.submit_order(Order("B1", "UST10Y", Side.BUY, OrderType.LIMIT, 99.50, 100, 1000))
    lob.submit_order(Order("A1", "UST10Y", Side.SELL, OrderType.LIMIT, 99.55, 60, 1001))
    snap = lob.snapshot_l2(depth=5)
    obi = order_book_imbalance(100.0, 60.0)
    mp = micro_price(99.50, 99.55, 100.0, 60.0)
    return {
        "best_bid": snap.bids[0].price,
        "best_ask": snap.asks[0].price,
        "obi": round(obi, 4),
        "micro_price": round(mp, 6),
        "invariants_ok": (snap.bids[0].price == 99.50 and snap.asks[0].price == 99.55 and 99.50 < mp < 99.55),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `bpipe-lob-engine` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "bpipe-lob-engine",
        "package": "bpipe_lob",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `bpipe_lob`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "bpipe-lob-engine",
        "package": "bpipe_lob",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
