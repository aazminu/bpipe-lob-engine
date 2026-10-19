"""
Comprehensive pytest test suite for bpipe-lob-engine.
Verifies price-time FIFO matching, partial fills, cancels, modifies, iceberg replenishment,
FIX 4.2/4.4 checksum & tag normalization, compact binary SBE codec, sequence-gap recovery,
and market microstructure invariants (OBI, micro-price, Roll spread, Kyle's lambda, VPIN,
TWAP/VWAP/Almgren-Chriss optimal execution).
"""

import pytest

from bpipe_lob import (
    BinaryTickCodec,
    FIXParserError,
    FIXTagValueParser,
    LimitOrderBook,
    NormalizedTick,
    Order,
    OrderType,
    SequenceGapRecoveryBuffer,
    Side,
    almgren_chriss_trajectory,
    kyle_lambda,
    micro_price,
    multi_level_obi,
    order_book_imbalance,
    realized_volatility,
    roll_effective_spread,
    simulate_schedule_execution,
    twap_schedule,
    vpin_toxicity,
    vwap_schedule,
    weighted_mid_price,
)


def _seed_book(symbol: str = "AAPL") -> LimitOrderBook:
    book = LimitOrderBook(symbol=symbol)
    # Resting bids
    book.submit_order(Order("B1", symbol, Side.BUY, OrderType.LIMIT, 100.00, 200, 1_000))
    book.submit_order(Order("B2", symbol, Side.BUY, OrderType.LIMIT, 100.00, 300, 1_010))
    book.submit_order(Order("B3", symbol, Side.BUY, OrderType.LIMIT, 99.90, 500, 1_020))
    # Resting asks
    book.submit_order(Order("A1", symbol, Side.SELL, OrderType.LIMIT, 100.10, 150, 1_030))
    book.submit_order(Order("A2", symbol, Side.SELL, OrderType.LIMIT, 100.10, 250, 1_040))
    book.submit_order(Order("A3", symbol, Side.SELL, OrderType.LIMIT, 100.20, 400, 1_050))
    return book


def test_l1_top_of_book_and_mid_spread():
    book = _seed_book()
    assert book.best_bid == 100.00
    assert book.best_ask == 100.10
    assert book.mid_price == pytest.approx(100.05)
    assert book.spread == pytest.approx(0.10)


def test_price_time_fifo_matching_order():
    book = _seed_book()
    # Aggressive buy of 250 @ 100.10 should fill A1 (150) first, then partial A2 (100 of 250)
    taker = Order("T1", "AAPL", Side.BUY, OrderType.LIMIT, 100.10, 250, 2_000)
    fills = book.submit_order(taker)
    assert len(fills) == 2
    assert fills[0].maker_order_id == "A1"
    assert fills[0].quantity == 150
    assert fills[0].price == 100.10
    assert fills[1].maker_order_id == "A2"
    assert fills[1].quantity == 100
    assert fills[1].price == 100.10

    # Remaining on A2 should be 150
    a2 = book.get_order("A2")
    assert a2 is not None
    assert a2.total_remaining == 150
    assert book.get_order("A1") is None


def test_multi_level_sweep_and_resting_residual():
    book = _seed_book()
    # Sweep all 400 at 100.10, plus 100 at 100.20, and rest 50 at 100.20?
    # Wait: if limit buy is 500 @ 100.15, it sweeps 400 @ 100.10 and rests 100 @ 100.15
    taker = Order("T_SWEEP", "AAPL", Side.BUY, OrderType.LIMIT, 100.15, 500, 2_100)
    fills = book.submit_order(taker)
    assert sum(f.quantity for f in fills) == 400
    assert book.best_bid == 100.15
    assert book.best_ask == 100.20
    resting = book.get_order("T_SWEEP")
    assert resting is not None and resting.total_remaining == 100


def test_market_order_sweeps_multiple_levels():
    book = _seed_book()
    mkt = Order("MKT1", "AAPL", Side.SELL, OrderType.MARKET, 0.0, 600, 2_200)
    fills = book.submit_order(mkt)
    assert len(fills) == 3
    assert [f.maker_order_id for f in fills] == ["B1", "B2", "B3"]
    assert [f.quantity for f in fills] == [200, 300, 100]
    assert book.best_bid == 99.90
    assert book.get_order("B3").total_remaining == 400


def test_ioc_order_fills_available_and_cancels_remainder():
    book = _seed_book()
    ioc = Order("IOC1", "AAPL", Side.BUY, OrderType.IOC, 100.10, 500, 2_300)
    fills = book.submit_order(ioc)
    assert sum(f.quantity for f in fills) == 400
    assert book.get_order("IOC1") is None
    assert book.best_ask == 100.20


def test_post_only_rejects_crossing_order_and_accepts_passive():
    book = _seed_book()
    crossing = Order("PO_CROSS", "AAPL", Side.BUY, OrderType.POST_ONLY, 100.10, 100, 2_400)
    fills = book.submit_order(crossing)
    assert fills == []
    assert book.get_order("PO_CROSS") is None

    passive = Order("PO_PASS", "AAPL", Side.BUY, OrderType.POST_ONLY, 100.05, 100, 2_410)
    book.submit_order(passive)
    assert book.best_bid == 100.05
    assert book.get_order("PO_PASS") is not None


def test_cancel_order_updates_best_bid_ask():
    book = _seed_book()
    assert book.cancel_order("A1", 3_000) is True
    assert book.cancel_order("A2", 3_010) is True
    assert book.best_ask == 100.20
    assert book.cancel_order("NON_EXISTENT", 3_020) is False


def test_modify_order_qty_decrease_preserves_priority_and_increase_loses_priority():
    book = _seed_book()
    # Decrease B1 from 200 to 120: B1 stays ahead of B2 at 100.00
    book.modify_order("B1", new_quantity=120, timestamp_ns=3_100)
    l3 = book.snapshot_l3()["bids"]
    assert l3[0][1] == "B1" and l3[0][2] == 120
    assert l3[1][1] == "B2" and l3[1][2] == 300

    # Increase B1 from 120 to 250: B1 moves behind B2 at 100.00
    book.modify_order("B1", new_quantity=250, timestamp_ns=3_200)
    l3_after = book.snapshot_l3()["bids"]
    assert l3_after[0][1] == "B2"
    assert l3_after[1][1] == "B1" and l3_after[1][2] == 250


def test_iceberg_order_replenishment_and_queue_priority():
    book = LimitOrderBook("MSFT")
    # Iceberg ask: total 300, peak 100 @ 200.00
    ice = Order(
        "ICE1", "MSFT", Side.SELL, OrderType.ICEBERG, 200.00, 300, 1_000, peak_size=100
    )
    lit = Order("LIT1", "MSFT", Side.SELL, OrderType.LIMIT, 200.00, 100, 1_010)
    book.submit_order(ice)
    book.submit_order(lit)

    snap = book.snapshot_l2()
    assert snap.asks[0].visible_quantity == 200
    assert snap.asks[0].hidden_quantity == 200

    # Buy 150 @ 200.00: should fill 100 from ICE1 (exhausting its first visible peak,
    # causing ICE1 to replenish 100 and move behind LIT1), then fill 50 from LIT1
    fills = book.submit_order(Order("BUY1", "MSFT", Side.BUY, OrderType.LIMIT, 200.00, 150, 2_000))
    assert len(fills) == 2
    assert fills[0].maker_order_id == "ICE1" and fills[0].quantity == 100
    assert fills[1].maker_order_id == "LIT1" and fills[1].quantity == 50

    l3_asks = book.snapshot_l3()["asks"]
    assert l3_asks[0][1] == "LIT1" and l3_asks[0][2] == 50
    assert l3_asks[1][1] == "ICE1" and l3_asks[1][2] == 100 and l3_asks[1][3] == 100


def test_fix_parser_valid_messages_and_checksum_verification():
    raw_d = FIXTagValueParser.build_fix_frame(
        [
            (8, "FIX.4.4"),
            (35, "D"),
            (34, "42"),
            (11, "CLORD-42"),
            (55, "NVDA"),
            (54, "1"),
            (38, "500"),
            (44, "125.50"),
            (60, "1700000000000000000"),
        ],
        delimiter="|",
    )
    msg = FIXTagValueParser.parse(raw_d)
    assert msg.msg_type == "D"
    assert msg.seq_num == 42
    tick = FIXTagValueParser.normalize_to_tick(msg)
    assert tick.symbol == "NVDA"
    assert tick.side == Side.BUY
    assert tick.price == pytest.approx(125.50)
    assert tick.quantity == 500
    assert tick.event_type == "NEW"


def test_fix_parser_rejects_corrupted_checksum():
    frame = FIXTagValueParser.build_fix_frame(
        [(8, "FIX.4.2"), (35, "X"), (34, "7"), (55, "SPY"), (269, "0"), (270, "500.25"), (271, "100")],
        delimiter="|",
    )
    corrupted = frame[:-7] + "10=999|"
    with pytest.raises(FIXParserError, match="CheckSum mismatch"):
        FIXTagValueParser.parse(corrupted)


def test_binary_tick_codec_roundtrip():
    original = NormalizedTick(
        sequence_num=1048576,
        timestamp_ns=1710000123456789000,
        symbol="UST10Y",
        event_type="TRADE",
        side=Side.SELL,
        price=99.8750,
        quantity=2500,
        venue_id=4,
    )
    wire = BinaryTickCodec.encode(original)
    assert len(wire) == BinaryTickCodec.FRAME_SIZE
    decoded = BinaryTickCodec.decode(wire)
    assert decoded.sequence_num == original.sequence_num
    assert decoded.timestamp_ns == original.timestamp_ns
    assert decoded.symbol == original.symbol
    assert decoded.event_type == original.event_type
    assert decoded.side == original.side
    assert decoded.price == pytest.approx(original.price)
    assert decoded.quantity == original.quantity
    assert decoded.venue_id == original.venue_id


def test_sequence_gap_recovery_buffer_arbitration():
    buf = SequenceGapRecoveryBuffer(initial_seq=1)
    t1 = NormalizedTick(1, 100, "AAPL", "NEW", Side.BUY, 100.0, 10)
    t2 = NormalizedTick(2, 200, "AAPL", "NEW", Side.BUY, 100.1, 10)
    t3 = NormalizedTick(3, 300, "AAPL", "NEW", Side.BUY, 100.2, 10)
    t4 = NormalizedTick(4, 400, "AAPL", "NEW", Side.BUY, 100.3, 10)

    # Ingest 1 in-order
    out1 = buf.ingest(t1)
    assert [x.sequence_num for x in out1] == [1]

    # Packet 2 dropped on A-feed; 3 and 4 arrive out-of-order
    assert buf.ingest(t4) == []
    assert buf.ingest(t3) == []
    assert buf.missing_sequences == [2]
    assert buf.buffered_count == 2

    # Duplicate packet 1 arrives on B-feed -> ignored
    assert buf.ingest(t1) == []
    assert buf.duplicates_dropped == 1

    # Retransmission delivers missing seq 2 -> drains [2, 3, 4] atomically
    drained = buf.ingest(t2)
    assert [x.sequence_num for x in drained] == [2, 3, 4]
    assert buf.buffered_count == 0
    assert buf.expected_seq == 5


def test_order_book_imbalance_and_micro_price_invariants():
    book = _seed_book()
    snap = book.snapshot_l2()
    # Best bid 100.00 has 500 shares; best ask 100.10 has 400 shares
    obi_l1 = order_book_imbalance(500, 400)
    assert obi_l1 == pytest.approx((500 - 400) / 900.0)
    assert -1.0 <= multi_level_obi(snap, levels=2) <= 1.0

    mp = weighted_mid_price(snap)
    # Since bid_qty (500) > ask_qty (400), micro-price must be strictly above mid_price (100.05)
    assert book.mid_price < mp < book.best_ask
    assert mp == pytest.approx((500 * 100.10 + 400 * 100.00) / 900.0)


def test_roll_effective_spread_and_kyle_lambda():
    # Alternating bid-ask bounce between 100.00 and 100.10
    bounce_prices = [100.00, 100.10, 100.00, 100.10, 100.00, 100.10, 100.00, 100.10]
    est_spread = roll_effective_spread(bounce_prices)
    assert est_spread > 0.08

    # Kyle's lambda synthetic linear price impact: dP = 0.002 * OFI
    flows = [-200.0, -100.0, 0.0, 100.0, 200.0, 300.0]
    d_prices = [0.002 * x for x in flows]
    lam = kyle_lambda(d_prices, flows)
    assert lam == pytest.approx(0.002)


def test_vpin_toxicity_and_realized_volatility():
    # Balanced flow -> low VPIN; one-sided aggressive flow -> VPIN near 1.0
    one_sided_px = [100.0 + 0.05 * i for i in range(25)]
    vols = [100.0] * 25
    vpin_high = vpin_toxicity(one_sided_px, vols, bucket_volume=200.0, window_buckets=5)
    assert vpin_high == pytest.approx(1.0)

    rvol = realized_volatility(one_sided_px, periods_per_year=252.0)
    assert rvol > 0.0


def test_twap_vwap_and_almgren_chriss_schedules():
    twap = twap_schedule(total_quantity=1_000, num_slices=4, start_ts_ns=1_000, interval_ns=100)
    assert sum(s.quantity for s in twap) == 1_000
    assert [s.quantity for s in twap] == [250, 250, 250, 250]

    vwap = vwap_schedule(
        total_quantity=1_000,
        volume_profileWeights=[0.4, 0.1, 0.1, 0.4],
        start_ts_ns=1_000,
        interval_ns=100,
    )
    assert sum(s.quantity for s in vwap) == 1_000
    assert [s.quantity for s in vwap] == [400, 100, 100, 400]

    ac_high_risk = almgren_chriss_trajectory(
        total_quantity=1_000,
        num_slices=5,
        risk_aversion=1e-2,
        volatility=0.02,
        temporary_impact=1e-4,
    )
    assert sum(ac_high_risk) == 1_000
    # Risk-averse Almgren-Chriss front-loads execution: first slice > last slice
    assert ac_high_risk[0] > ac_high_risk[-1]


def test_simulate_schedule_execution_implementation_shortfall():
    book = _seed_book()
    sched = twap_schedule(total_quantity=300, num_slices=3, start_ts_ns=5_000, interval_ns=100)
    report = simulate_schedule_execution(book, Side.BUY, sched)
    assert report.executed_quantity == 300
    assert report.arrival_mid_price == pytest.approx(100.05)
    assert report.vwap_execution_price == pytest.approx(100.10)
    assert report.implementation_shortfall_bps > 0.0
