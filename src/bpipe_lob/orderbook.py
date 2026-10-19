"""
Price-Time Priority (FIFO) Limit Order Book Engine.

Supports:
- Limit, Market, Immediate-or-Cancel (IOC), Post-Only, and Iceberg/Hidden orders
- O(1) order lookup by order_id for fast cancel and modify operations
- Automatic iceberg peak replenishment preserving price-time priority rules
- L1 (top-of-book), L2 (aggregated depth), and L3 (order-by-order) snapshots
- Deterministic trade execution fill events
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import bisect
from typing import Deque, Dict, List, Optional, Tuple


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    IOC = "IOC"
    POST_ONLY = "POST_ONLY"
    ICEBERG = "ICEBERG"


@dataclass
class Order:
    """Represents an individual order in the Limit Order Book."""

    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    timestamp_ns: int
    display_qty: Optional[int] = None
    hidden_qty: int = 0
    peak_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type != OrderType.MARKET and self.price <= 0:
            raise ValueError(f"Limit price must be positive, got {self.price}")
        if self.order_type == OrderType.ICEBERG:
            if self.peak_size is None or self.peak_size <= 0:
                raise ValueError("Iceberg order requires positive peak_size")
            visible = min(self.quantity, self.peak_size)
            self.display_qty = visible
            self.hidden_qty = self.quantity - visible
        else:
            if self.display_qty is None:
                self.display_qty = self.quantity

    @property
    def total_remaining(self) -> int:
        return (self.display_qty or 0) + self.hidden_qty


@dataclass(frozen=True)
class TradeFill:
    """Deterministic trade execution record emitted on order matching."""

    trade_id: int
    symbol: str
    price: float
    quantity: int
    aggressor_side: Side
    maker_order_id: str
    taker_order_id: str
    timestamp_ns: int


@dataclass(frozen=True)
class L2Level:
    """Aggregated price level representation for L2 market data feeds."""

    price: float
    visible_quantity: int
    hidden_quantity: int
    order_count: int


@dataclass(frozen=True)
class OrderBookSnapshot:
    """L2 depth snapshot at a specific nanosecond timestamp."""

    symbol: str
    timestamp_ns: int
    sequence_num: int
    bids: List[L2Level]
    asks: List[L2Level]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5 * (self.best_bid + self.best_ask)
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return round(self.best_ask - self.best_bid, 10)
        return None


class LimitOrderBook:
    """
    Deterministic Price-Time Priority (FIFO) Limit Order Book.

    Maintains sorted price levels and FIFO queues at each price level.
    Iceberg orders display up to `peak_size` shares; when the visible portion
    is exhausted, a new visible clip is replenished from `hidden_qty` and
    placed at the back of the FIFO queue at that price level (matching standard
    exchange mechanics such as CME Globex / NASDAQ ITCH).
    """

    def __init__(self, symbol: str, tick_size: float = 0.01) -> None:
        self.symbol = symbol
        self.tick_size = tick_size
        # Bids: price -> deque of Order. Sorted keys stored ascending in _bid_prices (best bid is last)
        self._bids: Dict[float, Deque[Order]] = {}
        self._bid_prices: List[float] = []
        # Asks: price -> deque of Order. Sorted keys stored ascending in _ask_prices (best ask is first)
        self._asks: Dict[float, Deque[Order]] = {}
        self._ask_prices: List[float] = []
        # Fast lookup map: order_id -> Order
        self._orders: Dict[str, Order] = {}
        self._trade_seq: int = 0
        self._book_seq: int = 0
        self._last_ts_ns: int = 0
        self.trade_history: List[TradeFill] = []

    def _norm_price(self, price: float) -> float:
        return round(price, 8)

    @property
    def best_bid(self) -> Optional[float]:
        return self._bid_prices[-1] if self._bid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5 * (self.best_bid + self.best_ask)
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return round(self.best_ask - self.best_bid, 8)
        return None

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def submit_order(self, order: Order) -> List[TradeFill]:
        """
        Submit an incoming order to the matching engine.
        Returns a list of deterministic TradeFill events generated by matching.
        """
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol {order.symbol} does not match book {self.symbol}")
        if order.order_id in self._orders:
            raise ValueError(f"Duplicate order_id: {order.order_id}")

        self._book_seq += 1
        self._last_ts_ns = max(self._last_ts_ns, order.timestamp_ns)
        order.price = self._norm_price(order.price) if order.order_type != OrderType.MARKET else 0.0

        # Check Post-Only condition
        if order.order_type == OrderType.POST_ONLY:
            if order.side == Side.BUY and self.best_ask is not None and order.price >= self.best_ask:
                return []  # Rejected post-only that would cross book
            if order.side == Side.SELL and self.best_bid is not None and order.price <= self.best_bid:
                return []

        fills = self._match(order)

        # If residual remains and order can rest on book
        if order.total_remaining > 0 and order.order_type in (
            OrderType.LIMIT,
            OrderType.POST_ONLY,
            OrderType.ICEBERG,
        ):
            if order.order_type == OrderType.ICEBERG and order.peak_size is not None:
                visible = min(order.total_remaining, order.peak_size)
                order.display_qty = visible
                order.hidden_qty = order.total_remaining - visible
            else:
                order.display_qty = order.total_remaining
                order.hidden_qty = 0
            self._rest_order(order)

        return fills

    def _match(self, taker: Order) -> List[TradeFill]:
        fills: List[TradeFill] = []
        taker_remaining = taker.total_remaining

        if taker.side == Side.BUY:
            while taker_remaining > 0 and self._ask_prices:
                best_ask = self._ask_prices[0]
                if taker.order_type != OrderType.MARKET and taker.price < best_ask:
                    break
                level_queue = self._asks[best_ask]
                fills_at_level, taker_remaining = self._match_level(
                    taker, taker_remaining, best_ask, level_queue
                )
                fills.extend(fills_at_level)
                if not level_queue:
                    del self._asks[best_ask]
                    self._ask_prices.pop(0)
        else:
            while taker_remaining > 0 and self._bid_prices:
                best_bid = self._bid_prices[-1]
                if taker.order_type != OrderType.MARKET and taker.price > best_bid:
                    break
                level_queue = self._bids[best_bid]
                fills_at_level, taker_remaining = self._match_level(
                    taker, taker_remaining, best_bid, level_queue
                )
                fills.extend(fills_at_level)
                if not level_queue:
                    del self._bids[best_bid]
                    self._bid_prices.pop()

        # Update taker remaining quantities
        taker.quantity = taker_remaining
        if taker.order_type == OrderType.ICEBERG and taker.peak_size is not None:
            vis = min(taker_remaining, taker.peak_size)
            taker.display_qty = vis
            taker.hidden_qty = taker_remaining - vis
        else:
            taker.display_qty = taker_remaining
            taker.hidden_qty = 0

        return fills

    def _match_level(
        self,
        taker: Order,
        taker_remaining: int,
        match_price: float,
        level_queue: Deque[Order],
    ) -> Tuple[List[TradeFill], int]:
        fills: List[TradeFill] = []
        while taker_remaining > 0 and level_queue:
            maker = level_queue[0]
            maker_visible = maker.display_qty or 0
            exec_qty = min(taker_remaining, maker_visible)

            self._trade_seq += 1
            fill = TradeFill(
                trade_id=self._trade_seq,
                symbol=self.symbol,
                price=match_price,
                quantity=exec_qty,
                aggressor_side=taker.side,
                maker_order_id=maker.order_id,
                taker_order_id=taker.order_id,
                timestamp_ns=taker.timestamp_ns,
            )
            fills.append(fill)
            self.trade_history.append(fill)

            taker_remaining -= exec_qty
            maker.display_qty = maker_visible - exec_qty
            maker.quantity = maker.total_remaining

            if maker.display_qty == 0:
                if maker.hidden_qty > 0 and maker.peak_size is not None:
                    # Iceberg replenishment: replenish visible slice and move to back of FIFO queue
                    replenish = min(maker.hidden_qty, maker.peak_size)
                    maker.display_qty = replenish
                    maker.hidden_qty -= replenish
                    maker.quantity = maker.total_remaining
                    level_queue.popleft()
                    level_queue.append(maker)
                else:
                    level_queue.popleft()
                    self._orders.pop(maker.order_id, None)

        return fills, taker_remaining

    def _rest_order(self, order: Order) -> None:
        self._orders[order.order_id] = order
        if order.side == Side.BUY:
            if order.price not in self._bids:
                self._bids[order.price] = deque()
                bisect.insort(self._bid_prices, order.price)
            self._bids[order.price].append(order)
        else:
            if order.price not in self._asks:
                self._asks[order.price] = deque()
                bisect.insort(self._ask_prices, order.price)
            self._asks[order.price].append(order)

    def cancel_order(self, order_id: str, timestamp_ns: int = 0) -> bool:
        """Cancel a resting order by ID. Returns True if found and removed."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return False

        self._book_seq += 1
        self._last_ts_ns = max(self._last_ts_ns, timestamp_ns)

        if order.side == Side.BUY:
            queue = self._bids.get(order.price)
            if queue is not None:
                queue.remove(order)
                if not queue:
                    del self._bids[order.price]
                    self._bid_prices.remove(order.price)
        else:
            queue = self._asks.get(order.price)
            if queue is not None:
                queue.remove(order)
                if not queue:
                    del self._asks[order.price]
                    self._ask_prices.remove(order.price)
        return True

    def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[int] = None,
        new_price: Optional[float] = None,
        timestamp_ns: int = 0,
    ) -> List[TradeFill]:
        """
        Modify a resting order.
        - Quantity decrease at the same price preserves FIFO time priority.
        - Quantity increase or price change loses time priority (cancel + replace).
        """
        existing = self._orders.get(order_id)
        if existing is None:
            raise KeyError(f"Order {order_id} not found in order book")

        target_price = self._norm_price(new_price) if new_price is not None else existing.price
        target_qty = new_quantity if new_quantity is not None else existing.total_remaining

        if target_qty <= 0:
            self.cancel_order(order_id, timestamp_ns)
            return []

        # Preserve queue priority if price is unchanged and total quantity decreases
        if target_price == existing.price and target_qty <= existing.total_remaining:
            self._book_seq += 1
            self._last_ts_ns = max(self._last_ts_ns, timestamp_ns)
            existing.quantity = target_qty
            if existing.order_type == OrderType.ICEBERG and existing.peak_size is not None:
                vis = min(existing.display_qty or existing.peak_size, target_qty)
                existing.display_qty = vis
                existing.hidden_qty = target_qty - vis
            else:
                existing.display_qty = target_qty
                existing.hidden_qty = 0
            return []

        # Otherwise cancel and resubmit (loses queue priority or crosses book)
        side = existing.side
        otype = existing.order_type
        peak = existing.peak_size
        self.cancel_order(order_id, timestamp_ns)
        replacement = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            order_type=otype,
            price=target_price,
            quantity=target_qty,
            timestamp_ns=timestamp_ns or self._last_ts_ns,
            peak_size=peak,
        )
        return self.submit_order(replacement)

    def snapshot_l2(self, depth: int = 10) -> OrderBookSnapshot:
        """Generate an aggregated L2 market data snapshot up to `depth` levels."""
        bids: List[L2Level] = []
        for p in reversed(self._bid_prices[-depth:]):
            q = self._bids[p]
            vis_sum = sum(o.display_qty or 0 for o in q)
            hid_sum = sum(o.hidden_qty for o in q)
            bids.append(
                L2Level(
                    price=p,
                    visible_quantity=vis_sum,
                    hidden_quantity=hid_sum,
                    order_count=len(q),
                )
            )

        asks: List[L2Level] = []
        for p in self._ask_prices[:depth]:
            q = self._asks[p]
            vis_sum = sum(o.display_qty or 0 for o in q)
            hid_sum = sum(o.hidden_qty for o in q)
            asks.append(
                L2Level(
                    price=p,
                    visible_quantity=vis_sum,
                    hidden_quantity=hid_sum,
                    order_count=len(q),
                )
            )

        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp_ns=self._last_ts_ns,
            sequence_num=self._book_seq,
            bids=bids,
            asks=asks,
        )

    def snapshot_l3(self) -> Dict[str, List[ Tuple[float, str, int, int] ]]:
        """
        Return full L3 order-by-order depth queue.
        Each entry is (price, order_id, visible_qty, hidden_qty).
        """
        bids_l3: List[Tuple[float, str, int, int]] = []
        for p in reversed(self._bid_prices):
            for o in self._bids[p]:
                bids_l3.append((p, o.order_id, o.display_qty or 0, o.hidden_qty))

        asks_l3: List[Tuple[float, str, int, int]] = []
        for p in self._ask_prices:
            for o in self._asks[p]:
                asks_l3.append((p, o.order_id, o.display_qty or 0, o.hidden_qty))

        return {"bids": bids_l3, "asks": asks_l3}
