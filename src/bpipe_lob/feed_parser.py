"""
Deterministic FIX 4.2/4.4 Tag-Value & SBE-inspired Binary Frame Normalizer
with Sequence-Gap Detection and Replay Buffer Recovery.

Supported FIX Message Types (Tag 35):
- 35=D : NewOrderSingle
- 35=8 : ExecutionReport
- 35=W : MarketDataSnapshotFullRefresh
- 35=X : MarketDataIncrementalRefresh
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import Dict, List, Optional, Tuple

from .orderbook import Order, OrderType, Side


SOH = "\x01"


class FIXParserError(ValueError):
    """Raised when a FIX frame fails structural or checksum validation."""


@dataclass(frozen=True)
class FIXMessage:
    """Parsed FIX 4.2/4.4 message with validated checksum."""

    begin_string: str
    msg_type: str
    seq_num: int
    tags: Dict[int, str]
    raw_checksum: int


@dataclass(frozen=True)
class NormalizedTick:
    """
    Canonical normalized market data event emitted by B-PIPE / SBE / FIX feed handlers.
    Prices are represented in float and fixed-point (1e-4 ticks) for zero-drift transport.
    """

    sequence_num: int
    timestamp_ns: int
    symbol: str
    event_type: str  # "NEW", "CANCEL", "MODIFY", "TRADE", "SNAPSHOT"
    side: Side
    price: float
    quantity: int
    order_id: str = ""
    venue_id: int = 1


class FIXTagValueParser:
    """
    Deterministic FIX 4.2 / 4.4 tag-value parser and normalizer.
    Accepts either standard SOH (`\\x01`) or pipe (`|`) delimited wire frames.
    """

    @staticmethod
    def compute_checksum(payload_before_10: str) -> int:
        """Compute modulo-256 sum of ASCII bytes up to Tag 10."""
        return sum(payload_before_10.encode("ascii")) % 256

    @classmethod
    def build_fix_frame(cls, tags: List[Tuple[int, str]], delimiter: str = SOH) -> str:
        """
        Construct a valid FIX message string with automatic Tag 9 (BodyLength)
        and Tag 10 (CheckSum) computation.
        """
        begin_str = "FIX.4.4"
        body_pairs: List[Tuple[int, str]] = []
        for k, v in tags:
            if k == 8:
                begin_str = v
            elif k in (9, 10):
                continue
            else:
                body_pairs.append((k, str(v)))

        body_str = "".join(f"{k}={v}{delimiter}" for k, v in body_pairs)
        # BodyLength is byte length of body when using single-byte SOH
        body_soh = "".join(f"{k}={v}{SOH}" for k, v in body_pairs)
        body_len = len(body_soh.encode("ascii"))

        prefix_soh = f"8={begin_str}{SOH}9={body_len}{SOH}{body_soh}"
        checksum = cls.compute_checksum(prefix_soh)

        prefix_delim = f"8={begin_str}{delimiter}9={body_len}{delimiter}{body_str}"
        return f"{prefix_delim}10={checksum:03d}{delimiter}"

    @classmethod
    def parse(cls, raw_frame: str, validate_checksum: bool = True) -> FIXMessage:
        """Parse and validate a raw FIX 4.2/4.4 frame."""
        if not raw_frame:
            raise FIXParserError("Empty FIX frame")

        delimiter = SOH if SOH in raw_frame else "|"
        # Normalize to SOH for accurate checksum calculation
        soh_frame = raw_frame.replace("|", SOH) if delimiter == "|" else raw_frame

        idx_10 = soh_frame.rfind(f"{SOH}10=")
        if idx_10 == -1:
            raise FIXParserError("Missing mandatory FIX CheckSum tag (10=)")

        payload_before_10 = soh_frame[: idx_10 + 1]
        expected_cs = cls.compute_checksum(payload_before_10)

        tokens = [t for t in soh_frame.split(SOH) if t]
        tags: Dict[int, str] = {}
        for token in tokens:
            if "=" not in token:
                raise FIXParserError(f"Malformed tag=value pair: {token}")
            k_str, v_str = token.split("=", 1)
            try:
                k = int(k_str)
            except ValueError as exc:
                raise FIXParserError(f"Non-integer FIX tag: {k_str}") from exc
            tags[k] = v_str

        if 8 not in tags or 35 not in tags or 34 not in tags or 10 not in tags:
            raise FIXParserError("Missing mandatory header/trailer tags (8, 34, 35, 10)")

        wire_cs = int(tags[10])
        if validate_checksum and wire_cs != expected_cs:
            raise FIXParserError(
                f"FIX CheckSum mismatch: wire={wire_cs:03d}, computed={expected_cs:03d}"
            )

        return FIXMessage(
            begin_string=tags[8],
            msg_type=tags[35],
            seq_num=int(tags[34]),
            tags=tags,
            raw_checksum=wire_cs,
        )

    @classmethod
    def normalize_to_tick(cls, msg: FIXMessage) -> NormalizedTick:
        """
        Convert a parsed FIX message (35=D, 35=8, 35=W, 35=X) into a canonical
        NormalizedTick event suitable for the LimitOrderBook engine.
        """
        tags = msg.tags
        symbol = tags.get(55, "UNKNOWN")
        ts_ns = int(tags.get(60, "0"))

        # Side mapping: 54=1 (Buy), 54=2 (Sell) or 269=0 (Bid), 269=1 (Offer)
        if 54 in tags:
            side = Side.BUY if tags[54] == "1" else Side.SELL
        elif 269 in tags:
            side = Side.BUY if tags[269] == "0" else Side.SELL
        else:
            side = Side.BUY

        if msg.msg_type == "D":  # NewOrderSingle
            price = float(tags.get(44, "0.0"))
            qty = int(float(tags.get(38, "0")))
            order_id = tags.get(11, f"FIX-{msg.seq_num}")
            return NormalizedTick(
                sequence_num=msg.seq_num,
                timestamp_ns=ts_ns,
                symbol=symbol,
                event_type="NEW",
                side=side,
                price=price,
                quantity=qty,
                order_id=order_id,
            )
        elif msg.msg_type == "8":  # ExecutionReport
            exec_type = tags.get(150, "0")  # 0=New, 4=Canceled, F=Trade
            ev_map = {"0": "NEW", "4": "CANCEL", "5": "MODIFY", "F": "TRADE", "2": "TRADE"}
            event_type = ev_map.get(exec_type, "TRADE")
            price = float(tags.get(31, tags.get(44, "0.0")))
            qty = int(float(tags.get(32, tags.get(38, "0"))))
            order_id = tags.get(37, tags.get(11, f"EXEC-{msg.seq_num}"))
            return NormalizedTick(
                sequence_num=msg.seq_num,
                timestamp_ns=ts_ns,
                symbol=symbol,
                event_type=event_type,
                side=side,
                price=price,
                quantity=qty,
                order_id=order_id,
            )
        elif msg.msg_type in ("W", "X"):  # MarketDataSnapshot / IncrementalRefresh
            # Tag 279: 0=New, 1=Change, 2=Delete
            md_action = tags.get(279, "0")
            md_type = tags.get(269, "0")  # 0=Bid, 1=Offer, 2=Trade
            if md_type == "2":
                event_type = "TRADE"
            elif md_action == "2":
                event_type = "CANCEL"
            elif md_action == "1":
                event_type = "MODIFY"
            else:
                event_type = "SNAPSHOT" if msg.msg_type == "W" else "NEW"

            price = float(tags.get(270, "0.0"))
            qty = int(float(tags.get(271, "0")))
            order_id = tags.get(37, f"MD-{msg.seq_num}")
            return NormalizedTick(
                sequence_num=msg.seq_num,
                timestamp_ns=ts_ns,
                symbol=symbol,
                event_type=event_type,
                side=side,
                price=price,
                quantity=qty,
                order_id=order_id,
            )
        else:
            raise FIXParserError(f"Unsupported FIX MsgType (35={msg.msg_type})")


class BinaryTickCodec:
    """
    Zero-allocation-oriented compact binary SBE-style wire frame encoder/decoder.

    Wire layout (44 bytes, little-endian `<QQ8sBBqIH`):
      - sequence_num  : uint64 (8B)
      - timestamp_ns  : uint64 (8B)
      - symbol        : 8-byte ASCII padded (8B)
      - event_code    : uint8  (1B) [1=NEW, 2=CANCEL, 3=MODIFY, 4=TRADE, 5=SNAPSHOT]
      - side_code     : uint8  (1B) [1=BUY, 2=SELL]
      - price_mantissa: int64  (8B) [fixed-point price * 10^4]
      - quantity      : uint32 (4B)
      - venue_id      : uint16 (2B)
    Total struct size = 40 bytes.
    """

    STRUCT_FMT = "<QQ8sBBqIH"
    FRAME_SIZE = struct.calcsize(STRUCT_FMT)

    _EVENT_TO_CODE = {"NEW": 1, "CANCEL": 2, "MODIFY": 3, "TRADE": 4, "SNAPSHOT": 5}
    _CODE_TO_EVENT = {v: k for k, v in _EVENT_TO_CODE.items()}

    @classmethod
    def encode(cls, tick: NormalizedTick) -> bytes:
        sym_bytes = tick.symbol.encode("ascii")[:8].ljust(8, b"\x00")
        ev_code = cls._EVENT_TO_CODE.get(tick.event_type, 1)
        side_code = 1 if tick.side == Side.BUY else 2
        price_mantissa = int(round(tick.price * 10_000))
        return struct.pack(
            cls.STRUCT_FMT,
            tick.sequence_num,
            tick.timestamp_ns,
            sym_bytes,
            ev_code,
            side_code,
            price_mantissa,
            tick.quantity,
            tick.venue_id,
        )

    @classmethod
    def decode(cls, buffer: bytes) -> NormalizedTick:
        if len(buffer) != cls.FRAME_SIZE:
            raise ValueError(f"Expected {cls.FRAME_SIZE}-byte binary frame, got {len(buffer)}")
        seq, ts_ns, sym_bytes, ev_code, side_code, mantissa, qty, venue = struct.unpack(
            cls.STRUCT_FMT, buffer
        )
        symbol = sym_bytes.rstrip(b"\x00").decode("ascii")
        event_type = cls._CODE_TO_EVENT.get(ev_code, "NEW")
        side = Side.BUY if side_code == 1 else Side.SELL
        return NormalizedTick(
            sequence_num=seq,
            timestamp_ns=ts_ns,
            symbol=symbol,
            event_type=event_type,
            side=side,
            price=round(mantissa / 10_000.0, 4),
            quantity=qty,
            order_id=f"BIN-{seq}",
            venue_id=venue,
        )


class SequenceGapRecoveryBuffer:
    """
    Deterministic sequence-number arbitrator and out-of-order replay buffer
    modeling institutional UDP multicast feed recovery (A/B feed arbitration).

    Guarantees strictly monotonic in-order delivery of `NormalizedTick` packets.
    Out-of-order packets ahead of `expected_seq` are staged in a heap/dict buffer;
    once missing sequence numbers arrive via retransmission or B-feed, the buffer
    automatically drains contiguous ticks in exact sequence order.
    """

    def __init__(self, initial_seq: int = 1) -> None:
        self.expected_seq: int = initial_seq
        self._staging: Dict[int, NormalizedTick] = {}
        self.duplicates_dropped: int = 0
        self.gaps_detected: int = 0

    @property
    def buffered_count(self) -> int:
        return len(self._staging)

    @property
    def missing_sequences(self) -> List[int]:
        if not self._staging:
            return []
        max_staged = max(self._staging.keys())
        return [
            s for s in range(self.expected_seq, max_staged) if s not in self._staging
        ]

    def ingest(self, tick: NormalizedTick) -> List[NormalizedTick]:
        """
        Ingest a packet and return any contiguous in-order sequence of ticks
        ready for the LimitOrderBook engine.
        """
        seq = tick.sequence_num
        if seq < self.expected_seq or seq in self._staging:
            self.duplicates_dropped += 1
            return []

        if seq > self.expected_seq:
            self.gaps_detected += 1
            self._staging[seq] = tick
            return []

        # seq == self.expected_seq: emit immediately and drain contiguous staged packets
        ready: List[NormalizedTick] = [tick]
        self.expected_seq += 1
        while self.expected_seq in self._staging:
            ready.append(self._staging.pop(self.expected_seq))
            self.expected_seq += 1
        return ready
