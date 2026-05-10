"""
Position Manager for the autonomous paper trading system.

Pure logic layer - no API calls, no scheduling. Answers:
- Can we open a new position for ticker X?
- Which positions are due for exit?
- What's our current capacity vs max positions?

Safety rules enforced here:
- Max concurrent positions (default 5)
- One open position per ticker at a time
- Minimum signal probability to trade
- Position sizing: max % of portfolio per trade
"""

from datetime import date, datetime
from typing import Optional

import auto_trader_db as db


# Safety configuration - tune these before going live
MAX_CONCURRENT_POSITIONS = 5
MAX_POSITION_PCT_OF_PORTFOLIO = 0.20  # 20% max per position
MIN_SIGNAL_PROBA = 0.55  # Don't trade signals below this
DEFAULT_HOLDING_DAYS = 5


class PositionManagerError(Exception):
    """Raised when position management rules are violated."""
    pass


def can_open_position(ticker: str, signal_proba: float) -> tuple[bool, str]:
    """
    Check if we can open a new position for this ticker right now.
    Returns (allowed, reason).
    """
    if signal_proba < MIN_SIGNAL_PROBA:
        return False, f"Signal proba {signal_proba:.3f} below threshold {MIN_SIGNAL_PROBA}"

    open_positions = db.get_open_positions()
    pending = db.get_pending_orders()

    # Check concurrent position limit (open + pending counts)
    total_committed = len(open_positions) + len(pending)
    if total_committed >= MAX_CONCURRENT_POSITIONS:
        return False, f"At max concurrent positions ({total_committed}/{MAX_CONCURRENT_POSITIONS})"

    # Check per-ticker uniqueness (don't double up on same ticker)
    for pos in open_positions:
        if pos["ticker"] == ticker:
            return False, f"Already have open position in {ticker}"
    for pending_order in pending:
        if pending_order["ticker"] == ticker:
            return False, f"Already have pending order for {ticker}"

    # Check today's intent for this ticker (don't double-record)
    today = date.today().isoformat()
    recent = db.get_recent_trades(limit=20)
    for r in recent:
        if r["ticker"] == ticker and r["signal_date"] == today:
            return False, f"Already recorded intent for {ticker} today"

    return True, "ok"


def calculate_position_size(buying_power: float, ref_price: float) -> int:
    """
    Calculate share quantity for a new position.
    Returns share count, or 0 if can't size meaningfully.
    """
    if buying_power <= 0 or ref_price <= 0:
        return 0

    max_notional = buying_power * MAX_POSITION_PCT_OF_PORTFOLIO
    qty = int(max_notional // ref_price)
    return max(0, qty)


def get_capacity() -> dict:
    """Return current capacity vs max."""
    open_positions = db.get_open_positions()
    pending = db.get_pending_orders()

    return {
        "max_positions": MAX_CONCURRENT_POSITIONS,
        "open_count": len(open_positions),
        "pending_count": len(pending),
        "available_slots": max(0, MAX_CONCURRENT_POSITIONS - len(open_positions) - len(pending)),
        "open_tickers": [p["ticker"] for p in open_positions],
        "pending_tickers": [p["ticker"] for p in pending],
    }


def get_exits_due(today_iso: Optional[str] = None) -> list:
    """Positions whose target_exit_date is today or earlier."""
    return db.get_positions_due_for_exit(today_iso)


def select_signals_to_trade(predictions: list, max_new: Optional[int] = None) -> list:
    """
    Given a sorted list of predictions (best first), pick which ones to trade.
    Filters by:
    - Minimum signal probability
    - Available capacity
    - Per-ticker uniqueness (already have a position)

    Returns list of (prediction, reason) tuples for traded picks.
    Skipped signals are NOT included.
    """
    capacity = get_capacity()
    available = capacity["available_slots"]
    if max_new is not None:
        available = min(available, max_new)

    if available <= 0:
        return []

    selected = []
    for pred in predictions:
        if len(selected) >= available:
            break
        ticker = pred.get("ticker")
        proba = pred.get("y_proba", 0)
        allowed, reason = can_open_position(ticker, proba)
        if allowed:
            selected.append((pred, "ok"))

    return selected


def summarize_portfolio() -> dict:
    """Comprehensive snapshot of autonomous trader state."""
    capacity = get_capacity()
    stats = db.get_performance_stats()
    exits_due = get_exits_due()
    open_positions = db.get_open_positions()

    return {
        "capacity": capacity,
        "stats": stats,
        "exits_due_today": len(exits_due),
        "exits_due_tickers": [p["ticker"] for p in exits_due],
        "open_positions_detail": [
            {
                "ticker": p["ticker"],
                "qty": p["qty"],
                "entry_price": p["entry_price"],
                "entry_filled_at": p["entry_filled_at"],
                "target_exit_date": p["target_exit_date"],
                "signal_proba": p["signal_proba"],
            }
            for p in open_positions
        ],
    }
