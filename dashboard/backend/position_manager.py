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

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import auto_trader_db as db


# Safety configuration - tune these before going live
MAX_CONCURRENT_POSITIONS = 5
MAX_POSITION_PCT_OF_PORTFOLIO = 0.20  # 20% max per position (fallback when no risk context)
MIN_SIGNAL_PROBA = 0.55  # Don't trade signals below this
DEFAULT_HOLDING_DAYS = 5
MAX_SIGNAL_AGE_DAYS = 3  # Refuse to trade signals older than this

# Risk layer config (shared with src.risk; copied here so the dashboard venv
# doesn't depend on the capstone pipeline being importable).
RISK_LAYER_ENABLED = True
RISK_HISTORY_DAYS = 210  # ~10 months of trading days for vol + correlation + MA200


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


def calculate_position_size(buying_power: float, ref_price: float, weight: Optional[float] = None) -> int:
    """
    Calculate share quantity for a new position.

    Args:
        buying_power: Account buying power in $.
        ref_price: Reference price per share (typically ask or mid).
        weight: Fraction of buying_power to allocate. Defaults to
            MAX_POSITION_PCT_OF_PORTFOLIO when None (legacy behavior).

    Returns 0 if inputs are invalid.
    """
    if buying_power <= 0 or ref_price <= 0:
        return 0

    w = MAX_POSITION_PCT_OF_PORTFOLIO if weight is None else float(weight)
    if w <= 0:
        return 0
    notional = buying_power * w
    qty = int(notional // ref_price)
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


def select_signals_to_trade(
    predictions: list,
    max_new: Optional[int] = None,
    risk_context: Optional[dict] = None,
) -> tuple[list, dict]:
    """
    Given a sorted list of predictions (best first), pick which ones to trade
    AND assign each a position weight (fraction of buying power).

    Filters by:
    - Minimum signal probability
    - Available capacity
    - Per-ticker uniqueness (already have a position)

    If `risk_context` is provided (see build_risk_context()), per-pick weights
    come from src.risk.size_basket — vol-targeted with regime gate and
    correlation filter. Otherwise every pick gets MAX_POSITION_PCT_OF_PORTFOLIO.

    Returns (selections, diagnostics) where:
      - selections: list of (prediction, weight) tuples in rank order
      - diagnostics: regime info + per-pick weight breakdown for logging/UI
    """
    capacity = get_capacity()
    available = capacity["available_slots"]
    if max_new is not None:
        available = min(available, max_new)

    diagnostics: dict = {
        "regime": "unknown",
        "risk_layer": "off",
        "spy_close": None,
        "spy_ma200": None,
        "vix": None,
        "vix_pct_rank": None,
        "regime_multiplier": 1.0,
        "gross_weight": 0.0,
        "picks": [],
    }

    # Populate regime info up front whenever risk_context is available, so
    # the dashboard can always tell the user "what's the market regime today"
    # even when no picks survive the pending-orders / threshold filters.
    if RISK_LAYER_ENABLED and risk_context is not None:
        try:
            from src.risk import regime_label, vix_percentile_rank
            spy_h = risk_context.get("spy_history")
            vix_h = risk_context.get("vix_history")
            asof = pd.Timestamp(date.today())
            spy_upto = spy_h.loc[:asof].dropna() if spy_h is not None else pd.Series(dtype=float)
            if len(spy_upto) >= 200:
                diagnostics["spy_close"] = float(spy_upto.iloc[-1])
                diagnostics["spy_ma200"] = float(spy_upto.tail(200).mean())
            vix_upto = vix_h.loc[:asof].dropna() if vix_h is not None else pd.Series(dtype=float)
            if len(vix_upto) > 0:
                diagnostics["vix"] = float(vix_upto.iloc[-1])
                diagnostics["vix_pct_rank"] = vix_percentile_rank(vix_h, asof)
            diagnostics["regime"] = regime_label(
                diagnostics["spy_close"], diagnostics["spy_ma200"],
                diagnostics["vix"], diagnostics["vix_pct_rank"],
            )
            diagnostics["regime_multiplier"] = 0.5 if diagnostics["regime"] == "risk_off" else 1.0
            diagnostics["risk_layer"] = "on"
        except Exception as e:
            print(f"[position_manager] regime snapshot failed: {e}")

    if available <= 0:
        return [], diagnostics

    today = date.today()
    pre_selected: list = []
    for pred in predictions:
        if len(pre_selected) >= available:
            break
        ticker = pred.get("ticker")
        proba = pred.get("y_proba", 0)

        signal_date_str = pred.get("signal_date")
        if signal_date_str:
            try:
                signal_date = datetime.strptime(signal_date_str, "%Y-%m-%d").date()
                age_days = (today - signal_date).days
                if age_days > MAX_SIGNAL_AGE_DAYS:
                    continue
            except (ValueError, TypeError):
                continue

        allowed, _ = can_open_position(ticker, proba)
        if allowed:
            pre_selected.append(pred)

    if not pre_selected:
        return [], diagnostics

    # Apply risk layer if context provided
    if RISK_LAYER_ENABLED and risk_context is not None:
        try:
            from src.risk import size_basket  # available via PROJECT_ROOT on sys.path
        except ImportError:
            # Fall back gracefully if the capstone pipeline isn't on the path.
            size_basket = None

        if size_basket is not None:
            tickers = [p["ticker"] for p in pre_selected]
            asof = pd.Timestamp(today)
            weights, diag = size_basket(
                tickers,
                risk_context["prices_long"],
                risk_context["spy_history"],
                risk_context["vix_history"],
                asof,
            )
            diagnostics.update({
                "regime": diag.regime,
                "risk_layer": "on",
                "spy_close": diag.spy_close,
                "spy_ma200": diag.spy_ma200,
                "vix": diag.vix,
                "vix_pct_rank": diag.vix_pct_rank,
                "regime_multiplier": diag.regime_multiplier,
                "gross_weight": diag.gross_weight,
                "picks": [
                    {
                        "ticker": p.ticker,
                        "realized_vol_20d": p.realized_vol_20d,
                        "raw_weight": p.raw_weight,
                        "final_weight": p.final_weight,
                        "dropped_reason": p.dropped_reason,
                    }
                    for p in diag.picks
                ],
            })
            selections = [(p, weights.get(p["ticker"], 0.0)) for p in pre_selected]
            # Drop zero-weight picks (correlation filter, no vol data, etc.)
            selections = [(p, w) for p, w in selections if w > 0]
            return selections, diagnostics

    # Fallback: flat MAX_POSITION_PCT_OF_PORTFOLIO per pick
    selections = [(p, MAX_POSITION_PCT_OF_PORTFOLIO) for p in pre_selected]
    diagnostics["picks"] = [
        {"ticker": p["ticker"], "final_weight": MAX_POSITION_PCT_OF_PORTFOLIO, "raw_weight": MAX_POSITION_PCT_OF_PORTFOLIO}
        for p in pre_selected
    ]
    diagnostics["gross_weight"] = sum(w for _, w in selections)
    return selections, diagnostics


# Risk context builder


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_VIX_CACHE_PATH = _PROJECT_ROOT / "Data" / "raw" / "vix.parquet"


def _load_vix_history() -> pd.Series:
    """Load cached VIX series from the capstone data layer. Empty Series if missing."""
    if not _VIX_CACHE_PATH.exists():
        return pd.Series(dtype=float, name="vix_close")
    df = pd.read_parquet(_VIX_CACHE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["vix_close"].sort_index()


def build_risk_context(
    data_client,
    tickers: list[str],
    days: int = RISK_HISTORY_DAYS,
) -> Optional[dict]:
    """
    Fetch the inputs the risk module needs: long-format OHLCV for `tickers`
    (plus SPY) and a VIX series. Returns None if Alpaca bars are unavailable.

    Tickers should include SPY; if not, it's added automatically.
    """
    if data_client is None:
        return None

    universe = list(set(tickers) | {"SPY"})
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now()
        # Pad calendar days for weekends/holidays
        start = end - timedelta(days=int(days * 1.6))
        req = StockBarsRequest(
            symbol_or_symbols=universe,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = data_client.get_stock_bars(req)
    except Exception as e:
        print(f"[position_manager] build_risk_context: Alpaca bars fetch failed: {e}")
        return None

    rows: list[dict] = []
    df_bars = bars.df if hasattr(bars, "df") else None
    if df_bars is None or df_bars.empty:
        return None

    # bars.df has a (symbol, timestamp) multi-index
    flat = df_bars.reset_index()
    for _, r in flat.iterrows():
        ts = pd.Timestamp(r["timestamp"])
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        rows.append({
            "date": ts.normalize(),
            "ticker": r["symbol"],
            "close": float(r["close"]),
        })
    prices_long = pd.DataFrame(rows)
    if prices_long.empty:
        return None

    spy_history = (
        prices_long[prices_long["ticker"] == "SPY"]
        .set_index("date")["close"]
        .sort_index()
    )
    vix_history = _load_vix_history()

    return {
        "prices_long": prices_long,
        "spy_history": spy_history,
        "vix_history": vix_history,
        "asof": prices_long["date"].max(),
    }


def get_latest_signal_age() -> Optional[int]:
    """Returns days since the latest model prediction, or None if unavailable."""
    try:
        from pathlib import Path
        import pandas as pd
        backend_dir = Path(__file__).parent
        project_root = backend_dir.parent.parent
        pred_path = project_root / "results" / "predictions.parquet"
        df = pd.read_parquet(pred_path)
        df["date"] = pd.to_datetime(df["date"])
        latest = df["date"].max().date()
        return (date.today() - latest).days
    except Exception:
        return None


def summarize_portfolio() -> dict:
    """Comprehensive snapshot of autonomous trader state."""
    capacity = get_capacity()
    stats = db.get_performance_stats()
    exits_due = get_exits_due()
    open_positions = db.get_open_positions()
    signal_age = get_latest_signal_age()

    return {
        "capacity": capacity,
        "stats": stats,
        "exits_due_today": len(exits_due),
        "exits_due_tickers": [p["ticker"] for p in exits_due],
        "signal_age_days": signal_age,
        "signal_is_fresh": signal_age is not None and signal_age <= MAX_SIGNAL_AGE_DAYS,
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
