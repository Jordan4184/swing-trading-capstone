"""
Real-time probability heatmap for autonomous trader visualization.

Given a ticker, returns the model\'s predicted y_proba across a range of
hypothetical "today close" prices. As the intraday price moves, the user can
see at which prices the model would BUY vs HOLD vs reject the signal.

The model never saw intraday prices in training - all features are end-of-day.
This heatmap therefore shows: "if today closed at price X, what would the model
predict?" It is NOT a real-time live probability; it is a what-if curve.

Cached for 60 seconds per ticker to avoid hammering the model.
"""

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_v1.joblib"
DATA_CACHE = PROJECT_ROOT / "Data" / "raw" / "ohlcv.parquet"

# Same FEATURE_COLUMNS order as src/models.py
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "rsi_14",
    "bb_pct",
    "volume_ratio_20d",
    "spy_return_1d",
    "excess_return_1d",
]

# Heatmap parameters
N_PRICE_POINTS = 30
CACHE_TTL_SECONDS = 60
DEFAULT_RANGE_PCT = 0.05  # +/- 5% around current price

# Module-level cache for the loaded model and ohlcv data
_model_payload: Optional[dict] = None
_ohlcv_df: Optional[pd.DataFrame] = None
_curve_cache: dict[str, tuple[float, dict]] = {}


def _load_model_once():
    global _model_payload
    if _model_payload is None:
        _model_payload = joblib.load(MODEL_PATH)
    return _model_payload


def _load_ohlcv_once():
    global _ohlcv_df
    if _ohlcv_df is None:
        df = pd.read_parquet(DATA_CACHE)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        _ohlcv_df = df
    return _ohlcv_df


def reload_ohlcv():
    """Force reload of OHLCV cache - call this after predict-cycle updates the parquet."""
    global _ohlcv_df, _curve_cache
    _ohlcv_df = None
    _curve_cache.clear()


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI for a single point (the last value of the series)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi) > 0 else float("nan")


def _bollinger_pct_b(series: pd.Series, window: int = 20) -> float:
    """Compute Bollinger %B for the last value of the series."""
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    bb = (series - lower) / (upper - lower)
    return float(bb.iloc[-1]) if len(bb) > 0 else float("nan")


def _compute_features_for_hypothetical_close(
    ticker_history: pd.DataFrame,
    spy_history: pd.DataFrame,
    hypothetical_close: float,
    volume_today: float,
) -> dict:
    """
    Given the ticker\'s historical bars and a hypothetical "today close" price,
    compute all 11 features as the model expects them.

    ticker_history: last ~70 days, INCLUDING today\'s placeholder row.
    The last row\'s close will be replaced with hypothetical_close.
    """
    # Make a copy and replace the last close with the hypothetical price
    df = ticker_history.copy()
    df.loc[df.index[-1], "close"] = hypothetical_close
    df.loc[df.index[-1], "volume"] = volume_today

    # Multi-horizon returns based on close at index -1 vs index -N-1
    closes = df["close"].values
    last = closes[-1]
    feats = {}
    for h in [1, 5, 20, 60]:
        if len(closes) > h:
            feats[f"return_{h}d"] = (last / closes[-h - 1]) - 1
        else:
            feats[f"return_{h}d"] = np.nan

    # Volatility: rolling std of 1-day returns
    returns = df["close"].pct_change()
    for window in [20, 60]:
        if len(returns) >= window:
            feats[f"volatility_{window}d"] = float(returns.tail(window).std())
        else:
            feats[f"volatility_{window}d"] = np.nan

    # RSI(14)
    feats["rsi_14"] = _rsi(df["close"], period=14)

    # Bollinger %B (20-day)
    feats["bb_pct"] = _bollinger_pct_b(df["close"], window=20)

    # Volume ratio (today vs 20-day rolling avg)
    if len(df) >= 20:
        avg_vol = df["volume"].tail(20).mean()
        feats["volume_ratio_20d"] = volume_today / avg_vol if avg_vol > 0 else np.nan
    else:
        feats["volume_ratio_20d"] = np.nan

    # SPY return (1-day): SPY is shared, computed from spy_history
    if len(spy_history) >= 2:
        spy_today = float(spy_history["close"].iloc[-1])
        spy_yesterday = float(spy_history["close"].iloc[-2])
        feats["spy_return_1d"] = (spy_today / spy_yesterday) - 1
    else:
        feats["spy_return_1d"] = np.nan

    # Excess return: this ticker\'s 1d return minus SPY\'s 1d return
    feats["excess_return_1d"] = (
        feats["return_1d"] - feats["spy_return_1d"]
        if not (np.isnan(feats["return_1d"]) or np.isnan(feats["spy_return_1d"]))
        else np.nan
    )

    return feats


def compute_heatmap(
    ticker: str,
    current_price: float,
    range_pct: float = DEFAULT_RANGE_PCT,
) -> dict:
    """
    Compute the probability curve for a ticker.

    Args:
        ticker: e.g. "NVDA"
        current_price: latest known intraday price (used to center the range)
        range_pct: how wide to span around current_price (default +/- 5%)

    Returns:
        dict with: ticker, n_points, prices, probabilities, current_price,
                   model_version, computed_at, cache_status
    """
    # Check cache
    cache_key = f"{ticker}:{round(current_price, 2)}"
    now = time.time()
    if cache_key in _curve_cache:
        ts, cached = _curve_cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return {**cached, "cache_status": "hit"}

    payload = _load_model_once()
    model = payload["model"]
    feature_cols = payload["feature_columns"]
    model_version = payload["model_version"]

    df = _load_ohlcv_once()
    if df.empty:
        raise RuntimeError("OHLCV data not loaded")

    # Get ticker history - last 70 days
    ticker_df = df[df["ticker"] == ticker].sort_values("date").tail(70).copy()
    if len(ticker_df) < 65:
        raise RuntimeError(f"Insufficient history for {ticker}: only {len(ticker_df)} rows")

    spy_df = df[df["ticker"] == "SPY"].sort_values("date").tail(70).copy()

    # Use latest known volume as our "today\'s volume" assumption
    volume_today = float(ticker_df["volume"].iloc[-1])

    # Generate price range
    price_low = current_price * (1 - range_pct)
    price_high = current_price * (1 + range_pct)
    prices = np.linspace(price_low, price_high, N_PRICE_POINTS)

    # Compute features and probability for each hypothetical price
    feature_rows = []
    for price in prices:
        feats = _compute_features_for_hypothetical_close(
            ticker_df, spy_df, hypothetical_close=float(price), volume_today=volume_today,
        )
        feature_rows.append([feats[c] for c in feature_cols])

    X = np.array(feature_rows, dtype=float)
    # Replace any nans with the column median (model can\'t take NaN)
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        idx_nan = np.where(np.isnan(X))
        X[idx_nan] = np.take(col_medians, idx_nan[1])

    probabilities = model.predict_proba(X)[:, 1]

    # Find the index closest to current_price for the "current" marker
    current_idx = int(np.argmin(np.abs(prices - current_price)))

    result = {
        "ticker": ticker,
        "n_points": N_PRICE_POINTS,
        "prices": [round(float(p), 2) for p in prices],
        "probabilities": [round(float(p), 4) for p in probabilities],
        "current_price": round(float(current_price), 2),
        "current_idx": current_idx,
        "current_probability": round(float(probabilities[current_idx]), 4),
        "min_probability": round(float(probabilities.min()), 4),
        "max_probability": round(float(probabilities.max()), 4),
        "model_version": model_version,
        "computed_at": pd.Timestamp.now().isoformat(),
        "buy_threshold": 0.55,
        "yesterday_close": round(float(ticker_df["close"].iloc[-1]), 2),
    }

    _curve_cache[cache_key] = (now, result)
    return {**result, "cache_status": "miss"}
