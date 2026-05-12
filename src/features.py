"""
Feature engineering for swing trading capstone.

Each feature family is motivated by a market phenomenon documented in EDA
or financial literature. See README for details.
"""

import numpy as np
import pandas as pd

# Helper: Applying per-ticker rolling computation


def _per_ticker(df: pd.DataFrame, func, *args, **kwargs) -> pd.Series:
    """
    Apply a function to each ticker data seperately and then concatenate
    Avoids leakage across tickers
    """

    return df.groupby("ticker", group_keys=False).apply(func, *args, **kwargs)


# Family 1: Returns at multiple horizons
# Motivation: EDA showed mean reversion at lag 1 and weak momentum at longer
# horizons. Multi-horizon returns let the model learn both regimes.


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1, 5, 20, 60 day trailing returns per ticker
    """
    df = df.sort_values(["ticker", "date"]).copy()
    for horizon in [1, 5, 20, 60]:
        df[f"return_{horizon}d"] = df.groupby("ticker")["close"].pct_change(horizon)
    return df


# Family 2: Volatility
# Motivation: EDA showed TSLA/NVDA have ~3x the std-dev of JNJ/PFE.
# Vol-aware features help the model treat high- and low-vol regimes differently.


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling std of daily returns at 20 and 60 day windows."""
    df = df.sort_values(["ticker", "date"]).copy()
    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("ticker")["close"].pct_change()

    for window in [20, 60]:
        df[f"volatility_{window}d"] = df.groupby("ticker")["return_1d"].transform(
            lambda x: x.rolling(window).std()
        )
    return df


# Family 3: Technical indicators
# Motivation: Widely-used signals in retail and institutional trading.
# Including them captures patterns reactive traders create in the data.


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index. Bounded 0-100. >70 overbought, <30 oversold"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _bollinger_pct_b(series: pd.Series, window: int = 20) -> pd.Series:
    """Position within Bollinger Bands. 0 = lower band, 1 = upper band."""
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper, lower = ma + 2 * sd, ma - 2 * sd
    return (series - lower) / (upper - lower)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14) and Bollinger %B (20-day)."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["rsi_14"] = df.groupby("ticker")["close"].transform(_rsi)
    df["bb_pct"] = df.groupby("ticker")["close"].transform(_bollinger_pct_b)
    return df


# Family 4: Volume and market-relative features
# Motivation: EDA showed most variance is systematic (avg corr to SPY ~0.6).
# Stripping out market exposure isolates the stock-specific signal we want.


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume relative to its own 20-day average (volume regime)."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["volume_ratio_20d"] = df.groupby("ticker")["volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )
    return df


def add_market_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return of each stock minus the return of SPY on the same day.
    Isolates idiosyncratic (stock-specific) move from market move.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("ticker")["close"].pct_change()

    spy_returns = (
        df[df["ticker"] == "SPY"].set_index("date")["return_1d"].rename("spy_return_1d")
    )
    df = df.merge(spy_returns, left_on="date", right_index=True, how="left")
    df["excess_return_1d"] = df["return_1d"] - df["spy_return_1d"]
    return df


# Target: 5-day forward return + classification labels


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 5-day forward return.
    Also creates a classification label: 1 if stock is in top quintile
    of next-5d returns *across the universe on that date*, else 0.
    This is the cross-sectional ranking problem from EDA.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["fwd_return_5d"] = df.groupby("ticker")["close"].pct_change(5).shift(-5)
    # Rank within each date: top 20% across universe = positive class
    df["fwd_return_rank"] = df.groupby("date")["fwd_return_5d"].rank(pct=True)
    df["target"] = (df["fwd_return_rank"] >= 0.8).astype(int)
    return df


# Master pipeline


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature family in order, then drop NaN warmup rows."""
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_technical_features(df)
    df = add_volume_features(df)
    df = add_market_relative_features(df)
    df = add_target(df)

    # Drop the first ~60 rows per ticker (warmup for rolling windows)
    # and the last 5 rows per ticker (target needs future data)
    df = df.dropna().reset_index(drop=True)
    return df


# Rank-feature variant (Phase 2 Block 2)
#
# Motivation: the model's target is cross-sectional ("top quintile per date")
# but its features are absolute (RSI-14 = 50, volatility_20d = 0.02). Recasting
# each numeric feature as its **per-date rank-pct across the universe** removes
# the absolute-vs-relative mismatch and lets the tree splits operate directly on
# the comparison that matters. Quant agent's call: "highest leverage-per-hour
# bet on the board." Expected lift: 1-3pp walk-forward AUC.

FEATURE_COLUMNS_TO_RANK = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "rsi_14",
    "bb_pct",
    "volume_ratio_20d",
    "spy_return_1d",       # constant across tickers on a given date — rank will collapse to 1.0
    "excess_return_1d",
]


def build_features_ranked(df: pd.DataFrame) -> pd.DataFrame:
    """
    Like build_features() but every numeric feature is replaced by its per-date
    rank-pct in [0, 1] across the universe. Preserves column names so the
    downstream model code doesn't need to change.
    """
    df = build_features(df)
    for col in FEATURE_COLUMNS_TO_RANK:
        if col in df.columns:
            df[col] = df.groupby("date")[col].rank(pct=True)
    return df


# CLI entry point

if __name__ == "__main__":
    from src.data_loader import load_data

    raw = load_data()
    feats = build_features(raw)

    print("\n=== Feature columns ===")
    print(feats.columns.tolist())
    print(f"\n=== Shape ===")
    print(f"Before features: {len(raw):,} rows")
    print(f"After features:  {len(feats):,} rows (warmup + target dropped)")
    print(f"\n=== Sample ===")
    print(feats.head())
    print(f"\n=== Target balance ===")
    print(feats["target"].value_counts(normalize=True))
    print(f"\n=== Feature stats (selected) ===")
    feature_cols = [
        "return_1d",
        "return_20d",
        "volatility_20d",
        "rsi_14",
        "bb_pct",
        "excess_return_1d",
    ]
    print(feats[feature_cols].describe())
