"""
Inference module for the autonomous trading system.

Loads a saved production model and generates predictions for the most
recent available date. Designed to be run daily before market open.

Usage (CLI):
    python -m src.predict

Usage (programmatic):
    from src.predict import generate_latest_predictions
    df = generate_latest_predictions()
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data_loader import download_data, get_universe, DATA_DIR
from src.features import build_features_ranked as build_features
from src.models import FEATURE_COLUMNS, load_model


def fetch_recent_data(lookback_days: int = 120) -> pd.DataFrame:
    """
    Pull the last N days of OHLCV from yfinance for the trading universe.
    We need at least 60 days for the rolling windows + buffer for market closures.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    df = download_data(
        tickers=get_universe(),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    return df


def merge_with_cache(fresh: pd.DataFrame) -> pd.DataFrame:
    """
    Combine fresh download with existing cache, dedup on (ticker, date).
    Keeps the freshest row when there is a conflict.
    """
    cache_path = DATA_DIR / "ohlcv.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        combined = pd.concat([cached, fresh], ignore_index=True)
        # Deduplicate, keeping the freshly downloaded version (last)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        # Update cache for next run
        combined.to_parquet(cache_path, index=False)
        print(f"Merged: {len(cached):,} cached + {len(fresh):,} fresh -> {len(combined):,} total")
        return combined
    else:
        fresh.to_parquet(cache_path, index=False)
        return fresh


def generate_latest_predictions(target_date: str = None) -> pd.DataFrame:
    """
    Generate predictions for the most recent available date.

    Args:
        target_date: Optional. If None, uses the latest date in the data.
                     Format: "YYYY-MM-DD". For the autonomous trader, leave None.

    Returns:
        DataFrame with columns: date, ticker, y_proba, signal_rank
        Sorted by y_proba descending.
    """
    # 1. Get fresh data
    print("Fetching recent market data...")
    fresh = fetch_recent_data(lookback_days=120)
    df = merge_with_cache(fresh)

    # 2. Build features (this requires the full historical context for rolling windows)
    print("Building features...")
    feats = build_features(df)
    print(f"Feature matrix: {len(feats):,} rows")

    # 3. Determine target date
    if target_date is None:
        # Use the most recent date in the features (this excludes the last 5 rows
        # because build_features drops them due to forward-target requirement,
        # but for inference we want to predict on TODAY, which has no target yet.
        # We need to recompute features WITHOUT dropping today's row.)
        # Workaround: rebuild features but skip the target dropna
        target_date = _most_recent_market_date(df)

    target_date = pd.Timestamp(target_date)
    print(f"Target prediction date: {target_date.date()}")

    # 4. Generate features without dropping the target rows (since we want today)
    feats_for_inference = _build_features_for_inference(df, target_date)

    if len(feats_for_inference) == 0:
        raise RuntimeError(f"No feature rows available for {target_date.date()}")

    # 5. Load model and predict
    print("Loading model...")
    payload = load_model()
    model = payload["model"]

    X = feats_for_inference[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "date": feats_for_inference["date"].values,
        "ticker": feats_for_inference["ticker"].values,
        "y_proba": proba,
        "y_true": -1,  # unknown, will be filled in retrospectively
        "fwd_return_5d": float("nan"),
        "fold": -1,  # production model, not from CV
    })
    predictions = predictions.sort_values("y_proba", ascending=False).reset_index(drop=True)
    predictions["signal_rank"] = predictions.index + 1

    print(f"Generated predictions for {len(predictions)} tickers on {target_date.date()}")
    return predictions


def _most_recent_market_date(df: pd.DataFrame) -> pd.Timestamp:
    """Latest date in the OHLCV frame across all tickers."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df["date"].max()


def _build_features_for_inference(raw_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Like build_features_ranked() but does NOT drop the most recent N rows
    that lack forward returns. We need predictions for the latest row, even
    though we can't compute the realized 5-day forward return yet.

    Crucially, the per-date rank-pct is computed across the universe
    BEFORE filtering to target_date — that's what makes it a cross-
    sectional rank rather than a degenerate single-row constant.
    """
    from src.features import (
        FEATURE_COLUMNS_TO_RANK,
        add_market_relative_features,
        add_return_features,
        add_technical_features,
        add_volatility_features,
        add_volume_features,
    )

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_technical_features(df)
    df = add_volume_features(df)
    df = add_market_relative_features(df)

    # Cross-sectional rank-pct per date (matches production training).
    for col in FEATURE_COLUMNS_TO_RANK:
        if col in df.columns:
            df[col] = df.groupby("date")[col].rank(pct=True)

    # Filter to target date
    df = df[df["date"] == target_date].copy()

    # Drop rows where any feature is NaN (e.g. ticker has insufficient history)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def write_predictions_parquet(predictions: pd.DataFrame, append: bool = True) -> Path:
    """
    Write predictions to results/predictions.parquet.

    Args:
        predictions: DataFrame from generate_latest_predictions()
        append: If True, append to existing file (deduping by date+ticker).
                If False, overwrite.
    """
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    pred_path = results_dir / "predictions.parquet"

    if append and pred_path.exists():
        existing = pd.read_parquet(pred_path)
        existing["date"] = pd.to_datetime(existing["date"])
        predictions["date"] = pd.to_datetime(predictions["date"])
        combined = pd.concat([existing, predictions], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        combined.to_parquet(pred_path, index=False)
        print(f"Appended {len(predictions)} new rows. Total: {len(combined)} predictions on file.")
    else:
        predictions.to_parquet(pred_path, index=False)
        print(f"Wrote {len(predictions)} predictions to {pred_path}")

    return pred_path


# CLI entry point
if __name__ == "__main__":
    print("=" * 60)
    print("DAILY PREDICTION GENERATION")
    print("=" * 60)
    preds = generate_latest_predictions()
    print()
    print("Top signals:")
    print(preds.head(10).to_string(index=False))
    print()
    write_predictions_parquet(preds, append=True)
