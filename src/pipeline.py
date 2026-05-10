"""
End-to-end pipeline CLI for swing trading capstone.

Usage:
    python -m src.pipeline --all          # run everything
    python -m src.pipeline --download-data  # just download
    python -m src.pipeline --train        # just train models
    python -m src.pipeline --backtest     # just run backtest
"""

import argparse
import sys
from pathlib import Path

# Pipeline steps


def run_download(force: bool = False) -> None:
    """Step 1: Download OHLCV data and cache it."""
    from src.data_loader import load_data

    print("\n" + "=" * 60)
    print("STEP 1: Download data")
    print("=" * 60)
    df = load_data(use_cache=not force)
    print(f"Loaded {len(df):,} rows for {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


def run_features() -> None:
    """Step 2: Build features (runs as part of train and backtest, exposed for testing)."""
    from src.data_loader import load_data
    from src.features import build_features

    print("\n" + "=" * 60)
    print("STEP 2: Build features")
    print("=" * 60)
    df = build_features(load_data())
    print(f"Built features: {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")


def run_train() -> None:
    """Step 3: Train models with walk-forward validation."""
    import json

    from src.data_loader import load_data
    from src.features import build_features
    from src.models import FEATURE_COLUMNS, compare_models, plot_feature_importance

    print("\n" + "=" * 60)
    print("STEP 3: Train and evaluate models")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    df = build_features(load_data())
    print(f"Feature matrix: {len(df):,} rows, {len(FEATURE_COLUMNS)} features")

    results = compare_models(df, n_splits=5)

    summary = {
        name: {
            "mean_auc": float(r["mean_auc"]),
            "mean_accuracy": float(r["mean_accuracy"]),
            "mean_precision": float(r["mean_precision"]),
            "mean_recall": float(r["mean_recall"]),
            "mean_f1": float(r["mean_f1"]),
        }
        for name, r in results.items()
    }
    with open(results_dir / "model_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_model = max(results.keys(), key=lambda k: results[k]["mean_auc"])
    print(f"\nBest model by AUC: {best_model}")
    results[best_model]["predictions"].to_parquet(
        results_dir / "predictions.parquet", index=False
    )

    plot_feature_importance(df, str(results_dir / "05_feature_importance.png"))


def run_backtest() -> None:
    """Step 4: Run backtest on saved predictions."""
    print("\n" + "=" * 60)
    print("STEP 4: Backtest")
    print("=" * 60)

    # Delegate to backtest module's main behavior by re-importing
    # (cleaner than copying the logic)
    import runpy

    runpy.run_module("src.backtest", run_name="__main__")


# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end ML swing trading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline --all
  python -m src.pipeline --download-data
  python -m src.pipeline --train --backtest
        """,
    )
    parser.add_argument(
        "--all", action="store_true", help="Run the full pipeline end-to-end"
    )
    parser.add_argument(
        "--download-data", action="store_true", help="Download OHLCV data"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if cache exists (use with --download-data)",
    )
    parser.add_argument(
        "--features", action="store_true", help="Build features (smoke-test)"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train models with walk-forward validation"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest on saved predictions"
    )
    parser.add_argument(
        "--retrain-final",
        action="store_true",
        help="Refresh data + train a new versioned production model",
    )

    args = parser.parse_args()

    # If no flags given, show help
    if not any(
        [args.all, args.download_data, args.features, args.train, args.backtest, args.retrain_final]
    ):
        parser.print_help()
        return 0

    try:
        if args.all or args.download_data:
            run_download(force=args.force_download)

        if args.all or args.features:
            run_features()

        if args.all or args.train:
            run_train()

        if args.all or args.backtest:
            run_backtest()
        if args.retrain_final:
            run_retrain_final()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1




def run_retrain_final() -> dict:
    """
    Retrain a production model on all available data (no walk-forward).

    Refreshes the OHLCV cache to today, rebuilds features, trains a single
    model, saves as a new versioned file, and compares to the current model.
    Does NOT promote the new model automatically.
    """
    import pandas as pd
    from datetime import datetime
    from src.data_loader import download_data, get_universe, DATA_DIR
    from src.features import build_features
    from src.models import (
        next_model_version,
        train_final_model,
        save_model,
        compare_to_current_production,
        PRODUCTION_MODEL_DIR,
    )

    print("\n" + "=" * 60)
    print("RETRAIN: refresh data + train v_next production model")
    print("=" * 60)

    # 1. Refresh OHLCV cache to today
    today = datetime.now()
    print(f"\nFetching market data through {today.date()}...")
    fresh = download_data(get_universe(), start="2018-01-01", end=today.strftime("%Y-%m-%d"))

    cache_path = DATA_DIR / "ohlcv.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["date"] = pd.to_datetime(cached["date"])
        fresh["date"] = pd.to_datetime(fresh["date"])
        combined = pd.concat([cached, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        combined.to_parquet(cache_path, index=False)
        delta = len(combined) - len(cached)
        print(f"Cache: {len(cached):,} cached + {len(fresh):,} fresh -> {len(combined):,} total ({delta:+,} new rows)")
    else:
        combined = fresh
        fresh.to_parquet(cache_path, index=False)
        print(f"Created new cache: {len(combined):,} rows")

    # 2. Rebuild features over full history
    print("\nBuilding features over full history...")
    df = build_features(combined)
    print(f"Feature matrix: {len(df):,} rows / 11 features")

    # 3. Train new versioned model
    new_version = next_model_version("rf")
    print(f"\nTraining {new_version}...")
    new_model = train_final_model(df)
    new_path = PRODUCTION_MODEL_DIR / f"{new_version}.joblib"
    save_model(new_model, path=new_path)

    # 4. Compare to current production model
    print("\nComparing new model to current production model (5-fold walk-forward)...")
    comparison = compare_to_current_production(new_model, df)
    print(f"\n=== Comparison Result ===")
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    # 5. Output instructions
    print(f"\n=== Next Steps ===")
    print(f"New model saved at: {new_path}")
    print(f"Current production: models/rf_v1.joblib")
    print(f"To PROMOTE {new_version} to production:")
    print(f"  cp {new_path} models/rf_v1.joblib")
    print(f"  (This is a manual step - no auto-promotion.)")

    return {
        "new_version": new_version,
        "new_model_path": str(new_path),
        "n_training_rows": len(df),
        "comparison": comparison,
    }


if __name__ == "__main__":
    sys.exit(main())
