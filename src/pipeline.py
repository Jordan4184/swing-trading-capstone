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

    args = parser.parse_args()

    # If no flags given, show help
    if not any(
        [args.all, args.download_data, args.features, args.train, args.backtest]
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

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
