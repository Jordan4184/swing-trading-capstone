"""
Model training and evaluation for swing trading capstone.

Uses walk-forward (time-series) cross-validation to avoid lookahead bias.
Trains and compares three classifiers:
    1. Logistic Regression (linear baseline)
    2. Random Forest (non-linear, handles interactions)
    3. LightGBM (gradient boosted trees, typical SOTA for tabular)
"""

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Feature Columns

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

TARGET_COLUMN = "target"

# Data Prep


def prepare_xy(df: pd.DataFrame):
    """Split a feature df into X, y, and metadata (date, ticker)"""
    df = df.sort_values("date").reset_index(drop=True)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    meta = df[["date", "ticker", "fwd_return_5d"]].copy()
    return X, y, meta


# Model factory


def make_models() -> dict:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "lightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            rum_leaves=31,
            min_child_samples=50,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }


# Walk Forward


def walk_forward_evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """
    Train and evaluate a model using TimeSeriesSplit.
    Returns metrics aggregated across folds, plus per-fold detail.

    Each fold: train on past, predict on next chunk.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        meta_test = meta.iloc[test_idx]

        # Clone the model fresh for each fold (avoid leakage between folds)
        from sklearn.base import clone

        m = clone(model)
        m.fit(X_train, y_train)

        proba = m.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "auc": roc_auc_score(y_test, proba),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
            }
        )

        all_predictions.append(
            pd.DataFrame(
                {
                    "date": meta_test["date"].values,
                    "ticker": meta_test["ticker"].values,
                    "y_true": y_test.values,
                    "y_proba": proba,
                    "fwd_return_5d": meta_test["fwd_return_5d"].values,
                    "fold": fold_idx,
                }
            )
        )

    fold_df = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(all_predictions, ignore_index=True)

    return {
        "fold_metrics": fold_df,
        "predictions": pred_df,
        "mean_auc": fold_df["auc"].mean(),
        "mean_accuracy": fold_df["accuracy"].mean(),
        "mean_precision": fold_df["precision"].mean(),
        "mean_recall": fold_df["recall"].mean(),
        "mean_f1": fold_df["f1"].mean(),
    }


# Comparison runner


def compare_models(df: pd.DataFrame, n_splits: int = 5) -> dict:
    X, y, meta = prepare_xy(df)
    models = make_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        result = walk_forward_evaluate(model, X, y, meta, n_splits=n_splits)
        results[name] = result

        print(f"  Mean AUC:       {result['mean_auc']:.4f}")
        print(f"  Mean Accuracy:  {result['mean_accuracy']:.4f}")
        print(f"  Mean Precision: {result['mean_precision']:.4f}")
        print(f"  Mean Recall:    {result['mean_recall']:.4f}")
        print(f"  Mean F1:        {result['mean_f1']:.4f}")

    return results


# Feature importance


def plot_feature_importance(df: pd.DataFrame, save_path: str):
    X, y, _ = prepare_xy(df)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance["feature"], importance["importance"])
    ax.set_title("Feature Importance (LightGBM)")
    ax.set_xlabel("Importance (split count)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFeature importance plot saved to {save_path}")
    return importance


# CLI entry point

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.features import build_features

    print("Loading data and building features...")
    df = build_features(load_data())
    print(f"Feature matrix: {len(df):,} rows, {len(FEATURE_COLUMNS)} features")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Target balance: {df['target'].mean():.4f} positive class")

    results = compare_models(df, n_splits=5)

    # Save results summary
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        name: {
            "mean_auc": r["mean_auc"],
            "mean_accuracy": r["mean_accuracy"],
            "mean_precision": r["mean_precision"],
            "mean_recall": r["mean_recall"],
            "mean_f1": r["mean_f1"],
        }
        for name, r in results.items()
    }

    with open(results_dir / "model_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults summary saved to {results_dir / 'model_comparison.json'}")

    # Save predictions for the best model (by AUC) — used by backtest module later
    best_model = max(results.keys(), key=lambda k: results[k]["mean_auc"])
    print(f"\nBest model by AUC: {best_model}")
    results[best_model]["predictions"].to_parquet(
        results_dir / "predictions.parquet", index=False
    )
    print(f"Predictions saved to {results_dir / 'predictions.parquet'}")

    # Feature importance
    plot_feature_importance(df, str(results_dir / "05_feature_importance.png"))
