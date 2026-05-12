"""
Feature ablation: absolute features vs per-date cross-sectional ranks.

Quant agent's read: "your target is relative (top quintile per date) but
your features are absolute — the model is wasting capacity learning the
cross-sectional transformation. Recasting features as per-date rank-pct
should lift walk-forward AUC by 1-3 points with zero new data dependencies."

This script trains the production model (RandomForest) twice — once on
absolute features (current v1) and once on rank-pct features — over the
same walk-forward TimeSeriesSplit, and reports per-fold AUC + the delta.

Decision logic at the bottom: promote / shelve / investigate, mirroring
the v2 backtest acceptance bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.data_loader import load_data
from src.features import build_features, build_features_ranked
from src.models import FEATURE_COLUMNS, make_models, prepare_xy


def walk_forward_auc(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[float]:
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = m.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y.iloc[test_idx], proba)
        aucs.append(float(auc))
        print(f"  fold {fold_idx}: AUC = {auc:.4f}")
    return aucs


def run_feature_ablation(n_splits: int = 5) -> dict:
    raw = load_data()
    rf = make_models()["RandomForest"]

    print("\n=== Absolute features (current v1) ===")
    abs_feats = build_features(raw)
    X_abs, y_abs, _ = prepare_xy(abs_feats)
    abs_aucs = walk_forward_auc(rf, X_abs, y_abs, n_splits=n_splits)

    print("\n=== Per-date rank features ===")
    rank_feats = build_features_ranked(raw)
    X_rk, y_rk, _ = prepare_xy(rank_feats)
    rk_aucs = walk_forward_auc(rf, X_rk, y_rk, n_splits=n_splits)

    abs_mean, abs_std = float(np.mean(abs_aucs)), float(np.std(abs_aucs))
    rk_mean, rk_std = float(np.mean(rk_aucs)), float(np.std(rk_aucs))
    delta = rk_mean - abs_mean

    if delta > 0.005:
        recommendation = "PROMOTE: rank features lift AUC meaningfully"
    elif abs(delta) <= 0.005:
        recommendation = "INDIFFERENT: no meaningful difference — keep current model"
    else:
        recommendation = "INVESTIGATE: rank features hurt AUC; do not promote"

    print("\n=== Comparison ===")
    print(f"  Absolute features mean AUC: {abs_mean:.4f} (± {abs_std:.4f})")
    print(f"  Rank features mean AUC:     {rk_mean:.4f} (± {rk_std:.4f})")
    print(f"  Δ AUC:                      {delta:+.4f}")
    print(f"  Recommendation:             {recommendation}")

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_splits": n_splits,
        "feature_columns": FEATURE_COLUMNS,
        "absolute": {
            "fold_aucs": abs_aucs,
            "mean_auc": abs_mean,
            "std_auc": abs_std,
        },
        "ranked": {
            "fold_aucs": rk_aucs,
            "mean_auc": rk_mean,
            "std_auc": rk_std,
        },
        "delta_auc": delta,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    out = run_feature_ablation(n_splits=5)
    path = results_dir / "feature_ablation.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")
