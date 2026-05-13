"""
Per-prediction SHAP attribution for the production Random Forest.

Computes signed contributions (positive class = top-quintile) for every
(date, ticker) prediction in results/predictions.parquet and persists them
to results/shap_values.parquet. Columns:

    date, ticker, base_value, <one column per feature with its SHAP value>

The dashboard's ConvictionLedger and Failure-Mode case study both read from
this artifact. For live predictions (today's), the backend computes SHAP
on-the-fly using a TreeExplainer initialized at startup.

Notes
-----
- TreeExplainer is exact for tree ensembles — no sampling, no approximation.
- For RandomForestClassifier the explainer returns shape (n_samples, n_features, n_classes).
  We keep only the positive-class slice [:, :, 1].
- base_value is the model's expected value on the training distribution (≈ 0.5
  for our balanced top-quintile target). SHAP rows sum to predicted log-odds
  minus base on each row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.data_loader import load_data
from src.features import build_features
from src.models import FEATURE_COLUMNS, load_model


def compute_shap_for_predictions(
    preds: pd.DataFrame,
    features: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Join predictions to the feature frame, run TreeExplainer, return a long-
    format DataFrame: (date, ticker, base_value, feature_1_shap, ...).
    """
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])

    # Inner join — only predictions whose feature row survives the warmup
    # + target-NaN dropna in build_features get explained. Equivalent to
    # ~17K rows once historical + recent live predictions are merged.
    merged = preds.merge(
        features[["date", "ticker"] + FEATURE_COLUMNS],
        on=["date", "ticker"],
        how="inner",
        suffixes=("", "_feat"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["date", "ticker", "base_value"] + FEATURE_COLUMNS)

    X = merged[FEATURE_COLUMNS]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)  # (n_samples, n_features, n_classes)

    if isinstance(sv, list):  # older shap returned a list
        positive = sv[1]
        base = float(np.asarray(explainer.expected_value)[1])
    elif sv.ndim == 3:
        positive = sv[:, :, 1]
        base = float(np.asarray(explainer.expected_value)[1])
    else:
        positive = sv
        base = float(explainer.expected_value)

    out = pd.DataFrame(positive, columns=FEATURE_COLUMNS)
    out.insert(0, "ticker", merged["ticker"].values)
    out.insert(0, "date", merged["date"].dt.strftime("%Y-%m-%d").values)
    out["base_value"] = base
    return out


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    print("Loading...")
    preds = pd.read_parquet(results_dir / "predictions.parquet")
    features = build_features(load_data())
    payload = load_model()
    model = payload["model"]
    print(f"  predictions: {len(preds):,} rows")
    print(f"  features:    {len(features):,} rows")
    print(f"  model:       {payload.get('model_version', '?')}")

    print("\nComputing SHAP values (exact, TreeExplainer)...")
    shap_df = compute_shap_for_predictions(preds, features, model)
    print(f"  output: {len(shap_df):,} rows × {shap_df.shape[1]} cols")

    out = results_dir / "shap_values.parquet"
    shap_df.to_parquet(out, index=False)
    print(f"\nSaved: {out}")

    # Sanity print on the most extreme contributors for the latest date
    latest = shap_df.sort_values("date").iloc[-1]
    print("\n=== Sample (latest row) ===")
    print(f"  {latest['ticker']} on {latest['date']}, base={latest['base_value']:.4f}")
    contribs = latest[FEATURE_COLUMNS].astype(float).sort_values(key=abs, ascending=False)
    for feat, val in contribs.head(5).items():
        sign = "+" if val >= 0 else ""
        print(f"    {feat:22s} {sign}{val:.4f}")
