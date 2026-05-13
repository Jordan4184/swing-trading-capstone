"""
SHAP Interaction Surface with Failure-Mode Pins.

Picks the feature pair with the strongest empirical interaction in the
production Random Forest (largest mean |Φ_ij| off the diagonal), bins
all observed (feature_i, feature_j) values into a 20×20 grid, computes
mean Φ_ij per cell, and pins the 6 failure-case picks at their actual
(value_i, value_j, Φ_ij) coords.

Rendered server-side once and persisted to results/shap_surface.json
so the dashboard reads it cheaply.

The 3rd axis is mathematically load-bearing: Φ_ij is the cross-partial
contribution to the positive class, which two side-by-side PDPs cannot
isolate (they show f(x) + g(y), not the interaction term).

Notes
-----
- Random forests are step functions. Each cell shows the mean Φ_ij of
  the predictions that landed in that bin — we render this as a
  piecewise-constant tiled surface on the frontend, not a smooth mesh.
- For interpretability we keep only Φ for the positive class. SHAP returns
  shape (n_samples, n_features, n_features, n_classes) so we take [:, :, :, 1].
- Φ_ij is symmetric — we only need the off-diagonal upper triangle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.data_loader import load_data
from src.features import build_features
from src.models import FEATURE_COLUMNS, load_model

GRID_BINS = 20  # resolution of the surface
RANDOM_SEED = 42

# Pinned failure cases come from the same hand-picked dates as the
# Failure-Mode case studies (see src/failure_modes.py).
PINNED_DATES = ["2020-02-18", "2022-06-06", "2024-04-12"]


def compute_interaction_values(model, X: pd.DataFrame) -> np.ndarray:
    """
    Run TreeExplainer.shap_interaction_values, return the positive-class
    slice with shape (n_samples, n_features, n_features).
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_interaction_values(X)
    # raw may be a list (per-class), tuple, or ndarray depending on shap version
    if isinstance(raw, list):
        sv = raw[1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 4:
        # (n_samples, n_features, n_features, n_classes)
        sv = raw[:, :, :, 1]
    else:
        sv = raw  # already (n_samples, n_features, n_features)
    return np.asarray(sv)


def pick_dominant_pair(interactions: np.ndarray, feature_names: list[str]) -> tuple[str, str, float]:
    """
    Pick the off-diagonal (i, j) with the largest mean absolute interaction.
    Off-diagonal because Φ_ii is the main effect, not an interaction.
    """
    mean_abs = np.abs(interactions).mean(axis=0)  # (n_features, n_features)
    np.fill_diagonal(mean_abs, 0.0)
    flat_idx = np.argmax(mean_abs)
    i, j = np.unravel_index(flat_idx, mean_abs.shape)
    if i > j:
        i, j = j, i
    return feature_names[i], feature_names[j], float(mean_abs[i, j])


def build_grid(
    X: pd.DataFrame,
    interactions: np.ndarray,
    feat_i: str,
    feat_j: str,
    bins: int = GRID_BINS,
) -> dict:
    """
    Bin all observed (feat_i, feat_j) values into a bins×bins grid.
    For each cell, compute mean Φ_{feat_i, feat_j} of predictions that landed
    in that cell. Cells with no observations come back as None (frontend
    renders them as gaps in the surface, preserving step-function honesty).
    """
    feat_idx = {f: k for k, f in enumerate(X.columns)}
    i_idx, j_idx = feat_idx[feat_i], feat_idx[feat_j]
    phi = interactions[:, i_idx, j_idx]  # one Φ per sample (and Φ is symmetric)

    x_vals = X[feat_i].to_numpy()
    y_vals = X[feat_j].to_numpy()

    # Use percentile-based edges so each bin has roughly equal density.
    # Then add tiny padding on the extremes so all points land in a bin.
    x_edges = np.unique(np.quantile(x_vals, np.linspace(0, 1, bins + 1)))
    y_edges = np.unique(np.quantile(y_vals, np.linspace(0, 1, bins + 1)))
    # In case of ties (e.g., rank features), fall back to uniform spacing
    if len(x_edges) < bins + 1:
        x_edges = np.linspace(x_vals.min(), x_vals.max(), bins + 1)
    if len(y_edges) < bins + 1:
        y_edges = np.linspace(y_vals.min(), y_vals.max(), bins + 1)
    x_edges[0] -= 1e-9
    y_edges[0] -= 1e-9

    x_bins = np.digitize(x_vals, x_edges[1:-1])
    y_bins = np.digitize(y_vals, y_edges[1:-1])

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    z = np.full((ny, nx), np.nan)  # shape (y, x) so Plotly's Surface(z=...) reads correctly
    counts = np.zeros((ny, nx), dtype=int)

    for k in range(len(phi)):
        bx, by = x_bins[k], y_bins[k]
        if 0 <= bx < nx and 0 <= by < ny:
            if np.isnan(z[by, bx]):
                z[by, bx] = phi[k]
                counts[by, bx] = 1
            else:
                counts[by, bx] += 1
                # Running mean
                z[by, bx] += (phi[k] - z[by, bx]) / counts[by, bx]

    # Bin centers for the axes
    x_centers = ((x_edges[:-1] + x_edges[1:]) / 2).tolist()
    y_centers = ((y_edges[:-1] + y_edges[1:]) / 2).tolist()

    # Replace NaN with None for JSON
    z_list: list[list[float | None]] = [
        [None if np.isnan(v) else float(v) for v in row]
        for row in z
    ]

    return {
        "x_centers": x_centers,
        "y_centers": y_centers,
        "z": z_list,
        "counts": counts.tolist(),
        "n_bins_x": nx,
        "n_bins_y": ny,
    }


def build_pins(
    preds: pd.DataFrame,
    features: pd.DataFrame,
    interactions: np.ndarray,
    feat_i: str,
    feat_j: str,
    feature_columns: list[str],
) -> list[dict]:
    """
    For each pinned failure-case date, find the picks (from v2 trades),
    look up their feature values and Φ_ij, return labeled pins.
    """
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    v2_trades = pd.read_parquet(results_dir / "v2_trades.parquet")
    v2_trades["date"] = pd.to_datetime(v2_trades["date"])

    feat_idx_i = feature_columns.index(feat_i)
    feat_idx_j = feature_columns.index(feat_j)

    # Build (date, ticker) -> position-in-X dict so we can pull interactions
    feat_index = {(r["date"], r["ticker"]): k for k, r in features.reset_index(drop=True).iterrows()}

    pins: list[dict] = []
    for date_str in PINNED_DATES:
        d = pd.Timestamp(date_str)
        trade = v2_trades[v2_trades["date"] == d]
        if trade.empty:
            continue
        tickers = [t.strip() for t in trade.iloc[0]["picks_csv"].split(",") if t.strip()]
        basket_return = float(trade.iloc[0]["basket_return_net"])
        for ticker in tickers:
            key = (d, ticker)
            k = feat_index.get(key)
            if k is None:
                continue
            x_val = float(features.iloc[k][feat_i])
            y_val = float(features.iloc[k][feat_j])
            phi = float(interactions[k, feat_idx_i, feat_idx_j])
            pins.append({
                "date": date_str,
                "ticker": ticker,
                "x": x_val,
                "y": y_val,
                "z": phi,
                "basket_return": basket_return,
            })
    return pins


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    print("Loading inputs...")
    preds = pd.read_parquet(results_dir / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    features_full = build_features(load_data())
    features_full["date"] = pd.to_datetime(features_full["date"])
    payload = load_model()
    model = payload["model"]

    # Align features to predictions (same shape we use for SHAP main values)
    merged = preds.merge(
        features_full[["date", "ticker"] + FEATURE_COLUMNS],
        on=["date", "ticker"],
        how="inner",
        suffixes=("", "_feat"),
    )
    print(f"  Aligned rows: {len(merged):,}")
    X = merged[FEATURE_COLUMNS].reset_index(drop=True)
    features_aligned = merged[["date", "ticker"] + FEATURE_COLUMNS].reset_index(drop=True)

    print(f"\nComputing SHAP interaction values (TreeExplainer, exact)...")
    print("  This is O(TLD²) — expect a couple minutes for {} rows × {} features.".format(len(X), len(FEATURE_COLUMNS)))
    interactions = compute_interaction_values(model, X)
    print(f"  shape: {interactions.shape}")

    print("\nPicking dominant interaction pair...")
    feat_i, feat_j, magnitude = pick_dominant_pair(interactions, FEATURE_COLUMNS)
    print(f"  Strongest interaction: {feat_i} × {feat_j} (mean |Φ| = {magnitude:.6f})")
    # Also report top-5 for sanity
    mean_abs = np.abs(interactions).mean(axis=0)
    np.fill_diagonal(mean_abs, 0.0)
    pairs = []
    for a in range(len(FEATURE_COLUMNS)):
        for b in range(a + 1, len(FEATURE_COLUMNS)):
            pairs.append((FEATURE_COLUMNS[a], FEATURE_COLUMNS[b], float(mean_abs[a, b])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    print("\n  Top 5 interaction pairs by mean |Φ|:")
    for fa, fb, mag in pairs[:5]:
        print(f"    {fa:22s} × {fb:22s}  {mag:.6f}")

    print(f"\nBuilding {GRID_BINS}×{GRID_BINS} grid for ({feat_i}, {feat_j})...")
    grid = build_grid(X, interactions, feat_i, feat_j, bins=GRID_BINS)
    populated = sum(1 for row in grid["z"] for v in row if v is not None)
    print(f"  populated cells: {populated} / {grid['n_bins_x'] * grid['n_bins_y']}")

    print(f"\nBuilding failure-case pins...")
    pins = build_pins(preds, features_aligned, interactions, feat_i, feat_j, FEATURE_COLUMNS)
    print(f"  {len(pins)} pins:")
    for p in pins:
        print(f"    {p['date']} {p['ticker']:>5s}: x={p['x']:+.4f}, y={p['y']:+.4f}, Φ={p['z']:+.6f} (basket {p['basket_return']*100:+.2f}%)")

    out = {
        "feature_x": feat_i,
        "feature_y": feat_j,
        "interaction_magnitude": magnitude,
        "top_pairs": [{"feature_x": fa, "feature_y": fb, "magnitude": mag} for fa, fb, mag in pairs[:5]],
        "grid": grid,
        "pins": pins,
        "n_samples": int(len(X)),
        "model_version": payload.get("model_version"),
    }
    path = results_dir / "shap_surface.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")
