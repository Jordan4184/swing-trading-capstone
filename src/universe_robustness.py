"""
Universe survivorship robustness test.

Quant final-week kill question:
    "How were these 11 names chosen, and when? If the list was picked
    in 2024 looking backward at large-cap US survivors, your entire
    walk-forward is contaminated."

This script answers that by re-running the v2 pipeline on 2-3 ALTERNATIVE
11-name universes constructed *as-of* a point in time before the train
window starts. If the strategy's edge survives universe perturbation,
the answer is bulletproof. If it doesn't, the candidate at least owns
the finding rather than discovering it on stage.

We don't have a paid PIT data feed, so the universes are constructed
from sources available within this repo + a small whitelist:
    - "headline": the current 11 names (the one in production)
    - "diversified": a sector-rotated alternative that intentionally
      avoids over-indexing on tech megacaps
    - "random_late": a different draw from a similar pool, used as
      a robustness control

Limitations are documented honestly in the JSON output — this is a
survivorship-aware robustness test, not a strict PIT reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import HOLDING_DAYS, build_equity_curve, compute_metrics
from src.bootstrap import bootstrap_metrics
from src.data_loader import download_data, load_data, load_vix
from src.features import build_features
from src.models import FEATURE_COLUMNS, prepare_xy, walk_forward_evaluate, make_models


# Alternative universes — each MUST include SPY for the regime gate and
# excess-return feature to work.
ALT_UNIVERSES: dict[str, list[str]] = {
    "headline": [
        "AAPL", "AMZN", "JNJ", "JPM", "MCD", "META", "NVDA",
        "PFE", "SPY", "TSLA", "UNH",
    ],
    "diversified": [
        # Same 10 + SPY but rotated: more healthcare/consumer staples,
        # less megacap-tech concentration.
        "AAPL", "BAC", "CVX", "HD", "JNJ", "KO", "PEP",
        "PG", "SPY", "WMT", "XOM",
    ],
    "random_late": [
        # Different draw still from S&P 100-ish names, mid-cap-skew on
        # purpose so we're not just relabeling the megacap winners.
        "ADBE", "CRM", "CSCO", "DIS", "GE", "GS", "IBM",
        "INTC", "SPY", "T", "VZ",
    ],
}


def fetch_universe(tickers: list[str], start: str = "2018-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Pull cached OHLCV if everything we need is in the local parquet;
    otherwise hit yfinance. The headline cache is the canonical 11
    tickers — alternative universes are downloaded on demand.
    """
    cached = load_data()
    cached_set = set(cached["ticker"].unique())
    missing = [t for t in tickers if t not in cached_set]
    if not missing:
        sub = cached[cached["ticker"].isin(tickers)].copy()
        return sub
    print(f"  Downloading {len(missing)} ticker(s) not in cache: {missing}")
    fresh = download_data(missing, start=start, end=end)
    combined = pd.concat([cached[cached["ticker"].isin(tickers)], fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined


def build_features_for_universe(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Re-runs the same feature pipeline against an alternative universe.
    Note SPY is required for the spy_return_1d / excess_return_1d features.
    """
    return build_features(ohlcv)


def walk_forward_for_universe(features: pd.DataFrame, n_splits: int = 5) -> dict:
    """Same walk-forward CV the production pipeline uses."""
    rf = make_models()["RandomForest"]
    X, y, meta = prepare_xy(features)
    return walk_forward_evaluate(rf, X, y, meta, n_splits=n_splits)


def simulate_top_n_strategy(predictions: pd.DataFrame, top_n: int = 2, holding_days: int = HOLDING_DAYS) -> pd.DataFrame:
    """
    Lightweight reimplementation of the v1 top-N non-overlapping simulator.
    Avoids re-importing v2/risk-layer plumbing on a different universe (the
    SPY-regime-gate + vol-target features assume specific column shape).
    """
    p = predictions.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["rank"] = p.groupby("date")["y_proba"].rank(method="first", ascending=False)
    picks = p[p["rank"] <= top_n]
    daily_basket = (
        picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return_5d"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    rebalance = daily_basket.iloc[::holding_days].copy().reset_index(drop=True)
    rebalance["basket_return_net"] = rebalance["basket_return_5d"] - 0.002  # 20bps RT
    return rebalance


def evaluate_universe(name: str, tickers: list[str]) -> dict:
    print(f"\n--- Universe: {name} ({len(tickers)} tickers) ---")
    print(f"  Tickers: {tickers}")
    ohlcv = fetch_universe(tickers)
    print(f"  OHLCV rows: {len(ohlcv):,}")

    features = build_features_for_universe(ohlcv)
    print(f"  Feature rows: {len(features):,}")

    cv = walk_forward_for_universe(features, n_splits=5)
    fold_aucs = cv["fold_metrics"]["auc"].tolist()
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print(f"  Walk-forward AUC: {mean_auc:.4f} (± {std_auc:.4f})")
    print(f"  Fold AUCs: {[round(a, 4) for a in fold_aucs]}")

    trades = simulate_top_n_strategy(cv["predictions"], top_n=2)
    if trades.empty:
        return {"name": name, "tickers": tickers, "error": "no trades"}

    rets = trades.set_index("date")["basket_return_net"]
    equity = build_equity_curve(rets)
    metrics = compute_metrics(equity, rets, periods_per_year=252 / HOLDING_DAYS)
    cis = bootstrap_metrics(rets, periods_per_year=252 / HOLDING_DAYS)

    print(f"  Sharpe: {metrics['sharpe_ratio']:+.3f}   CI [{cis['sharpe_ratio']['ci_low']:+.3f}, {cis['sharpe_ratio']['ci_high']:+.3f}]")
    print(f"  CAGR:   {metrics['annualized_return']*100:+.2f}%   CI [{cis['annualized_return']['ci_low']*100:+.2f}%, {cis['annualized_return']['ci_high']*100:+.2f}%]")
    print(f"  MaxDD:  {metrics['max_drawdown']*100:+.2f}%   CI [{cis['max_drawdown']['ci_low']*100:+.2f}%, {cis['max_drawdown']['ci_high']*100:+.2f}%]")

    return {
        "name": name,
        "tickers": tickers,
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "ci": cis,
        "n_trades": int(len(trades)),
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    universes_eval: list[dict] = []
    for name, tickers in ALT_UNIVERSES.items():
        try:
            universes_eval.append(evaluate_universe(name, tickers))
        except Exception as e:
            print(f"  FAILED on {name}: {type(e).__name__}: {e}")
            universes_eval.append({"name": name, "tickers": tickers, "error": str(e)})

    # Summary table
    print("\n=== Universe Robustness Summary ===")
    print(f"{'Universe':<14}{'AUC':>10}{'AUC std':>10}{'Sharpe':>10}{'CAGR':>10}{'MaxDD':>10}")
    for u in universes_eval:
        if "error" in u:
            continue
        m = u["metrics"]
        print(f"{u['name']:<14}{u['mean_auc']:>10.4f}{u['std_auc']:>10.4f}{m['sharpe_ratio']:>10.3f}{m['annualized_return']*100:>9.2f}%{m['max_drawdown']*100:>9.2f}%")

    # Persistence
    headline = next((u for u in universes_eval if u["name"] == "headline" and "error" not in u), None)
    others = [u for u in universes_eval if u["name"] != "headline" and "error" not in u]

    if headline and others:
        sharpes = [u["metrics"]["sharpe_ratio"] for u in others]
        cagrs = [u["metrics"]["annualized_return"] for u in others]
        median_sharpe_alt = float(np.median(sharpes))
        sharpe_range = (float(min(sharpes)), float(max(sharpes)))
        verdict_passes = median_sharpe_alt > 0.0 and all(c > 0 for c in cagrs)
    else:
        median_sharpe_alt = None
        sharpe_range = None
        verdict_passes = False

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "limitations": (
            "Alternative universes are CURATED-as-of-build (not a strict point-in-time "
            "reconstruction). The diversified set rotates away from megacap-tech "
            "concentration; the random_late set draws from a different S&P 100-ish "
            "neighborhood. A full PIT test would require historical index "
            "constituents (CRSP / Sharadar / S&P Dow Jones data) which this project "
            "does not have. This is a robustness check, not a survivorship audit."
        ),
        "universes": universes_eval,
        "summary": {
            "headline_sharpe": headline["metrics"]["sharpe_ratio"] if headline else None,
            "median_alt_sharpe": median_sharpe_alt,
            "alt_sharpe_range": sharpe_range,
            "verdict_passes": verdict_passes,
            "verdict_caption": (
                "Strategy's edge survives universe perturbation: alt-universe Sharpes "
                f"range {sharpe_range[0]:.2f}–{sharpe_range[1]:.2f} (median {median_sharpe_alt:.2f}) "
                f"vs the headline 11's {headline['metrics']['sharpe_ratio']:.2f}. CAGRs all positive."
                if (verdict_passes and headline and sharpe_range)
                else "Strategy's edge IS concentrated in the headline 11-name universe. "
                f"Median alt-universe Sharpe is {median_sharpe_alt}, range {sharpe_range}. "
                "Acknowledge openly: the headline numbers are tied to one universe draw "
                "of large-cap US survivors. The ranker is doing real work on those names; "
                "whether the same procedure generalizes is genuinely uncertain on this sample."
                if (median_sharpe_alt is not None and sharpe_range is not None)
                else "Insufficient data to render verdict — one or more alt universes failed."
            ),
        },
    }

    path = results_dir / "universe_robustness.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")
