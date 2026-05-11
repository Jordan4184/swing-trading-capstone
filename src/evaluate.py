"""
Model + autonomous trader evaluation framework.

Three categories of evaluation:
    A. Trade-level: real P&L of placed trades vs expectations (needs live trades)
    B. Model-quality: does y_proba predict actual forward returns? (works with just data)
    C. Drift detection: feature/probability distributions over time (works with just data)

Usage:
    python -m src.evaluate
    python -m src.evaluate --json > report.json
"""

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRADER_DB = PROJECT_ROOT / "dashboard" / "backend" / "auto_trader.db"
DATA_CACHE = PROJECT_ROOT / "Data" / "raw" / "ohlcv.parquet"
PREDICTIONS = RESULTS_DIR / "predictions.parquet"


def _load_predictions():
    if not PREDICTIONS.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PREDICTIONS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_ohlcv():
    if not DATA_CACHE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_closed_trades():
    if not TRADER_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(TRADER_DB))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE exit_filled_at IS NOT NULL ORDER BY exit_filled_at DESC",
            conn,
        )
        return df
    finally:
        conn.close()


def _compute_forward_returns(ohlcv, horizon=5):
    df = ohlcv.sort_values(["ticker", "date"]).copy()
    df["fwd_return_Nd"] = df.groupby("ticker")["close"].pct_change(horizon).shift(-horizon)
    return df[["ticker", "date", "close", "fwd_return_Nd"]]


def evaluate_trade_level():
    trades = _load_closed_trades()
    if trades.empty:
        return {
            "status": "no_trades",
            "n_closed_trades": 0,
            "note": "Autonomous trader has not closed any trades yet.",
        }

    n = len(trades)
    n_wins = int((trades["pnl"] > 0).sum())
    n_losses = int((trades["pnl"] <= 0).sum())
    hit_rate = n_wins / n if n > 0 else 0.0
    actual_returns = trades["pnl_pct"].astype(float)

    ohlcv = _load_ohlcv()
    universe_avg_returns = []
    if not ohlcv.empty:
        ohlcv_with_fwd = _compute_forward_returns(ohlcv, horizon=5)
        for _, t in trades.iterrows():
            sig_date = pd.Timestamp(t["signal_date"])
            sub = ohlcv_with_fwd[ohlcv_with_fwd["date"] == sig_date]["fwd_return_Nd"].dropna()
            if len(sub) > 0:
                universe_avg_returns.append(float(sub.mean()))

    benchmark_avg = float(np.nanmean(universe_avg_returns)) if universe_avg_returns else None
    strategy_avg = float(actual_returns.mean()) if len(actual_returns) > 0 else 0.0
    alpha_per_trade = (strategy_avg - benchmark_avg) if benchmark_avg is not None else None

    by_ticker = trades.groupby("ticker").agg(
        n_trades=("id", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl_pct=("pnl_pct", "mean"),
        hit_rate=("pnl", lambda x: float((x > 0).mean())),
    ).round(4).reset_index().to_dict(orient="records")

    return {
        "status": "ok",
        "n_closed_trades": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "hit_rate": round(hit_rate, 4),
        "total_pnl": round(float(trades["pnl"].sum()), 2),
        "avg_pnl": round(float(trades["pnl"].mean()), 2),
        "avg_return_pct": round(strategy_avg, 6),
        "benchmark_universe_avg_return": round(benchmark_avg, 6) if benchmark_avg is not None else None,
        "alpha_per_trade_pct": round(alpha_per_trade, 6) if alpha_per_trade is not None else None,
        "best_trade": round(float(trades["pnl"].max()), 2),
        "worst_trade": round(float(trades["pnl"].min()), 2),
        "by_ticker": by_ticker,
    }


def evaluate_model_quality():
    preds = _load_predictions()
    if preds.empty:
        return {"status": "no_predictions"}

    ohlcv = _load_ohlcv()
    if ohlcv.empty:
        return {"status": "no_ohlcv"}

    ohlcv_fwd = _compute_forward_returns(ohlcv, horizon=5)
    merged = preds.merge(ohlcv_fwd, on=["date", "ticker"], how="left", suffixes=("", "_realized"))
    merged["fwd_return_5d_actual"] = merged["fwd_return_Nd"]
    merged = merged.dropna(subset=["fwd_return_5d_actual", "y_proba"])

    if len(merged) < 20:
        return {"status": "insufficient_data", "n_predictions": len(merged)}

    merged["rank"] = merged.groupby("date")["y_proba"].rank(ascending=False, method="first")
    top2 = merged[merged["rank"] <= 2]
    top2_avg = float(top2["fwd_return_5d_actual"].mean())
    universe_avg = float(merged["fwd_return_5d_actual"].mean())
    top2_alpha = top2_avg - universe_avg

    daily_corr = []
    for date, group in merged.groupby("date"):
        if len(group) >= 3:
            corr = group["y_proba"].corr(group["fwd_return_5d_actual"], method="spearman")
            if not (corr is None or (isinstance(corr, float) and math.isnan(corr))):
                daily_corr.append(float(corr))
    avg_spearman = float(np.mean(daily_corr)) if daily_corr else None

    merged["proba_bucket"] = pd.cut(
        merged["y_proba"],
        bins=[0, 0.45, 0.50, 0.55, 0.60, 1.0],
        labels=["<0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", ">0.60"],
    )
    merged["actual_top_quintile"] = (
        merged.groupby("date")["fwd_return_5d_actual"].rank(pct=True) >= 0.8
    ).astype(int)

    cal = merged.groupby("proba_bucket", observed=True)["actual_top_quintile"].agg(["count", "mean"]).round(4).reset_index()
    cal = cal.rename(columns={"mean": "actual_top_quintile_rate"})
    cal["proba_bucket"] = cal["proba_bucket"].astype(str)
    calibration_records = cal.to_dict(orient="records")
    top2_hit_rate = float((top2["fwd_return_5d_actual"] > 0).mean())

    if avg_spearman is None:
        interp = "unknown"
    elif avg_spearman > 0.2:
        interp = "strong positive"
    elif avg_spearman > 0.05:
        interp = "weak positive"
    elif abs(avg_spearman) <= 0.05:
        interp = "no signal"
    elif avg_spearman > -0.2:
        interp = "weak negative"
    else:
        interp = "strong negative (concerning)"

    return {
        "status": "ok",
        "n_predictions_evaluated": len(merged),
        "n_dates_evaluated": int(merged["date"].nunique()),
        "top2_avg_return": round(top2_avg, 6),
        "universe_avg_return": round(universe_avg, 6),
        "top2_alpha_per_holding_period": round(top2_alpha, 6),
        "top2_hit_rate": round(top2_hit_rate, 4),
        "avg_spearman_correlation": round(avg_spearman, 4) if avg_spearman is not None else None,
        "spearman_interpretation": interp,
        "calibration_buckets": calibration_records,
    }


def evaluate_drift():
    preds = _load_predictions()
    if preds.empty:
        return {"status": "no_predictions"}

    preds = preds.sort_values("date").copy()
    preds["date"] = pd.to_datetime(preds["date"])
    latest_date = preds["date"].max()
    cutoff = latest_date - pd.Timedelta(days=30)

    recent = preds[preds["date"] >= cutoff]
    historical = preds[preds["date"] < cutoff]

    if len(recent) < 5 or len(historical) < 20:
        return {"status": "insufficient_data", "n_recent": len(recent), "n_historical": len(historical)}

    recent_mean = float(recent["y_proba"].mean())
    historical_mean = float(historical["y_proba"].mean())
    recent_std = float(recent["y_proba"].std())
    historical_std = float(historical["y_proba"].std())

    z_score_diff = (recent_mean - historical_mean) / historical_std if historical_std > 0 else 0.0

    if abs(z_score_diff) > 1.5:
        severity = "high"
    elif abs(z_score_diff) > 0.75:
        severity = "moderate"
    else:
        severity = "low"

    recent_above_055 = float((recent["y_proba"] >= 0.55).mean())
    historical_above_055 = float((historical["y_proba"] >= 0.55).mean())

    by_ticker_drift = []
    for ticker, group in preds.groupby("ticker"):
        t_recent = group[group["date"] >= cutoff]["y_proba"]
        t_hist = group[group["date"] < cutoff]["y_proba"]
        if len(t_recent) >= 2 and len(t_hist) >= 5:
            by_ticker_drift.append({
                "ticker": ticker,
                "recent_mean": round(float(t_recent.mean()), 4),
                "historical_mean": round(float(t_hist.mean()), 4),
                "delta": round(float(t_recent.mean() - t_hist.mean()), 4),
            })
    by_ticker_drift = sorted(by_ticker_drift, key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "status": "ok",
        "recent_window": f"last 30 days (from {cutoff.date()})",
        "n_recent": len(recent),
        "n_historical": len(historical),
        "recent_mean_proba": round(recent_mean, 4),
        "historical_mean_proba": round(historical_mean, 4),
        "z_score_difference": round(z_score_diff, 4),
        "drift_severity": severity,
        "recent_std": round(recent_std, 4),
        "historical_std": round(historical_std, 4),
        "recent_buy_signal_rate": round(recent_above_055, 4),
        "historical_buy_signal_rate": round(historical_above_055, 4),
        "buy_signal_rate_delta": round(recent_above_055 - historical_above_055, 4),
        "by_ticker_top5_drift": by_ticker_drift[:5],
    }


def run_full_report():
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "category_a_trade_level": evaluate_trade_level(),
        "category_b_model_quality": evaluate_model_quality(),
        "category_c_drift": evaluate_drift(),
    }


def _pretty_print(report):
    print()
    print("=" * 70)
    print("MODEL + AUTO-TRADER EVALUATION REPORT")
    print("=" * 70)
    print(f"Generated: {report['generated_at']}")
    print()

    a = report["category_a_trade_level"]
    print("--- A: Trade-Level (live autonomous trades) ---")
    if a["status"] == "no_trades":
        print(f"  {a['note']}")
    else:
        print(f"  Closed trades: {a['n_closed_trades']} ({a['n_wins']}W / {a['n_losses']}L)")
        print(f"  Hit rate: {a['hit_rate']:.1%}")
        print(f"  Total P&L: ${a['total_pnl']:.2f}")
        print(f"  Strategy avg return: {a['avg_return_pct']:.3%}")
        if a.get("benchmark_universe_avg_return") is not None:
            print(f"  Benchmark (universe avg): {a['benchmark_universe_avg_return']:.3%}")
            print(f"  Alpha per trade: {a['alpha_per_trade_pct']:+.3%}")
    print()

    b = report["category_b_model_quality"]
    print("--- B: Model Quality (predictions vs realized returns) ---")
    if b["status"] != "ok":
        print(f"  {b['status']}")
    else:
        print(f"  Predictions evaluated: {b['n_predictions_evaluated']} across {b['n_dates_evaluated']} dates")
        print(f"  Top-2 avg return: {b['top2_avg_return']:.3%}")
        print(f"  Universe avg return: {b['universe_avg_return']:.3%}")
        print(f"  Top-2 alpha (per 5-day hold): {b['top2_alpha_per_holding_period']:+.3%}")
        print(f"  Top-2 hit rate: {b['top2_hit_rate']:.1%}")
        if b.get("avg_spearman_correlation") is not None:
            print(f"  Rank correlation (Spearman): {b['avg_spearman_correlation']:+.4f} ({b['spearman_interpretation']})")
        print(f"  Calibration:")
        for bucket in b["calibration_buckets"]:
            label = bucket["proba_bucket"]
            count = bucket["count"]
            rate = bucket["actual_top_quintile_rate"]
            print(f"    {label:>12s}: n={count:>5} -> {rate:.1%} in top quintile")
    print()

    c = report["category_c_drift"]
    print("--- C: Drift Detection (recent vs historical predictions) ---")
    if c["status"] != "ok":
        print(f"  {c['status']}")
    else:
        print(f"  Window: {c['recent_window']}")
        print(f"  Recent mean proba: {c['recent_mean_proba']} (vs historical {c['historical_mean_proba']})")
        print(f"  Z-score difference: {c['z_score_difference']:+.4f} -> drift severity: {c['drift_severity']}")
        print(f"  Buy signal rate: recent {c['recent_buy_signal_rate']:.1%} vs historical {c['historical_buy_signal_rate']:.1%}")
        if c["by_ticker_top5_drift"]:
            print(f"  Top 5 tickers by drift magnitude:")
            for row in c["by_ticker_top5_drift"]:
                print(f"    {row['ticker']:>4s}: recent {row['recent_mean']:.3f} vs historical {row['historical_mean']:.3f} (delta {row['delta']:+.3f})")
    print()


if __name__ == "__main__":
    import sys
    report = run_full_report()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2, default=str))
    else:
        _pretty_print(report)
