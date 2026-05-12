"""
Stacked-additive ablation of the v2 risk layer.

Runs four backtests on the same picks (from predictions.parquet), each
adding one more risk component to the previous configuration:

    Row 0: v1 baseline (equal-weight top-2, no risk layer)
    Row 1: + vol-targeted sizing
    Row 2: + vol + regime gate
    Row 3: + vol + regime + correlation filter   (= full v2)

For each row: metrics (Sharpe / CAGR / MaxDD / hit_rate / total_return) and
bootstrap 95% CIs. Also computes delta-vs-previous-row and
delta-vs-baseline so the table can show the marginal contribution of each
risk component AND the cumulative effect.

Output: results/ablation_v2.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest import (
    HOLDING_DAYS,
    build_equity_curve,
    compute_metrics,
)
from src.backtest_v2 import simulate_strategy_riskmanaged
from src.bootstrap import bootstrap_metrics
from src.data_loader import load_data, load_vix


CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("v1 baseline (equal-weight)", {"enable_vol_target": False, "enable_regime_gate": False, "enable_corr_filter": False}),
    ("+ vol-targeted sizing",      {"enable_vol_target": True,  "enable_regime_gate": False, "enable_corr_filter": False}),
    ("+ regime gate",              {"enable_vol_target": True,  "enable_regime_gate": True,  "enable_corr_filter": False}),
    ("+ correlation filter (v2)",  {"enable_vol_target": True,  "enable_regime_gate": True,  "enable_corr_filter": True}),
]


def run_ablation(preds: pd.DataFrame, ohlcv: pd.DataFrame, vix: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    periods_per_year = 252 / HOLDING_DAYS

    for label, flags in CONFIGS:
        print(f"\n--- {label} ---")
        trades, _ = simulate_strategy_riskmanaged(preds, ohlcv, vix, **flags)
        rets = trades.set_index("date")["basket_return_net"]
        equity = build_equity_curve(rets)
        metrics = compute_metrics(equity, rets, periods_per_year=periods_per_year)
        cis = bootstrap_metrics(rets, periods_per_year=periods_per_year)

        n_risk_off = 0
        avg_gross = float(trades["gross_weight"].mean()) if "gross_weight" in trades.columns and len(trades) else 0.0
        if "regime" in trades.columns:
            n_risk_off = int((trades["regime"] == "risk_off").sum())

        print(f"  Sharpe {metrics['sharpe_ratio']:+.3f}  CAGR {metrics['annualized_return']*100:+.2f}%  "
              f"MaxDD {metrics['max_drawdown']*100:+.2f}%  n_trades {len(trades)}  "
              f"avg_gross {avg_gross:.3f}  n_risk_off {n_risk_off}")

        rows.append({
            "layer": label,
            "flags": flags,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "ci": cis,
            "n_trades": int(len(trades)),
            "avg_gross_weight": avg_gross,
            "n_risk_off_rebalances": n_risk_off,
        })

    # Marginal (vs previous row) and cumulative (vs baseline) deltas
    baseline = rows[0]["metrics"]
    for i in range(len(rows)):
        cur = rows[i]["metrics"]
        if i == 0:
            rows[i]["delta_vs_prev"] = None
            rows[i]["delta_vs_baseline"] = None
            continue
        prev = rows[i - 1]["metrics"]
        rows[i]["delta_vs_prev"] = {
            "sharpe_ratio": cur["sharpe_ratio"] - prev["sharpe_ratio"],
            "annualized_return": cur["annualized_return"] - prev["annualized_return"],
            "max_drawdown": cur["max_drawdown"] - prev["max_drawdown"],
            "total_return": cur["total_return"] - prev["total_return"],
        }
        rows[i]["delta_vs_baseline"] = {
            "sharpe_ratio": cur["sharpe_ratio"] - baseline["sharpe_ratio"],
            "annualized_return": cur["annualized_return"] - baseline["annualized_return"],
            "max_drawdown": cur["max_drawdown"] - baseline["max_drawdown"],
            "total_return": cur["total_return"] - baseline["total_return"],
        }

    return rows


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading inputs...")
    preds = pd.read_parquet(results_dir / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    ohlcv = load_data()
    vix = load_vix()
    print(f"  Predictions: {len(preds):,} rows")
    print(f"  OHLCV:       {len(ohlcv):,} rows, {ohlcv['ticker'].nunique()} tickers")
    print(f"  VIX:         {len(vix):,} rows")

    rows = run_ablation(preds, ohlcv, vix)

    print("\n=== Ablation Summary ===")
    print(f"{'Layer':<32}{'Sharpe':>10}{'CAGR':>10}{'MaxDD':>10}{'Δ Sharpe':>12}{'Δ MaxDD':>12}")
    for r in rows:
        m = r["metrics"]
        d = r.get("delta_vs_prev")
        ds = f"{d['sharpe_ratio']:+.3f}" if d else "—"
        dm = f"{d['max_drawdown']*100:+.2f}pp" if d else "—"
        print(f"{r['layer']:<32}{m['sharpe_ratio']:+10.3f}{m['annualized_return']*100:+9.2f}% "
              f"{m['max_drawdown']*100:+9.2f}%{ds:>12}{dm:>12}")

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_resamples": rows[0]["ci"]["n_resamples"],
        "rows": rows,
    }
    with open(results_dir / "ablation_v2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {results_dir / 'ablation_v2.json'}")
