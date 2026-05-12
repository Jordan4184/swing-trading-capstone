"""
Risk-managed v2 backtest.

Same picks as v1 (top-N by predicted probability per rebalance, non-overlapping
5-day holds), but per-pick weights come from src.risk.size_basket — vol-targeted
sizing, correlation filter, regime gate. Unused capital sits in cash (0% return).

Persists `results/backtest_v2_riskmanaged.json` and overlay plot
`results/06b_equity_curve_v2.png`. Does NOT touch v1 artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import (
    COMMISSION_PER_TRADE,
    HOLDING_DAYS,
    INITIAL_CAPITAL,
    SLIPPAGE_PER_TRADE,
    TOP_N,
    build_equity_curve,
    compute_metrics,
)
from src.data_loader import load_data, load_vix
from src.risk import size_basket


COST_PER_LEG = (COMMISSION_PER_TRADE + SLIPPAGE_PER_TRADE) * 2  # round-trip per leg


def simulate_strategy_riskmanaged(
    preds: pd.DataFrame,
    ohlcv: pd.DataFrame,
    vix: pd.DataFrame,
    top_n: int = TOP_N,
    holding_days: int = HOLDING_DAYS,
    enable_vol_target: bool = True,
    enable_regime_gate: bool = True,
    enable_corr_filter: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Non-overlapping risk-managed backtest.

    The three enable_* flags exist so the ablation harness can disable
    each risk layer independently (see src/ablation.py). All True = full v2.

    Returns:
        trades: DataFrame with date, picks, weights, regime, basket_return_net
        diagnostics: list of per-rebalance diagnostic dicts (for an audit page)
    """
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds_ranked = preds.copy()
    preds_ranked["rank"] = preds_ranked.groupby("date")["y_proba"].rank(
        method="first", ascending=False
    )
    daily_picks = preds_ranked[preds_ranked["rank"] <= top_n].copy()

    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    spy = ohlcv[ohlcv["ticker"] == "SPY"].set_index("date")["close"].sort_index()

    vix = vix.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix_series = vix.set_index("date")["vix_close"].sort_index()

    # Rebalance dates: same non-overlapping cadence as v1
    rebalance_dates = sorted(daily_picks["date"].unique())[::holding_days]

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for asof in rebalance_dates:
        asof_ts = pd.Timestamp(asof)
        day_picks = daily_picks[daily_picks["date"] == asof_ts].sort_values("rank")
        tickers = day_picks["ticker"].tolist()
        if not tickers:
            continue

        weights, diag = size_basket(
            tickers, ohlcv, spy, vix_series, asof_ts,
            enable_vol_target=enable_vol_target,
            enable_regime_gate=enable_regime_gate,
            enable_corr_filter=enable_corr_filter,
        )

        # Per-pick weighted forward return; tickers absent from `weights` are dropped
        fwd_by_ticker = dict(zip(day_picks["ticker"], day_picks["fwd_return_5d"]))
        gross_return = sum(weights.get(t, 0.0) * fwd_by_ticker.get(t, 0.0) for t in tickers)
        gross_weight = sum(weights.values())
        cost = COST_PER_LEG * gross_weight
        net_return = gross_return - cost

        rows.append({
            "date": asof_ts,
            "picks": tickers,
            "weights": weights,
            "regime": diag.regime,
            "gross_weight": gross_weight,
            "basket_return_gross": gross_return,
            "basket_return_net": net_return,
            "cost": cost,
        })
        diagnostics.append({
            "date": asof_ts.strftime("%Y-%m-%d"),
            "regime": diag.regime,
            "spy_close": diag.spy_close,
            "spy_ma200": diag.spy_ma200,
            "vix": diag.vix,
            "vix_pct_rank": diag.vix_pct_rank,
            "regime_multiplier": diag.regime_multiplier,
            "gross_weight": diag.gross_weight,
            "picks": [
                {
                    "ticker": p.ticker,
                    "realized_vol_20d": p.realized_vol_20d,
                    "raw_weight": p.raw_weight,
                    "final_weight": p.final_weight,
                    "dropped_reason": p.dropped_reason,
                }
                for p in diag.picks
            ],
        })

    trades = pd.DataFrame(rows)
    return trades, diagnostics


def plot_overlay(
    v1_eq: pd.Series,
    v2_eq: pd.Series,
    spy_eq: pd.Series,
    eq_weight_eq: pd.Series,
    save_path: Path,
) -> None:
    """Equity-curve overlay: v1 vs v2 vs benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(v1_eq.index, v1_eq.values, label="v1 (no risk layer)",
            linewidth=2, color="#A23B72", linestyle="--")
    ax.plot(v2_eq.index, v2_eq.values, label="v2 (vol-target + regime gate)",
            linewidth=2.5, color="#2E86AB")
    ax.plot(eq_weight_eq.index, eq_weight_eq.values, label="Equal-Weight Universe",
            linewidth=1.5, color="#06A77D", linestyle="-.", alpha=0.8)
    ax.plot(spy_eq.index, spy_eq.values, label="SPY Buy & Hold",
            linewidth=1.5, color="#888", linestyle=":", alpha=0.8)
    ax.set_title("Strategy v1 vs v2 vs Benchmarks — Equity Curve", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


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

    print("\nSimulating v2 (risk-managed)...")
    trades, diagnostics = simulate_strategy_riskmanaged(preds, ohlcv, vix)
    print(f"  {len(trades)} non-overlapping rebalances")

    # Equity curves
    v2_returns = trades.set_index("date")["basket_return_net"]
    v2_equity = build_equity_curve(v2_returns)

    # Benchmarks aligned to v2 window
    ohlcv_dt = ohlcv.copy()
    ohlcv_dt["date"] = pd.to_datetime(ohlcv_dt["date"])
    spy = ohlcv_dt[ohlcv_dt["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    spy_returns = spy.pct_change().loc[v2_returns.index.min(): v2_returns.index.max()]
    spy_equity = build_equity_curve(spy_returns)

    non_spy = ohlcv_dt[ohlcv_dt["ticker"] != "SPY"].copy()
    non_spy = non_spy.sort_values(["ticker", "date"])
    non_spy["ret_1d"] = non_spy.groupby("ticker")["close"].pct_change()
    eq_weight_returns = (
        non_spy.groupby("date")["ret_1d"].mean()
        .loc[v2_returns.index.min(): v2_returns.index.max()]
    )
    eq_weight_equity = build_equity_curve(eq_weight_returns)

    # v1 baseline (re-simulate from predictions so the windows align exactly)
    from src.backtest import simulate_strategy
    v1_trades = simulate_strategy(preds, top_n=TOP_N)
    v1_trades = v1_trades.sort_values("date")
    v1_trades["date"] = pd.to_datetime(v1_trades["date"])
    v1_returns = v1_trades.set_index("date")["basket_return_net"]
    v1_equity = build_equity_curve(v1_returns)

    # Metrics — annualization factor matches v1 (per-trade compounding, ~50/year)
    v2_metrics = compute_metrics(v2_equity, v2_returns, periods_per_year=252 / HOLDING_DAYS)
    v1_metrics = compute_metrics(v1_equity, v1_returns, periods_per_year=252 / HOLDING_DAYS)
    spy_metrics = compute_metrics(spy_equity, spy_returns, periods_per_year=252)
    eq_metrics = compute_metrics(eq_weight_equity, eq_weight_returns, periods_per_year=252)

    # Bootstrap 95% CIs on both v1 and v2 (IID resample, sound because trades
    # are non-overlapping).
    from src.bootstrap import bootstrap_metrics
    v1_cis = bootstrap_metrics(v1_returns, periods_per_year=252 / HOLDING_DAYS)
    v2_cis = bootstrap_metrics(v2_returns, periods_per_year=252 / HOLDING_DAYS)

    def banner(title: str, m: dict) -> None:
        print(f"\n=== {title} ===")
        for k, v in m.items():
            if isinstance(v, float) and abs(v) < 1:
                print(f"  {k:25s} {v:>10.4f}  ({v*100:.2f}%)")
            else:
                print(f"  {k:25s} {v:>10.4f}")

    banner("v1 (no risk layer)", v1_metrics)
    banner("v2 (vol-target + regime gate)", v2_metrics)
    banner("SPY", spy_metrics)
    banner("Equal-Weight Universe", eq_metrics)

    # Regime breakdown
    regime_counts = trades["regime"].value_counts().to_dict()
    print(f"\nRebalances by regime: {regime_counts}")
    print(f"Avg gross_weight in risk_on: {trades.loc[trades['regime']=='risk_on', 'gross_weight'].mean():.3f}")
    if (trades["regime"] == "risk_off").any():
        print(f"Avg gross_weight in risk_off: {trades.loc[trades['regime']=='risk_off', 'gross_weight'].mean():.3f}")

    # Acceptance check. MaxDDs are negative; v2 better => v2 - v1 is positive.
    maxdd_gain = v2_metrics["max_drawdown"] - v1_metrics["max_drawdown"]
    sharpe_delta = v2_metrics["sharpe_ratio"] - v1_metrics["sharpe_ratio"]
    print("\n=== Acceptance ===")
    print(f"  MaxDD improvement: {maxdd_gain*100:+.2f}pp (target ≥ +10pp)")
    print(f"  Sharpe delta: {sharpe_delta:+.3f} (target within ±0.3)")
    accept = bool((maxdd_gain >= 0.10) and (abs(sharpe_delta) <= 0.3))
    print(f"  Result: {'PASS' if accept else 'INVESTIGATE'}")

    print("\n=== Bootstrap 95% CIs (1000 resamples) ===")
    for label, cis in (("v1", v1_cis), ("v2", v2_cis)):
        print(f"  -- {label} --")
        for k in ("sharpe_ratio", "annualized_return", "max_drawdown", "hit_rate", "total_return"):
            c = cis[k]
            print(f"    {k:22s} {c['point']:+.4f}   CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")

    # Persist
    summary = {
        "v2_strategy": {k: float(v) for k, v in v2_metrics.items()},
        "v2_strategy_ci": v2_cis,
        "v1_strategy": {k: float(v) for k, v in v1_metrics.items()},
        "v1_strategy_ci": v1_cis,
        "spy": {k: float(v) for k, v in spy_metrics.items()},
        "equal_weight": {k: float(v) for k, v in eq_metrics.items()},
        "config": {
            "top_n": TOP_N,
            "holding_days": HOLDING_DAYS,
            "cost_per_leg_bps": COST_PER_LEG * 10000,
            "initial_capital": INITIAL_CAPITAL,
            "target_vol_annual": 0.15,
            "max_weight": 0.40,
            "gross_cap": 1.0,
            "regime_off_multiplier": 0.5,
            "corr_threshold": 0.80,
        },
        "regime_counts": regime_counts,
        "acceptance": {
            "maxdd_improvement_pp": round(maxdd_gain * 100, 4),
            "sharpe_delta": round(sharpe_delta, 4),
            "passed": accept,
        },
    }
    with open(results_dir / "backtest_v2_riskmanaged.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {results_dir / 'backtest_v2_riskmanaged.json'}")

    with open(results_dir / "backtest_v2_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Diagnostics saved: {results_dir / 'backtest_v2_diagnostics.json'}")

    trades_persist = trades[["date", "basket_return_net", "gross_weight", "regime"]].copy()
    trades_persist["picks_csv"] = trades["picks"].apply(lambda p: ",".join(p))
    trades_persist.to_parquet(results_dir / "v2_trades.parquet", index=False)
    print(f"Trades saved: {results_dir / 'v2_trades.parquet'}")

    plot_overlay(v1_equity, v2_equity, spy_equity, eq_weight_equity,
                 results_dir / "06b_equity_curve_v2.png")
    print(f"Plot saved: {results_dir / '06b_equity_curve_v2.png'}")
