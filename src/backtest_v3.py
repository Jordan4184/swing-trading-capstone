"""
v3 = v2 risk layer + ATR trailing stop with gap-aware fills.

Trader agent's #1 unfixed leak: v1/v2 hold positions through the full
5-day clock with no price-based exit. Tail trades ride to Friday's close
through gaps and earnings prints. v3 adds:

    stop = entry_close - ATR_MULTIPLIER * ATR(10)   (initial stop, not trailed
                                                     intra-day for simplicity)

Per-day walk through the 5-bar hold:
    - If day.open <= stop  → exit at open (honor gap risk; we get filled worse
                              than the stop because the market gapped through)
    - Elif day.low  <= stop → exit at stop (intraday touch)
    - Else continue
If never stopped: exit at day-N close (existing v2 behavior).

Per-pick weights come from src.risk.size_basket (same as v2 — vol-target,
regime gate, correlation filter, gross cap). The per-trade return is the
realized return from entry to actual exit (which can be earlier than day N
if stopped). Basket return is the weighted average across surviving picks.

Cost model: same 20bps round-trip per leg as v1/v2 — slippage already
includes some friction, and stopping out doesn't necessarily mean MORE
slippage than a scheduled exit at the close.

Acceptance bar (per trader memo): MaxDD improves by 4-8pp with minor
CAGR cost. If Sharpe degrades or MaxDD doesn't move, that's an honest
finding the candidate should narrate rather than promote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
from src.bootstrap import bootstrap_metrics
from src.data_loader import load_data, load_vix
from src.risk import size_basket


COST_PER_LEG = (COMMISSION_PER_TRADE + SLIPPAGE_PER_TRADE) * 2
ATR_WINDOW = 10
ATR_MULTIPLIER = 3.0


def compute_atr(ohlcv_for_ticker: pd.DataFrame, asof_date: pd.Timestamp, window: int = ATR_WINDOW) -> float | None:
    """
    Standard ATR(window) at asof_date. Requires the last `window+1` daily bars.
    Returns None if there isn't enough history.

    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
    ATR  = mean(TR over last `window` days)
    """
    df = ohlcv_for_ticker[ohlcv_for_ticker["date"] <= asof_date].sort_values("date")
    if len(df) < window + 1:
        return None
    bars = df.tail(window + 1).reset_index(drop=True)
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.iloc[1:].mean())  # drop the first row where prev_close is NaN


def simulate_trade_with_stop(
    ticker_bars: pd.DataFrame,
    entry_date: pd.Timestamp,
    holding_days: int,
    atr_multiplier: float,
    atr_window: int,
) -> tuple[float, str, int, float | None]:
    """
    Walk daily bars after entry_date for up to `holding_days` trading days.
    Returns (realized_return, exit_reason, days_held, atr).

    exit_reason ∈ {"scheduled", "stop_gap", "stop_intraday", "no_bars"}.
    """
    sub = ticker_bars[ticker_bars["date"] >= entry_date].sort_values("date").reset_index(drop=True)
    if len(sub) < 2:
        return 0.0, "no_bars", 0, None

    entry_close = float(sub.iloc[0]["close"])
    atr = compute_atr(ticker_bars, entry_date, window=atr_window)
    if atr is None or atr <= 0:
        # Fall back to scheduled exit — no stop logic if ATR isn't computable.
        exit_idx = min(holding_days, len(sub) - 1)
        exit_price = float(sub.iloc[exit_idx]["close"])
        return (exit_price - entry_close) / entry_close, "scheduled", exit_idx, None

    stop_price = entry_close - atr_multiplier * atr

    for d in range(1, min(holding_days, len(sub) - 1) + 1):
        bar = sub.iloc[d]
        open_p = float(bar["open"])
        low_p = float(bar["low"])
        if open_p <= stop_price:
            # Gap through the stop — honest fill at the open
            return (open_p - entry_close) / entry_close, "stop_gap", d, atr
        if low_p <= stop_price:
            return (stop_price - entry_close) / entry_close, "stop_intraday", d, atr

    # Never stopped — scheduled exit at day-N close (or last available)
    exit_idx = min(holding_days, len(sub) - 1)
    exit_close = float(sub.iloc[exit_idx]["close"])
    return (exit_close - entry_close) / entry_close, "scheduled", exit_idx, atr


def simulate_strategy_v3(
    preds: pd.DataFrame,
    ohlcv: pd.DataFrame,
    vix: pd.DataFrame,
    top_n: int = TOP_N,
    holding_days: int = HOLDING_DAYS,
    atr_multiplier: float = ATR_MULTIPLIER,
    atr_window: int = ATR_WINDOW,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Non-overlapping risk-managed backtest with ATR stops.

    Returns:
        trades: DataFrame with date, picks, weights, regime, basket_return_net
        diagnostics: per-rebalance diagnostic dicts
        per_trade: per-trade records for exit-reason analysis
    """
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    preds_ranked = preds.copy()
    preds_ranked["rank"] = preds_ranked.groupby("date")["y_proba"].rank(method="first", ascending=False)
    daily_picks = preds_ranked[preds_ranked["rank"] <= top_n].copy()

    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    spy = ohlcv[ohlcv["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    bars_by_ticker = {t: ohlcv[ohlcv["ticker"] == t][["date", "open", "high", "low", "close"]] for t in ohlcv["ticker"].unique()}

    vix = vix.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix_series = vix.set_index("date")["vix_close"].sort_index()

    rebalance_dates = sorted(daily_picks["date"].unique())[::holding_days]

    rows: list[dict] = []
    diagnostics: list[dict] = []
    per_trade: list[dict] = []

    for asof in rebalance_dates:
        asof_ts = pd.Timestamp(asof)
        day_picks = daily_picks[daily_picks["date"] == asof_ts].sort_values("rank")
        tickers = day_picks["ticker"].tolist()
        if not tickers:
            continue

        weights, diag = size_basket(tickers, ohlcv, spy, vix_series, asof_ts)

        per_pick_returns: dict[str, float] = {}
        for t in tickers:
            if t not in bars_by_ticker:
                continue
            ret, reason, days_held, atr = simulate_trade_with_stop(
                bars_by_ticker[t], asof_ts, holding_days, atr_multiplier, atr_window
            )
            per_pick_returns[t] = ret
            per_trade.append({
                "rebalance_date": asof_ts.strftime("%Y-%m-%d"),
                "ticker": t,
                "weight": float(weights.get(t, 0.0)),
                "realized_return": float(ret),
                "exit_reason": reason,
                "days_held": int(days_held),
                "atr": float(atr) if atr is not None else None,
            })

        gross_return = sum(weights.get(t, 0.0) * per_pick_returns.get(t, 0.0) for t in tickers)
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
            "gross_weight": diag.gross_weight,
        })

    trades = pd.DataFrame(rows)
    return trades, diagnostics, per_trade


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading inputs...")
    preds = pd.read_parquet(results_dir / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    ohlcv = load_data()
    vix = load_vix()

    print(f"\nSimulating v3 (ATR {ATR_MULTIPLIER}× / window {ATR_WINDOW})...")
    trades, _, per_trade = simulate_strategy_v3(preds, ohlcv, vix)
    print(f"  {len(trades)} rebalances, {len(per_trade)} per-trade records")

    v3_returns = trades.set_index("date")["basket_return_net"]
    v3_equity = build_equity_curve(v3_returns)
    v3_metrics = compute_metrics(v3_equity, v3_returns, periods_per_year=252 / HOLDING_DAYS)

    # Reload v2 summary for the comparison baseline
    with open(results_dir / "backtest_v2_riskmanaged.json") as f:
        v2_summary = json.load(f)
    v2_metrics = v2_summary["v2_strategy"]

    print("\n=== v3 metrics ===")
    for k, v in v3_metrics.items():
        if isinstance(v, float) and abs(v) < 1:
            print(f"  {k:25s} {v:>10.4f}  ({v*100:.2f}%)")
        else:
            print(f"  {k:25s} {v:>10.4f}")

    print("\n=== v2 vs v3 ===")
    sharpe_delta = v3_metrics["sharpe_ratio"] - v2_metrics["sharpe_ratio"]
    maxdd_delta = v3_metrics["max_drawdown"] - v2_metrics["max_drawdown"]
    cagr_delta = v3_metrics["annualized_return"] - v2_metrics["annualized_return"]
    print(f"  Δ Sharpe:  {sharpe_delta:+.3f}  (v2 {v2_metrics['sharpe_ratio']:.3f} → v3 {v3_metrics['sharpe_ratio']:.3f})")
    print(f"  Δ CAGR:    {cagr_delta*100:+.2f}pp  (v2 {v2_metrics['annualized_return']*100:.2f}% → v3 {v3_metrics['annualized_return']*100:.2f}%)")
    print(f"  Δ MaxDD:   {maxdd_delta*100:+.2f}pp  (v2 {v2_metrics['max_drawdown']*100:.2f}% → v3 {v3_metrics['max_drawdown']*100:.2f}%)")

    # Exit-reason breakdown
    pt_df = pd.DataFrame(per_trade)
    reason_counts = pt_df["exit_reason"].value_counts().to_dict()
    print(f"\nExit reasons: {reason_counts}")
    if (pt_df["exit_reason"] != "scheduled").any():
        stopped = pt_df[pt_df["exit_reason"].isin(["stop_gap", "stop_intraday"])]
        print(f"  Stopped trades: {len(stopped)}/{len(pt_df)} ({len(stopped)/len(pt_df)*100:.1f}%)")
        print(f"  Avg stopped return: {stopped['realized_return'].mean()*100:+.2f}%")
        print(f"  Avg scheduled return: {pt_df[pt_df['exit_reason']=='scheduled']['realized_return'].mean()*100:+.2f}%")

    # Acceptance bar from trader memo: -4 to -8pp DD improvement, minor CAGR cost
    accept = (maxdd_delta >= 0.04) and (cagr_delta >= -0.05)
    print(f"\nAcceptance: {'PASS' if accept else 'INVESTIGATE'} "
          f"(target MaxDD improvement ≥ +4pp, CAGR delta ≥ -5pp)")

    # Bootstrap CIs
    print("\nComputing bootstrap CIs...")
    v3_cis = bootstrap_metrics(v3_returns, periods_per_year=252 / HOLDING_DAYS)

    summary = {
        "v3_strategy": {k: float(v) for k, v in v3_metrics.items()},
        "v3_strategy_ci": v3_cis,
        "v2_strategy": {k: float(v) for k, v in v2_metrics.items()},
        "config": {
            "top_n": TOP_N,
            "holding_days": HOLDING_DAYS,
            "atr_multiplier": ATR_MULTIPLIER,
            "atr_window": ATR_WINDOW,
            "cost_per_leg_bps": COST_PER_LEG * 10000,
            "initial_capital": INITIAL_CAPITAL,
        },
        "exit_reasons": reason_counts,
        "n_per_trade": len(per_trade),
        "acceptance": {
            "maxdd_delta_pp": round(maxdd_delta * 100, 4),
            "cagr_delta_pp": round(cagr_delta * 100, 4),
            "sharpe_delta": round(sharpe_delta, 4),
            "passed": bool(accept),
        },
    }
    with open(results_dir / "backtest_v3_atrstop.json", "w") as f:
        json.dump(summary, f, indent=2)

    trades_persist = trades[["date", "basket_return_net", "gross_weight", "regime"]].copy()
    trades_persist["picks_csv"] = trades["picks"].apply(lambda p: ",".join(p))
    trades_persist.to_parquet(results_dir / "v3_trades.parquet", index=False)

    pt_df.to_parquet(results_dir / "v3_per_trade.parquet", index=False)

    print(f"\nSaved: {results_dir / 'backtest_v3_atrstop.json'}")
    print(f"Saved: {results_dir / 'v3_trades.parquet'}")
    print(f"Saved: {results_dir / 'v3_per_trade.parquet'}")
