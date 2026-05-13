"""
Strategy vs. vol-targeted SPY — the null test.

Panel chair's kill question:
    "Your v2 Sharpe is 1.09. At what point on its bootstrap CI does the
    strategy stop being distinguishable from a vol-targeted SPY position
    over the same window?"

This script answers that exactly. It builds a "null" benchmark that:
    - Holds ONLY SPY (no model, no ranker)
    - Uses the SAME vol-targeting rule as v2 (15% annual target, 0.40 cap)
    - Uses the SAME SPY/VIX regime gate as v2 (half-size when SPY < 200dMA
      AND VIX > 75th pct rolling 2y)
    - Rebalances on the SAME dates as v2 (non-overlapping 5-day cadence)
    - Applies the SAME 20bps round-trip cost

Then it runs a paired bootstrap on the per-rebalance Sharpe ratio diff
to produce a p-value for "strategy excess return ≠ vol-targeted SPY."

Outputs results/null_test.json with overlaid equity series, CIs,
distributional comparisons, and the p-value.

What this earns at defense
--------------------------
Converts the "your alpha is beta in disguise" trap into the strongest
slide: "I held the risk-managed v2 against a vol-targeted SPY null over
the same dates. The Sharpe difference is X with 95% CI [a,b], paired
bootstrap p = z. Vol-targeting is NOT the alpha — the residual edge
attributable to the ML ranker is the band above the null curve."
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
    build_equity_curve,
    compute_metrics,
)
from src.bootstrap import bootstrap_metrics
from src.data_loader import load_data, load_vix
from src.risk import (
    REGIME_OFF_MULTIPLIER,
    realized_vol,
    regime_label,
    vix_percentile_rank,
    vol_target_weight,
)


COST_PER_LEG = (COMMISSION_PER_TRADE + SLIPPAGE_PER_TRADE) * 2  # 20bps round-trip


def build_vol_targeted_spy(
    rebalance_dates: list[pd.Timestamp],
    spy: pd.Series,
    vix_series: pd.Series,
    target_vol: float = 0.15,
    max_weight: float = 0.40,
    holding_days: int = HOLDING_DAYS,
) -> pd.DataFrame:
    """
    For each rebalance date, compute SPY's vol-targeted weight (with the same
    regime gate as v2), look up SPY's realized return over the next holding_days,
    and produce the per-rebalance net return after costs.

    Returns a DataFrame with columns:
        date, spy_close, spy_ma200, vix, vix_pct_rank, regime, regime_mult,
        realized_vol_20d, weight, fwd_return_5d, cost, net_return
    """
    rows: list[dict] = []
    for asof in rebalance_dates:
        asof_ts = pd.Timestamp(asof)

        # Per-pick vol target on SPY (treating it as if it's the single pick)
        prior = spy.loc[:asof_ts]
        if len(prior) < 21:
            continue
        rv = realized_vol(prior.pct_change(), window=20)
        if np.isnan(rv) or rv <= 0:
            continue
        w_raw = vol_target_weight(rv, target=target_vol, max_w=max_weight)

        # Regime gate (same primitives as src.risk.size_basket)
        spy_close = float(prior.iloc[-1])
        spy_ma200 = float(prior.tail(200).mean()) if len(prior) >= 200 else float("nan")
        vix_upto = vix_series.loc[:asof_ts].dropna()
        if vix_upto.empty:
            continue
        vix_val = float(vix_upto.iloc[-1])
        vix_pct = vix_percentile_rank(vix_series, asof_ts)
        regime = regime_label(spy_close, spy_ma200, vix_val, vix_pct)
        regime_mult = REGIME_OFF_MULTIPLIER if regime == "risk_off" else 1.0

        # Final weight on SPY
        weight = w_raw * regime_mult
        weight = min(weight, 1.0)  # gross cap

        # Forward 5-day SPY return
        fwd_window = spy.loc[asof_ts:].iloc[: holding_days + 1]
        if len(fwd_window) < 2:
            continue
        fwd_return = float(fwd_window.iloc[-1] / fwd_window.iloc[0] - 1)

        gross = weight * fwd_return
        cost = COST_PER_LEG * weight
        net = gross - cost

        rows.append({
            "date": asof_ts,
            "spy_close": spy_close,
            "spy_ma200": spy_ma200,
            "vix": vix_val,
            "vix_pct_rank": vix_pct,
            "regime": regime,
            "regime_mult": regime_mult,
            "realized_vol_20d": float(rv),
            "weight": float(weight),
            "fwd_return_5d": fwd_return,
            "cost": float(cost),
            "net_return": float(net),
        })
    return pd.DataFrame(rows)


def paired_bootstrap_sharpe_diff(
    strategy_returns: pd.Series,
    null_returns: pd.Series,
    periods_per_year: float,
    n_resamples: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Resample paired rebalance returns (strategy_i, null_i) with replacement,
    recompute Sharpe of each, return distribution of the DIFFERENCE.
    Output the empirical p-value for "strategy Sharpe > null Sharpe".
    """
    rng = np.random.default_rng(seed)
    s = strategy_returns.dropna().to_numpy()
    n = null_returns.dropna().to_numpy()
    # Pair by index (we will align before calling this)
    assert len(s) == len(n), f"length mismatch: {len(s)} vs {len(n)}"
    n_obs = len(s)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_obs, size=n_obs)
        s_re = s[idx]
        n_re = n[idx]
        s_std = s_re.std(ddof=1)
        n_std = n_re.std(ddof=1)
        s_sh = (s_re.mean() / s_std) * np.sqrt(periods_per_year) if s_std > 0 else 0.0
        n_sh = (n_re.mean() / n_std) * np.sqrt(periods_per_year) if n_std > 0 else 0.0
        diffs[i] = s_sh - n_sh

    # Point estimate (un-resampled)
    s_std0 = s.std(ddof=1)
    n_std0 = n.std(ddof=1)
    s_sh0 = (s.mean() / s_std0) * np.sqrt(periods_per_year) if s_std0 > 0 else 0.0
    n_sh0 = (n.mean() / n_std0) * np.sqrt(periods_per_year) if n_std0 > 0 else 0.0
    point_diff = float(s_sh0 - n_sh0)

    # Two-sided p-value: fraction of bootstrap samples where diff has opposite sign
    if point_diff >= 0:
        p = float((diffs <= 0).mean())
    else:
        p = float((diffs >= 0).mean())
    # Two-sided
    p_two = min(2.0 * p, 1.0)

    return {
        "n_resamples": n_resamples,
        "point_diff": point_diff,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "strategy_sharpe": float(s_sh0),
        "null_sharpe": float(n_sh0),
        "p_value_one_sided": p,
        "p_value_two_sided": p_two,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    print("Loading inputs...")
    v2_trades = pd.read_parquet(results_dir / "v2_trades.parquet")
    v2_trades["date"] = pd.to_datetime(v2_trades["date"])
    ohlcv = load_data()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    spy = ohlcv[ohlcv["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    vix = load_vix()
    vix["date"] = pd.to_datetime(vix["date"])
    vix_series = vix.set_index("date")["vix_close"].sort_index()
    print(f"  v2 rebalances: {len(v2_trades)}")
    print(f"  SPY: {len(spy)}, VIX: {len(vix)}")

    print("\nBuilding vol-targeted SPY null benchmark on same rebalance dates...")
    rebalance_dates = v2_trades["date"].tolist()
    null_df = build_vol_targeted_spy(rebalance_dates, spy, vix_series)
    print(f"  null rebalances: {len(null_df)}")

    # Align strategy and null on shared dates
    v2_indexed = v2_trades.set_index("date")["basket_return_net"]
    null_indexed = null_df.set_index("date")["net_return"]
    shared = v2_indexed.index.intersection(null_indexed.index)
    strategy_aligned = v2_indexed.loc[shared].sort_index()
    null_aligned = null_indexed.loc[shared].sort_index()
    print(f"  aligned: {len(shared)} paired rebalances")

    # Build equity curves on identical date axis
    strategy_eq = build_equity_curve(strategy_aligned)
    null_eq = build_equity_curve(null_aligned)

    pp_year = 252 / HOLDING_DAYS
    strategy_metrics = compute_metrics(strategy_eq, strategy_aligned, periods_per_year=pp_year)
    null_metrics = compute_metrics(null_eq, null_aligned, periods_per_year=pp_year)
    strategy_cis = bootstrap_metrics(strategy_aligned, periods_per_year=pp_year)
    null_cis = bootstrap_metrics(null_aligned, periods_per_year=pp_year)

    print("\n=== v2 (strategy) ===")
    for k in ("sharpe_ratio", "annualized_return", "max_drawdown", "hit_rate", "total_return"):
        c = strategy_cis[k]
        print(f"  {k:22s} {c['point']:+.4f}   CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")
    print("\n=== vol-targeted SPY (null) ===")
    for k in ("sharpe_ratio", "annualized_return", "max_drawdown", "hit_rate", "total_return"):
        c = null_cis[k]
        print(f"  {k:22s} {c['point']:+.4f}   CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")

    print("\n=== Paired bootstrap on Sharpe difference (5,000 resamples) ===")
    diff = paired_bootstrap_sharpe_diff(strategy_aligned, null_aligned, periods_per_year=pp_year)
    print(f"  strategy Sharpe: {diff['strategy_sharpe']:+.4f}")
    print(f"  null Sharpe:     {diff['null_sharpe']:+.4f}")
    print(f"  Δ Sharpe:        {diff['point_diff']:+.4f}   95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]")
    print(f"  p_one_sided:     {diff['p_value_one_sided']:.4f}")
    print(f"  p_two_sided:     {diff['p_value_two_sided']:.4f}")

    verdict_passed = diff["point_diff"] > 0 and diff["p_value_two_sided"] < 0.10
    print(f"\nVerdict (Δ Sharpe > 0 AND p < 0.10): {'PASS — alpha is not just vol-targeting' if verdict_passed else 'INVESTIGATE — within bootstrap noise of vol-targeted SPY'}")

    # Persist equity series for the dashboard chart
    strategy_series = [
        {"date": d.strftime("%Y-%m-%d"), "equity": round(float(e), 2), "return": round(float(r), 6)}
        for d, e, r in zip(strategy_eq.index, strategy_eq.values, strategy_aligned.values)
    ]
    null_series = [
        {"date": d.strftime("%Y-%m-%d"), "equity": round(float(e), 2), "return": round(float(r), 6)}
        for d, e, r in zip(null_eq.index, null_eq.values, null_aligned.values)
    ]

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_paired_rebalances": int(len(shared)),
        "strategy_metrics": {k: float(v) for k, v in strategy_metrics.items()},
        "strategy_ci": strategy_cis,
        "null_metrics": {k: float(v) for k, v in null_metrics.items()},
        "null_ci": null_cis,
        "sharpe_diff_test": diff,
        "verdict": "PASS" if verdict_passed else "INVESTIGATE",
        "verdict_caption": (
            "Strategy Sharpe exceeds vol-targeted SPY by Δ "
            f"{diff['point_diff']:+.3f} (paired bootstrap p_two_sided = "
            f"{diff['p_value_two_sided']:.3f}). The residual edge attributable "
            "to the ML ranker is the band above the null curve."
            if verdict_passed else
            "Strategy and vol-targeted SPY differ by Δ Sharpe "
            f"{diff['point_diff']:+.3f} but the paired bootstrap fails to reject "
            f"the null at the 10% level (p_two_sided = {diff['p_value_two_sided']:.3f}). "
            "Acknowledge openly: on this 6.4-year sample with 11 large-cap survivors, "
            "vol-targeting alone may carry most of the apparent edge."
        ),
        "config": {
            "cost_per_leg_bps": COST_PER_LEG * 10000,
            "holding_days": HOLDING_DAYS,
            "initial_capital": INITIAL_CAPITAL,
            "vol_target_annual": 0.15,
            "max_weight": 0.40,
            "regime_off_multiplier": REGIME_OFF_MULTIPLIER,
        },
        "strategy_series": strategy_series,
        "null_series": null_series,
    }
    path = results_dir / "null_test.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")
