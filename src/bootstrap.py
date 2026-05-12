"""
IID bootstrap CIs for backtest metrics.

Sound for this project because trades are **non-overlapping** (the
backtest's load-bearing invariant) — per-trade returns are treated as
independent draws from a single distribution, so resample-with-replacement
gives a valid sampling distribution for Sharpe / CAGR / MaxDD without
needing block bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import build_equity_curve, compute_metrics, INITIAL_CAPITAL


DEFAULT_N_RESAMPLES = 1000
DEFAULT_ALPHA = 0.05  # 95% CI


def _quantile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def bootstrap_metrics(
    returns: pd.Series,
    periods_per_year: float,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict:
    """
    Resample `returns` (per-trade return series) with replacement n_resamples
    times. For each resample: rebuild the equity curve and recompute metrics.
    Return point estimates + (lo, hi) percentile CIs for the headline metrics.

    The point estimates come from the un-resampled series, not the mean of
    the bootstrap distribution, so they match what the existing JSON shows.
    """
    rets = returns.dropna().to_numpy()
    if len(rets) < 5:
        raise ValueError(f"Bootstrap requires at least 5 trades; got {len(rets)}")

    rng = np.random.default_rng(seed)
    n = len(rets)

    # Point estimates from the original sequence (preserves trade ordering
    # so the equity curve dates line up with the actual backtest).
    base_equity = build_equity_curve(returns)
    base = compute_metrics(base_equity, returns, periods_per_year=periods_per_year)

    sharpes = np.empty(n_resamples)
    cagrs = np.empty(n_resamples)
    maxdds = np.empty(n_resamples)
    hit_rates = np.empty(n_resamples)
    total_returns = np.empty(n_resamples)

    n_years = base["n_years"] if base["n_years"] > 0 else n / periods_per_year

    for i in range(n_resamples):
        sample = rng.choice(rets, size=n, replace=True)
        # Equity curve from a bare ndarray, no date index needed for metrics
        equity = INITIAL_CAPITAL * np.cumprod(1.0 + sample)

        total_return = equity[-1] / INITIAL_CAPITAL - 1.0
        cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        std = sample.std(ddof=1)
        sharpe = (sample.mean() / std) * np.sqrt(periods_per_year) if std > 0 else 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        maxdd = float(drawdown.min())

        sharpes[i] = sharpe
        cagrs[i] = cagr
        maxdds[i] = maxdd
        hit_rates[i] = float((sample > 0).mean())
        total_returns[i] = total_return

    def pack(point: float, samples: np.ndarray) -> dict:
        lo, hi = _quantile_ci(samples, alpha)
        return {
            "point": float(point),
            "ci_low": lo,
            "ci_high": hi,
            "ci_alpha": alpha,
        }

    return {
        "n_resamples": n_resamples,
        "alpha": alpha,
        "ci_level": 1 - alpha,
        "n_trades": int(n),
        "n_years": float(n_years),
        "sharpe_ratio": pack(base["sharpe_ratio"], sharpes),
        "annualized_return": pack(base["annualized_return"], cagrs),
        "max_drawdown": pack(base["max_drawdown"], maxdds),
        "hit_rate": pack(base["hit_rate"], hit_rates),
        "total_return": pack(base["total_return"], total_returns),
    }


if __name__ == "__main__":
    # Smoke test against the v1 strategy output
    import json
    from pathlib import Path

    from src.backtest import simulate_strategy, HOLDING_DAYS, TOP_N

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    preds = pd.read_parquet(results_dir / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    trades = simulate_strategy(preds, top_n=TOP_N)
    rets = trades.set_index("date")["basket_return_net"]

    cis = bootstrap_metrics(rets, periods_per_year=252 / HOLDING_DAYS, n_resamples=1000)

    print("=== Bootstrap CIs on v1 (1000 resamples) ===")
    for k, v in cis.items():
        if isinstance(v, dict):
            print(f"  {k:22s} {v['point']:+.4f}   95% CI [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
        else:
            print(f"  {k:22s} {v}")
