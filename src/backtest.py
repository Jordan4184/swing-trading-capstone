"""
Backtest module for swing trading capstone.

Takes the model's predictions and simulates a trading strategy:
- Each day, rank tickers by predicted probability
- Buy top N, hold for HOLDING_DAYS, non-overlapping
- Compare equity curve to SPY benchmark AND equal-weight universe benchmark
- Account for transaction costs

Author: Jordan Donaldson
"""

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_N = 2
HOLDING_DAYS = 5
COMMISSION_PER_TRADE = 0.0005
SLIPPAGE_PER_TRADE = 0.0005
INITIAL_CAPITAL = 10_000


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def load_predictions_and_prices(predictions_path: Path, df_features: pd.DataFrame):
    """Load predictions; build SPY + equal-weight universe benchmarks."""
    preds = pd.read_parquet(predictions_path)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.sort_values(["date", "y_proba"], ascending=[True, False]).reset_index(
        drop=True
    )

    # SPY benchmark
    spy = df_features[df_features["ticker"] == "SPY"][["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["spy_return_1d"] = spy["close"].pct_change()

    # Equal-weight universe benchmark (excludes SPY)
    non_spy = df_features[df_features["ticker"] != "SPY"].copy()
    non_spy["date"] = pd.to_datetime(non_spy["date"])
    non_spy = non_spy.sort_values(["ticker", "date"])
    non_spy["return_1d"] = non_spy.groupby("ticker")["close"].pct_change()
    eq_weight = (
        non_spy.groupby("date", as_index=False)["return_1d"]
        .mean()
        .rename(columns={"return_1d": "eq_weight_return_1d"})
    )

    return preds, spy, eq_weight


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------


def simulate_strategy(preds: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Non-overlapping top-N strategy."""
    preds_ranked = preds.copy()
    preds_ranked["rank"] = preds_ranked.groupby("date")["y_proba"].rank(
        method="first", ascending=False
    )
    daily_picks = preds_ranked[preds_ranked["rank"] <= top_n].copy()

    daily_basket = (
        daily_picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return_5d"})
    )

    daily_basket = daily_basket.sort_values("date").reset_index(drop=True)
    rebalance_basket = daily_basket.iloc[::HOLDING_DAYS].copy().reset_index(drop=True)

    cost = (COMMISSION_PER_TRADE + SLIPPAGE_PER_TRADE) * 2
    rebalance_basket["basket_return_net"] = rebalance_basket["basket_return_5d"] - cost

    return rebalance_basket


def build_equity_curve(
    returns: pd.Series, initial: float = INITIAL_CAPITAL
) -> pd.Series:
    """Compound returns into an equity curve."""
    return initial * (1 + returns.fillna(0)).cumprod()


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: float = 252,
) -> dict:
    """Standard performance metrics."""
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    start_date = pd.to_datetime(equity.index.min())
    end_date = pd.to_datetime(equity.index.max())
    n_calendar_days = (end_date - start_date).days
    n_years = n_calendar_days / 365.25

    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    period_mean = returns.mean()
    period_std = returns.std()
    sharpe = (
        (period_mean / period_std) * np.sqrt(periods_per_year) if period_std > 0 else 0
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    hit_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_periods": len(returns),
        "n_years": round(n_years, 2),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_equity_curves(
    strategy_eq: pd.Series,
    spy_eq: pd.Series,
    eq_weight_eq: pd.Series,
    save_path: Path,
):
    """Strategy vs SPY vs Equal-Weight Universe equity curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        strategy_eq.index,
        strategy_eq.values,
        label="ML Strategy",
        linewidth=2.5,
        color="#2E86AB",
    )
    ax.plot(
        eq_weight_eq.index,
        eq_weight_eq.values,
        label="Equal-Weight Universe (honest benchmark)",
        linewidth=2,
        color="#06A77D",
        linestyle="-.",
    )
    ax.plot(
        spy_eq.index,
        spy_eq.values,
        label="SPY Buy & Hold",
        linewidth=2,
        color="#A23B72",
        linestyle="--",
    )
    ax.set_title("Strategy vs. Benchmarks — Equity Curve (Out-of-Sample)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_drawdown(equity: pd.Series, save_path: Path):
    """Underwater chart for the strategy."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#E63946", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="#E63946", linewidth=1)
    ax.set_title("Strategy Drawdown ('Underwater' Chart)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Helper for printing metrics
# ---------------------------------------------------------------------------


def print_metrics(title: str, metrics: dict):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float) and abs(v) < 1:
            print(f"  {k:25s} {v:>10.4f}  ({v*100:.2f}%)")
        else:
            print(f"  {k:25s} {v:>10.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.features import build_features

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    print("Loading predictions and price data...")
    df = build_features(load_data())
    preds, spy, eq_weight = load_predictions_and_prices(
        results_dir / "predictions.parquet", df
    )
    print(f"  Predictions: {len(preds):,} rows")
    print(f"  Date range: {preds['date'].min().date()} to {preds['date'].max().date()}")

    print(
        f"\nSimulating strategy: long top {TOP_N}, {HOLDING_DAYS}-day non-overlapping..."
    )
    trades = simulate_strategy(preds, top_n=TOP_N)

    # Strategy: per-trade returns indexed by date
    strategy_returns = trades.set_index("date")["basket_return_net"]

    # Benchmarks aligned to strategy date range
    spy_returns = spy.set_index("date")["spy_return_1d"]
    spy_returns = spy_returns.loc[
        strategy_returns.index.min() : strategy_returns.index.max()
    ]

    eq_weight_returns = eq_weight.set_index("date")["eq_weight_return_1d"]
    eq_weight_returns = eq_weight_returns.loc[
        strategy_returns.index.min() : strategy_returns.index.max()
    ]

    # Equity curves
    strategy_equity = build_equity_curve(strategy_returns)
    spy_equity = build_equity_curve(spy_returns)
    eq_weight_equity = build_equity_curve(eq_weight_returns)

    # Metrics (note: strategy compounds per-trade, so periods_per_year = 252/HOLDING_DAYS)
    strategy_metrics = compute_metrics(
        strategy_equity, strategy_returns, periods_per_year=252 / HOLDING_DAYS
    )
    spy_metrics = compute_metrics(spy_equity, spy_returns, periods_per_year=252)
    eq_metrics = compute_metrics(
        eq_weight_equity, eq_weight_returns, periods_per_year=252
    )

    print_metrics("STRATEGY PERFORMANCE", strategy_metrics)
    print_metrics("SPY BENCHMARK", spy_metrics)
    print_metrics("EQUAL-WEIGHT UNIVERSE BENCHMARK (apples-to-apples)", eq_metrics)

    print("\n" + "=" * 60)
    print("STRATEGY vs SPY")
    print("=" * 60)
    print(
        f"  Total return advantage:    "
        f"{(strategy_metrics['total_return'] - spy_metrics['total_return'])*100:+.2f}%"
    )
    print(
        f"  Annualized advantage:      "
        f"{(strategy_metrics['annualized_return'] - spy_metrics['annualized_return'])*100:+.2f}%"
    )
    print(
        f"  Sharpe advantage:          "
        f"{strategy_metrics['sharpe_ratio'] - spy_metrics['sharpe_ratio']:+.4f}"
    )

    print("\n" + "=" * 60)
    print("STRATEGY vs EQUAL-WEIGHT UNIVERSE (the honest comparison)")
    print("=" * 60)
    print(
        f"  Total return advantage:    "
        f"{(strategy_metrics['total_return'] - eq_metrics['total_return'])*100:+.2f}%"
    )
    print(
        f"  Annualized advantage:      "
        f"{(strategy_metrics['annualized_return'] - eq_metrics['annualized_return'])*100:+.2f}%"
    )
    print(
        f"  Sharpe advantage:          "
        f"{strategy_metrics['sharpe_ratio'] - eq_metrics['sharpe_ratio']:+.4f}"
    )

    # Plots
    plot_equity_curves(
        strategy_equity,
        spy_equity,
        eq_weight_equity,
        results_dir / "06_equity_curve.png",
    )
    plot_drawdown(strategy_equity, results_dir / "07_drawdown.png")
    print(f"\nPlots saved to {results_dir}/")

    # Bootstrap CIs on strategy metrics (IID resample of per-trade returns;
    # sound because trades are non-overlapping).
    from src.bootstrap import bootstrap_metrics
    strategy_cis = bootstrap_metrics(strategy_returns, periods_per_year=252 / HOLDING_DAYS)
    print("\n=== Bootstrap CIs (95%, 1000 resamples) ===")
    for k in ("sharpe_ratio", "annualized_return", "max_drawdown", "hit_rate", "total_return"):
        c = strategy_cis[k]
        print(f"  {k:22s} {c['point']:+.4f}   CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")

    # Persist metrics
    summary = {
        "strategy": {k: float(v) for k, v in strategy_metrics.items()},
        "strategy_ci": strategy_cis,
        "spy": {k: float(v) for k, v in spy_metrics.items()},
        "equal_weight": {k: float(v) for k, v in eq_metrics.items()},
        "config": {
            "top_n": TOP_N,
            "holding_days": HOLDING_DAYS,
            "commission_per_trade_bps": COMMISSION_PER_TRADE * 10000,
            "slippage_per_trade_bps": SLIPPAGE_PER_TRADE * 10000,
            "initial_capital": INITIAL_CAPITAL,
        },
    }
    with open(results_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {results_dir / 'backtest_summary.json'}")
# Strategy simulation


def simulate_strategy(preds: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Each TRADING_DAY (every HOLDING_DAYS days): rank tickers by y_proba,
    take equal-weighted long positions in the top N names, hold for
    HOLDING_DAYS, realize the return.

    Critical: trades are NON-OVERLAPPING. We only enter a new position
    once the previous one closes. This avoids the lookahead-style inflation
    that comes from treating overlapping forward returns as independent.
    """
    # Rank within each date
    preds_ranked = preds.copy()
    preds_ranked["rank"] = preds_ranked.groupby("date")["y_proba"].rank(
        method="first", ascending=False
    )
    daily_picks = preds_ranked[preds_ranked["rank"] <= top_n].copy()

    # Average forward-5d return for the basket on each date
    daily_basket = (
        daily_picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return_5d"})
    )

    # NON-OVERLAPPING: take every Nth row where N = HOLDING_DAYS
    # This simulates entering a fresh trade only when the previous one closes
    daily_basket = daily_basket.sort_values("date").reset_index(drop=True)
    rebalance_basket = daily_basket.iloc[::HOLDING_DAYS].copy().reset_index(drop=True)

    # Subtract round-trip costs (commission + slippage)
    cost = (COMMISSION_PER_TRADE + SLIPPAGE_PER_TRADE) * 2
    rebalance_basket["basket_return_net"] = rebalance_basket["basket_return_5d"] - cost

    return rebalance_basket


def build_equity_curve(
    daily_returns: pd.Series, initial: float = INITIAL_CAPITAL
) -> pd.Series:
    """Compound a series of daily returns into an equity curve."""
    return initial * (1 + daily_returns.fillna(0)).cumprod()


# Performance metrics


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: float = 252,
) -> dict:
    """
    Standard performance metrics.
    `periods_per_year` lets us correctly annualize whether returns are
    daily (252), weekly (52), or per-5-day-trade (~50).
    """
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    # Use calendar time for annualization, not period count
    start_date = pd.to_datetime(equity.index.min())
    end_date = pd.to_datetime(equity.index.max())
    n_calendar_days = (end_date - start_date).days
    n_years = n_calendar_days / 365.25

    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe ratio: annualize using periods_per_year
    period_mean = returns.mean()
    period_std = returns.std()
    sharpe = (
        (period_mean / period_std) * np.sqrt(periods_per_year) if period_std > 0 else 0
    )

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Hit rate (% of periods with positive return)
    hit_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_periods": len(returns),
        "n_years": round(n_years, 2),
    }


# Plots


def plot_equity_curves(strategy_eq: pd.Series, spy_eq: pd.Series, save_path: Path):
    """Strategy vs SPY equity curve. The headline chart of the project."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        strategy_eq.index,
        strategy_eq.values,
        label="ML Strategy",
        linewidth=2,
        color="#2E86AB",
    )
    ax.plot(
        spy_eq.index,
        spy_eq.values,
        label="SPY Buy & Hold",
        linewidth=2,
        color="#A23B72",
        linestyle="--",
    )
    ax.set_title(
        "Strategy vs. SPY Benchmark — Equity Curve (Out-of-Sample)", fontsize=14
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_drawdown(equity: pd.Series, save_path: Path):
    """Underwater chart — visual of drawdown over time."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#E63946", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="#E63946", linewidth=1)
    ax.set_title("Strategy Drawdown ('Underwater' Chart)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# CLI entry point

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.features import build_features

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    print("Loading predictions and price data...")
    df = build_features(load_data())
    preds, spy, eq_weight = load_predictions_and_prices(
        results_dir / "predictions.parquet", df
    )
    print(f"  Predictions: {len(preds):,} rows")
    print(f"  Date range: {preds['date'].min().date()} to {preds['date'].max().date()}")

    print(
        f"\nSimulating strategy: long top {TOP_N} ranked names, {HOLDING_DAYS}-day hold..."
    )

    trades = simulate_strategy(preds, top_n=TOP_N)

    # Strategy: compound non-overlapping 5-day trade returns
    strategy_returns = trades.set_index("date")["basket_return_net"]

    # Benchmark: SPY daily returns over the same window, then compound to
    # match the strategy's date range
    spy_returns = spy.set_index("date")["spy_return_1d"]
    spy_returns = spy_returns.loc[
        strategy_returns.index.min() : strategy_returns.index.max()
    ]

    eq_weight_returns = eq_weight.set_index("date")["eq_weight_return_1d"]
    eq_weight_returns = eq_weight_returns.loc[
        strategy_returns.index.min() : strategy_returns.index.max()
    ]

    # Build equity curves
    strategy_equity = build_equity_curve(strategy_returns)
    spy_equity = build_equity_curve(spy_returns)
    eq_weight_equity = build_equity_curve(eq_weight_returns)

    # Compute metrics
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE")
    print("=" * 60)

    strategy_metrics = compute_metrics(strategy_equity, strategy_returns)
    for k, v in strategy_metrics.items():
        if isinstance(v, float) and abs(v) < 1:
            print(f"  {k:25s} {v:>10.4f}  ({v*100:.2f}%)")
        else:
            print(f"  {k:25s} {v:>10.4f}")

    print("\n" + "=" * 60)
    print("SPY BENCHMARK")
    print("=" * 60)
    spy_metrics = compute_metrics(spy_equity, spy_returns, periods_per_year=252)

    print("\n" + "=" * 60)
    print("EQUAL-WEIGHT UNIVERSE BENCHMARK (apples-to-apples)")
    print("=" * 60)
    eq_metrics = compute_metrics(
        eq_weight_equity, eq_weight_returns, periods_per_year=252
    )
    for k, v in eq_metrics.items():
        if isinstance(v, float) and abs(v) < 1:
            print(f"  {k:25s} {v:>10.4f}  ({v*100:.2f}%)")
        else:
            print(f"  {k:25s} {v:>10.4f}")

    for k, v in spy_metrics.items():
        if isinstance(v, float) and abs(v) < 1:
            print(f"  {k:25s} {v:>10.4f}  ({v*100:.2f}%)")
        else:
            print(f"  {k:25s} {v:>10.4f}")

    # Comparison

    print("\n" + "=" * 60)
    print("STRATEGY vs EQUAL-WEIGHT UNIVERSE (the honest comparison)")
    print("=" * 60)
    print(
        f"  Total return advantage:    {(strategy_metrics['total_return'] - eq_metrics['total_return'])*100:+.2f}%"
    )
    print(
        f"  Annualized advantage:      {(strategy_metrics['annualized_return'] - eq_metrics['annualized_return'])*100:+.2f}%"
    )
    print(
        f"  Sharpe advantage:          {strategy_metrics['sharpe_ratio'] - eq_metrics['sharpe_ratio']:+.4f}"
    )

    # Plots


def plot_equity_curves(
    strategy_eq: pd.Series,
    spy_eq: pd.Series,
    eq_weight_eq: pd.Series,
    save_path: Path,
):
    """Strategy vs SPY vs Equal-Weight Universe equity curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        strategy_eq.index,
        strategy_eq.values,
        label="ML Strategy",
        linewidth=2.5,
        color="#2E86AB",
    )
    ax.plot(
        eq_weight_eq.index,
        eq_weight_eq.values,
        label="Equal-Weight Universe",
        linewidth=2,
        color="#06A77D",
        linestyle="-.",
    )
    ax.plot(
        spy_eq.index,
        spy_eq.values,
        label="SPY Buy & Hold",
        linewidth=2,
        color="#A23B72",
        linestyle="--",
    )
    ax.set_title("Strategy vs. Benchmarks — Equity Curve (Out-of-Sample)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    plot_drawdown(strategy_equity, results_dir / "07_drawdown.png")
    print(f"\nPlots saved to {results_dir}/")

    # Persist metrics
    import json

    summary = {
        "strategy": {k: float(v) for k, v in strategy_metrics.items()},
        "spy": {k: float(v) for k, v in spy_metrics.items()},
        "config": {
            "top_n": TOP_N,
            "holding_days": HOLDING_DAYS,
            "commission_per_trade_bps": COMMISSION_PER_TRADE * 10000,
            "slippage_per_trade_bps": SLIPPAGE_PER_TRADE * 10000,
            "initial_capital": INITIAL_CAPITAL,
        },
    }
    with open(results_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {results_dir / 'backtest_summary.json'}")
