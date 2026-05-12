"""
Pure, stateless risk-management primitives shared by the offline backtest
and the live auto-trader.

No I/O, no globals. Every function takes data as arguments and returns a
new value. Composition lives in `size_basket`.

Defaults match the team-approved plan:
- target_vol = 15% annualized per pick
- max_weight = 0.40 per pick, gross_cap = 1.0
- realized vol window = 20 trading days, floor 8% annualized
- regime gate: half-size when SPY < 200dMA AND VIX > 75th pct of trailing 2y
- correlation cap: drop later picks with 60d correlation > 0.80 vs any kept earlier pick
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# Defaults
TARGET_VOL_ANNUAL = 0.15
MAX_WEIGHT = 0.40
GROSS_CAP = 1.0
VOL_WINDOW = 20
VOL_FLOOR_ANNUAL = 0.08
CORR_WINDOW = 60
CORR_THRESHOLD = 0.80
REGIME_VIX_PCT_THRESHOLD = 0.75
REGIME_VIX_LOOKBACK_DAYS = 504  # ~2 trading years
REGIME_OFF_MULTIPLIER = 0.5
TRADING_DAYS_PER_YEAR = 252

Regime = Literal["risk_on", "risk_off"]


@dataclass
class PickDiagnostic:
    ticker: str
    realized_vol_20d: float | None = None
    raw_weight: float | None = None
    final_weight: float = 0.0
    dropped_reason: str | None = None


@dataclass
class BasketDiagnostics:
    asof_date: str
    regime: Regime
    spy_close: float | None
    spy_ma200: float | None
    vix: float | None
    vix_pct_rank: float | None
    gross_weight: float
    regime_multiplier: float
    picks: list[PickDiagnostic] = field(default_factory=list)


# Primitives


def realized_vol(returns: pd.Series, window: int = VOL_WINDOW) -> float:
    """
    Annualized realized vol from the last `window` daily returns.
    Floored at VOL_FLOOR_ANNUAL to prevent absurd weights on quiet stocks.
    Returns NaN if there aren't enough non-null observations.
    """
    tail = returns.dropna().tail(window)
    if len(tail) < window:
        return float("nan")
    daily_std = float(tail.std(ddof=1))
    annual = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return max(annual, VOL_FLOOR_ANNUAL)


def vol_target_weight(
    pick_vol_annual: float,
    target: float = TARGET_VOL_ANNUAL,
    max_w: float = MAX_WEIGHT,
) -> float:
    """w = target / pick_vol, capped at max_w."""
    if pick_vol_annual <= 0 or np.isnan(pick_vol_annual):
        return 0.0
    w = target / pick_vol_annual
    return min(w, max_w)


def vix_percentile_rank(
    vix_series: pd.Series,
    asof_date: pd.Timestamp,
    lookback_days: int = REGIME_VIX_LOOKBACK_DAYS,
) -> float | None:
    """
    Percentile rank (0..1) of the most-recent VIX close at `asof_date`
    within the trailing `lookback_days` window. Rolling, no lookahead.
    Returns None if insufficient history.
    """
    s = vix_series.loc[:asof_date].dropna()
    if len(s) < lookback_days // 2:
        return None
    window = s.tail(lookback_days)
    current = window.iloc[-1]
    rank = (window <= current).mean()
    return float(rank)


def regime_label(
    spy_close: float | None,
    spy_ma200: float | None,
    vix: float | None,
    vix_pct_rank: float | None,
    vix_pct_threshold: float = REGIME_VIX_PCT_THRESHOLD,
) -> Regime:
    """
    risk_off iff SPY < 200dMA AND VIX percentile rank > threshold.
    Both conditions required; either alone fires too often (~30% of days)
    and kills CAGR. Returns risk_on whenever any input is missing.
    """
    if spy_close is None or spy_ma200 is None or vix is None or vix_pct_rank is None:
        return "risk_on"
    spy_below = spy_close < spy_ma200
    vix_elevated = vix_pct_rank > vix_pct_threshold
    return "risk_off" if (spy_below and vix_elevated) else "risk_on"


def correlation_filter(
    picks_in_rank_order: list[str],
    returns_window: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> tuple[list[str], dict[str, str]]:
    """
    Walk picks in rank order. Keep the first; for each subsequent pick,
    drop it if its `returns_window` correlation with any *already-kept*
    pick exceeds `threshold`.

    `returns_window` is a DataFrame indexed by date with one column per
    ticker (the trailing CORR_WINDOW of daily returns).

    Returns (kept_picks, drop_reasons) where drop_reasons maps the dropped
    ticker to a human-readable explanation.
    """
    kept: list[str] = []
    dropped: dict[str, str] = {}
    if returns_window.empty:
        return picks_in_rank_order, dropped

    corr = returns_window.corr()
    for t in picks_in_rank_order:
        if t not in corr.columns:
            kept.append(t)
            continue
        too_correlated_with = None
        for k in kept:
            if k in corr.columns and not pd.isna(corr.loc[t, k]) and corr.loc[t, k] > threshold:
                too_correlated_with = k
                break
        if too_correlated_with:
            dropped[t] = f"corr {corr.loc[t, too_correlated_with]:.2f} with {too_correlated_with} (>{threshold:.2f})"
        else:
            kept.append(t)
    return kept, dropped


def _wide_returns_from_prices(prices_long: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Pivot a long-format prices DataFrame (columns: date, ticker, close) into
    a wide returns DataFrame (one column per ticker).
    """
    sub = prices_long[prices_long["ticker"].isin(tickers)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    wide = sub.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide.pct_change()


# Orchestrator


def size_basket(
    picks_in_rank_order: list[str],
    prices_long: pd.DataFrame,
    spy_history: pd.Series,
    vix_history: pd.Series,
    asof_date: pd.Timestamp | str,
    target_vol: float = TARGET_VOL_ANNUAL,
    max_weight: float = MAX_WEIGHT,
    gross_cap: float = GROSS_CAP,
    corr_threshold: float = CORR_THRESHOLD,
    vol_window: int = VOL_WINDOW,
    corr_window: int = CORR_WINDOW,
    enable_vol_target: bool = True,
    enable_regime_gate: bool = True,
    enable_corr_filter: bool = True,
) -> tuple[dict[str, float], BasketDiagnostics]:
    """
    Convert a ranked list of picks into per-pick portfolio weights using:
      1. 20d realized-vol-targeted sizing per pick  (enable_vol_target)
      2. correlation filter to drop redundant later picks  (enable_corr_filter)
      3. regime multiplier (half-size when SPY < 200dMA AND VIX > 75th pct)
         (enable_regime_gate)
      4. gross cap (normalize down only if sum > gross_cap)

    The three `enable_*` flags exist for ablation studies. When False:
      - vol_target off  → equal weight 1/N per pick (matches v1 baseline)
      - regime_gate off → multiplier always 1.0
      - corr_filter off → no picks dropped on correlation

    Returns (weights, diagnostics). weights only contains kept tickers.
    """
    asof = pd.Timestamp(asof_date)

    diag = BasketDiagnostics(
        asof_date=asof.strftime("%Y-%m-%d"),
        regime="risk_on",
        spy_close=None,
        spy_ma200=None,
        vix=None,
        vix_pct_rank=None,
        gross_weight=0.0,
        regime_multiplier=1.0,
    )

    if not picks_in_rank_order:
        return {}, diag

    # 1. Per-pick raw weight. Either vol-targeted or equal-weight (baseline).
    returns_wide = _wide_returns_from_prices(prices_long, picks_in_rank_order)
    returns_upto = returns_wide.loc[:asof]

    pick_diags: dict[str, PickDiagnostic] = {t: PickDiagnostic(ticker=t) for t in picks_in_rank_order}
    equal_w = 1.0 / len(picks_in_rank_order)
    for t in picks_in_rank_order:
        if t in returns_upto.columns:
            rv = realized_vol(returns_upto[t], window=vol_window)
            pick_diags[t].realized_vol_20d = None if np.isnan(rv) else float(rv)
            if enable_vol_target:
                if not np.isnan(rv):
                    pick_diags[t].raw_weight = vol_target_weight(rv, target=target_vol, max_w=max_weight)
            else:
                # Baseline: equal weight as long as the ticker has any history.
                pick_diags[t].raw_weight = equal_w
        else:
            pick_diags[t].dropped_reason = "no price history"

    # 2. Correlation filter (optional)
    pickable = [t for t in picks_in_rank_order if pick_diags[t].raw_weight is not None]
    if enable_corr_filter:
        corr_window_df = returns_upto.tail(corr_window).dropna(how="all")
        kept, dropped_by_corr = correlation_filter(pickable, corr_window_df, threshold=corr_threshold)
        for t, reason in dropped_by_corr.items():
            pick_diags[t].dropped_reason = reason
        for t in pickable:
            if t not in kept and pick_diags[t].dropped_reason is None:
                pick_diags[t].dropped_reason = "filtered"
    else:
        kept = pickable

    # 3. Regime gate (optional). When disabled the multiplier stays 1.0 and
    # we don't even compute the regime label — keeps the ablation honest.
    if enable_regime_gate:
        spy_upto = spy_history.loc[:asof].dropna()
        if len(spy_upto) >= 200:
            diag.spy_close = float(spy_upto.iloc[-1])
            diag.spy_ma200 = float(spy_upto.tail(200).mean())
        vix_upto = vix_history.loc[:asof].dropna()
        if len(vix_upto) > 0:
            diag.vix = float(vix_upto.iloc[-1])
            diag.vix_pct_rank = vix_percentile_rank(vix_history, asof)
        diag.regime = regime_label(diag.spy_close, diag.spy_ma200, diag.vix, diag.vix_pct_rank)
        diag.regime_multiplier = REGIME_OFF_MULTIPLIER if diag.regime == "risk_off" else 1.0

    # 4. Apply regime + gross cap, write final weights
    raw_total = sum(pick_diags[t].raw_weight for t in kept)
    regime_scaled_total = raw_total * diag.regime_multiplier
    normalizer = (gross_cap / regime_scaled_total) if regime_scaled_total > gross_cap else 1.0

    weights: dict[str, float] = {}
    for t in kept:
        w = pick_diags[t].raw_weight * diag.regime_multiplier * normalizer
        pick_diags[t].final_weight = float(w)
        weights[t] = float(w)

    diag.gross_weight = float(sum(weights.values()))
    diag.picks = [pick_diags[t] for t in picks_in_rank_order]

    return weights, diag


# CLI smoke-test

if __name__ == "__main__":
    # Synthetic data: two tickers with different vols + a clearly risk-off day
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", "2024-01-01", freq="B")
    n = len(dates)

    # Ticker A: 10% annual vol; Ticker B: 30% annual vol (would naturally get 1/3 weight)
    a_rets = rng.normal(0.0003, 0.10 / np.sqrt(252), n)
    b_rets = rng.normal(0.0003, 0.30 / np.sqrt(252), n)
    a_close = 100 * np.exp(np.cumsum(a_rets))
    b_close = 100 * np.exp(np.cumsum(b_rets))

    prices = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["A"] * n + ["B"] * n,
        "close": list(a_close) + list(b_close),
    })

    # SPY in clear uptrend most of the time
    spy_rets = rng.normal(0.0005, 0.15 / np.sqrt(252), n)
    spy = pd.Series(100 * np.exp(np.cumsum(spy_rets)), index=dates)

    # VIX: a calm baseline with one spike near the end
    vix = pd.Series(15 + rng.normal(0, 2, n), index=dates).clip(lower=9)
    vix.iloc[-50:] = 35  # spike near end of sample

    # Drop SPY below MA200 too, so risk-off triggers
    spy.iloc[-50:] = spy.iloc[-300:-50].mean() * 0.85

    asof = dates[-1]
    weights, diag = size_basket(["A", "B"], prices, spy, vix, asof)
    print("=== End of sample (engineered risk-off) ===")
    print(f"  Regime: {diag.regime}, SPY {diag.spy_close:.2f} vs MA200 {diag.spy_ma200:.2f}, "
          f"VIX {diag.vix:.2f} (pct {diag.vix_pct_rank:.2f}), mult {diag.regime_multiplier}")
    for p in diag.picks:
        print(f"  {p.ticker}: vol={p.realized_vol_20d}, raw={p.raw_weight}, "
              f"final={p.final_weight:.4f}, dropped={p.dropped_reason}")
    print(f"  Gross: {diag.gross_weight:.4f}")

    # And in the middle of sample where we expect risk_on
    asof_mid = dates[len(dates) // 2]
    weights2, diag2 = size_basket(["A", "B"], prices, spy, vix, asof_mid)
    print("\n=== Middle of sample (engineered risk-on) ===")
    print(f"  Regime: {diag2.regime}, mult {diag2.regime_multiplier}")
    for p in diag2.picks:
        print(f"  {p.ticker}: final={p.final_weight:.4f}")
    print(f"  Gross: {diag2.gross_weight:.4f}")
