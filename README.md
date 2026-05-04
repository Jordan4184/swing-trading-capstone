# Swing Trading Capstone — ML-Based Cross-Sectional Equity Strategy

A machine-learning swing trading system built as a capstone project for the Institute of Data — Data Science & AI program. Trains classifiers to predict which stocks in a small US large-cap universe will rank in the top quintile of 5-day forward returns, then backtests the resulting strategy with realistic transaction costs against multiple benchmarks.

**Why this project, why now:** FINRA's removal of the Pattern Day Trader rule on June 4, 2026 eliminates the $25,000 minimum equity requirement that has historically gated retail algorithmic trading. Small-account algo strategies are about to become viable for a much larger audience. This project explores whether retail-grade ML — with cheap data and standard tools — can produce a genuine, defensible edge over passive benchmarks.

## Results Summary

Out-of-sample backtest, 2019-07 to 2025-12 (6.4 years), 10bps round-trip transaction costs, walk-forward validated:

| Metric            | ML Strategy | Equal-Weight Universe | SPY    |
| ----------------- | ----------- | --------------------- | ------ |
| Total Return      | **561%**    | 413%                  | 151%   |
| Annualized Return | **34.1%**   | 28.9%                 | 15.4%  |
| Sharpe Ratio      | **2.13**    | 1.22                  | 0.81   |
| Max Drawdown      | -54.1%      | -32.1%                | -33.7% |
| Hit Rate          | 54.8%       | 56.6%                 | 55.5%  |

**Honest alpha vs. equal-weight benchmark: +5.19pp annualized, +0.91 Sharpe.** The strategy outperforms a naive equal-weight allocation across the same universe — meaning the ML edge is distinct from simply being long high-momentum names.

Random Forest selected as best model by AUC (0.595) across walk-forward folds; LightGBM and Logistic Regression performed similarly, suggesting the predictive signal is captured by feature engineering rather than complex non-linear interactions.

## Methodology

### Data

- **Source:** Yahoo Finance via `yfinance`
- **Universe:** 11 large-cap US stocks (AAPL, AMZN, JNJ, JPM, MCD, META, NVDA, PFE, TSLA, UNH) plus SPY benchmark
- **Period:** 2018-01 to 2025-12, daily OHLCV
- **Quality:** Zero missing values, balanced panel (every ticker traded continuously through the period — no survivorship bias)

### Features

Four families of features, each motivated by EDA findings or financial literature:

1. **Multi-horizon returns** (1d, 5d, 20d, 60d): captures both short-term mean reversion and longer-horizon momentum
2. **Volatility** (20d, 60d rolling std): risk regime — TSLA/NVDA distributions are visibly different from JNJ/PFE
3. **Technical indicators** (RSI-14, Bollinger %B): standard quant signals
4. **Market-relative** (volume ratio, excess return vs. SPY): isolates idiosyncratic moves from systematic exposure

### Target

Binary classification: 1 if a stock ranks in the top 20% of its universe by 5-day forward return, else 0. Cross-sectional ranking (rather than absolute return prediction) is more learnable given the dominant noise in raw price series — chosen based on EDA findings on signal-to-noise.

### Models

Three classifiers compared via walk-forward (TimeSeriesSplit) cross-validation across 5 folds:

- **Logistic Regression** with StandardScaler — linear baseline
- **Random Forest** — non-linear, handles interactions
- **LightGBM** — gradient boosted trees, typical SOTA for tabular

Walk-forward validation is critical. Random k-fold on time-series data leaks future information into training and inflates measured performance. Each fold here trains on past data and tests on subsequent unseen data, simulating real deployment.

### Backtest

- Each trading day, rank tickers by predicted probability
- Buy equal-weighted top 2 names
- Hold for 5 days, then rebalance
- **Non-overlapping trades only** — a critical correction from the initial implementation, which inflated returns by treating overlapping forward-return windows as independent
- Realistic costs: 10bps round-trip (5bps commission + 5bps slippage)
- Compared against two benchmarks: SPY buy-and-hold (market) and equal-weighted universe (apples-to-apples for stock selection)

## Project Structure

```

```
