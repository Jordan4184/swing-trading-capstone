# Swing Trading Capstone — ML-Based Cross-Sectional Equity Strategy

> **TL;DR** — ML swing strategy on 11 US large-caps, 2019-07 → 2025-12 walk-forward backtest. **v2 Sharpe 1.09** with vol-targeting + regime gate, **statistically distinguishable** from a vol-targeted SPY null at paired-bootstrap *p* = 0.029. Edge **does not generalise** across two alternative 11-name universes (Sharpe collapses to 0.33 / 0.08) — documented, not hidden. Stack: Random Forest, Python pipeline + FastAPI + Next.js, paper-trading via Alpaca.

A machine-learning swing trading system built as a capstone project for the Institute of Data — Data Science & AI program. Trains classifiers to predict which stocks in a small US large-cap universe will rank in the top quintile of 5-day forward returns, then backtests the resulting strategy with realistic transaction costs against multiple benchmarks.

**Why this project, why now:** FINRA's removal of the Pattern Day Trader rule on June 4, 2026 eliminates the $25,000 minimum equity requirement that has historically gated retail algorithmic trading. Small-account algo strategies are about to become viable for a much larger audience. This project explores whether retail-grade ML — with cheap data and standard tools — can produce a genuine, defensible edge over passive benchmarks.

## Why I built this

I came to ML from a neuroscience background, and quantitative finance turned out to be the same problem in different clothing: noisy signals, low signal-to-noise ratio, the constant temptation to overfit within-subject variance, and the very human seduction of telling post-hoc stories about a single trial that should have been a single observation. Every failure mode that kills a neuroscience PhD's first model kills a quant model the same way. That made markets feel like a forcing function for the discipline I wanted to practice next.

The most important thing I learned wasn't in the headline numbers. At week 2 I believed that lifting walk-forward AUC was equivalent to lifting realized return — I had to ship and then *revert* a feature engineering change to learn otherwise. AUC measures ranking across the whole distribution; the strategy only takes the extreme tail of that distribution; the relationship between those two metrics at small N is not monotonic. That experiment is preserved in `results/feature_ablation.json` — I shipped it because the result is the lesson.

With three more months I'd build the implied-vol / earnings-proximity feature block the failure-mode case studies on the Evaluation page already point at, and a point-in-time-universe robustness test against survivorship bias. The architecture is set up for both; the artifact list keeps me honest about what's earned and what's still aspirational.

## Results Summary

Out-of-sample backtest, 2019-07 to 2025-12 (6.4 years), 20bps round-trip transaction costs, walk-forward validated:

| Metric            | v1 (no risk layer) | v2 (vol-target + regime gate) | Equal-Weight Universe | SPY    |
| ----------------- | ------------------ | ----------------------------- | --------------------- | ------ |
| Total Return      | 561%               | **314%**                      | 413%                  | 151%   |
| Annualized Return | 34.1%              | **24.7%**                     | 28.9%                 | 15.4%  |
| Sharpe Ratio      | 0.95               | **1.09**                      | 1.22                  | 0.81   |
| Max Drawdown      | -54.1%             | **-34.3%**                    | -32.1%                | -33.7% |
| Hit Rate          | 54.8%              | 54.5%                         | 56.6%                 | 55.5%  |

**Two flavours of the strategy.** v1 is the original ML ranker: top-2 picks per day, equal-weighted, non-overlapping 5-day holds. v2 layers vol-targeted position sizing (15% annualized per pick, 0.40 max weight), a regime gate (half-size when SPY < 200dMA AND VIX > 75th percentile of trailing 2y), and a 60-day correlation filter on the 2nd pick. v2 trades headline return for risk-adjusted return: roughly +0.14 Sharpe and -20pp on max drawdown, at the cost of ~9pp of annualized return.

Random Forest selected as best model by AUC (0.594) across walk-forward folds; LightGBM and Logistic Regression performed similarly, suggesting the predictive signal is captured by feature engineering rather than complex non-linear interactions.

### Does the edge survive stress tests?

The headline numbers above sit on one universe draw and one set of rebalance dates. Two stress tests stress them:

| Test | Result | Verdict |
| ---- | ------ | ------- |
| **Null test** — strategy vs vol-targeted SPY (same regime gate, same rebalance dates, same cost model — but no model) | Strategy 1.09 Sharpe vs null 0.33 Sharpe, Δ +0.76 [+0.11, +1.40], paired bootstrap *p* = 0.029 | **PASS** at 5% — alpha is distinguishable from vol-targeting alone |
| **Universe robustness** — re-run walk-forward CV + top-2 backtest on a sector-rotated 11-name set | Sharpe 1.00 → **0.33** | FAIL |
| **Universe robustness** — re-run on a different S&P 100 draw (mid-cap-skew, less megacap-tech) | Sharpe 1.00 → **0.08** | FAIL |

The null test rules out "you just took more risk with vol-targeting." The universe tests show the edge is **concentrated in the headline 11-name universe over 2018-2025** — a regime that favoured exactly those names. Both findings are persisted (`results/null_test.json`, `results/universe_robustness.json`) and visible on the dashboard's Evaluation page.

### Honest limits — what this project does and doesn't show

**The edge is real vs a fair benchmark.** A vol-targeted long-only SPY position over the same dates, with the same regime gate and cost model, delivers 0.33 Sharpe vs the strategy's 1.09. The Δ Sharpe CI is entirely above zero. Vol-targeting is not the whole story — the ranker is contributing measurable signal on this universe.

**The edge is universe-dependent.** When the same procedure is fit on two alternative 11-name universes — one a sector rotation away from megacap-tech, one a different draw from the S&P 100 neighbourhood — Sharpe collapses to 0.33 and 0.08 respectively. The honest interpretation is that the ML ranker is exploiting structure specific to AAPL/AMZN/META/NVDA/TSLA et al. across the 2018-2025 regime. Whether the *procedure* generalises to other universes, on out-of-sample windows, is genuinely uncertain on this sample. The fix is either re-fitting per universe or building features that abstract away from individual-name idiosyncrasy (implied vol regimes, earnings-proximity flags) — both are on the "what next" list above, neither is shipped.

### Feature engineering — what we tested but didn't ship

Because the model's target is cross-sectional ("top quintile per date") but the features are absolute (RSI=50, return=2%), we tested a **per-date rank-pct** alternative — each feature value replaced by its rank across the 11-ticker universe on that date. The infrastructure lives at `src/features.py::build_features_ranked` and the comparison is persisted in `results/feature_ablation.json`.

Walk-forward AUC improved from **0.594 → 0.606** (+1.19pp) and fold-to-fold std dropped from 0.031 → 0.022 (more stable). However, when we then ran the full pipeline through to a backtest, the realized top-2-pick Sharpe slipped (v1 Sharpe 0.95 → 0.76; v2 1.09 → 0.82) — well inside the bootstrap CI overlap, but the wrong direction nonetheless. **We chose to keep production on absolute features.** AUC measures ranking across the whole distribution; the strategy only takes the extreme high end, and the relationship between those two metrics is not monotonic at small N. This is exactly the kind of finding worth running and documenting: the rank-feature experiment is preserved as a research artifact on the Evaluation page, but production didn't change.

> **Note on Sharpe annualization.** Earlier versions of this README quoted 2.13 for v1; that figure annualized 5-day-trade returns with √252 instead of √(252/5). All numbers above use the correct √50.4 annualization, matching the convention used for the daily-frequency benchmarks (SPY, equal-weight).

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
swing-trading-capstone/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── src/                              # Reproducible pipeline (CLI-runnable)
│   ├── data_loader.py                # OHLCV download + caching
│   ├── features.py                   # Feature engineering
│   ├── models.py                     # Walk-forward training + comparison
│   ├── backtest.py                   # Strategy simulation with costs
│   └── pipeline.py                   # End-to-end CLI orchestration
├── notebooks/
│   └── 01_eda.ipynb                  # Exploratory data analysis (separate from pipeline)
├── docs/
│   └── roadmap/
│       └── 01_crowd_consensus_meter.md  # Future enhancement: contrarian-sentiment indicator
├── results/                          # Generated artifacts (plots, JSONs, predictions)
└── data/                             # Cached data (gitignored)
```

The notebook is intentionally separated from the pipeline — exploration is iterative and messy by design, while the pipeline is reproducible and deployable.

## Quick Start

### Setup

```bash
git clone https://github.com/Jordan4184/swing-trading-capstone.git
cd swing-trading-capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python -m src.pipeline --all
```

This downloads data, builds features, trains models, and runs the backtest. Takes ~2-3 minutes end-to-end.

### Run individual steps

```bash
python -m src.pipeline --download-data    # just refresh data
python -m src.pipeline --train            # retrain models
python -m src.pipeline --backtest         # rerun backtest only
```

### Explore in a notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Key Visualizations

Generated to `results/` after running the pipeline:

- `01_price_performance.png` — normalized price chart of the universe
- `02_target_distribution.png` — distribution of forward returns
- `03_autocorrelation.png` — autocorrelation by ticker and lag
- `04_correlation_matrix.png` — return correlation heatmap
- `05_feature_importance.png` — LightGBM feature importance
- `06_equity_curve.png` — strategy vs. benchmarks (the headline chart)
- `07_drawdown.png` — strategy drawdown over time

## Findings

### What worked

- Cross-sectional ranking is more learnable than absolute return prediction
- Volatility regime is the strongest single feature (top by importance)
- Multi-horizon returns capture both mean reversion and momentum
- Walk-forward validation produces honest performance estimates
- The strategy delivers ~5pp of annualized alpha vs. an equal-weight benchmark of the same universe

### What didn't

- Daily-horizon features rank lowest in importance — daily price action is dominated by noise
- Linear and non-linear models perform similarly, suggesting the signal lives in feature engineering rather than complex interactions
- The strategy's higher max drawdown (-54% vs. -32% for equal-weight) is the explicit cost of concentration

### Honest caveats

- Yahoo Finance is an unofficial data source; production deployment would use Polygon or similar
- 11-ticker universe is small; results may not generalize to broader universes
- 6.4-year period covers an exceptional bull market for tech; performance in a different regime is unknown
- Transaction costs may be optimistic — real slippage on small accounts can be larger

## Future Work

- **Crowd Consensus Meter** (designed in `docs/roadmap/01_crowd_consensus_meter.md`): a sentiment indicator combining Reddit, StockTwits, and Google Trends data to identify when crowd positioning is extreme — used as a filter on model signals to flag contrarian setups vs. consensus trades
- **Real-time deployment**: streaming OHLCV via Polygon WebSocket, paper-trading via Alpaca API
- **AI agent layer**: when the model surfaces a candidate, an LLM call summarizes recent news/SEC filings to provide qualitative context
- **Universe expansion**: scale from 11 to ~500 names (S&P 500); evaluate whether selection edge persists at scale
- **Regime-aware modeling**: separate models for trending vs. mean-reverting regimes, switching based on volatility/breadth indicators

## Stack

- **Python 3.12** — core language
- **pandas / numpy / pyarrow** — data manipulation
- **scikit-learn** — Logistic Regression, Random Forest, walk-forward CV, pipeline tooling
- **lightgbm** — gradient boosted trees
- **matplotlib / seaborn** — visualization
- **yfinance** — data source
- **Jupyter** — exploratory analysis

## Author

Jordan Donaldson

- Capstone project for Institute of Data — Data Science & AI program
- Background in Neuroscience; transitioning to ML/AI engineering
- [GitHub](https://github.com/Jordan4184) | itsjordandonaldson@gmail.com

## License

Educational project. Not financial advice. Past performance does not guarantee future results.
