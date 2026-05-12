# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project at a glance

ML swing-trading capstone for the **Institute of Data — Data Science & AI** program.

- **Universe (11)**: AAPL, AMZN, JNJ, JPM, MCD, META, NVDA, PFE, SPY, TSLA, UNH
- **Strategy**: 5-day swing, cross-sectional top-quintile ranker, Random Forest production model
- **Validation**: walk-forward CV — best fold-mean AUC ≈ **0.594**
- **Backtest v1** (no risk layer, 2019-07 → 2025-12, 20bps RT): 561% / **CAGR 34.1% / Sharpe 0.95 / MaxDD -54.1%**
- **Backtest v2** (vol-target + regime gate + correlation filter, same window): 314% / **CAGR 24.7% / Sharpe 1.09 / MaxDD -34.3%**
- **Stack**: Python 3.12 (pipeline + FastAPI backend), Next.js 16 + TypeScript + Tailwind v4 (frontend), SQLite for auto-trader state
- **Production model**: `models/rf_v1.joblib` (Random Forest, hand-promoted)
- **Venvs**: top-level `venv/` for the capstone pipeline; separate `dashboard/backend/venv/` for the API

> **Sharpe annualization gotcha.** `src/backtest.py::compute_metrics` annualizes with √(periods_per_year) where periods_per_year = 252/HOLDING_DAYS ≈ 50.4 (correct for 5-day-trade returns). An earlier version of the README used √252 instead, inflating v1 Sharpe from 0.95 to 2.13. If you re-encounter "Sharpe 2.13" anywhere, that's the bug. The numbers above use the corrected √50.4.

## Two layers, one project

This repo is **two stacks glued at the file system**:

1. **`src/`** — Pure-Python research pipeline (data → features → models → backtest). Reproducible CLI, no live anything. The original capstone deliverable.
2. **`dashboard/`** — Paper-trading dashboard layered on top. FastAPI backend (`dashboard/backend/`) + Next.js 16 frontend (`dashboard/frontend/`) + an autonomous scheduler that submits paper orders to Alpaca.

The dashboard **consumes** artifacts produced by the pipeline. Specifically, on startup `dashboard/backend/main.py` reads `results/predictions.parquet` and `results/backtest_summary.json` — if those don't exist the backend crashes on import. Run the pipeline at least once before touching the dashboard.

There are **two separate venvs**: a top-level `venv/` for the pipeline and `dashboard/backend/venv/` for the API. They have different dependency pins (notably the dashboard targets `alpaca-py`, `fastapi`, `apscheduler`); don't cross-install.

## Common commands

Pipeline (run from repo root with the top-level venv active):

```bash
python -m src.pipeline --all                # full pipeline: download → features → train → backtest
python -m src.pipeline --download-data      # refresh OHLCV cache from yfinance
python -m src.pipeline --train              # walk-forward CV + save predictions.parquet
python -m src.pipeline --backtest           # rerun backtest on existing predictions
python -m src.pipeline --retrain-final      # refresh data + train next versioned model (rf_vN)
python -m src.predict                       # generate predictions for today, append to parquet
```

Dashboard backend (`dashboard/backend/`, with its own venv):

```bash
uvicorn main:app --reload --port 8000
```

Dashboard frontend (`dashboard/frontend/`):

```bash
npm run dev      # next dev on :3000
npm run build
npm run lint
```

No test suite exists in this repo.

## Architecture — pipeline

**Universe** (`src/data_loader.py::UNIVERSE`): 10 large-caps + SPY. SPY plays a dual role — benchmark in the backtest AND its `spy_return_1d` is merged into every other ticker's row to produce the `excess_return_1d` feature. If you change the universe you'll likely break this merge.

**Target** is **cross-sectional**, not absolute (`src/features.py::add_target`): on each date, rank universe by 5-day forward return; positive class = top 20%. This was chosen over absolute return prediction because EDA showed daily noise dominates absolute signal. Keep this in mind — the model is a ranker disguised as a binary classifier.

**Feature engineering** uses per-ticker `groupby` for all rolling computations to avoid cross-ticker leakage. `build_features` drops warmup (first ~60 rows/ticker) and target-NaN (last 5 rows/ticker). The inference path (`src/predict.py::_build_features_for_inference`) deliberately bypasses the trailing dropna so we can predict on today's row even though forward return is unknown.

**Validation must be walk-forward** (`TimeSeriesSplit`). Never k-fold on this data — it leaks future into training and inflates AUC by ~5–10 points. Three models compete (LR / RandomForest / LightGBM) in `models.py::compare_models`; best-AUC predictions land in `results/predictions.parquet`.

**Backtest invariant**: trades are **non-overlapping** (`backtest.py::simulate_strategy` takes every Nth row where N = `HOLDING_DAYS`). The naïve overlapping version was an earlier bug that double-counted overlapping forward-return windows and tripled apparent returns. Two benchmarks: SPY (market) and equal-weighted universe (apples-to-apples for the selection edge).

**Risk-managed variant (v2)**: `src/backtest_v2.py` runs the same picks through `src/risk.py::size_basket`, which adds vol-targeted sizing (target 15% annualized, max 0.40 weight, gross cap 1.0), a regime gate (half-size when SPY < 200dMA **AND** VIX > 75th percentile rolling-2y), and a 60-day correlation filter that drops pick #2 if its trailing correlation with pick #1 exceeds 0.80. VIX comes from `src/data_loader.py::load_vix` (yfinance `^VIX`, cached at `data/raw/vix.parquet`). v2 saves to `results/backtest_v2_riskmanaged.json`, `results/backtest_v2_diagnostics.json`, `results/v2_trades.parquet`, and the overlay `results/06b_equity_curve_v2.png` — never touches v1 artifacts. The acceptance bar is "MaxDD ≥ 10pp better AND Sharpe within ±0.3 of v1."

> `src/backtest.py` currently has duplicate `simulate_strategy` / `plot_equity_curves` / CLI blocks left over from a refactor — the first definitions are the live ones; the trailing dupes are dead code that gets shadowed when the module loads. Don't be alarmed.

## Architecture — production model lifecycle

- Models live in `models/` as `rf_v1.joblib`, `rf_v2.joblib`, …
- `PRODUCTION_MODEL_NAME` in `src/models.py` is **hard-coded to `rf_v1`** — `load_model()` always loads that file. Promotion is **manual**: `cp models/rf_vN.joblib models/rf_v1.joblib`. There is no auto-promotion, by design.
- `--retrain-final` trains `rf_v{next}.joblib` and runs a side-by-side walk-forward comparison vs. the current `rf_v1`. It prints a `PROMOTE / KEEP CURRENT / INVESTIGATE` recommendation but never moves the file.
- Each joblib payload is a dict with `model`, `feature_columns`, `model_version`, `trained_at` — load via `load_model()`, never raw `joblib.load`.

## Architecture — dashboard

**Backend** (`dashboard/backend/main.py`): FastAPI app exposing capstone artifacts (`/api/predictions/*`, `/api/equity-curve`, `/api/equity-curve/v2`, `/api/summary`, `/api/summary/v2`, `/api/risk/today`), Alpaca paper-trading endpoints (`/api/account`, `/api/orders/*`, `/api/live-price/*`, `/api/historical-bars/*`), news (`/api/news/*`), Claude-powered analysis (`/api/intelligence/*`), heatmap (`/api/heatmap/*`), trade journal (`/api/journal/*`), and auto-trader control (`/api/autotrader/*`). Reads `.env` from `dashboard/backend/.env`: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ANTHROPIC_API_KEY`, `AUTO_TRADER_ENABLED`, `AUTO_STREAM_ENABLED`.

**Live risk sizing**: `dashboard/backend/position_manager.py::build_risk_context` pulls ~210 days of Alpaca daily bars + the cached VIX parquet, then `select_signals_to_trade` calls `src.risk.size_basket` to assign per-pick weights. The auto-scheduler forwards each weight to `calculate_position_size(buying_power, ref_price, weight)`. Falls back to flat 20% if the risk module can't be imported or Alpaca bars fail. The Decision Card on the main dashboard reads `/api/risk/today` which exposes regime + weights + recommended notional/shares as a single payload.

**Live data**: `stream_manager.py` runs an Alpaca `StockDataStream` on a daemon thread, buffering bars in `deque(maxlen=500)` per ticker. Initialized on FastAPI startup if creds are present. Frontend polls REST endpoints rather than connecting to the websocket directly — see `/api/historical-bars/{ticker}` (intraday timeframes supported: `1Min/5Min/15Min/30Min/1H/1D/1W`).

> The websocket thread is **incompatible with `uvicorn --reload`** — the daemon thread keeps the port bound across reloads and the worker fails to come back up. Run `AUTO_STREAM_ENABLED=false uvicorn main:app --reload --port 8000` for reload-friendly dev. The env var defaults to `true` so production behavior is unchanged.

**Auto-trader**: Opt-in via `AUTO_TRADER_ENABLED=true`. `auto_scheduler.py` runs entry cycle (weekdays 9:35 ET) and exit cycle (15:55 ET) via APScheduler. State lives in SQLite at `dashboard/backend/auto_trader.db` (gitignored) — schema in `auto_trader_db.py`. Hard safety caps: `MAX_ORDERS_PER_RUN=5`, `MAX_EXITS_PER_RUN=10`, `MAX_QTY_PER_ORDER=100`, `MAX_NOTIONAL_PER_ORDER=10_000`. These are intentionally not configurable from outside the modules.

**Trading is paper-only** (`TradingClient(..., paper=True)` in `main.py`). Do not change that flag without explicit instruction.

**Frontend**: Next.js **16** (React 19, Tailwind v4). `dashboard/frontend/AGENTS.md` is loaded by `CLAUDE.md` there and warns: this is not the Next.js you know — APIs and conventions differ from older versions. Before writing any frontend code, consult `dashboard/frontend/node_modules/next/dist/docs/` rather than relying on training data. The whole app is essentially one massive `app/page.tsx` (~87KB) plus three components under `app/components/`.

## Conventions

- **No heredoc-style Python patches.** Apply changes with the `Edit` tool's exact-string replacement (or rewrite the file with `Write`). Generating `cat <<'EOF' > file.py` blobs has bitten us — quoting/indent drift ends up corrupting working code. Learned the hard way.
- **Test in isolation before integration.** Exercise the pipeline module or backend endpoint directly (CLI, `python -c`, curl) before wiring it into the dashboard or scheduler. Don't debug through three layers of UI.
- **Commit early, commit often.** Small green commits over big speculative ones — the backtest invariant (non-overlapping trades) was a regression that would have been caught by a smaller commit cadence.

## Known bugs / quirks to be aware of

- **`src/data_loader.py::load_data` indentation bug** (latent crash): `DATA_DIR.mkdir(...)` and `cache_path = DATA_DIR / "ohlcv.parquet"` are nested inside the `if tickers is None:` block (lines 110–113). Calling `load_data(tickers=[...])` with an explicit list hits `NameError: name 'cache_path' is not defined` on line 115. Currently masked because every caller in the codebase relies on the default. Fix by dedenting those two lines out of the `if` block.
- **`src/models.py::make_models()` LightGBM kwarg typo**: `rum_leaves=31` should be `num_leaves=31`. LightGBM silently accepts unknown kwargs, so the model has been training with the default `num_leaves` all along. Be deliberate about fixing — it will change `rf_v1` reproducibility and the historical AUC≈0.594.
- `src/data_loader.py` defines `DATA_DIR` as `Data/raw` (capital D) but `.gitignore` ignores `data/raw/` (lowercase). Fine on case-insensitive macOS; not on Linux.
- `src/backtest.py` contains duplicate trailing definitions of `simulate_strategy`, `plot_equity_curves`, and the CLI block — dead code left over from a refactor, but visually confusing when scrolling.
