"""
FastAPI backend for the swing-trading dashboard.

Loads precomputed results from the capstone (predictions, backtest summary)
and exposes them via REST endpoints. Also integrates Alpaca for live market
data + historical bars + paper trading orders + news + LLM-powered analysis.

Run:
    uvicorn main:app --reload --port 8000
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import stream_manager
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

import llm_analyzer
import auto_trader_db
import position_manager
import auto_scheduler
import heatmap

# Make src.evaluate importable from the project root
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Environment + Alpaca clients
# ---------------------------------------------------------------------------

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    print("WARNING: Alpaca credentials missing. Live data endpoints will fail.")

if not ANTHROPIC_API_KEY:
    print("WARNING: Anthropic credentials missing. /api/intelligence will fail.")

data_client = (
    StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    if ALPACA_API_KEY and ALPACA_API_SECRET
    else None
)

news_client = (
    NewsClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    if ALPACA_API_KEY and ALPACA_API_SECRET
    else None
)

trading_client = (
    TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
    if ALPACA_API_KEY and ALPACA_API_SECRET
    else None
)

MAX_QTY_PER_ORDER = 100
MAX_NOTIONAL_PER_ORDER = 10_000

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Swing Trading Dashboard API", version="0.9.1")


@app.on_event("startup")
def _startup_autoscheduler():
    """Auto-start scheduler if AUTO_TRADER_ENABLED=true in env."""
    try:
        result = auto_scheduler.start_scheduler()
        if result.get("started"):
            print(f"Auto-trader scheduler: {result.get('reason')}")
        else:
            print(f"Auto-trader scheduler NOT started: {result.get('reason')}")
    except Exception as e:
        print(f"Auto-trader scheduler startup failed: {e}")

    # Initialize live websocket stream. Disable with AUTO_STREAM_ENABLED=false
    # to keep `uvicorn --reload` happy (the websocket thread breaks reloads).
    stream_enabled = os.getenv("AUTO_STREAM_ENABLED", "true").strip().lower() not in ("false", "0", "no", "off")
    if not stream_enabled:
        print("[main] Skipping stream - AUTO_STREAM_ENABLED=false")
    elif not (ALPACA_API_KEY and ALPACA_API_SECRET):
        print("[main] Skipping stream - Alpaca credentials missing")
    else:
        try:
            stream_manager.init_stream_manager(
                api_key=ALPACA_API_KEY,
                api_secret=ALPACA_API_SECRET,
                tickers=UNIVERSE,
                feed="iex",
            )
            print(f"[main] Stream manager initialized for {len(UNIVERSE)} tickers")
        except Exception as e:
            print(f"[main] Stream init failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Capstone artifact paths
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def _load_predictions() -> pd.DataFrame:
    path = RESULTS_DIR / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found at {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_backtest_summary() -> dict:
    path = RESULTS_DIR / "backtest_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Backtest summary not found at {path}")
    with open(path) as f:
        return json.load(f)


def _load_backtest_v2_summary() -> dict | None:
    """Risk-managed v2 backtest summary. None if pipeline hasn't generated it yet."""
    path = RESULTS_DIR / "backtest_v2_riskmanaged.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_v2_trades() -> pd.DataFrame | None:
    """Per-rebalance v2 trades (date, basket_return_net, gross_weight, regime, picks_csv)."""
    path = RESULTS_DIR / "v2_trades.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_ablation() -> dict | None:
    """Stacked-additive ablation of the v2 risk components. None if not yet generated."""
    path = RESULTS_DIR / "ablation_v2.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_feature_ablation() -> dict | None:
    """Absolute-vs-rank feature ablation. None if not yet generated."""
    path = RESULTS_DIR / "feature_ablation.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# Calibration buckets — match src/evaluate.py so the ribbon agrees with the
# existing calibration bar chart on the evaluation page.
CALIBRATION_BINS = [0.0, 0.45, 0.50, 0.55, 0.60, 1.01]  # 1.01 so exactly-1.0 lands in last bucket
CALIBRATION_LABELS = ["<0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", ">0.60"]


def _compute_calibration_buckets(preds: pd.DataFrame) -> list[dict]:
    """
    Bucketize historical predictions by y_proba and compute, per bucket:
      - count of predictions
      - actual_top_quintile_rate (realized "top 20% of 5d fwd return on date")
      - hit_rate_mean, hit_rate_low/high (1-sigma standard error for the rate)

    Only uses rows with non-null fwd_return_5d (live predictions are excluded
    because we don't know their outcome yet).
    """
    df = preds.dropna(subset=["fwd_return_5d"]).copy()
    if df.empty:
        return []
    df["proba_bucket"] = pd.cut(df["y_proba"], bins=CALIBRATION_BINS, labels=CALIBRATION_LABELS, include_lowest=True)
    df["actual_top_quintile"] = (
        df.groupby("date")["fwd_return_5d"].rank(pct=True) >= 0.8
    ).astype(int)

    out: list[dict] = []
    for label in CALIBRATION_LABELS:
        sub = df[df["proba_bucket"] == label]
        n = int(len(sub))
        if n == 0:
            out.append({
                "proba_bucket": label,
                "count": 0,
                "actual_top_quintile_rate": None,
                "se": None,
            })
            continue
        rate = float(sub["actual_top_quintile"].mean())
        # Standard error for a proportion: sqrt(p(1-p)/n)
        se = float(((rate * (1 - rate)) / n) ** 0.5) if n > 0 else None
        out.append({
            "proba_bucket": label,
            "count": n,
            "actual_top_quintile_rate": round(rate, 4),
            "se": round(se, 4) if se is not None else None,
        })
    return out


PREDICTIONS = _load_predictions()
BACKTEST = _load_backtest_summary()
BACKTEST_V2 = _load_backtest_v2_summary()
V2_TRADES = _load_v2_trades()
ABLATION = _load_ablation()
FEATURE_ABLATION = _load_feature_ablation()
CALIBRATION_BUCKETS = _compute_calibration_buckets(PREDICTIONS)
UNIVERSE = sorted(PREDICTIONS["ticker"].unique().tolist())

# ---------------------------------------------------------------------------
# Endpoints — root
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "swing-trading-dashboard-api",
        "status": "ok",
        "alpaca_connected": data_client is not None,
        "anthropic_connected": ANTHROPIC_API_KEY is not None,
        "trading_mode": "paper",
        "endpoints": [
            "/api/summary",
            "/api/predictions/latest",
            "/api/predictions/{ticker}",
            "/api/tickers",
            "/api/equity-curve",
            "/api/live-price/{ticker}",
            "/api/live-prices",
            "/api/historical-bars/{ticker}",
            "/api/prev-closes",
            "/api/account",
            "/api/orders/recent",
            "/api/orders/place",
            "/api/positions",
            "/api/news/recent",
            "/api/news/{ticker}",
            "/api/intelligence/{ticker}",
            "/api/intelligence/stats",
            "/api/autotrader/status",
            "/api/autotrader/trades",
            "/api/autotrader/runs",
            "/api/autotrader/entry-cycle (POST)",
            "/api/autotrader/exit-cycle (POST)",
            "/api/autotrader/predict-cycle (POST)",
            "/api/evaluation/report",
            "/api/heatmap/{ticker}?current_price=X",
            "/api/journal/comparison",
            "/api/journal/equity",
            "/api/summary/v2",
            "/api/equity-curve/v2",
            "/api/ablation",
            "/api/feature-ablation",
            "/api/calibration/buckets",
            "/api/risk/today",
        ],
    }

# ---------------------------------------------------------------------------
# Endpoints — capstone data
# ---------------------------------------------------------------------------

@app.get("/api/summary")
def get_summary():
    return BACKTEST


@app.get("/api/summary/v2")
def get_summary_v2():
    """Risk-managed v2 backtest summary (vol-target + regime gate + correlation filter)."""
    if BACKTEST_V2 is None:
        raise HTTPException(status_code=404, detail="v2 backtest not yet generated. Run `python -m src.backtest_v2`.")
    return BACKTEST_V2


@app.get("/api/calibration/buckets")
def get_calibration_buckets():
    """
    Historical resolution rate per probability bucket. Used by the
    CalibrationRibbon component next to every y_proba on the dashboard.
    Computed once at startup from predictions.parquet.

    Returns: { baseline_rate, buckets: [{ proba_bucket, count,
    actual_top_quintile_rate, se }, ...], total_n }.
    The baseline is 0.20 because the target is "top quintile per date".
    """
    return {
        "baseline_rate": 0.20,
        "bins": CALIBRATION_BINS,
        "buckets": CALIBRATION_BUCKETS,
        "total_n": sum(b["count"] for b in CALIBRATION_BUCKETS),
    }


@app.get("/api/ablation")
def get_ablation():
    """
    Stacked-additive ablation of the v2 risk components:
      v1 baseline → +vol target → +regime gate → +corr filter (= v2).
    Each row has metrics + 95% CIs + delta-vs-previous-row + delta-vs-baseline.
    """
    if ABLATION is None:
        raise HTTPException(status_code=404, detail="Ablation not yet generated. Run `python -m src.ablation`.")
    return ABLATION


@app.get("/api/feature-ablation")
def get_feature_ablation():
    """
    Absolute-features-vs-per-date-rank-features comparison on the same
    walk-forward TimeSeriesSplit. Research artifact — does NOT reflect the
    current production model, which still uses absolute features.
    """
    if FEATURE_ABLATION is None:
        raise HTTPException(status_code=404, detail="Feature ablation not yet generated. Run `python -m src.feature_ablation`.")
    return FEATURE_ABLATION


@app.get("/api/tickers")
def get_tickers():
    return {"tickers": UNIVERSE, "count": len(UNIVERSE)}


@app.get("/api/predictions/latest")
def get_latest_predictions(top_n: Optional[int] = None):
    latest_date = PREDICTIONS["date"].max()
    latest = (
        PREDICTIONS[PREDICTIONS["date"] == latest_date]
        .sort_values("y_proba", ascending=False)
        .copy()
    )
    if top_n:
        latest = latest.head(top_n)

    out = latest.assign(date=latest["date"].dt.strftime("%Y-%m-%d"))
    records = out.to_dict(orient="records")
    # Replace NaN with None in records since JSON cannot serialize NaN
    import math
    cleaned = []
    for rec in records:
        cleaned.append({
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in rec.items()
        })
    return {
        "date": latest_date.strftime("%Y-%m-%d"),
        "predictions": cleaned,
    }


@app.get("/api/predictions/{ticker}")
def get_ticker_predictions(ticker: str):
    ticker = ticker.upper()
    sub = PREDICTIONS[PREDICTIONS["ticker"] == ticker].copy()
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
    sub = sub.sort_values("date")
    sub_records = sub.assign(date=sub["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    import math
    sub_cleaned = []
    for rec in sub_records:
        sub_cleaned.append({
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in rec.items()
        })
    return {
        "ticker": ticker,
        "n_predictions": len(sub),
        "data": sub_cleaned,
    }


@app.get("/api/equity-curve")
def get_equity_curve(top_n: int = 2, holding_days: int = 5, cost_bps: float = 10.0):
    cost = cost_bps / 10000.0
    pr = PREDICTIONS.copy()
    pr["rank"] = pr.groupby("date")["y_proba"].rank(method="first", ascending=False)
    picks = pr[pr["rank"] <= top_n]
    daily = (
        picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    trades = daily.iloc[::holding_days].copy().reset_index(drop=True)
    trades["return_net"] = trades["basket_return"] - cost
    trades["equity"] = 10000 * (1 + trades["return_net"]).cumprod()

    return {
        "config": {"top_n": top_n, "holding_days": holding_days, "cost_bps": cost_bps, "initial_capital": 10000},
        "n_trades": len(trades),
        "data": [
            {"date": row["date"].strftime("%Y-%m-%d"), "equity": round(float(row["equity"]), 2), "return": round(float(row["return_net"]), 6)}
            for _, row in trades.iterrows()
        ],
    }


@app.get("/api/equity-curve/v2")
def get_equity_curve_v2(initial_capital: float = 10000.0):
    """
    Overlay v1 + v2 equity curves on the same date axis for the dashboard
    before/after panel. v1 is recomputed inline from PREDICTIONS (same logic
    as /api/equity-curve); v2 is read from results/v2_trades.parquet.
    """
    if V2_TRADES is None:
        raise HTTPException(status_code=404, detail="v2 trades not yet generated. Run `python -m src.backtest_v2`.")

    # v1: top-2, 5-day non-overlapping, 10bps cost — matches existing /api/equity-curve defaults
    v1_pr = PREDICTIONS.copy()
    v1_pr["rank"] = v1_pr.groupby("date")["y_proba"].rank(method="first", ascending=False)
    v1_picks = v1_pr[v1_pr["rank"] <= 2]
    v1_daily = (
        v1_picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    v1_trades = v1_daily.iloc[::5].copy().reset_index(drop=True)
    v1_trades["return_net"] = v1_trades["basket_return"] - 0.0020  # 20bps round-trip per leg, matches code
    v1_trades["equity"] = initial_capital * (1 + v1_trades["return_net"]).cumprod()

    # v2: load from artifact
    v2 = V2_TRADES.sort_values("date").reset_index(drop=True).copy()
    v2["equity"] = initial_capital * (1 + v2["basket_return_net"]).cumprod()

    return {
        "config": {
            "initial_capital": initial_capital,
            "v1": {"top_n": 2, "holding_days": 5, "cost_bps": 20.0, "sizing": "equal-weight"},
            "v2": {
                "top_n": 2, "holding_days": 5, "cost_bps": 20.0, "sizing": "vol-targeted",
                "target_vol": 0.15, "max_weight": 0.40, "gross_cap": 1.0,
                "regime_off_multiplier": 0.5, "corr_threshold": 0.80,
            },
        },
        "v1": [
            {"date": row["date"].strftime("%Y-%m-%d"),
             "equity": round(float(row["equity"]), 2),
             "return": round(float(row["return_net"]), 6)}
            for _, row in v1_trades.iterrows()
        ],
        "v2": [
            {"date": row["date"].strftime("%Y-%m-%d"),
             "equity": round(float(row["equity"]), 2),
             "return": round(float(row["basket_return_net"]), 6),
             "regime": row["regime"],
             "gross_weight": round(float(row["gross_weight"]), 4),
             "picks": row["picks_csv"]}
            for _, row in v2.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints — Alpaca live data
# ---------------------------------------------------------------------------

def _format_quote(ticker: str, quote) -> dict:
    bid = float(quote.bid_price) if quote.bid_price else None
    ask = float(quote.ask_price) if quote.ask_price else None
    mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
    return {
        "ticker": ticker,
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": round(mid, 2) if mid else None,
        "bid_size": int(quote.bid_size) if quote.bid_size else None,
        "ask_size": int(quote.ask_size) if quote.ask_size else None,
        "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
    }


@app.get("/api/live-price/{ticker}")
def get_live_price(ticker: str):
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")
    ticker = ticker.upper()
    try:
        request = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
        response = data_client.get_stock_latest_quote(request)
        if ticker not in response:
            raise HTTPException(status_code=404, detail=f"No quote data for '{ticker}'.")
        return _format_quote(ticker, response[ticker])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca request failed: {str(e)}")


@app.get("/api/live-prices")
def get_live_prices():
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")
    try:
        request = StockLatestQuoteRequest(symbol_or_symbols=UNIVERSE)
        response = data_client.get_stock_latest_quote(request)
        quotes = {}
        for ticker in UNIVERSE:
            if ticker in response:
                quotes[ticker] = _format_quote(ticker, response[ticker])
            else:
                quotes[ticker] = {"ticker": ticker, "bid_price": None, "ask_price": None}
        return {"count": len(UNIVERSE), "quotes": quotes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca request failed: {str(e)}")


@app.get("/api/prev-closes")
def get_prev_closes():
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")
    try:
        end = datetime.now() - timedelta(minutes=20)
        start = end - timedelta(days=7)
        request = StockBarsRequest(symbol_or_symbols=UNIVERSE, timeframe=TimeFrame.Day, start=start, end=end)
        response = data_client.get_stock_bars(request)
        prev_closes = {}
        for ticker in UNIVERSE:
            if ticker in response.data and len(response.data[ticker]) > 0:
                bars = response.data[ticker]
                latest_bar = bars[-1]
                prev_closes[ticker] = {"ticker": ticker, "prev_close": round(float(latest_bar.close), 2), "date": latest_bar.timestamp.strftime("%Y-%m-%d")}
            else:
                prev_closes[ticker] = {"ticker": ticker, "prev_close": None, "date": None}
        return {"count": len(UNIVERSE), "data": prev_closes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca prev-closes request failed: {str(e)}")


_TIMEFRAME_MAP = {
    "1Min": (TimeFrame.Minute, "intraday"),
    "5Min": (TimeFrame(5, TimeFrameUnit.Minute), "intraday"),
    "15Min": (TimeFrame(15, TimeFrameUnit.Minute), "intraday"),
    "30Min": (TimeFrame(30, TimeFrameUnit.Minute), "intraday"),
    "1H": (TimeFrame.Hour, "intraday"),
    "1D": (TimeFrame.Day, "daily"),
    "1W": (TimeFrame.Week, "daily"),
}


def _auto_timeframe_for_range(range_days: int) -> str:
    """Pick a sensible bar size given the range."""
    if range_days <= 1:
        return "5Min"
    if range_days <= 5:
        return "15Min"
    if range_days <= 30:
        return "1H"
    return "1D"


@app.get("/api/historical-bars/{ticker}")
def get_historical_bars(
    ticker: str,
    days: int = 90,
    range_days: int | None = None,
    timeframe: str | None = None,
):
    """
    Fetch OHLCV bars for a ticker.

    Args:
        ticker: e.g. NVDA
        days: legacy parameter for backward compatibility (90 default, capped at 365)
        range_days: new parameter, how many days of history to fetch
        timeframe: bar granularity. One of: 1Min, 5Min, 15Min, 30Min, 1H, 1D, 1W.
                   If omitted, chosen automatically based on range_days.
    """
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")

    ticker = ticker.upper()

    # Range: prefer range_days, fall back to days for backward compatibility
    effective_range = range_days if range_days is not None else days
    effective_range = max(1, min(effective_range, 365))

    # Timeframe: auto-pick if not provided
    if timeframe is None:
        timeframe = _auto_timeframe_for_range(effective_range)
    if timeframe not in _TIMEFRAME_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}. Valid: {list(_TIMEFRAME_MAP.keys())}")

    tf_obj, tf_kind = _TIMEFRAME_MAP[timeframe]

    try:
        end = datetime.now() - timedelta(minutes=20)
        start = end - timedelta(days=effective_range)
        request = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=tf_obj, start=start, end=end)
        response = data_client.get_stock_bars(request)
        if ticker not in response.data:
            return {"ticker": ticker, "n_bars": 0, "timeframe": timeframe, "range_days": effective_range, "kind": tf_kind, "data": []}
        bars = response.data[ticker]

        # For intraday, include full ISO timestamp; for daily/weekly, just date
        if tf_kind == "intraday":
            formatted = [
                {
                    "date": b.timestamp.isoformat(),
                    "open": round(float(b.open), 4),
                    "high": round(float(b.high), 4),
                    "low": round(float(b.low), 4),
                    "close": round(float(b.close), 4),
                    "volume": int(b.volume),
                }
                for b in bars
            ]
        else:
            formatted = [
                {
                    "date": b.timestamp.strftime("%Y-%m-%d"),
                    "open": round(float(b.open), 2),
                    "high": round(float(b.high), 2),
                    "low": round(float(b.low), 2),
                    "close": round(float(b.close), 2),
                    "volume": int(b.volume),
                }
                for b in bars
            ]
        return {
            "ticker": ticker,
            "n_bars": len(formatted),
            "timeframe": timeframe,
            "range_days": effective_range,
            "kind": tf_kind,
            "data": formatted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca bars request failed: {str(e)}")


@app.get("/api/account")
def get_account():
    if trading_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")
    try:
        account = trading_client.get_account()
        return {
            "account_status": account.status,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "equity": float(account.equity),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "currency": account.currency,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca account request failed: {str(e)}")

# ---------------------------------------------------------------------------
# Endpoints — News feed
# ---------------------------------------------------------------------------

def _format_news_article(article) -> dict:
    return {
        "id": str(article.id) if hasattr(article, "id") and article.id else None,
        "headline": article.headline if hasattr(article, "headline") else "",
        "summary": article.summary if hasattr(article, "summary") and article.summary else "",
        "author": article.author if hasattr(article, "author") and article.author else None,
        "source": article.source if hasattr(article, "source") and article.source else "",
        "url": article.url if hasattr(article, "url") and article.url else None,
        "symbols": list(article.symbols) if hasattr(article, "symbols") and article.symbols else [],
        "created_at": article.created_at.isoformat() if hasattr(article, "created_at") and article.created_at else None,
        "updated_at": article.updated_at.isoformat() if hasattr(article, "updated_at") and article.updated_at else None,
    }


@app.get("/api/news/recent")
def get_recent_news(limit: int = 20):
    if news_client is None:
        raise HTTPException(status_code=503, detail="Alpaca news client not configured.")
    limit = max(1, min(limit, 50))
    try:
        end = datetime.now()
        start = end - timedelta(days=7)
        request = NewsRequest(symbols=",".join(UNIVERSE), start=start, end=end, limit=limit, sort="desc")
        response = news_client.get_news(request)
        articles = response.data.get("news", []) if hasattr(response, "data") else []
        formatted = [_format_news_article(a) for a in articles]
        return {"count": len(formatted), "articles": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca news request failed: {str(e)}")


@app.get("/api/news/{ticker}")
def get_ticker_news(ticker: str, limit: int = 10):
    if news_client is None:
        raise HTTPException(status_code=503, detail="Alpaca news client not configured.")
    ticker = ticker.upper()
    limit = max(1, min(limit, 30))
    try:
        end = datetime.now()
        start = end - timedelta(days=14)
        request = NewsRequest(symbols=ticker, start=start, end=end, limit=limit, sort="desc")
        response = news_client.get_news(request)
        articles = response.data.get("news", []) if hasattr(response, "data") else []
        formatted = [_format_news_article(a) for a in articles]
        return {"ticker": ticker, "count": len(formatted), "articles": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca news request failed: {str(e)}")

# ---------------------------------------------------------------------------
# Endpoints — LLM Intelligence
# ---------------------------------------------------------------------------

@app.get("/api/intelligence/stats")
def get_intelligence_stats():
    """Monitoring endpoint: how many LLM calls today, cache state, etc."""
    return llm_analyzer.get_call_stats()


@app.get("/api/intelligence/{ticker}")
def get_ticker_intelligence(ticker: str, force_refresh: bool = False):
    """
    Get LLM-powered analysis for a ticker's recent news.
    Cached per (ticker, latest_article_id) to minimize LLM calls.
    """
    if news_client is None:
        raise HTTPException(status_code=503, detail="Alpaca news client not configured.")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Anthropic client not configured.")

    ticker = ticker.upper()

    # Fetch recent articles for this ticker
    try:
        end = datetime.now()
        start = end - timedelta(days=14)
        request = NewsRequest(symbols=ticker, start=start, end=end, limit=10, sort="desc")
        response = news_client.get_news(request)
        articles = response.data.get("news", []) if hasattr(response, "data") else []
        formatted_articles = [_format_news_article(a) for a in articles]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")

    if not formatted_articles:
        return {
            "ticker": ticker,
            "analysis": None,
            "metadata": {"reason": "no_articles"},
            "n_articles": 0,
        }

    # Run LLM analysis (uses cache automatically)
    result = llm_analyzer.analyze_news(ticker, formatted_articles, force_refresh=force_refresh)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="LLM analysis failed. Check backend logs and daily call limit.",
        )

    return {
        "ticker": ticker,
        "analysis": result["analysis"],
        "metadata": result["metadata"],
        "n_articles": len(formatted_articles),
    }

# ---------------------------------------------------------------------------
# Endpoints — Autonomous Trader
# ---------------------------------------------------------------------------

@app.get("/api/autotrader/status")
def autotrader_status():
    """Current state of the autonomous trader: scheduler, capacity, stats."""
    return {
        "scheduler": auto_scheduler.get_scheduler_status(),
        "portfolio": position_manager.summarize_portfolio(),
        "config": {
            "max_concurrent_positions": position_manager.MAX_CONCURRENT_POSITIONS,
            "min_signal_proba": position_manager.MIN_SIGNAL_PROBA,
            "max_position_pct": position_manager.MAX_POSITION_PCT_OF_PORTFOLIO,
            "default_holding_days": position_manager.DEFAULT_HOLDING_DAYS,
        },
    }


@app.get("/api/autotrader/trades")
def autotrader_trades(limit: int = 50):
    """Recent autonomous trader trades from the local DB."""
    return {
        "recent": auto_trader_db.get_recent_trades(limit=limit),
        "open_positions": auto_trader_db.get_open_positions(),
        "performance": auto_trader_db.get_performance_stats(),
    }


@app.get("/api/autotrader/runs")
def autotrader_runs(limit: int = 20):
    """Scheduler run history."""
    return {"runs": auto_trader_db.get_recent_runs(limit=limit)}


@app.post("/api/autotrader/entry-cycle")
def autotrader_entry(dry_run: bool = True):
    """
    Manually trigger the entry cycle.
    Defaults to dry_run=True for safety. Set dry_run=false to actually place orders.
    """
    return auto_scheduler.run_entry_cycle(dry_run=dry_run)


@app.post("/api/autotrader/exit-cycle")
def autotrader_exit(dry_run: bool = True):
    """
    Manually trigger the exit cycle.
    Defaults to dry_run=True for safety. Set dry_run=false to actually exit positions.
    """
    return auto_scheduler.run_exit_cycle(dry_run=dry_run)


# ---------------------------------------------------------------------------
# Endpoints — Heatmap (probability curve at varying prices)
# ---------------------------------------------------------------------------

@app.get("/api/heatmap/{ticker}")
def heatmap_curve(ticker: str, current_price: float, range_pct: float = 0.05):
    """
    Return the model\'s predicted y_proba across a range of hypothetical
    "today close" prices for the given ticker.

    Args:
        ticker: e.g. NVDA
        current_price: Current intraday price (centers the range)
        range_pct: How wide to span around current_price (default ±5%)

    Returns: prices[], probabilities[], current_idx, current_probability, etc.
    """
    try:
        result = heatmap.compute_heatmap(
            ticker=ticker.upper(),
            current_price=float(current_price),
            range_pct=float(range_pct),
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.post("/api/heatmap/reload")
def heatmap_reload():
    """Force reload of OHLCV cache (call after predict-cycle updates parquet)."""
    heatmap.reload_ohlcv()
    return {"reloaded": True}


@app.get("/api/heatmap-batch")
def heatmap_batch(tickers: str, prices: str, range_pct: float = 0.05):
    """
    Batch heatmap computation. One round trip for multiple tickers.

    Args:
        tickers: comma-separated, e.g. "NVDA,TSLA,META"
        prices: comma-separated, matching ticker order, e.g. "228,446,580"
        range_pct: width of price range (default 5%)

    Returns: { ticker: heatmap_result, ... }
    """
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices_list = [float(p) for p in prices.split(",") if p.strip()]

    if len(tickers_list) != len(prices_list):
        raise HTTPException(status_code=400, detail="tickers and prices must have same length")

    results = {}
    for ticker, price in zip(tickers_list, prices_list):
        try:
            results[ticker] = heatmap.compute_heatmap(
                ticker=ticker,
                current_price=price,
                range_pct=range_pct,
            )
        except Exception as e:
            results[ticker] = {"error": f"{type(e).__name__}: {str(e)}"}

    return results


# ---------------------------------------------------------------------------
# Endpoints — Live Stream
# ---------------------------------------------------------------------------
@app.get("/api/stream/status")
def stream_status():
    """Get the status of the live websocket stream."""
    mgr = stream_manager.get_stream_manager()
    if mgr is None:
        return {"running": False, "message": "Stream not initialized"}
    return mgr.get_status()


@app.get("/api/stream/bars/{ticker}")
def stream_bars(ticker: str, n: int = 100):
    """Get the latest N bars for a ticker from the live stream buffer."""
    mgr = stream_manager.get_stream_manager()
    if mgr is None:
        return {"ticker": ticker.upper(), "bars": [], "stream_running": False}
    bars = mgr.get_latest_bars(ticker.upper(), n=n)
    return {"ticker": ticker.upper(), "bars": bars, "stream_running": True, "n_bars": len(bars)}


@app.get("/api/stream/trade/{ticker}")
def stream_trade(ticker: str):
    """Get the most recent trade tick for a ticker."""
    mgr = stream_manager.get_stream_manager()
    if mgr is None:
        return {"ticker": ticker.upper(), "trade": None, "stream_running": False}
    trade = mgr.get_latest_trade(ticker.upper())
    return {"ticker": ticker.upper(), "trade": trade, "stream_running": True}


# ---------------------------------------------------------------------------
# Endpoints — Evaluation
# ---------------------------------------------------------------------------

@app.get("/api/evaluation/report")
def evaluation_report():
    """
    Run the full evaluation report:
      A. Trade-level (auto-trader closed trades)
      B. Model quality (predictions vs realized returns)
      C. Drift detection (recent vs historical predictions)

    Returns the same JSON shape as `python -m src.evaluate --json`.
    """
    try:
        # Import lazily so import errors don't kill app startup
        from src import evaluate as eval_mod
        # Force fresh evaluation each call
        report = eval_mod.run_full_report()
        return report
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"{type(e).__name__}: {str(e)}",
            "trace": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Endpoints — Risk / Decision Card
# ---------------------------------------------------------------------------

@app.get("/api/risk/today")
def risk_today(top_n: int = 5):
    """
    Single-payload feed for the Decision Card. Returns today's regime,
    risk-managed weights for the top-N predictions, recommended $ and
    share counts based on current buying power, and edge-over-universe.
    """
    if PREDICTIONS.empty:
        raise HTTPException(status_code=404, detail="No predictions loaded.")

    latest_date = PREDICTIONS["date"].max()
    today_rows = (
        PREDICTIONS[PREDICTIONS["date"] == latest_date]
        .sort_values("y_proba", ascending=False)
        .copy()
    )
    if today_rows.empty:
        raise HTTPException(status_code=404, detail="No predictions for latest date.")

    top_picks = today_rows.head(top_n)
    universe_mean_proba = float(today_rows["y_proba"].mean())
    top_proba = float(top_picks.iloc[0]["y_proba"])
    edge_over_universe = top_proba - universe_mean_proba

    # Build risk context (Alpaca bars + cached VIX). Falls back gracefully.
    risk_context = position_manager.build_risk_context(
        data_client,
        tickers=top_picks["ticker"].tolist() + ["SPY"],
    )

    # Display-only sizing: call size_basket directly, bypassing the freshness
    # and capacity filters that select_signals_to_trade applies for live trading.
    regime = "unknown"
    spy_close_val = spy_ma200_val = vix_val = vix_pct = None
    regime_mult = 1.0
    gross_weight = 0.0
    weight_by_ticker: dict[str, float] = {}
    drop_reason_by_ticker: dict[str, str | None] = {}
    realized_vol_by_ticker: dict[str, float | None] = {}
    risk_layer_state = "off"

    if risk_context is not None:
        try:
            from src.risk import size_basket
            picks = top_picks["ticker"].tolist()
            asof_ts = pd.Timestamp(latest_date)
            weights, diag = size_basket(
                picks,
                risk_context["prices_long"],
                risk_context["spy_history"],
                risk_context["vix_history"],
                asof_ts,
            )
            regime = diag.regime
            spy_close_val = diag.spy_close
            spy_ma200_val = diag.spy_ma200
            vix_val = diag.vix
            vix_pct = diag.vix_pct_rank
            regime_mult = diag.regime_multiplier
            gross_weight = diag.gross_weight
            weight_by_ticker = {t: float(w) for t, w in weights.items()}
            drop_reason_by_ticker = {p.ticker: p.dropped_reason for p in diag.picks}
            realized_vol_by_ticker = {p.ticker: p.realized_vol_20d for p in diag.picks}
            risk_layer_state = "on"
        except Exception as e:
            print(f"[risk_today] size_basket failed: {e}")

    # Recommended size in $/shares using current buying power and last close
    buying_power = None
    try:
        if trading_client is not None:
            account = trading_client.get_account()
            buying_power = float(account.buying_power)
    except Exception as e:
        print(f"[risk_today] buying_power fetch failed: {e}")

    # Latest close per ticker from the risk context's price history
    last_close: dict[str, float] = {}
    if risk_context is not None and not risk_context["prices_long"].empty:
        prices = risk_context["prices_long"]
        last = (
            prices.sort_values("date").groupby("ticker").tail(1)
            .set_index("ticker")["close"]
        )
        last_close = last.to_dict()

    picks_out = []
    for rank, (_, row) in enumerate(top_picks.iterrows(), start=1):
        t = row["ticker"]
        w = weight_by_ticker.get(t, 0.0)
        price = last_close.get(t)
        notional = (buying_power or 0.0) * w if buying_power else None
        rec_shares = int(notional // price) if (notional and price and price > 0) else None
        picks_out.append({
            "ticker": t,
            "y_proba": float(row["y_proba"]),
            "rank": rank,
            "weight_pct": round(w, 4),
            "rec_notional": round(notional, 2) if notional is not None else None,
            "rec_shares": rec_shares,
            "last_close": float(price) if price is not None else None,
            "realized_vol_20d": realized_vol_by_ticker.get(t),
            "dropped_reason": drop_reason_by_ticker.get(t),
        })

    return {
        "asof": latest_date.strftime("%Y-%m-%d"),
        "regime": regime,
        "risk_layer": risk_layer_state,
        "spy_close": spy_close_val,
        "spy_ma200": spy_ma200_val,
        "vix": vix_val,
        "vix_pct_rank": vix_pct,
        "regime_multiplier": regime_mult,
        "gross_weight": gross_weight,
        "buying_power": buying_power,
        "edge_over_universe": round(edge_over_universe, 6),
        "universe_mean_proba": round(universe_mean_proba, 6),
        "picks": picks_out,
    }


# ---------------------------------------------------------------------------
# Endpoints — Trade Journal (live vs backtest)
# ---------------------------------------------------------------------------

# Mirrors src.backtest.simulate_strategy without pulling matplotlib into the
# dashboard venv. Keep the constants in sync with src/backtest.py.
_JOURNAL_TOP_N = 2
_JOURNAL_HOLDING_DAYS = 5
_JOURNAL_ROUND_TRIP_COST = (0.0005 + 0.0005) * 2  # 10bps total


def _simulate_strategy_local(preds: pd.DataFrame, top_n: int = _JOURNAL_TOP_N) -> pd.DataFrame:
    """Non-overlapping top-N rebalance. Returns date + basket_return_net."""
    if preds.empty:
        return pd.DataFrame(columns=["date", "basket_return_5d", "basket_return_net"])
    ranked = preds.copy()
    ranked["rank"] = ranked.groupby("date")["y_proba"].rank(method="first", ascending=False)
    picks = ranked[ranked["rank"] <= top_n]
    daily_basket = (
        picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return_5d"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    rebalance = daily_basket.iloc[::_JOURNAL_HOLDING_DAYS].copy().reset_index(drop=True)
    rebalance["basket_return_net"] = rebalance["basket_return_5d"] - _JOURNAL_ROUND_TRIP_COST
    return rebalance


def _journal_collect_trades() -> list[dict]:
    """Pull all autotrader trades ordered chronologically by signal_date."""
    with auto_trader_db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY signal_date ASC, signal_rank ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _journal_status(row: dict) -> str:
    if row.get("exit_filled_at"):
        return "closed"
    if row.get("entry_filled_at"):
        return "open"
    return "pending"


@app.get("/api/journal/comparison")
def journal_comparison():
    """
    Per-trade live vs backtest comparison.

    For each autotrader trade we attach:
      - backtest_predict_pnl_pct: fwd_return_5d from predictions.parquet for
        the same (ticker, signal_date). What the historical backtester "saw"
        this pick deliver. NaN/None for live signals after the historical
        prediction window (no realized 5d forward yet).
      - backtest_basket_pnl_pct: the net basket return from re-running
        simulate_strategy() over the live window. Portfolio-level (top-N
        rebalance), matched to this trade's signal_date if it falls on a
        rebalance day.
    Plus the two corresponding gaps vs realized live_pnl_pct.
    """
    trades = _journal_collect_trades()
    if not trades:
        return {
            "rows": [],
            "n_trades": 0,
            "n_closed": 0,
            "n_open": 0,
            "n_pending": 0,
        }

    # Per-trade backtest counterfactual from predictions.parquet
    preds = PREDICTIONS.copy()
    preds["signal_date_str"] = preds["date"].dt.strftime("%Y-%m-%d")
    preds_lookup = preds.set_index(["ticker", "signal_date_str"])

    # Portfolio-level counterfactual from simulate_strategy over the live window
    go_live = trades[0]["signal_date"]
    bt_window = PREDICTIONS[PREDICTIONS["date"] >= pd.Timestamp(go_live)].copy()
    basket_by_date: dict[str, float] = {}
    if len(bt_window) > 0:
        baskets = _simulate_strategy_local(bt_window, top_n=2)
        for _, b in baskets.iterrows():
            basket_by_date[b["date"].strftime("%Y-%m-%d")] = float(b["basket_return_net"])

    enriched: list[dict] = []
    for r in trades:
        live_pnl_pct = float(r["pnl_pct"]) if r.get("pnl_pct") is not None else None

        # Predict counterfactual
        try:
            pred_row = preds_lookup.loc[(r["ticker"], r["signal_date"])]
            if isinstance(pred_row, pd.DataFrame):
                pred_row = pred_row.iloc[0]
            fwd = pred_row["fwd_return_5d"]
            predict_pnl_pct = float(fwd) if pd.notna(fwd) else None
        except KeyError:
            predict_pnl_pct = None

        # Basket counterfactual (portfolio-level)
        basket_pnl_pct = basket_by_date.get(r["signal_date"])

        gap_predict = (
            live_pnl_pct - predict_pnl_pct
            if (live_pnl_pct is not None and predict_pnl_pct is not None)
            else None
        )
        gap_basket = (
            live_pnl_pct - basket_pnl_pct
            if (live_pnl_pct is not None and basket_pnl_pct is not None)
            else None
        )

        enriched.append({
            "id": r["id"],
            "ticker": r["ticker"],
            "signal_date": r["signal_date"],
            "signal_proba": float(r["signal_proba"]),
            "signal_rank": r["signal_rank"],
            "entry_price": r.get("entry_price"),
            "entry_filled_at": r.get("entry_filled_at"),
            "exit_price": r.get("exit_price"),
            "exit_filled_at": r.get("exit_filled_at"),
            "qty": r.get("qty"),
            "live_pnl": r.get("pnl"),
            "live_pnl_pct": live_pnl_pct,
            "backtest_predict_pnl_pct": predict_pnl_pct,
            "backtest_basket_pnl_pct": basket_pnl_pct,
            "gap_vs_predict": gap_predict,
            "gap_vs_basket": gap_basket,
            "model_version": r.get("model_version"),
            "status": _journal_status(r),
        })

    n_closed = sum(1 for r in enriched if r["status"] == "closed")
    n_open = sum(1 for r in enriched if r["status"] == "open")
    n_pending = sum(1 for r in enriched if r["status"] == "pending")

    return {
        "rows": enriched,
        "n_trades": len(enriched),
        "n_closed": n_closed,
        "n_open": n_open,
        "n_pending": n_pending,
        "go_live_date": go_live,
    }


@app.get("/api/journal/equity")
def journal_equity():
    """
    Live cumulative equity curve (compounded closed-trade pnl_pct) overlaid
    on a backtest reference curve generated by simulate_strategy() over the
    same live window. Both rebased to $10K at go-live.
    """
    trades = _journal_collect_trades()
    if not trades:
        return {
            "go_live_date": None,
            "initial_capital": 10000,
            "live": [],
            "backtest": [],
            "n_live_trades_closed": 0,
            "n_backtest_baskets": 0,
        }

    go_live = trades[0]["signal_date"]
    closed = [
        r for r in trades
        if r.get("exit_filled_at") is not None and r.get("pnl_pct") is not None
    ]
    closed.sort(key=lambda r: r["exit_filled_at"])

    live_curve: list[dict] = []
    capital = 10000.0
    for t in closed:
        capital *= 1 + float(t["pnl_pct"])
        live_curve.append({
            "date": t["exit_filled_at"][:10],
            "equity": round(capital, 2),
            "ticker": t["ticker"],
            "trade_return": float(t["pnl_pct"]),
        })

    # Backtest reference curve over the live window
    bt_curve: list[dict] = []
    bt_window = PREDICTIONS[PREDICTIONS["date"] >= pd.Timestamp(go_live)].copy()
    if len(bt_window) > 0:
        baskets = _simulate_strategy_local(bt_window, top_n=2)
        cap2 = 10000.0
        for _, b in baskets.iterrows():
            ret = float(b["basket_return_net"]) if pd.notna(b["basket_return_net"]) else 0.0
            cap2 *= 1 + ret
            bt_curve.append({
                "date": b["date"].strftime("%Y-%m-%d"),
                "equity": round(cap2, 2),
                "return": ret,
            })

    return {
        "go_live_date": go_live,
        "initial_capital": 10000,
        "live": live_curve,
        "backtest": bt_curve,
        "n_live_trades_closed": len(closed),
        "n_backtest_baskets": len(bt_curve),
    }


# ---------------------------------------------------------------------------
# Endpoints — Autonomous Trader (continued)
# ---------------------------------------------------------------------------

@app.post("/api/autotrader/predict-cycle")
def autotrader_predict(timeout: int = 120):
    """
    Manually trigger the daily prediction cycle.
    Shells out to capstone venv to fetch market data, run inference,
    and append fresh predictions to results/predictions.parquet.
    """
    return auto_scheduler.run_prediction_cycle(timeout_seconds=timeout)


# ---------------------------------------------------------------------------
# Endpoints — Trading (PAPER ONLY)
# ---------------------------------------------------------------------------

class PlaceOrderRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    side: str = Field(..., description="'buy' or 'sell'")
    qty: int = Field(..., gt=0, le=MAX_QTY_PER_ORDER, description=f"Shares (1-{MAX_QTY_PER_ORDER})")

    @field_validator("ticker")
    @classmethod
    def ticker_in_universe(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in UNIVERSE:
            raise ValueError(f"Ticker '{v}' not in tradable universe: {UNIVERSE}")
        return v

    @field_validator("side")
    @classmethod
    def valid_side(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("buy", "sell"):
            raise ValueError(f"Side must be 'buy' or 'sell', got '{v}'")
        return v


@app.post("/api/orders/place")
def place_order(req: PlaceOrderRequest):
    if trading_client is None:
        raise HTTPException(status_code=503, detail="Alpaca trading not configured.")
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca data not configured.")

    try:
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=[req.ticker])
        quote_resp = data_client.get_stock_latest_quote(quote_req)
        if req.ticker not in quote_resp:
            raise HTTPException(status_code=400, detail=f"Cannot get quote for {req.ticker}")
        quote = quote_resp[req.ticker]
        ask = float(quote.ask_price) if quote.ask_price else None
        bid = float(quote.bid_price) if quote.bid_price else None
        ref_price = ask if (req.side == "buy" and ask) else (bid if bid else (ask or 0))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote lookup failed: {str(e)}")

    if ref_price <= 0:
        raise HTTPException(status_code=400, detail=f"No valid reference price for {req.ticker}. Markets may be closed.")

    estimated_notional = ref_price * req.qty
    if estimated_notional > MAX_NOTIONAL_PER_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"Estimated order value ${estimated_notional:,.2f} exceeds max ${MAX_NOTIONAL_PER_ORDER:,.0f} per order. Try fewer shares (current ref price: ${ref_price:.2f}).",
        )

    try:
        order_req = MarketOrderRequest(
            symbol=req.ticker,
            qty=req.qty,
            side=OrderSide.BUY if req.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_data=order_req)
        return {
            "status": "submitted",
            "order_id": str(order.id),
            "ticker": req.ticker,
            "side": req.side,
            "qty": req.qty,
            "estimated_price": ref_price,
            "estimated_notional": round(estimated_notional, 2),
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "alpaca_status": str(order.status),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Order submission failed: {str(e)}")


@app.get("/api/orders/recent")
def get_recent_orders(limit: int = 25):
    if trading_client is None:
        raise HTTPException(status_code=503, detail="Alpaca trading not configured.")
    limit = max(1, min(limit, 100))
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        orders = trading_client.get_orders(filter=request)
        return {
            "count": len(orders),
            "orders": [
                {
                    "order_id": str(o.id),
                    "ticker": o.symbol,
                    "side": str(o.side).lower().replace("orderside.", ""),
                    "qty": int(o.qty) if o.qty else 0,
                    "filled_qty": int(o.filled_qty) if o.filled_qty else 0,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "status": str(o.status).lower().replace("orderstatus.", ""),
                    "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                    "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                }
                for o in orders
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orders request failed: {str(e)}")


@app.get("/api/positions")
def get_positions():
    if trading_client is None:
        raise HTTPException(status_code=503, detail="Alpaca trading not configured.")
    try:
        positions = trading_client.get_all_positions()
        return {
            "count": len(positions),
            "positions": [
                {
                    "ticker": p.symbol,
                    "qty": int(p.qty) if p.qty else 0,
                    "side": str(p.side).lower().replace("positionside.", ""),
                    "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price else None,
                    "current_price": float(p.current_price) if p.current_price else None,
                    "market_value": float(p.market_value) if p.market_value else None,
                    "cost_basis": float(p.cost_basis) if p.cost_basis else None,
                    "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
                    "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
                }
                for p in positions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Positions request failed: {str(e)}")
