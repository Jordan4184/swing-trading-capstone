"""
FastAPI backend for the swing-trading dashboard.

Loads precomputed results from the capstone (predictions, backtest summary)
and exposes them via REST endpoints. Also integrates Alpaca for live market
data + historical bars (paper trading account).

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

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

# ---------------------------------------------------------------------------
# Environment + Alpaca clients
# ---------------------------------------------------------------------------

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    print("WARNING: Alpaca credentials missing. Live data endpoints will fail.")

data_client = (
    StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    if ALPACA_API_KEY and ALPACA_API_SECRET
    else None
)

trading_client = (
    TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
    if ALPACA_API_KEY and ALPACA_API_SECRET
    else None
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Swing Trading Dashboard API",
    description="Backend for the ML-based cross-sectional swing trading dashboard",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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

# ---------------------------------------------------------------------------
# Capstone data loading
# ---------------------------------------------------------------------------

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


PREDICTIONS = _load_predictions()
BACKTEST = _load_backtest_summary()
UNIVERSE = sorted(PREDICTIONS["ticker"].unique().tolist())

# ---------------------------------------------------------------------------
# Endpoints — capstone data
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "swing-trading-dashboard-api",
        "status": "ok",
        "alpaca_connected": data_client is not None,
        "endpoints": [
            "/api/summary",
            "/api/predictions/latest",
            "/api/predictions/{ticker}",
            "/api/tickers",
            "/api/equity-curve",
            "/api/live-price/{ticker}",
            "/api/live-prices",
            "/api/historical-bars/{ticker}",
            "/api/account",
        ],
    }


@app.get("/api/summary")
def get_summary():
    return BACKTEST


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

    return {
        "date": latest_date.strftime("%Y-%m-%d"),
        "predictions": latest.assign(
            date=latest["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }


@app.get("/api/predictions/{ticker}")
def get_ticker_predictions(ticker: str):
    ticker = ticker.upper()
    sub = PREDICTIONS[PREDICTIONS["ticker"] == ticker].copy()
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found.",
        )
    sub = sub.sort_values("date")
    return {
        "ticker": ticker,
        "n_predictions": len(sub),
        "data": sub.assign(
            date=sub["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
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
        "config": {
            "top_n": top_n,
            "holding_days": holding_days,
            "cost_bps": cost_bps,
            "initial_capital": 10000,
        },
        "n_trades": len(trades),
        "data": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "equity": round(float(row["equity"]), 2),
                "return": round(float(row["return_net"]), 6),
            }
            for _, row in trades.iterrows()
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

        return {
            "count": len(UNIVERSE),
            "quotes": quotes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca request failed: {str(e)}")


@app.get("/api/historical-bars/{ticker}")
def get_historical_bars(ticker: str, days: int = 90):
    """
    Fetch daily OHLC bars for a ticker over the past N days.
    Returns data formatted for charting.
    """
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")

    ticker = ticker.upper()
    days = max(1, min(days, 365))  # clamp 1-365

    try:
        # Alpaca free tier requires data >= 15 min old, so we use a small offset
        end = datetime.now() - timedelta(minutes=20)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        response = data_client.get_stock_bars(request)

        if ticker not in response.data:
            return {"ticker": ticker, "n_bars": 0, "data": []}

        bars = response.data[ticker]
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
