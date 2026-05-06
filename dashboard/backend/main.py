"""
FastAPI backend for the swing-trading dashboard.

Loads precomputed results from the capstone project (predictions parquet,
backtest summary JSON) and exposes them via REST endpoints for the
Next.js frontend.

Run:
    uvicorn main:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Swing Trading Dashboard API",
    description="Backend for the ML-based cross-sectional swing trading dashboard",
    version="0.1.0",
)

# CORS — allow the Next.js dev server (port 3000) to call this backend
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
# Paths to capstone artifacts
# ---------------------------------------------------------------------------

# This file lives at: dashboard/backend/main.py
# Capstone results live at: results/ (two levels up)
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading (cached in memory at startup)
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


# Load once at startup — these are small files, fine to keep in memory
PREDICTIONS = _load_predictions()
BACKTEST = _load_backtest_summary()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    """Health check / API discovery."""
    return {
        "service": "swing-trading-dashboard-api",
        "status": "ok",
        "endpoints": [
            "/api/summary",
            "/api/predictions/latest",
            "/api/predictions/{ticker}",
            "/api/tickers",
        ],
    }


@app.get("/api/summary")
def get_summary():
    """Backtest summary metrics for strategy, equal-weight, and SPY."""
    return BACKTEST


@app.get("/api/tickers")
def get_tickers():
    """List of tickers in the universe."""
    tickers = sorted(PREDICTIONS["ticker"].unique().tolist())
    return {"tickers": tickers, "count": len(tickers)}


@app.get("/api/predictions/latest")
def get_latest_predictions(top_n: Optional[int] = None):
    """
    Most recent predictions, ranked by buy probability.
    Optional ?top_n=N to limit results.
    """
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


@app.get("/api/equity-curve")
def get_equity_curve(top_n: int = 2, holding_days: int = 5, cost_bps: float = 10.0):
    """
    Compute the equity curve on the fly from predictions.

    Mirrors the non-overlapping top-N backtest from src/backtest.py.
    Defaults match the capstone's production config (TOP_N=2, HOLDING=5).
    """
    cost = cost_bps / 10000.0

    # Rank within each date, take top N
    pr = PREDICTIONS.copy()
    pr["rank"] = pr.groupby("date")["y_proba"].rank(method="first", ascending=False)
    picks = pr[pr["rank"] <= top_n]

    # Average forward-5d return per date for the basket
    daily = (
        picks.groupby("date", as_index=False)["fwd_return_5d"]
        .mean()
        .rename(columns={"fwd_return_5d": "basket_return"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Non-overlapping rebalance: take every Nth row
    trades = daily.iloc[::holding_days].copy().reset_index(drop=True)
    trades["return_net"] = trades["basket_return"] - cost
    trades["equity"] = 10000 * (1 + trades["return_net"]).cumprod()

    # Format for JSON
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


@app.get("/api/predictions/{ticker}")
def get_ticker_predictions(ticker: str):
    """All predictions for a single ticker, sorted by date."""
    ticker = ticker.upper()
    sub = PREDICTIONS[PREDICTIONS["ticker"] == ticker].copy()
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found. Try /api/tickers for valid options.",
        )
    sub = sub.sort_values("date")
    return {
        "ticker": ticker,
        "n_predictions": len(sub),
        "data": sub.assign(date=sub["date"].dt.strftime("%Y-%m-%d")).to_dict(
            orient="records"
        ),
    }
