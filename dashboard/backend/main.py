"""
FastAPI backend for the swing-trading dashboard.

Loads precomputed results from the capstone (predictions, backtest summary)
and exposes them via REST endpoints. Also integrates Alpaca for live market
data + historical bars + paper trading orders + news feed.

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
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

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

# ---------------------------------------------------------------------------
# Trading safety constants
# ---------------------------------------------------------------------------

MAX_QTY_PER_ORDER = 100
MAX_NOTIONAL_PER_ORDER = 10_000

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Swing Trading Dashboard API",
    version="0.7.0",
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
# Endpoints — root
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "swing-trading-dashboard-api",
        "status": "ok",
        "alpaca_connected": data_client is not None,
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
        ],
    }

# ---------------------------------------------------------------------------
# Endpoints — capstone data
# ---------------------------------------------------------------------------

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
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
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


@app.get("/api/prev-closes")
def get_prev_closes():
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")

    try:
        end = datetime.now() - timedelta(minutes=20)
        start = end - timedelta(days=7)

        request = StockBarsRequest(
            symbol_or_symbols=UNIVERSE,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        response = data_client.get_stock_bars(request)

        prev_closes = {}
        for ticker in UNIVERSE:
            if ticker in response.data and len(response.data[ticker]) > 0:
                bars = response.data[ticker]
                latest_bar = bars[-1]
                prev_closes[ticker] = {
                    "ticker": ticker,
                    "prev_close": round(float(latest_bar.close), 2),
                    "date": latest_bar.timestamp.strftime("%Y-%m-%d"),
                }
            else:
                prev_closes[ticker] = {
                    "ticker": ticker,
                    "prev_close": None,
                    "date": None,
                }

        return {
            "count": len(UNIVERSE),
            "data": prev_closes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca prev-closes request failed: {str(e)}")


@app.get("/api/historical-bars/{ticker}")
def get_historical_bars(ticker: str, days: int = 90):
    if data_client is None:
        raise HTTPException(status_code=503, detail="Alpaca client not configured.")

    ticker = ticker.upper()
    days = max(1, min(days, 365))

    try:
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

# ---------------------------------------------------------------------------
# Endpoints — News feed
# ---------------------------------------------------------------------------

def _format_news_article(article) -> dict:
    """Format an Alpaca news article into a JSON-friendly dict."""
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
    """
    Get recent news for the entire universe of tickers.
    Returns articles sorted newest first.
    """
    if news_client is None:
        raise HTTPException(status_code=503, detail="Alpaca news client not configured.")

    limit = max(1, min(limit, 50))

    try:
        # Fetch news for all universe tickers from past 7 days
        end = datetime.now()
        start = end - timedelta(days=7)

        request = NewsRequest(
            symbols=",".join(UNIVERSE),
            start=start,
            end=end,
            limit=limit,
            sort="desc",
        )
        response = news_client.get_news(request)

        articles = response.data.get("news", []) if hasattr(response, "data") else []
        formatted = [_format_news_article(a) for a in articles]

        return {
            "count": len(formatted),
            "articles": formatted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca news request failed: {str(e)}")


@app.get("/api/news/{ticker}")
def get_ticker_news(ticker: str, limit: int = 10):
    """News articles for a single ticker."""
    if news_client is None:
        raise HTTPException(status_code=503, detail="Alpaca news client not configured.")

    ticker = ticker.upper()
    limit = max(1, min(limit, 30))

    try:
        end = datetime.now()
        start = end - timedelta(days=14)

        request = NewsRequest(
            symbols=ticker,
            start=start,
            end=end,
            limit=limit,
            sort="desc",
        )
        response = news_client.get_news(request)

        articles = response.data.get("news", []) if hasattr(response, "data") else []
        formatted = [_format_news_article(a) for a in articles]

        return {
            "ticker": ticker,
            "count": len(formatted),
            "articles": formatted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca news request failed: {str(e)}")

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
        raise HTTPException(
            status_code=400,
            detail=f"No valid reference price for {req.ticker}. Markets may be closed.",
        )

    estimated_notional = ref_price * req.qty
    if estimated_notional > MAX_NOTIONAL_PER_ORDER:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Estimated order value ${estimated_notional:,.2f} exceeds "
                f"max ${MAX_NOTIONAL_PER_ORDER:,.0f} per order. "
                f"Try fewer shares (current ref price: ${ref_price:.2f})."
            ),
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
