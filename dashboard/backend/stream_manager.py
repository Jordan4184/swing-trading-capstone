"""
Real-time market data streaming via Alpaca websocket.

Maintains an in-memory buffer of the latest bars + most recent trades per ticker.
"""

import logging
from collections import deque
from typing import Optional
from dataclasses import dataclass

from alpaca.data.live import StockDataStream

logger = logging.getLogger(__name__)


@dataclass
class LiveBar:
    ticker: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class LiveTrade:
    ticker: str
    timestamp: str
    price: float
    size: int


class StreamManager:
    def __init__(self, api_key, api_secret, tickers, feed="iex"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.tickers = list(tickers)
        self.feed = feed
        self.bars = {t: deque(maxlen=500) for t in self.tickers}
        self.latest_trades = {}
        self._stream = None
        self._running = False

    async def _on_bar(self, bar):
        try:
            t = bar.symbol
            ts = bar.timestamp.isoformat() if hasattr(bar.timestamp, "isoformat") else str(bar.timestamp)
            lb = LiveBar(t, ts, float(bar.open), float(bar.high), float(bar.low), float(bar.close), int(bar.volume))
            self.bars[t].append(lb)
            logger.info("[stream] bar %s @ %s close=%s", t, ts, lb.close)
        except Exception as e:
            logger.error("[stream] _on_bar error: %s", e)

    async def _on_trade(self, trade):
        try:
            t = trade.symbol
            ts = trade.timestamp.isoformat() if hasattr(trade.timestamp, "isoformat") else str(trade.timestamp)
            self.latest_trades[t] = LiveTrade(t, ts, float(trade.price), int(trade.size))
        except Exception as e:
            logger.error("[stream] _on_trade error: %s", e)

    def start(self):
        if self._running:
            return
        self._stream = StockDataStream(self.api_key, self.api_secret, feed=self.feed)
        for t in self.tickers:
            self._stream.subscribe_bars(self._on_bar, t)
            self._stream.subscribe_trades(self._on_trade, t)
        self._running = True
        logger.info("[stream] starting stream for %d tickers on %s feed", len(self.tickers), self.feed)

        import threading

        def _run():
            try:
                self._stream.run()
            except Exception as e:
                logger.error("[stream] stream crashed: %s", e)
                self._running = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        logger.info("[stream] thread started")

    def stop(self):
        if self._stream and self._running:
            try:
                self._stream.stop()
            except Exception as e:
                logger.error("[stream] stop error: %s", e)
            self._running = False

    def get_latest_bars(self, ticker, n=100):
        bars = self.bars.get(ticker.upper())
        if not bars:
            return []
        recent = list(bars)[-n:]
        return [
            {"date": b.timestamp, "open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume}
            for b in recent
        ]

    def get_latest_trade(self, ticker):
        t = self.latest_trades.get(ticker.upper())
        if not t:
            return None
        return {"ticker": t.ticker, "timestamp": t.timestamp, "price": t.price, "size": t.size}

    def get_status(self):
        return {
            "running": self._running,
            "feed": self.feed,
            "tickers": self.tickers,
            "bars_buffered": {t: len(self.bars[t]) for t in self.tickers},
            "latest_trades_count": len(self.latest_trades),
        }


_stream_manager = None


def get_stream_manager():
    return _stream_manager


def init_stream_manager(api_key, api_secret, tickers, feed="iex"):
    global _stream_manager
    if _stream_manager is not None:
        return _stream_manager
    _stream_manager = StreamManager(api_key, api_secret, tickers, feed=feed)
    _stream_manager.start()
    return _stream_manager
