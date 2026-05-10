"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Prediction = {
  date: string;
  ticker: string;
  y_true: number;
  y_proba: number;
  fwd_return_5d: number;
  fold: number;
};

type LatestPredictionsResponse = {
  date: string;
  predictions: Prediction[];
};

type StrategyMetrics = {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  hit_rate: number;
};

type BacktestSummary = {
  strategy: StrategyMetrics;
  equal_weight: StrategyMetrics;
  spy: StrategyMetrics;
};

type EquityPoint = { date: string; equity: number; return: number };

type EquityCurveResponse = {
  config: { top_n: number; holding_days: number; cost_bps: number; initial_capital: number };
  n_trades: number;
  data: EquityPoint[];
};

type Quote = {
  ticker: string;
  bid_price: number | null;
  ask_price: number | null;
  mid_price?: number | null;
  bid_size?: number | null;
  ask_size?: number | null;
  timestamp?: string | null;
};

type LivePricesResponse = {
  count: number;
  quotes: Record<string, Quote>;
};

type PrevClose = {
  ticker: string;
  prev_close: number | null;
  date: string | null;
};

type PrevClosesResponse = {
  count: number;
  data: Record<string, PrevClose>;
};

type Account = {
  account_status: string;
  buying_power: number;
  cash: number;
  equity: number;
  portfolio_value: number;
  pattern_day_trader: boolean;
  trading_blocked: boolean;
  currency: string;
};

type Bar = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type BarsResponse = {
  ticker: string;
  n_bars: number;
  data: Bar[];
};

type Order = {
  order_id: string;
  ticker: string;
  side: string;
  qty: number;
  filled_qty: number;
  filled_avg_price: number | null;
  status: string;
  submitted_at: string | null;
  filled_at: string | null;
};

type OrdersResponse = {
  count: number;
  orders: Order[];
};

type Position = {
  ticker: string;
  qty: number;
  side: string;
  avg_entry_price: number | null;
  current_price: number | null;
  market_value: number | null;
  cost_basis: number | null;
  unrealized_pl: number | null;
  unrealized_plpc: number | null;
};

type PositionsResponse = {
  count: number;
  positions: Position[];
};

type FlashState = "up" | "dn" | null;
type Toast = { type: "success" | "error" | "info"; message: string };
type OrderModalState = { ticker: string; side: "buy" | "sell" } | null;

const API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 5_000;
const PREV_CLOSE_REFRESH_MS = 60_000;
const ORDERS_REFRESH_MS = 10_000;
const MAX_QTY_PER_ORDER = 100;
const MAX_NOTIONAL_PER_ORDER = 10_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtMoney = (n: number) =>
  n >= 1000 ? `$${(n / 1000).toFixed(1)}k` : `$${n.toFixed(2)}`;

const fmtPct = (n: number, withSign = true) => {
  const v = (n * 100).toFixed(2);
  return withSign && n > 0 ? `+${v}%` : `${v}%`;
};

const fmtPrice = (n: number | null | undefined) =>
  n == null ? "—" : n.toFixed(2);

const computeBarChange = (bars: Bar[] | undefined): { abs: number; pct: number } | null => {
  if (!bars || bars.length < 2) return null;
  const first = bars[0].close;
  const last = bars[bars.length - 1].close;
  return { abs: last - first, pct: (last - first) / first };
};

const computeDailyChange = (
  current: number | null | undefined,
  prevClose: number | null | undefined
): { abs: number; pct: number } | null => {
  if (current == null || prevClose == null || prevClose === 0) return null;
  return {
    abs: current - prevClose,
    pct: (current - prevClose) / prevClose,
  };
};

// ---------------------------------------------------------------------------
// Custom hook: track price changes and flash
// ---------------------------------------------------------------------------

function usePriceFlash(prices: Record<string, number | null | undefined>) {
  const previousRef = useRef<Record<string, number | null>>({});
  const [flashes, setFlashes] = useState<Record<string, FlashState>>({});
  const timeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  useEffect(() => {
    const newFlashes: Record<string, FlashState> = {};
    let anyChanged = false;

    Object.entries(prices).forEach(([ticker, current]) => {
      const prev = previousRef.current[ticker];
      if (prev != null && current != null && current !== prev) {
        newFlashes[ticker] = current > prev ? "up" : "dn";
        anyChanged = true;
        if (timeoutsRef.current[ticker]) clearTimeout(timeoutsRef.current[ticker]);
        timeoutsRef.current[ticker] = setTimeout(() => {
          setFlashes((f) => ({ ...f, [ticker]: null }));
        }, 700);
      }
      previousRef.current[ticker] = current ?? null;
    });

    if (anyChanged) {
      setFlashes((f) => ({ ...f, ...newFlashes }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(prices)]);

  useEffect(() => {
    return () => {
      Object.values(timeoutsRef.current).forEach(clearTimeout);
    };
  }, []);

  return flashes;
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [latest, setLatest] = useState<LatestPredictionsResponse | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurveResponse | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [livePrices, setLivePrices] = useState<LivePricesResponse | null>(null);
  const [prevCloses, setPrevCloses] = useState<PrevClosesResponse | null>(null);
  const [orders, setOrders] = useState<OrdersResponse | null>(null);
  const [positions, setPositions] = useState<PositionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>("NVDA");
  const [tickerPredictions, setTickerPredictions] = useState<Prediction[]>([]);
  const [selectedBars, setSelectedBars] = useState<BarsResponse | null>(null);
  const [spyBars, setSpyBars] = useState<BarsResponse | null>(null);
  const [pollCount, setPollCount] = useState(0);
  const [activityTab, setActivityTab] = useState<"orders" | "positions">("orders");
  const [orderModal, setOrderModal] = useState<OrderModalState>(null);
  const [toast, setToast] = useState<Toast | null>(null);

  const priceMap: Record<string, number | null | undefined> = {};
  if (livePrices) {
    Object.entries(livePrices.quotes).forEach(([t, q]) => {
      priceMap[t] = q.mid_price;
    });
  }
  const flashes = usePriceFlash(priceMap);

  // Toast auto-dismiss
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  // Initial load
  useEffect(() => {
    async function loadInitial() {
      try {
        const [summaryRes, latestRes, equityRes, accountRes, spyBarsRes, prevClosesRes, ordersRes, positionsRes] = await Promise.all([
          fetch(`${API_BASE}/api/summary`),
          fetch(`${API_BASE}/api/predictions/latest?top_n=10`),
          fetch(`${API_BASE}/api/equity-curve`),
          fetch(`${API_BASE}/api/account`),
          fetch(`${API_BASE}/api/historical-bars/SPY?days=90`),
          fetch(`${API_BASE}/api/prev-closes`),
          fetch(`${API_BASE}/api/orders/recent`),
          fetch(`${API_BASE}/api/positions`),
        ]);

        if (!summaryRes.ok) throw new Error("summary failed");
        if (!latestRes.ok) throw new Error("predictions failed");
        if (!equityRes.ok) throw new Error("equity-curve failed");
        if (!accountRes.ok) throw new Error("account failed");

        setSummary(await summaryRes.json());
        setLatest(await latestRes.json());
        setEquityCurve(await equityRes.json());
        setAccount(await accountRes.json());
        if (spyBarsRes.ok) setSpyBars(await spyBarsRes.json());
        if (prevClosesRes.ok) setPrevCloses(await prevClosesRes.json());
        if (ordersRes.ok) setOrders(await ordersRes.json());
        if (positionsRes.ok) setPositions(await positionsRes.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    loadInitial();
  }, []);

  // Live prices polling
  const fetchLivePrices = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/live-prices`);
      if (!res.ok) return;
      setLivePrices(await res.json());
      setPollCount((c) => c + 1);
    } catch {}
  }, []);

  useEffect(() => {
    fetchLivePrices();
    const interval = setInterval(fetchLivePrices, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchLivePrices]);

  // Refresh prev closes
  const fetchPrevCloses = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/prev-closes`);
      if (!res.ok) return;
      setPrevCloses(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    const interval = setInterval(fetchPrevCloses, PREV_CLOSE_REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchPrevCloses]);

  // Refresh orders + positions periodically
  const fetchOrdersAndPositions = useCallback(async () => {
    try {
      const [ordersRes, positionsRes, accountRes] = await Promise.all([
        fetch(`${API_BASE}/api/orders/recent`),
        fetch(`${API_BASE}/api/positions`),
        fetch(`${API_BASE}/api/account`),
      ]);
      if (ordersRes.ok) setOrders(await ordersRes.json());
      if (positionsRes.ok) setPositions(await positionsRes.json());
      if (accountRes.ok) setAccount(await accountRes.json());
    } catch {}
  }, []);

  useEffect(() => {
    const interval = setInterval(fetchOrdersAndPositions, ORDERS_REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchOrdersAndPositions]);

  // Load ticker-specific data when selection changes
  useEffect(() => {
    async function loadTickerData() {
      try {
        const [predRes, barsRes] = await Promise.all([
          fetch(`${API_BASE}/api/predictions/${selectedTicker}`),
          fetch(`${API_BASE}/api/historical-bars/${selectedTicker}?days=90`),
        ]);
        if (predRes.ok) {
          const data = await predRes.json();
          setTickerPredictions(data.data || []);
        }
        if (barsRes.ok) {
          setSelectedBars(await barsRes.json());
        }
      } catch {}
    }
    loadTickerData();
  }, [selectedTicker]);

  // Order placement
  async function handlePlaceOrder(ticker: string, side: "buy" | "sell", qty: number) {
    try {
      const res = await fetch(`${API_BASE}/api/orders/place`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, side, qty }),
      });
      const data = await res.json();
      if (!res.ok) {
        setToast({ type: "error", message: data.detail || "Order failed" });
        return;
      }
      setToast({
        type: "success",
        message: `${side.toUpperCase()} ${qty} ${ticker} @ ~$${data.estimated_price.toFixed(2)} submitted`,
      });
      setOrderModal(null);
      // Refresh orders/positions immediately
      fetchOrdersAndPositions();
    } catch (e) {
      setToast({ type: "error", message: e instanceof Error ? e.message : "Network error" });
    }
  }

  if (error) {
    return (
      <main style={{ padding: 20, color: "var(--red)" }}>
        <h1>Backend connection error: {error}</h1>
        <p style={{ color: "var(--text-muted)", marginTop: 8 }}>
          Make sure FastAPI is running on port 8000
        </p>
      </main>
    );
  }

  if (!summary || !latest || !equityCurve) {
    return (
      <main style={{ padding: 20, color: "var(--text-muted)" }}>
        Loading dashboard...
      </main>
    );
  }

  const s = summary.strategy;
  const ew = summary.equal_weight;

  const selectedQuote = livePrices?.quotes[selectedTicker];
  const selectedPrediction = latest.predictions.find((p) => p.ticker === selectedTicker);
  const selectedRank = latest.predictions.findIndex((p) => p.ticker === selectedTicker) + 1;

  const universeOrdered = latest.predictions
    .map((p) => p.ticker)
    .filter((t) => livePrices?.quotes[t]);

  const selectedPrevClose = prevCloses?.data[selectedTicker]?.prev_close;
  const selectedDailyChange = computeDailyChange(selectedQuote?.mid_price, selectedPrevClose);
  const selected90dChange = computeBarChange(selectedBars?.data);
  const spyDailyChange = computeDailyChange(
    livePrices?.quotes["SPY"]?.mid_price,
    prevCloses?.data["SPY"]?.prev_close
  );

  return (
    <div className="layout">
      {/* TOP TICKER STRIP */}
      <header className="ticker-strip">
        <div className="brand-cell">
          <div className="brand-logo">C</div>
        </div>
        <div className="ticker-tape">
          {universeOrdered.map((ticker) => {
            const q = livePrices?.quotes[ticker];
            const prev = prevCloses?.data[ticker]?.prev_close;
            const dc = computeDailyChange(q?.mid_price, prev);
            const flash = flashes[ticker];
            return (
              <div
                key={ticker}
                className={`tape-item ${flash === "up" ? "flash-up" : flash === "dn" ? "flash-dn" : ""}`}
                onClick={() => setSelectedTicker(ticker)}
                style={{ cursor: "pointer" }}
              >
                <span className="tape-sym">{ticker}</span>
                <span className="tape-price">{fmtPrice(q?.mid_price)}</span>
                <span className={`tape-chg ${dc && dc.pct >= 0 ? "up" : "dn"}`}>
                  {dc ? fmtPct(dc.pct) : "—"}
                </span>
              </div>
            );
          })}
          {!livePrices && (
            <div className="tape-item">
              <span className="tape-sym" style={{ color: "var(--text-muted)" }}>
                Loading prices...
              </span>
            </div>
          )}
        </div>
        <div className="market-clock">
          <span className="clock-pill">
            <span className="pulse"></span>
            {new Date().getDay() === 0 || new Date().getDay() === 6
              ? "MARKET CLOSED"
              : "MARKET OPEN"}
          </span>
          <span>
            {new Date().toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
              hour12: false,
            })} ET
          </span>
        </div>
        <div className="acct-mini">
          <div className="acct-mini-item">
            <span className="acct-mini-label">Equity</span>
            <span className="acct-mini-value">
              {account ? `$${account.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
            </span>
          </div>
          <div className="acct-mini-item">
            <span className="acct-mini-label">BP</span>
            <span className="acct-mini-value">
              {account ? fmtMoney(account.buying_power) : "—"}
            </span>
          </div>
        </div>
      </header>

      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="nav-btn active" title="Dashboard">▦</div>
        <div className="nav-btn" title="Watchlist">★</div>
        <div className="nav-btn" title="Signals">⚡</div>
        <div className="nav-btn" title="Positions">$</div>
        <div className="nav-btn" title="Backtest">∿</div>
        <div className="nav-btn" title="Models">▤</div>
        <div className="nav-divider"></div>
        <div className="nav-btn" title="News">◎</div>
        <div className="nav-btn" title="Alerts">⏰</div>
        <div className="nav-spacer"></div>
        <div className="nav-btn" title="Settings">⚙</div>
      </aside>

      {/* MAIN AREA */}
      <main className="main">
        {/* Stat strip */}
        <div className="stat-bar">
          <StatCell label="Total Return" value={fmtPct(s.total_return)} sub={`+${((s.total_return - summary.spy.total_return) * 100).toFixed(0)}pp vs SPY`} accent="up" />
          <StatCell label="Annualized" value={fmtPct(s.annualized_return)} sub={`+${((s.annualized_return - ew.annualized_return) * 100).toFixed(2)}pp vs EW`} accent="up" />
          <StatCell label="Sharpe" value={s.sharpe_ratio.toFixed(2)} sub={`+${(s.sharpe_ratio - ew.sharpe_ratio).toFixed(2)} vs EW`} accent="up" />
          <StatCell label="Max DD" value={fmtPct(s.max_drawdown, false)} sub="2023 cluster" accent="dn" />
          <StatCell label="Hit Rate" value={`${(s.hit_rate * 100).toFixed(1)}%`} sub={`${equityCurve.n_trades} trades`} accent="muted" />
          <StatCell label="Trades" value={equityCurve.n_trades.toString()} sub="non-overlapping" accent="muted" />
          <StatCell label="Universe" value="11" sub="tickers" accent="muted" />
          <StatCell label="Buying Power" value={account ? fmtMoney(account.buying_power) : "—"} sub="paper" accent="muted" />
        </div>

        {/* Toolbar */}
        <div className="toolbar">
          <div className="toolbar-section">
            <span className="toolbar-text"><strong>Layout</strong></span>
            <button className="layout-btn">1×1</button>
            <button className="layout-btn">2×1</button>
            <button className="layout-btn active">2×2</button>
            <button className="layout-btn">3×2</button>
          </div>
          <div className="toolbar-section">
            <span className="toolbar-text">Sync:</span>
            <button className="layout-btn active">Symbol</button>
            <button className="layout-btn">Timeframe</button>
          </div>
          <div className="toolbar-section">
            <span className="toolbar-text">Polls:</span>
            <span className="toolbar-text"><strong>{pollCount}</strong></span>
            <span className="toolbar-text" style={{ color: "var(--text-faint)" }}>· every {POLL_INTERVAL_MS / 1000}s</span>
          </div>
          <div className="toolbar-section" style={{ marginLeft: "auto" }}>
            <span className="toolbar-text">
              Selected: <strong>{selectedTicker}</strong>
            </span>
          </div>
        </div>

        {/* 2x2 chart grid */}
        <div className="chart-grid">
          <PriceChart
            symbol={selectedTicker}
            name="Selected · 90D"
            quote={selectedQuote}
            bars={selectedBars?.data}
            change={selectedDailyChange}
            color="#4ADE80"
            flash={flashes[selectedTicker]}
          />
          <PriceChart
            symbol="SPY"
            name="S&P 500 · 90D"
            quote={livePrices?.quotes["SPY"]}
            bars={spyBars?.data}
            change={spyDailyChange}
            color="#60A5FA"
            flash={flashes["SPY"]}
          />
          <EquityChartCell equityCurve={equityCurve} />
          <SignalsTable predictions={latest.predictions} onSelect={setSelectedTicker} selectedTicker={selectedTicker} />
        </div>

        {/* Bottom row */}
        <div className="bottom-row">
          <BottomCell title="Latest Signals" meta={`${latest.predictions.length} · click trade`}>
            <div style={{ overflow: "auto", maxHeight: "100%" }}>
              {latest.predictions.slice(0, 8).map((p, idx) => (
                <SignalRow
                  key={p.ticker}
                  rank={idx + 1}
                  prediction={p}
                  quote={livePrices?.quotes[p.ticker]}
                  prevClose={prevCloses?.data[p.ticker]?.prev_close}
                  selected={p.ticker === selectedTicker}
                  flash={flashes[p.ticker]}
                  onClick={() => setSelectedTicker(p.ticker)}
                  onTrade={(side) => setOrderModal({ ticker: p.ticker, side })}
                />
              ))}
            </div>
          </BottomCell>

          <BottomCell title={`Level II · ${selectedTicker}`} meta="placeholder · paid tier">
            <div style={{ padding: 12, color: "var(--text-muted)", fontSize: 10, textAlign: "center" }}>
              <p style={{ marginBottom: 8 }}>Level II depth requires Alpaca paid data tier</p>
              <p style={{ color: "var(--text-faint)" }}>Currently showing best bid/ask from free tier:</p>
              <div style={{ marginTop: 12, fontSize: 11 }}>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 16px" }}>
                  <span style={{ color: "var(--green)" }}>BID</span>
                  <span>{fmtPrice(selectedQuote?.bid_price)}</span>
                  <span style={{ color: "var(--text-muted)" }}>{selectedQuote?.bid_size ?? "—"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 16px" }}>
                  <span style={{ color: "var(--red)" }}>ASK</span>
                  <span>{fmtPrice(selectedQuote?.ask_price)}</span>
                  <span style={{ color: "var(--text-muted)" }}>{selectedQuote?.ask_size ?? "—"}</span>
                </div>
              </div>
            </div>
          </BottomCell>

          {/* TRADING ACTIVITY (Orders + Positions tabs) */}
          <div style={{ background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ padding: "0 10px", background: "var(--bg-elevated)", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex" }}>
                <button
                  onClick={() => setActivityTab("orders")}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    padding: "8px 12px",
                    fontSize: 11,
                    fontWeight: 600,
                    color: activityTab === "orders" ? "var(--text-primary)" : "var(--text-muted)",
                    borderBottom: activityTab === "orders" ? "2px solid var(--green)" : "2px solid transparent",
                    fontFamily: "inherit",
                  }}
                >
                  Orders ({orders?.count ?? 0})
                </button>
                <button
                  onClick={() => setActivityTab("positions")}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    padding: "8px 12px",
                    fontSize: 11,
                    fontWeight: 600,
                    color: activityTab === "positions" ? "var(--text-primary)" : "var(--text-muted)",
                    borderBottom: activityTab === "positions" ? "2px solid var(--green)" : "2px solid transparent",
                    fontFamily: "inherit",
                  }}
                >
                  Positions ({positions?.count ?? 0})
                </button>
              </div>
              <span style={{ fontSize: 9, color: "var(--text-faint)" }}>refreshes 10s</span>
            </div>
            <div style={{ flex: 1, overflow: "auto" }}>
              {activityTab === "orders" ? (
                <OrdersList orders={orders?.orders ?? []} />
              ) : (
                <PositionsList positions={positions?.positions ?? []} />
              )}
            </div>
          </div>
        </div>
      </main>

      {/* RIGHT PANEL — Ticker Detail */}
      <aside className="right-panel">
        <div className="ticker-detail-header">
          <div className="td-top-row">
            <div className="td-symbol-block">
              <div className="td-symbol">{selectedTicker}</div>
              <div className="td-name">
                {selectedPrediction
                  ? `Rank ${selectedRank} of ${latest.predictions.length}`
                  : "—"}
              </div>
              <div className="td-tags">
                {selectedPrediction && selectedRank <= 2 && (
                  <span className="td-tag signal">★ TOP SIGNAL</span>
                )}
                <span className="td-tag">11-stock universe</span>
              </div>
            </div>
            <div className="td-price-block">
              <div className="td-price">
                {fmtPrice(selectedQuote?.mid_price)}
              </div>
              {selectedDailyChange && (
                <div className={`td-change ${selectedDailyChange.pct >= 0 ? "up" : "dn"}`} style={{ fontSize: 13 }}>
                  {fmtPct(selectedDailyChange.pct)} today
                </div>
              )}
              {selected90dChange && (
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                  90d: <span className={selected90dChange.pct >= 0 ? "up" : "dn"}>{fmtPct(selected90dChange.pct)}</span>
                </div>
              )}
            </div>
          </div>
          {/* Trade buttons in right panel header */}
          <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
            <button
              onClick={() => setOrderModal({ ticker: selectedTicker, side: "buy" })}
              style={{
                flex: 1,
                background: "var(--green)",
                color: "var(--bg-base)",
                border: "none",
                borderRadius: 4,
                padding: "6px 0",
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              BUY
            </button>
            <button
              onClick={() => setOrderModal({ ticker: selectedTicker, side: "sell" })}
              style={{
                flex: 1,
                background: "var(--red)",
                color: "var(--bg-base)",
                border: "none",
                borderRadius: 4,
                padding: "6px 0",
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              SELL
            </button>
          </div>
        </div>

        <div className="detail-tabs">
          <span className="detail-tab active">Overview</span>
          <span className="detail-tab">Predictions</span>
          <span className="detail-tab" style={{ color: "var(--text-faint)" }}>News (soon)</span>
          <span className="detail-tab" style={{ color: "var(--text-faint)" }}>Earnings (soon)</span>
          <span className="detail-tab" style={{ color: "var(--text-faint)" }}>Options (soon)</span>
        </div>

        <div className="detail-body">
          <div className="detail-section">
            <div className="quote-stats">
              <QSCell label="Bid" value={fmtPrice(selectedQuote?.bid_price)} />
              <QSCell label="Ask" value={fmtPrice(selectedQuote?.ask_price)} />
              <QSCell label="Mid" value={fmtPrice(selectedQuote?.mid_price)} />
              <QSCell label="Prev Close" value={fmtPrice(selectedPrevClose)} />
              <QSCell label="Spread" value={
                selectedQuote?.bid_price && selectedQuote?.ask_price
                  ? `$${(selectedQuote.ask_price - selectedQuote.bid_price).toFixed(2)}`
                  : "—"
              } />
              <QSCell label="Day Change" value={
                selectedDailyChange
                  ? `${selectedDailyChange.abs >= 0 ? "+" : ""}$${selectedDailyChange.abs.toFixed(2)}`
                  : "—"
              } />
              {selectedBars?.data.length ? (
                <>
                  <QSCell label="90d High" value={`$${Math.max(...selectedBars.data.map((b) => b.high)).toFixed(2)}`} />
                  <QSCell label="90d Low" value={`$${Math.min(...selectedBars.data.map((b) => b.low)).toFixed(2)}`} />
                </>
              ) : null}
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-header">
              <span>ML Signal Detail</span>
              <span className="meta">Random Forest</span>
            </div>
            <div className="detail-section-body">
              {selectedPrediction ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 6, fontSize: 11, lineHeight: 1.6 }}>
                  <span style={{ color: "var(--text-muted)" }}>Buy probability</span>
                  <span style={{ fontWeight: 600 }} className={selectedPrediction.y_proba >= 0.55 ? "up" : ""}>
                    {selectedPrediction.y_proba.toFixed(3)}
                  </span>
                  <span style={{ color: "var(--text-muted)" }}>Rank in universe</span>
                  <span style={{ fontWeight: 600 }}>{selectedRank} of {latest.predictions.length}</span>
                  <span style={{ color: "var(--text-muted)" }}>Realized 5d return</span>
                  <span style={{ fontWeight: 600 }} className={selectedPrediction.fwd_return_5d >= 0 ? "up" : "dn"}>
                    {fmtPct(selectedPrediction.fwd_return_5d)}
                  </span>
                  <span style={{ color: "var(--text-muted)" }}>Decision</span>
                  <span style={{ fontWeight: 600 }} className={selectedPrediction.y_proba >= 0.55 ? "up" : ""}>
                    {selectedPrediction.y_proba >= 0.55 ? "BUY" : selectedPrediction.y_proba >= 0.50 ? "WATCH" : "HOLD"}
                  </span>
                  <span style={{ color: "var(--text-muted)" }}>Date</span>
                  <span style={{ fontWeight: 600 }}>{selectedPrediction.date}</span>
                  <span style={{ color: "var(--text-muted)" }}>CV fold</span>
                  <span style={{ fontWeight: 600 }}>{selectedPrediction.fold}</span>
                </div>
              ) : (
                <div style={{ color: "var(--text-muted)", fontSize: 11 }}>
                  No prediction available for {selectedTicker}
                </div>
              )}
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-header">
              <span>Prediction History</span>
              <span className="meta">{tickerPredictions.length} preds</span>
            </div>
            <div className="detail-section-body" style={{ padding: 0, fontSize: 10 }}>
              {tickerPredictions.length === 0 ? (
                <div style={{ padding: 12, color: "var(--text-muted)" }}>Loading...</div>
              ) : (
                tickerPredictions.slice(-10).reverse().map((p) => (
                  <div
                    key={`${p.date}-${p.ticker}`}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr auto auto",
                      gap: 8,
                      padding: "5px 12px",
                      borderBottom: "1px solid var(--border-soft)",
                      fontFeatureSettings: "'tnum'",
                    }}
                  >
                    <span style={{ color: "var(--text-muted)" }}>{p.date}</span>
                    <span style={{ fontWeight: 600 }}>{p.y_proba.toFixed(3)}</span>
                    <span className={p.fwd_return_5d >= 0 ? "up" : "dn"}>
                      {fmtPct(p.fwd_return_5d)}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* STATUS BAR */}
      <footer className="status-bar">
        <div className="status-item">
          <span className="status-dot"></span>
          API: ALPACA · {livePrices ? "LIVE" : "CONNECTING"}
        </div>
        <div className="status-item">POLLING: {POLL_INTERVAL_MS / 1000}s · {pollCount} polls</div>
        <div className="status-item">UNIVERSE: 11 tickers</div>
        <div className="status-item">PAPER · {orders?.count ?? 0} orders · {positions?.count ?? 0} positions</div>
        <div className="status-item" style={{ marginLeft: "auto" }}>
          v0.6 · {account?.account_status ?? "—"}
        </div>
      </footer>

      {/* ORDER MODAL */}
      {orderModal && (
        <OrderModal
          ticker={orderModal.ticker}
          side={orderModal.side}
          quote={livePrices?.quotes[orderModal.ticker]}
          buyingPower={account?.buying_power}
          onClose={() => setOrderModal(null)}
          onSubmit={(qty) => handlePlaceOrder(orderModal.ticker, orderModal.side, qty)}
        />
      )}

      {/* TOAST */}
      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: 30,
            right: 30,
            background: toast.type === "success" ? "var(--green-bg-strong)" : toast.type === "error" ? "rgba(248,113,113,0.2)" : "var(--bg-elevated)",
            border: `1px solid ${toast.type === "success" ? "var(--green)" : toast.type === "error" ? "var(--red)" : "var(--border)"}`,
            color: toast.type === "success" ? "var(--green)" : toast.type === "error" ? "var(--red)" : "var(--text-primary)",
            padding: "12px 20px",
            borderRadius: 6,
            fontSize: 12,
            fontWeight: 500,
            zIndex: 1000,
            maxWidth: 400,
          }}
        >
          {toast.message}
        </div>
      )}

      <style jsx>{`
        .layout {
          display: grid;
          grid-template-columns: 44px 1fr 320px;
          grid-template-rows: 32px 1fr 20px;
          height: 100vh;
          width: 100vw;
        }
        .ticker-strip {
          grid-column: 1 / -1;
          background: var(--bg-panel);
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: stretch;
        }
        .brand-cell {
          flex: 0 0 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-right: 1px solid var(--border);
        }
        .brand-logo {
          width: 24px; height: 24px;
          border-radius: 5px;
          background: linear-gradient(135deg, var(--green), var(--cyan));
          display: flex; align-items: center; justify-content: center;
          font-weight: 700; color: var(--bg-base);
          font-size: 12px;
        }
        .ticker-tape {
          flex: 1;
          display: flex;
          align-items: center;
          overflow-x: auto;
          padding: 0 4px;
        }
        .ticker-tape::-webkit-scrollbar { display: none; }
        .tape-item {
          padding: 0 10px;
          display: flex; gap: 6px; align-items: center;
          white-space: nowrap;
          border-right: 1px solid var(--border-soft);
          font-size: 10px;
        }
        .tape-item:hover { background: var(--bg-row); }
        .tape-sym { font-weight: 600; color: var(--text-primary); }
        .tape-price { color: var(--text-secondary); font-feature-settings: "tnum"; }
        .tape-chg { font-feature-settings: "tnum"; font-weight: 500; }
        .tape-chg.up { color: var(--green); }
        .tape-chg.dn { color: var(--red); }
        .market-clock {
          padding: 0 12px;
          display: flex; gap: 10px; align-items: center;
          border-left: 1px solid var(--border);
          font-size: 10px;
          color: var(--text-secondary);
        }
        .clock-pill {
          background: var(--green-bg);
          color: var(--green);
          padding: 2px 8px;
          border-radius: 99px;
          font-size: 9px;
          font-weight: 700;
          display: flex; gap: 4px; align-items: center;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .pulse {
          width: 5px; height: 5px;
          background: var(--green);
          border-radius: 50%;
          animation: pulse-dot 2s ease-in-out infinite;
        }
        .acct-mini {
          padding: 0 12px;
          border-left: 1px solid var(--border);
          display: flex; gap: 12px; align-items: center;
          font-size: 10px;
        }
        .acct-mini-item { display: flex; gap: 6px; }
        .acct-mini-label { color: var(--text-muted); }
        .acct-mini-value { font-weight: 600; }
        .sidebar {
          grid-column: 1;
          grid-row: 2;
          background: var(--bg-panel);
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 6px 0;
          gap: 3px;
        }
        .nav-btn {
          width: 32px; height: 32px;
          border-radius: 6px;
          display: flex; align-items: center; justify-content: center;
          color: var(--text-muted);
          cursor: pointer;
          font-size: 14px;
        }
        .nav-btn:hover { background: var(--bg-row); color: var(--text-primary); }
        .nav-btn.active {
          background: var(--green-bg-strong);
          color: var(--green);
        }
        .nav-divider { height: 1px; width: 20px; background: var(--border); margin: 4px 0; }
        .nav-spacer { flex: 1; }
        .main {
          grid-column: 2;
          grid-row: 2;
          display: grid;
          grid-template-rows: auto auto 1fr 180px;
          overflow: hidden;
          background: var(--bg-base);
        }
        .stat-bar {
          background: var(--bg-panel);
          border-bottom: 1px solid var(--border);
          display: grid;
          grid-template-columns: repeat(8, 1fr);
        }
        .toolbar {
          background: var(--bg-panel);
          border-bottom: 1px solid var(--border);
          padding: 5px 8px;
          display: flex;
          gap: 10px;
          align-items: center;
          font-size: 10px;
        }
        .toolbar-section {
          display: flex; gap: 3px; align-items: center;
          padding-right: 10px;
          border-right: 1px solid var(--border);
        }
        .toolbar-section:last-child { border-right: none; }
        .toolbar-text { color: var(--text-muted); font-size: 10px; }
        .toolbar-text strong { color: var(--text-primary); }
        .layout-btn {
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          color: var(--text-secondary);
          padding: 3px 7px;
          border-radius: 3px;
          cursor: pointer;
          font-size: 10px;
          font-family: inherit;
        }
        .layout-btn:hover { background: var(--bg-row); color: var(--text-primary); }
        .layout-btn.active { background: var(--green-bg); color: var(--green); border-color: var(--green); }
        .chart-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          grid-template-rows: 1fr 1fr;
          gap: 1px;
          background: var(--border);
          overflow: hidden;
        }
        .bottom-row {
          background: var(--bg-panel);
          border-top: 1px solid var(--border);
          display: grid;
          grid-template-columns: 1.5fr 1fr 1.5fr;
          gap: 1px;
          background: var(--border);
        }
        .right-panel {
          grid-column: 3;
          grid-row: 2;
          background: var(--bg-panel);
          border-left: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }
        .ticker-detail-header {
          padding: 10px 12px;
          border-bottom: 1px solid var(--border);
          background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-panel) 100%);
        }
        .td-top-row { display: flex; justify-content: space-between; align-items: flex-start; }
        .td-symbol-block { flex: 1; }
        .td-symbol { font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
        .td-name { font-size: 10px; color: var(--text-muted); margin-top: -2px; }
        .td-tags { display: flex; gap: 3px; margin-top: 5px; flex-wrap: wrap; }
        .td-tag {
          font-size: 8px;
          padding: 2px 5px;
          border-radius: 99px;
          background: var(--bg-row);
          color: var(--text-secondary);
          font-weight: 500;
        }
        .td-tag.signal { background: var(--green-bg); color: var(--green); font-weight: 700; }
        .td-price-block { text-align: right; }
        .td-price {
          font-size: 22px;
          font-weight: 700;
          letter-spacing: -0.02em;
        }
        .td-change {
          font-size: 12px;
          font-weight: 600;
          margin-top: -2px;
        }
        .detail-tabs {
          display: flex;
          overflow-x: auto;
          border-bottom: 1px solid var(--border);
          background: var(--bg-elevated);
        }
        .detail-tabs::-webkit-scrollbar { display: none; }
        .detail-tab {
          padding: 6px 10px;
          color: var(--text-muted);
          cursor: pointer;
          font-size: 10px;
          font-weight: 500;
          border-bottom: 2px solid transparent;
          white-space: nowrap;
        }
        .detail-tab.active { color: var(--text-primary); border-bottom-color: var(--green); }
        .detail-body { flex: 1; overflow: auto; }
        .detail-section { border-bottom: 1px solid var(--border); }
        .detail-section-header {
          padding: 8px 12px;
          background: var(--bg-elevated);
          border-bottom: 1px solid var(--border);
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 10px;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 700;
        }
        .detail-section-header .meta {
          color: var(--text-faint);
          font-weight: 500;
          text-transform: none;
          letter-spacing: 0;
          font-size: 10px;
        }
        .detail-section-body { padding: 8px 12px; }
        .quote-stats {
          display: grid;
          grid-template-columns: 1fr 1fr;
        }
        .status-bar {
          grid-column: 1 / -1;
          background: var(--bg-panel);
          border-top: 1px solid var(--border);
          display: flex;
          align-items: center;
          padding: 0 12px;
          gap: 16px;
          font-size: 10px;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .status-item { display: flex; gap: 6px; align-items: center; }
        .status-dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          background: var(--green);
        }
        .up { color: var(--green); }
        .dn { color: var(--red); }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCell({ label, value, sub, accent }: { label: string; value: string; sub: string; accent: "up" | "dn" | "muted" }) {
  return (
    <div style={{ padding: "6px 10px", borderRight: "1px solid var(--border-soft)" }}>
      <div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 700, marginTop: 1, letterSpacing: "-0.01em" }} className={accent === "up" ? "up" : accent === "dn" ? "dn" : ""}>{value}</div>
      <div style={{ fontSize: 9, color: accent === "muted" ? "var(--text-muted)" : "var(--text-secondary)", marginTop: 1 }} className={accent === "up" ? "up" : accent === "dn" ? "dn" : ""}>{sub}</div>
    </div>
  );
}

function PriceChart({ symbol, name, quote, bars, change, color, flash }: { symbol: string; name: string; quote?: Quote; bars?: Bar[]; change: { abs: number; pct: number } | null; color: string; flash?: FlashState }) {
  const gradId = `grad-${symbol}`;
  const lastClose = bars && bars.length > 0 ? bars[bars.length - 1].close : null;
  const displayPrice = quote?.mid_price ?? lastClose;

  return (
    <div style={{ background: "var(--bg-base)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div className={flash === "up" ? "flash-up" : flash === "dn" ? "flash-dn" : ""} style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "5px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
          <span style={{ fontSize: 12, fontWeight: 700 }}>{symbol}</span>
          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>{name}</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
          <span style={{ fontSize: 12, fontWeight: 700 }}>{fmtPrice(displayPrice)}</span>
          {change && <span style={{ fontSize: 10, fontWeight: 600 }} className={change.pct >= 0 ? "up" : "dn"}>{fmtPct(change.pct)}</span>}
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
        {!bars || bars.length === 0 ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-faint)", fontSize: 10 }}>Loading bars...</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={bars} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={gradId} x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2533" />
              <XAxis dataKey="date" stroke="#6A7488" tick={{ fontSize: 8 }} tickFormatter={(d: string) => d.slice(5)} interval={Math.max(0, Math.floor(bars.length / 5))} />
              <YAxis stroke="#6A7488" tick={{ fontSize: 8 }} domain={["auto", "auto"]} tickFormatter={(v: number) => `$${v.toFixed(0)}`} width={40} />
              <Tooltip contentStyle={{ backgroundColor: "#11151D", border: "1px solid #1F2533", fontSize: 10 }} formatter={(v: number) => [`$${v.toFixed(2)}`, "Close"]} />
              <Area type="monotone" dataKey="close" stroke={color} strokeWidth={1.5} fill={`url(#${gradId})`} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

function EquityChartCell({ equityCurve }: { equityCurve: EquityCurveResponse }) {
  return (
    <div style={{ background: "var(--bg-base)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "5px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
          <span style={{ fontSize: 12, fontWeight: 700 }}>STRATEGY</span>
          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>ML Equity · ALL · {equityCurve.n_trades} trades</span>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
          <span style={{ fontSize: 12, fontWeight: 700 }} className="up">${(equityCurve.data[equityCurve.data.length - 1]?.equity ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={equityCurve.data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="strat-grad" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#4ADE80" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#4ADE80" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2533" />
            <XAxis dataKey="date" stroke="#6A7488" tick={{ fontSize: 9 }} tickFormatter={(d: string) => d.slice(0, 7)} interval={Math.floor(equityCurve.data.length / 6)} />
            <YAxis stroke="#6A7488" tick={{ fontSize: 9 }} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} width={40} />
            <Tooltip contentStyle={{ backgroundColor: "#11151D", border: "1px solid #1F2533", fontSize: 10 }} formatter={(v: number) => [`$${v.toLocaleString()}`, "Equity"]} />
            <Area type="monotone" dataKey="equity" stroke="#4ADE80" strokeWidth={1.5} fill="url(#strat-grad)" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SignalsTable({ predictions, onSelect, selectedTicker }: { predictions: Prediction[]; onSelect: (t: string) => void; selectedTicker: string }) {
  return (
    <div style={{ background: "var(--bg-base)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "5px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 12, fontWeight: 700 }}>Signals · {predictions.length}</div>
        <div style={{ fontSize: 9, color: "var(--text-muted)" }}>click to select</div>
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        {predictions.map((p, idx) => (
          <div key={p.ticker} onClick={() => onSelect(p.ticker)} style={{ display: "grid", gridTemplateColumns: "24px 60px 1fr 50px 50px", gap: 8, padding: "5px 10px", borderBottom: "1px solid var(--border-soft)", alignItems: "center", cursor: "pointer", background: p.ticker === selectedTicker ? "var(--bg-row)" : undefined, fontSize: 11 }}>
            <span style={{ color: "var(--text-muted)", fontSize: 9, fontWeight: 700, textAlign: "center" }}>#{idx + 1}</span>
            <span style={{ fontWeight: 600 }}>{p.ticker}</span>
            <div style={{ height: 4, background: "var(--bg-row)", borderRadius: 2 }}>
              <div style={{ height: "100%", width: `${p.y_proba * 100}%`, background: idx <= 1 ? "linear-gradient(90deg, var(--green), var(--cyan))" : "var(--text-muted)", borderRadius: 2 }}></div>
            </div>
            <span style={{ fontWeight: 600, textAlign: "right" }}>{p.y_proba.toFixed(3)}</span>
            <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 3, textAlign: "center", background: p.y_proba >= 0.55 ? "var(--green-bg)" : p.y_proba >= 0.50 ? "rgba(251,191,36,0.1)" : "var(--bg-row)", color: p.y_proba >= 0.55 ? "var(--green)" : p.y_proba >= 0.50 ? "var(--amber)" : "var(--text-muted)" }}>
              {p.y_proba >= 0.55 ? "BUY" : p.y_proba >= 0.50 ? "WATCH" : "HOLD"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BottomCell({ title, meta, children }: { title: string; meta: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ padding: "6px 10px", background: "var(--bg-elevated)", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontWeight: 600 }}>
        <span>{title}</span>
        <span style={{ color: "var(--text-muted)", fontSize: 10, fontWeight: 400 }}>{meta}</span>
      </div>
      <div style={{ flex: 1, overflow: "hidden" }}>{children}</div>
    </div>
  );
}

function SignalRow({ rank, prediction, quote, prevClose, selected, flash, onClick, onTrade }: { rank: number; prediction: Prediction; quote?: Quote; prevClose?: number | null; selected: boolean; flash?: FlashState; onClick: () => void; onTrade: (side: "buy" | "sell") => void }) {
  const dailyChange = computeDailyChange(quote?.mid_price, prevClose);
  return (
    <div onClick={onClick} className={flash === "up" ? "flash-up" : flash === "dn" ? "flash-dn" : ""} style={{ display: "grid", gridTemplateColumns: "24px 50px 60px 50px 1fr 50px auto", gap: 6, padding: "5px 10px", alignItems: "center", borderBottom: "1px solid var(--border-soft)", fontSize: 11, cursor: "pointer", background: selected ? "var(--bg-row)" : undefined }}>
      <span style={{ width: 18, height: 18, borderRadius: "50%", background: rank <= 2 ? "var(--green-bg-strong)" : "var(--bg-row)", color: rank <= 2 ? "var(--green)" : "var(--text-secondary)", fontSize: 9, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center" }}>{rank}</span>
      <span style={{ fontWeight: 600 }}>{prediction.ticker}</span>
      <span style={{ fontSize: 10, color: "var(--text-muted)", textAlign: "right" }}>{fmtPrice(quote?.mid_price)}</span>
      <span style={{ fontSize: 10, textAlign: "right", fontWeight: 500 }} className={dailyChange && dailyChange.pct >= 0 ? "up" : dailyChange ? "dn" : ""}>{dailyChange ? fmtPct(dailyChange.pct) : "—"}</span>
      <div style={{ height: 4, background: "var(--bg-row)", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${prediction.y_proba * 100}%`, background: rank <= 2 ? "linear-gradient(90deg, var(--green), var(--cyan))" : "var(--text-muted)", borderRadius: 2 }}></div>
      </div>
      <span style={{ fontWeight: 600, textAlign: "right" }}>{prediction.y_proba.toFixed(3)}</span>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onTrade(prediction.y_proba >= 0.50 ? "buy" : "sell");
        }}
        style={{
          fontSize: 9,
          fontWeight: 700,
          padding: "2px 6px",
          borderRadius: 3,
          background: prediction.y_proba >= 0.55 ? "var(--green-bg)" : prediction.y_proba >= 0.50 ? "rgba(251,191,36,0.1)" : "var(--bg-row)",
          color: prediction.y_proba >= 0.55 ? "var(--green)" : prediction.y_proba >= 0.50 ? "var(--amber)" : "var(--text-muted)",
          border: "1px solid transparent",
          cursor: "pointer",
          fontFamily: "inherit",
        }}
        onMouseOver={(e) => (e.currentTarget.style.borderColor = "currentColor")}
        onMouseOut={(e) => (e.currentTarget.style.borderColor = "transparent")}
      >
        {prediction.y_proba >= 0.55 ? "TRADE" : prediction.y_proba >= 0.50 ? "TRADE" : "TRADE"}
      </button>
    </div>
  );
}

function QSCell({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ padding: "5px 10px", borderRight: "1px solid var(--border-soft)", borderBottom: "1px solid var(--border-soft)", display: "flex", justifyContent: "space-between", fontSize: 10 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function OrdersList({ orders }: { orders: Order[] }) {
  if (orders.length === 0) {
    return (
      <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>
        No orders yet. Click TRADE on any signal to place one.
      </div>
    );
  }
  return (
    <div>
      {orders.map((o) => (
        <div
          key={o.order_id}
          style={{
            display: "grid",
            gridTemplateColumns: "auto 60px 50px 1fr auto",
            gap: 8,
            padding: "6px 10px",
            borderBottom: "1px solid var(--border-soft)",
            alignItems: "center",
            fontSize: 10,
          }}
        >
          <span
            style={{
              fontSize: 9,
              fontWeight: 700,
              padding: "2px 6px",
              borderRadius: 2,
              background: o.side === "buy" ? "var(--green-bg)" : "var(--red-bg)",
              color: o.side === "buy" ? "var(--green)" : "var(--red)",
            }}
          >
            {o.side.toUpperCase()}
          </span>
          <span style={{ fontWeight: 600, fontSize: 11 }}>{o.ticker}</span>
          <span style={{ color: "var(--text-secondary)", textAlign: "right" }}>{o.qty}sh</span>
          <span style={{ color: "var(--text-muted)", fontSize: 9 }}>
            {o.filled_avg_price ? `@ $${o.filled_avg_price.toFixed(2)}` : "queued"}
            {o.submitted_at && ` · ${new Date(o.submitted_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`}
          </span>
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              padding: "1px 5px",
              borderRadius: 2,
              background:
                o.status === "filled"
                  ? "var(--green-bg)"
                  : o.status === "canceled" || o.status === "rejected"
                  ? "var(--red-bg)"
                  : "var(--bg-row)",
              color:
                o.status === "filled"
                  ? "var(--green)"
                  : o.status === "canceled" || o.status === "rejected"
                  ? "var(--red)"
                  : "var(--text-muted)",
              textTransform: "uppercase",
            }}
          >
            {o.status}
          </span>
        </div>
      ))}
    </div>
  );
}

function PositionsList({ positions }: { positions: Position[] }) {
  if (positions.length === 0) {
    return (
      <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>
        No open positions. Orders fill at next market open.
      </div>
    );
  }
  return (
    <div>
      {positions.map((p) => (
        <div
          key={p.ticker}
          style={{
            display: "grid",
            gridTemplateColumns: "60px 50px 1fr auto",
            gap: 8,
            padding: "6px 10px",
            borderBottom: "1px solid var(--border-soft)",
            alignItems: "center",
            fontSize: 10,
          }}
        >
          <span style={{ fontWeight: 600, fontSize: 11 }}>{p.ticker}</span>
          <span style={{ color: "var(--text-secondary)", textAlign: "right" }}>{p.qty}sh</span>
          <span style={{ color: "var(--text-muted)", fontSize: 9 }}>
            avg ${p.avg_entry_price?.toFixed(2) ?? "—"} · now ${p.current_price?.toFixed(2) ?? "—"}
          </span>
          <span
            style={{ fontWeight: 600, textAlign: "right" }}
            className={p.unrealized_pl != null && p.unrealized_pl >= 0 ? "up" : "dn"}
          >
            {p.unrealized_pl != null
              ? `${p.unrealized_pl >= 0 ? "+" : ""}$${p.unrealized_pl.toFixed(2)}`
              : "—"}
            {p.unrealized_plpc != null && (
              <span style={{ fontSize: 9, marginLeft: 4 }}>
                ({fmtPct(p.unrealized_plpc)})
              </span>
            )}
          </span>
        </div>
      ))}
    </div>
  );
}

function OrderModal({
  ticker,
  side,
  quote,
  buyingPower,
  onClose,
  onSubmit,
}: {
  ticker: string;
  side: "buy" | "sell";
  quote?: Quote;
  buyingPower?: number;
  onClose: () => void;
  onSubmit: (qty: number) => void;
}) {
  const [qty, setQty] = useState(1);
  const refPrice = side === "buy" ? quote?.ask_price ?? quote?.mid_price : quote?.bid_price ?? quote?.mid_price;
  const estimatedNotional = (refPrice ?? 0) * qty;

  const exceedsQty = qty > MAX_QTY_PER_ORDER;
  const exceedsNotional = estimatedNotional > MAX_NOTIONAL_PER_ORDER;
  const exceedsBP = side === "buy" && buyingPower != null && estimatedNotional > buyingPower;
  const cannotSubmit = qty < 1 || exceedsQty || exceedsNotional || exceedsBP || !refPrice;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 999,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--bg-panel)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          padding: 0,
          width: 380,
          maxWidth: "90vw",
          fontSize: 12,
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "14px 18px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <div style={{ fontSize: 14, fontWeight: 700 }}>
              <span style={{ color: side === "buy" ? "var(--green)" : "var(--red)" }}>
                {side.toUpperCase()}
              </span>{" "}
              {ticker}
            </div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
              Paper trading · market order · day
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: 18,
            }}
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: 18 }}>
          <div
            style={{
              background: "rgba(96, 165, 250, 0.08)",
              border: "1px solid rgba(96, 165, 250, 0.2)",
              borderRadius: 4,
              padding: "8px 10px",
              fontSize: 10,
              color: "var(--text-secondary)",
              marginBottom: 16,
            }}
          >
            <strong style={{ color: "var(--blue)" }}>PAPER ACCOUNT</strong> — no real money. Orders submitted on
            weekends queue for the next market open.
          </div>

          {/* Quote info */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 6,
              marginBottom: 16,
              fontSize: 11,
            }}
          >
            <div>
              <div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Bid</div>
              <div style={{ fontWeight: 600 }}>{fmtPrice(quote?.bid_price)}</div>
            </div>
            <div>
              <div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Ask</div>
              <div style={{ fontWeight: 600 }}>{fmtPrice(quote?.ask_price)}</div>
            </div>
            <div>
              <div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Mid</div>
              <div style={{ fontWeight: 600 }}>{fmtPrice(quote?.mid_price)}</div>
            </div>
            <div>
              <div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>
                Reference {side === "buy" ? "(ask)" : "(bid)"}
              </div>
              <div style={{ fontWeight: 600 }}>{fmtPrice(refPrice)}</div>
            </div>
          </div>

          {/* Qty input */}
          <div style={{ marginBottom: 14 }}>
            <label style={{ display: "block", color: "var(--text-muted)", fontSize: 10, textTransform: "uppercase", marginBottom: 4 }}>
              Shares (1-{MAX_QTY_PER_ORDER})
            </label>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="number"
                min={1}
                max={MAX_QTY_PER_ORDER}
                value={qty}
                onChange={(e) => setQty(Math.max(1, Math.min(MAX_QTY_PER_ORDER, parseInt(e.target.value) || 1)))}
                style={{
                  flex: 1,
                  background: "var(--bg-input)",
                  border: "1px solid var(--border-strong)",
                  borderRadius: 4,
                  padding: "8px 10px",
                  color: "var(--text-primary)",
                  fontSize: 14,
                  fontFamily: "inherit",
                }}
              />
              {[1, 5, 10, 25].map((n) => (
                <button
                  key={n}
                  onClick={() => setQty(n)}
                  style={{
                    background: qty === n ? "var(--green-bg)" : "var(--bg-elevated)",
                    color: qty === n ? "var(--green)" : "var(--text-secondary)",
                    border: "1px solid var(--border)",
                    borderRadius: 3,
                    padding: "5px 9px",
                    fontSize: 10,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Estimated total */}
          <div
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "10px 12px",
              fontSize: 12,
              marginBottom: 14,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ color: "var(--text-muted)" }}>Estimated total</span>
            <span style={{ fontWeight: 700, fontSize: 16 }}>
              ${estimatedNotional.toFixed(2)}
            </span>
          </div>

          {/* Warnings */}
          {exceedsQty && (
            <div style={{ color: "var(--red)", fontSize: 10, marginBottom: 8 }}>
              ⚠ Exceeds max {MAX_QTY_PER_ORDER} shares per order
            </div>
          )}
          {exceedsNotional && (
            <div style={{ color: "var(--red)", fontSize: 10, marginBottom: 8 }}>
              ⚠ Exceeds max ${MAX_NOTIONAL_PER_ORDER.toLocaleString()} per order
            </div>
          )}
          {exceedsBP && (
            <div style={{ color: "var(--red)", fontSize: 10, marginBottom: 8 }}>
              ⚠ Exceeds buying power ${buyingPower?.toFixed(2)}
            </div>
          )}
          {!refPrice && (
            <div style={{ color: "var(--amber)", fontSize: 10, marginBottom: 8 }}>
              ⚠ No live quote. Order may be rejected.
            </div>
          )}

          {/* Buttons */}
          <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
            <button
              onClick={onClose}
              style={{
                flex: 1,
                background: "var(--bg-elevated)",
                color: "var(--text-secondary)",
                border: "1px solid var(--border)",
                borderRadius: 4,
                padding: "10px 0",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Cancel
            </button>
            <button
              onClick={() => onSubmit(qty)}
              disabled={cannotSubmit}
              style={{
                flex: 2,
                background: cannotSubmit ? "var(--bg-elevated)" : side === "buy" ? "var(--green)" : "var(--red)",
                color: cannotSubmit ? "var(--text-muted)" : "var(--bg-base)",
                border: "none",
                borderRadius: 4,
                padding: "10px 0",
                fontSize: 12,
                fontWeight: 700,
                cursor: cannotSubmit ? "not-allowed" : "pointer",
                fontFamily: "inherit",
              }}
            >
              Confirm {side.toUpperCase()} {qty} {ticker}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
