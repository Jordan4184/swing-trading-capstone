"use client";

import { useEffect, useState, useCallback } from "react";
import {
  ComposedChart,
  Area,
  Line,
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

const API_BASE = "http://localhost:8000";

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

// Compute a fake daily change from the mid price (we don't have prev close yet)
// Markets closed on weekends — we just show the latest known
const computeChange = (q: Quote | undefined): { abs: number | null; pct: number | null } => {
  if (!q || !q.mid_price) return { abs: null, pct: null };
  // Without prev close from API, we'll show 0 for now — placeholder
  return { abs: 0, pct: 0 };
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [latest, setLatest] = useState<LatestPredictionsResponse | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurveResponse | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [livePrices, setLivePrices] = useState<LivePricesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>("NVDA");
  const [tickerPredictions, setTickerPredictions] = useState<Prediction[]>([]);

  // Initial load
  useEffect(() => {
    async function loadInitial() {
      try {
        const [summaryRes, latestRes, equityRes, accountRes] = await Promise.all([
          fetch(`${API_BASE}/api/summary`),
          fetch(`${API_BASE}/api/predictions/latest?top_n=10`),
          fetch(`${API_BASE}/api/equity-curve`),
          fetch(`${API_BASE}/api/account`),
        ]);

        if (!summaryRes.ok) throw new Error("summary failed");
        if (!latestRes.ok) throw new Error("predictions failed");
        if (!equityRes.ok) throw new Error("equity-curve failed");
        if (!accountRes.ok) throw new Error("account failed");

        setSummary(await summaryRes.json());
        setLatest(await latestRes.json());
        setEquityCurve(await equityRes.json());
        setAccount(await accountRes.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    loadInitial();
  }, []);

  // Live prices polling — every 30s
  const fetchLivePrices = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/live-prices`);
      if (!res.ok) return;
      setLivePrices(await res.json());
    } catch {
      // silent fail on polling
    }
  }, []);

  useEffect(() => {
    fetchLivePrices();
    const interval = setInterval(fetchLivePrices, 30_000);
    return () => clearInterval(interval);
  }, [fetchLivePrices]);

  // Load ticker-specific predictions when selection changes
  useEffect(() => {
    async function loadTickerData() {
      try {
        const res = await fetch(`${API_BASE}/api/predictions/${selectedTicker}`);
        if (!res.ok) return;
        const data = await res.json();
        setTickerPredictions(data.data || []);
      } catch {}
    }
    loadTickerData();
  }, [selectedTicker]);

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

  // Selected ticker data
  const selectedQuote = livePrices?.quotes[selectedTicker];
  const selectedPrediction = latest.predictions.find((p) => p.ticker === selectedTicker);
  const selectedRank = latest.predictions.findIndex((p) => p.ticker === selectedTicker) + 1;

  // Universe ordered by signal probability (top first)
  const universeOrdered = latest.predictions
    .map((p) => p.ticker)
    .filter((t) => livePrices?.quotes[t]);

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
            const ch = computeChange(q);
            return (
              <div
                key={ticker}
                className="tape-item"
                onClick={() => setSelectedTicker(ticker)}
                style={{ cursor: "pointer" }}
              >
                <span className="tape-sym">{ticker}</span>
                <span className="tape-price">{fmtPrice(q?.mid_price)}</span>
                <span className={`tape-chg ${ch.pct && ch.pct >= 0 ? "up" : "dn"}`}>
                  {ch.pct == null ? "—" : fmtPct(ch.pct)}
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
          <div className="toolbar-section" style={{ marginLeft: "auto" }}>
            <span className="toolbar-text">
              Selected: <strong>{selectedTicker}</strong>
            </span>
          </div>
        </div>

        {/* 2x2 chart grid */}
        <div className="chart-grid">
          <ChartCell
            symbol={selectedTicker}
            name={`Selected · 1D`}
            quote={selectedQuote}
            placeholder
          />
          <ChartCell
            symbol="SPY"
            name="S&P 500 ETF · 1D"
            quote={livePrices?.quotes["SPY"]}
            placeholder
          />
          <EquityChartCell equityCurve={equityCurve} />
          <SignalsTable predictions={latest.predictions} onSelect={setSelectedTicker} selectedTicker={selectedTicker} />
        </div>

        {/* Bottom row */}
        <div className="bottom-row">
          <BottomCell title="Latest Signals" meta={`As of ${latest.date}`}>
            <div style={{ overflow: "auto", maxHeight: "100%" }}>
              {latest.predictions.slice(0, 8).map((p, idx) => (
                <SignalRow
                  key={p.ticker}
                  rank={idx + 1}
                  prediction={p}
                  quote={livePrices?.quotes[p.ticker]}
                  selected={p.ticker === selectedTicker}
                  onClick={() => setSelectedTicker(p.ticker)}
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

          <BottomCell title="Activity Feed" meta="signals + events">
            <div style={{ overflow: "auto", maxHeight: "100%" }}>
              {latest.predictions.slice(0, 5).map((p) => (
                <ActivityRow
                  key={p.ticker}
                  time={latest.date.slice(5)}
                  type="signal"
                  desc={`${p.ticker} signal: prob ${p.y_proba.toFixed(3)}`}
                  meta={p.y_proba >= 0.55 ? "BUY" : p.y_proba >= 0.50 ? "WATCH" : "HOLD"}
                />
              ))}
              <ActivityRow time="—" type="news" desc="News feed not yet wired" meta="placeholder" />
            </div>
          </BottomCell>
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
              <div className="td-change" style={{ color: "var(--text-muted)", fontSize: 11 }}>
                last: {selectedQuote?.timestamp ? new Date(selectedQuote.timestamp).toLocaleString() : "—"}
              </div>
            </div>
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
          {/* Quote stats */}
          <div className="detail-section">
            <div className="quote-stats">
              <QSCell label="Bid" value={fmtPrice(selectedQuote?.bid_price)} />
              <QSCell label="Ask" value={fmtPrice(selectedQuote?.ask_price)} />
              <QSCell label="Mid" value={fmtPrice(selectedQuote?.mid_price)} />
              <QSCell label="Spread" value={
                selectedQuote?.bid_price && selectedQuote?.ask_price
                  ? `$${(selectedQuote.ask_price - selectedQuote.bid_price).toFixed(2)}`
                  : "—"
              } />
              <QSCell label="Bid Size" value={selectedQuote?.bid_size?.toString() ?? "—"} />
              <QSCell label="Ask Size" value={selectedQuote?.ask_size?.toString() ?? "—"} />
            </div>
          </div>

          {/* ML Signal Detail */}
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

          {/* Prediction history */}
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

          {/* Placeholder sections */}
          <div className="detail-section">
            <div className="detail-section-header">
              <span>Options Chain</span>
              <span className="meta">not yet wired</span>
            </div>
            <div className="detail-section-body" style={{ color: "var(--text-faint)", fontSize: 10 }}>
              Requires options data API. Future session.
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-header">
              <span>Earnings</span>
              <span className="meta">not yet wired</span>
            </div>
            <div className="detail-section-body" style={{ color: "var(--text-faint)", fontSize: 10 }}>
              Requires fundamentals API. Future session.
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-header">
              <span>Analyst Ratings</span>
              <span className="meta">not yet wired</span>
            </div>
            <div className="detail-section-body" style={{ color: "var(--text-faint)", fontSize: 10 }}>
              Requires analyst data API. Future session.
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
        <div className="status-item">
          POLLING: {livePrices ? "30s" : "—"}
        </div>
        <div className="status-item">
          UNIVERSE: 11 tickers
        </div>
        <div className="status-item" style={{ marginLeft: "auto" }}>
          v0.3 · paper account · {account?.account_status ?? "—"}
        </div>
      </footer>

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

function ChartCell({ symbol, name, quote, placeholder }: { symbol: string; name: string; quote?: Quote; placeholder?: boolean }) {
  return (
    <div style={{ background: "var(--bg-base)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "5px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
          <span style={{ fontSize: 12, fontWeight: 700 }}>{symbol}</span>
          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>{name}</span>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
          <span style={{ fontSize: 12, fontWeight: 700 }}>{fmtPrice(quote?.mid_price)}</span>
        </div>
      </div>
      <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-faint)", fontSize: 11 }}>
        {placeholder ? "Chart data — wire historical bars endpoint next" : ""}
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
          <span style={{ fontSize: 12, fontWeight: 700 }} className="up">
            ${(equityCurve.data[equityCurve.data.length - 1]?.equity ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={equityCurve.data} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
            <defs>
              <linearGradient id="strat-grad" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#4ADE80" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#4ADE80" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2533" />
            <XAxis
              dataKey="date"
              stroke="#6A7488"
              tick={{ fontSize: 9 }}
              tickFormatter={(d: string) => d.slice(0, 7)}
              interval={Math.floor(equityCurve.data.length / 6)}
            />
            <YAxis
              stroke="#6A7488"
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#11151D", border: "1px solid #1F2533", fontSize: 10 }}
              formatter={(v: number) => [`$${v.toLocaleString()}`, "Equity"]}
            />
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
          <div
            key={p.ticker}
            onClick={() => onSelect(p.ticker)}
            style={{
              display: "grid",
              gridTemplateColumns: "24px 60px 1fr 50px 50px",
              gap: 8,
              padding: "5px 10px",
              borderBottom: "1px solid var(--border-soft)",
              alignItems: "center",
              cursor: "pointer",
              background: p.ticker === selectedTicker ? "var(--bg-row)" : undefined,
              fontSize: 11,
            }}
          >
            <span style={{ color: "var(--text-muted)", fontSize: 9, fontWeight: 700, textAlign: "center" }}>#{idx + 1}</span>
            <span style={{ fontWeight: 600 }}>{p.ticker}</span>
            <div style={{ height: 4, background: "var(--bg-row)", borderRadius: 2 }}>
              <div style={{ height: "100%", width: `${p.y_proba * 100}%`, background: idx <= 1 ? "linear-gradient(90deg, var(--green), var(--cyan))" : "var(--text-muted)", borderRadius: 2 }}></div>
            </div>
            <span style={{ fontWeight: 600, textAlign: "right" }}>{p.y_proba.toFixed(3)}</span>
            <span style={{
              fontSize: 9,
              fontWeight: 700,
              padding: "2px 6px",
              borderRadius: 3,
              textAlign: "center",
              background: p.y_proba >= 0.55 ? "var(--green-bg)" : p.y_proba >= 0.50 ? "rgba(251,191,36,0.1)" : "var(--bg-row)",
              color: p.y_proba >= 0.55 ? "var(--green)" : p.y_proba >= 0.50 ? "var(--amber)" : "var(--text-muted)",
            }}>
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

function SignalRow({ rank, prediction, quote, selected, onClick }: { rank: number; prediction: Prediction; quote?: Quote; selected: boolean; onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        display: "grid",
        gridTemplateColumns: "24px 60px 70px 1fr 50px 50px",
        gap: 8,
        padding: "5px 10px",
        alignItems: "center",
        borderBottom: "1px solid var(--border-soft)",
        fontSize: 11,
        cursor: "pointer",
        background: selected ? "var(--bg-row)" : undefined,
      }}
    >
      <span style={{ width: 18, height: 18, borderRadius: "50%", background: rank <= 2 ? "var(--green-bg-strong)" : "var(--bg-row)", color: rank <= 2 ? "var(--green)" : "var(--text-secondary)", fontSize: 9, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center" }}>{rank}</span>
      <span style={{ fontWeight: 600 }}>{prediction.ticker}</span>
      <span style={{ fontSize: 10, color: "var(--text-muted)", textAlign: "right" }}>{fmtPrice(quote?.mid_price)}</span>
      <div style={{ height: 4, background: "var(--bg-row)", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${prediction.y_proba * 100}%`, background: rank <= 2 ? "linear-gradient(90deg, var(--green), var(--cyan))" : "var(--text-muted)", borderRadius: 2 }}></div>
      </div>
      <span style={{ fontWeight: 600, textAlign: "right" }}>{prediction.y_proba.toFixed(3)}</span>
      <span style={{
        fontSize: 9,
        fontWeight: 700,
        padding: "2px 6px",
        borderRadius: 3,
        textAlign: "center",
        background: prediction.y_proba >= 0.55 ? "var(--green-bg)" : prediction.y_proba >= 0.50 ? "rgba(251,191,36,0.1)" : "var(--bg-row)",
        color: prediction.y_proba >= 0.55 ? "var(--green)" : prediction.y_proba >= 0.50 ? "var(--amber)" : "var(--text-muted)",
      }}>
        {prediction.y_proba >= 0.55 ? "BUY" : prediction.y_proba >= 0.50 ? "WATCH" : "HOLD"}
      </span>
    </div>
  );
}

function ActivityRow({ time, type, desc, meta }: { time: string; type: "buy" | "sell" | "signal" | "news"; desc: string; meta: string }) {
  const colors: Record<string, { bg: string; fg: string }> = {
    buy: { bg: "var(--green-bg)", fg: "var(--green)" },
    sell: { bg: "var(--red-bg)", fg: "var(--red)" },
    signal: { bg: "rgba(96, 165, 250, 0.1)", fg: "var(--blue)" },
    news: { bg: "rgba(167, 139, 250, 0.1)", fg: "var(--purple)" },
  };
  return (
    <div style={{ display: "grid", gridTemplateColumns: "50px 60px 1fr auto", gap: 8, padding: "5px 10px", borderBottom: "1px solid var(--border-soft)", fontSize: 10, alignItems: "center" }}>
      <span style={{ color: "var(--text-muted)", fontFeatureSettings: "'tnum'" }}>{time}</span>
      <span style={{ fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 2, textAlign: "center", background: colors[type].bg, color: colors[type].fg }}>{type.toUpperCase()}</span>
      <span style={{ color: "var(--text-secondary)" }}>{desc}</span>
      <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{meta}</span>
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
