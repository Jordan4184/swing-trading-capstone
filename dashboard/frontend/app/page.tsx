"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import HeatmapStrip from "./components/HeatmapStrip";
import {
  ComposedChart,
  Brush,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Prediction = { date: string; ticker: string; y_true: number; y_proba: number; fwd_return_5d: number; fold: number };
type LatestPredictionsResponse = { date: string; predictions: Prediction[] };
type StrategyMetrics = { total_return: number; annualized_return: number; sharpe_ratio: number; max_drawdown: number; hit_rate: number };
type BacktestSummary = { strategy: StrategyMetrics; equal_weight: StrategyMetrics; spy: StrategyMetrics };
type EquityPoint = { date: string; equity: number; return: number };
type EquityCurveResponse = { config: { top_n: number; holding_days: number; cost_bps: number; initial_capital: number }; n_trades: number; data: EquityPoint[] };
type Quote = { ticker: string; bid_price: number | null; ask_price: number | null; mid_price?: number | null; bid_size?: number | null; ask_size?: number | null; timestamp?: string | null };
type LivePricesResponse = { count: number; quotes: Record<string, Quote> };
type PrevClose = { ticker: string; prev_close: number | null; date: string | null };
type PrevClosesResponse = { count: number; data: Record<string, PrevClose> };
type Account = { account_status: string; buying_power: number; cash: number; equity: number; portfolio_value: number; pattern_day_trader: boolean; trading_blocked: boolean; currency: string };
type BarRow = { date: string; open: number; high: number; low: number; close: number; volume: number };
type BarsResponse = { ticker: string; n_bars: number; data: BarRow[] };
type Order = { order_id: string; ticker: string; side: string; qty: number; filled_qty: number; filled_avg_price: number | null; status: string; submitted_at: string | null; filled_at: string | null };
type OrdersResponse = { count: number; orders: Order[] };
type Position = { ticker: string; qty: number; side: string; avg_entry_price: number | null; current_price: number | null; market_value: number | null; cost_basis: number | null; unrealized_pl: number | null; unrealized_plpc: number | null };
type PositionsResponse = { count: number; positions: Position[] };
type NewsArticle = { id: string | null; headline: string; summary: string; author: string | null; source: string; url: string | null; symbols: string[]; created_at: string | null; updated_at: string | null };
type NewsResponse = { count: number; articles: NewsArticle[] };

type Intelligence = {
  ticker: string;
  sentiment: "bullish" | "bearish" | "neutral";
  sentiment_confidence: number;
  impact: "high" | "medium" | "low";
  summary: string;
  key_events: string[];
  trade_rationale: string;
  risk_flags: string[];
};
type IntelligenceMeta = {
  model: string;
  input_tokens: number;
  output_tokens: number;
  elapsed_seconds: number;
  estimated_cost: number;
  cached: boolean;
  generated_at: string;
  n_articles_analyzed: number;
};
type IntelligenceResponse = {
  ticker: string;
  analysis: Intelligence | null;
  metadata: IntelligenceMeta | { reason: string };
  n_articles: number;
};

type FlashState = "up" | "dn" | null;
type Toast = { type: "success" | "error" | "info"; message: string };
type OrderModalState = { ticker: string; side: "buy" | "sell" } | null;
type ChartType = "candle" | "area";
type RangeOption = "1D" | "5D" | "1M" | "3M" | "6M" | "1Y";
type GranularityOption = "auto" | "1Min" | "5Min" | "15Min" | "30Min" | "1H" | "1D" | "1W";

const RANGE_TO_DAYS: Record<RangeOption, number> = { "1D": 1, "5D": 5, "1M": 30, "3M": 90, "6M": 180, "1Y": 365 };
const AUTO_GRANULARITY: Record<RangeOption, GranularityOption> = {
  "1D": "5Min", "5D": "15Min", "1M": "1H", "3M": "1D", "6M": "1D", "1Y": "1D",
};
type LayoutMode = "1x1" | "2x1" | "2x2" | "3x2" | "3x3";
type DetailTab = "overview" | "predictions" | "news" | "intel";
type ActivityTab = "orders" | "positions" | "news";

const LAYOUT_CONFIG: Record<LayoutMode, { cols: number; rows: number; cells: number }> = {
  "1x1": { cols: 1, rows: 1, cells: 1 },
  "2x1": { cols: 2, rows: 1, cells: 2 },
  "2x2": { cols: 2, rows: 2, cells: 4 },
  "3x2": { cols: 3, rows: 2, cells: 6 },
  "3x3": { cols: 3, rows: 3, cells: 9 },
};

const API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 5_000;
const PREV_CLOSE_REFRESH_MS = 60_000;
const ORDERS_REFRESH_MS = 10_000;
const NEWS_REFRESH_MS = 60_000;
const MAX_QTY_PER_ORDER = 100;
const MAX_NOTIONAL_PER_ORDER = 10_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtMoney = (n: number) => (n >= 1000 ? `$${(n / 1000).toFixed(1)}k` : `$${n.toFixed(2)}`);
const fmtPct = (n: number, withSign = true) => { const v = (n * 100).toFixed(2); return withSign && n > 0 ? `+${v}%` : `${v}%`; };
const fmtPrice = (n: number | null | undefined) => (n == null ? "—" : n.toFixed(2));
const computeBarChange = (bars: BarRow[] | undefined): { abs: number; pct: number } | null => {
  if (!bars || bars.length < 2) return null;
  const first = bars[0].close;
  const last = bars[bars.length - 1].close;
  return { abs: last - first, pct: (last - first) / first };
};
const computeDailyChange = (current: number | null | undefined, prevClose: number | null | undefined): { abs: number; pct: number } | null => {
  if (current == null || prevClose == null || prevClose === 0) return null;
  return { abs: current - prevClose, pct: (current - prevClose) / prevClose };
};

const fmtRelativeTime = (iso: string | null): string => {
  if (!iso) return "—";
  const date = new Date(iso);
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMs / 3600000);
  const diffDay = Math.floor(diffMs / 86400000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

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
    if (anyChanged) setFlashes((f) => ({ ...f, ...newFlashes }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(prices)]);
  useEffect(() => {
    return () => Object.values(timeoutsRef.current).forEach(clearTimeout);
  }, []);
  return flashes;
}

// ---------------------------------------------------------------------------
// Main
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
  const [recentNews, setRecentNews] = useState<NewsResponse | null>(null);
  const [tickerNews, setTickerNews] = useState<NewsResponse | null>(null);
  const [intelligence, setIntelligence] = useState<IntelligenceResponse | null>(null);
  const [intelLoading, setIntelLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string>("NVDA");
  const [tickerPredictions, setTickerPredictions] = useState<Prediction[]>([]);
  const [selectedBars, setSelectedBars] = useState<BarsResponse | null>(null);
  const [spyBars, setSpyBars] = useState<BarsResponse | null>(null);
  const [pollCount, setPollCount] = useState(0);
  const [activityTab, setActivityTab] = useState<ActivityTab>("orders");
  const [detailTab, setDetailTab] = useState<DetailTab>("overview");
  const [orderModal, setOrderModal] = useState<OrderModalState>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const [chartType, setChartType] = useState<ChartType>("candle");
  const [globalRange, setGlobalRange] = useState<RangeOption>("3M");
  const [globalGranularity, setGlobalGranularity] = useState<GranularityOption>("auto");
  // Clear bars cache when timeframe changes so charts refetch
  useEffect(() => {
    setAllBars({});
    setSelectedBars(null);
    setSpyBars(null);
  }, [globalRange, globalGranularity]);

  // Refetch SPY bars when timeframe changes
  useEffect(() => {
    const tf = globalGranularity === "auto" ? AUTO_GRANULARITY[globalRange] : globalGranularity;
    fetch(`${API_BASE}/api/historical-bars/SPY?range_days=${RANGE_TO_DAYS[globalRange]}&timeframe=${tf}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setSpyBars(data); })
      .catch(() => {});
  }, [globalRange, globalGranularity]);



  const [layout, setLayout] = useState<LayoutMode>("2x2");
  const [cellTickers, setCellTickers] = useState<Record<number, string>>({});

  // Live stream polling: when intraday timeframe is active, poll the websocket
  // buffer every 3 seconds and merge incoming bars with existing historical data.
  useEffect(() => {
    const tf = globalGranularity === "auto" ? AUTO_GRANULARITY[globalRange] : globalGranularity;
    const intraday = ["1Min", "5Min", "15Min", "30Min"].includes(tf);
    if (!intraday) return;

    let cancelled = false;

    async function pollStream() {
      // Build set of visible tickers
      const visible = new Set<string>();
      visible.add(selectedTicker);
      visible.add("SPY");
      Object.values(cellTickers).forEach((t) => { if (t) visible.add(t); });

      const promises = Array.from(visible).map(async (ticker) => {
        try {
          const res = await fetch(`${API_BASE}/api/stream/bars/${ticker}?n=200`);
          if (!res.ok) return null;
          const data = await res.json();
          if (!data?.bars?.length) return null;
          return { ticker, bars: data.bars };
        } catch {
          return null;
        }
      });

      const results = await Promise.all(promises);
      if (cancelled) return;

      results.forEach((r) => {
        if (!r) return;
        const streamBars = r.bars;
        // Merge: keep historical bars whose timestamp is before earliest stream bar,
        // then append all stream bars. This avoids duplicates.
        const earliestStream = streamBars[0].date;

        if (r.ticker === selectedTicker) {
          setSelectedBars((prev) => {
            if (!prev?.data) return { ...prev, data: streamBars, n_bars: streamBars.length } as any;
            const historical = prev.data.filter((b: any) => b.date < earliestStream);
            return { ...prev, data: [...historical, ...streamBars], n_bars: historical.length + streamBars.length };
          });
        }
        if (r.ticker === "SPY") {
          setSpyBars((prev) => {
            if (!prev?.data) return { ...prev, data: streamBars, n_bars: streamBars.length } as any;
            const historical = prev.data.filter((b: any) => b.date < earliestStream);
            return { ...prev, data: [...historical, ...streamBars], n_bars: historical.length + streamBars.length };
          });
        }
        // For other cells, update allBars
        if (r.ticker !== selectedTicker && r.ticker !== "SPY") {
          setAllBars((prev) => {
            const existing = prev[r.ticker];
            if (!existing?.data) return { ...prev, [r.ticker]: { ticker: r.ticker, data: streamBars, n_bars: streamBars.length } as any };
            const historical = existing.data.filter((b: any) => b.date < earliestStream);
            return { ...prev, [r.ticker]: { ...existing, data: [...historical, ...streamBars], n_bars: historical.length + streamBars.length } };
          });
        }
      });
    }

    // Initial fetch + interval
    pollStream();
    const id = setInterval(pollStream, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, [globalRange, globalGranularity, selectedTicker, cellTickers]);

  // Fetch heatmap probability curves for currently displayed tickers.
  // Triggered by changes in livePrices (so heatmaps stay in sync with market).
  useEffect(() => {
    const visibleTickers = Array.from(new Set(Object.values(cellTickers).concat([selectedTicker])));
    if (visibleTickers.length === 0 || !livePrices) return;
    const pricesArr: number[] = [];
    const validTickers: string[] = [];
    for (const t of visibleTickers) {
      const p = livePrices.prices?.[t]?.mid_price;
      if (p != null && Number.isFinite(p)) {
        pricesArr.push(p);
        validTickers.push(t);
      }
    }
    if (validTickers.length === 0) return;
    const url = `http://localhost:8000/api/heatmap-batch?tickers=${validTickers.join(",")}&prices=${pricesArr.join(",")}`;
    fetch(url)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => { if (data) setHeatmaps(data); })
      .catch(() => { /* ignore - non-critical */ });
  }, [livePrices, cellTickers, selectedTicker]);
  const [allBars, setAllBars] = useState<Record<string, BarsResponse>>({});
  const [heatmaps, setHeatmaps] = useState<Record<string, HeatmapData>>({});

  const priceMap: Record<string, number | null | undefined> = {};
  if (livePrices) Object.entries(livePrices.quotes).forEach(([t, q]) => { priceMap[t] = q.mid_price; });
  const flashes = usePriceFlash(priceMap);

  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  useEffect(() => {
    async function loadInitial() {
      try {
        const [summaryRes, latestRes, equityRes, accountRes, spyBarsRes, prevClosesRes, ordersRes, positionsRes, newsRes] = await Promise.all([
          fetch(`${API_BASE}/api/summary`),
          fetch(`${API_BASE}/api/predictions/latest?top_n=10`),
          fetch(`${API_BASE}/api/equity-curve`),
          fetch(`${API_BASE}/api/account`),
          fetch(`${API_BASE}/api/historical-bars/SPY?range_days=${RANGE_TO_DAYS[globalRange]}&timeframe=${globalGranularity === "auto" ? AUTO_GRANULARITY[globalRange] : globalGranularity}`),
          fetch(`${API_BASE}/api/prev-closes`),
          fetch(`${API_BASE}/api/orders/recent`),
          fetch(`${API_BASE}/api/positions`),
          fetch(`${API_BASE}/api/news/recent?limit=20`),
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
        if (newsRes.ok) setRecentNews(await newsRes.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    loadInitial();
  }, []);

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

  const fetchRecentNews = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/news/recent?limit=20`);
      if (res.ok) setRecentNews(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    const interval = setInterval(fetchRecentNews, NEWS_REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchRecentNews]);

  // Load ticker-specific data on selection change
  useEffect(() => {
    async function loadTickerData() {
      try {
        const [predRes, barsRes, newsRes] = await Promise.all([
          fetch(`${API_BASE}/api/predictions/${selectedTicker}`),
          fetch(`${API_BASE}/api/historical-bars/${selectedTicker}?range_days=${RANGE_TO_DAYS[globalRange]}&timeframe=${globalGranularity === "auto" ? AUTO_GRANULARITY[globalRange] : globalGranularity}`),
          fetch(`${API_BASE}/api/news/${selectedTicker}?limit=15`),
        ]);
        if (predRes.ok) {
          const data = await predRes.json();
          setTickerPredictions(data.data || []);
        }
        if (barsRes.ok) setSelectedBars(await barsRes.json());
        if (newsRes.ok) setTickerNews(await newsRes.json());
      } catch {}
    }
    loadTickerData();
    // Reset intel when ticker changes
    setIntelligence(null);
  }, [selectedTicker, globalRange, globalGranularity]);

  // Fetch intelligence when intel tab opened or ticker changes (only if tab is intel)
  const fetchIntelligence = useCallback(async (forceRefresh = false) => {
    setIntelLoading(true);
    try {
      const url = `${API_BASE}/api/intelligence/${selectedTicker}${forceRefresh ? "?force_refresh=true" : ""}`;
      const res = await fetch(url);
      if (res.ok) {
        setIntelligence(await res.json());
      } else {
        const err = await res.json();
        setToast({ type: "error", message: err.detail || "Intelligence request failed" });
      }
    } catch (e) {
      setToast({ type: "error", message: e instanceof Error ? e.message : "Intelligence error" });
    } finally {
      setIntelLoading(false);
    }
  }, [selectedTicker]);

  useEffect(() => {
    if (detailTab === "intel" && !intelligence && !intelLoading) {
      fetchIntelligence();
    }
  }, [detailTab, intelligence, intelLoading, fetchIntelligence]);

  useEffect(() => {
    if (!latest) return;
    const cfg = LAYOUT_CONFIG[layout];
    const universe = latest.predictions.map((p) => p.ticker);
    const needed = new Set<string>();
    for (let i = 0; i < cfg.cells; i++) {
      let t = cellTickers[i];
      if (!t) {
        if (i === 0) t = selectedTicker;
        else if (i === 1) t = "SPY";
        else {
          const remaining = universe.filter((x) => x !== selectedTicker && x !== "SPY");
          const idx = i - 2;
          if (idx < remaining.length) t = remaining[idx];
        }
      }
      if (t && t !== selectedTicker && t !== "SPY" && !allBars[t]) {
        needed.add(t);
      }
    }
    needed.forEach(async (ticker) => {
      try {
        const res = await fetch(`${API_BASE}/api/historical-bars/${ticker}?range_days=${RANGE_TO_DAYS[globalRange]}&timeframe=${globalGranularity === "auto" ? AUTO_GRANULARITY[globalRange] : globalGranularity}`);
        if (res.ok) {
          const data = await res.json();
          setAllBars((prev) => ({ ...prev, [ticker]: data }));
        }
      } catch {}
    });
  }, [layout, cellTickers, selectedTicker, latest, allBars, globalRange, globalGranularity]);

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
      setToast({ type: "success", message: `${side.toUpperCase()} ${qty} ${ticker} @ ~$${data.estimated_price.toFixed(2)} submitted` });
      setOrderModal(null);
      fetchOrdersAndPositions();
    } catch (e) {
      setToast({ type: "error", message: e instanceof Error ? e.message : "Network error" });
    }
  }

  if (error) {
    return (
      <main style={{ padding: 20, color: "var(--red)" }}>
        <h1>Backend connection error: {error}</h1>
        <p style={{ color: "var(--text-muted)", marginTop: 8 }}>Make sure FastAPI is running on port 8000</p>
      </main>
    );
  }

  if (!summary || !latest || !equityCurve) {
    return <main style={{ padding: 20, color: "var(--text-muted)" }}>Loading dashboard...</main>;
  }

  const s = summary.strategy;
  const ew = summary.equal_weight;
  const selectedQuote = livePrices?.quotes[selectedTicker];
  const selectedPrediction = latest.predictions.find((p) => p.ticker === selectedTicker);
  const selectedRank = latest.predictions.findIndex((p) => p.ticker === selectedTicker) + 1;
  const universeOrdered = latest.predictions.map((p) => p.ticker).filter((t) => livePrices?.quotes[t]);
  const universe = latest.predictions.map((p) => p.ticker);
  const selectedPrevClose = prevCloses?.data[selectedTicker]?.prev_close;
  const selectedDailyChange = computeDailyChange(selectedQuote?.mid_price, selectedPrevClose);
  const selected90dChange = computeBarChange(selectedBars?.data);
  const spyDailyChange = computeDailyChange(livePrices?.quotes["SPY"]?.mid_price, prevCloses?.data["SPY"]?.prev_close);
  const cfg = LAYOUT_CONFIG[layout];

  const resolveCellTicker = (i: number): string | null => {
    if (cellTickers[i]) return cellTickers[i];
    if (i === 0) return selectedTicker;
    if (i === 1 && cfg.cells >= 2) return "SPY";
    const remaining = universe.filter((t) => t !== selectedTicker && t !== "SPY");
    const idx = i - 2;
    if (idx >= 0 && idx < remaining.length) return remaining[idx];
    return null;
  };

  return (
    <div className="layout">
      <header className="ticker-strip">
        <div className="brand-cell"><div className="brand-logo">C</div></div>
        <div className="ticker-tape">
          {universeOrdered.map((ticker) => {
            const q = livePrices?.quotes[ticker];
            const prev = prevCloses?.data[ticker]?.prev_close;
            const dc = computeDailyChange(q?.mid_price, prev);
            const flash = flashes[ticker];
            return (
              <div key={ticker} className={`tape-item ${flash === "up" ? "flash-up" : flash === "dn" ? "flash-dn" : ""}`} onClick={() => setSelectedTicker(ticker)} style={{ cursor: "pointer" }}>
                <span className="tape-sym">{ticker}</span>
                <span className="tape-price">{fmtPrice(q?.mid_price)}</span>
                <span className={`tape-chg ${dc && dc.pct >= 0 ? "up" : "dn"}`}>{dc ? fmtPct(dc.pct) : "—"}</span>
              </div>
            );
          })}
          {!livePrices && <div className="tape-item"><span className="tape-sym" style={{ color: "var(--text-muted)" }}>Loading prices...</span></div>}
        </div>
        <div className="market-clock">
          <span className="clock-pill">
            <span className="pulse"></span>
            {new Date().getDay() === 0 || new Date().getDay() === 6 ? "MARKET CLOSED" : "MARKET OPEN"}
          </span>
          <span>{new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })} ET</span>
        </div>
        <div className="acct-mini">
          <div className="acct-mini-item"><span className="acct-mini-label">Equity</span><span className="acct-mini-value">{account ? `$${account.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}</span></div>
          <div className="acct-mini-item"><span className="acct-mini-label">BP</span><span className="acct-mini-value">{account ? fmtMoney(account.buying_power) : "—"}</span></div>
        </div>
      </header>

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

      <main className="main">
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

        <div className="toolbar">
          <div className="toolbar-section">
            <span className="toolbar-text"><strong>Layout</strong></span>
            {(["1x1", "2x1", "2x2", "3x2", "3x3"] as LayoutMode[]).map((l) => (
              <button key={l} className={`layout-btn ${layout === l ? "active" : ""}`} onClick={() => setLayout(l)}>
                {l.replace("x", "×")}
              </button>
            ))}
          </div>
          <div className="toolbar-section">
            <span className="toolbar-text"><strong>Chart</strong></span>
            <button className={`layout-btn ${chartType === "candle" ? "active" : ""}`} onClick={() => setChartType("candle")}>Candle</button>
            <button className={`layout-btn ${chartType === "area" ? "active" : ""}`} onClick={() => setChartType("area")}>Area</button>
          </div>
          <div className="toolbar-section">
            <span className="toolbar-text"><strong>Range</strong></span>
            {(["1D", "5D", "1M", "3M", "6M", "1Y"] as RangeOption[]).map((r) => (
              <button key={r} className={`layout-btn ${globalRange === r ? "active" : ""}`} onClick={() => setGlobalRange(r)}>{r}</button>
            ))}
          </div>
          <div className="toolbar-section">
            <span className="toolbar-text"><strong>Bars</strong></span>
            <select value={globalGranularity} onChange={(e) => setGlobalGranularity(e.target.value as GranularityOption)} style={{ background: "var(--bg-elevated)", color: "var(--text-primary)", border: "1px solid var(--border-strong)", borderRadius: 3, padding: "2px 6px", fontSize: 10, fontFamily: "inherit", cursor: "pointer", outline: "none" }}>
              <option value="auto">auto ({AUTO_GRANULARITY[globalRange]})</option>
              <option value="1Min">1Min</option>
              <option value="5Min">5Min</option>
              <option value="15Min">15Min</option>
              <option value="30Min">30Min</option>
              <option value="1H">1H</option>
              <option value="1D">1D</option>
              <option value="1W">1W</option>
            </select>
          </div>
          <div className="toolbar-section">
            <span className="toolbar-text">Polls:</span>
            <span className="toolbar-text"><strong>{pollCount}</strong></span>
            <span className="toolbar-text" style={{ color: "var(--text-faint)" }}>· every {POLL_INTERVAL_MS / 1000}s</span>
          </div>
          <div className="toolbar-section" style={{ marginLeft: "auto" }}>
            <span className="toolbar-text">Selected: <strong>{selectedTicker}</strong></span>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: `repeat(${cfg.cols}, 1fr)`, gridTemplateRows: `repeat(${cfg.rows}, 1fr)`, gap: 1, background: "var(--border)", overflow: "hidden" }}>
          {Array.from({ length: cfg.cells }).map((_, i) => {
            const t = resolveCellTicker(i);
            const isLast = i === cfg.cells - 1;
            const isSecondToLast = i === cfg.cells - 2;
            const noTicker = t === null;
            if (layout === "2x2" && i === 2) return <EquityChartCell key={`cell-${i}`} equityCurve={equityCurve} />;
            if (layout === "2x2" && i === 3) return <SignalsTable key={`cell-${i}`} predictions={latest.predictions} onSelect={setSelectedTicker} selectedTicker={selectedTicker} />;
            if (noTicker) {
              if (isSecondToLast && cfg.cells >= 4) return <EquityChartCell key={`cell-${i}`} equityCurve={equityCurve} />;
              if (isLast && cfg.cells >= 4) return <SignalsTable key={`cell-${i}`} predictions={latest.predictions} onSelect={setSelectedTicker} selectedTicker={selectedTicker} />;
              return <div key={`cell-${i}`} style={{ background: "var(--bg-base)" }} />;
            }
            const q = livePrices?.quotes[t];
            const prev = prevCloses?.data[t]?.prev_close;
            const dc = computeDailyChange(q?.mid_price, prev);
            const bars = t === selectedTicker ? selectedBars?.data : t === "SPY" ? spyBars?.data : allBars[t]?.data;
            const color = i === 0 ? "#4ADE80" : i === 1 ? "#60A5FA" : "#A78BFA";
            return (
              <PriceChartCell
                key={`cell-${i}`}
                cellIndex={i}
                symbol={t}
                universe={universe}
                quote={q}
                bars={bars}
                change={dc}
                color={color}
                flash={flashes[t]}
                chartType={chartType}
                onTickerChange={(newTicker) => {
                  setCellTickers((prev) => ({ ...prev, [i]: newTicker }));
                  if (i === 0) setSelectedTicker(newTicker);
                }}
                heatmap={heatmaps[t]}
              />
            );
          })}
        </div>

        <div className="bottom-row">
          <BottomCell title="Latest Signals" meta={`${latest.predictions.length} · click TRADE`}>
            <div style={{ overflow: "auto", maxHeight: "100%" }}>
              {latest.predictions.slice(0, 8).map((p, idx) => (
                <SignalRow key={p.ticker} rank={idx + 1} prediction={p} quote={livePrices?.quotes[p.ticker]} prevClose={prevCloses?.data[p.ticker]?.prev_close} selected={p.ticker === selectedTicker} flash={flashes[p.ticker]} onClick={() => setSelectedTicker(p.ticker)} onTrade={(side) => setOrderModal({ ticker: p.ticker, side })} />
              ))}
            </div>
          </BottomCell>

          <BottomCell title={`Level II · ${selectedTicker}`} meta="placeholder · paid tier">
            <div style={{ padding: 12, color: "var(--text-muted)", fontSize: 10, textAlign: "center" }}>
              <p style={{ marginBottom: 8 }}>Level II depth requires Alpaca paid data tier</p>
              <p style={{ color: "var(--text-faint)" }}>Currently showing best bid/ask from free tier:</p>
              <div style={{ marginTop: 12, fontSize: 11 }}>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 16px" }}>
                  <span style={{ color: "var(--green)" }}>BID</span><span>{fmtPrice(selectedQuote?.bid_price)}</span><span style={{ color: "var(--text-muted)" }}>{selectedQuote?.bid_size ?? "—"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 16px" }}>
                  <span style={{ color: "var(--red)" }}>ASK</span><span>{fmtPrice(selectedQuote?.ask_price)}</span><span style={{ color: "var(--text-muted)" }}>{selectedQuote?.ask_size ?? "—"}</span>
                </div>
              </div>
            </div>
          </BottomCell>

          <div style={{ background: "var(--bg-panel)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ padding: "0 10px", background: "var(--bg-elevated)", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex" }}>
                <button onClick={() => setActivityTab("orders")} style={tabBtnStyle(activityTab === "orders")}>Orders ({orders?.count ?? 0})</button>
                <button onClick={() => setActivityTab("positions")} style={tabBtnStyle(activityTab === "positions")}>Positions ({positions?.count ?? 0})</button>
                <button onClick={() => setActivityTab("news")} style={tabBtnStyle(activityTab === "news")}>News ({recentNews?.count ?? 0})</button>
              </div>
              <span style={{ fontSize: 9, color: "var(--text-faint)" }}>refreshes 10-60s</span>
            </div>
            <div style={{ flex: 1, overflow: "auto" }}>
              {activityTab === "orders" && <OrdersList orders={orders?.orders ?? []} />}
              {activityTab === "positions" && <PositionsList positions={positions?.positions ?? []} />}
              {activityTab === "news" && <NewsList articles={recentNews?.articles ?? []} onTickerClick={setSelectedTicker} universe={universe} />}
            </div>
          </div>
        </div>
      </main>

      <aside className="right-panel">
        <div className="ticker-detail-header">
          <div className="td-top-row">
            <div className="td-symbol-block">
              <div className="td-symbol">{selectedTicker}</div>
              <div className="td-name">{selectedPrediction ? `Rank ${selectedRank} of ${latest.predictions.length}` : "—"}</div>
              <div className="td-tags">
                {selectedPrediction && selectedRank <= 2 && <span className="td-tag signal">★ TOP SIGNAL</span>}
                <span className="td-tag">11-stock universe</span>
              </div>
            </div>
            <div className="td-price-block">
              <div className="td-price">{fmtPrice(selectedQuote?.mid_price)}</div>
              {selectedDailyChange && <div className={`td-change ${selectedDailyChange.pct >= 0 ? "up" : "dn"}`} style={{ fontSize: 13 }}>{fmtPct(selectedDailyChange.pct)} today</div>}
              {selected90dChange && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>90d: <span className={selected90dChange.pct >= 0 ? "up" : "dn"}>{fmtPct(selected90dChange.pct)}</span></div>}
            </div>
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
            <button onClick={() => setOrderModal({ ticker: selectedTicker, side: "buy" })} style={{ flex: 1, background: "var(--green)", color: "var(--bg-base)", border: "none", borderRadius: 4, padding: "6px 0", fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>BUY</button>
            <button onClick={() => setOrderModal({ ticker: selectedTicker, side: "sell" })} style={{ flex: 1, background: "var(--red)", color: "var(--bg-base)", border: "none", borderRadius: 4, padding: "6px 0", fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>SELL</button>
          </div>
        </div>

        <div className="detail-tabs">
          <span className={`detail-tab ${detailTab === "overview" ? "active" : ""}`} onClick={() => setDetailTab("overview")}>Overview</span>
          <span className={`detail-tab ${detailTab === "predictions" ? "active" : ""}`} onClick={() => setDetailTab("predictions")}>Predictions</span>
          <span className={`detail-tab ${detailTab === "news" ? "active" : ""}`} onClick={() => setDetailTab("news")}>News ({tickerNews?.count ?? 0})</span>
          <span className={`detail-tab ${detailTab === "intel" ? "active" : ""}`} onClick={() => setDetailTab("intel")}>
            <span style={{ background: "linear-gradient(90deg, var(--green), var(--cyan))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text", fontWeight: 700 }}>AI Intel</span>
          </span>
        </div>

        <div className="detail-body">
          {detailTab === "overview" && (
            <>
              <div className="detail-section">
                <div className="quote-stats">
                  <QSCell label="Bid" value={fmtPrice(selectedQuote?.bid_price)} />
                  <QSCell label="Ask" value={fmtPrice(selectedQuote?.ask_price)} />
                  <QSCell label="Mid" value={fmtPrice(selectedQuote?.mid_price)} />
                  <QSCell label="Prev Close" value={fmtPrice(selectedPrevClose)} />
                  <QSCell label="Spread" value={selectedQuote?.bid_price && selectedQuote?.ask_price ? `$${(selectedQuote.ask_price - selectedQuote.bid_price).toFixed(2)}` : "—"} />
                  <QSCell label="Day Change" value={selectedDailyChange ? `${selectedDailyChange.abs >= 0 ? "+" : ""}$${selectedDailyChange.abs.toFixed(2)}` : "—"} />
                  {selectedBars?.data.length ? (
                    <>
                      <QSCell label="90d High" value={`$${Math.max(...selectedBars.data.map((b) => b.high)).toFixed(2)}`} />
                      <QSCell label="90d Low" value={`$${Math.min(...selectedBars.data.map((b) => b.low)).toFixed(2)}`} />
                    </>
                  ) : null}
                </div>
              </div>
              <div className="detail-section">
                <div className="detail-section-header"><span>ML Signal Detail</span><span className="meta">Random Forest</span></div>
                <div className="detail-section-body">
                  {selectedPrediction ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 6, fontSize: 11, lineHeight: 1.6 }}>
                      <span style={{ color: "var(--text-muted)" }}>Buy probability</span><span style={{ fontWeight: 600 }} className={selectedPrediction.y_proba >= 0.55 ? "up" : ""}>{selectedPrediction.y_proba.toFixed(3)}</span>
                      <span style={{ color: "var(--text-muted)" }}>Rank in universe</span><span style={{ fontWeight: 600 }}>{selectedRank} of {latest.predictions.length}</span>
                      <span style={{ color: "var(--text-muted)" }}>Realized 5d return</span><span style={{ fontWeight: 600 }} className={selectedPrediction.fwd_return_5d >= 0 ? "up" : "dn"}>{fmtPct(selectedPrediction.fwd_return_5d)}</span>
                      <span style={{ color: "var(--text-muted)" }}>Decision</span><span style={{ fontWeight: 600 }} className={selectedPrediction.y_proba >= 0.55 ? "up" : ""}>{selectedPrediction.y_proba >= 0.55 ? "BUY" : selectedPrediction.y_proba >= 0.50 ? "WATCH" : "HOLD"}</span>
                      <span style={{ color: "var(--text-muted)" }}>Date</span><span style={{ fontWeight: 600 }}>{selectedPrediction.date}</span>
                      <span style={{ color: "var(--text-muted)" }}>CV fold</span><span style={{ fontWeight: 600 }}>{selectedPrediction.fold}</span>
                    </div>
                  ) : <div style={{ color: "var(--text-muted)", fontSize: 11 }}>No prediction available for {selectedTicker}</div>}
                </div>
              </div>
            </>
          )}

          {detailTab === "predictions" && (
            <div className="detail-section">
              <div className="detail-section-header"><span>Prediction History</span><span className="meta">{tickerPredictions.length} preds</span></div>
              <div className="detail-section-body" style={{ padding: 0, fontSize: 10 }}>
                {tickerPredictions.length === 0 ? (
                  <div style={{ padding: 12, color: "var(--text-muted)" }}>Loading...</div>
                ) : (
                  tickerPredictions.slice(-30).reverse().map((p) => (
                    <div key={`${p.date}-${p.ticker}`} style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 8, padding: "5px 12px", borderBottom: "1px solid var(--border-soft)", fontFeatureSettings: "'tnum'" }}>
                      <span style={{ color: "var(--text-muted)" }}>{p.date}</span>
                      <span style={{ fontWeight: 600 }}>{p.y_proba.toFixed(3)}</span>
                      <span className={p.fwd_return_5d >= 0 ? "up" : "dn"}>{fmtPct(p.fwd_return_5d)}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {detailTab === "news" && (
            <div className="detail-section">
              <div className="detail-section-header"><span>{selectedTicker} News</span><span className="meta">{tickerNews?.count ?? 0} articles</span></div>
              <div style={{ padding: 0 }}>
                {!tickerNews || tickerNews.articles.length === 0 ? (
                  <div style={{ padding: 12, color: "var(--text-muted)", fontSize: 11 }}>
                    {tickerNews ? `No recent news for ${selectedTicker}` : "Loading..."}
                  </div>
                ) : (
                  tickerNews.articles.map((article) => (
                    <NewsItem key={article.id ?? article.headline} article={article} compact />
                  ))
                )}
              </div>
            </div>
          )}

          {detailTab === "intel" && (
            <IntelligencePanel
              ticker={selectedTicker}
              intelligence={intelligence}
              loading={intelLoading}
              onRefresh={() => fetchIntelligence(true)}
            />
          )}
        </div>
      </aside>

      <footer className="status-bar">
        <div className="status-item"><span className="status-dot"></span>API: ALPACA · {livePrices ? "LIVE" : "CONNECTING"}</div>
        <div className="status-item">POLLING: {POLL_INTERVAL_MS / 1000}s · {pollCount} polls</div>
        <div className="status-item">UNIVERSE: 11 tickers</div>
        <div className="status-item">PAPER · {orders?.count ?? 0} orders · {positions?.count ?? 0} positions · {recentNews?.count ?? 0} news</div>
        <div className="status-item" style={{ marginLeft: "auto" }}>v1.0 · {account?.account_status ?? "—"}</div>
      </footer>

      {orderModal && (
        <OrderModal ticker={orderModal.ticker} side={orderModal.side} quote={livePrices?.quotes[orderModal.ticker]} buyingPower={account?.buying_power} onClose={() => setOrderModal(null)} onSubmit={(qty) => handlePlaceOrder(orderModal.ticker, orderModal.side, qty)} />
      )}

      {toast && (
        <div style={{ position: "fixed", bottom: 30, right: 30, background: toast.type === "success" ? "var(--green-bg-strong)" : toast.type === "error" ? "rgba(248,113,113,0.2)" : "var(--bg-elevated)", border: `1px solid ${toast.type === "success" ? "var(--green)" : toast.type === "error" ? "var(--red)" : "var(--border)"}`, color: toast.type === "success" ? "var(--green)" : toast.type === "error" ? "var(--red)" : "var(--text-primary)", padding: "12px 20px", borderRadius: 6, fontSize: 12, fontWeight: 500, zIndex: 1000, maxWidth: 400 }}>
          {toast.message}
        </div>
      )}

      <style jsx>{`
        .layout { display: grid; grid-template-columns: 44px 1fr 320px; grid-template-rows: 32px 1fr 20px; height: 100vh; width: 100vw; }
        .ticker-strip { grid-column: 1 / -1; background: var(--bg-panel); border-bottom: 1px solid var(--border); display: flex; align-items: stretch; }
        .brand-cell { flex: 0 0 44px; display: flex; align-items: center; justify-content: center; border-right: 1px solid var(--border); }
        .brand-logo { width: 24px; height: 24px; border-radius: 5px; background: linear-gradient(135deg, var(--green), var(--cyan)); display: flex; align-items: center; justify-content: center; font-weight: 700; color: var(--bg-base); font-size: 12px; }
        .ticker-tape { flex: 1; display: flex; align-items: center; overflow-x: auto; padding: 0 4px; }
        .ticker-tape::-webkit-scrollbar { display: none; }
        .tape-item { padding: 0 10px; display: flex; gap: 6px; align-items: center; white-space: nowrap; border-right: 1px solid var(--border-soft); font-size: 10px; }
        .tape-item:hover { background: var(--bg-row); }
        .tape-sym { font-weight: 600; color: var(--text-primary); }
        .tape-price { color: var(--text-secondary); font-feature-settings: "tnum"; }
        .tape-chg { font-feature-settings: "tnum"; font-weight: 500; }
        .tape-chg.up { color: var(--green); }
        .tape-chg.dn { color: var(--red); }
        .market-clock { padding: 0 12px; display: flex; gap: 10px; align-items: center; border-left: 1px solid var(--border); font-size: 10px; color: var(--text-secondary); }
        .clock-pill { background: var(--green-bg); color: var(--green); padding: 2px 8px; border-radius: 99px; font-size: 9px; font-weight: 700; display: flex; gap: 4px; align-items: center; text-transform: uppercase; letter-spacing: 0.05em; }
        .pulse { width: 5px; height: 5px; background: var(--green); border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }
        .acct-mini { padding: 0 12px; border-left: 1px solid var(--border); display: flex; gap: 12px; align-items: center; font-size: 10px; }
        .acct-mini-item { display: flex; gap: 6px; }
        .acct-mini-label { color: var(--text-muted); }
        .acct-mini-value { font-weight: 600; }
        .sidebar { grid-column: 1; grid-row: 2; background: var(--bg-panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; align-items: center; padding: 6px 0; gap: 3px; }
        .nav-btn { width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: var(--text-muted); cursor: pointer; font-size: 14px; }
        .nav-btn:hover { background: var(--bg-row); color: var(--text-primary); }
        .nav-btn.active { background: var(--green-bg-strong); color: var(--green); }
        .nav-divider { height: 1px; width: 20px; background: var(--border); margin: 4px 0; }
        .nav-spacer { flex: 1; }
        .main { grid-column: 2; grid-row: 2; display: grid; grid-template-rows: auto auto 1fr 180px; overflow: hidden; background: var(--bg-base); }
        .stat-bar { background: var(--bg-panel); border-bottom: 1px solid var(--border); display: grid; grid-template-columns: repeat(8, 1fr); }
        .toolbar { background: var(--bg-panel); border-bottom: 1px solid var(--border); padding: 5px 8px; display: flex; gap: 10px; align-items: center; font-size: 10px; }
        .toolbar-section { display: flex; gap: 3px; align-items: center; padding-right: 10px; border-right: 1px solid var(--border); }
        .toolbar-section:last-child { border-right: none; }
        .toolbar-text { color: var(--text-muted); font-size: 10px; }
        .toolbar-text strong { color: var(--text-primary); }
        .layout-btn { background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-secondary); padding: 3px 7px; border-radius: 3px; cursor: pointer; font-size: 10px; font-family: inherit; }
        .layout-btn:hover { background: var(--bg-row); color: var(--text-primary); }
        .layout-btn.active { background: var(--green-bg); color: var(--green); border-color: var(--green); }
        .bottom-row { background: var(--bg-panel); border-top: 1px solid var(--border); display: grid; grid-template-columns: 1.5fr 1fr 1.5fr; gap: 1px; background: var(--border); }
        .right-panel { grid-column: 3; grid-row: 2; background: var(--bg-panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
        .ticker-detail-header { padding: 10px 12px; border-bottom: 1px solid var(--border); background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-panel) 100%); }
        .td-top-row { display: flex; justify-content: space-between; align-items: flex-start; }
        .td-symbol-block { flex: 1; }
        .td-symbol { font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
        .td-name { font-size: 10px; color: var(--text-muted); margin-top: -2px; }
        .td-tags { display: flex; gap: 3px; margin-top: 5px; flex-wrap: wrap; }
        .td-tag { font-size: 8px; padding: 2px 5px; border-radius: 99px; background: var(--bg-row); color: var(--text-secondary); font-weight: 500; }
        .td-tag.signal { background: var(--green-bg); color: var(--green); font-weight: 700; }
        .td-price-block { text-align: right; }
        .td-price { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
        .td-change { font-size: 12px; font-weight: 600; margin-top: -2px; }
        .detail-tabs { display: flex; overflow-x: auto; border-bottom: 1px solid var(--border); background: var(--bg-elevated); }
        .detail-tabs::-webkit-scrollbar { display: none; }
        .detail-tab { padding: 6px 10px; color: var(--text-muted); cursor: pointer; font-size: 10px; font-weight: 500; border-bottom: 2px solid transparent; white-space: nowrap; }
        .detail-tab.active { color: var(--text-primary); border-bottom-color: var(--green); }
        .detail-body { flex: 1; overflow: auto; }
        .detail-section { border-bottom: 1px solid var(--border); }
        .detail-section-header { padding: 8px 12px; background: var(--bg-elevated); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }
        .detail-section-header .meta { color: var(--text-faint); font-weight: 500; text-transform: none; letter-spacing: 0; font-size: 10px; }
        .detail-section-body { padding: 8px 12px; }
        .quote-stats { display: grid; grid-template-columns: 1fr 1fr; }
        .status-bar { grid-column: 1 / -1; background: var(--bg-panel); border-top: 1px solid var(--border); display: flex; align-items: center; padding: 0 12px; gap: 16px; font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
        .status-item { display: flex; gap: 6px; align-items: center; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }
        .up { color: var(--green); }
        .dn { color: var(--red); }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tabBtnStyle(active: boolean): React.CSSProperties {
  return {
    background: "none",
    border: "none",
    cursor: "pointer",
    padding: "8px 12px",
    fontSize: 11,
    fontWeight: 600,
    color: active ? "var(--text-primary)" : "var(--text-muted)",
    borderBottom: active ? "2px solid var(--green)" : "2px solid transparent",
    fontFamily: "inherit",
  };
}

// ---------------------------------------------------------------------------
// Candlestick rendering
// ---------------------------------------------------------------------------

function CandleShape(props: any) {
  const { x, y, width, height, payload, yMin, yMax, chartHeight } = props;
  if (!payload || yMin == null || yMax == null || !chartHeight) return null;
  const { open, high, low, close } = payload as BarRow;
  const valueRange = yMax - yMin;
  if (valueRange <= 0) return null;
  const valueToY = (v: number) => ((yMax - v) / valueRange) * chartHeight;
  const yOpen = valueToY(open);
  const yClose = valueToY(close);
  const yHigh = valueToY(high);
  const yLow = valueToY(low);
  const isUp = close >= open;
  const color = isUp ? "#4ADE80" : "#F87171";
  const bodyTop = Math.min(yOpen, yClose);
  const bodyBottom = Math.max(yOpen, yClose);
  const bodyHeight = Math.max(1, bodyBottom - bodyTop);
  const candleWidth = Math.max(2, Math.min(width * 0.7, 8));
  const cx = x + width / 2;
  return (
    <g>
      <line x1={cx} y1={yHigh} x2={cx} y2={yLow} stroke={color} strokeWidth={1} />
      <rect x={cx - candleWidth / 2} y={bodyTop} width={candleWidth} height={bodyHeight} fill={color} stroke={color} />
    </g>
  );
}

type HeatmapData = {
  ticker: string;
  prices: number[];
  probabilities: number[];
  current_price: number;
  current_idx: number;
  current_probability: number;
  min_probability: number;
  max_probability: number;
  buy_threshold: number;
  error?: string;
};

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

function PriceChartCell({ cellIndex, symbol, universe, quote, bars, change, color, flash, chartType, onTickerChange, heatmap }: { cellIndex: number; symbol: string; universe: string[]; quote?: Quote; bars?: BarRow[]; change: { abs: number; pct: number } | null; color: string; flash?: FlashState; chartType: ChartType; onTickerChange: (newTicker: string) => void; heatmap?: HeatmapData }) {
  const lastClose = bars && bars.length > 0 ? bars[bars.length - 1].close : null;
  const displayPrice = quote?.mid_price ?? lastClose;
  const yDomain: [number | string, number | string] = bars && bars.length > 0
    ? [Math.min(...bars.map((b) => b.low)) * 0.995, Math.max(...bars.map((b) => b.high)) * 1.005]
    : ["auto", "auto"];
  const allTickers = universe.includes("SPY") ? universe : [...universe, "SPY"];
  return (
    <div style={{ background: "var(--bg-base)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div className={flash === "up" ? "flash-up" : flash === "dn" ? "flash-dn" : ""} style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "5px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select value={symbol} onChange={(e) => onTickerChange(e.target.value)} style={{ background: "var(--bg-elevated)", color: "var(--text-primary)", border: "1px solid var(--border-strong)", borderRadius: 3, padding: "2px 6px", fontSize: 11, fontFamily: "inherit", fontWeight: 700, cursor: "pointer", outline: "none" }}>
            {allTickers.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>· 90D</span>
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
                <linearGradient id={`grad-${cellIndex}-${symbol}`} x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2533" />
              <XAxis dataKey="date" stroke="#6A7488" tick={{ fontSize: 8 }} tickFormatter={(d: string) => d.includes("T") ? d.slice(11, 16) : d.slice(5)} interval={Math.max(0, Math.floor(bars.length / 5))} />
              <YAxis stroke="#6A7488" tick={{ fontSize: 8 }} domain={yDomain} tickFormatter={(v: number) => `$${v.toFixed(0)}`} width={40} />
              <Tooltip contentStyle={{ backgroundColor: "#11151D", border: "1px solid #1F2533", fontSize: 10 }} formatter={(v: number, name: string, props: any) => {
                if (chartType === "candle" && props?.payload) {
                  const p = props.payload as BarRow;
                  return [`O ${p.open.toFixed(2)} H ${p.high.toFixed(2)} L ${p.low.toFixed(2)} C ${p.close.toFixed(2)}`, ""];
                }
                return [`$${v.toFixed(2)}`, "Close"];
              }} />
              {chartType === "candle" ? (
                <Bar dataKey="close" shape={(p: any) => <CandleShape {...p} yMin={Math.min(...bars.map((b) => b.low)) * 0.995} yMax={Math.max(...bars.map((b) => b.high)) * 1.005} chartHeight={p.background?.height ?? 200} />} isAnimationActive={false}>
                  {bars.map((entry, idx) => (
                    <Cell key={idx} fill={entry.close >= entry.open ? "#4ADE80" : "#F87171"} />
                  ))}
                </Bar>
              ) : (
                <Area type="monotone" dataKey="close" stroke={color} strokeWidth={1.5} fill={`url(#grad-${cellIndex}-${symbol})`} />
              )}
              <Brush
                dataKey="date"
                height={18}
                stroke="#4A5568"
                fill="#11151D"
                travellerWidth={6}
                tickFormatter={(d: string) => typeof d === "string" && d.includes("T") ? d.slice(11, 16) : (typeof d === "string" ? d.slice(5) : "")}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
        {bars && bars.length > 0 && heatmap && (
          <div style={{ position: "absolute", top: 4, right: 4, bottom: 4, width: 48, pointerEvents: "none" }}>
            <HeatmapStrip
              data={heatmap}
              yMin={Math.min(...bars.map((b) => b.low)) * 0.995}
              yMax={Math.max(...bars.map((b) => b.high)) * 1.005}
              height={200}
              width={48}
            />
          </div>
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
      <button onClick={(e) => { e.stopPropagation(); onTrade(prediction.y_proba >= 0.50 ? "buy" : "sell"); }} style={{ fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 3, background: prediction.y_proba >= 0.55 ? "var(--green-bg)" : prediction.y_proba >= 0.50 ? "rgba(251,191,36,0.1)" : "var(--bg-row)", color: prediction.y_proba >= 0.55 ? "var(--green)" : prediction.y_proba >= 0.50 ? "var(--amber)" : "var(--text-muted)", border: "1px solid transparent", cursor: "pointer", fontFamily: "inherit" }} onMouseOver={(e) => (e.currentTarget.style.borderColor = "currentColor")} onMouseOut={(e) => (e.currentTarget.style.borderColor = "transparent")}>TRADE</button>
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
  if (orders.length === 0) return <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>No orders yet. Click TRADE on any signal to place one.</div>;
  return (
    <div>
      {orders.map((o) => (
        <div key={o.order_id} style={{ display: "grid", gridTemplateColumns: "auto 60px 50px 1fr auto", gap: 8, padding: "6px 10px", borderBottom: "1px solid var(--border-soft)", alignItems: "center", fontSize: 10 }}>
          <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 2, background: o.side === "buy" ? "var(--green-bg)" : "var(--red-bg)", color: o.side === "buy" ? "var(--green)" : "var(--red)" }}>{o.side.toUpperCase()}</span>
          <span style={{ fontWeight: 600, fontSize: 11 }}>{o.ticker}</span>
          <span style={{ color: "var(--text-secondary)", textAlign: "right" }}>{o.qty}sh</span>
          <span style={{ color: "var(--text-muted)", fontSize: 9 }}>
            {o.filled_avg_price ? `@ $${o.filled_avg_price.toFixed(2)}` : "queued"}
            {o.submitted_at && ` · ${new Date(o.submitted_at).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`}
          </span>
          <span style={{ fontSize: 9, fontWeight: 600, padding: "1px 5px", borderRadius: 2, background: o.status === "filled" ? "var(--green-bg)" : o.status === "canceled" || o.status === "rejected" ? "var(--red-bg)" : "var(--bg-row)", color: o.status === "filled" ? "var(--green)" : o.status === "canceled" || o.status === "rejected" ? "var(--red)" : "var(--text-muted)", textTransform: "uppercase" }}>{o.status}</span>
        </div>
      ))}
    </div>
  );
}

function PositionsList({ positions }: { positions: Position[] }) {
  if (positions.length === 0) return <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>No open positions. Orders fill at next market open.</div>;
  return (
    <div>
      {positions.map((p) => (
        <div key={p.ticker} style={{ display: "grid", gridTemplateColumns: "60px 50px 1fr auto", gap: 8, padding: "6px 10px", borderBottom: "1px solid var(--border-soft)", alignItems: "center", fontSize: 10 }}>
          <span style={{ fontWeight: 600, fontSize: 11 }}>{p.ticker}</span>
          <span style={{ color: "var(--text-secondary)", textAlign: "right" }}>{p.qty}sh</span>
          <span style={{ color: "var(--text-muted)", fontSize: 9 }}>avg ${p.avg_entry_price?.toFixed(2) ?? "—"} · now ${p.current_price?.toFixed(2) ?? "—"}</span>
          <span style={{ fontWeight: 600, textAlign: "right" }} className={p.unrealized_pl != null && p.unrealized_pl >= 0 ? "up" : "dn"}>
            {p.unrealized_pl != null ? `${p.unrealized_pl >= 0 ? "+" : ""}$${p.unrealized_pl.toFixed(2)}` : "—"}
            {p.unrealized_plpc != null && <span style={{ fontSize: 9, marginLeft: 4 }}>({fmtPct(p.unrealized_plpc)})</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

function NewsList({ articles, onTickerClick, universe }: { articles: NewsArticle[]; onTickerClick: (t: string) => void; universe: string[] }) {
  if (articles.length === 0) {
    return <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>No recent news. Refreshes every 60s.</div>;
  }
  return (
    <div>
      {articles.map((article) => (
        <NewsItem key={article.id ?? article.headline} article={article} onTickerClick={onTickerClick} universe={universe} />
      ))}
    </div>
  );
}

function NewsItem({ article, onTickerClick, universe, compact }: { article: NewsArticle; onTickerClick?: (t: string) => void; universe?: string[]; compact?: boolean }) {
  const universeSymbols = article.symbols.filter((s) => !universe || universe.includes(s));
  return (
    <div style={{ padding: "8px 12px", borderBottom: "1px solid var(--border-soft)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 10 }}>
        <span style={{ color: "var(--green)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>{article.source}</span>
        <span style={{ color: "var(--text-muted)" }}>{fmtRelativeTime(article.created_at)}</span>
      </div>
      {article.url ? (
        <a href={article.url} target="_blank" rel="noopener noreferrer" style={{ color: "var(--text-primary)", fontSize: 11, lineHeight: 1.45, fontWeight: 500, textDecoration: "none", display: "block" }}>
          {article.headline}
        </a>
      ) : (
        <div style={{ fontSize: 11, lineHeight: 1.45, fontWeight: 500 }}>{article.headline}</div>
      )}
      {!compact && article.summary && (
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4, lineHeight: 1.5 }}>
          {article.summary.length > 200 ? article.summary.slice(0, 200) + "..." : article.summary}
        </div>
      )}
      {universeSymbols.length > 0 && onTickerClick && (
        <div style={{ display: "flex", gap: 4, marginTop: 5, flexWrap: "wrap" }}>
          {universeSymbols.slice(0, 4).map((sym) => (
            <button
              key={sym}
              onClick={() => onTickerClick(sym)}
              style={{ background: "var(--bg-row)", color: "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: 99, padding: "1px 6px", fontSize: 9, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}
            >
              {sym}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function IntelligencePanel({ ticker, intelligence, loading, onRefresh }: { ticker: string; intelligence: IntelligenceResponse | null; loading: boolean; onRefresh: () => void }) {
  if (loading && !intelligence) {
    return (
      <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>
        <div style={{ marginBottom: 8 }}>🧠 Claude is analyzing the news...</div>
        <div style={{ fontSize: 10, color: "var(--text-faint)" }}>Usually takes 5-10 seconds</div>
      </div>
    );
  }

  if (!intelligence || !intelligence.analysis) {
    return (
      <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 11, textAlign: "center" }}>
        {intelligence?.metadata && "reason" in intelligence.metadata && intelligence.metadata.reason === "no_articles"
          ? `No recent news for ${ticker} to analyze.`
          : "AI analysis unavailable."}
        <button onClick={onRefresh} style={{ display: "block", margin: "12px auto 0", background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: 4, padding: "5px 12px", fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>Try Again</button>
      </div>
    );
  }

  const a = intelligence.analysis;
  const m = intelligence.metadata as IntelligenceMeta;

  const sentimentColor = a.sentiment === "bullish" ? "var(--green)" : a.sentiment === "bearish" ? "var(--red)" : "var(--text-muted)";
  const sentimentBg = a.sentiment === "bullish" ? "var(--green-bg)" : a.sentiment === "bearish" ? "var(--red-bg)" : "var(--bg-row)";
  const impactColor = a.impact === "high" ? "var(--red)" : a.impact === "medium" ? "var(--amber)" : "var(--text-muted)";
  const impactBg = a.impact === "high" ? "var(--red-bg)" : a.impact === "medium" ? "rgba(251,191,36,0.1)" : "var(--bg-row)";

  return (
    <>
      <div className="detail-section">
        <div className="detail-section-header">
          <span>AI Analysis · Claude Haiku</span>
          <button onClick={onRefresh} disabled={loading} style={{ background: "var(--bg-row)", color: "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: 3, padding: "2px 8px", fontSize: 9, cursor: loading ? "not-allowed" : "pointer", fontFamily: "inherit", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            {loading ? "..." : "↻ Refresh"}
          </button>
        </div>

        {/* Sentiment + Impact badges */}
        <div style={{ padding: "10px 12px", display: "flex", gap: 6, flexWrap: "wrap", borderBottom: "1px solid var(--border-soft)" }}>
          <span style={{ background: sentimentBg, color: sentimentColor, padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", border: `1px solid ${sentimentColor}` }}>
            {a.sentiment} · {(a.sentiment_confidence * 100).toFixed(0)}%
          </span>
          <span style={{ background: impactBg, color: impactColor, padding: "4px 10px", borderRadius: 4, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", border: `1px solid ${impactColor}` }}>
            Impact: {a.impact}
          </span>
        </div>

        {/* Summary */}
        <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border-soft)" }}>
          <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 4 }}>Summary</div>
          <div style={{ fontSize: 11, lineHeight: 1.55, color: "var(--text-primary)" }}>{a.summary}</div>
        </div>

        {/* Key events */}
        {a.key_events && a.key_events.length > 0 && (
          <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border-soft)" }}>
            <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 6 }}>Key Events</div>
            <ul style={{ paddingLeft: 16, fontSize: 10, lineHeight: 1.5, color: "var(--text-secondary)", margin: 0 }}>
              {a.key_events.map((event, i) => (
                <li key={i} style={{ marginBottom: 3 }}>{event}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Trade rationale */}
        {a.trade_rationale && (
          <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border-soft)" }}>
            <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 4 }}>Implications</div>
            <div style={{ fontSize: 10, lineHeight: 1.55, color: "var(--text-secondary)" }}>{a.trade_rationale}</div>
          </div>
        )}

        {/* Risk flags */}
        {a.risk_flags && a.risk_flags.length > 0 && (
          <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border-soft)" }}>
            <div style={{ fontSize: 9, color: "var(--amber)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 6 }}>⚠ Risk Flags</div>
            <ul style={{ paddingLeft: 16, fontSize: 10, lineHeight: 1.5, color: "var(--text-secondary)", margin: 0 }}>
              {a.risk_flags.map((flag, i) => (
                <li key={i} style={{ marginBottom: 3 }}>{flag}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Disclaimer */}
        <div style={{ padding: "8px 12px", background: "rgba(251,191,36,0.05)", fontSize: 9, color: "var(--amber)", lineHeight: 1.5 }}>
          ⚠ AI-generated analysis from {intelligence.n_articles} articles. NOT financial advice. Verify before acting.
        </div>

        {/* Metadata */}
        <div style={{ padding: "6px 12px", fontSize: 9, color: "var(--text-faint)", display: "flex", justifyContent: "space-between" }}>
          <span>{m.cached ? "📋 Cached" : "🆕 Fresh"} · {fmtRelativeTime(m.generated_at)}</span>
          <span>{m.input_tokens + m.output_tokens} tokens · ${m.estimated_cost?.toFixed(5) ?? "—"}</span>
        </div>
      </div>
    </>
  );
}

function OrderModal({ ticker, side, quote, buyingPower, onClose, onSubmit }: { ticker: string; side: "buy" | "sell"; quote?: Quote; buyingPower?: number; onClose: () => void; onSubmit: (qty: number) => void }) {
  const [qty, setQty] = useState(1);
  const refPrice = side === "buy" ? quote?.ask_price ?? quote?.mid_price : quote?.bid_price ?? quote?.mid_price;
  const estimatedNotional = (refPrice ?? 0) * qty;
  const exceedsQty = qty > MAX_QTY_PER_ORDER;
  const exceedsNotional = estimatedNotional > MAX_NOTIONAL_PER_ORDER;
  const exceedsBP = side === "buy" && buyingPower != null && estimatedNotional > buyingPower;
  const cannotSubmit = qty < 1 || exceedsQty || exceedsNotional || exceedsBP || !refPrice;

  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 999 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: 8, padding: 0, width: 380, maxWidth: "90vw", fontSize: 12 }}>
        <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700 }}>
              <span style={{ color: side === "buy" ? "var(--green)" : "var(--red)" }}>{side.toUpperCase()}</span> {ticker}
            </div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>Paper trading · market order · day</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 18 }}>×</button>
        </div>
        <div style={{ padding: 18 }}>
          <div style={{ background: "rgba(96, 165, 250, 0.08)", border: "1px solid rgba(96, 165, 250, 0.2)", borderRadius: 4, padding: "8px 10px", fontSize: 10, color: "var(--text-secondary)", marginBottom: 16 }}>
            <strong style={{ color: "var(--blue)" }}>PAPER ACCOUNT</strong> — no real money. Orders submitted on weekends queue for the next market open.
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 16, fontSize: 11 }}>
            <div><div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Bid</div><div style={{ fontWeight: 600 }}>{fmtPrice(quote?.bid_price)}</div></div>
            <div><div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Ask</div><div style={{ fontWeight: 600 }}>{fmtPrice(quote?.ask_price)}</div></div>
            <div><div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Mid</div><div style={{ fontWeight: 600 }}>{fmtPrice(quote?.mid_price)}</div></div>
            <div><div style={{ color: "var(--text-muted)", fontSize: 9, textTransform: "uppercase" }}>Reference {side === "buy" ? "(ask)" : "(bid)"}</div><div style={{ fontWeight: 600 }}>{fmtPrice(refPrice)}</div></div>
          </div>
          <div style={{ marginBottom: 14 }}>
            <label style={{ display: "block", color: "var(--text-muted)", fontSize: 10, textTransform: "uppercase", marginBottom: 4 }}>Shares (1-{MAX_QTY_PER_ORDER})</label>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input type="number" min={1} max={MAX_QTY_PER_ORDER} value={qty} onChange={(e) => setQty(Math.max(1, Math.min(MAX_QTY_PER_ORDER, parseInt(e.target.value) || 1)))} style={{ flex: 1, background: "var(--bg-input)", border: "1px solid var(--border-strong)", borderRadius: 4, padding: "8px 10px", color: "var(--text-primary)", fontSize: 14, fontFamily: "inherit" }} />
              {[1, 5, 10, 25].map((n) => (
                <button key={n} onClick={() => setQty(n)} style={{ background: qty === n ? "var(--green-bg)" : "var(--bg-elevated)", color: qty === n ? "var(--green)" : "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: 3, padding: "5px 9px", fontSize: 10, cursor: "pointer", fontFamily: "inherit" }}>{n}</button>
              ))}
            </div>
          </div>
          <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: "10px 12px", fontSize: 12, marginBottom: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ color: "var(--text-muted)" }}>Estimated total</span>
            <span style={{ fontWeight: 700, fontSize: 16 }}>${estimatedNotional.toFixed(2)}</span>
          </div>
          {exceedsQty && <div style={{ color: "var(--red)", fontSize: 10, marginBottom: 8 }}>⚠ Exceeds max {MAX_QTY_PER_ORDER} shares per order</div>}
          {exceedsNotional && <div style={{ color: "var(--red)", fontSize: 10, marginBottom: 8 }}>⚠ Exceeds max ${MAX_NOTIONAL_PER_ORDER.toLocaleString()} per order</div>}
          {exceedsBP && <div style={{ color: "var(--red)", fontSize: 10, marginBottom: 8 }}>⚠ Exceeds buying power ${buyingPower?.toFixed(2)}</div>}
          {!refPrice && <div style={{ color: "var(--amber)", fontSize: 10, marginBottom: 8 }}>⚠ No live quote. Order may be rejected.</div>}
          <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
            <button onClick={onClose} style={{ flex: 1, background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: 4, padding: "10px 0", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Cancel</button>
            <button onClick={() => onSubmit(qty)} disabled={cannotSubmit} style={{ flex: 2, background: cannotSubmit ? "var(--bg-elevated)" : side === "buy" ? "var(--green)" : "var(--red)", color: cannotSubmit ? "var(--text-muted)" : "var(--bg-base)", border: "none", borderRadius: 4, padding: "10px 0", fontSize: 12, fontWeight: 700, cursor: cannotSubmit ? "not-allowed" : "pointer", fontFamily: "inherit" }}>Confirm {side.toUpperCase()} {qty} {ticker}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
