"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Legend,
} from "recharts";
import CalibrationRibbon from "./CalibrationRibbon";

const API_BASE = "http://localhost:8000";

type TradeStatus = "closed" | "open" | "pending";

type JournalRow = {
  id: number;
  ticker: string;
  signal_date: string;
  signal_proba: number;
  signal_rank: number;
  entry_price: number | null;
  entry_filled_at: string | null;
  exit_price: number | null;
  exit_filled_at: string | null;
  qty: number | null;
  live_pnl: number | null;
  live_pnl_pct: number | null;
  backtest_predict_pnl_pct: number | null;
  backtest_basket_pnl_pct: number | null;
  gap_vs_predict: number | null;
  gap_vs_basket: number | null;
  model_version: string | null;
  status: TradeStatus;
};

type ComparisonResponse = {
  rows: JournalRow[];
  n_trades: number;
  n_closed: number;
  n_open: number;
  n_pending: number;
  go_live_date?: string;
};

type EquityPoint = { date: string; equity: number; ticker?: string; trade_return?: number; return?: number };

type EquityResponse = {
  go_live_date: string | null;
  initial_capital: number;
  live: EquityPoint[];
  backtest: EquityPoint[];
  n_live_trades_closed: number;
  n_backtest_baskets: number;
};

const fmtPct = (n: number | null | undefined, digits = 2) =>
  n == null || isNaN(n) ? "—" : `${(n * 100).toFixed(digits)}%`;
const fmtSigned = (n: number | null | undefined, digits = 2) =>
  n == null || isNaN(n) ? "—" : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(digits)}%`;
const fmtMoney = (n: number | null | undefined) =>
  n == null || isNaN(n) ? "—" : `$${n.toFixed(2)}`;

type SortKey = keyof JournalRow | "ticker";

export default function JournalPanel() {
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [equity, setEquity] = useState<EquityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("signal_date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [cmpRes, eqRes] = await Promise.all([
        fetch(`${API_BASE}/api/journal/comparison`),
        fetch(`${API_BASE}/api/journal/equity`),
      ]);
      if (!cmpRes.ok || !eqRes.ok) {
        throw new Error(`HTTP ${cmpRes.status}/${eqRes.status}`);
      }
      setComparison(await cmpRes.json());
      setEquity(await eqRes.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const sortedRows = useMemo(() => {
    if (!comparison) return [];
    const rows = [...comparison.rows];
    rows.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      const as = String(av);
      const bs = String(bv);
      return sortDir === "asc" ? as.localeCompare(bs) : bs.localeCompare(as);
    });
    return rows;
  }, [comparison, sortKey, sortDir]);

  const closedRows = useMemo(
    () => sortedRows.filter((r) => r.status === "closed" && r.live_pnl_pct != null),
    [sortedRows],
  );

  const aggregates = useMemo(() => {
    if (!comparison) return null;
    const closed = comparison.rows.filter((r) => r.status === "closed" && r.live_pnl_pct != null);
    if (closed.length === 0) return null;
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    const liveVals = closed.map((r) => r.live_pnl_pct!);
    const predictVals = closed
      .filter((r) => r.backtest_predict_pnl_pct != null)
      .map((r) => r.backtest_predict_pnl_pct!);
    const basketVals = closed
      .filter((r) => r.backtest_basket_pnl_pct != null)
      .map((r) => r.backtest_basket_pnl_pct!);
    return {
      n: closed.length,
      meanLive: mean(liveVals),
      meanPredict: predictVals.length ? mean(predictVals) : null,
      meanBasket: basketVals.length ? mean(basketVals) : null,
      hitRate: liveVals.filter((v) => v > 0).length / liveVals.length,
    };
  }, [comparison]);

  if (loading && !comparison) {
    return <div style={{ padding: 20, color: "var(--text-muted)" }}>Loading journal…</div>;
  }

  if (error) {
    return (
      <div style={{ padding: 20, color: "var(--red)" }}>
        Error: {error}
        <button onClick={fetchAll} style={btn}>Retry</button>
      </div>
    );
  }

  if (!comparison || !equity) {
    return <div style={{ padding: 20, color: "var(--text-muted)" }}>No journal data.</div>;
  }

  const totalTrades = comparison.n_trades;
  const goLive = equity.go_live_date ?? comparison.go_live_date ?? "—";

  return (
    <div style={{ padding: "12px 16px", color: "var(--text-primary)", overflow: "auto", height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em", margin: 0 }}>
            Trade Journal — Live vs Backtest
          </h1>
          <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: 4 }}>
            Go-live: {goLive} · {totalTrades} trade{totalTrades === 1 ? "" : "s"} (
            {comparison.n_closed} closed · {comparison.n_open} open · {comparison.n_pending} pending)
          </div>
        </div>
        <button onClick={fetchAll} disabled={loading} style={btn}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {/* === Headline stats === */}
      <Section title="Headline">
        {aggregates ? (
          <StatGrid>
            <Stat label="Closed Trades" value={aggregates.n.toString()} sub="" />
            <Stat
              label="Mean Live Return"
              value={fmtSigned(aggregates.meanLive)}
              sub=""
              accent={aggregates.meanLive >= 0 ? "up" : "dn"}
            />
            <Stat
              label="Mean Backtest (predict)"
              value={fmtSigned(aggregates.meanPredict)}
              sub="per-trade fwd_return_5d"
              accent={(aggregates.meanPredict ?? 0) >= 0 ? "up" : "dn"}
            />
            <Stat
              label="Live Hit Rate"
              value={fmtPct(aggregates.hitRate, 1)}
              sub=""
              accent={aggregates.hitRate >= 0.5 ? "up" : "dn"}
            />
          </StatGrid>
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            No closed trades yet. Headline stats activate after the first exit fills.
          </div>
        )}
      </Section>

      {/* === Trade table === */}
      <Section title={`Trade Table (${totalTrades})`}>
        {totalTrades === 0 ? (
          <Empty msg="No trades on file." />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <div style={tableHead}>
              <SortableHeader label="Signal Date" k="signal_date" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Ticker" k="ticker" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Rank" k="signal_rank" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Proba" k="signal_proba" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Entry" k="entry_price" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Exit" k="exit_price" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Live %" k="live_pnl_pct" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="BT-Predict %" k="backtest_predict_pnl_pct" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="BT-Basket %" k="backtest_basket_pnl_pct" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <SortableHeader label="Gap vs Predict" k="gap_vs_predict" sortKey={sortKey} sortDir={sortDir} onChange={(k, d) => { setSortKey(k); setSortDir(d); }} />
              <span>Status</span>
            </div>
            {sortedRows.map((r) => (
              <div key={r.id} style={tableRow}>
                <span>{r.signal_date}</span>
                <span style={{ fontWeight: 600 }}>{r.ticker}</span>
                <span>#{r.signal_rank}</span>
                <span style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ color: "var(--text-muted)" }}>{r.signal_proba.toFixed(3)}</span>
                  <CalibrationRibbon proba={r.signal_proba} size="xs" />
                </span>
                <span>{fmtMoney(r.entry_price)}</span>
                <span>{fmtMoney(r.exit_price)}</span>
                <span style={{ fontWeight: 600, color: pctColor(r.live_pnl_pct) }}>{fmtSigned(r.live_pnl_pct)}</span>
                <span style={{ color: pctColor(r.backtest_predict_pnl_pct) }}>{fmtSigned(r.backtest_predict_pnl_pct)}</span>
                <span style={{ color: pctColor(r.backtest_basket_pnl_pct) }}>{fmtSigned(r.backtest_basket_pnl_pct)}</span>
                <span style={{ fontWeight: 600, color: pctColor(r.gap_vs_predict) }}>{fmtSigned(r.gap_vs_predict)}</span>
                <span><StatusBadge status={r.status} /></span>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* === Per-trade scatter === */}
      <Section title="Live vs Backtest-Predict (per trade)">
        {closedRows.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            Scatter activates once at least one trade closes. Each point = one live trade,
            with x = backtest-predict % and y = realized live %. Points on the 45° line mean
            live matched the backtest.
          </div>
        ) : (
          <div style={{ height: 320, background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: 12 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
                <CartesianGrid stroke="var(--border-soft)" strokeDasharray="2 2" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Backtest %"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 10, fill: "var(--text-muted)" }}
                  stroke="var(--text-faint)"
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "Backtest-Predict %", position: "insideBottom", offset: -10, fill: "var(--text-muted)", fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Live %"
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 10, fill: "var(--text-muted)" }}
                  stroke="var(--text-faint)"
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  label={{ value: "Live %", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ background: "var(--bg-panel)", border: "1px solid var(--border)", fontSize: 11 }}
                  formatter={(value: number, name: string) => [`${(value * 100).toFixed(2)}%`, name]}
                  labelFormatter={() => ""}
                />
                <ReferenceLine
                  segment={[{ x: -1, y: -1 }, { x: 1, y: 1 }]}
                  stroke="var(--text-faint)"
                  strokeDasharray="3 3"
                  ifOverflow="extendDomain"
                />
                <ReferenceLine x={0} stroke="var(--border)" />
                <ReferenceLine y={0} stroke="var(--border)" />
                <Scatter
                  data={closedRows
                    .filter((r) => r.backtest_predict_pnl_pct != null)
                    .map((r) => ({ x: r.backtest_predict_pnl_pct, y: r.live_pnl_pct, ticker: r.ticker, date: r.signal_date }))}
                  fill="var(--cyan)"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* === Equity overlay === */}
      <Section title="Cumulative Equity — Live vs Backtest Reference">
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
          Both curves rebased to ${equity.initial_capital.toLocaleString()} at go-live ({goLive}).
          Backtest curve = simulate_strategy() top-2, non-overlapping, 10bps round-trip.
          Live curve = compounded closed-trade returns.
        </div>
        {equity.live.length === 0 && equity.backtest.length === 0 ? (
          <Empty msg="Both curves empty — need closed trades and at least one rebalance window of backtest reference." />
        ) : (
          <div style={{ height: 320, background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: 12 }}>
            <EquityChart live={equity.live} backtest={equity.backtest} initial={equity.initial_capital} />
          </div>
        )}
        <div style={{ display: "flex", gap: 16, fontSize: 10, color: "var(--text-muted)", marginTop: 8 }}>
          <span>Live points: {equity.n_live_trades_closed}</span>
          <span>Backtest baskets: {equity.n_backtest_baskets}</span>
        </div>
      </Section>
    </div>
  );
}

function EquityChart({ live, backtest, initial }: { live: EquityPoint[]; backtest: EquityPoint[]; initial: number }) {
  const merged = useMemo(() => {
    const byDate = new Map<string, { date: string; live?: number; backtest?: number }>();
    for (const p of backtest) {
      byDate.set(p.date, { date: p.date, backtest: p.equity });
    }
    for (const p of live) {
      const existing = byDate.get(p.date) ?? { date: p.date };
      existing.live = p.equity;
      byDate.set(p.date, existing);
    }
    return Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
  }, [live, backtest]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={merged} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid stroke="var(--border-soft)" strokeDasharray="2 2" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-muted)" }} stroke="var(--text-faint)" />
        <YAxis
          tick={{ fontSize: 10, fill: "var(--text-muted)" }}
          stroke="var(--text-faint)"
          tickFormatter={(v: number) => `$${v.toLocaleString()}`}
          domain={["auto", "auto"]}
        />
        <Tooltip
          contentStyle={{ background: "var(--bg-panel)", border: "1px solid var(--border)", fontSize: 11 }}
          formatter={(v: number) => [`$${v.toFixed(2)}`, ""]}
        />
        <Legend wrapperStyle={{ fontSize: 10, color: "var(--text-muted)" }} />
        <ReferenceLine y={initial} stroke="var(--text-faint)" strokeDasharray="3 3" />
        <Line type="monotone" dataKey="backtest" stroke="var(--purple)" strokeWidth={2} dot={false} name="Backtest reference" connectNulls />
        <Line type="monotone" dataKey="live" stroke="var(--cyan)" strokeWidth={2} dot={{ r: 3 }} name="Live (paper)" connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

function SortableHeader({
  label,
  k,
  sortKey,
  sortDir,
  onChange,
}: {
  label: string;
  k: SortKey;
  sortKey: SortKey;
  sortDir: "asc" | "desc";
  onChange: (k: SortKey, d: "asc" | "desc") => void;
}) {
  const active = sortKey === k;
  return (
    <span
      style={{ cursor: "pointer", userSelect: "none", color: active ? "var(--text-secondary)" : "var(--text-muted)" }}
      onClick={() => {
        if (active) onChange(k, sortDir === "asc" ? "desc" : "asc");
        else onChange(k, "desc");
      }}
    >
      {label}{active ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
    </span>
  );
}

function StatusBadge({ status }: { status: TradeStatus }) {
  const colors = {
    closed: { bg: "var(--green-bg)", fg: "var(--green)" },
    open: { bg: "var(--green-bg)", fg: "var(--cyan)" },
    pending: { bg: "var(--bg-row)", fg: "var(--text-muted)" },
  } as const;
  const { bg, fg } = colors[status];
  return (
    <span style={{ background: bg, color: fg, padding: "1px 6px", borderRadius: 3, fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>
      {status}
    </span>
  );
}

function pctColor(n: number | null | undefined) {
  if (n == null || isNaN(n)) return "var(--text-muted)";
  return n >= 0 ? "var(--green)" : "var(--red)";
}

const btn: React.CSSProperties = {
  background: "var(--bg-elevated)",
  color: "var(--text-secondary)",
  border: "1px solid var(--border)",
  borderRadius: 4,
  padding: "6px 14px",
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "inherit",
};

const tableHead: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "90px 60px 50px 92px 70px 70px 80px 90px 90px 100px 70px",
  gap: 8,
  padding: "6px 0",
  color: "var(--text-muted)",
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  fontWeight: 700,
  borderBottom: "1px solid var(--border-soft)",
};

const tableRow: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "90px 60px 50px 92px 70px 70px 80px 90px 90px 100px 70px",
  gap: 8,
  padding: "6px 0",
  fontSize: 11,
  borderBottom: "1px solid var(--border-soft)",
  alignItems: "center",
};

function StatGrid({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, background: "var(--border)", borderRadius: 4, overflow: "hidden" }}>
      {children}
    </div>
  );
}

function Stat({ label, value, sub, accent }: { label: string; value: string; sub: string; accent?: "up" | "dn" | "muted" }) {
  return (
    <div style={{ background: "var(--bg-panel)", padding: "10px 12px" }}>
      <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.01em", color: accent === "up" ? "var(--green)" : accent === "dn" ? "var(--red)" : "var(--text-primary)" }}>{value}</div>
      {sub && <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 18, background: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: 4, overflow: "hidden" }}>
      <div style={{ padding: "8px 12px", background: "var(--bg-elevated)", borderBottom: "1px solid var(--border)", fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700 }}>
        {title}
      </div>
      <div style={{ padding: 12 }}>{children}</div>
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "4px 0" }}>{msg}</div>;
}
