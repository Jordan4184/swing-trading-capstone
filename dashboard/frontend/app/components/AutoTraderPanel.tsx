"use client";

import { useEffect, useState, useCallback } from "react";

const API_BASE = "http://localhost:8000";

type Capacity = {
  max_positions: number;
  open_count: number;
  pending_count: number;
  available_slots: number;
  open_tickers: string[];
  pending_tickers: string[];
};

type Stats = {
  n_closed_trades: number;
  n_wins: number;
  n_losses: number;
  hit_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_pnl_pct: number;
  best_trade: number;
  worst_trade: number;
  n_open_positions: number;
  n_pending_orders: number;
};

type OpenPos = {
  ticker: string;
  qty: number;
  entry_price: number;
  entry_filled_at: string;
  target_exit_date: string;
  signal_proba: number;
};

type Portfolio = {
  capacity: Capacity;
  stats: Stats;
  exits_due_today: number;
  exits_due_tickers: string[];
  signal_age_days: number | null;
  signal_is_fresh: boolean;
  open_positions_detail: OpenPos[];
};

type SchedulerStatus = {
  running: boolean;
  enabled_via_env: boolean;
  jobs?: { id: string; name: string; next_run_time: string | null }[];
};

type Config = {
  max_concurrent_positions: number;
  min_signal_proba: number;
  max_position_pct: number;
  default_holding_days: number;
};

type StatusResponse = {
  scheduler: SchedulerStatus;
  portfolio: Portfolio;
  config: Config;
};

type Trade = {
  id: number;
  ticker: string;
  signal_date: string;
  signal_proba: number;
  signal_rank: number;
  intent_at: string;
  alpaca_order_id: string | null;
  submission_status: string;
  submission_error: string | null;
  entry_price: number | null;
  entry_filled_at: string | null;
  qty: number | null;
  target_exit_date: string | null;
  exit_price: number | null;
  exit_filled_at: string | null;
  exit_reason: string | null;
  pnl: number | null;
  pnl_pct: number | null;
};

type TradesResponse = {
  recent: Trade[];
  open_positions: Trade[];
  performance: Stats;
};

type Run = {
  id: number;
  run_type: string;
  run_at: string;
  n_signals_evaluated: number | null;
  n_orders_placed: number | null;
  n_positions_exited: number | null;
  notes: string | null;
  error: string | null;
};

type RunsResponse = { runs: Run[] };

type CycleResult = {
  dry_run: boolean;
  started_at: string;
  signals_evaluated?: number;
  signals_selected?: number;
  orders_attempted?: number;
  orders_succeeded?: number;
  positions_due?: number;
  exits_attempted?: number;
  exits_succeeded?: number;
  trades?: { ticker: string; qty?: number; would_trade?: boolean }[];
  exits?: { ticker: string; qty: number; would_exit?: boolean }[];
  errors?: string[];
};

const fmtMoney = (n: number) =>
  n >= 1000 || n <= -1000 ? `$${(n / 1000).toFixed(2)}k` : `$${n.toFixed(2)}`;
const fmtPct = (n: number) => `${(n * 100).toFixed(2)}%`;

export default function AutoTraderPanel() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [trades, setTrades] = useState<TradesResponse | null>(null);
  const [runs, setRuns] = useState<RunsResponse | null>(null);
  const [tab, setTab] = useState<"overview" | "trades" | "runs">("overview");
  const [cycleResult, setCycleResult] = useState<CycleResult | null>(null);
  const [cycleLoading, setCycleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(async () => {
    try {
      const [s, t, r] = await Promise.all([
        fetch(`${API_BASE}/api/autotrader/status`),
        fetch(`${API_BASE}/api/autotrader/trades`),
        fetch(`${API_BASE}/api/autotrader/runs`),
      ]);
      if (s.ok) setStatus(await s.json());
      if (t.ok) setTrades(await t.json());
      if (r.ok) setRuns(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15_000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  async function runCycle(kind: "entry" | "exit", dryRun: boolean) {
    setCycleLoading(true);
    setCycleResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/autotrader/${kind}-cycle?dry_run=${dryRun}`, {
        method: "POST",
      });
      if (res.ok) {
        setCycleResult(await res.json());
        fetchAll();
      } else {
        const e = await res.json();
        setError(e.detail || "Cycle failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setCycleLoading(false);
    }
  }

  if (error) {
    return (
      <div style={{ padding: 20, color: "var(--red)" }}>
        Error: {error}
        <button onClick={() => { setError(null); fetchAll(); }} style={btnStyle}>Retry</button>
      </div>
    );
  }

  if (!status || !trades || !runs) {
    return <div style={{ padding: 20, color: "var(--text-muted)" }}>Loading auto-trader state...</div>;
  }

  const p = status.portfolio;
  const sched = status.scheduler;
  const stats = p.stats;

  return (
    <div style={{ padding: "12px 16px", color: "var(--text-primary)", overflow: "auto", height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em", margin: 0 }}>Autonomous Trader</h1>
        <span style={{ fontSize: 10, color: "var(--text-faint)" }}>refreshes 15s</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 1, background: "var(--border)", borderRadius: 4, overflow: "hidden", marginBottom: 16 }}>
        <StatusCell label="Scheduler" value={sched.running ? "RUNNING" : sched.enabled_via_env ? "ENABLED" : "DISABLED"} accent={sched.running ? "up" : "muted"} />
        <StatusCell label="Signal Age" value={p.signal_age_days != null ? `${p.signal_age_days}d` : "—"} accent={p.signal_is_fresh ? "up" : "dn"} />
        <StatusCell label="Signal Status" value={p.signal_is_fresh ? "FRESH" : "STALE"} accent={p.signal_is_fresh ? "up" : "dn"} />
        <StatusCell label="Slots" value={`${p.capacity.open_count + p.capacity.pending_count}/${p.capacity.max_positions}`} accent={p.capacity.available_slots > 0 ? "up" : "dn"} />
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button onClick={() => runCycle("entry", true)} disabled={cycleLoading} style={btnStyle}>{cycleLoading ? "..." : "Entry Cycle (dry)"}</button>
        <button onClick={() => runCycle("entry", false)} disabled={cycleLoading} style={{ ...btnStyle, borderColor: "var(--green)", color: "var(--green)" }}>{cycleLoading ? "..." : "Entry Cycle (LIVE)"}</button>
        <button onClick={() => runCycle("exit", true)} disabled={cycleLoading} style={btnStyle}>Exit Cycle (dry)</button>
        <button onClick={() => runCycle("exit", false)} disabled={cycleLoading} style={{ ...btnStyle, borderColor: "var(--red)", color: "var(--red)" }}>Exit Cycle (LIVE)</button>
      </div>

      {cycleResult && (
        <div style={{ background: cycleResult.errors && cycleResult.errors.length > 0 ? "rgba(248,113,113,0.08)" : "var(--green-bg)", border: `1px solid ${cycleResult.errors && cycleResult.errors.length > 0 ? "var(--red)" : "var(--green)"}`, borderRadius: 4, padding: "10px 12px", marginBottom: 16, fontSize: 11 }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>{cycleResult.dry_run ? "DRY RUN" : "LIVE CYCLE"} · {cycleResult.started_at?.slice(11, 19)} ET</div>
          <div style={{ color: "var(--text-secondary)" }}>
            {cycleResult.signals_evaluated != null && <>Signals: {cycleResult.signals_evaluated} evaluated, {cycleResult.signals_selected ?? 0} selected · </>}
            {cycleResult.orders_succeeded != null && <>Orders: {cycleResult.orders_succeeded}/{cycleResult.orders_attempted ?? 0} placed</>}
            {cycleResult.exits_succeeded != null && <>Exits: {cycleResult.exits_succeeded}/{cycleResult.exits_attempted ?? 0} completed</>}
          </div>
          {cycleResult.errors && cycleResult.errors.length > 0 && <div style={{ marginTop: 6, color: "var(--red)" }}>Errors: {cycleResult.errors.join("; ")}</div>}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, background: "var(--border)", borderRadius: 4, overflow: "hidden", marginBottom: 16 }}>
        <StatCell label="Closed Trades" value={stats.n_closed_trades.toString()} sub="" />
        <StatCell label="Hit Rate" value={fmtPct(stats.hit_rate)} sub={`${stats.n_wins}W ${stats.n_losses}L`} />
        <StatCell label="Total P&L" value={fmtMoney(stats.total_pnl)} sub={`avg ${fmtMoney(stats.avg_pnl)}`} accent={stats.total_pnl >= 0 ? "up" : "dn"} />
        <StatCell label="Best / Worst" value={`${fmtMoney(stats.best_trade)} / ${fmtMoney(stats.worst_trade)}`} sub="" />
      </div>

      <div style={{ display: "flex", borderBottom: "1px solid var(--border)", marginBottom: 12 }}>
        {(["overview", "trades", "runs"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)} style={tabStyle(tab === t)}>
            {t === "overview" ? "Overview" : t === "trades" ? `Trades (${trades.recent.length})` : `Runs (${runs.runs.length})`}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <>
          <Section title="Capacity">
            <div style={{ fontSize: 11, lineHeight: 1.7 }}>
              <Row label="Open positions" value={`${p.capacity.open_count} (${p.capacity.open_tickers.join(", ") || "none"})`} />
              <Row label="Pending orders" value={`${p.capacity.pending_count} (${p.capacity.pending_tickers.join(", ") || "none"})`} />
              <Row label="Available slots" value={`${p.capacity.available_slots} of ${p.capacity.max_positions}`} />
              <Row label="Exits due today" value={`${p.exits_due_today} (${p.exits_due_tickers.join(", ") || "none"})`} />
            </div>
          </Section>

          <Section title={`Open Positions (${p.open_positions_detail.length})`}>
            {p.open_positions_detail.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: 11 }}>No open positions.</div>
            ) : (
              <div style={{ fontSize: 11 }}>
                <div style={{ display: "grid", gridTemplateColumns: "60px 50px 80px 100px 80px 50px", gap: 8, padding: "6px 0", color: "var(--text-muted)", fontSize: 10, textTransform: "uppercase", borderBottom: "1px solid var(--border-soft)" }}>
                  <span>Ticker</span><span>Qty</span><span>Entry</span><span>Filled</span><span>Exit Target</span><span>Proba</span>
                </div>
                {p.open_positions_detail.map((pos) => (
                  <div key={pos.ticker} style={{ display: "grid", gridTemplateColumns: "60px 50px 80px 100px 80px 50px", gap: 8, padding: "6px 0", borderBottom: "1px solid var(--border-soft)" }}>
                    <span style={{ fontWeight: 600 }}>{pos.ticker}</span>
                    <span>{pos.qty}</span>
                    <span>${pos.entry_price?.toFixed(2)}</span>
                    <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{pos.entry_filled_at?.slice(0, 10) ?? "—"}</span>
                    <span>{pos.target_exit_date}</span>
                    <span>{pos.signal_proba?.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
          </Section>

          <Section title="Configuration">
            <div style={{ fontSize: 11, lineHeight: 1.7 }}>
              <Row label="Max concurrent positions" value={status.config.max_concurrent_positions.toString()} />
              <Row label="Min signal probability" value={status.config.min_signal_proba.toFixed(2)} />
              <Row label="Max position size (%)" value={`${(status.config.max_position_pct * 100).toFixed(0)}%`} />
              <Row label="Default holding days" value={status.config.default_holding_days.toString()} />
              <Row label="Scheduled jobs" value={sched.jobs?.length.toString() ?? "0"} />
              {sched.jobs?.map((j) => (
                <Row key={j.id} label={`  → ${j.name}`} value={j.next_run_time?.slice(0, 19) ?? "not scheduled"} />
              ))}
            </div>
          </Section>
        </>
      )}

      {tab === "trades" && (
        <Section title={`Recent Trades (${trades.recent.length})`}>
          {trades.recent.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 11 }}>No trades yet.</div>
          ) : (
            <div style={{ fontSize: 11 }}>
              <div style={{ display: "grid", gridTemplateColumns: "60px 70px 60px 60px 70px 70px 80px 70px", gap: 8, padding: "6px 0", color: "var(--text-muted)", fontSize: 10, textTransform: "uppercase", borderBottom: "1px solid var(--border-soft)" }}>
                <span>Ticker</span><span>Sig Date</span><span>Proba</span><span>Qty</span><span>Entry</span><span>Exit</span><span>Status</span><span>P&L</span>
              </div>
              {trades.recent.map((t) => (
                <div key={t.id} style={{ display: "grid", gridTemplateColumns: "60px 70px 60px 60px 70px 70px 80px 70px", gap: 8, padding: "6px 0", borderBottom: "1px solid var(--border-soft)", fontSize: 10 }}>
                  <span style={{ fontWeight: 600 }}>{t.ticker}</span>
                  <span style={{ color: "var(--text-muted)" }}>{t.signal_date}</span>
                  <span>{t.signal_proba.toFixed(3)}</span>
                  <span>{t.qty ?? "—"}</span>
                  <span>{t.entry_price ? `$${t.entry_price.toFixed(2)}` : "—"}</span>
                  <span>{t.exit_price ? `$${t.exit_price.toFixed(2)}` : "—"}</span>
                  <StatusBadge status={t.submission_status} />
                  <span style={{ fontWeight: 600, color: t.pnl == null ? "var(--text-muted)" : t.pnl >= 0 ? "var(--green)" : "var(--red)" }}>
                    {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "—"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Section>
      )}

      {tab === "runs" && (
        <Section title={`Scheduler Runs (${runs.runs.length})`}>
          {runs.runs.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 11 }}>No runs yet.</div>
          ) : (
            <div style={{ fontSize: 11 }}>
              {runs.runs.map((r) => (
                <div key={r.id} style={{ padding: "8px 0", borderBottom: "1px solid var(--border-soft)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em", fontSize: 10, color: r.error ? "var(--red)" : "var(--green)" }}>{r.run_type}</span>
                    <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{r.run_at}</span>
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                    {r.notes ?? ""}
                    {r.n_signals_evaluated != null && ` · ${r.n_signals_evaluated} signals evaluated`}
                    {r.n_orders_placed != null && ` · ${r.n_orders_placed} orders placed`}
                    {r.n_positions_exited != null && ` · ${r.n_positions_exited} positions exited`}
                  </div>
                  {r.error && <div style={{ fontSize: 10, color: "var(--red)", marginTop: 3 }}>Error: {r.error}</div>}
                </div>
              ))}
            </div>
          )}
        </Section>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: "var(--bg-elevated)",
  color: "var(--text-secondary)",
  border: "1px solid var(--border)",
  borderRadius: 4,
  padding: "6px 12px",
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "inherit",
};

function tabStyle(active: boolean): React.CSSProperties {
  return {
    background: "none",
    border: "none",
    cursor: "pointer",
    padding: "8px 14px",
    fontSize: 11,
    fontWeight: 600,
    color: active ? "var(--text-primary)" : "var(--text-muted)",
    borderBottom: active ? "2px solid var(--green)" : "2px solid transparent",
    fontFamily: "inherit",
  };
}

function StatusCell({ label, value, accent }: { label: string; value: string; accent: "up" | "dn" | "muted" }) {
  return (
    <div style={{ background: "var(--bg-panel)", padding: "10px 12px" }}>
      <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: "-0.01em", color: accent === "up" ? "var(--green)" : accent === "dn" ? "var(--red)" : "var(--text-primary)" }}>{value}</div>
    </div>
  );
}

function StatCell({ label, value, sub, accent }: { label: string; value: string; sub: string; accent?: "up" | "dn" }) {
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
    <div style={{ marginBottom: 16, background: "var(--bg-panel)", border: "1px solid var(--border)", borderRadius: 4, overflow: "hidden" }}>
      <div style={{ padding: "8px 12px", background: "var(--bg-elevated)", borderBottom: "1px solid var(--border)", fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700 }}>{title}</div>
      <div style={{ padding: "8px 12px" }}>{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const color = status === "submitted" ? "var(--blue)" : status === "filled" ? "var(--green)" : status === "canceled" || status === "rejected" || status === "error" ? "var(--red)" : "var(--text-muted)";
  const bg = status === "submitted" ? "rgba(96,165,250,0.1)" : status === "filled" ? "var(--green-bg)" : status === "canceled" || status === "rejected" || status === "error" ? "var(--red-bg)" : "var(--bg-row)";
  return (
    <span style={{ background: bg, color, padding: "1px 6px", borderRadius: 2, fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em", textAlign: "center" }}>
      {status}
    </span>
  );
}
