"use client";

import { useCallback, useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";
const REFRESH_MS = 15_000;

type Check = {
  label: string;
  status: "green" | "amber" | "red";
  value: string;
};

type LatestPick = {
  rank: number;
  ticker: string;
  y_proba: number;
  tradable: boolean;
  reasons: string[];
};

type LastRun = {
  id: number;
  run_type: string;
  run_at: string;
  n_signals_evaluated: number | null;
  n_orders_placed: number | null;
  notes: string | null;
} | null;

type Health = {
  overall_status: "green" | "amber" | "red";
  checks: Check[];
  latest_prediction_date: string | null;
  days_since_predictions: number | null;
  config: {
    min_signal_proba: number;
    max_signal_age_days: number;
    max_concurrent_positions: number;
    default_holding_days: number;
  };
  latest_picks: LatestPick[];
  last_entry_run: LastRun;
  next_entry_run_eta: string | null;
};

const STATUS_COLOR: Record<"green" | "amber" | "red", string> = {
  green: "var(--green)",
  amber: "var(--amber)",
  red: "var(--red)",
};

const STATUS_BG: Record<"green" | "amber" | "red", string> = {
  green: "var(--green-bg)",
  amber: "rgba(251, 191, 36, 0.12)",
  red: "var(--red-bg)",
};

export default function AutotraderHealth() {
  const [data, setData] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMsg, setRefreshMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/autotrader/health`);
      if (!r.ok) {
        setErr(`HTTP ${r.status}`);
        return;
      }
      setData(await r.json());
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "fetch failed");
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  async function refreshPredictions() {
    setRefreshing(true);
    setRefreshMsg(null);
    try {
      const r = await fetch(`${API_BASE}/api/autotrader/predict-cycle`, { method: "POST" });
      const j = await r.json().catch(() => ({}));
      if (r.ok) {
        setRefreshMsg("Predictions regenerated. Refreshing health…");
        setTimeout(load, 800);
      } else {
        setRefreshMsg(`Failed: ${j.detail ?? r.status}`);
      }
    } catch (e) {
      setRefreshMsg(`Network error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRefreshing(false);
    }
  }

  if (err) {
    return (
      <div style={shell}>
        <Header overall="red" />
        <div style={{ padding: 14, color: "var(--red)", fontSize: 11 }}>Error: {err}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div style={shell}>
        <Header overall="amber" />
        <div style={{ padding: 14, color: "var(--text-muted)", fontSize: 11 }}>Loading health…</div>
      </div>
    );
  }

  const tradable = data.latest_picks.filter((p) => p.tradable);
  const blocked = data.latest_picks.filter((p) => !p.tradable);

  return (
    <div style={shell}>
      <Header overall={data.overall_status} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8, padding: 14 }}>
        {data.checks.map((c) => (
          <div key={c.label} style={{
            padding: "8px 10px",
            background: STATUS_BG[c.status],
            border: `1px solid ${c.status === "green" ? "rgba(74, 222, 128, 0.3)" : c.status === "amber" ? "rgba(251, 191, 36, 0.35)" : "rgba(248, 113, 113, 0.35)"}`,
            borderRadius: 4,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: STATUS_COLOR[c.status], display: "inline-block" }} />
              <span style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 700 }}>{c.label}</span>
            </div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", marginTop: 4 }}>{c.value}</div>
          </div>
        ))}
      </div>

      <div style={{ padding: "0 14px 12px", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button
          onClick={refreshPredictions}
          disabled={refreshing}
          style={{
            background: "var(--green-bg)",
            color: "var(--green)",
            border: "1px solid rgba(74, 222, 128, 0.4)",
            borderRadius: 4,
            padding: "6px 14px",
            fontSize: 11,
            fontWeight: 700,
            cursor: refreshing ? "wait" : "pointer",
            fontFamily: "inherit",
            opacity: refreshing ? 0.5 : 1,
          }}
          title="Triggers the same predict-cycle the scheduler runs at 9:30 ET. Pulls fresh OHLCV, runs inference on the production model, appends to predictions.parquet."
        >
          {refreshing ? "Regenerating…" : "↻ Refresh predictions now"}
        </button>
        {refreshMsg && (
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{refreshMsg}</span>
        )}
        <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-faint)" }}>
          Last predictions: {data.latest_prediction_date ?? "—"} · Next entry cycle: {fmtNextRun(data.next_entry_run_eta)}
        </span>
      </div>

      {/* Per-pick filter table */}
      <div style={{ padding: "0 14px 14px" }}>
        <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 700, marginBottom: 6 }}>
          Top picks &amp; filter outcome
        </div>
        <div style={tableHead}>
          <span>#</span>
          <span>Ticker</span>
          <span>Proba</span>
          <span>Status</span>
          <span>Reason</span>
        </div>
        {data.latest_picks.map((p) => (
          <div key={p.ticker} style={tableRow}>
            <span style={{ color: "var(--text-faint)" }}>#{p.rank}</span>
            <span style={{ fontWeight: 600 }}>{p.ticker}</span>
            <span style={{ fontFeatureSettings: "'tnum'" }}>{(p.y_proba * 100).toFixed(1)}%</span>
            <span>
              <span style={{
                background: p.tradable ? "var(--green-bg)" : "var(--bg-row)",
                color: p.tradable ? "var(--green)" : "var(--text-muted)",
                padding: "1px 7px",
                borderRadius: 3,
                fontSize: 9,
                fontWeight: 700,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}>
                {p.tradable ? "tradable" : "blocked"}
              </span>
            </span>
            <span style={{ fontSize: 10, color: p.tradable ? "var(--text-faint)" : "var(--amber)" }}>
              {p.reasons.join(" · ")}
            </span>
          </div>
        ))}
        <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: 8, lineHeight: 1.55 }}>
          Filter thresholds: min proba {data.config.min_signal_proba} · max signal age {data.config.max_signal_age_days}d ·
          max concurrent positions {data.config.max_concurrent_positions} · default hold {data.config.default_holding_days}d.
          {data.last_entry_run && (
            <>
              {" "}Last entry-cycle: <span style={{ color: "var(--text-secondary)" }}>{data.last_entry_run.run_at}</span> →
              {" "}{data.last_entry_run.n_orders_placed ?? 0} orders placed
              {data.last_entry_run.notes ? <em style={{ color: "var(--text-muted)" }}> ({data.last_entry_run.notes})</em> : ""}
            </>
          )}
          {" "}Health summary: <strong style={{ color: STATUS_COLOR[data.overall_status] }}>{data.overall_status.toUpperCase()}</strong> ·
          {" "}{tradable.length} would-trade / {blocked.length} blocked.
        </div>
      </div>
    </div>
  );
}

function Header({ overall }: { overall: "green" | "amber" | "red" }) {
  const label = overall === "green" ? "Healthy" : overall === "amber" ? "Degraded" : "Failing";
  return (
    <div style={{
      padding: "10px 14px",
      borderBottom: "1px solid var(--border)",
      background: "var(--bg-elevated)",
      display: "flex",
      alignItems: "center",
      gap: 10,
    }}>
      <span style={{ width: 10, height: 10, borderRadius: "50%", background: STATUS_COLOR[overall], display: "inline-block", boxShadow: `0 0 8px ${STATUS_COLOR[overall]}` }} />
      <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>Autotrader Health</span>
      <span style={{
        fontSize: 9, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
        color: STATUS_COLOR[overall], padding: "2px 8px", borderRadius: 3, background: STATUS_BG[overall],
      }}>{label}</span>
      <span style={{ marginLeft: "auto", fontSize: 9, color: "var(--text-faint)" }}>refreshes every 15s</span>
    </div>
  );
}

function fmtNextRun(s: string | null): string {
  if (!s) return "—";
  // ISO with tz; just show local date+time without seconds
  try {
    const d = new Date(s);
    return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return s;
  }
}

const shell: React.CSSProperties = {
  background: "var(--bg-panel)",
  border: "1px solid var(--border)",
  borderRadius: 4,
  marginBottom: 14,
  fontFeatureSettings: "'tnum'",
};

const tableHead: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "32px 60px 60px 100px 1fr",
  gap: 8,
  padding: "5px 0",
  fontSize: 9,
  color: "var(--text-muted)",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  fontWeight: 700,
  borderBottom: "1px solid var(--border-soft)",
};

const tableRow: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "32px 60px 60px 100px 1fr",
  gap: 8,
  padding: "6px 0",
  fontSize: 11,
  borderBottom: "1px solid var(--border-soft)",
  alignItems: "center",
};
