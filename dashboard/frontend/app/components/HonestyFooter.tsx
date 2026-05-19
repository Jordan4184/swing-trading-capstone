"use client";

import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";
const REFRESH_MS = 30_000;

type HonestyData = {
  mode: string;
  trading_mode_safe: boolean;
  positions: { pending: number; open: number; closed: number };
  backtest: { v1_sharpe: number | null; v2_sharpe: number | null };
  model: { version: string | null; trained_at: string | null; days_since_train: number | null };
  predictions?: {
    latest_date: string | null;
    days_old: number | null;
    stale_threshold_days: number;
    is_stale: boolean;
  };
  universe_size: number;
};

export default function HonestyFooter() {
  const [data, setData] = useState<HonestyData | null>(null);

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const r = await fetch(`${API_BASE}/api/honesty-footer`);
        if (!r.ok) return;
        const d = await r.json();
        if (alive) setData(d);
      } catch {
        /* silent — footer is non-critical */
      }
    }
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Always render the shell so layout doesn't shift when data arrives.
  return (
    <footer
      style={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        height: 22,
        background: "var(--bg-panel)",
        borderTop: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "0 12px",
        fontSize: 10,
        color: "var(--text-muted)",
        fontFeatureSettings: "'tnum'",
        zIndex: 1000,
        whiteSpace: "nowrap",
        overflow: "hidden",
      }}
    >
      <ModeBadge mode={data?.mode ?? "—"} safe={data?.trading_mode_safe ?? false} />
      <Dot />
      <Item label="Autotrader" value={data ? `${data.positions.pending} pending · ${data.positions.open} open · ${data.positions.closed} closed` : "loading…"} />
      <Dot />
      <Item label="v1 Sharpe" value={data?.backtest.v1_sharpe?.toFixed(2) ?? "—"} />
      <Item label="v2 Sharpe" value={data?.backtest.v2_sharpe?.toFixed(2) ?? "—"} accent={(data?.backtest.v2_sharpe ?? 0) > (data?.backtest.v1_sharpe ?? 0) ? "up" : undefined} />
      <Dot />
      <Item label="Model" value={data?.model.version ?? "—"} />
      <Item
        label="Trained"
        value={data?.model.days_since_train != null ? `${data.model.days_since_train.toFixed(1)}d ago` : "—"}
        accent={(data?.model.days_since_train ?? 0) > 30 ? "warn" : undefined}
      />
      <Item
        label="Preds"
        value={data?.predictions?.days_old != null ? `${data.predictions.days_old.toFixed(1)}d ago` : "—"}
        accent={data?.predictions?.is_stale ? "warn" : undefined}
        title={
          data?.predictions
            ? `Latest prediction: ${data.predictions.latest_date ?? "—"} · autotrader rejects signals older than ${data.predictions.stale_threshold_days}d`
            : undefined
        }
      />
      <Dot />
      <Item label="Universe" value={data ? `${data.universe_size} tickers` : "—"} />
      <span style={{ marginLeft: "auto", color: "var(--text-faint)", fontSize: 9 }}>
        every claim on this page is auditable
      </span>
    </footer>
  );
}

function ModeBadge({ mode, safe }: { mode: string; safe: boolean }) {
  return (
    <span
      style={{
        background: safe ? "var(--green-bg)" : "var(--red-bg)",
        color: safe ? "var(--green)" : "var(--red)",
        padding: "2px 7px",
        borderRadius: 3,
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
      }}
    >
      {mode}
    </span>
  );
}

function Item({ label, value, accent, title }: { label: string; value: string; accent?: "up" | "warn"; title?: string }) {
  const valueColor =
    accent === "up" ? "var(--green)" : accent === "warn" ? "var(--amber)" : "var(--text-secondary)";
  return (
    <span style={{ display: "flex", gap: 4, alignItems: "baseline" }} title={title}>
      <span style={{ color: "var(--text-faint)", fontSize: 9, textTransform: "uppercase", letterSpacing: "0.04em", fontWeight: 700 }}>{label}</span>
      <span style={{ color: valueColor, fontWeight: 600 }}>{value}</span>
    </span>
  );
}

function Dot() {
  return <span style={{ color: "var(--text-faint)", opacity: 0.5 }}>·</span>;
}
