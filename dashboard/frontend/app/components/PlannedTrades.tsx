"use client";

import { useEffect, useState, useCallback } from "react";
import { cappedTradableShares, MAX_NOTIONAL_PER_ORDER } from "../constants";

const API_BASE = "http://localhost:8000";
const REFRESH_MS = 30_000;

type Pick = {
  ticker: string;
  y_proba: number;
  rank: number;
  weight_pct: number;
  rec_notional: number | null;
  rec_shares: number | null;
  last_close: number | null;
  realized_vol_20d: number | null;
  dropped_reason: string | null;
};

type RiskToday = {
  asof: string;
  regime: "risk_on" | "risk_off" | "unknown";
  regime_multiplier: number;
  gross_weight: number;
  buying_power: number | null;
  picks: Pick[];
};

interface Props {
  onTrade?: (ticker: string, shares: number) => void;
}

const MIN_PROBA_TO_PLAN = 0.55;

export default function PlannedTrades({ onTrade }: Props) {
  const [data, setData] = useState<RiskToday | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const fetchPlan = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/risk/today?top_n=5`);
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
    fetchPlan();
    const id = setInterval(fetchPlan, REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchPlan]);

  if (err) {
    return <Placeholder text={`Error: ${err}`} />;
  }
  if (!data) {
    return <Placeholder text="Loading planned trades…" />;
  }

  // Filter to picks the autoscheduler would actually consider (proba threshold,
  // non-zero weight). Dropped picks (e.g. correlation-filtered) still render
  // but disabled.
  const tradable = data.picks.filter(
    (p) => p.y_proba >= MIN_PROBA_TO_PLAN && !p.dropped_reason && (p.rec_shares ?? 0) > 0,
  );
  const dropped = data.picks.filter((p) => p.dropped_reason || (p.rec_shares ?? 0) <= 0);
  const isRiskOff = data.regime === "risk_off";

  return (
    <div className="detail-section">
      <div className="detail-section-header">
        <span>Planned Trades · next open</span>
        <span className="meta">
          {tradable.length} planned · {data.regime.replace("_", " ")}
        </span>
      </div>
      <div style={{ padding: "8px 12px 4px", fontSize: 10, color: "var(--text-muted)", lineHeight: 1.5 }}>
        What the autoscheduler will execute at <strong style={{ color: "var(--text-secondary)" }}>9:35 ET</strong> on the next trading day. Sizing already vol-adjusted &amp; regime-gated.
        {isRiskOff && (
          <span style={{ color: "var(--amber)", fontWeight: 600 }}>
            {" "}Regime: HALF-size active.
          </span>
        )}
      </div>

      {tradable.length === 0 && dropped.length === 0 ? (
        <div style={{ padding: "12px", color: "var(--text-muted)", fontSize: 11 }}>
          No picks pass the {(MIN_PROBA_TO_PLAN * 100).toFixed(0)}% probability threshold today.
        </div>
      ) : null}

      {tradable.map((p) => (
        <PlannedRow key={p.ticker} pick={p} onTrade={onTrade} />
      ))}

      {dropped.length > 0 && (
        <div style={{ borderTop: "1px dashed var(--border-soft)", padding: "6px 12px", fontSize: 9, color: "var(--text-faint)" }}>
          Filtered out: {dropped.map((p) => `${p.ticker} (${p.dropped_reason ?? "below threshold"})`).join(" · ")}
        </div>
      )}

      <div style={{ padding: "6px 12px 10px", fontSize: 9, color: "var(--text-faint)", lineHeight: 1.55 }}>
        BP {data.buying_power != null ? `$${Math.round(data.buying_power).toLocaleString()}` : "—"} · gross commitment {Math.round(data.gross_weight * 100)}%<br />
        BUY * = qty capped by per-order safety limit (${MAX_NOTIONAL_PER_ORDER.toLocaleString()} notional max).
      </div>
    </div>
  );
}

function PlannedRow({ pick, onTrade }: { pick: Pick; onTrade?: (ticker: string, shares: number) => void }) {
  const recShares = pick.rec_shares ?? 0;
  const recNotional = pick.rec_notional ?? 0;
  const { shares: buyableShares, isCapped, reason } = cappedTradableShares(recShares, pick.last_close);
  const buyableNotional = (pick.last_close ?? 0) * buyableShares;
  const canBuy = !!onTrade && buyableShares >= 1;

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "20px 56px 1fr auto",
      gap: 8,
      alignItems: "center",
      padding: "8px 12px",
      borderBottom: "1px solid var(--border-soft)",
    }}>
      <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-faint)" }}>#{pick.rank}</span>
      <div>
        <div style={{ fontWeight: 700, fontSize: 12, color: "var(--text-primary)" }}>{pick.ticker}</div>
        <div style={{ fontSize: 9, color: "var(--text-muted)" }}>
          {(pick.y_proba * 100).toFixed(1)}% conf
        </div>
      </div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", lineHeight: 1.4 }}>
        <div style={{ color: "var(--text-secondary)", fontWeight: 600, fontSize: 11 }}>
          ${Math.round(recNotional).toLocaleString()}
        </div>
        <div>
          rec {recShares} sh · {(pick.weight_pct * 100).toFixed(0)}% BP
        </div>
        {isCapped && (
          <div style={{ color: "var(--amber)", fontSize: 9, marginTop: 1 }}>
            order capped → {buyableShares} sh / ${Math.round(buyableNotional).toLocaleString()}
          </div>
        )}
      </div>
      <button
        onClick={() => canBuy && onTrade?.(pick.ticker, buyableShares)}
        disabled={!canBuy}
        style={{
          background: canBuy ? "var(--green)" : "var(--bg-row)",
          color: canBuy ? "var(--bg-base)" : "var(--text-faint)",
          border: "none",
          borderRadius: 4,
          padding: "6px 10px",
          fontSize: 10,
          fontWeight: 700,
          cursor: canBuy ? "pointer" : "not-allowed",
          fontFamily: "inherit",
          letterSpacing: "0.04em",
          opacity: canBuy ? 1 : 0.5,
          minWidth: 58,
        }}
        title={
          !canBuy
            ? `Cannot fit a single share of ${pick.ticker} under $${MAX_NOTIONAL_PER_ORDER.toLocaleString()}`
            : isCapped
              ? `Buy ${buyableShares} ${pick.ticker} — capped by ${reason}, full rec is ${recShares}`
              : `Buy ${buyableShares} ${pick.ticker} (full recommendation)`
        }
      >
        BUY {buyableShares}{isCapped ? "*" : ""}
      </button>
    </div>
  );
}

function Placeholder({ text }: { text: string }) {
  return (
    <div className="detail-section">
      <div className="detail-section-header">
        <span>Planned Trades · next open</span>
        <span className="meta">—</span>
      </div>
      <div style={{ padding: "12px", color: "var(--text-muted)", fontSize: 11 }}>{text}</div>
    </div>
  );
}
