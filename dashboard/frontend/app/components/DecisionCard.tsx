"use client";

import { useEffect, useState, useCallback } from "react";
import CalibrationRibbon from "./CalibrationRibbon";
import ConvictionLedger from "./ConvictionLedger";

const API_BASE = "http://localhost:8000";

type RiskPick = {
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
  risk_layer: "on" | "off";
  spy_close: number | null;
  spy_ma200: number | null;
  vix: number | null;
  vix_pct_rank: number | null;
  regime_multiplier: number;
  gross_weight: number;
  buying_power: number | null;
  edge_over_universe: number;
  universe_mean_proba: number;
  picks: RiskPick[];
};

type Intelligence = {
  ticker: string;
  analysis?: {
    summary?: string;
    trade_rationale?: string;
    sentiment?: "bullish" | "bearish" | "neutral";
    sentiment_confidence?: number;
    impact?: "high" | "medium" | "low";
  };
  metadata?: { cached?: boolean; elapsed_seconds?: number };
};

const fmtPct = (n: number | null | undefined, digits = 1) =>
  n == null || isNaN(n) ? "—" : `${(n * 100).toFixed(digits)}%`;
const fmtSignedPct = (n: number | null | undefined, digits = 2) =>
  n == null || isNaN(n) ? "—" : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(digits)}pp`;
const fmtMoney0 = (n: number | null | undefined) =>
  n == null || isNaN(n) ? "—" : `$${Math.round(n).toLocaleString()}`;

export default function DecisionCard() {
  const [risk, setRisk] = useState<RiskToday | null>(null);
  const [thesis, setThesis] = useState<string | null>(null);
  const [sentiment, setSentiment] = useState<{ label: string; confidence: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRisk = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/risk/today?top_n=5`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: RiskToday = await res.json();
      setRisk(data);

      // Fire-and-forget thesis pull for the top pick. First-time generation
      // takes ~4-5s via Haiku; subsequent calls are <100ms (cached).
      const top = data.picks?.[0]?.ticker;
      if (top) {
        try {
          const t = await fetch(`${API_BASE}/api/intelligence/${top}`);
          if (t.ok) {
            const tj: Intelligence = await t.json();
            const a = tj.analysis;
            setThesis(a?.summary ?? a?.trade_rationale ?? null);
            if (a?.sentiment) {
              setSentiment({
                label: a.sentiment,
                confidence: a.sentiment_confidence ?? 0,
              });
            }
          }
        } catch {
          // Thesis is non-critical
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRisk();
  }, [fetchRisk]);

  if (loading && !risk) {
    return (
      <div style={shellStyle}>
        <div style={{ padding: 14, color: "var(--text-muted)", fontSize: 11 }}>Loading today&apos;s decision…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={shellStyle}>
        <div style={{ padding: 14, color: "var(--red)", fontSize: 11 }}>
          Decision Card error: {error}
          <button onClick={fetchRisk} style={btn}>Retry</button>
        </div>
      </div>
    );
  }

  if (!risk || risk.picks.length === 0) {
    return (
      <div style={shellStyle}>
        <div style={{ padding: 14, color: "var(--text-muted)", fontSize: 11 }}>No picks for today.</div>
      </div>
    );
  }

  const top = risk.picks[0];
  const isRiskOff = risk.regime === "risk_off";
  const regimeColor = isRiskOff ? "var(--amber)" : "var(--green)";
  const regimeBg = isRiskOff ? "rgba(251,191,36,0.12)" : "var(--green-bg)";

  return (
    <div style={shellStyle}>
      {/* Risk-off banner */}
      {isRiskOff && (
        <div style={{
          background: regimeBg,
          color: regimeColor,
          padding: "5px 14px",
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          borderBottom: "1px solid var(--border)",
        }}>
          ⚠ Half size active — SPY below 200dMA AND VIX elevated. Gross exposure capped at {(risk.gross_weight * 100).toFixed(0)}%.
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(280px, 1.2fr) 1fr 1fr minmax(220px, 1.4fr)", gap: 0, minHeight: 110 }}>
        {/* Top pick — hero */}
        <div style={{ padding: "12px 16px", borderRight: "1px solid var(--border-soft)" }}>
          <div style={labelStyle}>Top Pick · Rank 1 · asof {risk.asof}</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginTop: 4 }}>
            <span style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>{top.ticker}</span>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>${top.last_close?.toFixed(2)}</span>
          </div>
          <div style={{ display: "flex", gap: 16, marginTop: 6 }}>
            <div>
              <Metric label="Confidence" value={(top.y_proba * 100).toFixed(1) + "%"} sub={`${top.y_proba >= 0.6 ? "high" : top.y_proba >= 0.55 ? "above threshold" : "marginal"}`} accent="up" />
              <div style={{ marginTop: 4 }}>
                <CalibrationRibbon proba={top.y_proba} size="sm" />
              </div>
            </div>
            <Metric label="Edge vs Universe" value={fmtSignedPct(risk.edge_over_universe, 2)} sub={`mean ${(risk.universe_mean_proba * 100).toFixed(1)}%`} accent={risk.edge_over_universe > 0 ? "up" : "dn"} />
          </div>
        </div>

        {/* Recommended size */}
        <div style={{ padding: "12px 16px", borderRight: "1px solid var(--border-soft)" }}>
          <div style={labelStyle}>Recommended Size</div>
          <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.01em", marginTop: 6, color: "var(--text-primary)" }}>
            {fmtMoney0(top.rec_notional)}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
            {top.rec_shares ?? "—"} shares · {fmtPct(top.weight_pct, 1)} of BP
          </div>
          <div style={{ fontSize: 9, color: "var(--text-faint)", marginTop: 6 }}>
            vol-target 15% · 20d realized {fmtPct(top.realized_vol_20d, 0)}
          </div>
        </div>

        {/* Regime */}
        <div style={{ padding: "12px 16px", borderRight: "1px solid var(--border-soft)" }}>
          <div style={labelStyle}>Regime</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
            <span style={{
              background: regimeBg, color: regimeColor,
              padding: "3px 10px", borderRadius: 3, fontSize: 11, fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.06em",
            }}>{risk.regime.replace("_", " ")}</span>
            <span style={{ fontSize: 10, color: "var(--text-faint)" }}>×{risk.regime_multiplier}</span>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.5 }}>
            SPY {risk.spy_close?.toFixed(0) ?? "—"} / 200dMA {risk.spy_ma200?.toFixed(0) ?? "—"}<br />
            VIX {risk.vix?.toFixed(1) ?? "—"} ({risk.vix_pct_rank != null ? `${(risk.vix_pct_rank * 100).toFixed(0)}th pct` : "—"})
          </div>
        </div>

        {/* Thesis */}
        <div style={{ padding: "12px 16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={labelStyle}>{top.ticker} Thesis (AI)</span>
            {sentiment && (
              <span style={{
                background: sentiment.label === "bullish" ? "var(--green-bg)" : sentiment.label === "bearish" ? "var(--red-bg)" : "var(--bg-row)",
                color: sentiment.label === "bullish" ? "var(--green)" : sentiment.label === "bearish" ? "var(--red)" : "var(--text-muted)",
                padding: "1px 6px", borderRadius: 3, fontSize: 9, fontWeight: 700,
                textTransform: "uppercase", letterSpacing: "0.04em",
              }}>
                {sentiment.label} {sentiment.confidence ? `${Math.round(sentiment.confidence * 100)}%` : ""}
              </span>
            )}
          </div>
          <div style={{
            fontSize: 11, color: "var(--text-secondary)", marginTop: 6,
            lineHeight: 1.5, display: "-webkit-box",
            WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden",
            minHeight: 54,
          }}>
            {thesis ?? <span style={{ color: "var(--text-faint)" }}>Pulling thesis (first load takes ~5s)…</span>}
          </div>
        </div>
      </div>

      {/* Conviction ledger — top SHAP contributors to the top pick */}
      <ConvictionLedger ticker={top.ticker} date={risk.asof} />

      {/* Secondary picks strip */}
      {risk.picks.length > 1 && (
        <div style={{ borderTop: "1px solid var(--border-soft)", padding: "6px 16px", display: "flex", gap: 14, alignItems: "center", fontSize: 10, color: "var(--text-muted)" }}>
          <span style={{ fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", fontSize: 9 }}>Also today</span>
          {risk.picks.slice(1).map((p) => (
            <span key={p.ticker} style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>#{p.rank} {p.ticker}</span>
              <span>{(p.y_proba * 100).toFixed(1)}%</span>
              <CalibrationRibbon proba={p.y_proba} size="xs" />
              <span style={{ color: "var(--text-faint)" }}>·</span>
              <span>{fmtPct(p.weight_pct, 1)}</span>
              <span style={{ color: "var(--text-faint)" }}>·</span>
              <span>{fmtMoney0(p.rec_notional)}</span>
              {p.dropped_reason && (
                <span style={{ color: "var(--amber)", fontSize: 9, fontStyle: "italic" }}>
                  dropped: {p.dropped_reason}
                </span>
              )}
            </span>
          ))}
          <span style={{ marginLeft: "auto", fontSize: 9, color: "var(--text-faint)" }}>
            gross {fmtPct(risk.gross_weight, 0)} · BP {fmtMoney0(risk.buying_power)}
          </span>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, sub, accent }: { label: string; value: string; sub: string; accent?: "up" | "dn" }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, color: accent === "up" ? "var(--green)" : accent === "dn" ? "var(--red)" : "var(--text-primary)", marginTop: 1 }}>{value}</div>
      <div style={{ fontSize: 9, color: "var(--text-faint)", marginTop: 1 }}>{sub}</div>
    </div>
  );
}

const shellStyle: React.CSSProperties = {
  background: "var(--bg-panel)",
  borderBottom: "1px solid var(--border)",
  fontFeatureSettings: "'tnum'",
};

const labelStyle: React.CSSProperties = {
  fontSize: 9,
  color: "var(--text-muted)",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  fontWeight: 700,
};

const btn: React.CSSProperties = {
  background: "var(--bg-elevated)",
  color: "var(--text-secondary)",
  border: "1px solid var(--border)",
  borderRadius: 4,
  padding: "4px 10px",
  fontSize: 10,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "inherit",
  marginLeft: 8,
};
