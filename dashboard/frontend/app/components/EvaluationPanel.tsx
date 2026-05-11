"use client";

import { useEffect, useState, useCallback } from "react";

const API_BASE = "http://localhost:8000";

type TradeLevel = {
  status: "no_trades" | "ok";
  n_closed_trades?: number;
  n_wins?: number;
  n_losses?: number;
  hit_rate?: number;
  total_pnl?: number;
  avg_pnl?: number;
  avg_return_pct?: number;
  benchmark_universe_avg_return?: number | null;
  alpha_per_trade_pct?: number | null;
  best_trade?: number;
  worst_trade?: number;
  by_ticker?: { ticker: string; n_trades: number; total_pnl: number; avg_pnl_pct: number; hit_rate: number }[];
  note?: string;
};

type CalibrationBucket = {
  proba_bucket: string;
  count: number;
  actual_top_quintile_rate: number;
};

type ModelQuality = {
  status: string;
  n_predictions_evaluated?: number;
  n_dates_evaluated?: number;
  top2_avg_return?: number;
  universe_avg_return?: number;
  top2_alpha_per_holding_period?: number;
  top2_hit_rate?: number;
  avg_spearman_correlation?: number | null;
  spearman_interpretation?: string;
  calibration_buckets?: CalibrationBucket[];
};

type Drift = {
  status: string;
  recent_window?: string;
  n_recent?: number;
  n_historical?: number;
  recent_mean_proba?: number;
  historical_mean_proba?: number;
  z_score_difference?: number;
  drift_severity?: "low" | "moderate" | "high";
  recent_buy_signal_rate?: number;
  historical_buy_signal_rate?: number;
  buy_signal_rate_delta?: number;
  by_ticker_top5_drift?: { ticker: string; recent_mean: number; historical_mean: number; delta: number }[];
};

type Report = {
  generated_at: string;
  category_a_trade_level: TradeLevel;
  category_b_model_quality: ModelQuality;
  category_c_drift: Drift;
  error?: string;
};

const fmtPct = (n: number | undefined | null, digits = 2) =>
  n == null ? "—" : `${(n * 100).toFixed(digits)}%`;
const fmtSigned = (n: number | undefined | null, digits = 2) =>
  n == null ? "—" : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(digits)}%`;
const fmtMoney = (n: number | undefined | null) =>
  n == null ? "—" : `$${n.toFixed(2)}`;

export default function EvaluationPanel() {
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/evaluation/report`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setReport(data);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReport();
  }, [fetchReport]);

  if (loading && !report) {
    return <div style={{ padding: 20, color: "var(--text-muted)" }}>Computing evaluation report...</div>;
  }

  if (error) {
    return (
      <div style={{ padding: 20, color: "var(--red)" }}>
        Error: {error}
        <button onClick={fetchReport} style={btn}>Retry</button>
      </div>
    );
  }

  if (!report) {
    return <div style={{ padding: 20, color: "var(--text-muted)" }}>No report loaded.</div>;
  }

  const a = report.category_a_trade_level;
  const b = report.category_b_model_quality;
  const c = report.category_c_drift;

  return (
    <div style={{ padding: "12px 16px", color: "var(--text-primary)", overflow: "auto", height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em", margin: 0 }}>Model Evaluation</h1>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "var(--text-faint)" }}>
            Generated {new Date(report.generated_at).toLocaleString()}
          </span>
          <button onClick={fetchReport} disabled={loading} style={btn}>
            {loading ? "Computing..." : "Recompute"}
          </button>
        </div>
      </div>

      {/* === Category B === */}
      <Section title="B. Model Quality">
        {b.status !== "ok" ? (
          <Empty msg={b.status} />
        ) : (
          <>
            <StatGrid>
              <Stat label="Top-2 Alpha" value={fmtSigned(b.top2_alpha_per_holding_period)} sub="vs universe avg, per 5-day hold" accent={(b.top2_alpha_per_holding_period ?? 0) > 0 ? "up" : "dn"} />
              <Stat label="Top-2 Hit Rate" value={fmtPct(b.top2_hit_rate, 1)} sub={`vs 50% coin flip`} accent={(b.top2_hit_rate ?? 0) > 0.5 ? "up" : "dn"} />
              <Stat label="Top-2 Avg Return" value={fmtSigned(b.top2_avg_return)} sub={`vs ${fmtSigned(b.universe_avg_return)} universe`} />
              <Stat label="Rank Correlation" value={b.avg_spearman_correlation != null ? b.avg_spearman_correlation.toFixed(4) : "—"} sub={b.spearman_interpretation ?? ""} accent={(b.avg_spearman_correlation ?? 0) > 0.05 ? "up" : (b.avg_spearman_correlation ?? 0) < -0.05 ? "dn" : "muted"} />
            </StatGrid>

            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 8 }}>
                Calibration ({b.n_predictions_evaluated?.toLocaleString()} predictions across {b.n_dates_evaluated} dates)
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12 }}>
                Each bar shows the actual % of predictions in that probability bucket that ended up in the top quintile of next-5-day returns. Dashed line = 20% random baseline.
              </div>
              <CalibrationChart buckets={b.calibration_buckets ?? []} />
            </div>
          </>
        )}
      </Section>

      {/* === Category C === */}
      <Section title="C. Drift Detection">
        {c.status !== "ok" ? (
          <Empty msg={c.status} />
        ) : (
          <>
            <StatGrid>
              <Stat label="Recent Mean Proba" value={c.recent_mean_proba?.toFixed(4) ?? "—"} sub={`vs ${c.historical_mean_proba?.toFixed(4)} historical`} />
              <Stat label="Z-Score Diff" value={(c.z_score_difference ?? 0) >= 0 ? `+${c.z_score_difference?.toFixed(3)}` : `${c.z_score_difference?.toFixed(3)}`} sub={`drift severity: ${c.drift_severity}`} accent={c.drift_severity === "high" ? "dn" : c.drift_severity === "moderate" ? "muted" : "up"} />
              <Stat label="Buy Signal Rate (recent)" value={fmtPct(c.recent_buy_signal_rate, 1)} sub={`vs ${fmtPct(c.historical_buy_signal_rate, 1)} historical`} accent={(c.buy_signal_rate_delta ?? 0) > 0.1 ? "dn" : "muted"} />
              <Stat label="Window" value={c.n_recent + " recent"} sub={`${c.n_historical} historical`} />
            </StatGrid>

            {c.by_ticker_top5_drift && c.by_ticker_top5_drift.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 8 }}>
                  Top 5 Tickers by Drift
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "50px 90px 90px 90px", gap: 8, padding: "6px 0", color: "var(--text-muted)", fontSize: 10, textTransform: "uppercase", borderBottom: "1px solid var(--border-soft)" }}>
                  <span>Ticker</span><span>Recent Mean</span><span>Historical</span><span>Delta</span>
                </div>
                {c.by_ticker_top5_drift.map((t) => (
                  <div key={t.ticker} style={{ display: "grid", gridTemplateColumns: "50px 90px 90px 90px", gap: 8, padding: "6px 0", fontSize: 11, borderBottom: "1px solid var(--border-soft)" }}>
                    <span style={{ fontWeight: 600 }}>{t.ticker}</span>
                    <span>{t.recent_mean.toFixed(4)}</span>
                    <span style={{ color: "var(--text-muted)" }}>{t.historical_mean.toFixed(4)}</span>
                    <span style={{ fontWeight: 600, color: t.delta >= 0 ? "var(--green)" : "var(--red)" }}>{t.delta >= 0 ? "+" : ""}{t.delta.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </Section>

      {/* === Category A === */}
      <Section title="A. Trade-Level (live trades)">
        {a.status === "no_trades" ? (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            {a.note} Once trades close, this section will populate with realized P&L, hit rate, and alpha vs universe.
          </div>
        ) : (
          <>
            <StatGrid>
              <Stat label="Closed Trades" value={(a.n_closed_trades ?? 0).toString()} sub={`${a.n_wins}W / ${a.n_losses}L`} />
              <Stat label="Hit Rate" value={fmtPct(a.hit_rate, 1)} sub="" accent={(a.hit_rate ?? 0) > 0.5 ? "up" : "dn"} />
              <Stat label="Total P&L" value={fmtMoney(a.total_pnl)} sub={`avg ${fmtMoney(a.avg_pnl)}`} accent={(a.total_pnl ?? 0) >= 0 ? "up" : "dn"} />
              <Stat label="Alpha per Trade" value={fmtSigned(a.alpha_per_trade_pct, 3)} sub={`vs ${fmtSigned(a.benchmark_universe_avg_return)} benchmark`} accent={(a.alpha_per_trade_pct ?? 0) > 0 ? "up" : "dn"} />
            </StatGrid>

            {a.by_ticker && a.by_ticker.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 8 }}>
                  By Ticker
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "60px 60px 90px 100px 70px", gap: 8, padding: "6px 0", color: "var(--text-muted)", fontSize: 10, textTransform: "uppercase", borderBottom: "1px solid var(--border-soft)" }}>
                  <span>Ticker</span><span>N</span><span>Total P&L</span><span>Avg Return</span><span>Hit Rate</span>
                </div>
                {a.by_ticker.map((t) => (
                  <div key={t.ticker} style={{ display: "grid", gridTemplateColumns: "60px 60px 90px 100px 70px", gap: 8, padding: "6px 0", fontSize: 11, borderBottom: "1px solid var(--border-soft)" }}>
                    <span style={{ fontWeight: 600 }}>{t.ticker}</span>
                    <span>{t.n_trades}</span>
                    <span style={{ color: t.total_pnl >= 0 ? "var(--green)" : "var(--red)" }}>{fmtMoney(t.total_pnl)}</span>
                    <span>{fmtSigned(t.avg_pnl_pct, 3)}</span>
                    <span>{fmtPct(t.hit_rate, 1)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </Section>
    </div>
  );
}

function CalibrationChart({ buckets }: { buckets: CalibrationBucket[] }) {
  if (!buckets || buckets.length === 0) {
    return <div style={{ color: "var(--text-muted)" }}>No calibration data.</div>;
  }
  const maxRate = Math.max(...buckets.map((b) => b.actual_top_quintile_rate), 0.5);
  const baseline = 0.20;
  const chartWidth = 480;
  const labelWidth = 90;
  const countWidth = 80;
  const innerWidth = chartWidth - labelWidth - countWidth - 16;

  return (
    <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: 12 }}>
      {buckets.map((b, idx) => {
        const widthPct = (b.actual_top_quintile_rate / maxRate) * 100;
        const baselinePct = (baseline / maxRate) * 100;
        const aboveBaseline = b.actual_top_quintile_rate > baseline;
        return (
          <div key={idx} style={{ display: "grid", gridTemplateColumns: `${labelWidth}px 1fr ${countWidth}px`, gap: 8, alignItems: "center", padding: "5px 0" }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>{b.proba_bucket}</span>
            <div style={{ position: "relative", height: 22, background: "var(--bg-row)", borderRadius: 3, overflow: "hidden" }}>
              {/* Bar */}
              <div style={{
                position: "absolute",
                left: 0, top: 0, bottom: 0,
                width: `${widthPct}%`,
                background: aboveBaseline ? "var(--green)" : "var(--text-muted)",
                opacity: aboveBaseline ? 0.85 : 0.4,
                borderRadius: 3,
              }} />
              {/* Baseline marker line */}
              <div style={{
                position: "absolute",
                left: `${baselinePct}%`, top: -2, bottom: -2,
                width: 1,
                background: "var(--text-faint)",
                borderRight: "1px dashed var(--text-faint)",
              }} />
              {/* Rate label inside bar */}
              <div style={{
                position: "absolute",
                left: `min(${widthPct}% + 6px, calc(100% - 50px))`,
                top: "50%",
                transform: "translateY(-50%)",
                fontSize: 11,
                fontWeight: 600,
                color: aboveBaseline ? "var(--green)" : "var(--text-secondary)",
              }}>
                {(b.actual_top_quintile_rate * 100).toFixed(1)}%
              </div>
            </div>
            <span style={{ fontSize: 10, color: "var(--text-muted)", textAlign: "right" }}>n={b.count.toLocaleString()}</span>
          </div>
        );
      })}
      <div style={{ marginTop: 8, fontSize: 10, color: "var(--text-faint)", borderTop: "1px dashed var(--border-soft)", paddingTop: 6 }}>
        Dashed line = 20% random baseline. Green bars = lift above random.
      </div>
    </div>
  );
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
  return <div style={{ color: "var(--text-muted)", fontSize: 12 }}>Status: {msg}</div>;
}
