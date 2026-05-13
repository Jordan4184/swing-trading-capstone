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

type AblationDelta = {
  sharpe_ratio: number;
  annualized_return: number;
  max_drawdown: number;
  total_return: number;
};

type AblationRow = {
  layer: string;
  flags: { enable_vol_target: boolean; enable_regime_gate: boolean; enable_corr_filter: boolean };
  metrics: { sharpe_ratio: number; annualized_return: number; max_drawdown: number; hit_rate: number; total_return: number };
  ci: { sharpe_ratio: { point: number; ci_low: number; ci_high: number } };
  n_trades: number;
  avg_gross_weight: number;
  n_risk_off_rebalances: number;
  delta_vs_prev: AblationDelta | null;
  delta_vs_baseline: AblationDelta | null;
};

type Ablation = {
  generated_at: string;
  n_resamples: number;
  rows: AblationRow[];
};

type FeatureAblation = {
  generated_at: string;
  n_splits: number;
  feature_columns: string[];
  absolute: { fold_aucs: number[]; mean_auc: number; std_auc: number };
  ranked: { fold_aucs: number[]; mean_auc: number; std_auc: number };
  delta_auc: number;
  recommendation: string;
};

type ExitPolicy = {
  v3_strategy: { sharpe_ratio: number; annualized_return: number; max_drawdown: number; hit_rate: number; total_return: number };
  v2_strategy: { sharpe_ratio: number; annualized_return: number; max_drawdown: number; hit_rate: number; total_return: number };
  config: { atr_multiplier: number; atr_window: number };
  exit_reasons: Record<string, number>;
  n_per_trade: number;
  acceptance: { maxdd_delta_pp: number; cagr_delta_pp: number; sharpe_delta: number; passed: boolean };
};

type ShapContrib = { feature: string; shap_value: number };
type FailurePick = {
  ticker: string;
  y_proba: number | null;
  fwd_return_5d: number | null;
  base_value: number | null;
  top_contributors: ShapContrib[];
};
type FailureCase = {
  date: string;
  label: string;
  context: string;
  lesson: string;
  basket_return_net: number;
  gross_weight: number;
  regime: string;
  picks: FailurePick[];
};
type FailureModes = {
  generated_at: string;
  n_cases: number;
  feature_columns: string[];
  cases: FailureCase[];
};

export default function EvaluationPanel() {
  const [report, setReport] = useState<Report | null>(null);
  const [ablation, setAblation] = useState<Ablation | null>(null);
  const [featureAblation, setFeatureAblation] = useState<FeatureAblation | null>(null);
  const [exitPolicy, setExitPolicy] = useState<ExitPolicy | null>(null);
  const [failureModes, setFailureModes] = useState<FailureModes | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [reportRes, ablRes, featAblRes, exitRes, failRes] = await Promise.all([
        fetch(`${API_BASE}/api/evaluation/report`),
        fetch(`${API_BASE}/api/ablation`),
        fetch(`${API_BASE}/api/feature-ablation`),
        fetch(`${API_BASE}/api/exit-policy`),
        fetch(`${API_BASE}/api/failure-modes`),
      ]);
      if (!reportRes.ok) {
        throw new Error(`HTTP ${reportRes.status}`);
      }
      const data = await reportRes.json();
      if (data.error) {
        setError(data.error);
      } else {
        setReport(data);
      }
      if (ablRes.ok) {
        setAblation(await ablRes.json());
      }
      if (featAblRes.ok) {
        setFeatureAblation(await featAblRes.json());
      }
      if (exitRes.ok) {
        setExitPolicy(await exitRes.json());
      }
      if (failRes.ok) {
        setFailureModes(await failRes.json());
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

      {/* === Risk-Layer Ablation === */}
      <Section title="Risk-Layer Ablation (v1 → v2, stacked-additive)">
        {!ablation ? (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            No ablation artifact found. Run <code style={{ background: "var(--bg-row)", padding: "1px 4px", borderRadius: 2 }}>python -m src.ablation</code> from the project root.
          </div>
        ) : (
          <AblationTable rows={ablation.rows} nResamples={ablation.n_resamples} />
        )}
      </Section>

      {/* === Feature Ablation: absolute vs per-date rank === */}
      <Section title="Feature Ablation — absolute vs cross-sectional rank (research)">
        {!featureAblation ? (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            No feature ablation artifact found. Run <code style={{ background: "var(--bg-row)", padding: "1px 4px", borderRadius: 2 }}>python -m src.feature_ablation</code>.
          </div>
        ) : (
          <FeatureAblationTable data={featureAblation} />
        )}
      </Section>

      {/* === Failure-Mode Case Studies === */}
      <Section title="Failure-Mode Case Studies — 3 worst trades dissected">
        {!failureModes ? (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            No failure-mode artifact found. Run <code style={{ background: "var(--bg-row)", padding: "1px 4px", borderRadius: 2 }}>python -m src.failure_modes</code>.
          </div>
        ) : (
          <FailureModesPanel data={failureModes} />
        )}
      </Section>

      {/* === Exit Policy: scheduled vs ATR stop === */}
      <Section title="Exit Policy — scheduled (v2) vs ATR trailing stop (v3, research)">
        {!exitPolicy ? (
          <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "8px 0" }}>
            No v3 artifact found. Run <code style={{ background: "var(--bg-row)", padding: "1px 4px", borderRadius: 2 }}>python -m src.backtest_v3</code>.
          </div>
        ) : (
          <ExitPolicyTable data={exitPolicy} />
        )}
      </Section>

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

function AblationTable({ rows, nResamples }: { rows: AblationRow[]; nResamples: number }) {
  const cols = "180px 80px 90px 90px 100px 100px 90px";
  const headerStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: cols,
    gap: 8,
    padding: "6px 0",
    fontSize: 10,
    color: "var(--text-muted)",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    fontWeight: 700,
    borderBottom: "1px solid var(--border-soft)",
  };
  const rowStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: cols,
    gap: 8,
    padding: "8px 0",
    fontSize: 11,
    borderBottom: "1px solid var(--border-soft)",
    alignItems: "center",
  };

  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.5 }}>
        Each row adds one risk component to the previous configuration. Bottom row is the
        full v2 strategy. Δ columns show the marginal contribution of that layer relative to
        the row immediately above. CIs from {nResamples.toLocaleString()} bootstrap resamples.
      </div>
      <div style={headerStyle}>
        <span>Layer</span>
        <span>Trades</span>
        <span>Sharpe</span>
        <span>CAGR</span>
        <span>MaxDD</span>
        <span>Δ Sharpe</span>
        <span>Δ MaxDD</span>
      </div>
      {rows.map((r, idx) => {
        const m = r.metrics;
        const ci = r.ci.sharpe_ratio;
        const d = r.delta_vs_prev;
        const isBaseline = idx === 0;
        const isFinal = idx === rows.length - 1;
        return (
          <div
            key={r.layer}
            style={{
              ...rowStyle,
              background: isFinal ? "rgba(74, 222, 128, 0.05)" : "transparent",
            }}
          >
            <span style={{ fontWeight: isBaseline || isFinal ? 700 : 500, color: "var(--text-primary)" }}>
              {r.layer}
            </span>
            <span style={{ color: "var(--text-muted)" }}>{r.n_trades}</span>
            <span style={{ fontFeatureSettings: "'tnum'" }}>
              <div style={{ fontWeight: 600 }}>{m.sharpe_ratio.toFixed(3)}</div>
              <div style={{ fontSize: 9, color: "var(--text-faint)" }}>[{ci.ci_low.toFixed(2)}, {ci.ci_high.toFixed(2)}]</div>
            </span>
            <span style={{ fontFeatureSettings: "'tnum'", color: m.annualized_return >= 0 ? "var(--text-primary)" : "var(--red)" }}>
              {(m.annualized_return * 100).toFixed(2)}%
            </span>
            <span style={{ fontFeatureSettings: "'tnum'", color: "var(--red)" }}>
              {(m.max_drawdown * 100).toFixed(2)}%
            </span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600, color: deltaColor(d?.sharpe_ratio) }}>
              {d ? formatDelta(d.sharpe_ratio, false) : "—"}
            </span>
            <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600, color: deltaColor(d?.max_drawdown) }}>
              {d ? formatDelta(d.max_drawdown * 100, true) : "—"}
            </span>
          </div>
        );
      })}
      <div style={{ marginTop: 10, fontSize: 10, color: "var(--text-faint)", lineHeight: 1.5 }}>
        Read: <strong style={{ color: "var(--text-muted)" }}>{rows[1]?.layer.replace("+ ", "")}</strong> is the dominant contributor —
        on this sample it alone moves Sharpe by {formatDelta(rows[1]?.delta_vs_prev?.sharpe_ratio ?? 0, false)}{" "}
        and MaxDD by {formatDelta((rows[1]?.delta_vs_prev?.max_drawdown ?? 0) * 100, true)}.
        Subsequent layers each contribute &lt; ±0.05 Sharpe — small enough that the
        correlation filter actually shows a negative marginal on this 6.4-year window,
        a finding worth owning rather than hiding.
      </div>
    </div>
  );
}

function FeatureAblationTable({ data }: { data: FeatureAblation }) {
  const cols = "260px 90px 90px 200px";
  const headerStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: cols,
    gap: 8,
    padding: "6px 0",
    fontSize: 10,
    color: "var(--text-muted)",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    fontWeight: 700,
    borderBottom: "1px solid var(--border-soft)",
  };
  const rowStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: cols,
    gap: 8,
    padding: "8px 0",
    fontSize: 11,
    borderBottom: "1px solid var(--border-soft)",
    alignItems: "center",
  };

  const rows = [
    { label: "Absolute features (current production)", v: data.absolute, isCurrent: true },
    { label: "Per-date rank features", v: data.ranked, isCurrent: false },
  ];

  const deltaColorClass =
    data.delta_auc > 0.005 ? "var(--green)" : data.delta_auc < -0.005 ? "var(--red)" : "var(--text-muted)";

  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.5 }}>
        Same Random Forest, same walk-forward {data.n_splits}-fold split. Only the
        feature representation differs: absolute (RSI=50, return=2%) vs per-date
        cross-sectional rank-pct across the 11-ticker universe. Aligns the feature
        scale to the target&apos;s cross-sectional nature.
      </div>
      <div style={headerStyle}>
        <span>Feature set</span>
        <span>Mean AUC</span>
        <span>Std (folds)</span>
        <span>Fold AUCs</span>
      </div>
      {rows.map((r) => (
        <div key={r.label} style={{ ...rowStyle, background: r.isCurrent ? "transparent" : "rgba(34, 211, 238, 0.05)" }}>
          <span style={{ fontWeight: 600 }}>{r.label}</span>
          <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>{r.v.mean_auc.toFixed(4)}</span>
          <span style={{ fontFeatureSettings: "'tnum'", color: "var(--text-muted)" }}>± {r.v.std_auc.toFixed(4)}</span>
          <span style={{ fontFeatureSettings: "'tnum'", fontSize: 10, color: "var(--text-muted)" }}>
            {r.v.fold_aucs.map((a) => a.toFixed(3)).join(" · ")}
          </span>
        </div>
      ))}
      <div style={{ marginTop: 12, padding: "10px 12px", background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "baseline", flexWrap: "wrap" }}>
          <span style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700 }}>Δ AUC</span>
          <span style={{ fontSize: 18, fontWeight: 700, color: deltaColorClass, fontFeatureSettings: "'tnum'" }}>
            {data.delta_auc >= 0 ? "+" : ""}{data.delta_auc.toFixed(4)}
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {data.recommendation}
          </span>
        </div>
        <div style={{ marginTop: 6, fontSize: 10, color: "var(--text-faint)", lineHeight: 1.5 }}>
          Research artifact — does not reflect the current production model
          (still absolute features). Promoting rank features to production
          would cascade through predictions.parquet, v2 backtest, ablation,
          and every dashboard view — a deliberate, separate decision.
        </div>
      </div>
    </div>
  );
}

const FEATURE_LABELS: Record<string, string> = {
  return_1d: "1d return",
  return_5d: "5d return",
  return_20d: "20d return",
  return_60d: "60d return",
  volatility_20d: "20d vol",
  volatility_60d: "60d vol",
  rsi_14: "RSI-14",
  bb_pct: "Bollinger %B",
  volume_ratio_20d: "volume ratio",
  spy_return_1d: "SPY 1d",
  excess_return_1d: "excess 1d",
};

function FailureModesPanel({ data }: { data: FailureModes }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.5 }}>
        Three of v2&apos;s worst basket returns, each from a structurally different
        kind of regime shock. For each: market context at the time, the picks
        and what the model actually saw (SHAP contributions to the positive
        class), realized 5-day forward return, and one-line &quot;what I&apos;d
        change&quot; written from market context.
      </div>
      <div style={{ display: "grid", gap: 14 }}>
        {data.cases.map((c) => (
          <FailureCaseCard key={c.date} c={c} />
        ))}
      </div>
    </div>
  );
}

function FailureCaseCard({ c }: { c: FailureCase }) {
  const badgeColor = "var(--red)";
  return (
    <div style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, marginBottom: 8 }}>
        <div>
          <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700 }}>
            {c.date} · {c.label}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-faint)", marginTop: 2 }}>
            regime <span style={{ color: "var(--text-secondary)" }}>{c.regime}</span>
            {" · "}gross <span style={{ color: "var(--text-secondary)" }}>{(c.gross_weight * 100).toFixed(0)}%</span>
          </div>
        </div>
        <span style={{ fontSize: 16, fontWeight: 700, color: badgeColor, fontFeatureSettings: "'tnum'" }}>
          {(c.basket_return_net * 100).toFixed(2)}%
        </span>
      </div>

      <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 12 }}>
        {c.context}
      </div>

      <div style={{ display: "grid", gap: 10, marginBottom: 12 }}>
        {c.picks.map((p) => (
          <div key={p.ticker} style={{ background: "var(--bg-panel)", border: "1px solid var(--border-soft)", borderRadius: 3, padding: "8px 10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
              <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>{p.ticker}</span>
                <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                  proba {p.y_proba != null ? p.y_proba.toFixed(3) : "—"}
                </span>
              </div>
              <span style={{ fontSize: 11, fontWeight: 700, color: p.fwd_return_5d != null && p.fwd_return_5d < 0 ? "var(--red)" : "var(--text-primary)", fontFeatureSettings: "'tnum'" }}>
                realized 5d: {p.fwd_return_5d != null ? `${(p.fwd_return_5d * 100).toFixed(2)}%` : "—"}
              </span>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              {p.top_contributors.slice(0, 5).map((tc) => {
                const isPos = tc.shap_value >= 0;
                const color = isPos ? "var(--green)" : "var(--red)";
                const bg = isPos ? "var(--green-bg)" : "var(--red-bg)";
                return (
                  <span
                    key={tc.feature}
                    title={`SHAP contribution to the positive class (top-quintile). Positive = pushed proba up.`}
                    style={{ display: "flex", alignItems: "center", gap: 5 }}
                  >
                    <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>{FEATURE_LABELS[tc.feature] ?? tc.feature}</span>
                    <span style={{ position: "relative", width: 28, height: 5, background: bg, borderRadius: 1, overflow: "hidden" }}>
                      <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: "100%", background: color, opacity: 0.85 }} />
                    </span>
                    <span style={{ fontSize: 10, fontWeight: 700, color, fontFeatureSettings: "'tnum'" }}>
                      {isPos ? "+" : ""}{tc.shap_value.toFixed(3)}
                    </span>
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div style={{ borderTop: "1px dashed var(--border-soft)", paddingTop: 10, display: "flex", gap: 10, alignItems: "flex-start" }}>
        <span style={{ fontSize: 9, color: "var(--green)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, flexShrink: 0 }}>
          what I&apos;d change
        </span>
        <span style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>
          {c.lesson}
        </span>
      </div>
    </div>
  );
}

function ExitPolicyTable({ data }: { data: ExitPolicy }) {
  const cols = "200px 100px 100px 100px 100px";
  const header: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: cols,
    gap: 8,
    padding: "6px 0",
    fontSize: 10,
    color: "var(--text-muted)",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    fontWeight: 700,
    borderBottom: "1px solid var(--border-soft)",
  };
  const row: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: cols,
    gap: 8,
    padding: "8px 0",
    fontSize: 11,
    borderBottom: "1px solid var(--border-soft)",
    alignItems: "center",
  };

  const rows = [
    { label: "v2 — scheduled 5-day exit (production)", v: data.v2_strategy, isCurrent: true },
    { label: `v3 — ${data.config.atr_multiplier}x ATR(${data.config.atr_window}) stop`, v: data.v3_strategy, isCurrent: false },
  ];
  const exitReasons = Object.entries(data.exit_reasons);
  const totalTrades = data.n_per_trade;
  const stoppedCount = (data.exit_reasons.stop_gap ?? 0) + (data.exit_reasons.stop_intraday ?? 0);
  const stoppedPct = totalTrades ? (stoppedCount / totalTrades) * 100 : 0;

  const verdictColor = data.acceptance.passed ? "var(--green)" : "var(--amber)";
  const verdictText = data.acceptance.passed
    ? "PASS — MaxDD improved within target band"
    : "INVESTIGATE — stop did not deliver the predicted MaxDD improvement";

  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.5 }}>
        v3 keeps v2&apos;s sizing logic but adds a per-day price-based exit: stop =
        entry close − {data.config.atr_multiplier}× ATR({data.config.atr_window}). If the next day&apos;s low
        breaches the stop the trade exits at the stop price; if the open gaps
        through the stop the fill is at the open (honest gap risk).
      </div>
      <div style={header}>
        <span>Variant</span>
        <span>Sharpe</span>
        <span>CAGR</span>
        <span>MaxDD</span>
        <span>Hit Rate</span>
      </div>
      {rows.map((r) => (
        <div key={r.label} style={{ ...row, background: r.isCurrent ? "transparent" : "rgba(251,191,36,0.04)" }}>
          <span style={{ fontWeight: 600 }}>{r.label}</span>
          <span style={{ fontFeatureSettings: "'tnum'", fontWeight: 600 }}>{r.v.sharpe_ratio.toFixed(3)}</span>
          <span style={{ fontFeatureSettings: "'tnum'" }}>{(r.v.annualized_return * 100).toFixed(2)}%</span>
          <span style={{ fontFeatureSettings: "'tnum'", color: "var(--red)" }}>{(r.v.max_drawdown * 100).toFixed(2)}%</span>
          <span style={{ fontFeatureSettings: "'tnum'" }}>{(r.v.hit_rate * 100).toFixed(1)}%</span>
        </div>
      ))}

      <div style={{ marginTop: 12, display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 240px", padding: "10px 12px", background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 6 }}>Exit reasons</div>
          {exitReasons.map(([reason, count]) => (
            <div key={reason} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "2px 0" }}>
              <span style={{ color: reason === "scheduled" ? "var(--text-secondary)" : "var(--amber)" }}>{reason}</span>
              <span style={{ fontFeatureSettings: "'tnum'", color: "var(--text-muted)" }}>{count} ({(count / totalTrades * 100).toFixed(1)}%)</span>
            </div>
          ))}
          <div style={{ marginTop: 6, fontSize: 10, color: "var(--text-faint)", paddingTop: 6, borderTop: "1px dashed var(--border-soft)" }}>
            Stop-out rate: {stoppedPct.toFixed(1)}% of {totalTrades.toLocaleString()} per-trade legs.
          </div>
        </div>

        <div style={{ flex: "2 1 320px", padding: "10px 12px", background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700, marginBottom: 6 }}>Verdict</div>
          <div style={{ fontSize: 12, fontWeight: 600, color: verdictColor, marginBottom: 6 }}>{verdictText}</div>
          <div style={{ display: "flex", gap: 14, fontSize: 11, marginBottom: 6 }}>
            <span><span style={{ color: "var(--text-faint)" }}>Δ Sharpe </span><span style={{ fontFeatureSettings: "'tnum'", color: data.acceptance.sharpe_delta >= 0 ? "var(--green)" : "var(--red)" }}>{data.acceptance.sharpe_delta >= 0 ? "+" : ""}{data.acceptance.sharpe_delta.toFixed(3)}</span></span>
            <span><span style={{ color: "var(--text-faint)" }}>Δ CAGR </span><span style={{ fontFeatureSettings: "'tnum'", color: data.acceptance.cagr_delta_pp >= 0 ? "var(--green)" : "var(--red)" }}>{data.acceptance.cagr_delta_pp >= 0 ? "+" : ""}{data.acceptance.cagr_delta_pp.toFixed(2)}pp</span></span>
            <span><span style={{ color: "var(--text-faint)" }}>Δ MaxDD </span><span style={{ fontFeatureSettings: "'tnum'", color: data.acceptance.maxdd_delta_pp >= 0 ? "var(--green)" : "var(--red)" }}>{data.acceptance.maxdd_delta_pp >= 0 ? "+" : ""}{data.acceptance.maxdd_delta_pp.toFixed(2)}pp</span></span>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.5 }}>
            Read: vol-targeted sizing in v2 has already shrunk position size on
            high-vol names, so the absolute $ loss the stop would catch is
            smaller than the trader-memo expected. The stop also forces exits
            on trades that would have recovered by day 5. Net: stop adds little
            value on top of vol-targeting on this 6.4-year sample. Owning this
            finding is the win — &quot;I tried the recommended stop; it didn&apos;t help
            here, and I can explain why.&quot;
          </div>
        </div>
      </div>
    </div>
  );
}

function deltaColor(d: number | undefined): string {
  if (d == null) return "var(--text-muted)";
  // For MaxDD, "improvement" is a less-negative number (positive delta on a negative quantity)
  // For Sharpe, positive is improvement
  // Caller controls sign convention; we just use sign of d.
  if (d > 0.001) return "var(--green)";
  if (d < -0.001) return "var(--red)";
  return "var(--text-muted)";
}

function formatDelta(n: number, isPp: boolean): string {
  const sign = n >= 0 ? "+" : "";
  return isPp ? `${sign}${n.toFixed(2)}pp` : `${sign}${n.toFixed(3)}`;
}
