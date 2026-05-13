"use client";

import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

type Contributor = {
  feature: string;
  shap_value: number;
};

type Explain = {
  ticker: string;
  date: string;
  base_value: number;
  top_contributors: Contributor[];
  all_contributors: Contributor[];
};

interface Props {
  ticker: string | undefined;
  date: string | undefined;
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

export default function ConvictionLedger({ ticker, date }: Props) {
  const [data, setData] = useState<Explain | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    if (!ticker || !date) return;
    let alive = true;
    setData(null);
    setMissing(false);
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/explain/${ticker}/${date}`);
        if (r.status === 404) {
          if (alive) setMissing(true);
          return;
        }
        if (!r.ok) return;
        const j = await r.json();
        if (alive) setData(j);
      } catch {
        /* silent — non-critical */
      }
    })();
    return () => {
      alive = false;
    };
  }, [ticker, date]);

  if (missing) {
    return (
      <div style={shell}>
        <div style={{ fontSize: 10, color: "var(--text-faint)", padding: "0 12px", display: "flex", alignItems: "center", height: "100%" }}>
          Attribution not yet computed for this date. Run <code style={{ background: "var(--bg-row)", padding: "0 4px" }}>python -m src.explain</code>.
        </div>
      </div>
    );
  }
  if (!data) {
    return (
      <div style={shell}>
        <div style={{ fontSize: 10, color: "var(--text-faint)", padding: "0 12px", display: "flex", alignItems: "center", height: "100%" }}>
          Loading attribution…
        </div>
      </div>
    );
  }

  // Build bars with the top-N contributors. Domain symmetric around 0 so positive
  // and negative contributions render at the same visual scale.
  const top = data.top_contributors.slice(0, 5);
  const maxAbs = Math.max(0.001, ...top.map((c) => Math.abs(c.shap_value)));

  return (
    <div style={shell} title={`Top contributions to ${data.ticker} on ${data.date} (base value ${data.base_value.toFixed(3)})`}>
      <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", padding: "0 12px 0 14px", whiteSpace: "nowrap" }}>
        Why {data.ticker}?
      </div>
      <div style={{ display: "flex", flex: 1, alignItems: "center", gap: 14, paddingRight: 14, overflow: "hidden" }}>
        {top.map((c) => {
          const value = c.shap_value;
          const isPos = value >= 0;
          const widthPct = (Math.abs(value) / maxAbs) * 100;
          const color = isPos ? "var(--green)" : "var(--red)";
          const bg = isPos ? "var(--green-bg)" : "var(--red-bg)";
          return (
            <div
              key={c.feature}
              style={{ display: "flex", alignItems: "center", gap: 5, minWidth: 0 }}
              title={`${FEATURE_LABELS[c.feature] ?? c.feature}: ${isPos ? "+" : ""}${value.toFixed(4)} (positive = pushes proba up)`}
            >
              <span style={{ fontSize: 10, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
                {FEATURE_LABELS[c.feature] ?? c.feature}
              </span>
              <span style={{ position: "relative", width: 36, height: 6, background: bg, borderRadius: 1, overflow: "hidden", flexShrink: 0 }}>
                <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${widthPct}%`, background: color, opacity: 0.85 }} />
              </span>
              <span style={{ fontSize: 10, fontWeight: 700, color, fontFeatureSettings: "'tnum'", whiteSpace: "nowrap" }}>
                {isPos ? "+" : ""}{value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const shell: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  height: 28,
  background: "var(--bg-elevated)",
  borderTop: "1px solid var(--border-soft)",
  fontFeatureSettings: "'tnum'",
};
