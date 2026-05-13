"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://localhost:8000";

// Dynamic import: plotly.js touches `window` at import time and will crash
// SSR. Use the gl3d build (~1.2MB) rather than the full plotly bundle.
const Plot = dynamic(
  async () => {
    // @ts-expect-error — no upstream types for the gl3d build
    const Plotly = (await import("plotly.js-gl3d-dist-min")).default;
    const createPlotlyComponent = (await import("react-plotly.js/factory")).default;
    return createPlotlyComponent(Plotly);
  },
  { ssr: false, loading: () => <SurfacePlaceholder text="Loading 3D surface…" /> },
);

type Pin = {
  date: string;
  ticker: string;
  x: number;
  y: number;
  z: number;
  basket_return: number;
  label: string;
  lesson: string;
  context: string;
};

type TourStep = {
  id: number;
  title: string;
  caption: string;
  camera: { x: number; y: number; z: number };
  spotlight: [string, string] | null;
};

type ShapSurface = {
  feature_x: string;
  feature_y: string;
  interaction_magnitude: number;
  top_pairs: { feature_x: string; feature_y: string; magnitude: number }[];
  grid: {
    x_centers: number[];
    y_centers: number[];
    z: (number | null)[][];
    counts: number[][];
    n_bins_x: number;
    n_bins_y: number;
  };
  pins: Pin[];
  tour_steps: TourStep[];
  n_samples: number;
  model_version: string;
};

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

const DEFAULT_CAMERA = { x: 1.5, y: -1.6, z: 0.9 };

export default function ShapInteractionSurface() {
  const [data, setData] = useState<ShapSurface | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tourStep, setTourStep] = useState(0); // 0 = inactive, 1..N = step

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/shap-surface`);
        if (!r.ok) {
          if (alive) setErr(`HTTP ${r.status}`);
          return;
        }
        const j = await r.json();
        if (alive) setData(j);
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : "fetch failed");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Recompute plotly traces only when data changes — not on every tour click.
  const traces = useMemo(() => {
    if (!data) return null;
    const fx = FEATURE_LABELS[data.feature_x] ?? data.feature_x;
    const fy = FEATURE_LABELS[data.feature_y] ?? data.feature_y;
    let zMin = 0;
    let zMax = 0;
    for (const row of data.grid.z) {
      for (const v of row) {
        if (v == null) continue;
        if (v < zMin) zMin = v;
        if (v > zMax) zMax = v;
      }
    }
    const zAbs = Math.max(Math.abs(zMin), Math.abs(zMax), 0.001);

    const surfaceTrace = {
      type: "surface" as const,
      x: data.grid.x_centers,
      y: data.grid.y_centers,
      z: data.grid.z,
      connectgaps: false,
      showscale: true,
      cmin: -zAbs,
      cmax: zAbs,
      colorscale: [
        [0.0, "#F87171"],
        [0.5, "#11151D"],
        [1.0, "#4ADE80"],
      ],
      colorbar: {
        title: { text: "Φ_ij", font: { color: "#A0A8B8", size: 11 } },
        tickfont: { color: "#A0A8B8", size: 10 },
        thickness: 14,
        len: 0.55,
        bgcolor: "#0A0D13",
        bordercolor: "#1F2533",
        borderwidth: 1,
      },
      hovertemplate: `<b>${fx}</b> %{x:.4f}<br><b>${fy}</b> %{y:.4f}<br><b>Φ_ij</b> %{z:.5f}<extra></extra>`,
      contours: {
        z: { show: true, usecolormap: true, project: { z: true }, width: 1.5 },
      },
      lighting: { ambient: 0.75, diffuse: 0.35, roughness: 0.85 },
    };

    return { fx, fy, surfaceTrace, zAbs };
  }, [data]);

  const pinTrace = useMemo(() => {
    if (!data) return null;
    return {
      type: "scatter3d" as const,
      mode: "markers+text" as const,
      x: data.pins.map((p) => p.x),
      y: data.pins.map((p) => p.y),
      z: data.pins.map((p) => p.z),
      text: data.pins.map((p) => `${p.ticker} · ${p.date.slice(5)}`),
      textfont: { color: "#FBBF24", size: 11, family: "Inter, sans-serif" },
      textposition: "top center" as const,
      marker: {
        size: 8,
        color: "#FBBF24",
        symbol: "diamond" as const,
        line: { color: "#0A0D13", width: 1 },
      },
      hovertemplate: data.pins.map((p) =>
        `<b>${p.ticker}</b> · ${p.date}<br>` +
        `<i>${p.label}</i><br>` +
        `basket return ${(p.basket_return * 100).toFixed(2)}%<br>` +
        `Φ_ij ${p.z.toFixed(5)}<br><br>` +
        `<i style="font-size:10px">${escapeHtml(p.lesson)}</i><extra></extra>`,
      ),
      name: "Failure cases",
      showlegend: false,
    };
  }, [data]);

  if (err) {
    return <SurfacePlaceholder text={`Error: ${err}`} />;
  }
  if (!data || !traces || !pinTrace) {
    return <SurfacePlaceholder text="Loading SHAP interaction data…" />;
  }

  const isTouring = tourStep > 0;
  const activeStep: TourStep | null =
    isTouring && data.tour_steps[tourStep - 1] ? data.tour_steps[tourStep - 1] : null;
  const camera = activeStep ? activeStep.camera : DEFAULT_CAMERA;
  const stepCount = data.tour_steps.length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {/* Tour control bar */}
      <div style={tourBarStyle}>
        {!isTouring ? (
          <>
            <span style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5, flex: 1 }}>
              <strong style={{ color: "var(--text-secondary)" }}>{traces.fx} × {traces.fy}</strong> —
              the dominant feature interaction in the model (top of 55 candidate pairs by mean
              |Φ_ij|, magnitude {data.interaction_magnitude.toFixed(5)}). Six failure-case picks
              are pinned. Hover any pin for the case lesson; orbit with click-and-drag.
            </span>
            <button onClick={() => setTourStep(1)} style={primaryButton}>
              ▶ Take the tour ({stepCount} steps)
            </button>
          </>
        ) : (
          <>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <span style={stepIndicator}>
                Step {tourStep} / {stepCount}
              </span>
              <strong style={{ fontSize: 12, color: "var(--text-primary)" }}>{activeStep?.title}</strong>
            </div>
            <div style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
              <button
                disabled={tourStep === 1}
                onClick={() => setTourStep((s) => Math.max(1, s - 1))}
                style={tourStep === 1 ? disabledButton : secondaryButton}
              >
                ← Prev
              </button>
              <button
                onClick={() => setTourStep(0)}
                style={secondaryButton}
              >
                Exit
              </button>
              <button
                onClick={() => {
                  if (tourStep < stepCount) setTourStep((s) => s + 1);
                  else setTourStep(0);
                }}
                style={primaryButton}
              >
                {tourStep < stepCount ? "Next →" : "Done"}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Tour caption */}
      {activeStep && (
        <div style={tourCaptionStyle}>{activeStep.caption}</div>
      )}

      {/* Plot */}
      <div style={{ height: 720, background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, overflow: "hidden" }}>
        <Plot
          data={[traces.surfaceTrace as unknown as Record<string, unknown>, pinTrace as unknown as Record<string, unknown>]}
          layout={{
            autosize: true,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: "#181D27",
            scene: {
              bgcolor: "#11151D",
              xaxis: {
                title: { text: traces.fx, font: { color: "#A0A8B8", size: 12 } },
                tickfont: { color: "#6A7488", size: 10 },
                gridcolor: "#1F2533",
                zerolinecolor: "#2D3548",
                color: "#A0A8B8",
              },
              yaxis: {
                title: { text: traces.fy, font: { color: "#A0A8B8", size: 12 } },
                tickfont: { color: "#6A7488", size: 10 },
                gridcolor: "#1F2533",
                zerolinecolor: "#2D3548",
                color: "#A0A8B8",
              },
              zaxis: {
                title: { text: "Φ_ij (SHAP interaction)", font: { color: "#A0A8B8", size: 11 } },
                tickfont: { color: "#6A7488", size: 10 },
                gridcolor: "#1F2533",
                zerolinecolor: "#2D3548",
                color: "#A0A8B8",
              },
              camera: { eye: camera },
              dragmode: "orbit" as const,
            },
            showlegend: false,
            hoverlabel: {
              bgcolor: "#0A0D13",
              bordercolor: "#2D3548",
              font: { color: "#F0F3F8", size: 11, family: "Inter, sans-serif" },
              align: "left" as const,
            },
          }}
          config={{
            displayModeBar: false,
            responsive: true,
            plotGlPixelRatio: 2,
          } as unknown as Record<string, unknown>}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>

      {/* Static footer */}
      <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.6, padding: "4px 2px" }}>
        Read: cross-partial Φ_ij from <code>TreeExplainer.shap_interaction_values</code> — not a
        smoothed partial-dependence surface. Empty cells (no observations binned there) appear
        as gaps; the model is a Random Forest and its response is piecewise-constant by
        construction. The three axes are mathematically load-bearing: collapsing to 2D shows
        only main effects (<code>f(x) + g(y)</code>), which cannot expose synergy or antagonism
        between the features.
      </div>
    </div>
  );
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function SurfacePlaceholder({ text }: { text: string }) {
  return (
    <div style={{
      height: 720,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: "var(--bg-elevated)",
      border: "1px solid var(--border)",
      borderRadius: 4,
      color: "var(--text-muted)",
      fontSize: 11,
    }}>
      {text}
    </div>
  );
}

const tourBarStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "10px 14px",
  background: "var(--bg-elevated)",
  border: "1px solid var(--border)",
  borderRadius: 4,
};

const tourCaptionStyle: React.CSSProperties = {
  padding: "12px 16px",
  background: "linear-gradient(90deg, rgba(34, 211, 238, 0.06), rgba(74, 222, 128, 0.04))",
  border: "1px solid rgba(34, 211, 238, 0.25)",
  borderRadius: 4,
  fontSize: 12,
  color: "var(--text-primary)",
  lineHeight: 1.6,
};

const primaryButton: React.CSSProperties = {
  background: "var(--green-bg)",
  color: "var(--green)",
  border: "1px solid rgba(74, 222, 128, 0.4)",
  borderRadius: 3,
  padding: "5px 12px",
  fontSize: 11,
  fontWeight: 700,
  cursor: "pointer",
  fontFamily: "inherit",
  letterSpacing: "0.02em",
  whiteSpace: "nowrap",
};

const secondaryButton: React.CSSProperties = {
  background: "var(--bg-row)",
  color: "var(--text-secondary)",
  border: "1px solid var(--border-strong)",
  borderRadius: 3,
  padding: "5px 12px",
  fontSize: 11,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "inherit",
  letterSpacing: "0.02em",
  whiteSpace: "nowrap",
};

const disabledButton: React.CSSProperties = {
  ...secondaryButton,
  opacity: 0.4,
  cursor: "not-allowed",
};

const stepIndicator: React.CSSProperties = {
  fontSize: 9,
  color: "var(--cyan)",
  background: "rgba(34, 211, 238, 0.1)",
  padding: "3px 8px",
  borderRadius: 3,
  fontWeight: 700,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
};
