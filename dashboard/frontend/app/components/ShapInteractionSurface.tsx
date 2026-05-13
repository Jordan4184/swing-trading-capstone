"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

// Dynamic import: plotly.js touches `window` at import time and will crash
// SSR. We also import the gl3d build (~1.2MB) rather than the full
// plotly.js-dist (~3.5MB).
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

export default function ShapInteractionSurface() {
  const [data, setData] = useState<ShapSurface | null>(null);
  const [err, setErr] = useState<string | null>(null);

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

  if (err) {
    return <SurfacePlaceholder text={`Error: ${err}`} />;
  }
  if (!data) {
    return <SurfacePlaceholder text="Loading SHAP interaction data…" />;
  }

  const fx = FEATURE_LABELS[data.feature_x] ?? data.feature_x;
  const fy = FEATURE_LABELS[data.feature_y] ?? data.feature_y;

  // Divergent symmetric color range so positive and negative interactions
  // read at equal saturation. Pick the larger of |min|/|max| as the range.
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
      [0.0, "#F87171"], // red — antagonism (Φ pushes proba down)
      [0.5, "#11151D"], // dark — near zero
      [1.0, "#4ADE80"], // green — synergy (Φ pushes proba up)
    ],
    colorbar: {
      title: { text: "Φ_ij", font: { color: "#A0A8B8", size: 10 } },
      tickfont: { color: "#A0A8B8", size: 9 },
      thickness: 12,
      len: 0.6,
      bgcolor: "#0A0D13",
    },
    hovertemplate: `${fx}: %{x:.4f}<br>${fy}: %{y:.4f}<br>Φ_ij: %{z:.5f}<extra></extra>`,
    contours: {
      z: { show: true, usecolormap: true, project: { z: true }, width: 1 },
    },
    lighting: { ambient: 0.7, diffuse: 0.4, roughness: 0.9 },
  };

  // Failure-case pins. Group them visually with a thin yellow line up from the
  // surface for legibility (a tiny stalk so labels don't get lost in the mesh).
  const pinTrace = {
    type: "scatter3d" as const,
    mode: "markers+text" as const,
    x: data.pins.map((p) => p.x),
    y: data.pins.map((p) => p.y),
    z: data.pins.map((p) => p.z),
    text: data.pins.map((p) => `${p.ticker} · ${p.date.slice(5)}`),
    textfont: { color: "#FBBF24", size: 10 },
    textposition: "top center" as const,
    marker: {
      size: 6,
      color: "#FBBF24",
      symbol: "diamond" as const,
      line: { color: "#FBBF24", width: 1 },
    },
    hovertemplate: data.pins
      .map(
        (p) =>
          `<b>${p.ticker}</b> · ${p.date}<br>${fx}: ${p.x.toFixed(4)}<br>${fy}: ${p.y.toFixed(4)}<br>Φ_ij: ${p.z.toFixed(5)}<br>basket: ${(p.basket_return * 100).toFixed(2)}%`,
      )
      .map((s) => s + "<extra></extra>"),
    name: "Failure cases",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5 }}>
        Mean SHAP interaction value <code>Φ_ij</code> across <strong>{data.n_samples.toLocaleString()}</strong> walk-forward
        predictions, binned 20×20 over the dominant feature pair (
        <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>{fx} × {fy}</span>
        ). Green = synergy (the pair&apos;s joint signal pushes proba up beyond their individual contributions);
        red = antagonism. The yellow diamonds are the picks from the three failure-mode case studies — they sit on
        the surface at the exact <code>(x, y, Φ)</code> the model saw at the moment of the bad decision.
      </div>
      <div style={{ height: 520, background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, overflow: "hidden" }}>
        <Plot
          data={[surfaceTrace as unknown as Record<string, unknown>, pinTrace as unknown as Record<string, unknown>]}
          layout={{
            autosize: true,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: "#181D27",
            scene: {
              bgcolor: "#11151D",
              xaxis: {
                title: { text: fx, font: { color: "#A0A8B8", size: 11 } },
                tickfont: { color: "#6A7488", size: 9 },
                gridcolor: "#1F2533",
                zerolinecolor: "#2D3548",
                color: "#A0A8B8",
              },
              yaxis: {
                title: { text: fy, font: { color: "#A0A8B8", size: 11 } },
                tickfont: { color: "#6A7488", size: 9 },
                gridcolor: "#1F2533",
                zerolinecolor: "#2D3548",
                color: "#A0A8B8",
              },
              zaxis: {
                title: { text: "Φ_ij (SHAP interaction)", font: { color: "#A0A8B8", size: 10 } },
                tickfont: { color: "#6A7488", size: 9 },
                gridcolor: "#1F2533",
                zerolinecolor: "#2D3548",
                color: "#A0A8B8",
              },
              camera: { eye: { x: 1.5, y: -1.6, z: 0.9 } },
              dragmode: "orbit" as const,
            },
            showlegend: false,
            hoverlabel: { bgcolor: "#11151D", bordercolor: "#2D3548", font: { color: "#F0F3F8", size: 11 } },
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
      <div style={{ fontSize: 10, color: "var(--text-faint)", lineHeight: 1.5, padding: "4px 0" }}>
        Read: this is the cross-partial Φ_ij from <code>TreeExplainer.shap_interaction_values</code> — not a smoothed
        partial-dependence surface. Empty cells (no observations binned there) appear as gaps; the model is a
        Random Forest and its response is piecewise-constant by construction. The three axes are mathematically
        load-bearing: collapsing to 2D shows only main effects (<code>f(x) + g(y)</code>), which cannot expose
        synergy or antagonism between the features.
      </div>
    </div>
  );
}

function SurfacePlaceholder({ text }: { text: string }) {
  return (
    <div style={{
      height: 520,
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
