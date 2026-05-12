"use client";

import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

type Bucket = {
  proba_bucket: string;
  count: number;
  actual_top_quintile_rate: number | null;
  se: number | null;
};

type CalibrationData = {
  baseline_rate: number;
  bins: number[];
  buckets: Bucket[];
  total_n: number;
};

// Module-level cache so the same data isn't re-fetched once per ribbon.
let _cache: CalibrationData | null = null;
let _inFlight: Promise<CalibrationData | null> | null = null;
const _listeners = new Set<() => void>();

async function loadCalibration(): Promise<CalibrationData | null> {
  if (_cache) return _cache;
  if (_inFlight) return _inFlight;
  _inFlight = (async () => {
    try {
      const r = await fetch(`${API_BASE}/api/calibration/buckets`);
      if (!r.ok) return null;
      _cache = await r.json();
      _listeners.forEach((fn) => fn());
      return _cache;
    } catch {
      return null;
    } finally {
      _inFlight = null;
    }
  })();
  return _inFlight;
}

function useCalibration(): CalibrationData | null {
  const [data, setData] = useState<CalibrationData | null>(_cache);
  useEffect(() => {
    if (_cache) {
      setData(_cache);
      return;
    }
    let alive = true;
    const sub = () => alive && setData(_cache);
    _listeners.add(sub);
    loadCalibration().then((d) => alive && setData(d));
    return () => {
      _listeners.delete(sub);
      alive = false;
    };
  }, []);
  return data;
}

function bucketFor(proba: number, data: CalibrationData): Bucket | null {
  // The backend bins are [0, 0.45, 0.50, 0.55, 0.60, 1.01], with labels
  // matching the order of data.buckets. Walk the bins to find the slot.
  for (let i = 0; i < data.bins.length - 1; i++) {
    const lo = data.bins[i];
    const hi = data.bins[i + 1];
    if (proba >= lo && proba < hi) return data.buckets[i] ?? null;
  }
  // Defensive: edge case proba == 1.0
  return data.buckets[data.buckets.length - 1] ?? null;
}

type Size = "xs" | "sm";

interface Props {
  proba: number;
  size?: Size;
  /** If true, the ribbon stretches to fill its container width. Defaults
   *  to a compact fixed-width layout suitable for inline use under a
   *  probability number. */
  block?: boolean;
}

export default function CalibrationRibbon({ proba, size = "sm", block = false }: Props) {
  const data = useCalibration();

  if (!data) {
    return (
      <span style={{ fontSize: size === "xs" ? 8 : 9, color: "var(--text-faint)" }}>
        cal …
      </span>
    );
  }

  const bucket = bucketFor(proba, data);
  if (!bucket || bucket.actual_top_quintile_rate == null) {
    return (
      <span style={{ fontSize: size === "xs" ? 8 : 9, color: "var(--text-faint)" }}>
        no cal data
      </span>
    );
  }

  const baseline = data.baseline_rate;
  const rate = bucket.actual_top_quintile_rate;
  const liftPp = (rate - baseline) * 100;
  // Color: green if meaningfully above baseline, red if below, muted otherwise.
  const color =
    liftPp > 3
      ? "var(--green)"
      : liftPp < -3
      ? "var(--red)"
      : "var(--text-muted)";

  // Visual scaling for the bar:
  // domain is 0..50% (well above any historical bucket). Baseline mark at 20%.
  const domain = 0.5;
  const barWidth = size === "xs" ? 36 : 56;
  const barHeight = size === "xs" ? 4 : 6;
  const fillPx = Math.min(barWidth, (rate / domain) * barWidth);
  const baselinePx = (baseline / domain) * barWidth;

  const fontSize = size === "xs" ? 8 : 9;

  return (
    <span
      style={{
        display: block ? "flex" : "inline-flex",
        alignItems: "center",
        gap: 5,
        fontSize,
        color: "var(--text-faint)",
        fontFeatureSettings: "'tnum'",
        whiteSpace: "nowrap",
        width: block ? "100%" : undefined,
      }}
      title={`${bucket.proba_bucket} bucket: ${(rate * 100).toFixed(1)}% historical top-quintile rate (n=${bucket.count.toLocaleString()}), baseline ${(baseline * 100).toFixed(0)}%`}
    >
      <span
        style={{
          position: "relative",
          width: barWidth,
          height: barHeight,
          background: "var(--bg-row)",
          borderRadius: 1,
          overflow: "hidden",
          flexShrink: 0,
        }}
      >
        <span
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: fillPx,
            background: color,
            opacity: 0.8,
          }}
        />
        <span
          style={{
            position: "absolute",
            left: baselinePx,
            top: -1,
            bottom: -1,
            width: 1,
            background: "var(--text-faint)",
          }}
        />
      </span>
      <span style={{ color }}>{(rate * 100).toFixed(0)}%</span>
      <span style={{ color: "var(--text-faint)" }}>n={bucket.count.toLocaleString()}</span>
    </span>
  );
}
