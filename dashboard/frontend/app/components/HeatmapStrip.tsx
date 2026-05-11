"use client";

/**
 * Renders a vertical probability strip alongside a candle chart.
 *
 * Y-axis is price (matches the parent chart\'s YAxis domain).
 * X-axis is probability (0.4 to 0.65 range typically).
 * Color: red (<0.45) -> neutral (0.45-0.55) -> green (>0.55).
 * Horizontal marker line shows current price + current probability.
 * Vertical dashed line at the 0.55 buy threshold.
 */

type HeatmapData = {
  ticker: string;
  prices: number[];
  probabilities: number[];
  current_price: number;
  current_idx: number;
  current_probability: number;
  min_probability: number;
  max_probability: number;
  buy_threshold: number;
  error?: string;
};

type Props = {
  data: HeatmapData | null | undefined;
  yMin: number;
  yMax: number;
  height: number;
  width?: number;
};

export default function HeatmapStrip({ data, yMin, yMax, height, width = 56 }: Props) {
  if (!data || data.error) {
    return (
      <div style={{
        width, height,
        background: "var(--bg-elevated)",
        border: "1px solid var(--border-soft)",
        borderRadius: 2,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-faint)", fontSize: 8,
        writingMode: "vertical-rl",
      }}>
        {data?.error ? "err" : "..."}
      </div>
    );
  }

  // Fixed probability domain. 0.4-0.65 covers the realistic range.
  const probMin = 0.40;
  const probMax = 0.65;
  const threshold = data.buy_threshold ?? 0.55;

  // Map probability to color (red -> neutral gray -> green)
  function colorFor(p: number): string {
    if (p >= 0.60) return "#4ADE80";
    if (p >= threshold) return "#86EFAC";
    if (p >= 0.50) return "#94A3B8";
    if (p >= 0.45) return "#FCA5A5";
    return "#F87171";
  }

  // SVG coords: x = probability, y = price
  // Note: SVG y-axis is inverted (0 at top), so we map high prices to top.
  const innerWidth = width - 2;
  const innerHeight = height - 2;

  const xFor = (prob: number) =>
    1 + ((prob - probMin) / (probMax - probMin)) * (innerWidth - 1);
  const yFor = (price: number) =>
    1 + ((yMax - price) / (yMax - yMin)) * (innerHeight - 1);

  // Threshold x position
  const thresholdX = xFor(threshold);

  // Build the curve path - one point per price level
  // Reverse the data because SVG goes top-down but prices go bottom-up
  const pathPoints = data.prices.map((price, i) => {
    const x = xFor(data.probabilities[i]);
    const y = yFor(price);
    return { x, y, prob: data.probabilities[i] };
  });

  // Sort by y (SVG top-to-bottom = high price to low price)
  pathPoints.sort((a, b) => a.y - b.y);

  // Path string
  const pathD = pathPoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
    .join(" ");

  // Filled area under curve (to threshold line if above, or to left edge)
  const fillD = [
    `M 1 ${pathPoints[0].y}`,
    ...pathPoints.map((p) => `L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`),
    `L 1 ${pathPoints[pathPoints.length - 1].y}`,
    "Z",
  ].join(" ");

  const currentY = yFor(data.current_price);
  const currentX = xFor(data.current_probability);
  const currentColor = colorFor(data.current_probability);

  return (
    <div style={{
      width, height,
      background: "var(--bg-elevated)",
      border: "1px solid var(--border-soft)",
      borderRadius: 2,
      position: "relative",
      overflow: "hidden",
    }}>
      <svg width={width} height={height} style={{ display: "block" }}>
        {/* Background gradient bands - subtle vertical color zones */}
        <defs>
          <linearGradient id={`gradient-${data.ticker}`} x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#F87171" stopOpacity="0.08" />
            <stop offset={`${((0.50 - probMin) / (probMax - probMin)) * 100}%`} stopColor="#94A3B8" stopOpacity="0.05" />
            <stop offset={`${((threshold - probMin) / (probMax - probMin)) * 100}%`} stopColor="#94A3B8" stopOpacity="0.05" />
            <stop offset={`${((threshold - probMin) / (probMax - probMin)) * 100}%`} stopColor="#4ADE80" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#4ADE80" stopOpacity="0.15" />
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={width} height={height} fill={`url(#gradient-${data.ticker})`} />

        {/* Buy threshold dashed vertical line at 0.55 */}
        <line
          x1={thresholdX} x2={thresholdX} y1={0} y2={height}
          stroke="#86EFAC" strokeWidth={0.5} strokeDasharray="2 2" opacity={0.6}
        />

        {/* Filled area under probability curve */}
        <path d={fillD} fill={currentColor} fillOpacity={0.2} />

        {/* Probability curve */}
        <path d={pathD} fill="none" stroke={currentColor} strokeWidth={1.2} opacity={0.9} />

        {/* Current price marker - horizontal line + dot */}
        <line
          x1={0} x2={width} y1={currentY} y2={currentY}
          stroke="var(--text-faint)" strokeWidth={0.4} strokeDasharray="1 2"
        />
        <circle cx={currentX} cy={currentY} r={2.5} fill={currentColor} stroke="#000" strokeWidth={0.5} />

        {/* Probability label at top */}
        <text x={width - 2} y={9} fontSize={8} fill="var(--text-muted)" textAnchor="end" fontWeight={600}>
          {(data.current_probability * 100).toFixed(0)}%
        </text>
      </svg>
    </div>
  );
}
