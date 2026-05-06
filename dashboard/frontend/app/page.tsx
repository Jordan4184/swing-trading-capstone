"use client";

import { useEffect, useState } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type Prediction = {
  date: string;
  ticker: string;
  y_true: number;
  y_proba: number;
  fwd_return_5d: number;
  fold: number;
};

type LatestPredictionsResponse = {
  date: string;
  predictions: Prediction[];
};

type StrategyMetrics = {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  hit_rate: number;
};

type BacktestSummary = {
  strategy: StrategyMetrics;
  equal_weight: StrategyMetrics;
  spy: StrategyMetrics;
};

type EquityPoint = {
  date: string;
  equity: number;
  return: number;
};

type EquityCurveResponse = {
  config: {
    top_n: number;
    holding_days: number;
    cost_bps: number;
    initial_capital: number;
  };
  n_trades: number;
  data: EquityPoint[];
};

type CombinedEquityPoint = {
  date: string;
  production: number;
  simulated: number | null;
};

type CombinedDrawdownPoint = {
  date: string;
  productionDD: number;
  simulatedDD: number | null;
};

const API_BASE = "http://localhost:8000";

function computeMetrics(curve: EquityPoint[]) {
  if (curve.length === 0) {
    return { totalReturn: 0, sharpe: 0, maxDD: 0, finalEquity: 0 };
  }
  const initial = 10000;
  const final = curve[curve.length - 1].equity;
  const totalReturn = final / initial - 1;

  const returns = curve.map((p) => p.return);
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance =
    returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;
  const std = Math.sqrt(variance);
  const periodsPerYear = curve.length > 1
    ? 252 / Math.round(
        (new Date(curve[1].date).getTime() - new Date(curve[0].date).getTime()) /
        (1000 * 60 * 60 * 24)
      )
    : 50;
  const sharpe = std > 0 ? (mean / std) * Math.sqrt(periodsPerYear) : 0;

  let peak = curve[0].equity;
  let maxDD = 0;
  for (const p of curve) {
    if (p.equity > peak) peak = p.equity;
    const dd = (p.equity - peak) / peak;
    if (dd < maxDD) maxDD = dd;
  }

  return { totalReturn, sharpe, maxDD, finalEquity: final };
}

// Compute drawdown at each point: (equity - running_peak) / running_peak
function computeDrawdownSeries(curve: EquityPoint[]): { date: string; dd: number }[] {
  const result: { date: string; dd: number }[] = [];
  let peak = curve[0]?.equity ?? 0;
  for (const p of curve) {
    if (p.equity > peak) peak = p.equity;
    const dd = peak > 0 ? (p.equity - peak) / peak : 0;
    result.push({ date: p.date, dd });
  }
  return result;
}

function mergeEquityCurves(
  production: EquityPoint[],
  simulated: EquityPoint[] | null
): CombinedEquityPoint[] {
  if (!simulated) {
    return production.map((p) => ({
      date: p.date,
      production: p.equity,
      simulated: null,
    }));
  }
  const simByDate = new Map(simulated.map((p) => [p.date, p.equity]));
  return production.map((p) => ({
    date: p.date,
    production: p.equity,
    simulated: simByDate.get(p.date) ?? null,
  }));
}

function mergeDrawdownCurves(
  production: EquityPoint[],
  simulated: EquityPoint[] | null
): CombinedDrawdownPoint[] {
  const prodDD = computeDrawdownSeries(production);
  if (!simulated) {
    return prodDD.map((p) => ({
      date: p.date,
      productionDD: p.dd,
      simulatedDD: null,
    }));
  }
  const simDD = computeDrawdownSeries(simulated);
  const simByDate = new Map(simDD.map((p) => [p.date, p.dd]));
  return prodDD.map((p) => ({
    date: p.date,
    productionDD: p.dd,
    simulatedDD: simByDate.get(p.date) ?? null,
  }));
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [latest, setLatest] = useState<LatestPredictionsResponse | null>(null);
  const [productionCurve, setProductionCurve] = useState<EquityCurveResponse | null>(null);
  const [simulatedCurve, setSimulatedCurve] = useState<EquityCurveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [simTopN, setSimTopN] = useState(2);
  const [simHolding, setSimHolding] = useState(5);
  const [simCost, setSimCost] = useState(10);
  const [simRunning, setSimRunning] = useState(false);

  useEffect(() => {
    async function loadData() {
      try {
        const [summaryRes, latestRes, equityRes] = await Promise.all([
          fetch(`${API_BASE}/api/summary`),
          fetch(`${API_BASE}/api/predictions/latest?top_n=5`),
          fetch(`${API_BASE}/api/equity-curve`),
        ]);

        if (!summaryRes.ok || !latestRes.ok || !equityRes.ok) {
          throw new Error("Failed to fetch from backend");
        }

        setSummary(await summaryRes.json());
        setLatest(await latestRes.json());
        setProductionCurve(await equityRes.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    loadData();
  }, []);

  async function runSimulation() {
    setSimRunning(true);
    try {
      const url = `${API_BASE}/api/equity-curve?top_n=${simTopN}&holding_days=${simHolding}&cost_bps=${simCost}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Simulation failed");
      const data = await res.json();
      setSimulatedCurve(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation error");
    } finally {
      setSimRunning(false);
    }
  }

  function clearSimulation() {
    setSimulatedCurve(null);
  }

  if (error) {
    return (
      <main className="min-h-screen bg-slate-950 text-red-400 p-8">
        <h1 className="text-2xl font-bold">Error</h1>
        <p className="mt-4">{error}</p>
        <p className="mt-4 text-slate-400 text-sm">
          Is the FastAPI backend running on port 8000?
        </p>
      </main>
    );
  }

  if (!summary || !latest || !productionCurve) {
    return (
      <main className="min-h-screen bg-slate-950 text-slate-400 p-8 flex items-center justify-center">
        <p>Loading dashboard...</p>
      </main>
    );
  }

  const s = summary.strategy;
  const ew = summary.equal_weight;
  const alphaPP = (s.annualized_return - ew.annualized_return) * 100;
  const equityChartData = mergeEquityCurves(
    productionCurve.data,
    simulatedCurve?.data ?? null
  );
  const drawdownChartData = mergeDrawdownCurves(
    productionCurve.data,
    simulatedCurve?.data ?? null
  );
  const simMetrics = simulatedCurve
    ? computeMetrics(simulatedCurve.data)
    : null;

  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Swing Trading Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">
          ML-based cross-sectional strategy &middot; As of {latest.date}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total Return"
          value={`${(s.total_return * 100).toFixed(1)}%`}
          subtitle={`vs SPY: +${((s.total_return - summary.spy.total_return) * 100).toFixed(0)}pp`}
          accent="green"
        />
        <StatCard
          label="Annualized"
          value={`${(s.annualized_return * 100).toFixed(1)}%`}
          subtitle={`+${alphaPP.toFixed(2)}pp vs equal-weight`}
          accent="green"
        />
        <StatCard
          label="Sharpe Ratio"
          value={s.sharpe_ratio.toFixed(2)}
          subtitle={`vs EW: +${(s.sharpe_ratio - ew.sharpe_ratio).toFixed(2)}`}
          accent="blue"
        />
        <StatCard
          label="Max Drawdown"
          value={`${(s.max_drawdown * 100).toFixed(1)}%`}
          subtitle="concentration cost"
          accent="red"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-8">
        <div className="lg:col-span-1 bg-slate-900 rounded-lg p-6 border border-slate-800">
          <h2 className="text-lg font-semibold mb-1">Strategy Simulator</h2>
          <p className="text-slate-500 text-xs mb-6">
            Adjust parameters and re-run the backtest. Production curve stays as reference.
          </p>

          <SliderControl
            label="TOP_N"
            sublabel="names held"
            value={simTopN}
            min={1}
            max={5}
            step={1}
            onChange={setSimTopN}
          />

          <SliderControl
            label="HOLDING_DAYS"
            sublabel="rebalance period"
            value={simHolding}
            min={1}
            max={20}
            step={1}
            onChange={setSimHolding}
          />

          <SliderControl
            label="COST"
            sublabel="round-trip bps"
            value={simCost}
            min={0}
            max={50}
            step={1}
            onChange={setSimCost}
          />

          <button
            onClick={runSimulation}
            disabled={simRunning}
            className="w-full mt-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded transition"
          >
            {simRunning ? "Running..." : "Run Simulation"}
          </button>

          {simulatedCurve && simMetrics && (
            <>
              <button
                onClick={clearSimulation}
                className="w-full mt-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm py-2 px-4 rounded transition"
              >
                Clear simulation
              </button>

              <div className="mt-6 pt-4 border-t border-slate-800 space-y-2 text-sm">
                <p className="text-slate-400 text-xs uppercase tracking-wide mb-2">
                  Sim Result
                </p>
                <Stat label="Total Return" value={`${(simMetrics.totalReturn * 100).toFixed(1)}%`} />
                <Stat label="Sharpe (approx)" value={simMetrics.sharpe.toFixed(2)} />
                <Stat label="Max Drawdown" value={`${(simMetrics.maxDD * 100).toFixed(1)}%`} />
                <Stat label="Trades" value={simulatedCurve.n_trades.toString()} />
                <Stat label="Final Equity" value={`$${simMetrics.finalEquity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />

                <div className="pt-2 mt-2 border-t border-slate-800">
                  <p className="text-slate-500 text-xs">vs production:</p>
                  <Stat
                    label="Return delta"
                    value={`${((simMetrics.totalReturn - (productionCurve.data[productionCurve.data.length - 1].equity / 10000 - 1)) * 100).toFixed(1)}pp`}
                  />
                </div>
              </div>
            </>
          )}
        </div>

        <div className="lg:col-span-3 space-y-4">
          <div className="bg-slate-900 rounded-lg p-6 border border-slate-800">
            <div className="flex justify-between items-baseline mb-4">
              <div>
                <h2 className="text-xl font-semibold">Equity Curve</h2>
                <p className="text-slate-500 text-xs mt-1">
                  Production: TOP_N={productionCurve.config.top_n}, HOLD={productionCurve.config.holding_days}d, cost={productionCurve.config.cost_bps}bps
                  {simulatedCurve && (
                    <>
                      {" \u00B7 "}
                      Sim: TOP_N={simulatedCurve.config.top_n}, HOLD={simulatedCurve.config.holding_days}d, cost={simulatedCurve.config.cost_bps}bps
                    </>
                  )}
                </p>
              </div>
              {simulatedCurve && (
                <span className="text-xs bg-amber-500/20 text-amber-300 px-2 py-1 rounded border border-amber-500/30">
                  COMPARING
                </span>
              )}
            </div>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={equityChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="productionGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="date"
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(d: string) => d.slice(0, 7)}
                    interval={Math.floor(equityChartData.length / 8)}
                  />
                  <YAxis
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "0.5rem",
                      color: "#e2e8f0",
                    }}
                    formatter={(value: number | null, name: string) =>
                      value === null ? ["—", name] : [`$${value.toLocaleString()}`, name]
                    }
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 12, paddingTop: 10 }}
                    iconType="line"
                  />
                  <Area
                    type="monotone"
                    dataKey="production"
                    name="Production strategy"
                    stroke="#10b981"
                    strokeWidth={2}
                    fill="url(#productionGradient)"
                  />
                  <Line
                    type="monotone"
                    dataKey="simulated"
                    name="Simulation"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    strokeDasharray="6 4"
                    dot={false}
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-slate-900 rounded-lg p-6 border border-slate-800">
            <div className="flex justify-between items-baseline mb-4">
              <div>
                <h2 className="text-xl font-semibold">Drawdown</h2>
                <p className="text-slate-500 text-xs mt-1">
                  Underwater chart &middot; peak-to-trough decline at each point
                </p>
              </div>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={drawdownChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.05} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0.4} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="date"
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(d: string) => d.slice(0, 7)}
                    interval={Math.floor(drawdownChartData.length / 8)}
                  />
                  <YAxis
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    domain={[(dataMin: number) => Math.floor(dataMin * 10) / 10, 0]}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "0.5rem",
                      color: "#e2e8f0",
                    }}
                    formatter={(value: number | null, name: string) =>
                      value === null ? ["—", name] : [`${(value * 100).toFixed(2)}%`, name]
                    }
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 12, paddingTop: 10 }}
                    iconType="line"
                  />
                  <Area
                    type="monotone"
                    dataKey="productionDD"
                    name="Production drawdown"
                    stroke="#ef4444"
                    strokeWidth={2}
                    fill="url(#drawdownGradient)"
                  />
                  <Line
                    type="monotone"
                    dataKey="simulatedDD"
                    name="Simulated drawdown"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    strokeDasharray="6 4"
                    dot={false}
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-900 rounded-lg p-6 border border-slate-800">
        <h2 className="text-xl font-semibold mb-4">
          Latest Signals &middot; Top {latest.predictions.length}
        </h2>
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase">
              <th className="pb-2">Rank</th>
              <th className="pb-2">Ticker</th>
              <th className="pb-2 text-right">Buy Probability</th>
              <th className="pb-2 text-right">Realized 5d Return</th>
            </tr>
          </thead>
          <tbody>
            {latest.predictions.map((p, idx) => (
              <tr key={p.ticker} className="border-b border-slate-800 last:border-0">
                <td className="py-3 text-slate-500">{idx + 1}</td>
                <td className="py-3 font-bold">{p.ticker}</td>
                <td className="py-3 text-right font-mono">
                  {p.y_proba.toFixed(3)}
                </td>
                <td
                  className={`py-3 text-right font-mono ${
                    p.fwd_return_5d >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {(p.fwd_return_5d * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-8 text-xs text-slate-600">
        Backend: {API_BASE} &middot; Next.js + FastAPI
      </p>
    </main>
  );
}

function StatCard({
  label,
  value,
  subtitle,
  accent,
}: {
  label: string;
  value: string;
  subtitle: string;
  accent: "green" | "blue" | "red";
}) {
  const accentClasses = {
    green: "text-green-400",
    blue: "text-blue-400",
    red: "text-red-400",
  };

  return (
    <div className="bg-slate-900 rounded-lg p-5 border border-slate-800">
      <p className="text-slate-400 text-xs uppercase tracking-wide">{label}</p>
      <p className={`text-3xl font-bold mt-2 ${accentClasses[accent]}`}>
        {value}
      </p>
      <p className="text-slate-500 text-xs mt-1">{subtitle}</p>
    </div>
  );
}

function SliderControl({
  label,
  sublabel,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  sublabel: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-5">
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
        <span className="text-lg font-mono font-bold text-emerald-400">{value}</span>
      </div>
      <p className="text-xs text-slate-500 mb-2">{sublabel}</p>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-emerald-500"
      />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}
