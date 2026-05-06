"use client";

import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
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

const API_BASE = "http://localhost:8000";

export default function DashboardPage() {
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [latest, setLatest] = useState<LatestPredictionsResponse | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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

        const summaryData = await summaryRes.json();
        const latestData = await latestRes.json();
        const equityData = await equityRes.json();

        setSummary(summaryData);
        setLatest(latestData);
        setEquityCurve(equityData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    loadData();
  }, []);

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

  if (!summary || !latest || !equityCurve) {
    return (
      <main className="min-h-screen bg-slate-950 text-slate-400 p-8 flex items-center justify-center">
        <p>Loading dashboard...</p>
      </main>
    );
  }

  const s = summary.strategy;
  const ew = summary.equal_weight;
  const alphaPP = (s.annualized_return - ew.annualized_return) * 100;

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

      <div className="bg-slate-900 rounded-lg p-6 border border-slate-800 mb-8">
        <div className="flex justify-between items-baseline mb-4">
          <h2 className="text-xl font-semibold">Strategy Equity Curve</h2>
          <p className="text-slate-500 text-xs">
            {equityCurve.n_trades} trades &middot;
            TOP_N={equityCurve.config.top_n} &middot;
            HOLD={equityCurve.config.holding_days}d &middot;
            cost={equityCurve.config.cost_bps}bps
          </p>
        </div>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityCurve.data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
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
                interval={Math.floor(equityCurve.data.length / 8)}
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
                formatter={(value: number) => [`$${value.toLocaleString()}`, "Equity"]}
              />
              <Area
                type="monotone"
                dataKey="equity"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#equityGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
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
