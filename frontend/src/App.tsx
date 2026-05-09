import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Boxes,
  BrainCircuit,
  CircuitBoard,
  Database,
  Gauge,
  LineChart,
  Play,
  Radio,
  ShieldCheck,
  TrendingUp
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart as ReLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { fetchDashboard } from "./api";
import { formatCompact, formatCurrency, formatMetric, labelForStrategy } from "./format";
import type { BacktestRow, DashboardData, MetricCard, ModelComparison } from "./types";

const chartColors: Record<string, string> = {
  buy_hold: "#67E8F9",
  sma_crossover: "#34D399",
  momentum: "#FBBF24",
  random: "#FB7185"
};

export function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [data, setData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    fetchDashboard(ticker)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Dashboard unavailable");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  const topBacktests = useMemo(() => {
    if (!data) return [];
    return [...data.backtests].sort((a, b) => b.sharpe - a.sharpe);
  }, [data]);

  if (error) {
    return (
      <Shell>
        <div className="flex h-full items-center justify-center">
          <div className="panel max-w-xl p-8">
            <div className="text-lg font-semibold text-white">Dashboard API unavailable</div>
            <div className="mt-3 text-sm text-slate-400">{error}</div>
            <div className="mt-5 rounded border border-slate-700 bg-graphite-900 p-4 text-sm text-slate-300">
              Start the backend with <span className="font-mono text-cyan-200">uvicorn backend.main:app --reload</span>.
            </div>
          </div>
        </div>
      </Shell>
    );
  }

  return (
    <Shell>
      <main className="min-h-screen flex-1 overflow-y-auto px-5 py-5 lg:px-7">
        <TopBar data={data} ticker={ticker} setTicker={setTicker} isLoading={isLoading} />
        {data ? (
          <div className="mt-5 grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
            <section className="space-y-5">
              <PortfolioHeader data={data} />
              <MetricGrid metrics={data.metrics} />
              <Panel title="Equity Curve" icon={<LineChart size={17} />} action={data.portfolio.strategy}>
                <div className="h-[330px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ReLineChart data={data.equityCurve} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="#1F2B3B" strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fill: "#94A3B8", fontSize: 11 }} minTickGap={30} />
                      <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} tickFormatter={formatCompact} width={54} />
                      <Tooltip content={<ChartTooltip />} />
                      {Object.keys(chartColors).map((strategy) => (
                        <Line
                          key={strategy}
                          type="monotone"
                          dataKey={strategy}
                          stroke={chartColors[strategy]}
                          strokeWidth={strategy === data.portfolio.strategy ? 2.8 : 1.6}
                          dot={false}
                          name={labelForStrategy(strategy)}
                        />
                      ))}
                    </ReLineChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
              <div className="grid gap-5 lg:grid-cols-2">
                <Panel title="Drawdown Surface" icon={<Gauge size={17} />}>
                  <div className="h-[260px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={data.drawdownCurve} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
                        <defs>
                          <linearGradient id="drawdownFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#FB7185" stopOpacity={0.7} />
                            <stop offset="100%" stopColor="#FB7185" stopOpacity={0.05} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#1F2B3B" strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={{ fill: "#94A3B8", fontSize: 10 }} minTickGap={32} />
                        <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
                        <Tooltip content={<ChartTooltip percentKeys={["drawdown"]} />} />
                        <Area type="monotone" dataKey="drawdown" stroke="#FB7185" fill="url(#drawdownFill)" strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
                <Panel title="Reward & Evaluation Analytics" icon={<Activity size={17} />}>
                  <TrainingAnalyticsPanel data={data} />
                </Panel>
              </div>
              <BacktestTable rows={topBacktests} />
            </section>
            <aside className="space-y-5">
              <ModelComparisonPanel models={data.modelComparison} />
              <ExperimentPanel data={data} />
              <TradeTimeline events={data.tradeTimeline} />
              <InferencePanel data={data} />
              <Safeguards items={data.safeguards} />
            </aside>
          </div>
        ) : (
          <SkeletonDashboard />
        )}
      </main>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-graphite-950 text-slate-100">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(103,232,249,0.10),transparent_34%),linear-gradient(135deg,rgba(52,211,153,0.08),transparent_38%)]" />
      <div className="relative flex min-h-screen">
        <aside className="hidden w-[84px] border-r border-slate-800/80 bg-graphite-900/78 px-4 py-5 backdrop-blur xl:block">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-cyan-300/30 bg-cyan-300/10 text-cyan-200">
            <CircuitBoard size={22} />
          </div>
          <nav className="mt-8 space-y-3">
            {[BarChart3, BrainCircuit, Database, ShieldCheck, Radio].map((Icon, index) => (
              <button
                key={index}
                className={`flex h-11 w-11 items-center justify-center rounded-lg border transition ${
                  index === 0
                    ? "border-cyan-300/40 bg-cyan-300/10 text-cyan-100"
                    : "border-transparent text-slate-500 hover:border-slate-700 hover:bg-graphite-800 hover:text-slate-200"
                }`}
                aria-label={`Navigation ${index + 1}`}
              >
                <Icon size={19} />
              </button>
            ))}
          </nav>
        </aside>
        {children}
      </div>
    </div>
  );
}

function TopBar({
  data,
  ticker,
  setTicker,
  isLoading
}: {
  data: DashboardData | null;
  ticker: string;
  setTicker: (ticker: string) => void;
  isLoading: boolean;
}) {
  return (
    <header className="flex flex-col gap-4 border-b border-slate-800/70 pb-5 lg:flex-row lg:items-center lg:justify-between">
      <div>
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-300/10 text-emerald-200 xl:hidden">
            <CircuitBoard size={20} />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-normal text-white sm:text-2xl">Deep RL Trading Agent</h1>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-400">
              <span>{data?.mode.dataSource ?? "Local trading intelligence API"}</span>
              <span className="h-1 w-1 rounded-full bg-slate-600" />
              <span>{data?.mode.label ?? "Loading"}</span>
              {data?.mode.vecNormalize ? <span className="status-chip cyan">VecNormalize</span> : null}
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={ticker}
          onChange={(event) => setTicker(event.target.value)}
          className="h-10 rounded-lg border border-slate-700 bg-graphite-900 px-3 text-sm text-slate-100 outline-none transition focus:border-cyan-300"
        >
          {(data?.tickers ?? ["AAPL", "MSFT", "SPY", "BTC-USD"]).map((symbol) => (
            <option key={symbol} value={symbol}>
              {symbol}
            </option>
          ))}
        </select>
        <div className="status-chip emerald">
          <span className={`h-2 w-2 rounded-full ${isLoading ? "bg-amber-300" : "bg-emerald-300"}`} />
          {isLoading ? "Syncing" : "Online"}
        </div>
      </div>
    </header>
  );
}

function PortfolioHeader({ data }: { data: DashboardData }) {
  return (
    <section className="panel overflow-hidden p-5">
      <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="text-sm font-medium text-cyan-200">{data.portfolio.label}</div>
          <div className="mt-2 flex flex-wrap items-end gap-3">
            <div className="text-4xl font-semibold text-white">{formatCurrency(data.portfolio.endingValue)}</div>
            <div className={`pb-1 text-sm font-semibold ${data.portfolio.totalReturn >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
              {formatMetric(data.portfolio.totalReturn, "percent")}
            </div>
          </div>
          <div className="mt-3 text-sm text-slate-400">
            Selected benchmark: <span className="text-slate-200">{labelForStrategy(data.portfolio.strategy)}</span>
          </div>
        </div>
        <div className="grid min-w-[280px] grid-cols-2 gap-3">
              <MiniStat label="Evaluation capital" value={formatCurrency(data.portfolio.initialValue)} />
          <MiniStat label="Runtime" value={data.portfolio.status} />
        </div>
      </div>
    </section>
  );
}

function MetricGrid({ metrics }: { metrics: MetricCard[] }) {
  return (
    <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
      {metrics.map((metric) => (
        <div key={metric.label} className="panel p-4">
          <div className="text-xs font-medium uppercase text-slate-500">{metric.label}</div>
          <div className="mt-3 text-2xl font-semibold text-white">{formatMetric(metric.value, metric.format)}</div>
        </div>
      ))}
    </section>
  );
}

function Panel({
  title,
  icon,
  action,
  children
}: {
  title: string;
  icon: React.ReactNode;
  action?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="panel p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          <span className="text-cyan-200">{icon}</span>
          {title}
        </div>
        {action ? <span className="text-xs text-slate-500">{labelForStrategy(action)}</span> : null}
      </div>
      {children}
    </section>
  );
}

function ModelComparisonPanel({ models }: { models: ModelComparison[] }) {
  return (
    <Panel title="PPO vs SAC" icon={<BrainCircuit size={17} />}>
      <div className="space-y-3">
        {models.map((model) => (
          <div key={model.name} className="rounded-lg border border-slate-800 bg-graphite-900/70 p-4">
            <div className="flex items-center justify-between">
              <div className="text-lg font-semibold text-white">{model.name}</div>
              <span className={`status-chip ${model.status === "ready" ? "emerald" : "amber"}`}>{model.status}</span>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <MiniStat label="Policy" value={model.policy} />
              <MiniStat label="Timesteps" value={formatCompact(model.timesteps)} />
              <MiniStat label="LR" value={model.learningRate ? model.learningRate.toExponential(1) : "N/A"} />
              <MiniStat label="Reward" value={model.reward} />
            </div>
            <div className="mt-3 text-xs leading-5 text-slate-400">{model.notes}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function ExperimentPanel({ data }: { data: DashboardData }) {
  return (
    <Panel title="Experiment Tracking" icon={<Boxes size={17} />}>
      <div className="space-y-3">
        {data.experiments.map((item) => (
          <div key={item.name} className="flex items-start gap-3 rounded-lg border border-slate-800 bg-graphite-900/60 p-3">
            <div className={`mt-1 h-2.5 w-2.5 rounded-full ${item.status === "ready" ? "bg-emerald-300" : item.status === "pending" ? "bg-amber-300" : "bg-cyan-300"}`} />
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-100">{item.name}</div>
              <div className="mt-1 text-xs text-slate-500">{item.detail}</div>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function TradeTimeline({ events }: { events: DashboardData["tradeTimeline"] }) {
  return (
    <Panel title="Trade Timeline" icon={<TrendingUp size={17} />}>
      <div className="space-y-3">
        {events.map((event) => (
          <div key={`${event.date}-${event.equity}`} className="relative border-l border-slate-800 pl-4">
            <div className="absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full bg-cyan-300" />
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-medium text-slate-100">{event.side}</div>
              <div className="text-xs text-slate-500">{event.date}</div>
            </div>
            <div className="mt-1 flex justify-between text-xs text-slate-400">
              <span>Position {event.position.toFixed(2)}</span>
              <span>{formatCurrency(event.equity)}</span>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function InferencePanel({ data }: { data: DashboardData }) {
  return (
    <Panel title="Inference Demo" icon={<Play size={17} />}>
      <div className="mb-4 flex items-center justify-between">
        <div className="text-sm text-slate-300">{labelForStrategy(data.inferenceDemo.strategy)}</div>
        <span className="status-chip cyan">{data.inferenceDemo.mode}</span>
      </div>
      <div className="mb-3 text-xs leading-5 text-slate-500">{data.inferenceDemo.message}</div>
      <div className="space-y-2">
        {data.inferenceDemo.steps.map((step) => (
          <div key={step.date} className="grid grid-cols-[1fr_64px_74px] items-center gap-2 rounded-lg bg-graphite-900/70 px-3 py-2 text-xs">
            <div>
              <div className="font-medium text-slate-200">{step.date}</div>
              <div className="text-slate-500">Close {step.close.toFixed(2)}</div>
            </div>
            <div className="text-right text-cyan-200">{step.action.toFixed(2)}</div>
            <div className="text-right text-slate-300">{formatCompact(step.equity)}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function TrainingAnalyticsPanel({ data }: { data: DashboardData }) {
  const analytics = Array.isArray(data.trainingAnalytics)
    ? {
        label: "Legacy telemetry shape detected",
        points: data.trainingAnalytics,
        channels: ["policy reward", "validation Sharpe", "drawdown", "turnover", "evaluation return"]
      }
    : data.trainingAnalytics;
  if (analytics.points.length === 0) {
    return (
      <div className="flex min-h-[260px] flex-col justify-between rounded-lg border border-slate-800 bg-graphite-900/60 p-4">
        <div>
          <div className="status-chip amber">Awaiting telemetry</div>
          <div className="mt-4 text-lg font-semibold text-white">{analytics.label}</div>
          <div className="mt-2 text-sm leading-6 text-slate-400">
            This panel is reserved for checkpoint-backed SB3/W&B training curves. No reward or Sharpe trajectory is simulated here.
          </div>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-2 2xl:grid-cols-2">
          {analytics.channels.map((channel) => (
            <div key={channel} className="rounded-lg border border-slate-800 bg-graphite-950/60 px-3 py-2 text-xs text-slate-400">
              {channel}
            </div>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <ReLineChart data={analytics.points} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#1F2B3B" strokeDasharray="3 3" />
          <XAxis dataKey="step" tick={{ fill: "#94A3B8", fontSize: 10 }} tickFormatter={formatCompact} />
          <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} />
          <Tooltip content={<ChartTooltip />} />
          <Line type="monotone" dataKey="ppoReward" stroke="#67E8F9" strokeWidth={2} dot={false} name="PPO Reward" />
          <Line type="monotone" dataKey="sacReward" stroke="#34D399" strokeWidth={2} dot={false} name="SAC Reward" />
          <Line type="monotone" dataKey="evalSharpe" stroke="#FBBF24" strokeWidth={2} dot={false} name="Eval Sharpe" />
        </ReLineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Safeguards({ items }: { items: string[] }) {
  return (
    <Panel title="Validation Guardrails" icon={<ShieldCheck size={17} />}>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item} className="flex gap-2 text-xs leading-5 text-slate-400">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-300" />
            <span>{item}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function BacktestTable({ rows }: { rows: BacktestRow[] }) {
  return (
    <Panel title="Backtesting Results" icon={<BarChart3 size={17} />}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr className="border-b border-slate-800">
              <th className="pb-3 font-medium">Strategy</th>
              <th className="pb-3 font-medium">Return</th>
              <th className="pb-3 font-medium">Sharpe</th>
              <th className="pb-3 font-medium">Max DD</th>
              <th className="pb-3 font-medium">Trades</th>
              <th className="pb-3 font-medium">Hold</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.ticker}-${row.strategy}`} className="border-b border-slate-900 text-slate-300">
                <td className="py-3">
                  <div className="font-medium text-slate-100">{labelForStrategy(row.strategy)}</div>
                  <div className="text-xs text-slate-500">{row.ticker}</div>
                </td>
                <td className="py-3 text-emerald-300">{formatMetric(row.annualized_return, "percent")}</td>
                <td className="py-3">{row.sharpe.toFixed(2)}</td>
                <td className="py-3 text-rose-300">{formatMetric(row.max_drawdown, "percent")}</td>
                <td className="py-3">{row.number_of_trades.toFixed(0)}</td>
                <td className="py-3">{row.average_holding_period.toFixed(1)}d</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-graphite-900/60 px-3 py-2">
      <div className="text-[11px] uppercase text-slate-500">{label}</div>
      <div className="mt-1 truncate text-sm font-medium text-slate-100">{value}</div>
    </div>
  );
}

function ChartTooltip({ active, payload, label, percentKeys = [] }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-700 bg-graphite-900/95 px-3 py-2 text-xs shadow-panel">
      <div className="mb-1 text-slate-400">{label}</div>
      {payload.map((item: any) => {
        const value = Number(item.value);
        const text = percentKeys.includes(item.dataKey) ? formatMetric(value, "percent") : value > 1000 ? formatCurrency(value) : value.toFixed(3);
        return (
          <div key={item.dataKey} className="flex min-w-[160px] justify-between gap-4">
            <span style={{ color: item.color }}>{item.name ?? item.dataKey}</span>
            <span className="text-slate-100">{text}</span>
          </div>
        );
      })}
    </div>
  );
}

function SkeletonDashboard() {
  return (
    <div className="mt-5 grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
      <div className="space-y-5">
        <div className="panel h-36 animate-pulse" />
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="panel h-24 animate-pulse" />
          ))}
        </div>
        <div className="panel h-[380px] animate-pulse" />
      </div>
      <div className="space-y-5">
        <div className="panel h-72 animate-pulse" />
        <div className="panel h-72 animate-pulse" />
      </div>
    </div>
  );
}
