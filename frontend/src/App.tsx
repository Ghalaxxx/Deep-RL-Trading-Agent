import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
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
  Target,
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
import type {
  AgentSignal,
  BacktestRow,
  DashboardData,
  MarketFeed,
  MarketRegime,
  MetricCard,
  ModelComparison,
  PaperPortfolio,
  RiskState,
  TrainingSession
} from "./types";

const chartColors: Record<string, string> = {
  buy_hold: "#67E8F9",
  sma_crossover: "#34D399",
  momentum: "#FBBF24",
  random: "#FB7185"
};

export function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [data, setData] = useState<DashboardData | null>(null);
  const [marketFeed, setMarketFeed] = useState<MarketFeed | null>(null);
  const [paperPortfolio, setPaperPortfolio] = useState<PaperPortfolio | null>(null);
  const [risk, setRisk] = useState<RiskState | null>(null);
  const [marketRegime, setMarketRegime] = useState<MarketRegime | null>(null);
  const [agentSignal, setAgentSignal] = useState<AgentSignal | null>(null);
  const [trainingSessions, setTrainingSessions] = useState<TrainingSession[]>([]);
  const [connection, setConnection] = useState<"connecting" | "replay" | "reconnecting" | "polling">("connecting");
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);
    fetchDashboard(ticker)
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setMarketFeed(payload.marketFeed);
          setPaperPortfolio(payload.paperPortfolio);
          setRisk(payload.risk);
          setMarketRegime(payload.marketRegime);
          setAgentSignal(payload.agentExplainability);
          setTrainingSessions(payload.trainingSessions);
          setLastUpdate(payload.marketFeed.asOf);
        }
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

  useEffect(() => {
    if (!data) return;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let closedByEffect = false;
    const connect = () => {
      setConnection((current) => (current === "replay" ? current : "connecting"));
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.hostname}:8000/ws/replay/${ticker}`);
      socket.onopen = () => setConnection("replay");
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (!message.events) return;
        for (const update of message.events) {
          if (update.type === "MARKET_REPLAY_UPDATE") {
            setMarketFeed(update.payload);
            setLastUpdate(update.payload.asOf);
          }
          if (update.type === "PAPER_TRADE_UPDATE") setPaperPortfolio(update.payload);
          if (update.type === "RISK_UPDATE") setRisk(update.payload);
          if (update.type === "REGIME_UPDATE") setMarketRegime(update.payload);
          if (update.type === "AGENT_SIGNAL") setAgentSignal(update.payload);
          if (update.type === "EXPERIMENT_UPDATE") setTrainingSessions(update.payload);
        }
      };
      socket.onerror = () => setConnection("polling");
      socket.onclose = () => {
        if (closedByEffect) return;
        setConnection("reconnecting");
        reconnectTimer = window.setTimeout(connect, 2500);
      };
    };
    connect();
    return () => {
      closedByEffect = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [data, ticker]);

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
        <TopBar
          data={data}
          ticker={ticker}
          setTicker={setTicker}
          isLoading={isLoading}
          connection={connection}
          lastUpdate={lastUpdate}
        />
        {data && marketFeed && paperPortfolio && risk && marketRegime && agentSignal ? (
          <div className="mt-5 grid gap-5 xl:grid-cols-[1.3fr_0.7fr]">
            <section className="space-y-5">
              <MarketReplayPanel feed={marketFeed} />
              <PortfolioHeader data={data} />
              <PaperPortfolioPanel portfolio={paperPortfolio} />
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
              <StrategyHeatmapPanel data={data} />
              <TrainingSessionsPanel sessions={trainingSessions} />
            </section>
            <aside className="space-y-5">
              <RiskPanel risk={risk} />
              <RegimePanel regime={marketRegime} />
              <ExplainabilityPanel signal={agentSignal} />
              <ModelComparisonPanel models={data.modelComparison} />
              <ExperimentPanel data={data} />
              <PaperTradeTimeline trades={paperPortfolio.trades} />
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
  isLoading,
  connection,
  lastUpdate
}: {
  data: DashboardData | null;
  ticker: string;
  setTicker: (ticker: string) => void;
  isLoading: boolean;
  connection: "connecting" | "replay" | "reconnecting" | "polling";
  lastUpdate: string | null;
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
              <span className="h-1 w-1 rounded-full bg-slate-600" />
              <span>{data?.marketFeed.mode ?? "Replay feed pending"}</span>
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
        <div className="status-chip cyan">
          <span className={`h-2 w-2 rounded-full ${isLoading ? "bg-amber-300" : "bg-cyan-300"}`} />
          {connectionLabel(connection)}
        </div>
        <div className="hidden text-right text-xs text-slate-500 sm:block">
          <div>Replay update</div>
          <div className="text-slate-300">{lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : "pending"}</div>
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

function MarketReplayPanel({ feed }: { feed: MarketFeed }) {
  const positive = feed.change >= 0;
  const isHistorical = feed.sourceKind === "historical_replay";
  return (
    <section className="panel p-5">
      <div className="grid gap-4 lg:grid-cols-[0.9fr_1.4fr]">
        <div>
          <div className="mb-3 text-xs font-semibold uppercase text-slate-500">
            {isHistorical ? "Historical Replay Engine" : "Delayed Market Snapshot Mode"}
          </div>
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-cyan-200">{feed.symbol}</div>
              <div className="mt-2 text-3xl font-semibold text-white">{formatCurrency(feed.price)}</div>
            </div>
            <span className={`status-chip ${isHistorical ? "amber" : "cyan"}`}>{feed.mode}</span>
          </div>
          <div className="mt-2 text-xs leading-5 text-slate-500">{feed.dataBasis}</div>
          <div className={`mt-3 text-sm font-semibold ${positive ? "text-emerald-300" : "text-rose-300"}`}>
            {positive ? "+" : ""}
            {feed.change.toFixed(2)} ({formatMetric(feed.percentChange, "percent")})
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <MiniStat label={feed.sourceLabel} value={feed.sourceTimestamp} />
            <MiniStat label="Replay engine update" value={new Date(feed.asOf).toLocaleTimeString()} />
          </div>
        </div>
        <div className="h-[160px]">
          <ResponsiveContainer width="100%" height="100%">
            <ReLineChart data={feed.history} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
              <CartesianGrid stroke="#1F2B3B" strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fill: "#94A3B8", fontSize: 10 }} minTickGap={26} />
              <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} width={54} domain={["dataMin", "dataMax"]} />
              <Tooltip content={<ChartTooltip />} />
              <Line type="monotone" dataKey="price" stroke="#67E8F9" strokeWidth={2.2} dot={false} name="Price" />
            </ReLineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

function PaperPortfolioPanel({ portfolio }: { portfolio: PaperPortfolio }) {
  const pnl = portfolio.realizedPnl + portfolio.unrealizedPnl;
  return (
    <Panel title="Paper Portfolio" icon={<Target size={17} />} action={portfolio.mode}>
      <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="grid grid-cols-2 gap-3">
          <MiniStat label="Cash" value={formatCurrency(portfolio.cash)} />
          <MiniStat label="Total equity" value={formatCurrency(portfolio.totalEquity)} />
          <MiniStat label="Position" value={`${portfolio.positionShares.toFixed(3)} sh`} />
          <MiniStat label="Exposure" value={formatMetric(portfolio.exposure, "percent")} />
          <MiniStat label="Realized PnL" value={formatCurrency(portfolio.realizedPnl)} />
          <MiniStat label="Unrealized PnL" value={formatCurrency(portfolio.unrealizedPnl)} />
        </div>
        <div>
          <div className="mb-3 flex items-center justify-between">
            <span className={`status-chip ${pnl >= 0 ? "emerald" : "amber"}`}>{portfolio.currentAction}</span>
            <span className="text-xs text-slate-500">
              Cost {(portfolio.executionAssumptions.transactionCost * 100).toFixed(2)}% + slippage {(portfolio.executionAssumptions.slippage * 100).toFixed(2)}%
            </span>
          </div>
          <div className="h-[170px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={portfolio.equityCurve} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                <defs>
                  <linearGradient id="paperEquity" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34D399" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#34D399" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1F2B3B" strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fill: "#94A3B8", fontSize: 10 }} minTickGap={26} />
                <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} tickFormatter={formatCompact} width={50} />
                <Tooltip content={<ChartTooltip />} />
                <Area type="monotone" dataKey="equity" stroke="#34D399" fill="url(#paperEquity)" strokeWidth={2} name="Paper Equity" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function RiskPanel({ risk }: { risk: RiskState }) {
  const tone = risk.status === "Normal" ? "emerald" : risk.status === "Warning" ? "amber" : "rose";
  return (
    <Panel title="Risk Management" icon={<AlertTriangle size={17} />}>
      <div className="mb-4 flex items-center justify-between">
        <span className={`status-chip ${tone === "rose" ? "amber" : tone}`}>{risk.status}</span>
        <span className="text-xs text-slate-500">{risk.mode}</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <MiniStat label="Current DD" value={formatMetric(risk.currentDrawdown, "percent")} />
        <MiniStat label="Max DD" value={formatMetric(risk.maxDrawdown, "percent")} />
        <MiniStat label="Volatility" value={formatMetric(risk.volatility, "percent")} />
        <MiniStat label="Kill switch" value={risk.killSwitchActive ? "Active" : "Armed"} />
      </div>
      <div className="mt-4 space-y-2">
        {risk.guardrails.map((rule) => (
          <div key={rule.name} className="flex items-center justify-between rounded-lg border border-slate-800 bg-graphite-900/70 px-3 py-2 text-xs">
            <span className="text-slate-300">{rule.name}</span>
            <span className={rule.passing ? "text-emerald-300" : "text-amber-300"}>{rule.passing ? "Pass" : "Breach"}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function RegimePanel({ regime }: { regime: MarketRegime }) {
  return (
    <Panel title="Current Market Regime" icon={<Gauge size={17} />}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-lg font-semibold text-white">{regime.label}</div>
          <div className="mt-1 text-xs text-cyan-200">{regime.method}</div>
        </div>
        <span className="status-chip cyan">{formatMetric(regime.confidence, "percent")}</span>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-400">{regime.explanation}</p>
      <div className="mt-4 space-y-2">
        {regime.factors.map((factor) => (
          <div key={factor.name} className="flex items-center justify-between text-xs">
            <span className="text-slate-400">{factor.name}</span>
            <span className="text-slate-200">{factor.status}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function ExplainabilityPanel({ signal }: { signal: AgentSignal }) {
  return (
    <Panel title="Why This Action?" icon={<BrainCircuit size={17} />}>
      <div className="flex items-center justify-between">
        <span className="text-2xl font-semibold text-white">{signal.action}</span>
        <span className="status-chip cyan">{signal.mode}</span>
      </div>
      <div className="mt-2 text-sm text-slate-400">
        Target exposure {formatMetric(signal.targetExposure, "percent")} | confidence {formatMetric(signal.confidence, "percent")}
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-400">{signal.rationale}</p>
      <div className="mt-4 space-y-2">
        {signal.signals.map((item) => (
          <div key={item.name} className="rounded-lg border border-slate-800 bg-graphite-900/70 px-3 py-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300">{item.name}</span>
              <span className="text-cyan-200">{item.interpretation}</span>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function StrategyHeatmapPanel({ data }: { data: DashboardData }) {
  const labels: Record<string, string> = {
    annualized_return: "Return",
    sharpe: "Sharpe",
    sortino: "Sortino",
    max_drawdown: "Max DD",
    volatility: "Vol",
    win_rate: "Win",
    number_of_trades: "Trades",
    profit_factor: "Profit Factor",
    calmar: "Calmar"
  };
  return (
    <Panel title="Strategy Comparison Heatmap" icon={<BarChart3 size={17} />}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[880px] border-separate border-spacing-1 text-left text-xs">
          <thead className="text-slate-500">
            <tr>
              <th className="sticky left-0 z-10 bg-graphite-900 px-3 py-2 font-medium">Strategy</th>
              {data.strategyHeatmap.metrics.map((metric) => (
                <th key={metric} className="px-3 py-2 font-medium">
                  {labels[metric] ?? metric}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.strategyHeatmap.rows.map((row) => (
              <tr key={row.strategy}>
                <td className="sticky left-0 z-10 min-w-[154px] rounded-lg border border-slate-800 bg-graphite-900 px-3 py-2">
                  <div className="font-semibold text-slate-100">{row.strategy}</div>
                  <div className="mt-1 text-[11px] text-slate-500">{row.status}</div>
                </td>
                {data.strategyHeatmap.metrics.map((metric) => {
                  const value = row.metrics[metric];
                  return (
                    <td key={`${row.strategy}-${metric}`} className={`rounded-lg border px-3 py-2 ${heatmapCellClass(metric, value)}`}>
                      {value === null ? <span className="text-slate-500">Awaiting checkpoint</span> : metricDisplay(metric, value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-xs leading-5 text-slate-500">
        PPO and SAC cells remain empty until local checkpoint-backed evaluations exist. Baseline rows come from the repository backtest reports.
      </div>
    </Panel>
  );
}

function TrainingSessionsPanel({ sessions }: { sessions: TrainingSession[] }) {
  return (
    <Panel title="Training Session Manager" icon={<Activity size={17} />}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr className="border-b border-slate-800">
              <th className="pb-3 font-medium">Algorithm</th>
              <th className="pb-3 font-medium">Asset</th>
              <th className="pb-3 font-medium">Timesteps</th>
              <th className="pb-3 font-medium">Reward</th>
              <th className="pb-3 font-medium">Status</th>
              <th className="pb-3 font-medium">Checkpoint</th>
              <th className="pb-3 font-medium">Best Metric</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((session) => (
              <tr key={session.id} className="border-b border-slate-900 text-slate-300">
                <td className="py-3 font-semibold text-slate-100">{session.algorithm}</td>
                <td className="py-3">{session.ticker}</td>
                <td className="py-3">{formatCompact(session.timesteps)}</td>
                <td className="py-3">{session.rewardFunction}</td>
                <td className="py-3">
                  <span className={`status-chip ${session.status === "completed" ? "emerald" : session.status === "failed" ? "amber" : "cyan"}`}>
                    {session.status}
                  </span>
                </td>
                <td className="py-3">{session.checkpointAvailable ? "Available" : "Awaiting checkpoint"}</td>
                <td className="py-3">{session.bestValidationMetric === null ? "N/A" : session.bestValidationMetric.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-xs leading-5 text-slate-500">
        This manager reflects local configs and checkpoint artifacts first; it does not simulate running training progress.
      </div>
    </Panel>
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

function PaperTradeTimeline({ trades }: { trades: PaperPortfolio["trades"] }) {
  return (
    <Panel title="Paper Trade Timeline" icon={<TrendingUp size={17} />} action="paper">
      {trades.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-graphite-900/60 p-4 text-sm leading-6 text-slate-400">
          No paper trades in the current replay window. The simulator only records fills after cost-aware exposure changes.
        </div>
      ) : (
        <div className="space-y-3">
          {trades
            .slice()
            .reverse()
            .map((trade) => (
              <div key={`${trade.timestamp}-${trade.side}-${trade.price}-${trade.shares}`} className="relative border-l border-slate-800 pl-4">
                <div className={`absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full ${trade.side === "BUY" ? "bg-emerald-300" : "bg-rose-300"}`} />
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-medium text-slate-100">
                    {trade.side} <span className="text-xs text-slate-500">paper</span>
                  </div>
                  <div className="text-xs text-slate-500">{trade.timestamp}</div>
                </div>
                <div className="mt-1 grid grid-cols-3 gap-2 text-xs text-slate-400">
                  <span>{Math.abs(trade.shares).toFixed(3)} sh</span>
                  <span>{formatCurrency(trade.price)}</span>
                  <span className="text-right">Cost {formatCurrency(trade.cost)}</span>
                </div>
              </div>
            ))}
        </div>
      )}
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

function metricDisplay(metric: string, value: number): string {
  if (["annualized_return", "max_drawdown", "volatility", "win_rate"].includes(metric)) {
    return formatMetric(value, "percent");
  }
  if (metric === "number_of_trades") return value.toFixed(0);
  return value.toFixed(2);
}

function heatmapCellClass(metric: string, value: number | null): string {
  if (value === null) return "border-slate-800 bg-graphite-900/70";
  if (metric === "max_drawdown") {
    const intensity = Math.min(Math.abs(value) / 0.35, 1);
    return intensity > 0.45
      ? "border-rose-300/25 bg-rose-400/15 text-rose-100"
      : "border-emerald-300/20 bg-emerald-400/10 text-emerald-100";
  }
  if (metric === "volatility" || metric === "number_of_trades") {
    const high = metric === "volatility" ? value > 0.32 : value > 35;
    return high ? "border-amber-300/25 bg-amber-400/12 text-amber-100" : "border-slate-700 bg-graphite-900/70 text-slate-200";
  }
  const positive = value > 0;
  return positive ? "border-emerald-300/25 bg-emerald-400/12 text-emerald-100" : "border-rose-300/25 bg-rose-400/12 text-rose-100";
}

function connectionLabel(connection: "connecting" | "replay" | "reconnecting" | "polling"): string {
  if (connection === "replay") return "Replay connected";
  if (connection === "reconnecting") return "Replay reconnecting";
  if (connection === "polling") return "Polling replay";
  return "Connecting replay";
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
