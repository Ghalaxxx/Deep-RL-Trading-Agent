export type MetricCard = {
  label: string;
  value: number | null;
  format: "percent" | "number" | "integer" | "days";
};

export type LiveFeed = {
  mode: "Live market data" | "Simulated live replay" | "Cached demo mode";
  symbol: string;
  price: number;
  previousClose: number;
  change: number;
  percentChange: number;
  asOf: string;
  sourceTimestamp: string;
  latencyMs: number;
  history: Array<{ time: string; price: number }>;
};

export type PaperTrade = {
  timestamp: string;
  side: "BUY" | "SELL";
  price: number;
  shares: number;
  notional: number;
  cost: number;
  mode: "paper";
};

export type PaperPortfolio = {
  mode: string;
  cash: number;
  positionShares: number;
  positionMarketValue: number;
  exposure: number;
  entryPrice: number | null;
  realizedPnl: number;
  unrealizedPnl: number;
  totalEquity: number;
  currentAction: string;
  allowLeverage: boolean;
  equityCurve: Array<{
    time: string;
    equity: number;
    cash: number;
    exposure: number;
    drawdown: number;
    unrealizedPnl: number;
  }>;
  trades: PaperTrade[];
  executionAssumptions: {
    transactionCost: number;
    slippage: number;
    realMoney: boolean;
    brokerConnected: boolean;
  };
};

export type RiskGuardrail = {
  name: string;
  passing: boolean;
  current: number;
  limit: number;
  severity: "warning" | "halt";
};

export type RiskState = {
  status: "Normal" | "Warning" | "Halted";
  currentDrawdown: number;
  maxDrawdown: number;
  volatility: number;
  dailyLossLimit: number;
  maxExposure: number;
  positionLimit: number;
  stopLossThreshold: number;
  killSwitchActive: boolean;
  violations: RiskGuardrail[];
  guardrails: RiskGuardrail[];
  mode: string;
};

export type MarketRegime = {
  label: string;
  confidence: number;
  method: string;
  explanation: string;
  factors: Array<{ name: string; value: number; status: string }>;
};

export type AgentSignal = {
  mode: string;
  action: string;
  targetExposure: number;
  confidence: number;
  rationale: string;
  signals: Array<{ name: string; value: number; interpretation: string }>;
};

export type Portfolio = {
  label: string;
  strategy: string;
  initialValue: number;
  endingValue: number;
  totalReturn: number;
  sharpe: number | null;
  maxDrawdown: number | null;
  status: string;
};

export type ModelComparison = {
  name: "PPO" | "SAC";
  status: string;
  policy: string;
  reward: string;
  timesteps: number;
  learningRate: number | null;
  checkpoint: string | null;
  notes: string;
};

export type BacktestRow = {
  ticker: string;
  strategy: string;
  annualized_return: number;
  volatility: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  win_rate: number | null;
  profit_factor: number | null;
  average_win: number | null;
  average_loss: number | null;
  number_of_trades: number;
  average_holding_period: number;
};

export type TradeEvent = {
  date: string;
  strategy: string;
  side: string;
  position: number;
  return: number;
  equity: number;
};

export type TrainingPoint = {
  step: number;
  ppoReward: number;
  sacReward: number;
  evalSharpe: number;
};

export type TrainingAnalytics = {
  source: string;
  label: string;
  isDemo: boolean;
  points: TrainingPoint[];
  channels: string[];
};

export type ExperimentItem = {
  name: string;
  status: string;
  detail: string;
};

export type TrainingSession = {
  id: string;
  algorithm: "PPO" | "SAC";
  ticker: string;
  timesteps: number;
  rewardFunction: string;
  status: "pending" | "running" | "completed" | "failed";
  bestValidationMetric: number | null;
  checkpointAvailable: boolean;
  checkpointPath: string | null;
  notes: string;
};

export type StrategyHeatmap = {
  metrics: string[];
  rows: Array<{
    strategy: string;
    status: string;
    metrics: Record<string, number | null>;
  }>;
};

export type RealtimeEvent =
  | { type: "PRICE_UPDATE"; payload: LiveFeed }
  | { type: "PAPER_TRADE_UPDATE"; payload: PaperPortfolio }
  | { type: "RISK_UPDATE"; payload: RiskState }
  | { type: "REGIME_UPDATE"; payload: MarketRegime }
  | { type: "AGENT_SIGNAL"; payload: AgentSignal }
  | { type: "EXPERIMENT_UPDATE"; payload: TrainingSession[] };

export type InferenceStep = {
  date: string;
  close: number;
  action: number;
  equity: number;
  drawdown: number;
};

export type DashboardData = {
  mode: {
    label: string;
    hasCheckpoint: boolean;
    vecNormalize: boolean;
    dataSource: string;
  };
  ticker: string;
  tickers: string[];
  liveFeed: LiveFeed;
  paperPortfolio: PaperPortfolio;
  risk: RiskState;
  marketRegime: MarketRegime;
  agentExplainability: AgentSignal;
  portfolio: Portfolio;
  metrics: MetricCard[];
  equityCurve: Array<Record<string, string | number>>;
  drawdownCurve: Array<{ date: string; drawdown: number }>;
  modelComparison: ModelComparison[];
  backtests: BacktestRow[];
  tradeTimeline: TradeEvent[];
  trainingAnalytics: TrainingAnalytics;
  experiments: ExperimentItem[];
  trainingSessions: TrainingSession[];
  strategyHeatmap: StrategyHeatmap;
  inferenceDemo: {
    mode: string;
    strategy: string;
    message: string;
    steps: InferenceStep[];
  };
  realtime: {
    transport: string;
    endpoint: string;
    fallback: string;
    eventTypes: string[];
  };
  safeguards: string[];
};
