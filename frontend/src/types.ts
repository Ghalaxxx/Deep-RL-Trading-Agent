export type MetricCard = {
  label: string;
  value: number | null;
  format: "percent" | "number" | "integer" | "days";
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
  portfolio: Portfolio;
  metrics: MetricCard[];
  equityCurve: Array<Record<string, string | number>>;
  drawdownCurve: Array<{ date: string; drawdown: number }>;
  modelComparison: ModelComparison[];
  backtests: BacktestRow[];
  tradeTimeline: TradeEvent[];
  trainingAnalytics: TrainingAnalytics;
  experiments: ExperimentItem[];
  inferenceDemo: {
    mode: string;
    strategy: string;
    message: string;
    steps: InferenceStep[];
  };
  safeguards: string[];
};
