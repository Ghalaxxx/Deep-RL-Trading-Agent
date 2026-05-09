"""Curriculum scheduling for staged market difficulty."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class CurriculumStage:
    name: str
    min_step: int
    tickers: list[str]
    start: str
    end: str
    volatility: str


@dataclass(slots=True)
class CurriculumScheduler:
    """Three-stage scheduler with Sharpe-based advancement criteria."""

    validation_threshold: float = 0.5
    required_consecutive_successes: int = 3
    current_stage_index: int = 0
    consecutive_successes: int = 0
    stages: list[CurriculumStage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = [
                CurriculumStage(
                    name="stage_1_low_volatility",
                    min_step=0,
                    tickers=["MSFT", "AAPL"],
                    start="2018-01-01",
                    end="2020-01-01",
                    volatility="low",
                ),
                CurriculumStage(
                    name="stage_2_medium_volatility",
                    min_step=200_000,
                    tickers=["MSFT", "AAPL", "SPY", "QQQ"],
                    start="2018-01-01",
                    end="2021-01-01",
                    volatility="medium",
                ),
                CurriculumStage(
                    name="stage_3_full_dataset",
                    min_step=500_000,
                    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "GLD", "BTC-USD"],
                    start="2015-01-01",
                    end="2024-12-31",
                    volatility="full",
                ),
            ]

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_index]

    def update(self, global_step: int, validation_sharpe: float) -> CurriculumStage:
        """Advance when both timestep and validation-Sharpe criteria are met."""

        if self.current_stage_index >= len(self.stages) - 1:
            return self.current_stage

        next_stage = self.stages[self.current_stage_index + 1]
        if validation_sharpe > self.validation_threshold:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        if (
            global_step >= next_stage.min_step
            and self.consecutive_successes >= self.required_consecutive_successes
        ):
            self.current_stage_index += 1
            self.consecutive_successes = 0
        return self.current_stage

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the date slice for the active curriculum stage."""

        stage = self.current_stage
        if not isinstance(df.index, pd.DatetimeIndex):
            return df.copy()
        return df.loc[stage.start : stage.end].copy()
