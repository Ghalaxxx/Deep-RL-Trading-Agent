"""Visualization helpers for equity curves and agent behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_equity_curves(
    curves: dict[str, pd.Series | np.ndarray],
    title: str = "Equity Curves",
    output_path: Path | str | None = None,
) -> go.Figure:
    """Create an interactive Plotly equity-curve comparison."""

    fig = go.Figure()
    for name, curve in curves.items():
        values = curve if isinstance(curve, pd.Series) else pd.Series(curve)
        fig.add_trace(go.Scatter(x=values.index, y=values.values, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Date / Step", yaxis_title="Portfolio Value")
    if output_path is not None:
        fig.write_html(str(output_path))
    return fig


def plot_action_distribution(
    actions: np.ndarray,
    title: str = "Action Distribution",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a histogram of continuous actions."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.asarray(actions, dtype=float), bins=40, color="#2F855A", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=160)
    return fig
