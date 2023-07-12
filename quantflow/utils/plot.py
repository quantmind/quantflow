import os
from typing import Any

import pandas as pd

from .marginal import Marginal1D

PLOTLY_THEME = os.environ.get("PLOTLY_THEME", "plotly_dark")

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates.default = PLOTLY_THEME
except ImportError:
    px = None


def check_plotly() -> None:
    if px is None:
        raise ImportError("plotly is not installed")


def plot_lines(data: Any, template: str = PLOTLY_THEME, **kwargs: Any) -> Any:
    check_plotly()
    return px.line(data, template=template, **kwargs)


def plot_marginal_pdf(
    m: Marginal1D,
    n: int | None = None,
    analytical: str = "lines",
    marker_size: int = 8,
    marker_color: str = "rgba(30, 186, 64, .5)",
    **kwargs: Any
) -> Any:
    """Plot the marginal pdf on an input support"""
    check_plotly()
    result = m.pdf_from_characteristic(n, **kwargs)
    xx = result.x
    yy = result.y
    fig = go.Figure()
    if analytical:
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=m.pdf(xx),
                name="analytical",
                mode=analytical,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=yy,
            name="characteristic",
            mode="markers",
            marker_color=marker_color,
            marker_size=marker_size,
        )
    )
    return fig


def plot_characteristic(
    m: Marginal1D, n: int | None = None, max_frequency: float | None = None
) -> Any:
    check_plotly()
    df = m.characteristic_df(n=n, max_frequency=max_frequency)
    return px.line(
        df,
        x="frequency",
        y="characteristic",
        color="name",
        markers=True,
    )


def plot_vol_surface(
    data: Any,
    *,
    model: pd.DataFrame | None = None,
    template: str = PLOTLY_THEME,
    marker_size: int = 10,
    series: str = "implied_vol",
    **kwargs: Any
) -> Any:
    check_plotly()
    params = dict(
        x="moneyness_ttm",
        y=series,
        color="side",
        template=template,
    )
    params.update(kwargs)
    fig = px.scatter(data, **params)
    if model is not None:
        fig.add_trace(
            go.Scatter(
                x=model["moneyness_ttm"],
                y=model[series],
                name="model",
                mode="lines",
            )
        )
    fig.update_traces(marker_size=marker_size)
    return fig


def plot_vol_cross(
    data: pd.DataFrame,
    data2: pd.DataFrame | None = None,
    series: str = "implied_vol",
    marker_size: int = 10,
    **kwargs: Any
) -> Any:
    check_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["moneyness_ttm"],
            y=data[series],
            name="model",
            mode="lines",
        )
    )
    if data2 is not None:
        fig.add_trace(
            go.Scatter(
                x=data2["moneyness_ttm"],
                y=data2[series],
                name="model",
                mode="lines",
            )
        )
    return fig.update_layout(xaxis_title="moneyness_ttm", yaxis_title=series)
