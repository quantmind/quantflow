import os
from typing import Any

from .marginal import Marginal1D
from .types import FloatArray

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
    m: Marginal1D, x: FloatArray, analytical: str = "lines", **kwargs: Any
) -> Any:
    """Plot the marginal pdf on an input support"""
    check_plotly()
    n = len(x)
    result = m.pdf_from_characteristic(x)
    xx = result.x[:n]
    yy = result.y[:n]
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
        )
    )
    return fig


def plot_characteristic(
    m: Marginal1D, n: int | None = None, max_frequency: float | None = None
) -> Any:
    df = m.characteristic_df(n=n, max_frequency=max_frequency)
    return px.line(
        df,
        x="frequency",
        y="characteristic",
        color="name",
        markers=True,
    )


def plot_vol_surface(
    data: Any, template: str = PLOTLY_THEME, marker_size: int = 10, **kwargs: Any
) -> Any:
    if px is None:
        raise ImportError("plotly is not installed")
    params = dict(
        x="moneyness",
        y="implied_vol",
        color="side",
        template=template,
    )
    params.update(kwargs)
    fig = px.scatter(data, **params)
    fig.update_traces(marker_size=marker_size)
    return fig
