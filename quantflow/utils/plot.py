import os
from typing import Any

try:
    import plotly.express as px  # type: ignore
except ImportError:
    px = None

PLOTLY_THEME = os.environ.get("PLOTLY_THEME", "plotly_dark")


def plot_lines(data: Any, template: str = PLOTLY_THEME, **kwargs: Any) -> Any:
    if px is None:
        raise ImportError("plotly is not installed")
    return px.line(data, template=template, **kwargs)


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
