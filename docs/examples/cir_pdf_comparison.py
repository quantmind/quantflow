"""CIR process: compare analytical PDF with PDF from characteristic function."""

import numpy as np
import plotly.graph_objects as go

from docs.examples._utils import assets_path
from quantflow.sp.cir import CIR


def make_figure(cir: CIR, t: float, n: int = 128) -> go.Figure:
    m = cir.marginal(t)
    x = np.linspace(1e-6, m.mean() + 4 * float(m.std()), 300)

    pdf_analytical = cir.analytical_pdf(t, x)
    pdf_cf = m.pdf_from_characteristic(n, simpson_rule=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf_analytical,
            mode="lines",
            name="Analytical PDF",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pdf_cf.x,
            y=pdf_cf.y,
            mode="markers",
            name="PDF from characteristic function",
            marker=dict(color="#ff7f0e", size=6, symbol="circle"),
        )
    )
    fig.update_layout(
        title=(
            f"CIR PDF at t={t}"
            f" (κ={cir.kappa}, θ={cir.theta}, σ={cir.sigma}, x₀={cir.rate})"
        ),
        xaxis_title="x",
        yaxis_title="probability density",
        legend=dict(x=0.6, y=0.95),
    )
    return fig


def make_cf_figure(cir: CIR, t: float, n: int = 512) -> go.Figure:
    m = cir.marginal(t)
    max_frequency = float(np.asarray(m.frequency_range().ub).flat[0])
    u = np.linspace(0, max_frequency, n)
    cf = cir.characteristic(t, u)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u,
            y=np.abs(cf),
            mode="lines",
            name="|Φ(u)|",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u,
            y=cf.real,
            mode="lines",
            name="Re[Φ(u)]",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig.add_vline(
        x=max_frequency,
        line=dict(color="red", dash="dot", width=1),
        annotation_text="max_frequency",
        annotation_position="top left",
    )
    fig.update_layout(
        title=(
            f"CIR characteristic function at t={t}"
            f" (κ={cir.kappa}, θ={cir.theta}, σ={cir.sigma}, x₀={cir.rate})"
        ),
        xaxis_title="u",
        yaxis_title="Φ(u)",
    )
    return fig


if __name__ == "__main__":
    cir = CIR(kappa=1.0, theta=0.5, sigma=0.8, rate=3.0)

    fig1 = make_figure(cir, t=0.5)
    fig1.write_image(assets_path("cir_pdf_t05.png"), width=900, height=500)

    fig2 = make_figure(cir, t=2.0)
    fig2.write_image(assets_path("cir_pdf_t20.png"), width=900, height=500)

    fig3 = make_cf_figure(cir, t=2.0)
    fig3.write_image(assets_path("cir_cf_t20.png"), width=900, height=500)
