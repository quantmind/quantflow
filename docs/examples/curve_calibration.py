from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from docs.examples._utils import assets_path, btc_surface_loader
from quantflow.options.bs import implied_black_volatility
from quantflow.options.surface import VolCrossSection
from quantflow.rates import InterpolatedMonotonicCubicCurve, NelsonSiegelCurve

MATURITY = "2026-12-25"


def style(fig: go.Figure, font_size: int = 20) -> go.Figure:
    """Apply a readable font scale to a figure"""
    fig.update_layout(font_size=font_size, title_font_size=font_size + 4)
    fig.update_annotations(font_size=font_size)
    return fig


def smile(cross: VolCrossSection, forward: float, ttm: float) -> pd.DataFrame:
    """Invert out of the money mid quotes with the given forward price"""
    strikes, ks, prices, cps = [], [], [], []
    for sk in cross.strikes:
        strike = float(sk.strike)
        option = sk.put if strike <= forward else sk.call
        if option is None:
            continue
        strikes.append(strike)
        ks.append(np.log(strike / forward))
        # inverse options are quoted in BTC, already in forward space
        prices.append(float(option.bid.price + option.ask.price) / 2)
        cps.append(-1.0 if strike <= forward else 1.0)
    result = implied_black_volatility(
        k=np.array(ks),
        price=np.array(prices),
        ttm=np.full(len(ks), ttm),
        initial_sigma=np.full(len(ks), 0.5),
        call_put=np.array(cps),
    )
    frame = pd.DataFrame(
        dict(
            strike=strikes,
            iv=np.asarray(result.values).ravel(),
            side=["put" if cp < 0 else "call" for cp in cps],
        )
    )
    return frame[np.asarray(result.converged).ravel()]


def smile_panel(fig: go.Figure, frame: pd.DataFrame, col: int) -> None:
    for side, color in (("put", "#636efa"), ("call", "#ef553b")):
        data = frame[frame.side == side]
        fig.add_trace(
            go.Scatter(
                x=data.strike,
                y=data.iv,
                mode="markers",
                marker=dict(symbol="circle", color=color, size=9),
                name=f"{side}s",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )


loader = btc_surface_loader()
surface = loader.surface()
ref_date = surface.ref_date
spot = float(surface.spot_price())

cross = next(c for c in surface.maturities if str(c.maturity.date()) == MATURITY)
ttm = cross.ttm(ref_date)
future = float(cross.forward.mid)

# figure 1: the same quotes inverted with the spot and with the market future
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        f"forward = spot ({spot:,.0f})",
        f"forward = future ({future:,.0f})",
    ),
)
bad = smile(cross, spot, ttm)
smile_panel(fig, bad, 1)
smile_panel(fig, smile(cross, future, ttm), 2)
last_put = bad[bad.side == "put"].iloc[-1]
first_call = bad[bad.side == "call"].iloc[0]
jump = first_call.iv - last_put.iv
fig.add_annotation(
    x=first_call.strike,
    y=(first_call.iv + last_put.iv) / 2,
    xref="x1",
    yref="y1",
    text=f"{100 * jump:.1f} vol point jump",
    showarrow=True,
    arrowhead=2,
    ax=140,
    ay=40,
)
fig.update_layout(
    title=f"BTC {MATURITY} smile, out of the money mid quotes",
    legend=dict(x=0.62, y=1.12, orientation="h"),
)
fig.update_xaxes(title_text="strike")
fig.update_yaxes(title_text="implied volatility", tickformat=".0%")
style(fig, font_size=24)
fig.write_image(assets_path("curve_calibration_smile.png"), width=1600, height=800)


# figure 2: put-call parity regression at the front and at a long maturity
def parity_panel(fig: go.Figure, cross: VolCrossSection, col: int) -> None:
    """Add the parity regression of one maturity to a subplot column"""
    section = loader.maturities[cross.maturity]
    parities = section.put_call_parities(
        surface.spot_price(), ref_date=ref_date, max_pairs=100
    )
    cp = parities.regressand()
    k = parities.regressor()
    slope, intercept = np.polyfit(k, cp, 1)
    # the forward is the strike where the regression line crosses zero
    zero_crossing = -intercept / slope
    k_line = np.linspace(k.min(), k.max(), 50)
    fig.add_trace(
        go.Scatter(
            x=k_line,
            y=intercept + slope * k_line,
            mode="lines",
            line=dict(color="#636efa"),
            name="regression",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=k,
            y=cp,
            mode="markers",
            marker=dict(symbol="circle", size=10, color="#ef553b"),
            name="put-call pairs",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
    fig.add_vline(
        x=zero_crossing,
        line_dash="dash",
        annotation_text="implied forward",
        row=1,
        col=col,
    )


front = surface.maturities[0]
front_ttm = front.ttm(ref_date)
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        f"{front.maturity.date()}, ttm {front_ttm:.4f} years",
        f"{cross.maturity.date()}, ttm {ttm:.3f} years",
    ),
)
parity_panel(fig, front, 1)
parity_panel(fig, cross, 2)
fig.update_layout(
    title=f"BTC put-call parity regression, snapshot {ref_date:%Y-%m-%d %H:%M} UTC",
    legend=dict(x=0.62, y=1.12, orientation="h"),
)
fig.update_xaxes(title_text="strike / spot")
fig.update_yaxes(title_text="call - put (BTC)")
style(fig, font_size=24)
fig.write_image(assets_path("curve_calibration_parity.png"), width=1600, height=800)

# figure 3: the same parity regressions in moneyness space
BAND = 0.15


def parity_moneyness_panel(fig: go.Figure, cross: VolCrossSection, col: int) -> None:
    """Add the parity regression of one maturity in moneyness space"""
    section = loader.maturities[cross.maturity]
    parities = section.put_call_parities(
        surface.spot_price(), ref_date=ref_date, max_pairs=100
    )
    cp = parities.regressand()
    k = parities.regressor()
    ttm_cross = cross.ttm(ref_date)
    m = np.log(k) / np.sqrt(ttm_cross)
    slope, intercept = np.polyfit(k, cp, 1)
    zero_crossing = -intercept / slope
    m_line = np.linspace(m.min(), m.max(), 100)
    fig.add_trace(
        go.Scatter(
            x=m_line,
            y=intercept + slope * np.exp(m_line * np.sqrt(ttm_cross)),
            mode="lines",
            line=dict(color="#636efa"),
            name="regression",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=m,
            y=cp,
            mode="markers",
            marker=dict(symbol="circle", size=10, color="#ef553b"),
            name="put-call pairs",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
    fig.add_vline(
        x=float(np.log(zero_crossing) / np.sqrt(ttm_cross)),
        line_dash="dash",
        annotation_text="implied forward",
        row=1,
        col=col,
    )
    annotation = (
        dict(annotation_text="band", annotation_position="bottom left")
        if col == 1
        else {}
    )
    fig.add_vrect(
        x0=-BAND,
        x1=BAND,
        fillcolor="#00cc96",
        opacity=0.15,
        line_width=0,
        row=1,
        col=col,
        **annotation,
    )


fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        f"{front.maturity.date()}, ttm {front_ttm:.4f} years",
        f"{cross.maturity.date()}, ttm {ttm:.3f} years",
    ),
)
parity_moneyness_panel(fig, front, 1)
parity_moneyness_panel(fig, cross, 2)
fig.update_layout(
    title=(
        f"BTC put-call parity in moneyness space, "
        f"snapshot {ref_date:%Y-%m-%d %H:%M} UTC"
    ),
    legend=dict(x=0.62, y=1.12, orientation="h"),
)
fig.update_xaxes(title_text="moneyness log(K/S) / sqrt(ttm)")
fig.update_yaxes(title_text="call - put (BTC)")
style(fig, font_size=24)
fig.write_image(
    assets_path("curve_calibration_parity_moneyness.png"), width=1600, height=800
)

# figure 4: forward term structure, market futures against parity implied forwards
term = loader.implied_forward_term_structure(max_pairs=100)
implied = pd.DataFrame(term, columns=["maturity", "ttm", "forward"])
futures = pd.DataFrame(
    dict(
        ttm=[c.ttm(ref_date) for c in surface.maturities],
        forward=[float(c.forward.mid) for c in surface.maturities],
    )
)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=implied.ttm,
        y=implied.forward,
        mode="markers",
        marker=dict(symbol="diamond", size=10),
        name="parity implied",
    )
)
fig.add_trace(
    go.Scatter(
        x=futures.ttm,
        y=futures.forward,
        mode="markers",
        marker=dict(symbol="circle", size=10),
        name="listed futures",
    )
)
fig.add_hline(y=spot, line_dash="dash", annotation_text="spot")
fig.update_layout(
    title="BTC forward term structure",
    xaxis_title="time to maturity (years, log scale)",
    xaxis_type="log",
    yaxis_title="forward price (USD)",
)
style(fig)
fig.write_image(assets_path("curve_calibration_forwards.png"), width=900, height=500)

# figure 5: the parity forward error against the listed future, with a naive
# all pairs regression and with the calibrate_forwards algorithm
rows = []
for c in surface.maturities:
    section = loader.maturities[c.maturity]
    parities = section.put_call_parities(
        surface.spot_price(), ref_date=ref_date, max_pairs=100
    )
    cp_pairs = parities.regressand()
    k_pairs = parities.regressor()
    slope, intercept = np.polyfit(k_pairs, cp_pairs, 1)
    rows.append(dict(ttm=c.ttm(ref_date), forward=-intercept / slope * spot))
naive = pd.DataFrame(rows)
parity_forwards = loader.calibrate_forwards()
calibrated = pd.DataFrame(
    dict(
        ttm=[entry[1] for entry in parity_forwards],
        forward=[float(entry[2]) for entry in parity_forwards],
    )
)
fig = go.Figure()
for name, symbol, frame in (
    ("all pairs", "diamond", naive),
    ("calibrate_forwards", "circle", calibrated),
):
    merged = frame.merge(futures, on="ttm", suffixes=("", "_future"))
    fig.add_trace(
        go.Scatter(
            x=merged.ttm,
            y=(merged.forward - merged.forward_future) / merged.forward_future,
            mode="markers",
            marker=dict(symbol=symbol, size=10),
            name=name,
        )
    )
fig.add_hline(y=0, line_dash="dash")
fig.update_layout(
    title="Parity implied forward minus listed future",
    xaxis_title="time to maturity (years, log scale)",
    xaxis_type="log",
    yaxis_title="relative difference",
    yaxis_tickformat=".2%",
)
style(fig)
fig.write_image(
    assets_path("curve_calibration_forward_basis.png"), width=900, height=500
)

# figure 6: parametric curves pool information across maturities, interpolated
# curves fit one node per maturity and amplify the noise at the front
loader_ns = btc_surface_loader()
loader_ns.calibrate_curves(quote_curve=NelsonSiegelCurve, asset_curve=NelsonSiegelCurve)
loader_in = btc_surface_loader()
loader_in.calibrate_curves(
    quote_curve=InterpolatedMonotonicCubicCurve,
    asset_curve=InterpolatedMonotonicCubicCurve,
)
quote_in = cast(InterpolatedMonotonicCubicCurve, loader_in.quote_curve)
asset_in = cast(InterpolatedMonotonicCubicCurve, loader_in.asset_curve)
max_ttm = max(float(c.ttm(ref_date)) for c in surface.maturities)
ttm_grid = np.linspace(1 / 52, max_ttm, 100)
node_ttm = np.array(
    [
        (date - ref_date).total_seconds() / (365 * 86400)
        for date in quote_in.anchor_dates
    ]
)
fig = go.Figure()
for name, curve in (("quote", loader_ns.quote_curve), ("asset", loader_ns.asset_curve)):
    fig.add_trace(
        go.Scatter(
            x=ttm_grid,
            y=np.atleast_1d(curve.continuously_compounded_rate(ttm_grid)),
            mode="lines",
            name=f"{name} Nelson-Siegel",
        )
    )
for name, curve in (("quote", quote_in), ("asset", asset_in)):
    fig.add_trace(
        go.Scatter(
            x=node_ttm,
            y=[float(r) for r in curve.anchor_rates],
            mode="markers",
            marker=dict(symbol="circle", size=10),
            name=f"{name} interpolated",
        )
    )
node_rates = [float(r) for curve in (quote_in, asset_in) for r in curve.anchor_rates]
worst = min(node_rates, key=lambda r: -abs(r))
off_scale = sum(1 for r in node_rates if abs(r) > 0.5)
fig.update_layout(
    title="Calibrated zero rates, parametric against interpolated",
    xaxis_title="time to maturity (years)",
    yaxis_title="zero rate",
    yaxis_tickformat=".0%",
    yaxis_range=[-0.5, 0.5],
)
note = (
    f"{off_scale} front nodes off scale, worst at {worst:.0%}"
    if off_scale
    else f"worst node at {worst:.0%}"
)
fig.add_annotation(x=0.02, y=-0.45, text=note, showarrow=False, xanchor="left")
style(fig)
fig.write_image(assets_path("curve_calibration_rates.png"), width=900, height=500)

# structured output: calibrated parity forwards against market futures
table = calibrated.merge(futures, on="ttm", suffixes=("_parity", "_future"))
table.insert(0, "maturity", [str(entry[0].date()) for entry in parity_forwards])
table["basis"] = table.forward_parity - table.forward_future
table = table.round(dict(ttm=3, forward_parity=1, forward_future=1, basis=1))
print(table.to_string(index=False))
