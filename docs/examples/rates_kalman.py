import asyncio
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from docs.examples._utils import assets_path, cached_df
from quantflow.data.fed import FederalReserve
from quantflow.rates.calibration import tenor_to_years
from quantflow.rates.vasicek import VasicekCurve


@cached_df(ttl=timedelta(days=1))
def fed_yield_curves() -> pd.DataFrame:
    async def fetch() -> pd.DataFrame:
        async with FederalReserve() as fed:
            return await fed.yield_curves()

    return asyncio.run(fetch())


# daily par-yield panel from the Federal Reserve (cached for a day)
df = fed_yield_curves()
# weekly panel (uniform 7-day step) using the average yield over each week
weekly = df.resample("W-WED").mean().dropna()

# calibrate the Vasicek short-rate model to the panel by Kalman-filter MLE
calibrator = VasicekCurve().calibrator()
curve = calibrator.calibrate_historical_rates_dataframe(weekly, frequency=2)
print(curve.model_dump_json(indent=2, exclude={"ref_date"}))

# rebuild model-implied yields from the filtered short rate path: y = (B r - A) / tau
ttm = np.array([tenor_to_years(c) for c in weekly.columns])
a, b = curve.affine_coefficients(ttm)
A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
short_rate = calibrator.filtered_short_rate  # one value per observation date

# observed (par -> continuous) and model yields per tenor, over time
tenors = ["1Y", "2Y", "5Y", "10Y"]
fig = make_subplots(rows=2, cols=2, subplot_titles=tenors)
for k, tenor in enumerate(tenors):
    i = weekly.columns.get_loc(tenor)
    observed = 2.0 * np.log1p(weekly[tenor].to_numpy() / 2.0) * 100
    model = (B[i] * short_rate - A[i]) / ttm[i] * 100
    row, col = k // 2 + 1, k % 2 + 1
    fig.add_trace(
        go.Scatter(
            x=weekly.index,
            y=observed,
            name="observed",
            legendgroup="observed",
            showlegend=k == 0,
            line=dict(color="#636efa"),
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly.index,
            y=model,
            name="model",
            legendgroup="model",
            showlegend=k == 0,
            line=dict(color="#ef553b"),
        ),
        row=row,
        col=col,
    )
fig.update_layout(title="Observed vs Vasicek model yields")
fig.update_yaxes(title_text="yield (%)")
fig.write_image(assets_path("rates_kalman.png"), width=1600, height=800)
