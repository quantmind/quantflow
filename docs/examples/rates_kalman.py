import asyncio
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from docs.examples._utils import assets_path, cached_df
from quantflow.data.fed import FederalReserve
from quantflow.rates.calibration import tenor_to_years
from quantflow.rates.cir import CIRCurve
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
ttm = np.array([tenor_to_years(c) for c in weekly.columns])

# Vasicek: linear-Gaussian, fitted with the exact Kalman filter
vasicek_cal = VasicekCurve().calibrator()
vasicek = vasicek_cal.calibrate_historical_rates_dataframe(weekly, frequency=2)
print("Vasicek (Kalman filter)")
print(vasicek.model_dump_json(indent=2, exclude={"ref_date"}))

# CIR: state-dependent variance, fitted with the unscented Kalman filter
cir_cal = CIRCurve().calibrator()
cir = cir_cal.calibrate_historical_rates_dataframe(weekly, frequency=2)
print("CIR (unscented Kalman filter)")
print(cir.model_dump_json(indent=2, exclude={"ref_date"}))

# rebuild model-implied yields from each filtered short rate path: y = (B r - A) / tau
va_a, va_b = vasicek.affine_coefficients(ttm)
va_A, va_B = np.asarray(va_a, dtype=float), np.asarray(va_b, dtype=float)
va_short = vasicek_cal.filtered_short_rate
cir_a, cir_b = cir.affine_coefficients(ttm)
cir_A, cir_B = np.asarray(cir_a, dtype=float), np.asarray(cir_b, dtype=float)
cir_short = cir_cal.filtered_short_rate

# observed (par -> continuous) and both model yields per tenor, over time
tenors = ["1Y", "2Y", "5Y", "10Y"]
colours = dict(observed="#636efa", Vasicek="#ef553b", CIR="#00cc96")
fig = make_subplots(rows=2, cols=2, subplot_titles=tenors)
for k, tenor in enumerate(tenors):
    i = weekly.columns.get_loc(tenor)
    row, col = k // 2 + 1, k % 2 + 1
    series = {
        "observed": 2.0 * np.log1p(weekly[tenor].to_numpy() / 2.0) * 100,
        "Vasicek": (va_B[i] * va_short - va_A[i]) / ttm[i] * 100,
        "CIR": (cir_B[i] * cir_short - cir_A[i]) / ttm[i] * 100,
    }
    for name, values in series.items():
        fig.add_trace(
            go.Scatter(
                x=weekly.index,
                y=values,
                name=name,
                legendgroup=name,
                showlegend=k == 0,
                line=dict(color=colours[name]),
            ),
            row=row,
            col=col,
        )
fig.update_layout(title="Observed vs Vasicek and CIR model yields")
fig.update_yaxes(title_text="yield (%)")
fig.write_image(assets_path("rates_kalman.png"), width=1600, height=800)
