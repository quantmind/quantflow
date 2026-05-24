import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from quantflow.sp.ou import Vasicek
from quantflow.sp.wiener import WienerProcess
from quantflow.ta.ohlc import OHLC

hurst_router = APIRouter()

DEFAULT_PERIODS = (
    "10s",
    "20s",
    "30s",
    "1min",
    "2min",
    "3min",
    "5min",
    "10min",
    "30min",
)
VASICEK_PERIODS = ("10min", "20min", "30min", "1h")


class HurstWienerResponse(BaseModel):
    dates: list[str] = Field(description="Datetime strings for the path")
    values: list[float] = Field(description="Path values")
    hurst_exponent: float = Field(description="Realized variance Hurst exponent")
    realized_std: float = Field(description="Realized standard deviation (scaled)")
    estimator_periods: list[str] = Field(
        description="Sampling periods for range-based estimators"
    )
    estimator_pk: list[float] = Field(description="Parkinson volatility estimates")
    estimator_gk: list[float] = Field(description="Garman-Klass volatility estimates")
    estimator_rs: list[float] = Field(
        description="Rogers-Satchell volatility estimates"
    )
    ohlc_hurst_pk: float = Field(description="OHLC Hurst exponent (Parkinson)")
    ohlc_hurst_gk: float = Field(description="OHLC Hurst exponent (Garman-Klass)")
    ohlc_hurst_rs: float = Field(description="OHLC Hurst exponent (Rogers-Satchell)")


class HurstVasicekResponse(BaseModel):
    dates: list[str] = Field(description="Datetime strings for the path")
    values: list[float] = Field(description="Path values")
    hurst_realized: float = Field(description="Hurst exponent from realized variance")
    hurst_pk: float = Field(description="Hurst exponent (Parkinson)")
    hurst_gk: float = Field(description="Hurst exponent (Garman-Klass)")
    hurst_rs: float = Field(description="Hurst exponent (Rogers-Satchell)")


def _range_std(pdf: Any, range_seconds: float, seconds_in_day: int) -> float:
    variance = pdf.mean()
    variance = seconds_in_day * variance / range_seconds
    return math.sqrt(variance)


def _ohlc_hurst(df: pd.DataFrame, serie: str, periods: tuple) -> dict[str, float]:
    template = OHLC(
        series=serie,
        period="10m",
        rogers_satchell_variance=True,
        parkinson_variance=True,
        garman_klass_variance=True,
    )
    estimator_names = ("pk", "gk", "rs")
    time_range = []
    estimators: dict[str, list] = defaultdict(list)
    for period in periods:
        ohlc = template.model_copy(update=dict(period=period))
        rf = ohlc(df)
        ts = pd.to_timedelta(period).to_pytimedelta().total_seconds()
        time_range.append(ts)
        for name in estimator_names:
            estimators[name].append(rf[f"{serie}_{name}"].mean())
    result = {}
    for name in estimator_names:
        result[name] = (
            float(np.polyfit(np.log(time_range), np.log(estimators[name]), 1)[0]) / 2.0
        )
    return result


@hurst_router.get("/hurst-wiener")
async def hurst_wiener(
    sigma: float = Query(2.0, description="Wiener process volatility", ge=0.1, le=10),
) -> HurstWienerResponse:
    from quantflow.utils.dates import start_of_day

    seconds_in_day = 24 * 60 * 60
    wiener = WienerProcess(sigma=sigma)
    paths = wiener.sample(n=1, time_horizon=1, time_steps=seconds_in_day)
    wiener_df = paths.as_datetime_df(start=start_of_day(), unit="D").reset_index()

    dates = [str(d) for d in wiener_df.iloc[:, 0]]
    values = [float(v) for v in wiener_df.iloc[:, 1]]
    hurst_exp = float(paths.hurst_exponent())
    realized_std = float(paths.paths_std(scaled=True)[0])

    periods = list(DEFAULT_PERIODS)
    pk_vals = []
    gk_vals = []
    rs_vals = []
    template = OHLC(
        series="0",
        period="10m",
        rogers_satchell_variance=True,
        parkinson_variance=True,
        garman_klass_variance=True,
    )
    for period in periods:
        ohlc = template.model_copy(update=dict(period=period))
        rf = ohlc(wiener_df)
        ts = pd.to_timedelta(period).to_pytimedelta().total_seconds()
        pk_vals.append(_range_std(rf["0_pk"], ts, seconds_in_day))
        gk_vals.append(_range_std(rf["0_gk"], ts, seconds_in_day))
        rs_vals.append(_range_std(rf["0_rs"], ts, seconds_in_day))

    ohlc_hurst = _ohlc_hurst(wiener_df, "0", DEFAULT_PERIODS)

    return HurstWienerResponse(
        dates=dates,
        values=values,
        hurst_exponent=hurst_exp,
        realized_std=realized_std,
        estimator_periods=periods,
        estimator_pk=pk_vals,
        estimator_gk=gk_vals,
        estimator_rs=rs_vals,
        ohlc_hurst_pk=ohlc_hurst["pk"],
        ohlc_hurst_gk=ohlc_hurst["gk"],
        ohlc_hurst_rs=ohlc_hurst["rs"],
    )


@hurst_router.get("/hurst-vasicek")
async def hurst_vasicek(
    kappa: float = Query(10.0, description="Mean reversion speed", ge=1.0, le=500.0),
) -> HurstVasicekResponse:
    from quantflow.utils.dates import start_of_day

    p = Vasicek(kappa=kappa)
    paths = p.sample(n=1, time_horizon=1, time_steps=24 * 60 * 6)
    hurst_real = float(paths.hurst_exponent())

    pdf = pd.DataFrame(
        {"0": paths.path(0)}, index=paths.dates(start=start_of_day())
    ).reset_index()

    ohlc_h = _ohlc_hurst(pdf, "0", VASICEK_PERIODS)

    dates = [str(d) for d in pdf.iloc[:, 0]]
    values = [float(v) for v in pdf.iloc[:, 1]]

    return HurstVasicekResponse(
        dates=dates,
        values=values,
        hurst_realized=hurst_real,
        hurst_pk=ohlc_h["pk"],
        hurst_gk=ohlc_h["gk"],
        hurst_rs=ohlc_h["rs"],
    )
