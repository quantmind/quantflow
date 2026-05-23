from datetime import timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .base import DataFrame


class OHLC(BaseModel):
    """Aggregates OHLC data over a given period and series

    Optionally calculates the range-based variance estimators for the series.
    Range-based estimator are called like that because they are calculated from the
    difference between the period high and low.
    """

    series: str
    """series to aggregate"""
    period: str | timedelta
    """down-sampling period, e.g. 1h, 1d, 1w"""
    index_column: str = "index"
    """column to group by"""
    parkinson_variance: bool = False
    """add Parkinson variance column"""
    garman_klass_variance: bool = False
    """add Garman Klass variance column"""
    rogers_satchell_variance: bool = False
    """add Rogers Satchell variance column"""
    percent_variance: bool = False
    """log-transform the variance columns"""

    def __call__(self, df: DataFrame) -> pd.DataFrame:
        """Returns a dataframe with OHLC data sampled over the given period"""
        data = (
            df.set_index(self.index_column) if self.index_column in df.columns else df
        )
        col = self._resolve_column(data)
        resampled = data[col].resample(self.period)
        s = self.series
        result = pd.DataFrame(
            {
                f"{s}_open": resampled.first(),
                f"{s}_high": resampled.max(),
                f"{s}_low": resampled.min(),
                f"{s}_close": resampled.last(),
                f"{s}_mean": resampled.mean(),
            }
        ).reset_index()
        if self.parkinson_variance:
            result = self.parkinson(result)
        if self.garman_klass_variance:
            result = self.garman_klass(result)
        if self.rogers_satchell_variance:
            result = self.rogers_satchell(result)
        return result

    def _resolve_column(self, df: pd.DataFrame) -> str | int:
        """Resolve the series column name, handling int vs string column names."""
        if self.series in df.columns:
            return self.series
        try:
            col: int = int(self.series)
            if col in df.columns:
                return col
        except ValueError:
            pass
        raise KeyError(f"Column '{self.series}' not found in dataframe")

    def _col(self, suffix: str) -> str:
        return f"{self.series}_{suffix}"

    def _get_col(self, df: pd.DataFrame, suffix: str) -> pd.Series:
        col = df[self._col(suffix)]
        return np.log(col) if self.percent_variance else col

    def parkinson(self, df: DataFrame) -> pd.DataFrame:
        """Adds parkinson variance column to the dataframe

        This requires the series high and low columns to be present
        """
        high = self._get_col(df, "high")
        low = self._get_col(df, "low")
        pk = (high - low) ** 2 / np.sqrt(4 * np.log(2))
        df = df.copy()
        df[self._col("pk")] = pk
        return df

    def garman_klass(self, df: DataFrame) -> pd.DataFrame:
        """Adds Garman Klass variance estimator column to the dataframe

        This requires the series high and low columns to be present.
        """
        o = self._get_col(df, "open")
        hh = self._get_col(df, "high") - o
        ll = self._get_col(df, "low") - o
        cc = self._get_col(df, "close") - o
        gk = (
            0.522 * (hh - ll) ** 2
            - 0.019 * (cc * (hh + ll) + 2.0 * ll * hh)
            - 0.383 * cc**2
        )
        df = df.copy()
        df[self._col("gk")] = gk
        return df

    def rogers_satchell(self, df: DataFrame) -> pd.DataFrame:
        """Adds Rogers Satchell variance estimator column to the dataframe

        This requires the series high and low columns to be present.
        """
        o = self._get_col(df, "open")
        hh = self._get_col(df, "high") - o
        ll = self._get_col(df, "low") - o
        cc = self._get_col(df, "close") - o
        rs = hh * (hh - cc) + ll * (ll - cc)
        df = df.copy()
        df[self._col("rs")] = rs
        return df
