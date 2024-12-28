from datetime import timedelta

import numpy as np
import polars as pl
from pydantic import BaseModel

from .base import DataFrame, to_polars


class OHLC(BaseModel):
    """Aggregates OHLC data over a given period and serie

    Optionally calculates the range-based variance estimators for the serie.
    Range-based estimator are called like that because they are calculated from the
    difference between the period high and low.
    """

    serie: str
    """serie to aggregate"""
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

    @property
    def open_col(self) -> pl.Expr:
        return self.var_column("open")

    @property
    def high_col(self) -> pl.Expr:
        return self.var_column("high")

    @property
    def low_col(self) -> pl.Expr:
        return self.var_column("low")

    @property
    def close_col(self) -> pl.Expr:
        return self.var_column("close")

    def __call__(self, df: DataFrame) -> pl.DataFrame:
        """Returns a dataframe with OHLC data sampled over the given period"""
        result = (
            to_polars(df, copy=True)
            .group_by_dynamic(self.index_column, every=self.period)
            .agg(
                pl.col(self.serie).first().alias(f"{self.serie}_open"),
                pl.col(self.serie).max().alias(f"{self.serie}_high"),
                pl.col(self.serie).min().alias(f"{self.serie}_low"),
                pl.col(self.serie).last().alias(f"{self.serie}_close"),
                pl.col(self.serie).mean().alias(f"{self.serie}_mean"),
            )
        )
        if self.parkinson_variance:
            result = self.parkinson(result)
        if self.garman_klass_variance:
            result = self.garman_klass(result)
        if self.rogers_satchell_variance:
            result = self.rogers_satchell(result)
        return result

    def parkinson(self, df: DataFrame) -> pl.DataFrame:
        """Adds parkinson variance column to the dataframe

        This requires the serie high and low columns to be present
        """
        c = (self.high_col - self.low_col) ** 2 / np.sqrt(4 * np.log(2))
        return to_polars(df).with_columns(c.alias(f"{self.serie}_pk"))

    def garman_klass(self, df: DataFrame) -> pl.DataFrame:
        """Adds Garman Klass variance estimator column to the dataframe

        This requires the serie high and low columns to be present.
        """
        open = self.open_col
        hh = self.high_col - open
        ll = self.low_col - open
        cc = self.close_col - open
        c = (
            0.522 * (hh - ll) ** 2
            - 0.019 * (cc * (hh + ll) + 2.0 * ll * hh)
            - 0.383 * cc**2
        )
        return to_polars(df).with_columns(c.alias(f"{self.serie}_gk"))

    def rogers_satchell(self, df: DataFrame) -> pl.DataFrame:
        """Adds Rogers Satchell variance estimator column to the dataframe

        This requires the serie high and low columns to be present.
        """
        open = self.open_col
        hh = self.high_col - open
        ll = self.low_col - open
        cc = self.close_col - open
        c = hh * (hh - cc) + ll * (ll - cc)
        return to_polars(df).with_columns(c.alias(f"{self.serie}_rs"))

    def var_column(self, suffix: str) -> pl.Expr:
        """Returns a polars expression for the OHLC column"""
        col = pl.col(f"{self.serie}_{suffix}")
        return col.log() if self.percent_variance else col
