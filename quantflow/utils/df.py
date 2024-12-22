from typing import Self, TypeAlias

import numpy as np
import pandas as pd
import polars as pl

DataFrame: TypeAlias = pl.DataFrame | pd.DataFrame


def to_polars(df: DataFrame) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.DataFrame(df)
    return df


class DFutils:
    def __init__(self, df: DataFrame, period: float = 1) -> None:
        self.df = to_polars(df)
        self.period = period

    def __repr__(self) -> str:
        return repr(self.df)

    def __str__(self) -> str:
        return str(self.df)

    def clone(self, df: DataFrame) -> Self:
        return self.__class__(df, self.period)

    def with_mid(self) -> Self:
        """Adds mid and spread columns to the dataframe"""
        df = self.df
        return self.clone(
            self.df.with_columns(
                mid=0.5 * (df["ask_price"] + df["bid_price"]),
                mid_volume=0.5 * (df["ask_amount"] + df["bid_amount"]),
                spread=20000
                * (df["ask_price"] - df["bid_price"])
                / (df["ask_price"] + df["bid_price"]),
            )
        )

    def with_parkinson(self) -> Self:
        """Adds parkinson volatility column to the dataframe

        This requires the high and low columns to be present
        """
        c = 10000 / np.sqrt(4 * self.period * np.log(2))
        return self.clone(
            df=self.df.with_columns(
                pk=c * (pl.col("high") / pl.col("low")).map_elements(np.log)
            )
        )

    def with_rogers_satchel(self) -> Self:
        """Adds Rogers-Satchel volatility column to the dataframe

        This requires the high and low columns to be present
        """
        c = 10000 / np.sqrt(self.period)
        return self.clone(
            df=self.df.with_columns(
                rs=c
                * (
                    (pl.col("high") / pl.col("open")).map_elements(np.log)
                    * (pl.col("high") / pl.col("close")).map_elements(np.log)
                    + (pl.col("low") / pl.col("open")).map_elements(np.log)
                    * (pl.col("low") / pl.col("close")).map_elements(np.log)
                ).map_elements(np.sqrt)
            )
        )
