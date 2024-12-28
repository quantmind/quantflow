from typing import TypeAlias

import pandas as pd
import polars as pl

DataFrame: TypeAlias = pl.DataFrame | pd.DataFrame


def to_polars(df: DataFrame) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.DataFrame(df)
    return df
