from typing import TypeAlias

import pandas as pd
import polars as pl

DataFrame: TypeAlias = pl.DataFrame | pd.DataFrame


def to_polars(df: DataFrame, *, copy: bool = False) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.DataFrame(df)
    elif copy:
        return df.clone()
    return df
