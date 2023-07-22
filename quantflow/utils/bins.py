from typing import Dict, Sequence

import numpy as np
from pandas import DataFrame

from .types import FloatArray


def pdf(
    data: FloatArray,
    num_bins: int | None = None,
    delta: float | None = None,
    symmetric: float | None = None,
    precision: int = 6,
) -> DataFrame:
    max_value = np.max(data)
    min_value = np.min(data)
    domain = max(abs(data)) if symmetric is not None else max_value - min_value
    if num_bins is None:
        if not delta:
            num_bins = 50
            delta_ = round(domain / (num_bins - 1), precision)
        else:
            delta_ = delta
            num_bins = round(domain / delta_)
    else:
        if delta:
            raise ValueError("Cannot specify both num_bins and delta")
        if num_bins < 2:
            raise ValueError("num_bins must be greater than 1")
        delta_ = round(domain / (num_bins - 1), precision)
    if symmetric is not None:
        b = (num_bins + 0.5) * delta_
        min_value = symmetric - b
        max_value = symmetric + b
    x = np.arange(min_value - delta_, max_value + 2 * delta_, delta_)
    bins = (x[:-1] + x[1:]) * 0.5
    pdf, _ = np.histogram(data, bins=bins, density=True)
    return DataFrame(dict(pdf=pdf), index=x[1:-1])


def event_density(df: DataFrame, columns: Sequence, num: int = 10) -> Dict:
    """Calculate the probability density of the number of events
    in the dataframe columns
    """
    bins = np.linspace(-0.5, num - 0.5, num + 1)
    data = dict(n=np.arange(num))
    for col in columns:
        counts, _ = np.histogram(df[col], bins=bins)
        counts = counts / np.sum(counts)
        data[col] = counts[:num]
    return data
