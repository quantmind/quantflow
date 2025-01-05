from typing import Any, Sequence, cast

import numpy as np
from pandas import DataFrame

from .types import FloatArray


def pdf(
    data: FloatArray,
    *,
    num_bins: int | None = None,
    delta: float | None = None,
    symmetric: float | None = None,
    precision: int = 6,
) -> DataFrame:
    """Extract a probability density function from the data as a DataFrame
    with index given by the bin centers and a single column `pdf` with the
    estimated probability density function values

    :param data: the data to extract the PDF from
    :param num_bins: the number of bins to use in the histogram,
        if not provided it is calculated from the `delta` parameter (if provided)
        or set to 50
    :param delta: the spacing between bins, if not provided it is calculated
        from the `num_bins`
    :param symmetric: if provided, the bins are centered around this value
    :param precision: the precision to use in the calculation
    """
    max_value = cast(float, np.max(data))
    min_value = cast(float, np.min(data))
    domain: float = max(abs(data)) if symmetric is not None else max_value - min_value  # type: ignore
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


def event_density(
    df: DataFrame, columns: Sequence[str], num: int = 10
) -> dict[str, Any]:
    """Calculate the probability density of the number of events
    in the dataframe columns
    """
    bins = np.linspace(-0.5, num - 0.5, num + 1)
    data = dict(n=np.arange(num))
    for col in columns:
        counts, _ = np.histogram(df[col], bins=bins)
        counts = counts / np.sum(counts)
        data[col] = counts[:num]  # type: ignore
    return data
