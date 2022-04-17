from typing import Dict, Sequence

import numpy as np
from pandas import DataFrame


def pdf(data: np.ndarray, num: int = 10, precision: int = 6) -> DataFrame:
    domain = max(abs(data))
    delta = round(domain / num, precision)
    b = (num + 0.5) * delta
    bins = np.arange(-b, b + delta, delta)
    counts, _ = np.histogram(data, bins=bins)
    counts = counts / np.sum(counts)
    x = (bins[:-1] + bins[1:]) * 0.5
    return DataFrame(dict(x=x, f=counts))


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
