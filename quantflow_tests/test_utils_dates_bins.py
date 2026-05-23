from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantflow.utils.bins import event_density, pdf
from quantflow.utils.dates import as_date, as_utc, isoformat, start_of_day, to_date_iso


def test_pdf_with_default_bins() -> None:
    data = np.array([0.0, 1.0, 2.0, 3.0])
    result = pdf(data)
    assert "pdf" in result.columns
    assert len(result) > 0


def test_pdf_with_delta_and_symmetric() -> None:
    data = np.array([-1.0, -0.5, 0.5, 1.0])
    result = pdf(data, delta=0.5, symmetric=0.0)
    assert result.index.min() < 0.0
    assert result.index.max() > 0.0


def test_pdf_invalid_num_bins_and_delta() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        pdf(np.array([1.0, 2.0]), num_bins=10, delta=0.1)
    with pytest.raises(ValueError, match="greater than 1"):
        pdf(np.array([1.0, 2.0]), num_bins=1)


def test_event_density() -> None:
    df = pd.DataFrame({"a": [0, 1, 1, 2], "b": [1, 1, 2, 2]})
    density = event_density(df, columns=["a", "b"], num=3)
    assert np.array_equal(density["n"], np.array([0, 1, 2]))
    assert np.isclose(np.sum(density["a"]), 1.0)
    assert np.isclose(np.sum(density["b"]), 1.0)


def test_date_helpers() -> None:
    dt = datetime(2024, 1, 2, 15, 30, tzinfo=timezone.utc)
    assert as_utc(dt) == dt
    assert as_utc(date(2024, 1, 2)) == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert isoformat("2024-01-02") == "2024-01-02"
    assert isoformat(date(2024, 1, 2)) == "2024-01-02"
    assert start_of_day(dt) == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert as_date(dt) == date(2024, 1, 2)
    assert as_date(date(2024, 1, 3)) == date(2024, 1, 3)
    assert to_date_iso(date(2024, 1, 2)) == "2024-01-02"
    assert to_date_iso("2024-01-02") == "2024-01-02"
    assert to_date_iso(None) is None
