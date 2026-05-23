from datetime import date, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.__main__ import crate_app


@pytest.fixture
def mock_fmp() -> AsyncMock:
    fmp = AsyncMock()
    start = date.today() - timedelta(days=30)
    dates = [start + timedelta(days=i) for i in range(30)]
    prices_df = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(90.0, 110.0, 30).tolist(),
        }
    )
    fmp.prices = AsyncMock(return_value=prices_df)
    return fmp


@pytest.fixture
def app(mock_fmp: AsyncMock) -> FastAPI:
    application = crate_app()
    application.state.fmp = mock_fmp
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_status(client: TestClient) -> None:
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_supersmoother(client: TestClient) -> None:
    response = client.get("/.api/supersmoother?period=10&symbol=BTCUSD")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 30
    point = data["data"][0]
    assert "date" in point
    assert "close" in point
    assert "supersmoother" in point
    assert "ewma" in point


def test_supersmoother_custom_period(client: TestClient) -> None:
    response = client.get("/.api/supersmoother?period=20&symbol=ETHUSD")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 30


def test_supersmoother_invalid_period(client: TestClient) -> None:
    response = client.get("/.api/supersmoother?period=1&symbol=BTCUSD")
    assert response.status_code == 422


def test_gaussian_sampling(client: TestClient) -> None:
    response = client.get("/.api/gaussian-sampling?kappa=1.0&samples=100")
    assert response.status_code == 200
    data = response.json()
    assert "x" in data
    assert "simulation" in data
    assert "analytical" in data
    assert len(data["x"]) == len(data["simulation"])


def test_poisson_sampling(client: TestClient) -> None:
    response = client.get("/.api/poisson-sampling?intensity=2.0&samples=100")
    assert response.status_code == 200
    data = response.json()
    assert "x" in data
    assert "simulation" in data
    assert "analytical" in data


def test_double_exponential_sampling(client: TestClient) -> None:
    response = client.get("/.api/double-exponential-sampling?log_kappa=0.1&samples=100")
    assert response.status_code == 200
    data = response.json()
    assert "x" in data
    assert "simulation" in data
    assert "analytical" in data
    assert "char_x" in data
    assert "char_y" in data


def test_heston_vol_surface_jd(client: TestClient) -> None:
    response = client.get("/.api/heston-vol-surface?model=jd&vol=0.4&sigma=0.5")
    assert response.status_code == 200
    data = response.json()
    assert "moneyness" in data
    assert "ttm" in data
    assert "iv" in data
    assert len(data["ttm"]) == 10
    assert len(data["moneyness"]) == 51


def test_heston_vol_surface_hj(client: TestClient) -> None:
    response = client.get(
        "/.api/heston-vol-surface?model=hj&vol=0.4&sigma=0.5&kappa=1.0&rho=-0.3"
    )
    assert response.status_code == 200
    data = response.json()
    assert "iv" in data


def test_hurst_wiener(client: TestClient) -> None:
    response = client.get("/.api/hurst-wiener?sigma=2.0")
    assert response.status_code == 200
    data = response.json()
    assert "dates" in data
    assert "values" in data
    assert "hurst_exponent" in data
    assert "estimator_periods" in data


def test_hurst_vasicek(client: TestClient) -> None:
    response = client.get("/.api/hurst-vasicek?kappa=10.0")
    assert response.status_code == 200
    data = response.json()
    assert "dates" in data
    assert "values" in data
    assert "hurst_realized" in data
    assert "hurst_pk" in data


def test_yield_curve(client: TestClient) -> None:
    response = client.get(
        "/.api/yield-curve?ttm=0.25&ttm=0.5&ttm=1.0&ttm=2.0"
        "&rates=0.04&rates=0.042&rates=0.045&rates=0.048"
    )
    assert response.status_code == 200
    data = response.json()
    assert "curve" in data
    assert "ttm" in data
    assert "rates" in data
