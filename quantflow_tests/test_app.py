import asyncio
from datetime import date, timedelta
from typing import Iterator
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fluid.utils.redis import FluidRedis

from app.__main__ import crate_app
from app.api.deps import RedisCache


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
def clear_cache() -> None:
    async def _clear() -> None:
        # Use a dedicated redis client so the temporary event loop created by
        # asyncio.run does not leave a stale connection in the app's shared
        # pool (which the TestClient lifespan would later fail to close).
        redis = FluidRedis.create()
        try:
            await RedisCache.clear(redis.redis_cli)
        finally:
            await redis.close()

    asyncio.run(_clear())


@pytest.fixture
def client(app: FastAPI, clear_cache: None) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


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
    assert len(data["data"]) > 0
    point = data["data"][0]
    assert "date" in point
    assert "close" in point
    assert "supersmoother" in point
    assert "ewma" in point


def test_supersmoother_custom_period(client: TestClient) -> None:
    response = client.get("/.api/supersmoother?period=20&symbol=ETHUSD")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) > 0


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


def test_cointegration_endpoint(app: FastAPI, client: TestClient) -> None:
    rng = np.random.default_rng(4)
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    trend = np.cumsum(rng.normal(0, 0.015, len(days))) + 5
    series = {
        "BTCUSD": trend + rng.normal(0, 0.01, len(days)),
        "ETHUSD": 0.9 * trend + rng.normal(0, 0.01, len(days)) + 0.3,
        "SOLUSD": 1.1 * trend + rng.normal(0, 0.01, len(days)) - 0.4,
    }

    async def prices(
        symbol: str, convert_to_date: bool, frequency: object
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": days.date if convert_to_date else days,
                "close": np.exp(series[symbol]),
            }
        )

    app.state.fmp.prices = AsyncMock(side_effect=prices)

    response = client.get("/.api/cointegration?frequency=1min")
    cached_response = client.get("/.api/cointegration?frequency=1min")

    assert response.status_code == 200
    assert cached_response.status_code == 200
    assert cached_response.json() == response.json()
    data = response.json()
    assert data["dates"][:2] == ["2024-01-01", "2024-01-02"]
    assert len(data["dates"]) == 120
    assert len(data["residuals"]) == 120
    assert len(data["deltas"]) == 3
    assert np.mean(data["residuals"]) == pytest.approx(0.0, abs=1.0e-12)
    assert app.state.fmp.prices.await_count == 3
