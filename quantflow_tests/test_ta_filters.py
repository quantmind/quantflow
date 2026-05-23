from __future__ import annotations

import pytest

from quantflow.ta.ewma import EWMA
from quantflow.ta.kalman import KalmanFilter
from quantflow.ta.supersmoother import SuperSmoother


def test_ewma_initialization_and_alpha() -> None:
    ewma = EWMA(period=10)
    assert ewma.current_value is None
    assert 0.0 < ewma.alpha < 1.0


def test_ewma_first_and_second_update() -> None:
    ewma = EWMA(period=5)
    first = ewma.update(10.0)
    second = ewma.update(20.0)
    assert first == 10.0
    assert 10.0 < second < 20.0
    assert ewma.current_value == second


def test_ewma_asymmetric_tau_branch() -> None:
    ewma = EWMA(period=4, tau=1.0)
    ewma.update(10.0)
    down = ewma.update(0.0)
    up = ewma.update(20.0)
    assert down == 10.0
    assert up > down


def test_ewma_factory_methods() -> None:
    ewma_half_life = EWMA.from_half_life(half_life=2.0)
    ewma_alpha = EWMA.from_alpha(alpha=0.5)
    assert ewma_half_life.period >= 1
    assert ewma_alpha.period >= 1


def test_kalman_initialization_and_first_update() -> None:
    kf = KalmanFilter(R=1.0, Q=0.1)
    assert kf.value() is None
    first = kf.update(5.0)
    assert first == 5.0
    assert kf.value() == 5.0
    assert kf.error_covariance == 1.0


def test_kalman_steady_updates_and_properties() -> None:
    kf = KalmanFilter(R=1.0, Q=0.1)
    kf.update(10.0)
    updated = kf.update(12.0)
    assert 10.0 < updated < 12.0
    assert 0.0 < kf.kalman_gain < 1.0
    assert kf.error_covariance > 0.0


def test_supersmoother_initialization() -> None:
    smoother = SuperSmoother(period=10)
    assert smoother.raw_value() is None
    assert smoother.value() is None


def test_supersmoother_first_two_updates() -> None:
    smoother = SuperSmoother(period=10)
    first = smoother.update(1.0)
    second = smoother.update(3.0)
    assert first == 1.0
    assert second == pytest.approx(2.0)
    assert smoother.raw_value() == 3.0
    assert smoother.value() == second


def test_supersmoother_steady_updates() -> None:
    smoother = SuperSmoother(period=8)
    smoother.update(1.0)
    smoother.update(3.0)
    third = smoother.update(5.0)
    assert isinstance(third, float)
    assert smoother.value() == third
