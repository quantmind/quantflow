from __future__ import annotations

import numpy as np
import pytest

from quantflow.dists import MvNormal
from quantflow.ta.kalman import (
    LinearGaussianModel,
    MeanAndCov,
    StateSpaceModel,
    UnscentedKalmanFilter,
)


def linear_1d() -> LinearGaussianModel:
    return LinearGaussianModel(
        F=np.array([[0.95]]),
        Q=np.array([[0.01]]),
        H=np.array([1.0]),
        R=np.eye(1) * 0.5,
        mu0=np.array([0.0]),
        cov0=np.array([[1.0]]),
    )


def linear_2d() -> LinearGaussianModel:
    return LinearGaussianModel(
        F=np.array([[0.9, 0.1], [0.0, 0.95]]),
        Q=np.eye(2) * 0.05,
        H=np.eye(2),
        R=np.eye(2) * 0.2,
        mu0=np.zeros(2),
        cov0=np.eye(2),
    )


class SineModel(StateSpaceModel):
    """A non-linear scalar model: x_t = sin(x_{t-1}) + noise, y_t = x_t + noise."""

    def get_px0(self) -> MvNormal:
        return MvNormal(mean=np.array([0.5]), cov=np.eye(1) * 0.5)

    def get_px(self, t: int, xp: np.ndarray) -> MvNormal:
        return MvNormal(mean=np.sin(xp), cov=np.eye(1) * 0.01)

    def get_py(self, t: int, xp: np.ndarray, x: np.ndarray) -> MvNormal:
        return MvNormal(mean=np.asarray(x, dtype=float), cov=np.eye(1) * 0.1)


class TestMeanAndCov:
    def test_unpacking(self) -> None:
        mc = MeanAndCov(mean=np.array([1.0]), cov=np.array([[2.0]]))
        m, c = mc
        assert m[0] == pytest.approx(1.0)
        assert c[0, 0] == pytest.approx(2.0)


class TestLinearGaussianModel:
    def test_shape_coercion(self) -> None:
        m = LinearGaussianModel(
            F=np.array(0.95),
            Q=np.array(0.01),
            H=np.array(1.0),
            R=np.array(0.5),
            mu0=np.array(0.0),
            cov0=np.array(1.0),
        )
        assert m.F.shape == (1, 1)
        assert m.Q.shape == (1, 1)
        assert m.R.shape == (1, 1)
        assert m.cov0.shape == (1, 1)
        assert m.mu0.shape == (1,)
        assert m.n_x == 1

    def test_get_px0(self) -> None:
        mean, cov = linear_1d().get_px0().mean_and_cov()
        assert mean[0] == pytest.approx(0.0)
        assert cov[0, 0] == pytest.approx(1.0)

    def test_get_px(self) -> None:
        law = linear_1d().get_px(1, np.array([2.0]))
        assert law.mean[0] == pytest.approx(0.95 * 2.0)
        assert law.cov[0, 0] == pytest.approx(0.01)

    def test_get_py(self) -> None:
        law = linear_1d().get_py(1, np.array([0.0]), np.array([3.0]))
        assert law.mean[0] == pytest.approx(3.0)
        assert law.cov[0, 0] == pytest.approx(0.5)


class TestKalmanFilter:
    def test_filter_shapes_1d(self) -> None:
        y = np.random.default_rng(0).normal(size=(15, 1))
        kf = linear_1d().kalman_filter(y)
        ll = kf.filter()
        assert len(kf.states) == 15
        assert np.isfinite(ll)
        for s in kf.states:
            assert s.mean.shape == (1,)
            assert s.cov.shape == (1, 1)

    def test_filter_shapes_2d(self) -> None:
        y = np.random.default_rng(1).normal(size=(20, 2))
        kf = linear_2d().kalman_filter(y)
        ll = kf.filter()
        assert len(kf.states) == 20
        assert ll < 0
        for s in kf.states:
            assert s.mean.shape == (2,)
            assert s.cov.shape == (2, 2)

    def test_observations_promoted_to_2d(self) -> None:
        kf = linear_1d().kalman_filter(np.zeros(10))
        assert kf.y.shape == (1, 10)

    def test_predict_step(self) -> None:
        # x=2, P=1 -> mean = 0.95*2, cov = 0.95^2 + 0.01
        kf = linear_1d().kalman_filter(np.zeros((1, 1)))
        pred = kf.predict(MeanAndCov(mean=np.array([2.0]), cov=np.array([[1.0]])))
        assert pred.mean[0] == pytest.approx(0.95 * 2.0)
        assert pred.cov[0, 0] == pytest.approx(0.95**2 + 0.01)

    def test_update_is_exact(self) -> None:
        # prior (0, 1), y=1.2, R=0.5 -> S=1.5, K=2/3, mean=0.8, cov=1/3
        model = linear_1d()
        kf = model.kalman_filter(np.array([[1.2]]))
        state, ll = kf.update(model.get_px0().mean_and_cov(), np.array([1.2]))
        assert state.mean[0] == pytest.approx(0.8)
        assert state.cov[0, 0] == pytest.approx(1.0 / 3.0)
        assert np.isfinite(ll)

    def test_variance_reduces_toward_observation(self) -> None:
        m = LinearGaussianModel(
            F=np.eye(1),
            Q=np.eye(1) * 1e-8,
            H=np.array([1.0]),
            R=np.eye(1) * 0.01,
            mu0=np.array([0.0]),
            cov0=np.eye(1) * 100.0,
        )
        kf = m.kalman_filter(np.ones((5, 1)) * 5.0)
        kf.filter()
        assert float(kf.states[-1].cov.item()) < 1.0
        assert float(kf.states[-1].mean.item()) == pytest.approx(5.0, abs=0.1)

    def test_observation_matrix_shapes_match(self) -> None:
        # a 1d column-vector H and the equivalent 2d (n_y, 1) H must agree.
        y = np.random.default_rng(3).normal(size=(12, 3))
        vec = LinearGaussianModel(
            F=np.array([[0.95]]),
            Q=np.array([[0.01]]),
            H=np.array([1.0, 2.0, 0.5]),
            R=np.eye(3) * 0.3,
            mu0=np.array([0.0]),
            cov0=np.array([[1.0]]),
        )
        mat = vec.model_copy(update={"H": np.array([[1.0], [2.0], [0.5]])})
        vec_kf = vec.kalman_filter(y)
        mat_kf = mat.kalman_filter(y)
        vec_ll = vec_kf.filter()
        mat_ll = mat_kf.filter()
        assert vec_ll == pytest.approx(mat_ll)
        for a, b in zip(vec_kf.states, mat_kf.states):
            assert np.allclose(a.mean, b.mean)
            assert np.allclose(a.cov, b.cov)


class TestUnscentedKalmanFilter:
    def test_reduces_to_kalman_filter(self) -> None:
        # On a linear-Gaussian model the UKF is exact: it matches the KF.
        y = np.random.default_rng(5).normal(size=(20, 2))
        kf = linear_2d().kalman_filter(y)
        ukf = UnscentedKalmanFilter(model=linear_2d(), y=y)
        kll = kf.filter()
        ull = ukf.filter()
        assert kll == pytest.approx(ull)
        for k, u in zip(kf.states, ukf.states):
            assert np.allclose(k.mean, u.mean)
            assert np.allclose(k.cov, u.cov)

    def test_runs_on_nonlinear_model(self) -> None:
        rng = np.random.default_rng(7)
        x_true, obs = 0.5, []
        for i in range(15):
            if i > 0:
                x_true = float(np.sin(x_true)) + 0.1 * rng.normal()
            obs.append([x_true + 0.3 * rng.normal()])
        ukf = UnscentedKalmanFilter(model=SineModel(), y=np.array(obs))
        ll = ukf.filter()
        assert len(ukf.states) == 15
        assert np.isfinite(ll)
        for s in ukf.states:
            assert s.mean.shape == (1,)
            assert s.cov.shape == (1, 1)

    def test_default_parameters(self) -> None:
        ukf = UnscentedKalmanFilter(model=SineModel(), y=np.zeros((8, 1)))
        assert ukf.alpha == pytest.approx(1e-3)
        assert ukf.beta == pytest.approx(2.0)
        assert ukf.kappa == pytest.approx(0.0)
        ll = ukf.filter()
        assert len(ukf.states) == 8
        assert np.isfinite(ll)
