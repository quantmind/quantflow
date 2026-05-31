from __future__ import annotations

import numpy as np
import pytest

from quantflow.dists import MvNormal
from quantflow.ta.kalman import (
    KalmanFilter,
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

    def get_proposal0(self, data: np.ndarray) -> MvNormal:
        return self.get_px0()

    def get_proposal(self, t: int, xp: np.ndarray, data: np.ndarray) -> MvNormal:
        return self.get_px(t, xp)


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

    def test_proposal0_is_exact_update(self) -> None:
        # prior (0, 1), y=1.2, R=0.5 -> S=1.5, K=2/3, mean=0.8, cov=1/3
        mean, cov = linear_1d().get_proposal0(np.array([[1.2]])).mean_and_cov()
        assert mean[0] == pytest.approx(0.8)
        assert cov[0, 0] == pytest.approx(1.0 / 3.0)


class TestKalmanFilter:
    def test_filter_shapes_1d(self) -> None:
        data = np.random.default_rng(0).normal(size=(15, 1))
        states, ll = KalmanFilter(model=linear_1d(), data=data).filter()
        assert len(states) == 15
        assert np.isfinite(ll)
        for s in states:
            assert s.mean.shape == (1,)
            assert s.cov.shape == (1, 1)

    def test_filter_shapes_2d(self) -> None:
        data = np.random.default_rng(1).normal(size=(20, 2))
        states, ll = KalmanFilter(model=linear_2d(), data=data).filter()
        assert len(states) == 20
        assert ll < 0
        for s in states:
            assert s.mean.shape == (2,)
            assert s.cov.shape == (2, 2)

    def test_data_promoted_to_2d(self) -> None:
        kf = KalmanFilter(model=linear_1d(), data=np.zeros(10))
        assert kf.data.shape == (1, 10)

    def test_variance_reduces_toward_observation(self) -> None:
        m = LinearGaussianModel(
            F=np.eye(1),
            Q=np.eye(1) * 1e-8,
            H=np.array([1.0]),
            R=np.eye(1) * 0.01,
            mu0=np.array([0.0]),
            cov0=np.eye(1) * 100.0,
        )
        states, _ = KalmanFilter(model=m, data=np.ones((5, 1)) * 5.0).filter()
        assert float(states[-1].cov.item()) < 1.0
        assert float(states[-1].mean.item()) == pytest.approx(5.0, abs=0.1)

    def test_sherman_morrison_matches_general(self) -> None:
        # H as a 1d column vector with scaled-identity R takes the O(n_y)
        # Sherman-Morrison path; the equivalent 2d H takes the general path.
        data = np.random.default_rng(3).normal(size=(12, 3))
        sm = LinearGaussianModel(
            F=np.array([[0.95]]),
            Q=np.array([[0.01]]),
            H=np.array([1.0, 2.0, 0.5]),
            R=np.eye(3) * 0.3,
            mu0=np.array([0.0]),
            cov0=np.array([[1.0]]),
        )
        general = sm.model_copy(update={"H": np.array([[1.0], [2.0], [0.5]])})
        sm_states, sm_ll = KalmanFilter(model=sm, data=data).filter()
        gen_states, gen_ll = KalmanFilter(model=general, data=data).filter()
        assert sm_ll == pytest.approx(gen_ll)
        for a, b in zip(sm_states, gen_states):
            assert np.allclose(a.mean, b.mean)
            assert np.allclose(a.cov, b.cov)


class TestUnscentedKalmanFilter:
    def test_reduces_to_kalman_filter(self) -> None:
        # On a linear-Gaussian model the UKF is exact: it matches the KF.
        data = np.random.default_rng(5).normal(size=(20, 2))
        ks, kll = KalmanFilter(model=linear_2d(), data=data).filter()
        us, ull = UnscentedKalmanFilter(model=linear_2d(), data=data).filter()
        assert kll == pytest.approx(ull)
        for k, u in zip(ks, us):
            assert np.allclose(k.mean, u.mean)
            assert np.allclose(k.cov, u.cov)

    def test_runs_on_nonlinear_model(self) -> None:
        rng = np.random.default_rng(7)
        x_true, obs = 0.5, []
        for i in range(15):
            if i > 0:
                x_true = float(np.sin(x_true)) + 0.1 * rng.normal()
            obs.append([x_true + 0.3 * rng.normal()])
        states, ll = UnscentedKalmanFilter(
            model=SineModel(), data=np.array(obs)
        ).filter()
        assert len(states) == 15
        assert np.isfinite(ll)
        for s in states:
            assert s.mean.shape == (1,)
            assert s.cov.shape == (1, 1)

    def test_default_parameters(self) -> None:
        ukf = UnscentedKalmanFilter(model=SineModel(), data=np.zeros((8, 1)))
        assert ukf.alpha == pytest.approx(1e-3)
        assert ukf.beta == pytest.approx(2.0)
        assert ukf.kappa == pytest.approx(0.0)
        states, ll = ukf.filter()
        assert len(states) == 8
        assert np.isfinite(ll)
