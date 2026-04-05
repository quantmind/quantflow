from __future__ import annotations

import numpy as np
import pytest

from quantflow.options.divfm import DIVFMPricer, DIVFMWeights
from quantflow.options.divfm.weights import LayerWeights, SubnetWeights
from quantflow.options.pricer import OptionPricerBase

try:
    import torch

    from quantflow.options.divfm.network import DIVFMNetwork
    from quantflow.options.divfm.trainer import DayData, DIVFMTrainer

    has_torch = True
except ImportError:
    has_torch = False


NUM_FACTORS = 5
HIDDEN_SIZE = 16
NUM_HIDDEN = 2
N = 40  # number of synthetic options


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer(
    in_size: int,
    out_size: int,
    apply_activation: bool = True,
) -> LayerWeights:
    """LayerWeights with zero linear weights (output = 0 before BN)."""
    return LayerWeights(
        weight=np.zeros((out_size, in_size), dtype=np.float32),
        bias=np.zeros(out_size, dtype=np.float32),
        bn_mean=np.zeros(out_size, dtype=np.float32),
        bn_var=np.ones(out_size, dtype=np.float32),
        bn_gamma=np.ones(out_size, dtype=np.float32),
        bn_beta=np.zeros(out_size, dtype=np.float32),
        apply_activation=apply_activation,
    )


def _make_subnet(input_size: int, output_size: int) -> SubnetWeights:
    layers = []
    in_size = input_size
    for _ in range(NUM_HIDDEN):
        layers.append(_make_layer(in_size, HIDDEN_SIZE, apply_activation=True))
        in_size = HIDDEN_SIZE
    layers.append(_make_layer(in_size, output_size, apply_activation=False))
    return SubnetWeights(layers=layers)


@pytest.fixture
def weights() -> DIVFMWeights:
    return DIVFMWeights(
        subnet_ttm=_make_subnet(1, 1),
        subnet_moneyness=_make_subnet(1, 1),
        subnet_joint=_make_subnet(2, NUM_FACTORS - 3),
        num_factors=NUM_FACTORS,
    )


@pytest.fixture
def pricer(weights: DIVFMWeights) -> DIVFMPricer:
    return DIVFMPricer(weights=weights)


def _synthetic_options(
    n: int = N,
    iv: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic option data with constant implied vol."""
    rng = np.random.default_rng(42)
    moneyness_ttm = rng.uniform(-2.0, 2.0, n).astype(np.float32)
    ttm = rng.uniform(0.1, 2.0, n).astype(np.float32)
    implied_vols = np.full(n, iv, dtype=np.float64)
    return moneyness_ttm, ttm, implied_vols


# ---------------------------------------------------------------------------
# DIVFMWeights tests (no torch required)
# ---------------------------------------------------------------------------


def test_weights_is_pricer_base(pricer: DIVFMPricer) -> None:
    assert isinstance(pricer, OptionPricerBase)


def test_forward_shape(weights: DIVFMWeights) -> None:
    moneyness_ttm, ttm, _ = _synthetic_options()
    F = weights.forward(moneyness_ttm, ttm)
    assert F.shape == (N, NUM_FACTORS)


def test_first_factor_is_constant_one(weights: DIVFMWeights) -> None:
    moneyness_ttm, ttm, _ = _synthetic_options()
    F = weights.forward(moneyness_ttm, ttm)
    np.testing.assert_array_equal(F[:, 0], np.ones(N))


def test_calibrate_constant_iv(pricer: DIVFMPricer) -> None:
    target_iv = 0.3
    moneyness_ttm, ttm, implied_vols = _synthetic_options(iv=target_iv)
    pricer.calibrate(moneyness_ttm, ttm, implied_vols)

    # With zero-weight subnets, all non-constant factors collapse to 0,
    # so the model reduces to sigma = beta_1 * 1 = mean(IV)
    assert pricer.betas[0] == pytest.approx(target_iv, rel=1e-4)
    np.testing.assert_array_almost_equal(pricer.betas[1:], 0.0)


def test_maturity_after_calibrate(weights: DIVFMWeights) -> None:
    # Use a tighter moneyness range so Black-Scholes IV inversion is reliable
    pricer = DIVFMPricer(weights=weights, max_moneyness_ttm=1.5, n=50)
    moneyness_ttm, ttm, implied_vols = _synthetic_options(iv=0.3)
    pricer.calibrate(moneyness_ttm, ttm, implied_vols)

    mp = pricer.maturity(0.5)
    assert mp.ttm == pytest.approx(0.5, rel=1e-3)
    assert len(mp.moneyness) == pricer.n
    assert len(mp.call) == pricer.n
    # call prices must be non-negative
    assert np.all(mp.call >= 0.0)
    # implied vols should be close to the calibrated constant in the central region
    central = np.abs(mp.moneyness) < 0.5
    assert np.allclose(mp.implied_vols[central], 0.3, atol=1e-3)


def test_maturity_cache(pricer: DIVFMPricer) -> None:
    moneyness_ttm, ttm, implied_vols = _synthetic_options()
    pricer.calibrate(moneyness_ttm, ttm, implied_vols)

    mp1 = pricer.maturity(0.25)
    mp2 = pricer.maturity(0.25)
    assert mp1 is mp2  # same object from cache


def test_calibrate_with_extra(weights: DIVFMWeights) -> None:
    """Extra features are stored and propagated to the pricing grid."""
    extra_features = 1
    subnet_ttm = _make_subnet(1 + extra_features, 1)
    subnet_joint = _make_subnet(2 + extra_features, NUM_FACTORS - 3)
    w = DIVFMWeights(
        subnet_ttm=subnet_ttm,
        subnet_moneyness=_make_subnet(1, 1),
        subnet_joint=subnet_joint,
        num_factors=NUM_FACTORS,
        extra_features=extra_features,
    )
    pricer = DIVFMPricer(weights=w, max_moneyness_ttm=1.5, n=20)
    moneyness_ttm, ttm, implied_vols = _synthetic_options()
    extra = np.zeros((N, extra_features), dtype=np.float32)

    pricer.calibrate(moneyness_ttm, ttm, implied_vols, extra=extra)

    assert pricer.extra is not None
    assert pricer.extra.shape == (1, extra_features)

    # Maturity must compute without error when extra is set
    mp = pricer.maturity(0.5)
    assert len(mp.call) == pricer.n


def test_reset_clears_cache(pricer: DIVFMPricer) -> None:
    moneyness_ttm, ttm, implied_vols = _synthetic_options()
    pricer.calibrate(moneyness_ttm, ttm, implied_vols)

    pricer.maturity(0.25)
    assert len(pricer.ttm) == 1
    pricer.reset()
    assert len(pricer.ttm) == 0


# ---------------------------------------------------------------------------
# DIVFMNetwork tests (requires torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_network_default_construction() -> None:
    net = DIVFMNetwork()
    assert net.num_factors == 5
    assert net.extra_features == 0


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_network_forward_shape() -> None:
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    net.eval()
    M = torch.zeros(N)
    T = torch.ones(N) * 0.5
    out = net(M, T)
    assert out.shape == (N, NUM_FACTORS)
    assert (out[:, 0] == 1.0).all()  # f_1 = 1


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_to_weights_forward_matches_network() -> None:
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    net.eval()

    rng = np.random.default_rng(0)
    M_np = rng.uniform(-2, 2, N).astype(np.float32)
    T_np = rng.uniform(0.1, 2.0, N).astype(np.float32)

    with torch.no_grad():
        torch_out = net(torch.tensor(M_np), torch.tensor(T_np)).numpy()

    w = net.to_weights()
    numpy_out = w.forward(M_np, T_np)

    np.testing.assert_allclose(torch_out, numpy_out, atol=1e-5)


# ---------------------------------------------------------------------------
# DIVFMTrainer tests (requires torch)
# ---------------------------------------------------------------------------


def _make_days(num_days: int = 20, iv: float = 0.3) -> list[DayData]:
    """Synthetic training set with a constant IV surface."""
    rng = np.random.default_rng(7)
    days = []
    for _ in range(num_days):
        n = rng.integers(20, 60)
        days.append(
            DayData(
                moneyness_ttm=rng.uniform(-2.0, 2.0, n).astype(np.float32),
                ttm=rng.uniform(0.1, 2.0, n).astype(np.float32),
                implied_vols=np.full(n, iv, dtype=np.float64),
            )
        )
    return days


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_trainer_construction() -> None:
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    trainer = DIVFMTrainer(net, lr=1e-3, batch_days=8)
    assert trainer.batch_days == 8
    assert trainer.network is net


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_trainer_step_returns_loss() -> None:
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    trainer = DIVFMTrainer(net, batch_days=4)
    days = _make_days(num_days=10)
    loss = trainer.step(days)
    assert isinstance(loss, float)
    assert loss >= 0.0


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_trainer_evaluate() -> None:
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    trainer = DIVFMTrainer(net)
    days = _make_days(num_days=5)
    val_loss = trainer.evaluate(days)
    assert isinstance(val_loss, float)
    assert val_loss >= 0.0


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_trainer_fit_loss_decreases() -> None:
    """Loss should decrease over training steps on a simple constant-IV surface."""
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    trainer = DIVFMTrainer(net, lr=1e-2, batch_days=8)
    days = _make_days(num_days=30)
    losses = trainer.fit(days, num_steps=50, log_every=0)
    assert len(losses) == 50
    # Average loss over the last 10 steps should be lower than the first 10
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_trainer_to_weights_produces_pricer() -> None:
    net = DIVFMNetwork(num_factors=NUM_FACTORS, hidden_size=HIDDEN_SIZE)
    trainer = DIVFMTrainer(net, batch_days=4)
    days = _make_days(num_days=10)
    trainer.fit(days, num_steps=5, log_every=0)

    weights = trainer.to_weights()
    assert isinstance(weights, DIVFMWeights)

    pricer = DIVFMPricer(weights=weights, max_moneyness_ttm=1.5, n=20)
    day = days[0]
    pricer.calibrate(day.moneyness_ttm, day.ttm, day.implied_vols)
    mp = pricer.maturity(0.5)
    assert len(mp.call) == pricer.n
