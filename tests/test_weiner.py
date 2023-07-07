import pytest

from quantflow.sp.weiner import WeinerProcess


@pytest.fixture
def weiner() -> WeinerProcess:
    return WeinerProcess(sigma=0.5)


def test_characteristic(weiner: WeinerProcess) -> None:
    assert weiner.characteristic(1, 0) == 1
    marginal = weiner.marginal(1, 0)
    assert marginal.mean() == 0
    assert marginal.mean_from_characteristic() == 0
    assert marginal.std() == 0.5
    assert marginal.variance_from_characteristic() == pytest.approx(0.25)
