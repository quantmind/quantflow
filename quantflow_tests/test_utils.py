from quantflow.utils.numbers import round_to_step, to_decimal


def test_round_to_step():
    assert str(round_to_step(1.234, 0.1)) == "1.2"
    assert str(round_to_step(1.234, 0.01)) == "1.23"
    assert str(round_to_step(1.236, 0.01)) == "1.24"
    assert str(round_to_step(1.1, 0.01)) == "1.10"
    assert str(round_to_step(1.1, 0.001)) == "1.100"
    assert str(round_to_step(2, 0.001)) == "2.000"
    assert str(round_to_step(to_decimal("2.00000000000"), 0.001)) == "2.000"
