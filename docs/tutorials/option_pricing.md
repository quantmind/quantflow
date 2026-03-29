# Option Pricing

This tutorial shows how to price a single European option using the
[OptionPricer][quantflow.options.pricer.OptionPricer].

The underlying model is [HestonJ][quantflow.sp.heston.HestonJ], a Heston stochastic
volatility model extended with jumps drawn from a
[DoubleExponential][quantflow.utils.distributions.DoubleExponential] distribution.

The result is a [ModelOptionPrice][quantflow.options.pricer.ModelOptionPrice] containing the
price, delta, and gamma in forward space as well as the implied Black volatility and its sensitivities delta, gamma, vega, etc.

```python
--8<-- "docs/examples/heston_volatility_pricer.py"
```

```json
--8<-- "docs/examples_output/heston_volatility_pricer.out"
```
