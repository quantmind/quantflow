# Option Pricing

This tutorial shows how to price a single European option using the
[OptionPricer][quantflow.options.pricer.OptionPricer].

When calculating the option price from a option pricer, the result is a [ModelOptionPrice][quantflow.options.pricer.ModelOptionPrice] containing the
price, delta, and gamma in forward space as well as the implied Black volatility and its sensitivities delta, gamma, vega, etc.

Model sensitivities such as delta and gamma are calculated using the model dynamics, and will differ from the Black-Scholes deltas and gammas. The implied Black vol is the volatility that, when plugged into the Black formula, gives the same price as the model price. The Black deltas and gammas are calculated by plugging the implied vol into the Black formula.

## Validation with Black-Scholes

The first example shows how to price an option using the Black-Scholes model and validate the results against the analytical Black formula. The implied volatility should be the same as the model volatility, and the deltas and gammas should be the same as well (within numerical precision).

```python
--8<-- "docs/examples/weiner_volatility_pricer.py"
```

```json
--8<-- "docs/examples/output/weiner_volatility_pricer.out"
```


## Heston with Jumps

The underlying model is [HestonJ][quantflow.sp.heston.HestonJ], a Heston stochastic
volatility model extended with jumps drawn from a
[DoubleExponential][quantflow.utils.distributions.DoubleExponential] distribution.

```python
--8<-- "docs/examples/heston_volatility_pricer.py"
```

```json
--8<-- "docs/examples/output/heston_volatility_pricer.out"
```
