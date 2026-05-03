# Pricing Method Comparison

This tutorial compares the three Fourier-based pricing methods available in
[OptionPricer][quantflow.options.pricer.OptionPricer]:
[Carr-Madan](../bibliography.md#carr_madan),
[Lewis](../bibliography.md#lewis), and
[COS](../bibliography.md#cos).
All three invert the characteristic function of the log-return to price European calls,
but they differ in how they handle the payoff transform, the integration contour, and
the discretisation strategy. See [Option Pricing](../theory/option_pricing.md) for the
full derivation of all three methods.

## Selecting the method

Pass `method` when constructing the pricer:

```python
from quantflow.options.pricer import OptionPricer, OptionPricingMethod
from quantflow.sp.heston import Heston

model = Heston.create(vol=0.2)
pricer = OptionPricer(model=model, method=OptionPricingMethod.COS)
result = pricer.maturity(1.0)
```

## Accuracy comparison

The charts below use a Heston model with $\sigma=0.8$, $\kappa=2$, $\rho=-0.2$,
$\text{vol}=0.5$. The reference solution is Lewis with $N=8192$.
Implied vol errors are clipped at 10% — Lewis can produce errors well above this
at low $N$, particularly at short maturities, which would otherwise dominate the scale.

At long maturities (TTM=1.0) all three methods converge quickly:

[![Accuracy TTM=1.0](../assets/examples/pricing_method_accuracy_ttm1_0.png)](../assets/examples/pricing_method_accuracy_ttm1_0.png){target="_blank"}

At medium maturities (TTM=0.25) COS tends to converge faster than Carr-Madan for the
same $N$:

[![Accuracy TTM=0.25](../assets/examples/pricing_method_accuracy_ttm0_25.png)](../assets/examples/pricing_method_accuracy_ttm0_25.png){target="_blank"}

At very short maturities (TTM=0.02) the differences are most pronounced. Carr-Madan
can struggle with the auto-selected $\alpha$, while Lewis and COS remain stable:

[![Accuracy TTM=0.02](../assets/examples/pricing_method_accuracy_ttm0_02.png)](../assets/examples/pricing_method_accuracy_ttm0_02.png){target="_blank"}

## Complexity

| Method | Complexity |
|---|---|
| Carr-Madan | $O(N \log N)$ via Fractional Fourier Transform |
| Lewis | $O(N \log N)$ via Fractional Fourier Transform |
| COS | $O(N^2)$ |

Unlike the FRFT methods, COS can price a single strike at a time without computing the
full grid, making it well suited for lazy or on-demand strike evaluation.

## Code for the above charts

```python
--8<-- "docs/examples/pricing_method_comparison.py"
```
