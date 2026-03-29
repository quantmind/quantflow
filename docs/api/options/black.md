# Black Pricing

Here we define the log strike `k` as
$$
  k = \log{\frac{K}{F}}
$$

where $K$ is the strike price and $F$ is the forward price of the underlying asset.
We also refers to this log-strike as `moneyness`, since it is zero for at-the-money (ATM) options,
negative for in-the-money (ITM) call options, and positive for out-of-the-money (OTM) call options.


::: quantflow.options.bs.black_price

::: quantflow.options.bs.black_call

::: quantflow.options.bs.black_delta

::: quantflow.options.bs.black_vega

::: quantflow.options.bs.BlackSensitivities

::: quantflow.options.bs.implied_black_volatility

::: quantflow.options.bs.ImpliedVols

::: quantflow.options.bs.ImpliedVol
