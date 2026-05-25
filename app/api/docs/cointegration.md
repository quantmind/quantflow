Johansen cointegration analysis of BTC, ETH, and SOL log prices.

Returns the mean-reverting spread (residuals) and the cointegrating vector
(eigenvector for the largest eigenvalue of the Johansen test).

Log prices are standardised before the test; the returned cointegrating vector
is rescaled back to the original units and normalised to unit length.
A residual near zero means the spread is close to its long-run equilibrium.
Large positive or negative residuals signal potential mean-reversion trades.
