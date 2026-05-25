Sample from a Vasicek mean-reverting Gaussian process and compare the empirical
distribution to the analytical marginal PDF.

The process is run for one unit of time with 1000 time steps. The terminal
distribution is binned into 50 bins and compared against the analytical Gaussian PDF.

A Kolmogorov-Smirnov test is included to quantify goodness of fit.
A high p-value (e.g. above 0.05) means the simulation is statistically consistent
with the analytical distribution.

Antithetic variates halve the variance of Monte Carlo estimates by pairing each
draw with its negative, without additional model evaluations.
