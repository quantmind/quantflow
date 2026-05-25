Sample from a Poisson jump process and compare the empirical jump count distribution
to the analytical Poisson PMF.

The process is run for one unit of time with 1000 time steps. Jump counts are
binned and compared against the analytical distribution using a chi-squared
goodness-of-fit test. Tail bins with expected count below 5 are merged to satisfy
the chi-squared validity requirement.

A high p-value (e.g. above 0.05) means the simulation is statistically consistent
with the analytical Poisson distribution.
