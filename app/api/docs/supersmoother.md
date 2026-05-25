Apply SuperSmoother and EWMA filters to two years of historical daily price data
for a given symbol.

SuperSmoother is an adaptive low-pass filter designed by John Ehlers. It suppresses
high-frequency noise more aggressively than EWMA while introducing less lag at
trend inflections.

Both filters use the same `period` parameter, making it easy to compare their
smoothing characteristics on the same data. Shorter periods follow price more
closely; longer periods produce smoother output with more lag.

Supported symbols include crypto pairs (e.g. BTCUSD, ETHUSD) and US equities.
