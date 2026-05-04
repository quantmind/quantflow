import numpy as np
from quantflow.ta.supersmoother import SuperSmoother
from quantflow.ta.ewma import EWMA

# Create test data with noise
np.random.seed(42)
values = np.sin(np.linspace(0, 4 * np.pi, 50)) + np.random.normal(0, 0.1, 50)

# Apply both filters with same period
period = 10
ss = SuperSmoother(period=period)
ewma = EWMA(period=period)

ss_smoothed = ss.smooth_array(values)
ewma_smoothed = ewma.smooth_array(values)

# Check if they're different
print(f"SuperSmoother: {ss_smoothed[:10]}")
print(f"EWMA:          {ewma_smoothed[:10]}")
print(f"\nAre they different? {not np.allclose(ss_smoothed, ewma_smoothed)}")
print(f"Max difference: {np.max(np.abs(ss_smoothed - ewma_smoothed)):.4f}")
print(f"Mean difference: {np.mean(np.abs(ss_smoothed - ewma_smoothed)):.4f}")
