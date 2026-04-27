import json

import pandas as pd

from quantflow.options.inputs import OptionInput
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs

# Load a saved volatility surface snapshot from JSON
with open("docs/examples/volsurface.json") as fp:
    surface_inputs = VolSurfaceInputs(**json.load(fp))

# Build the VolSurface from the inputs and calculate implied volatilities
surface: VolSurface = surface_from_inputs(surface_inputs)
surface.bs()
surface.disable_outliers()

# Print the term structure (forward prices and implied rates per maturity)
print(surface.term_structure().to_string(index=False))

# Display the surface inputs for converged options only
inputs = surface.inputs(converged=True)
option_inputs = [i for i in inputs.inputs if isinstance(i, OptionInput)]
df = pd.DataFrame([i.model_dump() for i in option_inputs])
print(
    df[["maturity", "strike", "option_type", "bid", "ask", "iv_bid", "iv_ask"]]
    .head(10)
    .to_string(index=False)
)
