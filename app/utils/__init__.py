import marimo as mo
import pandas as pd

pd.options.plotting.backend = "plotly"

def nav_menu():
    return mo.nav_menu(
        {
            "/": "Quantflow",
            "/api": "API Reference",
        }
    )
