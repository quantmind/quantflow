try:
    from .app import App
except ImportError:
    raise ImportError(
        "Cannot run qf command line, " "quantflow needs to be installed with cli extras"
    ) from None

main = App()
