import dotenv

dotenv.load_dotenv()

try:
    from .app import QfApp
except ImportError as ex:
    raise ImportError(
        "Cannot run qf command line, "
        "quantflow needs to be installed with cli & data extras, "
        "pip install quantflow[cli, data]"
    ) from ex

main = QfApp()
