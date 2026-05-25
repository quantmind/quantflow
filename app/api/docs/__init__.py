from pathlib import Path

_DOCS_PATH = Path(__file__).parent


def load_description(filename: str) -> str:
    return (_DOCS_PATH / filename).read_text(encoding="utf-8")
