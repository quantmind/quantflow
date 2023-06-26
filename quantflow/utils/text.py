from typing import Callable, Optional

try:
    from IPython.display import Markdown
except ImportError:
    Markdown = None  # type: ignore


def shift(text: str, n: int) -> str:
    space = " " * max(n, 0)
    return "\n".join(f"{space}{t}" for t in text.split("\n"))


def doc(method: Callable) -> str:
    """
    Return the docstring of a method.
    """
    if Markdown is None:
        raise RuntimeError("IPython is required to display markdown docs")
    return trim_docstring(method.__doc__)


def trim_docstring(docstring: Optional[str] = None) -> str:
    """Uniformly trims leading/trailing whitespace from docstrings.
    Based on
    http://www.python.org/peps/pep-0257.html#handling-docstring-indentation
    """
    if not docstring or not docstring.strip():
        return ""
    # Convert tabs to spaces and split into lines
    lines = docstring.expandtabs().splitlines()
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.lstrip())
    trimmed = [lines[0].lstrip()] + [line[indent:].rstrip() for line in lines[1:]]
    return "\n".join(trimmed).strip()
