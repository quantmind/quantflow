from pathlib import Path

APP_PATH = Path(__file__).parent.parent


def head_snippet(path: Path) -> str:
    return (path / "assets" / "logos" / "head-snippet.html").read_text()


def on_post_page(output: str, page, config) -> str:
    """Hook to inject custom HTML into the head of each page in mkdocs build."""
    snippet = head_snippet(APP_PATH.parent / "docs")
    return output.replace("</head>", f"{snippet}</head>", 1)
