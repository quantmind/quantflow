from pathlib import Path


APP_PATH = Path(__file__).parent.parent
DOCS_PATH = APP_PATH.parent / "docs"


def head_snippet() -> str:
    return (DOCS_PATH / "assets" / "logos" / "head-snippet.html").read_text()


def on_post_page(output: str, page, config) -> str:
    """Hook to inject custom HTML into the head of each page in mkdocs build."""
    snippet = head_snippet()
    return output.replace("</head>", f"{snippet}</head>", 1)
