from html import escape
from pathlib import Path
from typing import Any

APP_PATH = Path(__file__).parent.parent


def head_snippet(path: Path) -> str:
    return (path / "assets" / "logos" / "head-snippet.html").read_text()


def social_meta(page: Any, config: Any) -> str:
    site_name = escape(str(config.get("site_name", "")))
    site_description = escape(str(config.get("site_description", "")))
    site_url = str(config.get("site_url", "")).rstrip("/")
    title = escape(f"{page.title} | {site_name}" if page.title else site_name)
    page_url = escape(str(page.canonical_url))
    image = escape(f"{site_url}/assets/logos/png/quantflow-github-social-1280x640.png")
    image_alt = "QuantFlow logo and tagline"
    return "\n".join(
        (
            '<meta property="og:type" content="website">',
            f'<meta property="og:site_name" content="{site_name}">',
            f'<meta property="og:title" content="{title}">',
            f'<meta property="og:description" content="{site_description}">',
            f'<meta property="og:url" content="{page_url}">',
            f'<meta property="og:image" content="{image}">',
            f'<meta property="og:image:secure_url" content="{image}">',
            '<meta property="og:image:type" content="image/png">',
            '<meta property="og:image:width" content="1280">',
            '<meta property="og:image:height" content="640">',
            f'<meta property="og:image:alt" content="{image_alt}">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{title}">',
            f'<meta name="twitter:description" content="{site_description}">',
            f'<meta name="twitter:image" content="{image}">',
            f'<meta name="twitter:image:alt" content="{image_alt}">',
        )
    )


def on_post_page(output: str, page: Any, config: Any) -> str:
    """Hook to inject custom HTML into the head of each page in mkdocs build."""
    snippet = "\n".join(
        (
            head_snippet(APP_PATH.parent / "docs"),
            social_meta(page, config),
        )
    )
    return output.replace("</head>", f"{snippet}</head>", 1)
