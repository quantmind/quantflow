#!/usr/bin/env python3
"""Convert docs/references.bib to docs/bibliography.md.

Usage:
    uv run python docs/bib2md.py [--bib PATH] [--out PATH]

The BibTeX file is the source of truth; run this script whenever it changes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"@(\w+)\s*\{\s*([\w][\w:/-]*)\s*,\s*(.*?)\n\}",
    re.DOTALL,
)
_FIELD_RE = re.compile(
    r"""(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)")""",
    re.DOTALL,
)


def _strip_braces(text: str) -> str:
    """Remove single-level LaTeX capitalisation braces, e.g. {Lévy} -> Lévy."""
    return re.sub(r"\{([^{}]*)\}", r"\1", text)


def _clean(value: str) -> str:
    text = _strip_braces(" ".join(value.split()))
    # Convert LaTeX en-dash (--) to a plain hyphen
    return text.replace("--", "-")


def parse_bib(text: str) -> list[dict[str, str]]:
    entries = []
    for m in _ENTRY_RE.finditer(text):
        entry: dict[str, str] = {
            "type": m.group(1).lower(),
            "key": m.group(2),
        }
        for fm in _FIELD_RE.finditer(m.group(3)):
            raw = fm.group(2) if fm.group(2) is not None else fm.group(3)
            entry[fm.group(1).lower()] = _clean(raw)
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Author formatting
# ---------------------------------------------------------------------------

def _format_single_author(author: str) -> str:
    """Convert 'Lastname, Firstname' to 'Firstname Lastname'; leave others as-is."""
    author = author.strip()
    # Only reformat if a comma is clearly separating last from first name
    # (avoid splitting on commas inside name suffixes or freeform strings)
    parts = author.split(",", 1)
    if len(parts) == 2:
        last, first = parts[0].strip(), parts[1].strip()
        # Sanity: last name should be a single word (no spaces)
        if first and " " not in last:
            return f"{first} {last}"
    return author


def format_authors(author_field: str) -> str:
    # Normalise separators: "and" and "&" both split authors
    normalised = re.sub(r"\s+and\s+", " & ", author_field, flags=re.IGNORECASE)
    authors = [a.strip() for a in normalised.split("&") if a.strip()]
    return ", ".join(_format_single_author(a) for a in authors)


# ---------------------------------------------------------------------------
# Entry formatter
# ---------------------------------------------------------------------------

def _title_link(title: str, url: str) -> str:
    if not url:
        return title
    return f'[{title}]({url}){{target="_blank" rel="noopener"}}'


def _journal_detail(entry: dict[str, str]) -> str:
    """Build 'Journal, Vol(Num):Pages' string."""
    journal = entry.get("journal", "")
    volume = entry.get("volume", "")
    number = entry.get("number", "")
    pages = entry.get("pages", "")
    if not journal:
        return ""
    detail = journal
    if volume:
        vol_str = volume
        if number:
            vol_str += f"({number})"
        if pages:
            vol_str += f":{pages}"
        detail += f", {vol_str}"
    elif pages:
        detail += f", {pages}"
    return detail


def format_entry(entry: dict[str, str]) -> str:
    key = entry["key"]
    title = entry.get("title", "Unknown Title")
    year = entry.get("year", "")
    url = entry.get("url", "")
    author = entry.get("author", "")

    # Prefer doi as url when no url field present
    if not url and "doi" in entry:
        doi = entry["doi"].lstrip("doi:")
        url = f"https://doi.org/{doi}"

    author_str = format_authors(author) if author else ""
    title_md = _title_link(title, url)
    year_str = f"({year})" if year else ""

    # Prefix: "Author(s). (Year)"
    prefix_parts = []
    if author_str:
        prefix_parts.append(author_str.rstrip(".") + ".")
    if year_str:
        prefix_parts.append(year_str)
    prefix = " ".join(prefix_parts)

    # Suffix depends on entry type
    entry_type = entry.get("type", "article")
    if entry_type == "article":
        suffix = _journal_detail(entry)
    elif entry_type == "book":
        suffix = entry.get("publisher", "")
    elif entry_type in ("mastersthesis", "phdthesis"):
        suffix = entry.get("school", "")
    else:
        suffix = entry.get("publisher", entry.get("school", ""))

    body = f"{prefix} {title_md}" if prefix else title_md
    if suffix:
        body = f"{body}, {suffix}"

    return f"#### {key}\n\n{body}\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(bib_path: Path, out_path: Path) -> None:
    text = bib_path.read_text(encoding="utf-8")
    entries = parse_bib(text)
    if not entries:
        print(f"No entries found in {bib_path}", file=sys.stderr)
        sys.exit(1)

    entries.sort(key=lambda e: e["key"].lower())

    lines = ["# Bibliography\n", "\n---\n"]
    for entry in entries:
        lines.append("\n")
        lines.append(format_entry(entry))

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {out_path}")


def main() -> None:
    docs = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bib",
        type=Path,
        default=docs / "references.bib",
        help="Input BibTeX file (default: docs/references.bib)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=docs / "bibliography.md",
        help="Output Markdown file (default: docs/bibliography.md)",
    )
    args = parser.parse_args()
    convert(args.bib, args.out)


if __name__ == "__main__":
    main()
