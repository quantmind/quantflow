"""
Adapted from humps

https://github.com/nficano/humps
"""

import re

ACRONYM_RE = re.compile(r"([A-Z\d]+)(?=[A-Z\d]|$)")
SPLIT_RE = re.compile(r"([\-_]*(?<=[^0-9])(?=[A-Z])[^A-Z]*[\-_]*)")


def snake_case(string: str) -> str:
    """Convert a string into snake case."""
    return _separate_words(_fix_abbreviations(string)).lower()


def _fix_abbreviations(string: str) -> str:
    return ACRONYM_RE.sub(lambda m: m.group(0).title(), string)


def _separate_words(string: str, separator: str = "_") -> str:
    return separator.join(s for s in SPLIT_RE.split(string) if s)
