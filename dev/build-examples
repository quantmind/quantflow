#!/usr/bin/env python
"""Run all example scripts in docs/examples/ and capture their stdout to .out files."""
import sys

from docs.examples._utils import build_examples

failed = build_examples()

if failed:
    sys.exit(1)
