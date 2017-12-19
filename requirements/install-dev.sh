#!/usr/bin/env bash

set -e -x

pip install -U pip wheel
pip install -U setuptools
pip install -U -r requirements/hard.txt
pip install -U -r requirements/dev.txt
