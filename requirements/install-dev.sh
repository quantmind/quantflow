#!/usr/bin/env bash

set -e -x

pip install -U pip
pip install -U wheel setuptools
pip install -U -r requirements/hard.txt
pip install -U -r requirements/dev.txt
pip install -U git+git://github.com/quantmind/pulsar.git
