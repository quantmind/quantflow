#!/usr/bin/env python
import sys
import os
from multiprocessing import current_process

try:
    from pulsar.utils.path import Path
except ImportError:
    # pulsar not available, we are in dev
    path = os.path.join(os.path.dirname(os.getcwd()), 'pulsar')
    if os.path.isdir(path):
        sys.path.append(path)

from pulsar.apps.test import TestSuite
from pulsar.apps.test.plugins import bench, profile


def run(**params):
    args = params.get('argv', sys.argv)
    if '--coverage' in args or params.get('coverage'):
        import coverage
        p = current_process()
        p._coverage = coverage.coverage(data_suffix=True)
        p._coverage.start()
    runtests(**params)


def runtests(**params):
    suite = TestSuite(modules=['tests'],
                      plugins=(bench.BenchMark(), profile.Profile()),
                      **params).start()


if __name__ == '__main__':
    run()
