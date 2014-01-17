import os
import sys

try:
    import pulsar
except ImportError:
    path = os.path.join(os.path.dirname(os.getcwd()), 'pulsar')
    if os.path.isdir(path):
        sys.path.append(path)

from qa.data.bitcoin import Robot

if __name__ == '__main__':  #pragma    nocover
    Robot().start()
