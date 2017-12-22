import re
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd

from pulsar.apps.http import HttpClient

from .share import Share
from .currency import Currency



class Yahoo:
    """Yahoo Finance client
    Build with Yahoo Query Language - YQL

    https://developer.yahoo.com/yql/

    and yahoo Finance API
    """

    def __init__(self, http=None):
        self.http = http or HttpClient()
        self._symbols = {}

    def symbol(self, symbol):
        """"Share price information
        """
        symbol = symbol.upper()
        if symbol not in self._symbols:
            self._symbols[symbol] = Share(self, symbol)
        return self._symbols[symbol]
