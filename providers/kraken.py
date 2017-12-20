from urllib.parse import urlencode
from asyncio import async

from pulsar.apps import http

from quantflow.data import DataProvider


class ApiResponseError(Exception):
    pass


class KrakenClient(object):
    """REST API python client.

    See more at https://www.bitfinex.com/pages/api
    """
    API_HOST = 'https://api.kraken.com/'
    API_VERSION = '0'

    def __init__(self, api_key, api_secret):
        self.key = api_key
        self.secret = api_secret
        self.client = http.HttpClient()

    @property
    def _loop(self):
        return self.client._loop

    def assets(self):
        url = self._public('Assets')
        response = yield from self.client.get(url)
        return self._raise_error(response)

    def order_book(self, symbol, limit=None):
        url = self._public('Depth', pair=symbol, count=limit)
        response = yield from self.client.get(url)
        result = self._raise_error(response)
        return result[symbol]

    def tiker_info(self, symbol):
        url = self._public('Ticker', pair=symbol)
        response = yield from self.client.get(url)
        result = self._raise_error(response)
        return result[symbol]

    def _public(self, name, **params):
        base = '%s%s/public/%s' % (self.API_HOST, self.API_VERSION, name)
        return self._url(base, **params)

    def _private(self, name, **params):
        base = '%s%s/private/%s' % (self.API_HOST, self.API_VERSION, name)
        return self._url(base, **params)

    def _url(self, base, **params):
        query = urlencode([(key, value) for key, value in params.items()
                           if value is not None])
        return '%s?%s' % (base, query)

    def _raise_error(self, response):
        response.raise_for_status()
        data = response.json()
        error = data.get('error')
        if error:
            raise ApiResponseError('. '.join(error))
        return data.get('result')


class Kraken(DataProvider):

    def start(self):
        self.rest = KrakenClient()
        self._loop = self.rest._loop
        self.counter = 0
        self.symbols = ('XBTUSD', 'XBTEUR', 'XBTGBP')
        self._looping_call()

    def _looping_coro(self):
        if self.counter >= len(self.symbols):
            self.counter = 0
        pair = self.symbols[self.counter]
        self.counter += 1
        symbol = 'X%sZ%s' % (pair[:3], pair[3:])
        book = yield from self.rest.order_book(symbol)
        self.record('order_book', book)
        self._loop.call_later(2, self._looping_call)

    def _looping_call(self):
        async(self._looping_coro(), loop=self._loop)
