from pulsar.apps import Application
from pulsar.apps.http import HttpClient
from pulsar.apps.ws import WS


class BitCoin(object):
    root_url = 'https://www.bitstamp.net/api/'

    def __init__(self, http):
        self.http = http

    @property
    def _loop(self):
        return self.http._loop

    def ticker(self):
        return self.http.get(self.url('ticker/'))

    @in_loop
    def mtgox(self):
        res =  yield self.http.get('ws://ws.blockchain.info/inv',
                                   websocket_handler=MtGox())
        a = res.get_content()
        b = 1


