from pulsar.apps.http import HttpClient

from ..pusher import Pusher


class BitstampWs:
    rest = 'https://www.bitstamp.net/api/'
    key = 'de504dc5763aeef9ff52'
    beat = 2

    def __init__(self, http=None):
        self.pusher = Pusher(key=self.key, http=http)

    @property
    def http(self):
        return self.pusher.http

    def live_trades(self, callback):
        return self.subscribe('live_trades', callback, 'trade')

    def order_book(self, callback):
        return self.subscribe('order_book', callback, 'data')

    def diff_order_book(self, callback):
        return self.subscribe('diff_order_book', callback, 'data')

    async def subscribe(self, channel, callback, event):
        '''Subscribe to a ``channel`` and listen for ``event``
        '''
        ch = await self.pusher.subscribe(channel)
        ch.bind(event, callback)
        return ch
