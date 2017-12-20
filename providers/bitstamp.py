import copy
import hmac
import time

from pulsar import async
from pulsar.apps import http

from pusher import Pusher

from quantflow.data import DataProvider, models


class BitstampWs(object):
    rest = 'https://www.bitstamp.net/api/'
    key = 'de504dc5763aeef9ff52'
    beat = 2

    def __init__(self):
        self.pusher = Pusher(key=self.key)
        self.http = http.HttpClient()

    def live_trades(self, callback):
        return self.subscribe('live_trades', callback, 'trade')

    def order_book(self, callback):
        return self.subscribe('order_book', callback, 'data')

    def diff_order_book(self, callback):
        return self.subscribe('diff_order_book', callback, 'data')

    def subscribe(self, channel, callback, event):
        '''Subscribe to a ``channel`` and listen for ``event``
        '''
        ch = yield from self.pusher.subscribe(channel)
        ch.bind(event, callback)
        return ch


class BitstampRest(object):
    """Rest Clinet for Bitstamp.

    API description at https://www.bitstamp.net/api/
    """
    endpoint = 'https://www.bitstamp.net/api'

    def __init__(self, client_id, api_key, api_secret):
        self.client_id = client_id
        self.key = api_key
        self.secret = api_secret
        self.client = http.HttpClient()
        self.nonce = int(1000000000000000*time.monotonic())

    def request_unauth(self, url):
        """Perform unauthorized request."""
        response = yield from self.client.get(self.endpoint + url)
        response.raise_for_status()
        return response.json()

    def request(self, url, params_dict={}):
        """Performs request with authorization."""
        self.nonce += 1
        payload = (str(self.nonce) + self.client_id + self.key).encode('utf-8')
        hmac_signer = hmac.new(bytes(self.secret, encoding='ascii'),
                               msg=payload, digestmod='SHA256')
        signature = hmac_signer.hexdigest().upper()

        payload_dict = copy.deepcopy(params_dict)
        payload_dict.update({
            'key': self.key,
            'signature': signature,
            'nonce': self.nonce
            })

        response = yield from self.client.post(
            self.endpoint + url,
            data=payload_dict)
        response.raise_for_status()
        return response.json()

    def order_book(self):
        """Returns JSON dictionary with "bids" and "asks".

        Each is a list of open orders and each order is represented as
        a list of price and amount.
        """
        return self.request_unauth('/order_book/')

    def transactions(self, time_frame='hour'):
        """Returns descending JSON list of transactions."""

        return self.request_unauth('/transactions/?time={}'.format(time_frame))

    def user_transactions(self, offset=0, limit=100, sort='desc'):
        """Returns list of transactions.

        Parameters
        ----------
        offset : skip that many transactions before beginning to return
        results. Default: 0.

        limit : limit result to that many transactions. Default: 100. Maximum:
        1000.

        sort : sorting by date and time (asc - ascending; desc -
        descending). Default: desc.
        """
        assert(offset >= 0)
        assert(limit >= 0 and limit <= 1000)
        assert(sort in ('asc', 'desc'))

        params_dict = {
            'offset': offset,
            'limit': limit,
            'sort': sort
            }

        return self.request('/user_transactions/', params_dict)

    def orders(self):
        """List of open orders."""

        return self.request('/open_orders/')

    def order_cancel(self, order_id):
        """Cancel order."""

        return self.request('/cancel_order/', {'id': order_id})

    def order_new(self, side, amount, price, limit_price):
        """Buy limit order"""

        assert(side in ['buy', 'sell'])

        params_dict = {
            'amount': amount,
            'price': price,
            'limit_price': limit_price
            }

        return self.request('/{}/'.format(side), params_dict)


class Bitstamp(DataProvider):
    security = 'xbtusd'
    beat = 2

    def start(self):
        '''Start bitstamp data provider by connecting to bitstamp
        websocket API a subscribing for live trades and order book changes.
        '''
        self.ws = BitstampWs()
        self.rest = BitstampRest(self.config['client_id'],
                                 self.config['key'],
                                 self.config['secret'])
        return self._connect()

    def _connect(self):
        try:
            yield from self.rest.order_book()
        except OSError:
            self.logger.error('Could not connect, '
                              'retrying in %s seconds', self.beat)
            self._loop.call_later(self.beat, async, self._connect())
        else:
            # Register to bitstamp websocket events
            yield from self.ws.live_trades(self._live_trades)
            yield from self.ws.diff_order_book(self._diff_order_book)
            yield from self.ws.order_book(self._order_book)

    def _live_trades(self, trade, **kwargs):
        self.logger.debug('New XBTUSD trade')
        trade['volume'] = trade.pop('amount')
        trade['trade_id'] = trade.pop('id')
        book = self.book(self.security)
        book.record(update_type=models.TRADE, **trade)
        book.flush()
        self.publish('trade', {'exchange': self.name,
                               'security': self.security,
                               'trade': trade})

    def _order_book(self, book, **kwargs):
        self.logger.debug('New XBTUSD orders')
        self.publish('order_book', {'exchange': self.name,
                                    'security': self.security,
                                    'book': book})

    def _diff_order_book(self, data, **kwargs):
        if data['bids'] or data['asks']:
            book = self.book(self.security)
            for price, volume in data['bids']:
                book.record(update_type=models.ORDER,
                            side=models.BIDBUY,
                            price=float(price),
                            volume=float(volume))
            for price, volume in data['asks']:
                book.record(update_type=models.ORDER,
                            side=models.ASKSELL,
                            price=float(price),
                            volume=float(volume))
            book.flush()
