"""Bitfinex integration"""
import base64
import copy
import json
import hmac
import time
from datetime import date
from asyncio import async

from pulsar.apps import http
from pulsar.utils.httpurl import Headers

from quantflow.data import DataProvider, models

LIMIT = 10000


def date2timestamp(dte):
    return time.mktime(dte.timetuple())


class BitfinexRest(object):
    """REST API python client.

    See more at https://www.bitfinex.com/pages/api
    """
    API_HOST = 'https://api.bitfinex.com'
    API_PREFIX = '/v1'
    API_URL = API_HOST + API_PREFIX

    ORDER_TYPES = ['LIMIT', 'MARKET', 'STOP', 'TRAILING STOP', 'FILL-OR-KILL',
                   'EXCHANGE LIMIT', 'EXCHANGE MARKET', 'EXCHANGE STOP',
                   'EXCHANGE TRAILING STOP', 'EXCHANGE FILL-OR-KILL']

    def __init__(self, api_key, api_secret):
        self.key = api_key
        self.secret = api_secret
        self.client = http.HttpClient()
        # The nonce must be strictly increasing.
        self.nonce = int(100000000000000*time.monotonic())

    @property
    def _loop(self):
        return self.client._loop

    def request(self, url, params_dict={}):
        """Performs request with authorization."""
        self.nonce += 1
        payload_dict = copy.deepcopy(params_dict)
        payload_dict.update({
            'request': self.API_PREFIX + url,
            'nonce': str(self.nonce),
            })
        payload = base64.b64encode(json.dumps(payload_dict).encode('utf-8'))
        hmac_signer = hmac.new(bytes(self.secret, encoding='ascii'),
                               msg=payload, digestmod='SHA384')
        signature = str(hmac_signer.hexdigest())

        headers = Headers([
            ('X-BFX-APIKEY', self.key),
            ('X-BFX-PAYLOAD', payload.decode('ascii')),
            ('X-BFX-SIGNATURE', signature),
            ])
        response = yield from self.client.post(
            self.API_URL + url,
            headers=headers,
            data=payload_dict)
        response.raise_for_status()
        return response.json()

    def request_unauth(self, url):
        """Perform unauthorized request."""
        response = yield from self.client.get(self.API_URL + url)
        response.raise_for_status()
        return response.json()

    def pubticker(self, ticker):
        """Gives innermost bid and asks and information on the most recent trade.

        As well as high, low and volume of the last 24 hours.
        """
        return self.request_unauth('/pubticker/{}'.format(ticker))

    def lend_book(self, currency, limit_bids=50, limit_asks=50):
        """Get the full lend book.

        limit_bids : Optional. Limit the number of bids (loan demands)
        returned. May be 0 in which case the array of bids is empty

        limit_asks (int) : Optional. Limit the number of asks (loan offers)
        returned. May be 0 in which case the array of asks is empty
        """
        return self.request_unauth(
            '/lendbook/{}?limit_bids={}&limit_asks={}'.format(
                currency, limit_bids, limit_asks)
            )

    def order_book(self, symbol, limit_bids=50, limit_asks=50, group=True):
        """Get the full order book.

        limit_bids : Optional. Limit the number of bids (loan demands)
        returned. May be 0 in which case the array of bids is empty

        limit_asks (int) : Optional. Limit the number of asks (loan offers)
        returned. May be 0 in which case the array of asks is empty

        group : Optional. It True orders are grouped by price in the orderbook.
        """
        return self.request_unauth(
            '/book/{}?limit_bids={}&limit_asks={}&group={:d}'.format(
                symbol, limit_bids, limit_asks, group)
            )

    def trades(self, symbol, limit=50, since=None):
        """Get a list of the most recent trades for the given symbol.

        since : Unix time. Optional. Only show trades at or after this time.

        limit : Optional. Limit the number of trades returned.
        """
        assert(limit >= 1)
        request = '/trades/{}?limit_trades={}'.format(
            symbol, limit)
        if since is not None:
            request += '&timestamp={:d}'.format(since)
        return self.request_unauth(request)

    def lends(self, currency, limit=50, since=None):
        """Get a list of the most recent swaps data for the given currency.

        Total amount lent and Flash Return Rate (in % by 365 days) over time.

        since: Unix timestamp. Optional. Only show data at or after this
        timestamp.

        limit: Optional. Limit the number of swaps data returned.
        """
        assert(limit >= 1)
        request = '/lends/{}?limit_lends={}'.format(
            currency, limit)
        if since is not None:
            request += '&timestamp={:d}'.format(since)
        return self.request_unauth(request)

    def symbols(self):
        """Get a list of valid symbol IDs."""

        return self.request_unauth('/symbols/')

    def symbols_details(self):
        """Get a list of valid symbol IDs and the pair details."""

        return self.request_unauth('/symbols_details/')

    def balances(self):
        """See your balances."""

        return self.request('/balances')

    def orders(self):
        """View your active orders."""

        return self.request('/orders')

    def positions(self):
        """View your active positions."""

        return self.request('/positions')

    def offers(self):
        """View your active offers."""

        return self.request('/offers')

    def credits(self):
        """View your funds currently lent (active credits)."""

        return self.request('/credits')

    def taken_swaps(self):
        """An array of your active taken swaps."""

        return self.request('/taken_swaps')

    def account_infos(self):
        """Return information about your account (trading fees)."""

        return self.request('/account_infos')

    def margin_infos(self):
        """See your trading wallet information for margin trading."""

        return self.request('/margin_infos')

    def deposit_new(self, currency, method, wallet_name):
        """Return your deposit address to make a new deposit."""
        params = {
            'currency': currency,
            'method': method,
            'wallet_name': wallet_name,
        }
        return self.request('/deposit/new', params)

    def order_dict(self, side, amount, price, symbol, type, is_hidden=False,
                   exchange="bitfinex"):
        """Construct dictionary with order params."""

        assert(side in ['buy', 'sell'])
        assert(type.upper() in self.ORDER_TYPES)
        return {
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'side': side,
            'type': type,
            'exchange': exchange,
            'is_hidden': is_hidden,
        }

    def order_new(self, order_dict):
        """Submit a new order."""
        return self.request('/order/new', order_dict)

    def order_new_multi(self, order_dicts):
        """Submit several new orders at once."""
        return self.request('/order/new/multi', order_dicts)

    def order_cancel(self, order_id):
        """Cancel an order."""

        return self.request('/order/cancel', {'order_id': order_id})

    def order_cancel_multi(self, order_ids):
        """Cancel multiple orders ar once."""

        return self.request('/order/cancel/multi', order_ids)

    def order_cancel_all(self):
        """Cancel multiples orders at once."""

        return self.request('/order/cancel/all')

    def order_replace(self, order_id, new_order_dict):
        """Replace an order with a new one."""

        params = copy.deepcopy(new_order_dict)
        params['order_id'] = order_id
        return self.request('/order/cancel/replace', params)

    def order_status(self, order_id):
        """Get the status of an order."""

        return self.request('/order/status', {'order_id': order_id})

    def position_claim(self, position_id):
        """Claim a position."""

        return self.request('/position/claim', {'position_id': position_id})

    def history(self, currency, since=None, until=None, limit=500,
                wallet=None):
        """View all of your balance ledger entries."""

        params = {
            'currency': currency,
            'limit': limit,
        }
        if wallet is not None:
            params['wallet'] = wallet
        if since is not None:
            params['since'] = since
        if until is not None:
            params['until'] = until
        return self.request('/history', params)

    def history_movements(self, currency, method=None, since=None, until=None,
                          limit=500):
        """View your past deposits/withdrawals."""

        params = {
            'currency': currency,
            'limit': limit,
        }
        if method is not None:
            params['method'] = method
        if since is not None:
            params['since'] = since
        if until is not None:
            params['until'] = until
        return self.request('/history/movements', params)

    def mytrades(self, symbol, timestamp, limit_trades=50):
        """View your past trades."""

        params = {
            'symbol': symbol,
            'timestamp': timestamp,
            'limit_trades': limit_trades,
        }
        return self.request('/mytrades', params)

    def offer_new(self, currency, amount, rate, period, direction):
        """Submit a new offer."""

        params = {
            'currency': currency,
            'amount': amount,
            'rate': rate,
            'period': period,
            'direction': direction,
        }
        return self.request('/offer/new', params)

    def offer_cancel(self, offer_id):
        """Cancel an offer."""

        return self.request('/offer/cancel', {'offer_id': offer_id})

    def offer_status(self, offer_id):
        """Get the status of an offer"""

        return self.request('/offer/status', {'offer_id': offer_id})


class Bitfinex(DataProvider):
    """DataProvider for bitfinex

    Emulates asynchronous client for rest api by making requests periodically
    and posting events in case of different results
    """
    sides = {'buy': models.BIDBUY,
             'sell': models.ASKSELL}

    def start(self):
        self.rest = BitfinexRest(self.config['key'], self.config['secret'])
        self._last_call = None
        self._last_trades = None
        self._loop = self.rest._loop
        self.symbols = ('btcusd',)
        self.counter = 0
        # self._looping_call()

    def _looping_coro(self):
        if self.counter >= len(self.symbols):
            self.counter = 0
        symb = self.symbols[self.counter]
        self.counter += 1
        try:
            book = yield from self.rest.order_book(symb,
                                                   limit_bids=LIMIT,
                                                   limit_asks=LIMIT)
            last_call = self._last_call
            if not last_call:
                last_call = date2timestamp(date.today())
            self._last_call = time.time()
            trades = yield from self.rest.trades(symb, limit=LIMIT,
                                                 since=int(last_call))
            yield from self._loop.run_in_executor(None, self._order_book,
                                                  symb, book, trades)
        except Exception:
            self.logger.exception('Exception while retrieving bitfinex')

        self._loop.call_later(self.config.get('rest_refresh_interval', 4),
                              self._looping_call)

    def _order_book(self, symb, obook, trades):
        book = self.book(symb)
        for x in trades:
            if x['exchange'] == self.name:
                book.record(update_type=models.TRADE,
                            side=self.sides[x['type']],
                            price=float(x['price']),
                            size=float(x['amount']),
                            trade_id=x['tid'])
        self._add(symb, obook['asks'], models.ASKSELL, trades)
        self._add(symb, obook['bids'], models.BIDBUY, trades)

    def _add(self, symb, data, side, trades):
        book = self.book(symb)
        table = book.get_table()
        for x in data:
            timestamp = x['timestamp']
            price = float(x['price'])
            size = float(x['amount'])
            prices = table.where('((price=%s) & (side=%d))' % (price, side))
            prices = tuple(prices)
            update_type = models.ORDER
            if prices and prices[0] < timestamp:
                # last = prices[0]
                # we have a new update
                change = prices[0] - size
                if change < 0:
                    update_type = models.TRADE_OR_CANCEL

            book.record(update_type=update_type,
                        side=side,
                        price=float(x['price']),
                        size=float(x['amount']))

    def _looping_call(self):
        async(self._looping_coro(), loop=self._loop)
