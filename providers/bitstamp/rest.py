import os
import time
import hmac

from pulsar.apps.http import HttpClient

from ..utils import from_config


class BitstampRest:
    """Rest Clinet for Bitstamp.

    API description at https://www.bitstamp.net/api/
    """
    endpoint = 'https://www.bitstamp.net/api'

    def __init__(self, client_id=None, api_key=None, api_secret=None,
                 config_file=None, http=None):
        self.http = http or HttpClient()
        self.auth = from_config(dict(
                client_id=client_id or os.environ.get('BITSTAMP_CLIENT_ID'),
                api_key=api_key or os.environ.get('BITSTAMP_API_KEY'),
                api_secret=api_secret or os.environ.get('BITSTAMP_API_SECRET')
            ),
            config_file=config_file,
            entry='truefx'
        )
        self.nonce = int(1000000000000000*time.monotonic())

    def request_unauth(self, url):
        """Perform unauthorized request."""
        response = yield from self.http.get(self.endpoint + url)
        response.raise_for_status()
        return response.json()

    async def request(self, url, **data):
        """Performs request with authorization."""
        auth = self.auth
        self.nonce += 1
        payload = (
                str(self.nonce) + auth['client_id'] + auth['api_key']
        ).encode('utf-8')
        signer = hmac.new(auth['api_key'].encode('ascii'),
                          msg=payload, digestmod='SHA256')
        signature = signer.hexdigest().upper()

        data.update(key=auth['api_key'], signature=signature,
                    nonce=self.nonce)

        response = await self.client.post(self.endpoint + url, data=data)
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
