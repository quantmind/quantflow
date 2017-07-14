from pulsar.apps.http import HttpClient

from .share import Share
from .currency import Currency

# Yahoo! YQL API
PUBLIC_API_URL = 'https://query.yahooapis.com/v1/public/yql'
OAUTH_API_URL = 'https://query.yahooapis.com/v1/yql'
DATATABLES_URL = 'store://datatables.org/alltableswithkeys'


class Yahoo:

    def __init__(self, http=None):
        self.http = http or HttpClient()

    async def share(self, symbol):
        return await Share.get(self, symbol)

    async def execute(self, yql, token=None):
        response = await self.get(
            PUBLIC_API_URL,
            params=dict(
                q=yql,
                format='json',
                env=DATATABLES_URL
            )
        )
        response.raise_for_status()
        data = response.json()
        return data
