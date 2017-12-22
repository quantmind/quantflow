import os

from pulsar.apps.http import HttpClient

from ..utils import from_config


APPLICATION_JSON = 'application/json'
ERROR = 'Error Message'
TIME_SERIES_INTRADAY = 'TIME_SERIES_INTRADAY'
CURRENCY_EXCHANGE_RATE = 'CURRENCY_EXCHANGE_RATE'
REAL_TIME_FX_CODE = 'Realtime Currency Exchange Rate'
INTRADAY_COMPACT = 'compact'
INTRADAY_FULL = 'full'


class AlphaVantageError(ValueError):
    pass


class AlphaVantage:
    BASE_URL = 'https://www.alphavantage.co/query'

    def __init__(self, api_key=None, default_format=None,
                 http=None, config_file=None):
        self.http = http or HttpClient()
        self.default_format = default_format or 'json'
        self.api_key = from_config(dict(
                api_key=api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
            ),
            config_file=config_file,
            entry='alphavantage'
        )['api_key']

    async def intraday(self, symbol, interval=None, full=False, format=None):
        return await self.get(
            function=TIME_SERIES_INTRADAY,
            symbol=symbol,
            outputsize=INTRADAY_FULL if full else INTRADAY_COMPACT,
            interval=interval or '1min',
            datatype=format
        )

    async def fx(self, foreign, base=None):
        base = base or 'USD'
        data = await self.get(
            function=CURRENCY_EXCHANGE_RATE,
            to_currency=foreign,
            from_currency=base
        )
        data = data[REAL_TIME_FX_CODE]
        return dict(code='%s%s' % (base, foreign),
                    price=float(data['5. Exchange Rate']),
                    time=data['6. Last Refreshed'])

    async def get(self, datatype=None, **params):
        params['datatype'] = datatype or self.default_format
        params['apikey'] = self.api_key
        response = await self.http.get(self.BASE_URL, params=params)
        response.raise_for_status()
        ct = response.headers['content-type'].split(';')[0]
        if ct == APPLICATION_JSON:
            data = response.json()
            if ERROR in data:
                raise AlphaVantageError(data[ERROR])
            return data
        else:
            return response.json()

