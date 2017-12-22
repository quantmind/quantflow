import os

from pulsar.apps.http import HttpClient

from ..utils import from_config


class FigiError(ValueError):
    pass


class Figi:
    """The OpenFIGI REST API Client

    https://openfigi.com
    """
    Error = FigiError
    BASE_URL = 'https://api.openfigi.com/v1/mapping'
    ID_TYPES = dict(
        TICKER=(
            'Ticker is a specific identifier for a financial instrument'
            ' that reflects common usage'
        )
    )

    def __init__(self, api_key=None, http=None, config_file=None):
        self.http = http or HttpClient()
        api_key = from_config(dict(
            api_key=api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        ),
            config_file=config_file,
            entry='figi'
        )['api_key']
        self.headers = {'Content-Type': 'text/json'}
        if api_key:
            self.headers['X-OPENFIGI-APIKEY'] = api_key

    async def get(self, idType, idValue, **data):
        data.update(
            idType=idType,
            idValue=idValue
        )
        response = await self.http.post(
            self.BASE_URL, headers=self.headers, json=[data]
        )
        response.raise_for_status()
        result = response.json()
        assert len(result), "expected one result"
        result = result[0]
        if 'data' in result:
            return result['data']
        else:
            raise FigiError(result['error'])

    def ticker(self, value, **data):
        return self.get('TICKER', value, **data)
