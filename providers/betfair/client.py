import os
import configparser

from pulsar.apps.http import HttpClient

from .api import BetFairApi, BetFairResponseError, APIS


ENDPOINT = 'https://api.betfair.com/exchange/%s/json-rpc/v1'
LOGIN_URL = 'https://identitysso.betfair.com/api/certlogin'
CONFIG_FILE = '.fluidily'


class BetFair:
    """Client to Betfair JSON-RPC API

    .. attribute: cert

        Certificate, Key pair for authentication

    .. attribute: headers

        Dictionary of headers to send to betfair API server
    """

    def __init__(self, cert=None, http=None, key=None, username=None,
                 password=None):
        cert, cert_key = cert or (None, None)
        self.auth = from_config(
            dict(
                username=username or os.environ.get('BETFAIR_USERNAME'),
                password=password or os.environ.get('BETFAIR_PASSWORD'),
                key=key or os.environ.get('BETFAIR_KEY'),
                cert=cert or os.environ.get('BETFAIR_CERTIFICATE'),
                cert_key=cert_key or os.environ.get('BETFAIR_CERTIFICATE_KEY')
            ),
            entry='betfair'
        )
        self.cert = (cp(self.auth.pop('cert')), cp(self.auth.pop('cert_key')))
        self.http = http or HttpClient()
        self.headers = {'X-Application': self.auth.pop('key')}
        self.betting = self.create_api('betting')
        self.race_status = self.create_api('scores')
        self.account = self.create_api('account')

    async def login(self):
        headers = [('content-type', 'application/x-www-form-urlencoded')]
        headers.extend(self.headers.items())
        response = await self.http.post(
            LOGIN_URL, data=self.auth, headers=headers, cert=self.cert
        )
        response.raise_for_status()
        data = response.json()
        status = data['loginStatus']
        if status == 'SUCCESS':
            token = data['sessionToken']
            self.headers['X-Authentication'] = token
            return token
        else:
            raise BetFairResponseError(403, status)

    def create_api(self, name):
        return BetFairApi(
            ENDPOINT % name,
            http=self.http,
            headers=self.headers,
            **APIS[name]
        )


def from_config(keys, entry=None, filename=None):
    config = configparser.ConfigParser()
    filename = filename or CONFIG_FILE
    path = os.path.join(os.path.expanduser("~"), filename)
    if os.path.isfile(path):
        config.read(path)
        entry = entry or 'default'
        for key, value in keys.items():
            if not value:
                keys[key] = config.get(entry, key)
    return keys


def cp(path):
    if path:
        return os.path.expanduser(path)
