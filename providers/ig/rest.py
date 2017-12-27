"""
https://github.com/ig-python/ig-markets-api-python-library
"""
import os

from pulsar.apps.http import HttpClient

from .stream import IgStream
from ..utils import from_config


class IgRest:
    endpoint = 'https://api.ig.com/gateway/deal'
    demo_endpoint = 'https://demo-api.ig.com/gateway/deal'

    def __init__(self, username=None, api_key=None, password=None,
                 config_file=None, http=None, demo=False):
        if demo:
            self.endpoint = self.demo_endpoint
        self.stream_endpoint = None
        self.oauth_token = None
        self.http = http or HttpClient()
        self.auth = from_config(dict(
                username=username or os.environ.get('IG_USERNAME'),
                password=password or os.environ.get('IG_PASSWORD'),
                api_key=api_key or os.environ.get('IG_API_KEY')
            ),
            config_file=config_file,
            entry='igindex'
        )
        self.headers = {
            'X-IG-API-KEY': self.auth['api_key'],
            'Content-Type': 'application/json',
            'Accept': 'application/json; charset=UTF-8'
        }

    async def login(self):
        response = await self.http.post(
            '%s/session' % self.endpoint,
            headers=self.headers,
            json=dict(
                identifier=self.auth['username'],
                password=self.auth['password']
            )
        )
        response.raise_for_status()
        data = response.json()
        self.headers['CST'] = response.headers['CST']
        self.headers['X-SECURITY-TOKEN'] = response.headers['X-SECURITY-TOKEN']
        self.stream_endpoint = data['lightstreamerEndpoint']
        return data

    async def logout(self):
        response = await self.http.delete(
            '%s/session' % self.endpoint,
            headers=self.headers
        )
        response.raise_for_status()
        self.headers.pop('CST')
        self.headers.pop('X-SECURITY-TOKEN')

    async def stream(self):
        if not self.stream_endpoint:
            await self.login()
        stream = IgStream(self.stream_endpoint, http=self.http)
        await stream.connect(
            self.auth['username'],
            self.headers['CST'],
            self.headers['X-SECURITY-TOKEN']
        )
        return stream

    async def accounts(self):
        data = await self.request(
            'GET',
            'accounts'
        )
        return data['accounts']

    async def user(self):
        data = await self.request(
            'GET',
            'session'
        )
        return data

    async def market_navigation(self, id=None):
        path = 'marketnavigation/%s' % id if id else 'marketnavigation'
        data = await self.request('GET', path)
        return data

    async def markets(self, epic=None):
        path = 'markets/%s' % epic if epic else 'markets'
        data = await self.request('GET', path)
        return data

    async def request(self, method, path, headers=None, **kwargs):
        heads = self.headers
        if headers:
            heads = heads.copy()
            heads.update(headers)
        url = '%s/%s' % (self.endpoint, path)
        response = await self.http.request(
            method, url, headers=heads, **kwargs
        )
        response.raise_for_status()
        return response.json()
