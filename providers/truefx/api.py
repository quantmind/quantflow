import os
import uuid

import ccy

from pulsar.apps.http import HttpClient

from ..utils import from_config


CCYS = set(('AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK'))


class AlphaVantageError(ValueError):
    pass


class TrueFX:
    BASE_URL = 'http://webrates.truefx.com/rates/connect.html'

    def __init__(self, username=None, password=None,
                 http=None, config_file=None):
        self.http = http or HttpClient()
        self.auth = from_config(dict(
                u=username or os.environ.get('TRUEFX_USERNAME'),
                p=password or os.environ.get('TRUEFX_PASSWORD')
            ),
            config_file=config_file,
            entry='truefx'
        )

    async def intraday(self, ccys=None):
        if not ccys:
            ccys = ','.join((ccy.currency(c).as_cross('/') for c in CCYS))
        session = TrueFXSession(self, ccys)
        await session.start()
        return session


class TrueFXSession:

    def __init__(self, api, ccys):
        self.api = api
        self.ccys = ccys
        self.q = str(uuid.uuid4())[:8]
        self.id = None
        self.data = {}
        self.timestamp = 0

    async def start(self):
        params = self.api.auth.copy()
        params.update(c=self.ccys, f='csv', q=self.q, s='n')
        response = await self.api.http.get(self.api.BASE_URL, params=params)
        response.raise_for_status()
        self.id = response.text.strip()
        response = await self.api.http.get(
            self.api.BASE_URL, params=dict(id=self.id)
        )
        response.raise_for_status()
        for row in response.text.split('\n'):
            bits = row.split(',')
            if len(bits) != 9:
                break
            self.data[bits[0]] = dict(
                timestap=self.update_timestamp(bits[1]),
                bid=float(bits[2]+bits[3]),
                offer=float(bits[4] + bits[5])
            )

    async def refresh(self):
        assert self.id, "session not started"
        response = await self.api.http.get(
            self.api.BASE_URL, params=dict(id=self.id)
        )
        response.raise_for_status()
        updated = {}
        for row in response.text.split('\n'):
            bits = row.split(',')
            if len(bits) != 9:
                break
            entry = dict(
                timestap=self.update_timestamp(bits[1]),
                bid=float(bits[2]+bits[3]),
                offer=float(bits[4] + bits[5])
            )
            self.data[bits[0]] = entry
            updated[bits[0]] = entry
        return updated

    def update_timestamp(self, timestamp):
        timestamp = int(timestamp)
        self.timestamp = max(self.timestamp, timestamp)
        return timestamp
