import os
import time
from math import floor
from itertools import chain

from dateutil.parser import parse as parse_date

import bs4

from pulsar.apps.http import HttpClient

from ..utils import from_config, user_agent_info


LOGIN_URL = (
    "https://selftrade.co.uk/transactional/anonymous/login"
)

LOGIN_URL_2 = (
    "https://selftrade.co.uk/api/LoginApi/GetPinFormat"
)

API_KEY_START = "var  /* -transactional-anonymous-login */ apiKey = '"


class AuthenticationError(ValueError):
    pass


class Selftrade:
    version = '0.1.0'

    def __init__(self, an=None, dob=None, pin=None, config_file=None,
                 headers=None, http=None):
        self.auth = from_config(dict(
                an=an or os.environ.get('SELFTRADE_DOB'),
                dob=dob or os.environ.get('SELFTRADE_DOB'),
                pin=pin or os.environ.get('SELFTRADE_PIN')
            ),
            config_file=config_file,
            entry='selftrade')
        self.headers = headers or {}
        self.http = http or HttpClient()
        if 'user-agent' not in self.headers:
            self.headers['user-agent'] = (
                'selftrade-client/%s%s' % (self.version, user_agent_info())
            )

    async def login(self):
        dob = parse_date(self.auth['dob']).date()
        response = await self.http.get(LOGIN_URL, headers=self.headers)
        response.raise_for_status()
        bs = bs4.BeautifulSoup(response.content, 'html.parser')
        form = bs.find('form', id='LoginForm')
        data = {}
        for input in chain(form.find_all('input'), form.find_all('select')):
            name = input.get('name')
            if not name:
                continue
            value = input.get('value')
            if name == 'Username':
                value = self.auth['an']
            elif name == 'DateOfBirthViewModel.Day':
                value = dob.day
            elif name == 'DateOfBirthViewModel.Month':
                value = dob.month
            elif name == 'DateOfBirthViewModel.Year':
                value = dob.year
            if value:
                data[name] = value
        key = self.get_api_key(bs.find_all('script', type='text/javascript'))
        if not key:
            raise AuthenticationError('cannot find api key')
        response = await self.http.get(
            LOGIN_URL_2, headers=self.headers,
            params=dict(
                apiKey=key,
                username=data['Username'],
                _=floor(time.time()*1000)
            )
        )
        response.raise_for_status()
        data.update(response.json())
        pin = str(self.auth['pin'])
        data['PasswordCharacter1'] = pin[data['Character1Index']]
        data['PasswordCharacter2'] = pin[data['Character2Index']]
        data['PasswordCharacter3'] = pin[data['Character3Index']]
        response = await self.http.post(
            LOGIN_URL,
            headers=self.headers,
            data=data
        )
        response.raise_for_status()
        return response

    def get_api_key(self, scripts):
        for script in scripts:
            for line in script.text.split('\r\n'):
                line = line.strip();
                if line.startswith(API_KEY_START):
                    return line[len(API_KEY_START):-2]
