import os
import time
from math import floor
from itertools import chain

from dateutil.parser import parse as parse_date

import bs4

from pulsar.apps.http import HttpClient

from ..utils import from_config, user_agent_info

import requests

LOGIN_URL = (
    "https://selftrade.co.uk/transactional/anonymous/login"
)

API_URL = "https://selftrade.co.uk/api"


API_KEY_START = "var  /* -transactional-anonymous-login */ apiKey = '"


class AuthenticationError(ValueError):
    pass


class Selftrade:
    version = '0.1.0'
    api_key = None
    cookie_form_keys = ('__RequestVerificationToken',)

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
        response = requests.get(LOGIN_URL, headers=self.headers)
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
        self.api_key = self.get_api_key(
            bs.find_all('script', type='text/javascript')
        )
        if not self.api_key:
            raise AuthenticationError('cannot find api key')
        response = await self.api_request(
            'GET',
            'LoginApi/GetPinFormat',
            headers=self.headers,
            params=dict(username=data['Username'])
        )
        response.raise_for_status()
        data.update(response.json())
        pin = str(self.auth['pin'])
        data['PasswordCharacter1'] = pin[data['Character1Index']-1]
        data['PasswordCharacter2'] = pin[data['Character2Index']-1]
        data['PasswordCharacter3'] = pin[data['Character3Index']-1]
        headers = self.headers.copy()
        headers['content-type'] = 'multipart/form-data'
        cookies = dict(((key, data[key]) for key in self.cookie_form_keys))
        #
        # Perform login
        response = requests.post(
            LOGIN_URL,
            headers=headers,
            data=data,
            cookies=cookies
        )
        response.raise_for_status()
        #
        # get the new API key
        bs = bs4.BeautifulSoup(response.content, 'html.parser')
        key = self.api_key
        self.api_key = self.get_api_key(
            bs.find_all('script', type='text/javascript')
        )
        assert key != self.api_key
        response = await self.inbox_message_count()
        return response

    async def inbox_message_count(self):
        response = await self.api_request(
            'GET',
            'SecureInboxApi/GetSecureInboxMessageCount'
        )
        response.raise_for_status()
        return response

    async def api_request(self, method, path, params=None,
                          headers=None, **kwargs):
        headers = headers or self.headers
        url = '%s/%s' % (API_URL, path)
        if not params:
            params = {}
        params.update(
            apiKey=self.api_key,
            _=floor(time.time() * 1000)
        )
        response = requests.request(
            method, url, headers=headers, params=params, **kwargs
        )
        response.raise_for_status()
        return response

    def get_api_key(self, scripts):
        for script in scripts:
            for line in script.text.split('\r\n'):
                line = line.strip();
                if line.startswith(API_KEY_START):
                    return line[len(API_KEY_START):-2]
