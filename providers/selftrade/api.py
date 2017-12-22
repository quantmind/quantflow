import os
import sys
import time
import json
import asyncio
from math import floor
from datetime import date
from itertools import chain

from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

import bs4

from pulsar.apps.http import HttpClient

from ..utils import from_config, user_agent_info


d = os.path.dirname

STREAM_LIMIT = 2**20
PATH = d(d(d(__file__)))
PHANTOM_JS = os.path.join(PATH, 'node_modules', '.bin', 'phantomjs')
LOGIN_URL = (
    "https://selftrade.co.uk/transactional/anonymous/login"
)

WEB_URL = "https://selftrade.co.uk"
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
        # self.http = http or requests.Session()
        self.http = http or HttpClient()
        if 'user-agent' not in self.headers:
            self.headers['user-agent'] = (
                'selftrade-client/%s%s' % (self.version, user_agent_info())
            )

    async def get_data(self, stdout=None, stderr=None, limit=None):
        dob = parse_date(self.auth['dob']).date()
        commands = [
            'cd %s' % PATH,
            'export PHANTOMJS_EXECUTABLE=%s' % PHANTOM_JS,
            'export USERNAME=%s' % self.auth['an'],
            'export DAY=%s' % dob.day,
            'export MONTH=%s' % dob.month,
            'export YEAR=%s' % dob.year,
            'export PASSWORD=%s' % self.auth['pin'],
            './node_modules/.bin/casperjs providers/selftrade/worker.js'
        ]
        command_str = ' && '.join(commands)
        proc = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=limit or STREAM_LIMIT
        )
        msg, err = await asyncio.gather(
            _interact(proc, 1, stdout or sys.stdout),
            _interact(proc, 2, stderr or sys.stderr)
        )
        if proc.returncode:
            msg = '%s%s' % (msg.decode('utf-8'), err.decode('utf-8'))
            raise RuntimeError(msg.strip(), proc.returncode)
        return msg.decode('utf-8').strip()

    async def login(self):
        """Perform login
        """
        dob = parse_date(self.auth['dob']).date()
        response = await self.web_request(
            'GET',
            'transactional/anonymous/login'
        )
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
                data[name] = str(value)
        response = await self.api_request(
            'GET',
            'LoginApi/GetPinFormat',
            self.get_api_key(bs),
            headers=self.headers,
            params=dict(username=data['Username'])
        )
        response.raise_for_status()
        data.update(response.json())
        pin = str(self.auth['pin'])
        for idx in range(1, 4):
            key = 'Character%dIndex' % idx
            data['PasswordCharacter%d' % idx] = pin[data[key]-1]
            data[key] = str(data[key])
        headers = self.headers.copy()
        # headers['Content-Type'] = 'multipart/form-data'
        cookies = dict(((key, data[key]) for key in self.cookie_form_keys))
        #
        # Perform login
        response = await self.web_request(
            'POST',
            'transactional/anonymous/login',
            headers=headers,
            files=data,
            cookies=cookies
        )
        return response

    async def accounts(self):
        response = await self.web_request(
            'GET', 'transactional/authenticated/dashboard'
        )
        bs = bs4.BeautifulSoup(response.content, 'html.parser')
        info = await self.api_request(
            'GET',
            'SecureNavApi/GetSecureNavData',
            self.get_api_key(bs)
        )
        return info

    async def cash_statement(self, currency=None, fromdate=None, todate=None):
        todate = todate or date.today()
        fromdate = fromdate or todate - relativedelta(months=3)
        response = await self.api_request(
            'GET',
            'CashStatementApi/GetCashStatementData',
            params=dict(
                currency=currency or 'GBP',
                fromDate=fromdate,
                toDate=todate
            )
        )
        return response.json()

    async def inbox_message_count(self):
        response = await self.api_request(
            'GET',
            'SecureInboxApi/GetSecureInboxMessageCount'
        )
        response.raise_for_status()
        return response

    async def web_request(self, method, path,
                          headers=None, **kwargs):
        headers = headers or self.headers
        url = '%s/%s' % (WEB_URL, path)
        response = await self.http.request(
            method, url, headers=headers, **kwargs
        )
        response.raise_for_status()
        return response

    async def api_request(self, method, path, api_key, params=None,
                          headers=None, **kwargs):
        headers = headers or self.headers
        url = '%s/%s' % (API_URL, path)
        if not params:
            params = {}
        params.update(
            apiKey=api_key,
            _=floor(time.time() * 1000)
        )
        response = await self.http.request(
            method, url, headers=headers, params=params, **kwargs
        )
        response.raise_for_status()
        return response

    def get_api_key(self, bs):
        scripts = bs.find_all('script', type='text/javascript')
        for script in scripts:
            for line in script.text.split('\r\n'):
                line = line.strip();
                if line.startswith(API_KEY_START):
                    return line[len(API_KEY_START):-2]
        raise AuthenticationError('cannot find api key')


async def _interact(proc, fd, out):
    transport = proc._transport.get_pipe_transport(fd)
    stream = proc.stdout if fd == 1 else proc.stderr
    try:
        if fd == 1:
            interactions = []
            output = bytearray()
            while True:
                line = await stream.readline()
                if not line:
                    break
                out.write(line.decode('utf-8'))
                if line == b'undefined\n':
                    data = output.decode('utf-8')
                    try:
                        data = json.loads(data)
                    except Exception:
                        pass
                    interactions.append(data)
                    output = bytearray()
                else:
                    output.extend(line)
            return interactions
        else:
            err = await stream.read()
            return err
    finally:
        transport.close()
