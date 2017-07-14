from pulsar.apps.rpc import JsonProxy


ENDPOINT = 'https://api.betfair.com/exchange/betting/json-rpc/v1'
LOGIN_URL = 'https://identitysso.betfair.com/api/certlogin'


class BetFairError(RuntimeError):
    pass


class BetFair(JsonProxy):
    """Client to Betfair JSON-RPC API
    """

    def __init__(self, key, username, password, **kwargs):
        super().__init__(ENDPOINT, **kwargs)
        self.username = username
        self.password = password
        self._http.headers['X-Application'] = key

    async def login(self):
        headers = [('content-type', 'application/x-www-form-urlencoded')]
        body = dict(username=self.username, password=self.password)
        response = await self._http.post(LOGIN_URL, data=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        status = data['loginStatus']
        if status == 'SUCCESS':
            token = data['sessionToken']
            self._http.headers['X-Authentication'] = token
            return token
        else:
            raise BetFairError(status)

    def __getattr__(self, name):
        name = 'SportsAPING/v1.0/%s' % name
        return super().__getattr__(name)
