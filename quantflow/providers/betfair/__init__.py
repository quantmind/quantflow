from pulsar.apps.rpc import JsonProxy


ENDPOINT = 'https://api.betfair.com/exchange/betting/json-rpc/v1'
LOGIN_URL = 'https://identitysso.betfair.com/api/certlogin'


class BetFair(JsonProxy):

    def __init__(self, key, username, password, **kwargs):
        super().__init__(ENDPOINT, **kwargs)
        self.username = username
        self.password = password
        self._http.headers['X-Application'] = key

    def login(self):
        headers = [('content-type', 'application/x-www-form-urlencoded')]
        body = dict(username=self.username, password=self.password)
        response = yield from self._http.post(LOGIN_URL, data=body)
        response.raise_for_status()
        data = response.json()
        if data.get('loginStatus') == 'SUCCESS':
            token = data['sessionToken']
            self._http.headers['X-Authentication'] = token
            return token
        else:
            raise ValueError
