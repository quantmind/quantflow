import unittest

from quantflow.providers.betfair import BetFair


class TestBetfair(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        key = cls.cfg.get('betfair_key')
        username = cls.cfg.get('betfair_username')
        password = cls.cfg.get('betfair_password')
        if not key:
            raise unittest.SkipTest('betfair key missing')
        if not username:
            raise unittest.SkipTest('betfair username missing')
        if not password:
            raise unittest.SkipTest('betfair password missing')
        cls.betfair = BetFair(key, username, password)
        yield from cls.betfair.login()

    def test_(self):
        pass
