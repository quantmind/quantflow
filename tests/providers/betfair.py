import unittest

from quantflow.providers.betfair import BetFair


class TestBetfair(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        key = cls.cfg.get('betfair_key')
        username = cls.cfg.get('betfair_username')
        password = cls.cfg.get('betfair_password')
        certfile = cls.cfg.get('betfair_certfile')
        keyfile = cls.cfg.get('betfair_keyfile')
        if not key:
            raise unittest.SkipTest('betfair_key missing')
        if not username:
            raise unittest.SkipTest('betfair_username missing')
        if not password:
            raise unittest.SkipTest('betfair_password missing')
        if not certfile:
            raise unittest.SkipTest('betfair_certfile missing')
        if not keyfile:
            raise unittest.SkipTest('betfair_keyfile missing')
        cls.betfair = BetFair(key, username, password,
                              certfile=certfile, keyfile=keyfile)
        yield from cls.betfair.login()

    def test_listCompetions(self):
        all = yield from self.betfair.listCompetitions(filter={})
        self.assertTrue(all)

    def test_getEventTypes(self):
        all = yield from self.betfair.listEventTypes(filter={})
        self.assertTrue(all)
