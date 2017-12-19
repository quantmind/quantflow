import unittest

from bots.betfair import BetFair


class TestBetfair(unittest.TestCase):

    @classmethod
    async def setUpClass(cls):
        cls.bf = BetFair()
        await cls.bf.login()

    def test_client_auth(self):
        bf = self.bf
        self.assertTrue(bf.auth)
        self.assertTrue(bf.auth['username'])
        self.assertTrue(bf.auth['password'])
        self.assertTrue('X-Application' in bf.headers)

    def test_betting_api(self):
        bf = self.bf
        self.assertTrue(
            bf.betting.url,
            'https://api.betfair.com/exchange/betting/json-rpc/v1'
        )

    async def test_list_events(self):
        events = await self.bf.betting.listEventTypes(filter={})
        self.assertTrue(events)
        self.assertIsInstance(events, list)

    def test_race_status_api(self):
        bf = self.bf
        self.assertTrue(bf.race_status.url)

    def test_account_api(self):
        bf = self.bf
        self.assertTrue(bf.account.url)
