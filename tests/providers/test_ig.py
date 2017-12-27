import unittest

from pulsar.apps.test import sequential

from providers.ig import IgRest


@sequential
class IgRestTestCase(unittest.TestCase):
    client = None
    stream = None

    @classmethod
    async def setUpClass(cls):
        cls.client = IgRest()
        await cls.client.login()
        cls.stream = await cls.client.stream()

    @classmethod
    async def tearDownClass(cls):
        if cls.client:
            await cls.client.logout()
            if cls.stream:
                await cls.stream.close()
            await cls.client.http.close()

    async def test_accounts(self):
        accounts = await self.client.accounts()
        self.assertTrue(accounts)

    async def test_user(self):
        user = await self.client.user()
        self.assertTrue(user)

    async def test_market_navigation(self):
        data = await self.client.market_navigation()
        self.assertTrue(data)
        nodes = data['nodes']
        self.assertTrue(nodes)

    async def test_market_navigation_fx(self):
        data = await self.client.market_navigation(165333)
        self.assertTrue(len(data['nodes']), 4)
        # major FX markets
        data = await self.client.market_navigation(166904)
        self.assertTrue(len(data['markets']))

    async def test_markets(self):
        data = await self.client.markets('CS.D.EURUSD.TODAY.IP')
        self.assertTrue(data)

    async def test_stream(self):
        self.assertTrue(self.stream)
