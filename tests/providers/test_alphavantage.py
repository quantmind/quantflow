import unittest

from providers import AlphaVantage


class TestAlphaVantage(unittest.TestCase):

    @classmethod
    async def setUpClass(cls):
        cls.api = AlphaVantage()
        # await cls.st.login()

    async def test_intraday(self):
        response = await self.api.intraday('BP')
        return response

    async def test_fx(self):
        data = await self.api.fx('EUR')
        self.assertEqual(data['code'], 'USDEUR')
        data = await self.api.fx('BTC')
        self.assertEqual(data['code'], 'USDBTC')
        data = await self.api.fx('XRP')
        self.assertEqual(data['code'], 'USDXRP')
        data = await self.api.fx('USD', 'EUR')
        self.assertEqual(data['code'], 'EURUSD')
        data = await self.api.fx('USD', 'BTC')
        self.assertEqual(data['code'], 'BTCUSD')
        data = await self.api.fx('USD', 'XRP')
        self.assertEqual(data['code'], 'XRPUSD')
