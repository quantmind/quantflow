import unittest

from providers import Yahoo


class TestYahoo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.yahoo = Yahoo()

    @classmethod
    def tearDownClass(cls):
        return cls.yahoo.http.close()

    async def test_quote(self):
        share = self.yahoo.symbol('idv')
        self.assertEqual(share.symbol, 'IDV')
        await share.quote()
        float(share.price)

    async def test_historical_prices(self):
        share = self.yahoo.symbol('idv')
        ts = await share.prices()
        self.assertEqual(len(ts.columns), 7)

    async def test_historical_dividends(self):
        share = self.yahoo.symbol('idv')
        ts = await share.dividends()
        self.assertEqual(len(ts.columns), 2)

    async def __test_bad_share(self):
        share = await self.yahoo.share('shdgcv')
        self.assertEqual(share.symbol, 'shdgcv')
