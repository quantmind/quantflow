import unittest

from quantflow.providers import Yahoo


class TestBetfair(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.yahoo = Yahoo()

    async def test_share(self):
        share = await self.yahoo.share('idv')
        self.assertEqual(share.symbol, 'idv')
