import unittest

from providers import Figi


class TestAlphaVantage(unittest.TestCase):
    """https://www.alphavantage.co/documentation/
    """
    BASE_URL = 'https://www.alphavantage.co/query'

    @classmethod
    async def setUpClass(cls):
        cls.api = Figi()

    async def test_session(self):
        data = await self.api.ticker('GOOG')
        self.assertTrue(data)
        self.assertTrue(len(data) > 1)
        data = await self.api.ticker('GOOG', exchCode='US')
        self.assertTrue(data)
        self.assertEqual(len(data), 1)
        goog = data[0]
        self.assertEqual(goog['figi'], 'BBG009S3NB30')

