import unittest

from providers import TrueFX


class TestAlphaVantage(unittest.TestCase):
    """https://www.alphavantage.co/documentation/
    """
    BASE_URL = 'https://www.alphavantage.co/query'

    @classmethod
    async def setUpClass(cls):
        cls.api = TrueFX()

    async def test_session(self):
        session = await self.api.intraday()
        self.assertTrue(session.id)
        self.assertEqual(len(session.data), 9)
        self.assertTrue(session.timestamp)
        self.assertTrue(await session.refresh() <= 9)

