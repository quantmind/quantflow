import unittest

from providers.morningstar import MorningStar


class TestMorningStart(unittest.TestCase):

    @classmethod
    async def setUpClass(cls):
        cls.api = MorningStar()
        # await cls.st.login()

    async def test_ping(self):
        response = await self.api.auth()
        return response
