import unittest

from providers.selftrade import Selftrade


class TestBetfair(unittest.TestCase):

    @classmethod
    async def setUpClass(cls):
        cls.st = Selftrade()
        # await cls.st.login()

    async def test_auth(self):
        await self.st.get_data()
        accounts = await self.st.accounts()
        pass
        # self.assertTrue(bf.auth)
        # self.assertTrue(bf.auth['username'])
        # self.assertTrue(bf.auth['password'])
        # self.assertTrue('X-Application' in bf.headers)
