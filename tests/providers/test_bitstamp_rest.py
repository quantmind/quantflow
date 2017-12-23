import unittest

from pulsar.apps.test import test_timeout, sequential

from providers.bitstamp import BitstampRest


@sequential
class BitstampRestTestCase(unittest.TestCase):
    user_transaction_keys = ('datetime', 'id', 'type',
                             'usd', 'btc', 'fee', 'order_id')

    open_orders_keys = ('id', 'datetime', 'type', 'price', 'amount')

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        cls.client = BitstampRest()

    @classmethod
    def tearDownClass(cls):
        return cls.client.http.close()

    @test_timeout(60)
    async def test_order_book(self):
        order_book = await self.client.order_book()
        self.assertTrue('bids' in order_book)
        self.assertTrue('asks' in order_book)

    async def test_transactions(self):
        transactions = await self.client.transactions()
        self.assertTrue(len(transactions) > 0)
        checks = [(key in transactions[0]) for key in ('date', 'tid', 'price')]
        self.assertTrue(all(checks))

    async def test_user_transactions(self):
        transactions = await self.client.user_transactions()
        self.assertFalse('error' in transactions)
        self.assertTrue(len(transactions) >= 0)
        if(len(transactions) > 0):
            checks = [(key in transactions[0]) for key in
                      self.user_transaction_keys]
            self.assertTrue(all(checks))

    async def test_open_orders(self):
        res = await self.client.orders()
        self.assertFalse('error' in res)
        self.assertTrue(len(res) >= 0)
        if(len(res) > 0):
            checks = [(key in res[0]) for key in
                      self.open_orders_keys]
            self.assertTrue(all(checks))

    async def test_cancel_order(self):
        res = await self.client.order_cancel('0')
        self.assertEqual(res['error'], 'Order not found')
