import asyncio
import unittest

from pulsar.apps.test import test_timeout

from providers.bitstamp import BitstampWs


class TestBitStampWs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stamp = BitstampWs()
        return cls.stamp.pusher.connect()

    def test_base(self):
        self.assertTrue(self.stamp.pusher)

    @test_timeout(60)
    async def test_live_trade(self):
        stamp = self.stamp
        future = asyncio.Future()

        def live_trades(data, event=None):
            if future.done():
                return
            try:
                self.assertEqual(event, 'trade')
                self.assertIsInstance(data, dict)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        ch = await stamp.live_trades(live_trades)
        self.assertEqual(ch.name, 'live_trades')
        await future

    @test_timeout(60)
    async def test_order_book(self):
        stamp = self.stamp
        future = asyncio.Future()

        def order_book(data, event=None):
            try:
                self.assertEqual(event, 'data')
                self.assertIsInstance(data, dict)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        ch = await stamp.order_book(order_book)
        self.assertEqual(ch.name, 'order_book')
        await future
