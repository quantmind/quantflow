import unittest
from unittest import SkipTest

from quantflow.protocols import Pusher, PusherChannel


class PusherTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = cls.cfg.params
        app_id = cfg.get('pusher_app_id')
        app_key = cfg.get('pusher_app_key')
        app_secret = cfg.get('pusher_app_secret')
        if not app_id:
            raise SkipTest('Requires pusher_app_id in tests/config.py')
        if not app_key:
            raise SkipTest('Requires pusher_app_key in tests/config.py')
        if not app_secret:
            raise SkipTest('Requires pusher_app_secret in tests/config.py')
        cls.pusher = Pusher(app_id, key=app_key, secret=app_secret)

    def test_pusher(self):
        pusher = self.pusher
        self.assertTrue(pusher.app_id)
        self.assertTrue(pusher.host)
        self.assertEqual(pusher.port, 80)

    def test_channel(self):
        pusher = self.pusher
        channel = pusher['test_channel']
        self.assertTrue(channel)
        self.assertEqual(channel.name, 'test_channel')
        self.assertEqual(channel.path,
                         '/apps/%s/channels/test_channel/events' %
                         pusher.app_id)
        self.assertEqual(channel.pusher, pusher)
        self.assertEqual(channel, pusher['test_channel'])

    def test_bind(self):
        pusher = self.pusher
        channel = pusher['test_channel2']

        def test_data(data):
            pass

        # channel.bind(test_data)
        result = yield from channel.trigger('hello',
                                            {'message': 'Hello world'})
        self.assertEqual(result, True)
