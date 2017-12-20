import pulsar
from pulsar import ensure_future
from pulsar.utils.system import json
from pulsar.apps.data import create_store
from pulsar.apps.http import HttpClient, OAuth1
from pulsar.utils.importer import module_attribute

from pq import api


class Twitter(api.EventDriven):
    """A pulsar :class:`.Application` for connecting and streaming
    tweets from `twitter streaming api`_.
    This application requires the following parameters
    to be specified in your ``config.py`` file:
    * ``twitter_api_key`` the Consumer key of your application
    * ``twitter_api_secret``, the Consumer secret
    * ``twitter_access_token``, the application Access token
    * ``twitter_access_secret``, the Access token secret
    * ``twitter_stream_filter``, dictionary of parameters for
      `filtering tweets`_.
    """
    interval1 = 0
    interval2 = 0
    interval3 = 0
    public_stream = 'https://stream.twitter.com/1.1/statuses/filter.json'

    def __call__(self, on_message=None, **filters):
        api_key = self.get_param('twitter_api_key')
        client_secret = self.get_param('twitter_api_secret')
        access_token = self.get_param('twitter_access_token')
        access_secret = self.get_param('twitter_access_secret')
        self.on_message = module_attribute(on_message)
        self.filters = filters or self.get_param('twitter_filters')
        self._http = HttpClient(encode_multipart=False)
        oauth1 = OAuth1(api_key,
                        client_secret=client_secret,
                        resource_owner_key=access_token,
                        resource_owner_secret=access_secret)
        self._http.bind_event('pre_request', oauth1)
        self.buffer = []
        self.connect()

    def connect(self):
        '''Connect to twitter streaming endpoint.
        If the connection is dropped, the :meth:`reconnect` method is invoked
        according to twitter streaming connection policy_.
        '''
        coro = self._http.post(
            self.public_stream,
            data=self.filters,
            headers=[('content-type', 'application/x-www-form-urlencoded')],
            on_headers=self.connected,
            data_processed=self.process_data,
            post_request=self.reconnect)
        ensure_future(coro)

    def connected(self, response, **kw):
        '''Callback when a succesful connection is made.
        Reset reconnection intervals to 0
        '''
        if response.status_code == 200:
            self.logger.info('Successfully connected with twitter streaming')
            self.interval1 = 0
            self.interval2 = 0
            self.interval3 = 0

    def process_data(self, response, **kw):
        '''Callback passed to :class:`HttpClient` for processing
        streaming data.
        '''
        if response.status_code == 200:
            messages = []
            data = response.recv_body()
            while data:
                idx = data.find(b'\r\n')
                if idx < 0:     # incomplete data - add to buffer
                    self.buffer.append(data)
                    data = None
                else:
                    self.buffer.append(data[:idx])
                    data = data[idx+2:]
                    msg = b''.join(self.buffer)
                    self.buffer = []
                    if msg:
                        body = json.loads(msg.decode('utf-8'))
                        if 'disconnect' in body:
                            msg = body['disconnect']
                            self.logger.warning('Disconnecting (%d): %s',
                                                msg['code'], msg['reason'])
                        elif 'warning' in body:
                            message = body['warning']['message']
                            self.logger.warning(message)
                        else:
                            messages.append(body)
            if messages:
                # a list of messages is available
                if self.on_message:
                    self.on_message(self, messages)

    def reconnect(self, response, exc=None):
        '''Handle reconnection according to twitter streaming policy_
        .. _policy: https://dev.twitter.com/docs/streaming-apis/connecting
        '''
        loop = response._loop
        if response.status_code == 200:
            gap = 0
        elif not response.status_code:
            # This is a network error, back off lineraly 250ms up to 16s
            self.interval1 = gap = max(self.interval1+0.25, 16)
        elif response.status_code == 420:
            gap = 60 if not self.interval2 else 2*self.interval2
            self.interval2 = gap
        else:
            if response.status_code >= 400:
                self.logger.error('Could not connect to twitter spreaming API:'
                                  ' %d - %s',
                                  response.status_code,
                                  response.text().strip())
            gap = 5 if not self.interval3 else min(2*self.interval3, 320)
            self.interval3 = gap

        self.call_later(loop.time() + gap, self.connect, 'Reconnect in ')

    def get_param(self, name):
        value = self.cfg.get(name)
        if not value:
            raise pulsar.ImproperlyConfigured(
                'Please specify the "%s" parameter in your %s file' %
                (name, self.cfg.config))
        return value

    def call_later(self, t, callback, msg):
        next = 10
        loop = self._loop
        now = loop.time()
        gap = t - now
        if gap <= 0:
            callback()
        else:
            self.logger.info('%s in %d seconds', msg, round(gap) or 1)
            if gap < next:
                loop.call_at(t, callback)
            else:
                loop.call_later(next, self.call_later, t, callback, msg)


class PublishTweets:
    '''Tweets processor
    This callable class is passed to the :class:`.Twitter` application
    and it is invoked every time new messages are available.
    The callable method accept two parameters, the :class:`.Twitter`
    application and a list of messages received.
    '''
    store = None

    def __init__(self, channel):
        self.channel = channel

    def __call__(self, twitter, messages):
        if not self.store:
            self.store = create_store(twitter.cfg.data_store)
            self.pubsub = self.store.pubsub()
        ensure_future(self._publish(messages))

    async def _publish(self, messages):
        for message in messages:
            await self.pubsub.publish(self.channel, json.dumps(message))


def log_tweets(twitter, messages):
    for m in messages:
        print(m)
