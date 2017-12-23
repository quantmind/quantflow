import re
import time
import json
import hmac
import hashlib
import asyncio
import logging
from urllib.parse import quote, urlsplit

from pulsar.apps.http import HttpClient
from pulsar.apps.ws import WS


channel_name_re = re.compile('^[-a-zA-Z0-9_=@,.;]+$')
app_id_re = re.compile('^[0-9]+$')
PUSHER_ERROR = 'pusher:error'
PUSHER_CONNECTION = 'pusher:connection_established'
PUSHER_SUBSCRIBE = 'pusher:subscribe'
PUSHER_SUBSCRIBED = 'pusher_internal:subscription_succeeded'


def pusher_from_url(url):
    p = urlsplit(url)
    if not p.path.startswith("/apps/"):
        raise ValueError("invalid URL path")
    return Pusher(app_id=p.path[6:], key=p.username, secret=p.password,
                  host=p.hostname, port=p.port, secure=p.scheme == 'https')


class PusherError(Exception):

    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class PusherChannel(object):

    def __init__(self, name, pusher):
        self._registered_callbacks = {}
        self.pusher = pusher
        self.name = str(name)
        if not channel_name_re.match(self.name):
            raise NameError("Invalid channel id: %s" % self.name)
        self.path = '/apps/%s/channels/%s/events' % (self.pusher.app_id,
                                                     quote(self.name))

    async def trigger(self, event, data=None, socket_id=None):
        '''Trigger an ``event`` on this channel
        '''
        json_data = json.dumps(data, cls=self.pusher.encoder)
        query_string = self.signed_query(event, json_data, socket_id)
        signed_path = "%s?%s" % (self.path, query_string)
        pusher = self.pusher
        absolute_url = pusher.get_absolute_path(signed_path)
        response = await pusher.http.post(
            absolute_url, data=json_data,
            headers=[('Content-Type', 'application/json')])
        response.raise_for_status()
        return response.status_code == 202

    def bind(self, event, callback):
        '''Bind to an ``event`` in this channel. The ``callback`` function is
        executed every time the event is triggered.
        '''
        self._registered_callbacks[event] = callback
        return True

    def signed_query(self, event, json_data, socket_id):
        pusher = self.pusher
        query_string = pusher.compose_querystring(event, json_data, socket_id)
        string_to_sign = "POST\n%s\n%s" % (self.path, query_string)
        signature = pusher.sign(string_to_sign.encode('utf-8'))
        return "%s&auth_signature=%s" % (query_string, signature)

    def _event(self, event, data):
        try:
            callback = self._registered_callbacks.get(event)
            if callback:
                callback(data, event=event)
        except Exception:
            self.pusher.logger.exception(
                'Unhandled exception during event "%s" on channel "%s"',
                self.name, event)


class Pusher(WS):
    channel_type = PusherChannel
    host = 'api.pusherapp.com'
    wshost = 'ws.pusherapp.com'
    client = 'py-pulsarPusher'
    protocol = 7

    def __init__(self, app_id=None, key=None, secret=None, host=None,
                 port=None, encoder=None, secure=False, http=None):
        self.key = key or ''
        self.app_id = app_id or ''
        if self.app_id and not app_id_re.match(self.app_id):
            raise NameError('Invalid app id "%s"' % self.app_id)
        self.secret = secret or ''
        self.host = host or self.host
        self.port = port or (443 if secure else 80)
        self.secure = secure
        self.encoder = encoder
        self.logger = logging.getLogger('pulsar.pusher')
        self.http = http or HttpClient()
        self._consumer = None
        self._waiter = None
        self._channels = {}

    def __getitem__(self, name):
        if name not in self._channels:
            self._channels[name] = self.channel_type(name, self)
        return self._channels[name]

    async def connect(self):
        '''Connect to a Pusher websocket
        '''
        if not self._consumer:
            try:
                address = self._websocket_host()
                self.logger.info('Connect to %s', address)
                self._consumer = await self.http.get(
                    address, websocket_handler=self
                )
                if self._consumer.status_code != 101:
                    raise PusherError("Could not connect to websocket")
            except Exception as exc:
                waiter.set_exception(exc)
                raise
        return self._consumer

    async def subscribe(self, channel, data=None, auth=None):
        msg = {'channel': channel}
        if auth:
            pass
        if data:
            msg['channel_data'] = data
        channel = self[channel]
        await self.execute(PUSHER_SUBSCRIBE, msg)
        return channel

    # INTERNALS
    async def execute(self, event, data):
        websocket = await self.connect()
        websocket.write(json.dumps({'event': event,
                                    'data': data}))

    def on_message(self, websocket, message):
        '''Handle websocket incoming messages
        '''
        encoded = json.loads(message)
        event = encoded.get('event')
        channel = encoded.get('channel')
        data = json.loads(encoded.get('data'))
        try:
            if event == PUSHER_ERROR:
                raise PusherError(data['message'], data['code'])
            elif event == PUSHER_CONNECTION:
                self.socket_id = data.get('socket_id')
                self.logger.info('Succesfully connected on socket %s',
                                 self.socket_id)
            elif event == PUSHER_SUBSCRIBED:
                self.logger.info('Succesfully subscribed to %s',
                                 encoded.get('channel'))
            elif channel:
                self[channel]._event(event, data)
        except Exception:
            self.logger.exception('pusher error')

    def on_close(self, websocket):
        self.logger.warning('Disconnected from pusher')

    def sign(self, message):
        return hmac.new(self.secret.encode('utf-8'),
                        message, hashlib.sha256).hexdigest()

    def compose_querystring(self, event, json_data, socket_id):
        hasher = hashlib.md5()
        hasher.update(json_data.encode('UTF-8'))
        ret = ('auth_key=%s&auth_timestamp=%s&auth_version=1.0&'
               'body_md5=%s&name=%s' %
               (self.key, int(time.time()), hasher.hexdigest(), event))
        if socket_id:
            ret += "&socket_id=%s" % socket_id
        return ret

    def authenticate(self, socket_id, custom_data=None):
        if custom_data:
            custom_data = json.dumps(custom_data, cls=self.pusher.encoder)

        auth = self.authentication_string(socket_id, custom_data)
        r = {'auth': auth}

        if custom_data:
            r['channel_data'] = custom_data

        return r

    def authentication_string(self, socket_id, custom_string=None):
        if not socket_id:
            raise Exception("Invalid socket_id")

        string_to_sign = "%s:%s" % (socket_id, self.name)

        if custom_string:
            string_to_sign += ":%s" % custom_string

        signature = self.pusher.sign(string_to_sign.encode('utf-8'))

        return "%s:%s" % (self.pusher.key, signature)

    def get_absolute_path(self, signed_path):
        scheme = 'https' if self.secure else 'http'
        return '%s://%s%s' % (scheme, self.host, signed_path)

    def _websocket_host(self, protocol=None):
        scheme = 'wss' if self.secure else 'ws'
        protocol = protocol or self.protocol
        return ('%s://%s/app/%s?client=%s&protocol=%s' %
                (scheme, self.wshost, self.key, self.client, protocol))
