import asyncio

from pulsar.apps.http import HttpClient


ERROR_CMD = b'ERROR'
SYNC_ERROR_CMD = b'SYNC ERROR'
LOOP_CMD = b'LOOP'
END_CMD = b'END'
OK_CMD = b'OK'
PROBE_CMD = b'PROBE'
PREAMBLE = b'Preamble'


class LightstreamerError(ValueError):
    pass


class LightstreamerSyncError(LightstreamerError):
    pass


class IgStream:

    def __init__(self, endpoint, http=None):
        self.endpoint = endpoint
        self.http = http or HttpClient()
        self._worker = None
        self._session = {}
        self._subscriptions = {}
        self._listeners = []

    def add_listener(self, listener):
        self._listeners.append(listener)

    async def connect(self, identifier, cst, xst, adapter_set=''):
        password = 'CST-%s|XST-%s' % (cst, xst)
        response = await self.request(
            'lightstreamer/create_session.txt',
            stream=True,
            data={
                "LS_op2": 'create',
                "LS_cid": 'mgQkwtwdysogQz2BJ4Ji kOj2Bg',
                # "LS_adapter_set": adapter_set,
                "LS_user": identifier,
                "LS_password": password
            }
        )
        self._stream = response.raw
        self._worker = self.http._loop.create_task(self._read_stream())

    async def close(self):
        if self._worker and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._worker = None

    async def request(self, path, **kwargs):
        url = '%s/%s' % (self.endpoint, path)
        response = await self.http.post(url, **kwargs)
        response.raise_for_status()
        return response

    async def _read_stream(self):
        parse_session = False
        async for chunk in self._stream:
            for message in chunk.split(b'\r\n'):
                message = message.rstrip()

                if message is None:
                    break

                elif message == OK_CMD:
                    parse_session = True

                elif message == PROBE_CMD:
                    parse_session = False
                    continue

                elif message.startswith(ERROR_CMD):
                    raise LightstreamerError(message.decode('utf-8'))

                elif message.startswith(SYNC_ERROR_CMD):
                    raise LightstreamerSyncError(message.decode('utf-8'))

                elif message.startswith(LOOP_CMD):
                    break

                elif message.startswith(END_CMD):
                    break

                elif message.startswith(PREAMBLE):
                    parse_session = False
                    continue

                elif parse_session:

                    session_key, session_value = message.split(b':', 1)
                    self._session[session_key] = session_value

                elif message:

                    tok = message.split(b',', 1)
                    table, item = int(tok[0]), tok[1]
                    if table in self._subscriptions:
                        self._subscriptions[table].notifyupdate(item)
