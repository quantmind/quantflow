import asyncio
import abc

from quantflow.orderbook import store

from pq import api


class SimpleBackOff:

    def __init__(self, min_value=None, max_value=None, multiplier=None):
        self.min_value = min_value or 1
        self.max_value = max_value or 60
        self.multiplier = max(multiplier or 2, 1.01)
        self.value = None

    def reset(self):
        self.value = None

    def next(self):
        if self.value is None:
            self.value = self.min_value
        else:
            self.value = min(self.multiplier*self.value, self.max_value)
        return self.value


class Consumer(api.ConsumerAPI):
    backoff = SimpleBackOff()

    def start(self, _):
        self._worker = self._loop.create_task(
            self.aggregate()
        ).add_done_callback(self._done)

    @abc.abstractmethod
    async def aggregate(self):
        """Aggregator coroutine
        """

    async def publish(self, channel, message):
        try:
            await self.backend.publish(channel, message)
        except ConnectionError:
            self.logger.error(
                'cannot publish in %s channel, no connection', channel
            )
        except Exception:
            self.logger.exception(
                'Critical exception while publishing in channel %s', channel
            )

    async def sleep(self, delay=None, error_message=None):
        if not delay:
            delay = self.backoff.next()
        else:
            self.backoff.reset()
        if error_message:
            self.logger.error(
                '%s - retrying in %s seconds', error_message, delay
            )
        await asyncio.sleep(delay)

    def close(self):
        if self._worker and not self._worker.done():
            worker = self._worker
            self._worker = None
            worker.cancel()
            return worker

    def _done(self, fut):
        if not fut.cancelled():
            exc = fut.exception()
            if exc:
                self.logger.exception(
                    'Critical exception',
                    exc_info=(exc.__class__, exc, exc.__traceback__)
                )


class Trade:

    def __init__(self, exchange, security):
        self.exchange = exchange
        self.security = security

    def __call__(self, data, **kwargs):
        trade = self.trade(data)
        trades = store(self.security, self.exchange, 'trade', mode='w')
        trades.add(trade)
        trades.flush()
        self.publish('trade', {
            'exchange': self.exchange,
            'security': self.security,
            'trade': trade
        })

    def trades(self):
        pass
