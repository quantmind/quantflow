from ..consumer import Consumer
from .api import TrueFX


class Aggregator(Consumer):
    name = 'truefx.aggregator'
    session = None
    _worker = None

    async def aggregate(self):
        truefx = TrueFX(http=self.manager.http())
        self.logger.info('start aggregating forex data from truefx')
        session = None
        while True:
            try:
                if session is None:
                    session = await truefx.intraday()
                    await self.publish('truefx', session.data)
                    await self.sleep(0.5)

                data = await session.refresh()
                if data:
                    await self.publish('truefx', data)
            except Exception as exc:
                self.logger.exception('Could not get data from truefx')
                await self.sleep()
                session = None
            else:
                await self.sleep(0.5)
