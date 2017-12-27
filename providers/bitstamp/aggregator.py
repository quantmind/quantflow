from quantflow.orderbook import flags

from ..consumer import Consumer
from .ws import BitstampWs
from .rest import BitstampRest, PAIRS


class Aggregator(Consumer):
    name = 'bitstamp.aggregator'
    beat = 2

    async def aggregate(self):
        http = self.manager.http()
        ws = BitstampWs(http=http)

        try:
            #await rest.order_book()
            # Register to bitstamp websocket events
            await ws.live_trades(LiveTrades('bitstamp', 'btcusd'))
            #await ws.diff_order_book(self._diff_order_book)
            #await ws.order_book(self._order_book)
        except OSError:
            await self.sleep(error_message='Could not connect')
        except Exception:
            self.logger.exception('Critical exception')
            await self.sleep()

    def _live_trades(self, trade, **kwargs):
        self.logger.debug('New XBTUSD trade')
        trade['volume'] = trade.pop('amount')
        trade['trade_id'] = trade.pop('id')
        return
        book = self.book(self.security)
        book.record(update_type=flags.TRADE, **trade)
        book.flush()
        self.publish('trade', {'exchange': self.name,
                               'security': self.security,
                               'trade': trade})

    def _order_book(self, book, **kwargs):
        self.logger.debug('New XBTUSD orders')
        self.publish('order_book', {'exchange': self.name,
                                    'security': self.security,
                                    'book': book})

    def _diff_order_book(self, data, **kwargs):
        if data['bids'] or data['asks']:
            book = self.book(self.security)
            for price, volume in data['bids']:
                book.record(update_type=models.ORDER,
                            side=models.BIDBUY,
                            price=float(price),
                            volume=float(volume))
            for price, volume in data['asks']:
                book.record(update_type=models.ORDER,
                            side=models.ASKSELL,
                            price=float(price),
                            volume=float(volume))
            book.flush()


class BitstampTrade(Trade):

    def trade(self, data):
        return dict(
            id=data['id'],
            volume=float(data['amount']),
            price=float(data['price']),
            timestamp=int(data['timestamp']),
            type=int(data['type'])
        )


class LiveTrades:

    def __init__(self, exchange, pair):
        self.exchange = exchange
        self.pair = pair

    def __call__(self, data, **kwargs):
        trade = self.trade(data)
        trades = self.trades(self.pair)
        trades.add(trade)
        trades.flush()
        self.publish('trade', {
            'exchange': self.exchange,
            'security': self.pair,
            'trade': trade
        })
