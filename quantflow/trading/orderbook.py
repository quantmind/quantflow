from collection import Mapping


class OrderBook:

    def __init__(self):
        self.bids = {}
        self.asks = {}

    def update(self, book):
        if isinstance(book, OrderBook):
            self._update(self.bids, book.bids)
            self._update(self.asks, book.asks)
        elif isinstance(book, Mapping):
            self._update(self.bids, book.get('bids'))
            self._update(self.asks, book.get('asks'))
        else:
            raise ValueError('Could not update order book')

    def _update(self, data, newdata):
        if not newdata:
            return
        for price, size in newdata:
            pass
