


class BitCoin(DataFeed):
    root_url = 'https://www.bitstamp.net/api/'

    def ticker(self):
        return self.http.get(self.url('ticker/'))
