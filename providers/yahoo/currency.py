from .yql import Yql


class Currency(Yql):
    _table = 'xchange'
    _key = 'pair'

    def _fetch(self):
        data = super(Currency, self)._fetch()
        if data['Date'] and data['Time']:
            data[u'DateTimeUTC'] = edt_to_utc('{0} {1}'.format(data['Date'], data['Time']))
        return data

    def get_bid(self):
        return self.data_set['Bid']

    def get_ask(self):
        return self.data_set['Ask']

    def get_rate(self):
        return self.data_set['Rate']

    def get_trade_datetime(self):
        return self.data_set['DateTimeUTC']
