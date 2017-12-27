

def bitstamp_pairs(ccys):
    yield 'eurusd'
    for ccy in ccys:
        yield '%seur' % ccy
        yield '%susd' % ccy
        if ccy != 'btc':
            yield '%sbtc' % ccy
