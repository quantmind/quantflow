import re
import time
from io import StringIO
from datetime import date, timedelta

import pandas as pd

from .yql import Yql


DEFAULT_WINDOW = timedelta(days=365)
HISTORY_DATA_URL = 'https://query1.finance.yahoo.com/v7/finance/download/%s'
HISTORY_PAGE_URL = 'https://finance.yahoo.com/quote/%s/history'


class Share(Yql):
    table = 'quotes'
    key = 'symbol'
    _crumb = None
    ymap = dict(
        price='LastTradePriceOnly',
        change='Change',
        percent_change='PercentChange',
        volume='Volume',
        prev_close='PreviousClose',
        open='Open',
        avg_daily_volume='AverageDailyVolume',
        stock_exchange='StockExchange',
        market_cap='MarketCapitalization',
        book_value='BookValue',
        ebitda='EBITDA',
        dividend_share='DividendShare',
        dividend_yield='DividendYield',
        earnings_share='EarningsShare',
        days_high='DaysHigh',
        days_low='DaysLow',
        year_high='YearHigh',
        year_low='YearLow',
        moving_avg_50='FiftydayMovingAverage',
        moving_avg_200='TwoHundreddayMovingAverage',
        pe='PERatio',
        peg='PEGRatio',  # Price eraning Growth Ratio,
        ps='PriceSales',
        pb='PriceBook',
    )

    async def prices(self, startdate=None, enddate=None, interval=None):
        return await self._history(startdate, enddate, interval, 'history')

    async def dividends(self, startdate=None, enddate=None, interval=None):
        return await self._history(startdate, enddate, interval, 'div')

    async def _history(self, startdate, enddate, interval, events):
        enddate = enddate or date.today()
        startdate = startdate or enddate - DEFAULT_WINDOW
        if not self._crumb:
            self._crumb = await self._get_crumb()
        params = dict(
            period1=int(time.mktime(startdate.timetuple())),
            period2=int(time.mktime(enddate.timetuple())),
            interval=interval or '1d',
            events=events,
            crumb=self._crumb
        )
        response = await self.yahoo.http.get(
            HISTORY_DATA_URL % self.symbol, params=params
        )
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))

    async def _get_crumb(self):
        # Scrape a history page for a valid crumb ID:
        response = await self.yahoo.http.get(HISTORY_PAGE_URL % self.symbol)
        out = response.text
        # Matches: {"crumb":"AlphaNumeric"}
        rpat = '"CrumbStore":{"crumb":"([^"]+)"}'
        crumb = re.findall(rpat, out)[0]
        return crumb.encode('ascii').decode('unicode-escape')

