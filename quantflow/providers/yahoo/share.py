from .base import Base


class Share(Base):
    table = 'quotes'
    key = 'symbol'

    def _fetch(self):
        data = super(Share, self)._fetch()
        if data['LastTradeDate'] and data['LastTradeTime']:
            data[u'LastTradeDateTimeUTC'] = edt_to_utc('{0} {1}'.format(data['LastTradeDate'], data['LastTradeTime']))
        return data

    def get_price(self):
        return self.data_set['LastTradePriceOnly']

    def get_change(self):
        return self.data_set['Change']

    def get_percent_change(self):
        return self.data_set['PercentChange']

    def get_volume(self):
        return self.data_set['Volume']

    def get_prev_close(self):
        return self.data_set['PreviousClose']

    def get_open(self):
        return self.data_set['Open']

    def get_avg_daily_volume(self):
        return self.data_set['AverageDailyVolume']

    def get_stock_exchange(self):
        return self.data_set['StockExchange']

    def get_market_cap(self):
        return self.data_set['MarketCapitalization']

    def get_book_value(self):
        return self.data_set['BookValue']

    def get_ebitda(self):
        return self.data_set['EBITDA']

    def get_dividend_share(self):
        return self.data_set['DividendShare']

    def get_dividend_yield(self):
        return self.data_set['DividendYield']

    def get_earnings_share(self):
        return self.data_set['EarningsShare']

    def get_days_high(self):
        return self.data_set['DaysHigh']

    def get_days_low(self):
        return self.data_set['DaysLow']

    def get_year_high(self):
        return self.data_set['YearHigh']

    def get_year_low(self):
        return self.data_set['YearLow']

    def get_50day_moving_avg(self):
        return self.data_set['FiftydayMovingAverage']

    def get_200day_moving_avg(self):
        return self.data_set['TwoHundreddayMovingAverage']

    def get_price_earnings_ratio(self):
        return self.data_set['PERatio']

    def get_price_earnings_growth_ratio(self):
        return self.data_set['PEGRatio']

    def get_price_sales(self):
        return self.data_set['PriceSales']

    def get_price_book(self):
        return self.data_set['PriceBook']

    def get_short_ratio(self):
        return self.data_set['ShortRatio']

    def get_trade_datetime(self):
        return self.data_set['LastTradeDateTimeUTC']

    def get_name(self):
        return self.data_set['Name']

    def get_percent_change_from_year_high(self):
        return self.data_set['PercebtChangeFromYearHigh']  # spelling error in Yahoo API

    def get_change_from_50_day_moving_average(self):
        return self.data_set['ChangeFromFiftydayMovingAverage']

    def get_EPS_estimate_next_quarter(self):
        return self.data_set['EPSEstimateNextQuarter']

    def get_EPS_estimate_next_year(self):
        return self.data_set['EPSEstimateNextYear']

    def get_percent_change_from_200_day_moving_average(self):
        return self.data_set['PercentChangeFromTwoHundreddayMovingAverage']

    def get_change_from_year_low(self):
        return self.data_set['ChangeFromYearLow']

    def get_ex_dividend_date(self):
        return self.data_set['ExDividendDate']

    def get_change_from_year_high(self):
        return self.data_set['ChangeFromYearHigh']

    def get_EPS_estimate_current_year(self):
        return self.data_set['EPSEstimateCurrentYear']

    def get_price_EPS_estimate_next_year(self):
        return self.data_set['PriceEPSEstimateNextYear']

    def get_price_EPS_estimate_current_year(self):
        return self.data_set['PriceEPSEstimateCurrentYear']

    def get_one_yr_target_price(self):
        return self.data_set['OneyrTargetPrice']

    def get_change_percent_change(self):
        return self.data_set['Change_PercentChange']

    def get_dividend_pay_date(self):
        return self.data_set['DividendPayDate']

    def get_currency(self):
        return self.data_set['Currency']

    def get_days_range(self):
        return self.data_set['DaysRange']

    def get_percent_change_from_50_day_moving_average(self):
        return self.data_set['PercentChangeFromFiftydayMovingAverage']

    def get_last_trade_with_time(self):
        return self.data_set['LastTradeWithTime']

    def get_percent_change_from_year_low(self):
        return self.data_set['PercentChangeFromYearLow']

    def get_change_from_200_day_moving_average(self):
        return self.data_set['ChangeFromTwoHundreddayMovingAverage']

    def get_year_range(self):
        return self.data_set['YearRange']

    def get_historical(self, start_date, end_date):
        """
        Get Yahoo Finance Stock historical prices

        :param start_date: string date in format '2009-09-11'
        :param end_date: string date in format '2009-09-11'
        :return: list
        """
        hist = []
        for s, e in get_date_range(start_date, end_date):
            try:
                query = self.prepare_query('historicaldata', self.key, startDate=s, endDate=e)
                result = self._request(query)
                if isinstance(result, dict):
                    result = [result]
                hist.extend(result)
            except AttributeError:
                pass
        return hist
