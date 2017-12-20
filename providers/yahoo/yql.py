from datetime import datetime, timedelta


# Yahoo! YQL API
PUBLIC_API_URL = 'https://query.yahooapis.com/v1/public/yql'
OAUTH_API_URL = 'https://query.yahooapis.com/v1/yql'
DATATABLES_URL = 'store://datatables.org/alltableswithkeys'

QUERY = 'select * from yahoo.finance.{table} where {key} = "{symbol}"'


class YQLError(Exception):
    pass


class YQLQueryError(YQLError):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Query failed with error: "%s".' % repr(self.value)


class Yql:
    """Yahoo base ticker
    """
    table = ''
    key = ''
    data = None
    ymap = {}

    def __init__(self, yahoo, symbol):
        self.yahoo = yahoo
        self.symbol = symbol
        self.refresh({})

    async def quote(self):
        self.refresh(await self.execute())
        return self

    async def execute(self, **kwargs):
        yql = self.query(**kwargs)
        res = await self.yahoo.http.get(
            PUBLIC_API_URL,
            params=dict(
                q=yql,
                format='json',
                env=DATATABLES_URL
            )
        )
        res.raise_for_status()
        data = res.json()
        try:
            return data['query']['results']['quote']
        except KeyError:
            try:
                raise YQLQueryError(res['error']['description']) from None
            except KeyError:
                raise YQLError()

    def query(self, table=None, key=None, **kwargs):
        """Build YQL query
        """
        query = QUERY.format(
            symbol=self.symbol,
            table=table or self.table,
            key=key or self.key
        )
        if kwargs:
            query = '%s%s' % (
                query,
                ''.join(' and {0}="{1}"'.format(k, v)
                        for k, v in kwargs.items())
            )
        return query

    def refresh(self, data):
        for attr, key in self.ymap.items():
            setattr(self, attr, data.get(key))
