
QUERY = 'select * from yahoo.finance.{table} where {key} = "{symbol}"'


class Base:
    table = ''
    key = ''

    def __init__(self, yahoo, symbol):
        self.yahoo = yahoo
        self.symbol = symbol
        self.data = {}

    @classmethod
    async def get(cls, yahoo, symbol):
        symbol = cls(yahoo, symbol)
        await symbol.execute()
        return symbol

    async def execute(self, **kwargs):
        query = self.prepare_query(**kwargs)
        data = await self.yahoo.execute(query)
        return data

    def prepare_query(self, table=None, key=None, **kwargs):
        """
        Simple YQL query bulder

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
