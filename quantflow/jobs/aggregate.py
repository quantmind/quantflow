from pq import api


@api.job()
async def aggregate_data(self):
    """Aggregate data from registered providers
    """
    for provider in self.providers:
        await provider.start()
