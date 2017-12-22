from pulsar.apps.http import HttpClient


class MorningStar:
    url = "https://api.morningstar.com/v2"

    def __init__(self, http=None):
        self.http = http or HttpClient()

    async def auth(self, days=None):
        days = days or 90
        url = "%s/service/account/CreateAccesscode/%dd" % (self.url, days)
        response = await self.http.post(
            url, data=dict(
                # accountcode=self.accountcode,
                # password=self.password
            )
        )
        response.raise_for_status()
        return response
