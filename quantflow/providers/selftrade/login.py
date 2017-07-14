from lux.models import Schema, fields

LOGIN_URL = (
    "https://secure.selftrade.co.uk/rest/externalsecurity/checkFirstLoginP"
)


class SelfTradeLoginSchema(Schema):
    accountId = fields.Integer(required=True)
    pinPart1 = fields.Integer(required=True)
    pinPart2 = fields.Integer(required=True)
    pinPart3 = fields.Integer(required=True)


class SelftradeLogin:

    async def firstLogin(self, accountId):
        response = await self.http.get(
            LOGIN_URL,
            params=dict(
                accountId=accountId
            )
        )
        response.raise_for_status()
        text = response.text
