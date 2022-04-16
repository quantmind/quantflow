import json
import os
from typing import Any, Dict, Optional

from aiohttp import ClientResponse, ClientSession
from aiohttp.client_exceptions import ContentTypeError

ResponseType = dict | list


def compact(**kwargs: Any) -> dict:
    return {k: v for k, v in kwargs.items() if v}


class HttpResponseError(RuntimeError):
    def __init__(self, response: ClientResponse, data: dict) -> None:
        self.response = response
        self.data = data
        self.data["request_url"] = str(response.url)
        self.data["request_method"] = response.method
        self.data["response_status"] = response.status

    @property
    def status(self) -> int:
        return self.response.status

    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)


class HttpClient:
    session: Optional[ClientSession] = None
    user_agent: str = os.getenv("HTTP_USER_AGENT", "quantflow/data")
    content_type: str = "application/json"
    ResponseError: HttpResponseError = HttpResponseError
    ok_status = frozenset((200, 201))

    def get_session(self) -> ClientSession:
        if not self.session:
            self.session = ClientSession()
        return self.session

    async def close(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def get(self, url: str, **kwargs: Any) -> ResponseType:
        kwargs["method"] = "GET"
        return await self.execute(url, **kwargs)

    async def execute(
        self,
        url: str,
        *,
        method: str = "",
        headers: Optional[dict] = None,
        **kw: Any,
    ) -> ResponseType:
        session = self.get_session()
        _headers = self.default_headers()
        _headers.update(headers or ())
        method = method or "GET"
        response = await session.request(method, url, headers=_headers, **kw)
        if response.status in self.ok_status:
            return await self.response_data(response)
        elif response.status == 204:
            return {}
        else:
            return await self.response_error(response)

    def default_headers(self) -> Dict[str, str]:
        return {"user-agent": self.user_agent, "accept": self.content_type}

    def mock(self) -> None:
        pass

    @classmethod
    async def response_error(cls, response: ClientResponse) -> ResponseType:
        try:
            data = await cls.response_data(response)
        except ContentTypeError:
            data = dict(message=await response.text())
        raise cls.ResponseError(response, data)

    @classmethod
    async def response_data(cls, response: ClientResponse) -> ResponseType:
        return await response.json()