import json
import os
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientResponse, ClientSession
from aiohttp.client_exceptions import ContentTypeError

ResponseType = dict | list


def compact(**kwargs: Any) -> dict:
    return {k: v for k, v in kwargs.items() if v}


class HttpResponseError(RuntimeError):
    def __init__(self, response: ClientResponse, data: ResponseType) -> None:
        self.response = response
        self.data: dict[str, Any] = data if isinstance(data, dict) else {"data": data}
        self.data["request_url"] = str(response.url)
        self.data["request_method"] = response.method
        self.data["response_status"] = response.status

    @property
    def status(self) -> int:
        return self.response.status

    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)


@dataclass
class HttpClient:
    session: ClientSession | None = None
    user_agent: str = os.getenv("HTTP_USER_AGENT", "quantflow/data")
    content_type: str = "application/json"
    session_owner: bool = False
    ResponseError: type[HttpResponseError] = HttpResponseError
    ok_status: frozenset = frozenset((200, 201))

    def new_session(self, **kwargs: Any) -> ClientSession:
        return ClientSession(**kwargs)

    def get_session(self) -> ClientSession:
        if not self.session:
            self.session_owner = True
            self.session = ClientSession()
        return self.session

    async def close(self) -> None:
        if self.session and self.session_owner:
            await self.session.close()
            self.session = None

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def get(self, url: str, **kwargs: Any) -> ResponseType:
        return await self.request("GET", url, **kwargs)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict | None = None,
        **kw: Any,
    ) -> ResponseType:
        session = self.get_session()
        _headers = self.default_headers()
        _headers.update(headers or ())
        response = await session.request(method, url, headers=_headers, **kw)
        if response.status in self.ok_status:
            return await self.response_data(response)
        elif response.status == 204:
            return {}
        else:
            data = await self.response_error(response)
            raise self.ResponseError(response, data)

    def default_headers(self) -> dict[str, str]:
        return {"user-agent": self.user_agent, "accept": self.content_type}

    @classmethod
    async def response_error(cls, response: ClientResponse) -> ResponseType:
        try:
            return await cls.response_data(response)
        except ContentTypeError:
            return dict(message=await response.text())

    @classmethod
    async def response_data(cls, response: ClientResponse) -> ResponseType:
        return await response.json()
