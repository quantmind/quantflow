import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Generic, TypeVar, cast
from fluid.utils import log
from fastapi import Depends, FastAPI, Request
from fluid.utils.redis import FluidRedis
from pydantic import BaseModel
from redis import Redis

M = TypeVar("M", bound=BaseModel)
logger = log.get_logger(__name__)


def instrument_app(app: FastAPI) -> None:
    """Instrument the app with the necessary dependencies"""
    redis = FluidRedis.create()
    app.state.redis = redis.redis_cli
    app.router.on_shutdown.append(redis.close)


def get_redis(request: Request) -> Redis:
    """Get the redis client from the app state"""
    if not hasattr(request.app.state, "redis"):
        raise RuntimeError("Redis client not found in app state")
    return cast(Redis, request.app.state.redis)


@dataclass
class RedisCache(Generic[M]):
    redis: Redis
    Model: type[M]
    key: str
    ttl: int = 60

    async def from_cache(self, loader: Callable[[], Awaitable[M]]) -> M:
        """Get a value from the cache"""
        value = await self.redis.get(self.key)
        if value is None:
            return await self.set_cache(await loader())
        try:
            return self.Model.model_validate_json(value)
        except json.JSONDecodeError:
            logger.exception(f"Failed to decode cache value for key {self.key}")
            return await self.set_cache(await loader())

    async def set_cache(self, value: M) -> M:
        """Set a value in the cache"""
        payload = value.model_dump_json()
        await self.redis.set(self.key, payload, ex=self.ttl)
        return value


RedisDep = Annotated[Redis, Depends(get_redis)]
