from redis.asyncio import Redis
from loguru import logger

from src.config.loader import get_str_env, get_int_env


class AsyncRedisManager:
    def __init__(self):
        self.client: Redis

    async def init_connect(self):
        client = Redis(
            host=get_str_env("REDIS_HOST"),
            port=get_int_env("REDIS_PORT"),
            db=get_int_env("REDIS_DB"),
            password=get_str_env("REDIS_PASSWORD"),
            decode_responses=True,
            # read_from_replicas=True,
            # max_connections=10,
        )

        await client.initialize()
        logger.info("redis connect init finish")
        self.client = client

    async def get_client(self) -> Redis:
        if not self.client:
            await self.init_connect()
        return self.client

    async def close_all(self):
        await self.client.close()
        logger.info("redis connect close finish")


async_redis_manager = AsyncRedisManager()
