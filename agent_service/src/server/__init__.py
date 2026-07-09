from typing import Optional
from urllib.parse import urlparse, parse_qs

import aiomysql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from loguru import logger

from src.config.loader import get_bool_env, get_str_env, get_int_env


async def check(conn):
    await conn.execute("SELECT 1")


async def reset(conn):
    await conn.execute("ROLLBACK")


def parse_mysql_url(url: str) -> dict:
    u = urlparse(url)

    if u.scheme not in ("mysql", "mysql+aiomysql"):
        raise ValueError("invalid mysql url")

    query = parse_qs(u.query)

    return {
        "host": u.hostname,
        "port": u.port or 3306,
        "user": u.username,
        "password": u.password,
        "db": u.path.lstrip("/"),
        "charset": query.get("charset", ["utf8mb4"])[0],
        "autocommit": True,
    }


class CheckPointPoolManager:
    def __init__(self) -> None:
        self._pool = None
        self._checkpointer: Optional[BaseCheckpointSaver] = None

    async def init(self):
        # Initialize global connection pool based on configuration
        checkpoint_saver = get_bool_env("LANGGRAPH_CHECKPOINT_SAVER", False)
        checkpoint_url = get_str_env("LANGGRAPH_CHECKPOINT_DB_URL", "")

        if not checkpoint_saver or not checkpoint_url:
            logger.info(
                "Checkpoint saver not configured, skipping connection pool initialization"
            )
        else:
            pool_min_size = get_int_env("POOL_MIN_SIZE", 5)
            pool_max_size = get_int_env("POOL_MAX_SIZE", 20)
            # Initialize PostgreSQL connection pool
            if checkpoint_url.startswith("postgres://"):
                pool_timeout = get_int_env("PG_POOL_TIMEOUT", 60)
                pool_max_lifetime = get_int_env("PG_POOL_MAX_LIFETIME", 5 * 60)
                pool_max_idle = get_int_env("PG_POOL_MAX_IDLE", 4 * 60)

                connection_kwargs = {
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3,
                }

                logger.info(
                    f"Initializing global PostgreSQL connection pool: "
                    f"min_size={pool_min_size}, max_size={pool_max_size}, timeout={pool_timeout}s"
                )

                try:
                    self._pool = AsyncConnectionPool(
                        checkpoint_url,
                        kwargs=connection_kwargs,
                        min_size=pool_min_size,
                        max_size=pool_max_size,
                        timeout=pool_timeout,
                        max_idle=pool_max_idle,
                        max_lifetime=pool_max_lifetime,
                        check=check,
                        reset=reset,
                    )
                    await self._pool.open()

                    self._checkpointer = AsyncPostgresSaver(self._pool)
                    await self._checkpointer.setup()

                    logger.info(
                        "Global PostgreSQL connection pool initialized successfully"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize PostgreSQL connection pool: {e}"
                    )
                    self._pool = None
                    self._checkpointer = None
                    raise RuntimeError(
                        "Checkpoint persistence is explicitly configured with PostgreSQL, "
                        "but initialization failed. Application will not start."
                    ) from e
            # Initialize MySQLSQL connection pool
            elif checkpoint_url.startswith("mysql://"):
                pool_recycle = get_int_env("MYSQL_POOL_RECYCLE", 8 * 60 * 60)
                cfg = parse_mysql_url(checkpoint_url)
                try:
                    self._pool = await aiomysql.create_pool(
                        **cfg,
                        minsize=pool_min_size,
                        maxsize=pool_max_size,
                        pool_recycle=pool_recycle,
                    )
                    # await self._pool.open()
                    self._checkpointer = AIOMySQLSaver(self._pool)
                    await self._checkpointer.setup()

                    logger.info("Global MySQL connection pool initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize MySQL connection pool: {e}")
                    self._pool = None
                    self._checkpointer = None
                    raise RuntimeError(
                        "Checkpoint persistence is explicitly configured with MySQL, "
                        "but initialization failed. Application will not start."
                    ) from e
            else:
                logger.warning(f"Not match checkpoint_url: {checkpoint_url}")

    async def close(self):
        if self._pool:
            if isinstance(self._pool, AsyncConnectionPool):
                logger.info("Closing global PostgreSQL connection pool")
                await self._pool.close()
                logger.info("Global PostgreSQL connection pool closed")
            elif isinstance(self._pool, aiomysql.Pool):
                logger.info("Closing global MySQL connection pool")
                self._pool.close()
                await self._pool.wait_closed()
                logger.info("Global MySQL connection pool closed")


checkpointer_pool = CheckPointPoolManager()
