from collections.abc import Generator
from contextlib import contextmanager
import logging
import os

import redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

logger = logging.getLogger(__name__)


class RedisManager:
    """Centralized Redis connection management."""

    _pool: redis.ConnectionPool | None = None

    @classmethod
    def get_pool(cls) -> redis.ConnectionPool | None:
        """Get or create Redis connection pool."""
        if cls._pool is None:
            try:
                cls._pool = redis.ConnectionPool(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD"),
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    max_connections=20,
                    health_check_interval=30,
                )
                # Test connection
                with redis.Redis(connection_pool=cls._pool) as client:
                    client.ping()
                logger.info("Redis connection pool initialized successfully")
            except (ConnectionError, TimeoutError, RedisError) as e:
                logger.warning(f"Redis connection failed: {e}")
                cls._pool = None
            except Exception as e:
                logger.error(f"Unexpected error initializing Redis: {e}")
                cls._pool = None
        return cls._pool

    @classmethod
    @contextmanager
    def get_client(cls) -> Generator[redis.Redis | None, None, None]:
        """Context manager for Redis client."""
        pool = cls.get_pool()
        if pool is None:
            yield None
            return

        client = None
        try:
            client = redis.Redis(connection_pool=pool)
            yield client
        except (ConnectionError, TimeoutError, RedisError) as e:
            logger.warning(f"Redis operation failed: {e}")
            yield None
        except Exception as e:
            logger.error(f"Unexpected Redis error: {e}")
            yield None
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    @classmethod
    def close(cls):
        """Close Redis connection pool."""
        if cls._pool:
            try:
                cls._pool.disconnect()
                cls._pool = None
                logger.info("Redis connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing Redis pool: {e}")
