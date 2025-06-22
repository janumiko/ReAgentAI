import logging
import pickle

from aizynthfinder.context.config import Configuration

from src.reagentai.common.utils.redis import RedisManager
from src.reagentai.models.retrosynthesis import RouteCollection

logger = logging.getLogger(__name__)


class RetrosynthesisCache:
    """
    A cache for storing retrosynthesis routes based on target SMILES strings.
    Supports both in-memory and Redis backends with automatic fallback.
    """

    # Class-level cache for fast access
    _memory_cache: dict[str, RouteCollection] = {}
    finder_config: Configuration | None = None
    _cache_prefix = "retrosynthesis"
    _default_ttl = 86400  # 24 hours

    @classmethod
    def _serialize_data(cls, data: RouteCollection) -> bytes:
        """Serialize RouteCollection for Redis storage."""
        try:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            raise

    @classmethod
    def _deserialize_data(cls, data: bytes) -> RouteCollection:
        """Deserialize RouteCollection from Redis storage."""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise

    @classmethod
    def _get_cache_key(cls, target_smile: str) -> str:
        """Generate standardized cache key."""
        normalized_smile = target_smile.strip().lower()
        return f"{cls._cache_prefix}:{normalized_smile}"

    @classmethod
    def add(cls, target_smile: str, data: RouteCollection, ttl: int | None = None) -> bool:
        """Add route collection to cache."""
        if not target_smile or not data:
            logger.warning("Invalid input for cache add operation")
            return False

        # Always store in memory cache
        cls._memory_cache[target_smile] = data

        # Attempt Redis storage
        ttl = ttl or cls._default_ttl
        cache_key = cls._get_cache_key(target_smile)

        with RedisManager.get_client() as redis_client:
            if redis_client:
                try:
                    serialized_data = cls._serialize_data(data)
                    result = redis_client.setex(cache_key, ttl, serialized_data)
                    if result:
                        logger.debug(f"Cached to Redis: {cache_key}")
                        return True
                except Exception as e:
                    logger.warning(f"Failed to cache to Redis: {e}")

        logger.debug(f"Cached to memory only: {target_smile}")
        return True

    @classmethod
    def get(cls, target_smile: str) -> RouteCollection | None:
        """Retrieve route collection from cache."""
        if not target_smile:
            return None

        # Check memory cache first
        if target_smile in cls._memory_cache:
            logger.debug(f"Cache hit (memory): {target_smile}")
            return cls._memory_cache[target_smile]

        # Check Redis cache
        cache_key = cls._get_cache_key(target_smile)

        with RedisManager.get_client() as redis_client:
            if redis_client:
                try:
                    cached_data = redis_client.get(cache_key)
                    if cached_data and isinstance(cached_data, bytes):
                        data = cls._deserialize_data(cached_data)
                        cls._memory_cache[target_smile] = data
                        logger.debug(f"Cache hit (Redis): {target_smile}")
                        return data
                except Exception as e:
                    logger.warning(f"Failed to retrieve from Redis: {e}")

        logger.debug(f"Cache miss: {target_smile}")
        return None

    @classmethod
    def delete(cls, target_smile: str) -> bool:
        """Delete specific entry from cache."""
        if not target_smile:
            return False

        cls._memory_cache.pop(target_smile, None)
        cache_key = cls._get_cache_key(target_smile)

        with RedisManager.get_client() as redis_client:
            if redis_client:
                try:
                    result = redis_client.delete(cache_key)
                    logger.debug(f"Deleted from cache: {target_smile}")
                    return bool(result)
                except Exception as e:
                    logger.warning(f"Failed to delete from Redis: {e}")

        return True

    @classmethod
    def clear(cls) -> bool:
        """Clear all cached routes."""
        cls._memory_cache.clear()

        with RedisManager.get_client() as redis_client:
            if redis_client:
                try:
                    pipeline = redis_client.pipeline()
                    for key in redis_client.scan_iter(match=f"{cls._cache_prefix}:*", count=100):
                        pipeline.delete(key)
                    pipeline.execute()
                    logger.info("Cleared Redis cache")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to clear Redis cache: {e}")

        logger.info("Cleared memory cache")
        return True

    @classmethod
    def close(cls):
        """Close Redis connections and cleanup resources."""
        RedisManager.close()
