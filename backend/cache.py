"""
SatyaCheck AI - Distributed Redis Cache
"""

import os
import logging
import json
import pickle
from typing import Any, Optional
import redis
from redis.exceptions import RedisError, ConnectionError

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_SSL = os.getenv('REDIS_SSL', 'false').lower() == 'true'

DEFAULT_TTL = 3600
CLAIM_TTL = 3600
PIB_SEARCH_TTL = 86400
WEB_SEARCH_TTL = 43200


class DistributedCache:
    """Distributed Redis cache for multi-instance scaling."""

    def __init__(self):
        self.redis_client = None
        self.enabled = False

        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                ssl=REDIS_SSL,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                max_connections=50,
                health_check_interval=30
            )

            self.redis_client.ping()
            self.enabled = True
            logger.info(f"Redis cache connected: {REDIS_HOST}:{REDIS_PORT}")

        except (RedisError, ConnectionError) as e:
            logger.warning(f"Redis unavailable: {e}. Running without cache.")
            self.redis_client = None
            self.enabled = False

        except Exception as e:
            logger.warning(f"Redis connection error: {e}. Running without cache.")
            self.redis_client = None
            self.enabled = False

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return json.dumps(value).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            return pickle.loads(data)
        except Exception:
            try:
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Deserialization error: {e}")
                return None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enabled or not self.redis_client:
            return None

        try:
            data = self.redis_client.get(key)
            if data:
                logger.debug(f"Cache HIT: {key}")
                return self._deserialize(data)
            else:
                logger.debug(f"Cache MISS: {key}")
                return None

        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """Set value in cache with TTL."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            data = self._serialize(value)
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            result = self.redis_client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return result > 0

        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            return self.redis_client.exists(key) > 0

        except Exception as e:
            logger.warning(f"Cache exists error for {key}: {e}")
            return False

    def get_many(self, keys: list) -> dict:
        """Get multiple values at once using pipeline."""
        if not self.enabled or not self.redis_client:
            return {}

        try:
            pipeline = self.redis_client.pipeline()
            for key in keys:
                pipeline.get(key)

            results = pipeline.execute()

            output = {}
            for key, data in zip(keys, results):
                if data:
                    output[key] = self._deserialize(data)

            logger.debug(f"Cache GET_MANY: {len(output)}/{len(keys)} hits")
            return output

        except Exception as e:
            logger.warning(f"Cache get_many error: {e}")
            return {}

    def set_many(self, items: dict, ttl: int = DEFAULT_TTL) -> bool:
        """Set multiple values at once using pipeline."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            pipeline = self.redis_client.pipeline()
            for key, value in items.items():
                data = self._serialize(value)
                pipeline.setex(key, ttl, data)

            pipeline.execute()
            logger.debug(f"Cache SET_MANY: {len(items)} items (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Cache set_many error: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self.enabled or not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cache CLEAR: {deleted} keys matching '{pattern}'")
                return deleted
            return 0

        except Exception as e:
            logger.warning(f"Cache clear_pattern error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled or not self.redis_client:
            return {
                'enabled': False,
                'status': 'disabled'
            }

        try:
            info = self.redis_client.info('stats')
            return {
                'enabled': True,
                'status': 'healthy',
                'total_commands': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(info),
                'memory_used': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0)
            }

        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {
                'enabled': True,
                'status': 'error',
                'error': str(e)
            }

    def _calculate_hit_rate(self, info: dict) -> str:
        """Calculate cache hit rate percentage."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses

        if total == 0:
            return "0.00%"

        hit_rate = (hits / total) * 100
        return f"{hit_rate:.2f}%"

    def health_check(self) -> bool:
        """Check if cache is healthy."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.ping()
            return True

        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            return False


_cache_instance = None


def get_cache() -> DistributedCache:
    """Get global cache instance."""
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = DistributedCache()

    return _cache_instance


def cache_claim_result(claim_hash: str, result: dict) -> bool:
    """Cache claim analysis result."""
    cache = get_cache()
    key = f"claim:{claim_hash}"
    return cache.set(key, result, ttl=CLAIM_TTL)


def get_cached_claim_result(claim_hash: str) -> Optional[dict]:
    """Get cached claim analysis result."""
    cache = get_cache()
    key = f"claim:{claim_hash}"
    return cache.get(key)


def cache_pib_search(query: str, results: list) -> bool:
    """Cache PIB search results."""
    cache = get_cache()
    key = f"pib:{query}"
    return cache.set(key, results, ttl=PIB_SEARCH_TTL)


def get_cached_pib_search(query: str) -> Optional[list]:
    """Get cached PIB search results."""
    cache = get_cache()
    key = f"pib:{query}"
    return cache.get(key)


def cache_web_search(query: str, results: list) -> bool:
    """Cache web search results."""
    cache = get_cache()
    key = f"web:{query}"
    return cache.set(key, results, ttl=WEB_SEARCH_TTL)


def get_cached_web_search(query: str) -> Optional[list]:
    """Get cached web search results."""
    cache = get_cache()
    key = f"web:{query}"
    return cache.get(key)
