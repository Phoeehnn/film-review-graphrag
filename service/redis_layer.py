"""Optional Redis cache and fixed-window IP rate limit for the JSON query API."""
import hashlib
import json
import time
from uuid import uuid4

import redis.asyncio as redis

from .schemas import QueryResult
from .version import SERVICE_VERSION, runtime_source_hashes


RATE_SCRIPT = '''
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1])) end
return n
'''


class RedisLayer:
    def __init__(self, settings):
        self.settings = settings
        self.enabled = settings.cache_enabled or settings.rate_limit_enabled
        self.client = redis.from_url(settings.redis_url, decode_responses=True,
                                     socket_connect_timeout=.5, socket_timeout=.5) if self.enabled else None
        self.source_fingerprint = hashlib.sha256(json.dumps(runtime_source_hashes(), sort_keys=True).encode()).hexdigest()

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def ready(self):
        if not self.enabled:
            return {'ready': True, 'enabled': False}
        try:
            return {'ready': bool(await self.client.ping()), 'enabled': True,
                    'cache_enabled': self.settings.cache_enabled,
                    'rate_limit_enabled': self.settings.rate_limit_enabled}
        except redis.RedisError:
            return {'ready': False, 'enabled': True, 'reason': 'Redis 不可用'}

    def cache_key(self, request, dataset_fingerprint):
        body = {'version': SERVICE_VERSION, 'source': self.source_fingerprint,
                'dataset': dataset_fingerprint, 'request': request.model_dump(mode='json')}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':')).encode('utf-8')).hexdigest()
        return 'filmreview:query:' + digest

    async def get(self, key):
        return await self.client.get(key)

    async def set(self, key, result):
        await self.client.set(key, result.model_dump_json(), ex=self.settings.cache_ttl_seconds)

    async def allow(self, client_ip):
        digest = hashlib.sha256(client_ip.encode()).hexdigest()[:24]
        key = f'filmreview:rate:{digest}:{int(time.time()) // 60}'
        count = await self.client.eval(RATE_SCRIPT, 1, key, 120)
        return count <= self.settings.rate_limit_per_minute


def replay_cached(raw, started):
    previous = QueryResult.model_validate_json(raw)
    origin = previous.model_usage
    return previous.model_copy(update={
        'request_id': str(uuid4()), 'elapsed_seconds': round(time.monotonic() - started, 4),
        'model_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
                        'calls': 0, 'network_retries': 0, 'models': [], 'system_fingerprints': []},
        'stage_timings': {'redis_cache': round(time.monotonic() - started, 4)},
        'cache': {'hit': True, 'origin_request_id': previous.request_id,
                  'origin_model_usage': origin},
    })
