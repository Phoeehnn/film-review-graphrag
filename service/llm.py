import asyncio
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from .settings import Settings


@dataclass
class RetryBudget:
    used: int = 0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    network_retries: int = 0
    models: set = field(default_factory=set)
    fingerprints: set = field(default_factory=set)
    retry_budget: RetryBudget = field(default_factory=RetryBudget)

    def add(self, response):
        if response.model:
            self.models.add(response.model)
        if getattr(response, 'system_fingerprint', None):
            self.fingerprints.add(response.system_fingerprint)
        usage = getattr(response, 'usage', None)
        if usage:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0

    def dump(self):
        return {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens, 'total_tokens': self.total_tokens, 'calls': self.calls, 'network_retries': self.network_retries, 'models': sorted(self.models), 'system_fingerprints': sorted(self.fingerprints)}


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.deepseek_api_key or 'not-configured', base_url=settings.deepseek_base_url, max_retries=0, timeout=settings.model_timeout)

    async def close(self):
        await self.client.close()

    def arguments(self, messages, **kwargs):
        return {'model': self.settings.deepseek_model, 'messages': messages, 'temperature': self.settings.generation_temperature, 'max_tokens': self.settings.generation_max_tokens, 'extra_body': {'thinking': {'type': 'disabled'}}, **kwargs}

    async def complete(self, messages, usage: Usage, **kwargs):
        for attempt in range(2):
            try:
                usage.calls += 1
                response = await self.client.chat.completions.create(**self.arguments(messages, **kwargs))
                usage.add(response)
                return response
            except (APIConnectionError, APITimeoutError):
                if attempt or usage.retry_budget.used>=1:
                    raise
                usage.network_retries += 1
                usage.retry_budget.used += 1
                await asyncio.sleep(.2)

    async def stream(self, messages, usage: Usage):
        # Retry only before the first token, so a reconnect cannot duplicate content.
        started = False
        for attempt in range(2):
            stream = None
            try:
                usage.calls += 1
                stream = await self.client.chat.completions.create(**self.arguments(messages, stream=True, stream_options={'include_usage': True}))
                final_usage = None
                async for chunk in stream:
                    if chunk.model:
                        usage.models.add(chunk.model)
                    if getattr(chunk, 'system_fingerprint', None):
                        usage.fingerprints.add(chunk.system_fingerprint)
                    if chunk.usage:
                        final_usage = chunk
                    if chunk.choices and chunk.choices[0].delta.content:
                        started = True
                        yield chunk.choices[0].delta.content
                if final_usage:
                    usage.add(final_usage)
                return
            except (APIConnectionError, APITimeoutError):
                if attempt or started or usage.retry_budget.used>=1:
                    raise
                usage.network_retries += 1
                usage.retry_budget.used += 1
                await asyncio.sleep(.2)
            finally:
                if stream is not None:
                    await stream.close()

    async def ready(self):
        if not self.settings.deepseek_api_key:
            return {'ready': False, 'reason': '未配置模型密钥'}
        models = await self.client.models.list()
        ids = [m.id for m in models.data]
        return {'ready': self.settings.deepseek_model in ids, 'configured_model': self.settings.deepseek_model, 'available_models': ids}
