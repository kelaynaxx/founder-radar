"""LLM provider abstraction.

All language-model calls go through `BaseLLMProvider`. The OpenAI-compatible
implementation is the only concrete provider for now, but a future
`AnthropicProvider`, `OllamaProvider`, or `LocalProvider` can be dropped in
without touching the rest of the codebase.

Phase 1 does not actually call an LLM, but the abstraction is defined here
so Phase 3 doesn't require a refactor.
"""
from founder_radar.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from founder_radar.llm.openai_provider import OpenAICompatibleProvider

__all__ = [
    "BaseLLMProvider",
    "LLMMessage",
    "LLMResponse",
    "OpenAICompatibleProvider",
]