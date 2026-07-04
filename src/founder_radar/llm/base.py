"""Abstract LLM provider.

Every language-model interaction in Founder Radar goes through an instance
of `BaseLLMProvider`. Concrete providers (`OpenAIProvider`,
`AnthropicProvider`, ...) implement `complete()` which takes a prompt and
returns a text response.

Why a thin abstraction (not the official `openai` SDK everywhere)?
  - Phase 1 doesn't even call an LLM. By defining the interface now, Phase 3
    can plug in without touching other modules.
  - We support *any* OpenAI-compatible endpoint (LM Studio, Ollama with
    openai shim, vLLM, Together, Groq, ...) by changing one setting.
  - Tests can swap a `FakeLLMProvider` that returns canned responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class LLMMessage:
    """A single role-tagged chunk of a prompt.

    We model the minimal shape (role + content) rather than the full
    OpenAI/Anthropic message schema. Providers translate as needed.
    """

    role: str  # "system", "user", "assistant"
    content: str


@dataclass(slots=True)
class LLMResponse:
    """The provider's reply to a `complete()` call."""

    content: str
    model: str
    # Token accounting is optional and provider-dependent. We leave it as
    # an opaque dict so providers can add fields without breaking callers.
    usage: dict | None = None


class BaseLLMProvider(ABC):
    """Interface every concrete LLM provider must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'openai-compatible')."""

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send `messages` to the model and return its reply.

        Implementations should raise on network / auth errors with a
        message that includes the provider name so logs are searchable.
        """