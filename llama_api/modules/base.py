from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Iterator, List, TypeVar

from ..mixins.completion import CompletionMixin
from ..mixins.interrupt import InterruptMixin
from ..mixins.lock import LockMixin
from ..mixins.logits import LogitsMixin
from ..mixins.prompt_utils import PromptUtilsMixin
from ..schemas.api import (
    APIChatMessage,
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    TextGenerationSettings,
)
from ..utils.logger import ApiLogger

T = TypeVar("T")
logger = ApiLogger(__name__)


@dataclass
class BaseLLMModel:
    model_path: str = "/path/to/model"
    max_total_tokens: int = 2048

    @property
    def asdict(self) -> dict:
        return asdict(self)

    @property
    def model_path_resolved(self) -> str:
        return self.model_path


class BaseCompletionGenerator(
    ABC,
    PromptUtilsMixin,
    InterruptMixin,
    LogitsMixin,
    LockMixin,
    CompletionMixin,
):
    """Base class for all completion generators."""

    @abstractmethod
    def __del__(self):
        """Clean up resources."""

    @property
    @abstractmethod
    def llm_model(self) -> "BaseLLMModel":
        """The LLM model used by this generator."""

    @property
    def model_name(self) -> str:
        """Identifier for the model used by this generator."""
        return Path(self.llm_model.model_path_resolved).stem

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, llm_model: "BaseLLMModel"
    ) -> "BaseCompletionGenerator":
        """Load a pretrained model into RAM."""

    def generate_completion(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Completion:
        """Generate a completion for a given prompt."""
        completion_id = settings.completion_id
        completion_status = self.completion_status[completion_id]
        deque(
            self.generate_text(prompt=prompt, settings=settings),
            maxlen=0,
        )  # exhaust the generator
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "text": completion_status.generated_text,
                    "index": 0,
                    "logprobs": completion_status.logprobs
                    if settings.logprobs
                    else None,
                    "finish_reason": self.get_finish_reason(settings),
                }
            ],
            "usage": {
                "prompt_tokens": completion_status.input_tokens,
                "completion_tokens": completion_status.generated_tokens,
                "total_tokens": completion_status.input_tokens
                + completion_status.generated_tokens,
            },
        }

    def generate_completion_with_streaming(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[CompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        completion_id = settings.completion_id = (
            "chat" + settings.completion_id
        )
        completion_status = self.completion_status[completion_id]
        model = self.model_name
        for token in self.generate_text(prompt=prompt, settings=settings):
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time()),
                "model": model,
                "choices": [
                    {
                        "text": token,
                        "index": 0,
                        "logprobs": completion_status.logprobs
                        if settings.logprobs
                        else None,
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time()),
            "model": model,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": completion_status.logprobs
                    if settings.logprobs
                    else None,
                    "finish_reason": self.get_finish_reason(settings),
                }
            ],
        }

    def generate_chat_completion(
        self, messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        """Generate a completion for a given prompt."""
        completion_id = settings.completion_id = (
            "chat" + settings.completion_id
        )
        completion_status = self.completion_status[completion_id]
        deque(
            self.generate_text(
                prompt=self.convert_messages_into_prompt(
                    messages, settings=settings
                ),
                settings=settings,
            ),
            maxlen=0,
        )  # exhaust the generator
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion_status.generated_text,
                    },
                    "index": 0,
                    "finish_reason": self.get_finish_reason(settings),
                }
            ],
            "usage": {
                "prompt_tokens": completion_status.input_tokens,
                "completion_tokens": completion_status.generated_tokens,
                "total_tokens": completion_status.input_tokens
                + completion_status.generated_tokens,
            },
        }

    def generate_chat_completion_with_streaming(
        self, messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        completion_id = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        model = self.model_name
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        for token in self.generate_text(prompt=prompt, settings=settings):
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": self.get_finish_reason(settings),
                }
            ],
        }

    @abstractmethod
    def encode(self, text: str, **kwargs: Any) -> List[int]:
        """Encode a text string into a list of token IDs."""

    @abstractmethod
    def decode(self, ids: List[int], **kwargs: Any) -> str:
        """Decode a list of token IDs into a text string."""

    @abstractmethod
    def generate_text(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[str]:
        ...

    def accept_settings(
        self,
        prompt: str,
        prompt_tokens: int,
        settings: TextGenerationSettings,
    ) -> None:
        """Update the completion status."""
        # Check if the prompt is too long
        context_window = self.llm_model.max_total_tokens
        self.raise_for_token_limit(
            prompt_tokens=prompt_tokens, context_window=context_window
        )
        settings.max_tokens = min(
            settings.max_tokens, context_window - prompt_tokens
        )
        completion_id = settings.completion_id

        # Update completion status
        self.completion_status[completion_id].input_text = prompt
        self.completion_status[completion_id].input_tokens = prompt_tokens

        # Cache the stops for later use of stop_checker
        self.build_stops_from_settings(settings)


class BaseEmbeddingGenerator(ABC):
    @abstractmethod
    def __del__(self):
        """Clean up resources."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name: str) -> "BaseEmbeddingGenerator":
        """Load a pretrained model into RAM."""
        return cls

    @abstractmethod
    def generate_embeddings(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the model used by this generator."""
