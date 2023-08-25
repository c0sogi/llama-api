from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Iterator, List, TypeVar

from ..mixins.completion import CompletionMixin
from ..mixins.function_call import FunctionCallMixin
from ..mixins.interrupt import InterruptMixin
from ..mixins.lock import LockMixin
from ..mixins.logits import LogitsMixin
from ..mixins.prompt_utils import PromptUtilsMixin
from ..schemas.api import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
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
    FunctionCallMixin,
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
        self, request: CreateCompletionRequest
    ) -> Completion:
        """Generate a completion for a given prompt."""
        completion_id = request.completion_id
        completion_status = self.completion_status[completion_id]
        deque(
            self.generate_text(prompt=request.prompt, settings=request),
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
                    if request.logprobs
                    else None,
                    "finish_reason": self.get_finish_reason(request),
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
        self, request: CreateCompletionRequest
    ) -> Iterator[CompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        completion_id = request.completion_id
        completion_status = self.completion_status[completion_id]
        completion_chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": completion_status.logprobs
                    if request.logprobs
                    else None,
                    "finish_reason": None,
                }
            ],
        }  # type: CompletionChunk
        for token in self.generate_text(
            prompt=request.prompt, settings=request
        ):
            completion_chunk["created"] = int(time())
            completion_chunk["choices"][0]["text"] = token
            completion_chunk["choices"][0]["logprobs"] = (
                completion_status.logprobs if request.logprobs else None
            )
            yield completion_chunk
        completion_chunk["created"] = int(time())
        completion_chunk["choices"][0]["text"] = ""
        completion_chunk["choices"][0]["logprobs"] = None
        completion_chunk["choices"][0][
            "finish_reason"
        ] = self.get_finish_reason(request)
        yield completion_chunk

    def generate_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> ChatCompletion:
        """Generate a completion for a given prompt."""
        self.accept_function_call(request)
        deque(
            self.generate_text(
                prompt=self.convert_messages_into_prompt(request),
                settings=request,
            ),
            maxlen=0,
        )  # exhaust the generator
        completion_id = request.completion_id
        completion_status = self.completion_status[completion_id]
        finish_reason = self.get_finish_reason(request)
        if finish_reason == "function_call":
            message: ChatCompletionMessage = {
                "role": "assistant",
                "content": None,
                "function_call": self.generate_function_call(
                    completion_status.generated_text
                ),
            }
        else:
            message = {
                "role": "assistant",
                "content": completion_status.generated_text,
            }
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "message": message,
                    "index": 0,
                    "finish_reason": finish_reason,
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
        self,
        request: CreateChatCompletionRequest,
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        self.accept_function_call(request)
        token_generator = self.generate_text(
            prompt=self.convert_messages_into_prompt(request), settings=request
        )
        chat_completion_chunk = {
            "id": request.completion_id,
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }  # type: ChatCompletionChunk
        yield chat_completion_chunk
        if request.grammar:
            for function_call_chunk in self.generate_function_call_streaming(
                token_generator
            ):
                chat_completion_chunk["choices"][0]["delta"] = {
                    "function_call": function_call_chunk
                }
                chat_completion_chunk["created"] = int(time())
                yield chat_completion_chunk
        else:
            for token in token_generator:
                chat_completion_chunk["choices"][0]["delta"] = {
                    "content": token
                }
                chat_completion_chunk["created"] = int(time())
                yield chat_completion_chunk
        chat_completion_chunk["created"] = int(time())
        chat_completion_chunk["choices"][0]["delta"] = {}
        chat_completion_chunk["choices"][0][
            "finish_reason"
        ] = self.get_finish_reason(request)
        yield chat_completion_chunk

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
        """Accept the settings for a completion request.and update the
        completion status."""
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
