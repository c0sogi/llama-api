from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any, Iterator, List, Optional, TypeVar, Union

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
from ..shared.config import Config, MainCliArgs
from ..utils.logger import ApiLogger

T = TypeVar("T")
logger = ApiLogger(__name__)


@dataclass
class BaseLLMModel:
    model_path: str = "/path/to/model"
    max_total_tokens: int = 2048
    # The template that transforms the input messages into a prompt.
    instruction_template: Optional[str] = None
    # Enabling auto_truncate will automatically truncate the input prompt
    # if max_tokens + prompt_tokens > max_total_tokens.
    auto_truncate: bool = True

    def calculate_rope_alpha(self) -> float:
        """Calculate the RoPE alpha based on the n_ctx.
        Assume that the trained token length is 4096."""
        # The following formula is obtained by fitting the data points
        # (comp, alpha): [(1.0, 1.0) (1.75, 2.0), (2.75, 4.0), (4.1, 8.0)]
        compress_ratio = self.calculate_rope_compress_ratio()
        return (
            -0.00285883 * compress_ratio**4
            + 0.03674126 * compress_ratio**3
            + 0.23873223 * compress_ratio**2
            + 0.49519964 * compress_ratio
            + 0.23218571
        )

    def calculate_rope_freq(self) -> float:
        """Calculate the RoPE frequency based on the n_ctx.
        Assume that the trained token length is 4096."""
        return 10000.0 * self.calculate_rope_alpha() ** (64 / 63)

    def calculate_rope_compress_ratio(self) -> float:
        """Calculate the RoPE embedding compression ratio based on the n_ctx.
        Assume that the trained token length is 4096."""
        return max(self.max_total_tokens / Config.trained_tokens, 1.0)

    def calculate_rope_scale(self) -> float:
        """Calculate the RoPE scaling factor based on the n_ctx.
        Assume that the trained token length is 4096."""
        return 1 / self.calculate_rope_compress_ratio()

    @property
    def asdict(self) -> dict:
        return asdict(self)

    @property
    def model_path_resolved(self) -> str:
        return self.model_path

    def repr(self) -> str:
        return " / ".join(
            f"\033[4m{key}\033[0m: {value}"
            for key, value in self.asdict.items()
            if value is not None
        )


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

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, llm_model: "BaseLLMModel"
    ) -> "BaseCompletionGenerator":
        """Load a pretrained model into RAM."""

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

    @property
    def model_name(self) -> str:
        """Identifier for the model used by this generator."""
        return Path(self.llm_model.model_path_resolved).stem

    def generate_completion(
        self, request: CreateCompletionRequest
    ) -> Completion:
        """Generate a completion for a given prompt."""
        completion_id = request.completion_id
        completion_status = self.completion_status[completion_id]
        deque(self.get_text_generator(request), 0)  # exhaust the generator
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
        for token in self.get_text_generator(request):
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
        deque(self.get_text_generator(request), 0)  # exhaust the generator
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
        text_generator = self.get_text_generator(request)
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
                text_generator
            ):
                chat_completion_chunk["choices"][0]["delta"] = {
                    "function_call": function_call_chunk
                }
                chat_completion_chunk["created"] = int(time())
                yield chat_completion_chunk
        else:
            for token in text_generator:
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

    def get_text_generator(
        self,
        request: Union[CreateChatCompletionRequest, CreateCompletionRequest],
    ) -> Iterator[str]:
        """Accept a completion request and return a generator that yields
        tokens of the generated text."""
        # Convert messages into a prompt if necessary
        if isinstance(request, CreateChatCompletionRequest):
            self.accept_function_call(request)
            prompt = self.convert_messages_into_prompt(
                request, self.llm_model.instruction_template
            )
        else:
            prompt = request.prompt
        prompt_ids = self.encode(prompt)
        prompt_tokens = len(prompt_ids)
        context_window = self.llm_model.max_total_tokens

        if MainCliArgs.max_tokens_limit.value:
            request.max_tokens = min(
                request.max_tokens, MainCliArgs.max_tokens_limit.value
            )

        # Truncate the prompt if it is too long and auto_truncate is enabled
        if self.llm_model.auto_truncate:
            overflow_tokens = (
                prompt_tokens + request.max_tokens - context_window
            )
            if overflow_tokens > 0:
                logger.warning(
                    f"Prompt is too long, truncating {overflow_tokens} tokens."
                )
                prompt_ids = prompt_ids[overflow_tokens:]
                prompt = self.decode(prompt_ids)
                prompt_tokens = len(prompt_ids)

        # Check if the prompt is too long
        self.raise_for_token_limit(
            prompt_tokens=prompt_tokens, context_window=context_window
        )
        request.max_tokens = min(
            request.max_tokens, context_window - prompt_tokens
        )

        # Update completion status
        completion_id = request.completion_id
        self.completion_status[completion_id].input_text = prompt
        self.completion_status[completion_id].input_tokens = prompt_tokens

        # Cache the stops for later use of stop_checker
        self.build_stops_from_settings(request)
        return self.generate_text(prompt, request)


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
