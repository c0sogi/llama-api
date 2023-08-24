from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from re import search
from time import time
from typing import Any, Iterator, List, TypeVar

from orjson import JSONDecodeError, loads

from ..mixins.completion import CompletionMixin
from ..mixins.function_call import FunctionCallMixin
from ..mixins.interrupt import InterruptMixin
from ..mixins.lock import LockMixin
from ..mixins.logits import LogitsMixin
from ..mixins.prompt_utils import PromptUtilsMixin
from ..schemas.api import (  # noqa: F401
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    FunctionCompletionChunk,
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
        model = self.model_name
        for token in self.generate_text(
            prompt=request.prompt, settings=request
        ):
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
                        if request.logprobs
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
                    if request.logprobs
                    else None,
                    "finish_reason": self.get_finish_reason(request),
                }
            ],
        }

    def generate_chat_completion(
        self,
        request: CreateChatCompletionRequest,
    ) -> ChatCompletion:
        """Generate a completion for a given prompt."""
        self.accept_function_call(request)
        completion_id = request.completion_id
        completion_status = self.completion_status[completion_id]
        deque(
            self.generate_text(
                prompt=self.convert_messages_into_prompt(request),
                settings=request,
            ),
            maxlen=0,
        )  # exhaust the generator
        finish_reason = self.get_finish_reason(request)
        if finish_reason == "function_call":
            function_call_match = search(
                r'\{\s*"name"\s*:\s*"((?:[^"]|\\")*)"\s*,\s*"arguments"\s*:\s*({.*})\}\s*',  # noqa: E501
                completion_status.generated_text,
            )
            assert (
                function_call_match is not None
            ), f"Invalid function call: {completion_status.generated_text}"
            message = {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_call_match.group(1),
                    "arguments": function_call_match.group(2),
                },
            }
        else:
            message = {
                "role": "assistant",
                "content": completion_status.generated_text,
            }  # type: ChatCompletionMessage
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "message": message,
                    "index": 0,
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

    def generate_chat_completion_with_streaming(
        self,
        request: CreateChatCompletionRequest,
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        self.accept_function_call(request)
        prompt = self.convert_messages_into_prompt(request)
        model = self.model_name
        completion_id = request.completion_id
        function_call = last_function_name = ""
        begin_function_arguments = False
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
        for token in self.generate_text(prompt=prompt, settings=request):
            if not request.grammar:
                delta = {"content": token}  # type: ChatCompletionChunkDelta
            else:
                function_call += token
                name_match = search(
                    r'\{\s*"name"\s*:\s*"((?:[^"]|\\")*)', function_call
                )
                arguments_match = search(
                    r'"arguments"\s*:\s*{.*', function_call
                )
                if name_match is not None and arguments_match is None:
                    current_function_name = name_match.group(1)
                    if current_function_name == last_function_name:
                        continue
                    last_function_name = current_function_name
                    function_chunk = {
                        "name": token,
                    }  # type: FunctionCompletionChunk
                elif name_match is not None:
                    if not begin_function_arguments:
                        begin_function_arguments = True
                        token = token[token.rfind("{") :]  # noqa: E203
                    try:
                        loads(function_call)
                        continue
                    except JSONDecodeError:
                        function_chunk = {"arguments": token}
                else:
                    continue
                delta = {"function_call": function_chunk}
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time()),
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None}
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
                    "finish_reason": self.get_finish_reason(request),
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
