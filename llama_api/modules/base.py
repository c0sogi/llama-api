from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, List, TypeVar

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
):
    """Base class for all completion generators."""

    @abstractmethod
    def __del__(self):
        """Clean up resources."""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, llm_model: "BaseLLMModel"
    ) -> "BaseCompletionGenerator":
        """Load a pretrained model into RAM."""

    @abstractmethod
    def generate_completion(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Completion:
        """Generate a completion for a given prompt."""

    @abstractmethod
    def generate_completion_with_streaming(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[CompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""

    @abstractmethod
    def generate_chat_completion(
        self, messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        """Generate a completion for a given prompt."""

    @abstractmethod
    def generate_chat_completion_with_streaming(
        self, messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""

    @abstractmethod
    def encode(self, text: str, **kwargs: Any) -> List[int]:
        """Encode a text string into a list of token IDs."""

    @abstractmethod
    def decode(self, ids: List[int], **kwargs: Any) -> str:
        """Decode a list of token IDs into a text string."""

    @property
    @abstractmethod
    def llm_model(self) -> "BaseLLMModel":
        """The LLM model used by this generator."""

    @property
    def model_name(self) -> str:
        """Identifier for the model used by this generator."""
        return Path(self.llm_model.model_path_resolved).stem


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
