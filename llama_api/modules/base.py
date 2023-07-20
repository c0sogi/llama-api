from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar, Union

from tiktoken import Encoding, get_encoding

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
class UserChatRoles:
    ai: str
    system: str
    user: str


class BaseTokenizer(ABC, Generic[T]):
    _tokenizer: Optional[Union[T, Encoding]] = None

    def _load_tokenizer(self, loader: Callable[[], T]) -> Union[T, Encoding]:
        if self._tokenizer is None:
            try:
                self._tokenizer = loader()
            except Exception as e:
                logger.warning(e)
                self._tokenizer = self.fallback_tokenizer
        return self._tokenizer

    @abstractmethod
    def loader() -> T:
        ...

    @classmethod
    @property
    def fallback_tokenizer(cls) -> Union[T, Encoding]:
        logger.warning(
            "Fallback tokenizer is used! Please specify a tokenizer."
        )
        if cls._tokenizer is None:
            cls._tokenizer = get_encoding("cl100k_base")
        return cls._tokenizer

    @property
    def tokenizer(self) -> Union[T, Encoding]:
        return self._load_tokenizer(self.loader)

    @property
    def model_name(self) -> str:
        return "cl100k_base"

    def encode(self, message: str) -> list[int]:
        assert isinstance(self.tokenizer, Encoding)
        return self.tokenizer.encode(message)

    def decode(self, tokens: list[int]) -> str:
        assert isinstance(self.tokenizer, Encoding)
        return self.tokenizer.decode(tokens)

    def tokens_of(self, message: str) -> int:
        return len(self.encode(message))

    def split_text_on_tokens(
        self, text: str, tokens_per_chunk: int, chunk_overlap: int
    ) -> list[str]:
        """Split incoming text and return chunks."""
        splits: list[str] = []
        input_ids = self.encode(text)
        start_idx = 0
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(self.decode(chunk_ids))
            start_idx += tokens_per_chunk - chunk_overlap
            cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits

    def get_chunk_of(self, text: str, tokens: int) -> str:
        """Split incoming text and return chunks."""
        input_ids = self.encode(text)
        return self.decode(input_ids[: min(tokens, len(input_ids))])


@dataclass
class BaseLLMModel:
    model_path: str = "/path/to/model"
    max_total_tokens: int = 2048
    tokenizer: BaseTokenizer = field(default_factory=BaseTokenizer)
    user_chat_roles: UserChatRoles = field(
        default_factory=lambda: UserChatRoles(
            ai="assistant", system="system", user="user"
        ),
    )
    prefix_template: Optional[str] = field(
        default=None,
        metadata={
            "description": "A prefix to prepend to the generated text. "
            "If None, no prefix is prepended."
        },
    )
    suffix_template: Optional[str] = field(
        default=None,
        metadata={
            "description": "A suffix to prepend to the generated text. "
            "If None, no suffix is prepended."
        },
    )

    prefix: Optional[str] = field(init=False, repr=False, default=None)
    suffix: Optional[str] = field(init=False, repr=False, default=None)


class BaseCompletionGenerator(ABC):
    """Base class for all completion generators."""

    user_role: str = "user"
    system_role: str = "system"

    user_input_role: str = "User"
    system_input_role: str = "System"

    ai_fallback_input_role: str = "Assistant"

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
        ...

    @abstractmethod
    def generate_completion(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Completion:
        """Generate a completion for a given prompt."""
        ...

    @abstractmethod
    def generate_completion_with_streaming(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[CompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        ...

    @abstractmethod
    def generate_chat_completion(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        """Generate a completion for a given prompt."""
        ...

    @abstractmethod
    def generate_chat_completion_with_streaming(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        """Generate a completion for a given prompt,
        yielding chunks of text as they are generated."""
        ...

    @staticmethod
    def get_stop_strings(*roles: str) -> list[str]:
        """A helper method to generate stop strings for a given set of roles.
        Stop strings are required to stop text completion API from generating
        text that does not belong to the current chat turn.
        e.g. The common stop string is "### USER:",
        which can prevent ai from generating user's message itself."""

        prompt_stop = set()
        for role in roles:
            avoids = (
                f"{role}:",
                f"### {role}:",
                f"###{role}:",
            )
            prompt_stop.update(
                avoids,
                map(str.capitalize, avoids),
                map(str.upper, avoids),
                map(str.lower, avoids),
            )
        return list(prompt_stop)

    @classmethod
    def convert_messages_into_prompt(
        cls, messages: list[APIChatMessage], settings: TextGenerationSettings
    ) -> str:
        """A helper method to convert list of messages into one text prompt."""

        ai_input_role: str = cls.ai_fallback_input_role
        chat_history: str = ""
        for message in messages:
            if message.role.lower() == cls.user_role:
                input_role = cls.user_input_role
            elif message.role.lower() == cls.system_role:
                input_role = cls.system_input_role
            else:
                input_role = ai_input_role = message.role
            chat_history += f"### {input_role}:{message.content}"

        prompt_stop: list[str] = cls.get_stop_strings(
            cls.user_input_role, cls.system_input_role, ai_input_role
        )
        if isinstance(settings.stop, str):
            settings.stop = prompt_stop + [settings.stop]
        elif isinstance(settings.stop, list):
            settings.stop = prompt_stop + settings.stop
        else:
            settings.stop = prompt_stop
        return chat_history + f"### {ai_input_role}:"

    @staticmethod
    def is_possible_to_generate_stops(
        decoded_text: str, stops: list[str]
    ) -> bool:
        """A helper method to check if
        the decoded text contains any of the stop tokens."""

        for stop in stops:
            if stop in decoded_text or any(
                [
                    decoded_text.endswith(stop[: i + 1])
                    for i in range(len(stop))
                ]
            ):
                return True
        return False

    @property
    @abstractmethod
    def llm_model(self) -> "BaseLLMModel":
        """The LLM model used by this generator."""
        ...


class BaseEmbeddingGenerator(ABC):
    @abstractmethod
    def __del__(self):
        """Clean up resources."""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name: str) -> "BaseEmbeddingGenerator":
        """Load a pretrained model into RAM."""
        return cls

    @abstractmethod
    def generate_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the model used by this generator."""
        ...
