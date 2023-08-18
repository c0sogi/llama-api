"""Wrapper for llama_cpp to generate text completions."""
from inspect import signature
from typing import Callable, Iterator, List, Optional, Union

from ..schemas.api import (
    APIChatMessage,
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    TextGenerationSettings,
)
from ..schemas.models import LlamaCppModel
from ..shared.config import Config
from ..utils.completions import (
    convert_text_completion_chunks_to_chat,
    convert_text_completion_to_chat,
)
from ..utils.dependency import import_repository
from ..utils.llama_cpp import build_shared_lib
from ..utils.logger import ApiLogger
from .base import BaseCompletionGenerator

logger = ApiLogger(__name__)
logger.info("🦙 llama-cpp-python repository found!")
with import_repository(**Config.repositories["llama_cpp"]):
    build_shared_lib(logger=logger)
    from repositories.llama_cpp import llama_cpp


class LogitsProcessorList(
    List[Callable[[List[int], List[float]], List[float]]]
):
    def __call__(
        self, input_ids: List[int], scores: List[float]
    ) -> List[float]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


def _create_completion(
    client: llama_cpp.Llama,
    prompt: str,
    stream: bool,
    settings: TextGenerationSettings,
) -> Union[Completion, Iterator[CompletionChunk]]:
    logit_processors = LogitsProcessorList(
        [
            processor.without_torch
            for processor in BaseCompletionGenerator.get_logit_processors(
                settings=settings,
                encoder=lambda s: client.tokenize(
                    s.encode("utf-8"), add_bos=False
                ),
            )
        ]
    )
    return client.create_completion(
        stream=stream,
        prompt=prompt,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        logprobs=settings.logprobs,
        echo=settings.echo,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        repeat_penalty=settings.repeat_penalty,
        top_k=settings.top_k,
        tfs_z=settings.tfs_z,
        mirostat_mode=settings.mirostat_mode,
        mirostat_tau=settings.mirostat_tau,
        mirostat_eta=settings.mirostat_eta,
        logits_processor=logit_processors if logit_processors else None,  # type: ignore  # noqa: E501
        stop=settings.stop,
    )


def _create_chat_completion(
    client: llama_cpp.Llama,
    messages: List[APIChatMessage],
    stream: bool,
    settings: TextGenerationSettings,
) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
    prompt: str = LlamaCppCompletionGenerator.convert_messages_into_prompt(
        messages, settings=settings
    )
    completion_or_chunks = _create_completion(
        client=client, prompt=prompt, stream=stream, settings=settings
    )
    if isinstance(completion_or_chunks, Iterator):
        return convert_text_completion_chunks_to_chat(
            completion_or_chunks,
        )
    else:
        return convert_text_completion_to_chat(
            completion_or_chunks,
        )


class LlamaCppCompletionGenerator(BaseCompletionGenerator):
    generator: Optional[
        Iterator[Union[CompletionChunk, ChatCompletionChunk]]
    ] = None
    client: Optional[llama_cpp.Llama] = None
    _llm_model: Optional["LlamaCppModel"] = None

    def __del__(self) -> None:
        """Currnetly, VRAM is not freed when using cuBLAS.
        There is no cuda-related API in shared library(llama).
        Let's wait until llama.cpp fixes it.
        """
        if self.client is not None:
            getattr(self.client, "__del__", lambda: None)()
            del self.client
            self.client = None
            logger.info("🗑️ LlamaCppCompletionGenerator deleted!")

    @property
    def llm_model(self) -> "LlamaCppModel":
        assert self._llm_model is not None
        return self._llm_model

    @classmethod
    def from_pretrained(
        cls, llm_model: "LlamaCppModel"
    ) -> "LlamaCppCompletionGenerator":
        kwargs = {
            # Get all attributes of llm_model
            key: value
            for key, value in llm_model.asdict.items()
            # Hacky way to pass arguments to older versions of llama-cpp-python
            if key in signature(llama_cpp.Llama.__init__).parameters.keys()
        }
        kwargs["n_ctx"] = llm_model.max_total_tokens
        kwargs["model_path"] = llm_model.model_path_resolved
        kwargs["verbose"] = llm_model.verbose and llm_model.echo
        client = llama_cpp.Llama(**kwargs)
        if llm_model.cache:
            cache_type = llm_model.cache_type
            if cache_type is None:
                cache_type = "ram"
            cache_size = (
                2 << 30
                if llm_model.cache_size is None
                else llm_model.cache_size
            )
            if cache_type == "disk":
                if llm_model.echo:
                    logger.info(
                        f"🦙 Using disk cache with size {cache_size}",
                    )
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=cache_size)
            else:
                if llm_model.echo:
                    logger.info(
                        f"🦙 Using ram cache with size {cache_size}",
                    )
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=cache_size)
            client.set_cache(cache)
        self = cls()
        self.client = client
        self._llm_model = llm_model
        return self

    def generate_completion(
        self,
        prompt: str,
        settings: TextGenerationSettings = TextGenerationSettings(),
    ) -> Completion:
        assert self.client is not None
        completion = _create_completion(
            client=self.client, prompt=prompt, stream=False, settings=settings
        )
        assert not isinstance(completion, Iterator)
        return completion

    def generate_completion_with_streaming(
        self,
        prompt: str,
        settings: TextGenerationSettings = TextGenerationSettings(),
    ) -> Iterator[CompletionChunk]:
        assert self.client is not None
        completion_chunk_generator = _create_completion(
            client=self.client, prompt=prompt, stream=True, settings=settings
        )
        assert isinstance(completion_chunk_generator, Iterator)
        self.generator = completion_chunk_generator
        for chunk in completion_chunk_generator:
            if self.is_interrupted:
                yield chunk
                return  # the generator was interrupted
            yield chunk

    def generate_chat_completion(
        self, messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> ChatCompletion:
        assert self.client is not None
        chat_completion = _create_chat_completion(
            client=self.client,
            messages=messages,
            stream=False,
            settings=settings,
        )
        assert not isinstance(chat_completion, Iterator)
        return chat_completion

    def generate_chat_completion_with_streaming(
        self, messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> Iterator[ChatCompletionChunk]:
        assert self.client is not None
        chat_completion_chunk_generator = _create_chat_completion(
            client=self.client,
            messages=messages,
            stream=True,
            settings=settings,
        )
        assert isinstance(chat_completion_chunk_generator, Iterator)
        self.generator = chat_completion_chunk_generator
        for chunk in chat_completion_chunk_generator:
            if self.is_interrupted:
                yield chunk
                return  # the generator was interrupted
            yield chunk

    def encode(self, text: str, add_bos: bool = True, **kwargs) -> List[int]:
        assert self.client is not None, "Client is not initialized"
        return self.client.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos
        )

    def decode(self, ids: List[int], **kwargs) -> str:
        assert self.client is not None, "Client is not initialized"
        return self.client.detokenize(ids).decode("utf-8", errors="ignore")
