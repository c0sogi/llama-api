"""Wrapper for llama_cpp to generate text completions."""
from inspect import signature
from typing import Iterator, Literal, Optional

from ..schemas.api import (
    APIChatMessage,
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    TextGenerationSettings,
)
from ..schemas.models import LlamaCppModel
from ..utils.completions import (
    convert_text_completion_chunks_to_chat,
    convert_text_completion_to_chat,
)
from ..utils.llama_cpp import build_shared_lib
from ..utils.logger import ApiLogger
from ..utils.path import import_repository, resolve_model_path_to_posix
from .base import BaseCompletionGenerator

logger = ApiLogger(__name__)
logger.info("🦙 llama-cpp-python repository found!")
build_shared_lib(logger=logger)
try:
    with import_repository(
        git_path="https://github.com/abetlen/llama-cpp-python",
        disk_path="repositories/llama_cpp",
    ):
        from repositories.llama_cpp import llama_cpp
        from repositories.llama_cpp.llama_cpp.llama_cpp import GGML_USE_CUBLAS
except ImportError:
    logger.warning(
        "🦙 llama-cpp-python repository not found. "
        "Falling back to llama-cpp-python submodule."
    )

    import llama_cpp
    from llama_cpp import GGML_USE_CUBLAS as GGML_USE_CUBLAS


def _make_logit_bias_processor(
    llama: llama_cpp.Llama,
    logit_bias: dict[str, float],
    logit_bias_type: Optional[Literal["input_ids", "tokens"]],
):
    """Create a logit bias processor to bias the logit scores."""
    if logit_bias_type is None:
        logit_bias_type = "input_ids"

    to_bias: dict[int, float] = {}
    if logit_bias_type == "input_ids":
        for input_id_string, score in logit_bias.items():
            to_bias[int(input_id_string)] = score

    elif logit_bias_type == "tokens":
        for token, score in logit_bias.items():
            for input_id in llama.tokenize(
                token.encode("utf-8"), add_bos=False
            ):
                to_bias[input_id] = score

    def logit_bias_processor(
        input_ids: list[int],
        scores: list[float],
    ) -> list[float]:
        new_scores: list[float] = [0.0] * len(scores)
        for input_id, score in enumerate(scores):
            new_scores[input_id] = score + to_bias.get(input_id, 0.0)

        return new_scores

    return logit_bias_processor


def _create_completion(
    client: llama_cpp.Llama,
    prompt: str,
    stream: bool,
    settings: TextGenerationSettings,
) -> Completion | Iterator[CompletionChunk]:
    return client.create_completion(  # type: ignore
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
        logits_processor=llama_cpp.LogitsProcessorList(  # type: ignore
            [
                _make_logit_bias_processor(
                    client,
                    settings.logit_bias,
                    settings.logit_bias_type,
                ),
            ]
        )
        if settings.logit_bias is not None
        else None,
        stop=settings.stop,
    )


def _create_chat_completion(
    client: llama_cpp.Llama,
    messages: list[APIChatMessage],
    stream: bool,
    settings: TextGenerationSettings,
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    prompt: str = LlamaCppCompletionGenerator.convert_messages_into_prompt(
        messages, settings=settings
    )
    completion_or_chunks = client(
        prompt=prompt,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        stream=stream,
        max_tokens=settings.max_tokens,
        repeat_penalty=settings.repeat_penalty,
        presence_penalty=settings.presence_penalty,
        frequency_penalty=settings.frequency_penalty,
        tfs_z=settings.tfs_z,
        mirostat_mode=settings.mirostat_mode,
        mirostat_tau=settings.mirostat_tau,
        mirostat_eta=settings.mirostat_eta,
        logits_processor=llama_cpp.LogitsProcessorList(  # type: ignore
            [
                _make_logit_bias_processor(
                    client,
                    settings.logit_bias,
                    settings.logit_bias_type,
                ),
            ]
        )
        if settings.logit_bias is not None
        else None,
        stop=settings.stop,
    )
    if isinstance(completion_or_chunks, Iterator):
        return convert_text_completion_chunks_to_chat(
            completion_or_chunks,  # type: ignore
        )
    else:
        return convert_text_completion_to_chat(
            completion_or_chunks,  # type: ignore
        )


class LlamaCppCompletionGenerator(BaseCompletionGenerator):
    generator: Optional[Iterator[CompletionChunk | ChatCompletionChunk]] = None
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
        if GGML_USE_CUBLAS:
            logger.warning(
                "🗑️ Since you are using cuBLAS, unloading llama.cpp model"
                "will cause VRAM leak."
            )

    @property
    def llm_model(self) -> "LlamaCppModel":
        assert self._llm_model is not None
        return self._llm_model

    @classmethod
    def from_pretrained(
        cls, llm_model: "LlamaCppModel"
    ) -> "LlamaCppCompletionGenerator":
        additional_kwargs = {}
        arg_keys = signature(llama_cpp.Llama.__init__).parameters.keys()
        if "rope_freq_base" in arg_keys:
            additional_kwargs.update(
                {"rope_freq_base": llm_model.rope_freq_base},
            )
        if "rope_freq_scale" in arg_keys:
            additional_kwargs.update(
                {"rope_freq_scale": llm_model.rope_freq_scale}
            )
        client = llama_cpp.Llama(
            model_path=resolve_model_path_to_posix(
                llm_model.model_path,
                default_relative_directory="models/ggml",
            ),
            n_ctx=llm_model.max_total_tokens,
            n_parts=llm_model.n_parts,
            n_gpu_layers=llm_model.n_gpu_layers,
            seed=llm_model.seed,
            f16_kv=llm_model.f16_kv,
            logits_all=llm_model.logits_all,
            vocab_only=llm_model.vocab_only,
            use_mmap=llm_model.use_mmap,
            use_mlock=llm_model.use_mlock,
            embedding=llm_model.embedding,
            n_threads=llm_model.n_threads,
            n_batch=llm_model.n_batch,
            last_n_tokens_size=llm_model.last_n_tokens_size,
            lora_base=llm_model.lora_base,
            lora_path=llm_model.lora_path,
            low_vram=llm_model.low_vram,
            verbose=llm_model.echo,
            **additional_kwargs,
        )
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
        yield from completion_chunk_generator

    def generate_chat_completion(
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
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
        self, messages: list[APIChatMessage], settings: TextGenerationSettings
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
        yield from chat_completion_chunk_generator

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        assert self.client is not None, "Client is not initialized"
        return self.client.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos
        )

    def decode(self, tokens: list[int]) -> str:
        assert self.client is not None, "Client is not initialized"
        return self.client.detokenize(tokens).decode("utf-8", errors="ignore")
