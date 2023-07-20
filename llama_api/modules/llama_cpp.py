"""Wrapper for llama_cpp to generate text completions."""
from dataclasses import dataclass, field
from inspect import signature
from os import getpid, kill
from signal import SIGINT
from typing import Iterator, Literal, Optional

from transformers.models.llama import LlamaTokenizer

from ..common.templates import ChatTurnTemplates, DescriptionTemplates
from ..schemas.api import (
    APIChatMessage,
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    TextGenerationSettings,
)
from ..utils.completions import (
    convert_text_completion_chunks_to_chat,
    convert_text_completion_to_chat,
)
from ..utils.llama_cpp import build_shared_lib
from ..utils.logger import ApiLogger
from ..utils.path import RelativeImport, resolve_model_path_to_posix
from .base import (
    BaseCompletionGenerator,
    BaseLLMModel,
    BaseTokenizer,
    UserChatRoles,
)

build_shared_lib()
with RelativeImport("repositories/llama_cpp"):
    from repositories.llama_cpp import llama_cpp
    from repositories.llama_cpp.llama_cpp.llama_cpp import GGML_USE_CUBLAS


logger = ApiLogger(__name__)
logger.info("🦙 llama-cpp-python repository found!")


class LlamaCppTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self._model_name = model_name

    def encode(self, message: str, /) -> list[int]:
        return self.tokenizer.encode(message)

    def decode(self, tokens: list[int], /) -> str:
        return self.tokenizer.decode(tokens)

    def loader(self) -> LlamaTokenizer:
        paths = self._model_name.split("/")

        if len(paths) == 2:
            root_path = self._model_name
            subfolder = None
        elif len(paths) > 2:
            root_path = "/".join(paths[:2])
            subfolder = "/".join(paths[2:])
        else:
            raise Exception(
                f"Input string {paths} is not in the correct format"
            )
        return LlamaTokenizer.from_pretrained(root_path, subfolder=subfolder)

    @property
    def model_name(self) -> str:
        return self._model_name


@dataclass
class LlamaCppModel(BaseLLMModel):
    """Llama.cpp model that can be loaded from local path."""

    tokenizer: LlamaCppTokenizer = field(
        default_factory=lambda: LlamaCppTokenizer(model_name=""),
        metadata={"description": "The tokenizer to use for this model."},
    )
    model_path: str = field(
        default="YOUR_GGML.bin",
        metadata={
            "description": "The path to the model. Must end with .bin. "
            "You must put .bin file into 'models/ggml'"
        },
    )
    user_chat_roles: UserChatRoles = field(
        default_factory=lambda: UserChatRoles(
            ai="ASSISTANT",
            system="SYSTEM",
            user="USER",
        ),
    )
    prefix_template: Optional[str] = field(
        default_factory=lambda: DescriptionTemplates.USER_AI__DEFAULT,
    )
    chat_turn_prompt: str = field(
        default_factory=lambda: ChatTurnTemplates.ROLE_CONTENT_1,
        metadata={"description": "The prompt to use for chat turns."},
    )
    n_parts: int = field(
        default=-1,
        metadata={
            "description": "Number of parts to split the model into. If -1, "
            "the number of parts is automatically determined."
        },
    )
    n_gpu_layers: int = field(
        default=30,
        metadata={
            "description": "Number of layers to keep on the GPU. "
            "If 0, all layers are kept on the GPU."
        },
    )
    seed: int = field(
        default=-1,
        metadata={"description": "Seed. If -1, a random seed is used."},
    )
    f16_kv: bool = field(
        default=True,
        metadata={"description": "Use half-precision for key/value cache."},
    )
    logits_all: bool = field(
        default=False,
        metadata={
            "description": "Return logits for all tokens, "
            "not just the last token."
        },
    )
    vocab_only: bool = field(
        default=False,
        metadata={"description": "Only load the vocabulary, no weights."},
    )
    use_mlock: bool = field(
        default=True,
        metadata={"description": "Force system to keep model in RAM."},
    )
    n_batch: int = field(
        default=512,
        metadata={
            "description": "Number of tokens to process in parallel. "
            "Should be a number between 1 and n_ctx."
        },
    )
    last_n_tokens_size: int = field(
        default=64,
        metadata={
            "description": "The number of tokens to look back "
            "when applying the repeat_penalty."
        },
    )
    use_mmap: bool = True  # Whether to use memory mapping for the model.
    streaming: bool = True  # Whether to stream the results, token by token.
    cache: bool = (
        False  # The size of the cache in bytes. Only used if cache is True.
    )
    echo: bool = True  # Whether to echo the prompt.
    lora_base: Optional[str] = None  # The path to the Llama LoRA base model.
    lora_path: Optional[
        str
    ] = None  # The path to the Llama LoRA. If None, no LoRa is loaded.
    cache_type: Optional[Literal["disk", "ram"]] = "ram"
    cache_size: Optional[int] = (
        2 << 30
    )  # The size of the cache in bytes. Only used if cache is True.
    n_threads: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of threads to use. "
            "If None, the number of threads is automatically determined."
        },
    )
    low_vram: bool = False  # Whether to use less VRAM.
    embedding: bool = False  # Whether to use the embedding layer.

    # Refer: https://github.com/ggerganov/llama.cpp/pull/2054
    rope_freq_base: float = 10000.0  # I use 26000 for n_ctx=4096.
    rope_freq_scale: float = 1.0  # Generally, 2048 / n_ctx.


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
        logits_processor=llama_cpp.LogitsProcessorList(
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
            logger.info(
                "🗑️ Process will be killed since you are using cuBLAS, "
                "which can potentially cause VRAM leak."
            )
            kill(getpid(), SIGINT)

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
                {"rope_freq_base": llm_model.rope_freq_base}
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
