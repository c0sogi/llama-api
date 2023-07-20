"""Wrapper for exllama to generate text completions."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from torch import IntTensor, cuda

from ..common.templates import ChatTurnTemplates, DescriptionTemplates
from ..utils.completions import (
    make_chat_completion,
    make_chat_completion_chunk,
    make_completion,
    make_completion_chunk,
)
from ..utils.logger import ApiLogger
from ..utils.path import RelativeImport, resolve_model_path_to_posix
from .base import (
    BaseCompletionGenerator,
    BaseLLMModel,
    BaseTokenizer,
    UserChatRoles,
)

with RelativeImport("repositories/exllama"):
    from repositories.exllama.generator import ExLlamaGenerator
    from repositories.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
    from repositories.exllama.tokenizer import ExLlamaTokenizer

if TYPE_CHECKING:
    from ..schemas.api import (
        APIChatMessage,
        ChatCompletion,
        ChatCompletionChunk,
        Completion,
        CompletionChunk,
        TextGenerationSettings,
    )

logger = ApiLogger("||🦙 exllama.generator||")
assert cuda.is_available(), "CUDA must be available to use ExLlama."


class ExllamaTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        self._model_name = model_name

    def encode(self, message: str, /) -> list[int]:
        if isinstance(self.tokenizer, ExLlamaTokenizer):
            return self.tokenizer.encode(message).flatten().tolist()
        else:
            return self.tokenizer.encode(message)

    def decode(self, tokens: list[int], /) -> str:
        if isinstance(self.tokenizer, ExLlamaTokenizer):
            return str(self.tokenizer.decode(IntTensor(tokens)))
        else:
            return self.tokenizer.decode(tokens)

    def loader(self) -> ExLlamaTokenizer:
        model_folder_path = Path(
            resolve_model_path_to_posix(
                self.model_name,
                default_relative_directory="models/gptq",
            ),
        )
        return ExLlamaTokenizer(
            (model_folder_path / "tokenizer.model").as_posix(),
        )

    @property
    def model_name(self) -> str:
        return self._model_name


@dataclass
class ExllamaModel(BaseLLMModel):
    """Exllama model that can be loaded from local path."""

    model_path: str = field(
        default="YOUR_GPTQ_FOLDER_NAME",
        metadata={
            "description": "The GPTQ model path to the model."
            "e.g. If you have a model folder in 'models/gptq/your_model',"
            "then you should set this to 'your_model'."
        },
    )
    tokenizer: ExllamaTokenizer = field(
        default_factory=lambda: ExllamaTokenizer(model_name=""),
        metadata={"description": "The tokenizer to use for this model."},
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
        metadata={"description": "The prompt to use for each chat turn."},
    )
    compress_pos_emb: float = field(
        default=1.0,
        metadata={
            "description": "Increase to compress positional embeddings "
            "applied to sequence. This is useful when you want to "
            "extend context window size. e.g. If you want to extend context "
            "window size from 2048 to 4096, set this to 2.0."
        },
    )
    gpu_peer_fix: bool = field(
        default=False,
        metadata={
            "description": "Apparently Torch can have problems transferring "
            "tensors directly 1 GPU to another. Enable this to use system "
            "RAM as a buffer for GPU to GPU transfers."
        },
    )
    auto_map: Optional[list[float]] = field(
        default=None,
        metadata={
            "description": "List of floats with memory allocation in GB, "
            "per CUDA device, overrides device_map."
        },
    )

    # Optional parameters
    matmul_recons_thd: int = 8
    fused_mlp_thd: int = 2
    sdp_thd: int = 8
    fused_attn: bool = True
    matmul_fused_remap: bool = False
    rmsnorm_no_half2: bool = False
    rope_no_half2: bool = False
    matmul_no_half2: bool = False
    silu_no_half2: bool = False
    concurrent_streams: bool = False


class ExllamaCompletionGenerator(BaseCompletionGenerator):
    config: Optional[ExLlamaConfig] = None
    model: Optional[ExLlama] = None
    cache: Optional[ExLlamaCache] = None
    tokenizer: Optional[ExLlamaTokenizer] = None
    generator: Optional[ExLlamaGenerator] = None
    _llm_model: Optional["ExllamaModel"] = None
    _completion_status: dict[
        str, int
    ] = {}  # key: completion_id, value: number of completion tokens

    def __del__(self) -> None:
        if self.model is not None:
            self.model.free_unmanaged()
            del self.model
            self.model = None
            logger.info("🗑️ ExllamaCompletionGenerator model deleted")
        if self.tokenizer is not None:
            getattr(self.tokenizer, "__del__", lambda: None)()
            del self.tokenizer
            self.tokenizer = None
            logger.info("🗑️ ExllamaCompletionGenerator tokenizer deleted")
        if self.cache is not None:
            getattr(self.cache, "__del__", lambda: None)()
            del self.cache
            self.cache = None
            logger.info("🗑️ ExllamaCompletionGenerator cache deleted")

    @property
    def llm_model(self) -> "ExllamaModel":
        assert self._llm_model is not None
        return self._llm_model

    @classmethod
    def from_pretrained(
        cls, llm_model: "ExllamaModel"
    ) -> "ExllamaCompletionGenerator":
        if not isinstance(llm_model.tokenizer.tokenizer, ExLlamaTokenizer):
            raise ValueError(
                "ExllamaCompletionGenerator requires an ExLlamaTokenizer, "
                f"not {type(llm_model.tokenizer.tokenizer)}."
            )
        result = cls()
        result.config = _make_config(llm_model)
        result.tokenizer = llm_model.tokenizer.tokenizer
        result.model = ExLlama(result.config)
        result.cache = ExLlamaCache(result.model)
        result.generator = None
        result._llm_model = llm_model
        return result

    @contextmanager
    def _generator_context_manager(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> Iterator[ExLlamaGenerator]:
        """Make a generator object for the ExLlama model."""
        assert self.model is not None, "Model is not initialized."
        assert self.tokenizer is not None, "Tokenizer is not initialized."
        assert self.cache is not None, "Cache is not initialized."

        generator = ExLlamaGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            cache=self.cache,
        )
        generator.settings.temperature = settings.temperature
        generator.settings.top_p = settings.top_p
        generator.settings.top_k = settings.top_k
        generator.settings.typical = settings.typical_p
        generator.settings.token_repetition_penalty_max = (
            settings.repeat_penalty
        )
        if (
            settings.ban_eos_token
            and generator.tokenizer.eos_token_id is not None
        ):
            generator.disallow_tokens([generator.tokenizer.eos_token_id])

        generator.end_beam_search()
        generator.gen_begin_reuse(generator.tokenizer.encode(prompt))
        yield generator
        del generator

    def _generate_text(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> str:
        return "".join(
            self._generate_text_with_streaming(prompt, settings=settings)
        )

    def _generate_text_with_streaming(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> Iterator[str]:
        assert (
            self.model is not None
            and self.tokenizer is not None
            and self.cache is not None
        )

        # Make sure that the stop token is a list
        if isinstance(settings.stop, str):
            stops = [settings.stop]
        elif isinstance(settings.stop, list):
            stops = settings.stop
        else:
            stops = []

        with self._generator_context_manager(
            prompt, settings=settings
        ) as generator:
            # Start generation
            initial_len = generator.sequence[0].shape[0]
            has_leading_space: bool = False
            text_cursor: int = 0
            n_completion_tokens: int = 0

            for n_completion_tokens in range(1, settings.max_tokens + 1):
                token = generator.gen_single_token()
                if token.item() == generator.tokenizer.eos_token_id:
                    return
                if (
                    n_completion_tokens == 0
                    and generator.tokenizer.tokenizer.IdToPiece(
                        int(token)
                    ).startswith("▁")
                ):
                    has_leading_space = True

                decoded_text = str(
                    generator.tokenizer.decode(
                        generator.sequence[0][initial_len:]
                    )
                )
                if has_leading_space:
                    decoded_text = " " + decoded_text
                if self.is_possible_to_generate_stops(
                    decoded_text, stops=stops
                ):
                    for stop in stops:
                        if stop in decoded_text:
                            return
                    continue
                text_piece = decoded_text[text_cursor:]
                if "�" in text_piece:
                    continue
                yield text_piece
                text_cursor += len(text_piece)
            self._completion_status[
                settings.completion_id
            ] = n_completion_tokens

    def generate_completion_with_streaming(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> Iterator["CompletionChunk"]:
        assert self.config is not None and self.tokenizer is not None
        completion_id: str = settings.completion_id
        model_path: str = str(self.config.model_path)
        last_token: Optional[str] = None
        generated_text: str = ""
        for token in self._generate_text_with_streaming(
            prompt, settings=settings
        ):
            generated_text += token
            if last_token is not None:
                yield make_completion_chunk(
                    id=completion_id,
                    model=model_path,
                    text=last_token,
                    finish_reason=None,
                )
            last_token = token
        yield make_completion_chunk(
            id=completion_id,
            model=model_path,
            text=last_token if last_token is not None else "",
            finish_reason="length"
            if self._completion_status.get(
                completion_id, self.tokenizer.encode(generated_text).shape[1]
            )
            >= settings.max_tokens
            else "stop",
        )

    def generate_completion(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> "Completion":
        assert self.tokenizer is not None and self.config is not None
        completion_id: str = settings.completion_id
        generated_text: str = self._generate_text(prompt, settings=settings)
        n_prompt_tokens: int = self.tokenizer.encode(prompt).shape[1]
        n_completion_tokens: int = self._completion_status.get(
            completion_id, self.tokenizer.encode(generated_text).shape[1]
        )
        return make_completion(
            id=completion_id,
            model=str(self.config.model_path),
            text=generated_text,
            prompt_tokens=n_prompt_tokens,
            completion_tokens=n_completion_tokens,
            finish_reason="length"
            if n_completion_tokens >= settings.max_tokens
            else "stop",
        )

    def generate_chat_completion_with_streaming(
        self,
        messages: list["APIChatMessage"],
        settings: "TextGenerationSettings",
    ) -> Iterator["ChatCompletionChunk"]:
        assert self.config is not None and self.tokenizer is not None
        completion_id: str = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        model_path: str = str(self.config.model_path)
        last_token: Optional[str] = None
        generated_text: str = ""
        for token in self._generate_text_with_streaming(
            prompt, settings=settings
        ):
            generated_text += token
            if last_token is not None:
                yield make_chat_completion_chunk(
                    id=completion_id,
                    model=model_path,
                    content=last_token,
                    finish_reason=None,
                )
            last_token = token
        yield make_chat_completion_chunk(
            id=completion_id,
            model=model_path,
            content=last_token if last_token is not None else "",
            finish_reason="length"
            if self._completion_status.get(
                completion_id, self.tokenizer.encode(generated_text).shape[1]
            )
            else "stop",
        )

    def generate_chat_completion(
        self,
        messages: list["APIChatMessage"],
        settings: "TextGenerationSettings",
    ) -> "ChatCompletion":
        assert self.tokenizer is not None and self.config is not None
        completion_id: str = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        generated_text: str = self._generate_text(prompt, settings=settings)
        prompt_tokens: int = self.tokenizer.encode(prompt).shape[1]
        completion_tokens: int = self._completion_status.get(
            completion_id, self.tokenizer.encode(generated_text).shape[1]
        )
        return make_chat_completion(
            id=completion_id,
            model=str(self.config.model_path),
            content=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="length"
            if completion_tokens >= settings.max_tokens
            else "stop",
        )

    def encode(self, message: str, /) -> list[int]:
        assert self.tokenizer is not None, "Tokenizer is not initialized"
        return self.tokenizer.encode(message).flatten().tolist()

    def decode(self, tokens: list[int], /) -> str:
        assert self.tokenizer is not None, "Tokenizer is not initialized"
        return str(self.tokenizer.decode(IntTensor(tokens)))


def _make_config(llm_model: "ExllamaModel") -> ExLlamaConfig:
    """Create a config object for the ExLlama model."""
    model_folder_path = Path(
        resolve_model_path_to_posix(
            llm_model.model_path,
            default_relative_directory="models/gptq",
        ),
    )
    config = ExLlamaConfig((model_folder_path / "config.json").as_posix())

    # Find the model checkpoint
    model_file_found: list[Path] = []
    for ext in (".safetensors", ".pt", ".bin"):
        model_file_found.extend(model_folder_path.glob(f"*{ext}"))
        if model_file_found:
            if len(model_file_found) > 1:
                logger.warning(
                    f"More than one {ext} model has been found. "
                    "The last one will be selected. It could be wrong."
                )

            break
    if not model_file_found:
        raise FileNotFoundError(
            f"No model has been found in {model_folder_path}."
        )
    config.model_path = model_file_found[-1].as_posix()  # type: ignore
    config.max_seq_len = llm_model.max_total_tokens
    config.max_input_len = llm_model.max_total_tokens
    config.max_attention_size = 2048**2
    config.compress_pos_emb = llm_model.compress_pos_emb
    config.gpu_peer_fix = llm_model.gpu_peer_fix
    config.auto_map = llm_model.auto_map
    config.matmul_fused_remap = llm_model.matmul_fused_remap
    config.fused_mlp_thd = llm_model.fused_mlp_thd
    config.sdp_thd = llm_model.sdp_thd
    config.fused_attn = llm_model.fused_attn
    config.matmul_fused_remap = llm_model.matmul_fused_remap
    config.rmsnorm_no_half2 = llm_model.rmsnorm_no_half2
    config.rope_no_half2 = llm_model.rope_no_half2
    config.matmul_fused_remap = llm_model.matmul_fused_remap
    config.silu_no_half2 = llm_model.silu_no_half2
    config.concurrent_streams = llm_model.concurrent_streams
    return config
