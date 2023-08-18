"""Wrapper for exllama to generate text completions."""
# flake8: noqa
from os import environ

from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)
if environ.get("LLAMA_API_XFORMERS") == "1":
    with logger.log_any_error(
        "xformers mode is enabled, but xformers is not installed",
        suppress_exception=True,
    ):
        from ..modules.xformers import hijack_attention_forward

        hijack_attention_forward()
from gc import collect
from pathlib import Path
from time import time
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

from torch import IntTensor, Tensor, cuda, version
from torch.cuda import empty_cache
from torch.nn.functional import log_softmax

from ..logits.base import BaseLogitProcessor
from ..schemas.api import (
    APIChatMessage,
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    TextGenerationSettings,
)
from ..schemas.models import ExllamaModel
from ..shared.config import Config
from ..utils.dependency import import_repository
from ..utils.system import deallocate_memory
from .base import BaseCompletionGenerator
from .exllama_lora import ExLlamaLora

with import_repository(**Config.repositories["exllama"]):
    from repositories.exllama.generator import ExLlamaGenerator
    from repositories.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
    from repositories.exllama.tokenizer import ExLlamaTokenizer


assert cuda.is_available(), "CUDA must be available to use ExLlama."

_stop_checker = BaseCompletionGenerator.is_possible_to_generate_stops


def _make_config(
    model_folder_path: Path, llm_model: "ExllamaModel"
) -> ExLlamaConfig:
    """Create a config object for the ExLlama model."""

    # Find the model checkpoint
    model_file_found: List[Path] = []
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

    # Find the model checkpoint
    model_file_found: List[Path] = []
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

    config = ExLlamaConfig((model_folder_path / "config.json").as_posix())
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
    if llm_model.alpha_value is not None:
        config.alpha_value = llm_model.alpha_value
        config.calculate_rotary_embedding_base()
    if version.hip:
        config.rmsnorm_no_half2 = True
        config.rope_no_half2 = True
        config.matmul_no_half2 = True
        config.silu_no_half2 = True
    return config


def _apply_settings_to_generator(
    cg: "ExllamaCompletionGenerator",
    settings: "TextGenerationSettings",
) -> ExLlamaGenerator:
    """Apply the settings to the generator."""
    # Make sure that the batch size is correct
    required_batch_size = 1 if settings.guidance_scale <= 1 else 2
    cache_batch_size = cg.cache.batch_size  # type: int
    if cache_batch_size != required_batch_size:
        cg._cache = None
        deallocate_memory(cg._cache)
        cg._cache = ExLlamaCache(cg._model, batch_size=required_batch_size)
        cg._generator = ExLlamaGenerator(
            model=cg._model, tokenizer=cg._tokenizer, cache=cg._cache
        )
    # Temperature cannot be 0.0, so we use a very small value instead.
    # 0.0 will cause a division by zero error.
    generator = cg.generator
    generator.settings.temperature = settings.temperature or 0.01
    generator.settings.top_p = settings.top_p
    generator.settings.top_k = settings.top_k
    generator.settings.typical = settings.typical_p
    generator.settings.token_repetition_penalty_max = settings.repeat_penalty
    generator.settings.token_repetition_penalty_sustain = (
        -1
        if settings.repetition_penalty_range <= 0
        else settings.repetition_penalty_range
    )
    disallowed_tokens = (
        [generator.tokenizer.eos_token_id] if settings.ban_eos_token else None
    )
    generator.disallow_tokens(disallowed_tokens)
    return generator


def _gen_single_token_with_cfg(
    generator: ExLlamaGenerator, mask: Tensor, cfg_alpha: float
) -> int:
    logits = generator.model.forward(
        generator.sequence[:, -1:],
        cache=generator.cache,
        input_mask=mask,
    )  # type: Tensor  # type: ignore
    generator.apply_rep_penalty(logits)
    probs = log_softmax(logits, dim=-1)
    token, _ = generator.sample_current(
        cfg_alpha * probs[0] + (1 - cfg_alpha) * probs[1]
    )
    generator.gen_accept_token(token.repeat(2, 1))
    return int(token.item())


def _gen_single_token_without_cfg(
    generator: ExLlamaGenerator,
    initial_len: int,
    constraints: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    logit_processors: Optional[Iterable[BaseLogitProcessor]] = None,
) -> int:
    generator.end_beam_search()

    # Simple sampling case:
    if generator.sequence is not None:
        logits = generator.model.forward(
            generator.sequence[:, -1:],
            cache=generator.cache,
            lora=generator.lora,
            input_mask=mask,
        )  # type: Tensor  # type: ignore
        generator.apply_rep_penalty(logits)
        logits[:, :, generator.tokenizer.bos_token_id] = -10000.0

        if logit_processors is not None:
            input_ids = generator.sequence[0][initial_len:]
            for logit_processor in logit_processors:
                logits = logit_processor.with_torch(input_ids, logits)

        if constraints is not None:
            for constraint in constraints:
                logits[:, :, constraint] += 10000.0
            logits[:, :, :] -= 10000.0

        token, _ = generator.batched_sample(
            logits,
            generator.settings.temperature,
            generator.settings.top_k,
            generator.settings.top_p,
            generator.settings.min_p + 0.01
            if constraints is not None
            else 0.0,
            generator.settings.typical,
        )

    else:
        if constraints is not None:
            token = constraints[0]
        else:
            token = Tensor([[generator.tokenizer.bos_token_id]]).long()

    generator.gen_accept_token(token)
    return int(token.item())


def _generator(
    cg: "ExllamaCompletionGenerator",
    settings: "TextGenerationSettings",
    stops: List[str],
    cfg_mask: Optional[Tensor] = None,
) -> Iterator[str]:
    IdToPiece = cg.tokenizer.tokenizer.IdToPiece
    decoder = cg.tokenizer.decode
    generator = cg.generator

    cfg_alpha = settings.guidance_scale  # type: float
    initial_len = generator.sequence[0].shape[0]  # type: int
    eos_token_id = generator.tokenizer.eos_token_id  # type: int
    has_leading_space = False  # type: bool
    text_cursor = 0  # type: int
    n_tokens = 0  # type: int
    logit_processors = (
        [
            processor
            for processor in cg.get_logit_processors(
                settings=settings,
                encoder=cg.encode,
            )
        ]
        if cfg_mask is None
        else None
    )  # type: Optional[Iterable[BaseLogitProcessor]]
    for n_tokens in range(1, settings.max_tokens + 1):
        if cg.is_interrupted:
            break  # the generator was interrupted

        # Predict the next token id
        if cfg_mask is not None:
            token_id = _gen_single_token_with_cfg(
                generator, mask=cfg_mask, cfg_alpha=cfg_alpha
            )
        else:
            token_id = _gen_single_token_without_cfg(
                generator,
                initial_len=initial_len,
                logit_processors=logit_processors or None,
            )
        if cg.is_interrupted or token_id == eos_token_id:
            break

        # Yield the text piece
        if n_tokens == 1:
            has_leading_space = IdToPiece(token_id).startswith("▁")
        decoded_text = (
            " " + str(decoder(generator.sequence[0][initial_len:]))
            if has_leading_space
            else str(decoder(generator.sequence[0][initial_len:]))
        )
        text_piece = decoded_text[text_cursor:]
        if "�" in text_piece:  # Decode error when decoding multi-byte char
            continue
        if _stop_checker(text_piece, stops=stops):  # Stop token found maybe
            if any(stop in decoded_text for stop in stops):
                break  # Stop token found
            continue
        yield text_piece
        text_cursor += len(text_piece)
    # End of generation
    cg._completion_status[settings.completion_id] = n_tokens


def _generate_text_with_streaming(
    cg: "ExllamaCompletionGenerator",
    prompt: str,
    settings: "TextGenerationSettings",
) -> Iterator[str]:
    with logger.log_any_error():
        # Make sure that the stop token is a list
        if isinstance(settings.stop, str):
            stops = [settings.stop]  # type: List[str]
        elif isinstance(settings.stop, list):
            stops = settings.stop
        else:
            stops = []

        # Apply the settings to the generator
        generator = _apply_settings_to_generator(cg, settings=settings)

        # Apply the LORA model
        if cg.lora:
            generator.lora = cg.lora  # type: ignore

        # Start the generator
        context_window = cg.llm_model.max_total_tokens
        if settings.guidance_scale == 1:
            ids = _encode(cg.tokenizer, prompt)
            prompt_tokens = ids.shape[-1]
            cg.raise_for_token_limit(
                prompt_tokens=prompt_tokens, context_window=context_window
            )
            mask = None  # type: Optional[Tensor]
            generator.end_beam_search()
            generator.gen_begin_reuse(ids)
        else:
            ids, mask = _encode(
                cg.tokenizer,
                [prompt, settings.negative_prompt or ""],
                return_mask=True,
            )
            prompt_tokens = ids.shape[-1]
            cg.raise_for_token_limit(
                prompt_tokens=prompt_tokens, context_window=context_window
            )
            generator.gen_begin(ids, mask=mask)

        settings.max_tokens = min(
            settings.max_tokens, context_window - prompt_tokens
        )

        yield from _generator(
            cg, settings=settings, cfg_mask=mask, stops=stops
        )


class ExllamaCompletionGenerator(BaseCompletionGenerator):
    _config: Optional[ExLlamaConfig] = None
    _model: Optional[ExLlama] = None
    _cache: Optional[ExLlamaCache] = None
    _tokenizer: Optional[ExLlamaTokenizer] = None
    _generator: Optional[ExLlamaGenerator] = None
    _llm_model: Optional["ExllamaModel"] = None
    _lora: Optional["ExLlamaLora"] = None
    _completion_status: Dict[
        str, int
    ] = {}  # key: completion_id, value: number of completion tokens

    @property
    def llm_model(self) -> "ExllamaModel":
        assert self._llm_model is not None
        return self._llm_model

    @property
    def generator(self) -> ExLlamaGenerator:
        assert self._generator is not None, "Generator is not initialized."
        return self._generator

    @property
    def tokenizer(self) -> ExLlamaTokenizer:
        assert self._tokenizer is not None, "Tokenizer is not initialized."
        return self._tokenizer

    @property
    def cache(self) -> ExLlamaCache:
        assert self._cache is not None, "Cache is not initialized."
        return self._cache

    @property
    def model(self) -> ExLlama:
        assert self._model is not None, "Model is not initialized."
        return self._model

    @property
    def config(self) -> ExLlamaConfig:
        assert self._config is not None, "Config is not initialized."
        return self._config

    @property
    def lora(self) -> Optional[ExLlamaLora]:
        return self._lora

    @classmethod
    def from_pretrained(
        cls, llm_model: "ExllamaModel"
    ) -> "ExllamaCompletionGenerator":
        model_folder_path = Path(llm_model.model_path_resolved)
        lora_path = model_folder_path / "adapter_model.bin"
        lora_config_path = model_folder_path / "adapter_config.json"

        result = cls()
        result._llm_model = llm_model
        result._config = _make_config(model_folder_path, llm_model)
        result._tokenizer = ExLlamaTokenizer(
            (model_folder_path / "tokenizer.model").as_posix()
        )
        result._model = ExLlama(result._config)
        if lora_path.exists() and lora_config_path.exists():
            logger.info(f"🦙 LORA model found for {result.model_name}")
            with logger.log_any_error(
                f"🦙 LORA model loading failed for {result.model_name}"
            ):
                result._lora = ExLlamaLora(
                    model=result._model,
                    lora_config_path=lora_config_path.as_posix(),
                    lora_path=lora_path.as_posix(),
                )
            logger.info(f"🦙 LORA model loaded for {result.model_name}")
        result._cache = ExLlamaCache(result._model)
        result._generator = ExLlamaGenerator(
            result._model, result._tokenizer, result._cache
        )
        return result

    def generate_completion_with_streaming(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> Iterator["CompletionChunk"]:
        completion_id = settings.completion_id
        model = self.model_name
        generated_text = ""  # type: str
        for token in _generate_text_with_streaming(
            self, prompt=prompt, settings=settings
        ):
            generated_text += token
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time()),
                "model": model,
                "choices": [
                    {
                        "text": token,
                        "index": 0,
                        "logprobs": None,
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
                    "logprobs": None,
                    "finish_reason": _get_finish_reason(
                        self, settings, completion_id, generated_text
                    ),
                }
            ],
        }

    def generate_completion(
        self, prompt: str, settings: "TextGenerationSettings"
    ) -> "Completion":
        completion_id = settings.completion_id
        generated_text = "".join(
            _generate_text_with_streaming(
                self, prompt=prompt, settings=settings
            )
        )
        prompt_tokens = _encode(self.tokenizer, prompt).shape[1]
        completion_tokens = self._completion_status.get(
            completion_id, _encode(self.tokenizer, generated_text).shape[1]
        )
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": _get_finish_reason(
                        self, settings, completion_id, generated_text
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def generate_chat_completion_with_streaming(
        self,
        messages: List["APIChatMessage"],
        settings: "TextGenerationSettings",
    ) -> Iterator["ChatCompletionChunk"]:
        completion_id = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        model = self.model_name
        generated_text = ""  # type: str
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
        for token in _generate_text_with_streaming(
            self, prompt=prompt, settings=settings
        ):
            generated_text += token
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
                    "finish_reason": _get_finish_reason(
                        self, settings, completion_id, generated_text
                    ),
                }
            ],
        }

    def generate_chat_completion(
        self,
        messages: List["APIChatMessage"],
        settings: "TextGenerationSettings",
    ) -> "ChatCompletion":
        completion_id = settings.completion_id
        prompt = self.convert_messages_into_prompt(messages, settings=settings)
        generated_text = "".join(
            _generate_text_with_streaming(
                self, prompt=prompt, settings=settings
            )
        )
        prompt_tokens = _encode(self.tokenizer, prompt).shape[1]
        completion_tokens = self._completion_status.get(
            completion_id, _encode(self.tokenizer, generated_text).shape[1]
        )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time()),
            "model": self.model_name,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "index": 0,
                    "finish_reason": _get_finish_reason(
                        self, settings, completion_id, generated_text
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def encode(self, text: str) -> List[int]:
        assert self._tokenizer is not None, "Tokenizer is not initialized"
        return _encode(self._tokenizer, text).flatten().tolist()

    def decode(self, ids: List[int], **kwargs) -> str:
        assert self._tokenizer is not None, "Tokenizer is not initialized"
        return str(self._tokenizer.decode(IntTensor(ids)))

    def __del__(self) -> None:
        if self._tokenizer is not None:
            getattr(self._tokenizer, "__del__", lambda: None)()
            del self._tokenizer
            self._tokenizer = None
            logger.info("🗑️ ExllamaCompletionGenerator tokenizer deleted")
        if self._cache is not None:
            getattr(self._cache, "__del__", lambda: None)()
            del self._cache
            self._cache = None
            logger.info("🗑️ ExllamaCompletionGenerator cache deleted")
        if self._generator is not None:
            getattr(self._generator, "__del__", lambda: None)()
            del self._generator
            self._generator = None
            logger.info("🗑️ ExllamaCompletionGenerator generator deleted")
        if self._model is not None:
            self._model.free_unmanaged()
            del self._model
            self._model = None
            logger.info("🗑️ ExllamaCompletionGenerator model deleted")
        collect()
        empty_cache()


@overload
def _encode(
    tokenizer: ExLlamaTokenizer,
    text: str,
    return_mask: bool = False,
) -> Tensor:
    ...


@overload
def _encode(
    tokenizer: ExLlamaTokenizer,
    text: List[str],
    return_mask: bool = True,
) -> Tuple[Tensor, Tensor]:
    ...


def _encode(
    tokenizer: ExLlamaTokenizer,
    text: Union[str, List[str]],
    return_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Encode a text string into a tensor."""
    result = tokenizer.encode(text, return_mask=return_mask)
    if return_mask:
        ids, mask = result
        assert isinstance(ids, Tensor) and isinstance(mask, Tensor)
        return ids, mask
    else:
        ids = result[0] if isinstance(result, tuple) else result
        assert isinstance(ids, Tensor)
        return ids


def _get_finish_reason(
    cg: ExllamaCompletionGenerator,
    settings: "TextGenerationSettings",
    completion_id: str,
    generated_text: str,
) -> Literal["length", "stop"]:
    return (
        "length"
        if cg._completion_status.get(
            completion_id,
            _encode(cg.tokenizer, generated_text).shape[1],
        )
        >= settings.max_tokens
        else "stop"
    )
