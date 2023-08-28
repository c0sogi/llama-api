"""Wrapper for exllama to generate text completions."""
# flake8: noqa
from array import array
from os import environ

from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)
# if environ.get("XFORMERS") == "1":
#     with logger.log_any_error(
#         "xformers mode is enabled, but xformers is not installed",
#         suppress_exception=True,
#     ):
#         from ..modules.xformers import hijack_attention_forward

#         hijack_attention_forward()
from gc import collect
from pathlib import Path
from re import compile
from typing import Iterable, Iterator, List, Optional, Tuple, Union, overload

from torch import IntTensor, Tensor, cuda, version
from torch.cuda import empty_cache
from torch.nn.functional import log_softmax

from ..logits.base import BaseLogitProcessor
from ..schemas.api import TextGenerationSettings
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


class ExllamaCompletionGenerator(BaseCompletionGenerator):
    _config: Optional[ExLlamaConfig] = None
    _model: Optional[ExLlama] = None
    _cache: Optional[ExLlamaCache] = None
    _tokenizer: Optional[ExLlamaTokenizer] = None
    _generator: Optional[ExLlamaGenerator] = None
    _llm_model: Optional["ExllamaModel"] = None
    _lora: Optional["ExLlamaLora"] = None
    _byte_pattern = compile(r"<0x([0-9a-fA-F]{2})>")

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
        if self._lora is not None:
            getattr(self._lora, "__del__", lambda: None)()
            del self._lora
            self._lora = None
            logger.info("🗑️ ExllamaCompletionGenerator lora deleted")
        if self._model is not None:
            self._model.free_unmanaged()
            del self._model
            self._model = None
            logger.info("🗑️ ExllamaCompletionGenerator model deleted")
        collect()
        empty_cache()

    def generate_text(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[str]:
        with logger.log_any_error():
            # Encode the prompt
            if settings.guidance_scale == 1:
                ids = _encode(self.tokenizer, prompt or " ")
                mask = None  # type: Optional[Tensor]
            else:
                ids, mask = _encode(
                    self.tokenizer,
                    [prompt or " ", settings.negative_prompt or ""],
                    return_mask=True,
                )

            # Accept and apply the settings
            self.accept_settings(
                prompt=prompt,
                prompt_tokens=ids.shape[-1],
                settings=settings,
            )
            generator = _apply_settings_to_generator(self, settings=settings)

            # Apply LoRA
            if self.lora:
                generator.lora = self.lora  # type: ignore

            # Inject the prompt
            if mask is not None:
                generator.gen_begin(ids, mask=mask)
            else:
                generator.end_beam_search()
                generator.gen_begin_reuse(ids)

            # Generate text
            yield from self._generate_text(settings, mask)

    def _generate_text(
        self,
        settings: TextGenerationSettings,
        cfg_mask: Optional[Tensor] = None,
    ) -> Iterator[str]:
        # Set up the variables
        IdToPiece = self.tokenizer.tokenizer.IdToPiece
        generator = self.generator
        initial_len = generator.sequence[0].shape[0]  # type: int
        eos_token_id = generator.tokenizer.eos_token_id  # type: int
        completion_status = self.completion_status[settings.completion_id]
        text_buffer = ""  # type: str
        byte_array = array("B")  # type: array[int]
        byte_pattern = self._byte_pattern
        logit_processors = (
            [
                processor
                for processor in self.get_logit_processors(
                    settings=settings, encoder=self.encode
                )
            ]
            if cfg_mask is None
            else None
        ) or None

        for _ in range(settings.max_tokens):
            # If the generator was interrupted, stop the generation
            if self.check_interruption(completion_status):
                break

            # Predict next token id
            token_id = (
                _gen_single_token_with_cfg(
                    generator=generator,
                    mask=cfg_mask,
                    cfg_alpha=settings.guidance_scale,
                )
                if cfg_mask is not None
                else _gen_single_token_without_cfg(
                    generator=generator,
                    input_ids=generator.sequence[0][initial_len:],
                    logit_processors=logit_processors,
                )
            )  # type: int

            # Check if the token is a stop token
            if (
                self.check_interruption(completion_status)
                or token_id == eos_token_id
            ):
                break

            # Update the completion status
            completion_status.generated_tokens += 1

            # Try to decode the token
            piece = IdToPiece(token_id)  # type: str
            if piece[0] == "<" and piece[-1] == ">":
                byte_match = byte_pattern.match(piece)
                if byte_match is None:
                    continue
                try:
                    byte_array.append(int(byte_match.group(1), 16))
                    piece = byte_array.tobytes().decode()
                    del byte_array[:]
                except UnicodeDecodeError:
                    continue
            text_to_yield = text_buffer + piece.replace("▁", " ")

            # Check if the decoded text contains any of the stop tokens.
            stop_status = self.stop_checker(text_to_yield)
            if stop_status is None:  # Good to go
                text_buffer = ""  # Clear the buffer
                completion_status.generated_text += text_to_yield
                yield text_to_yield
            elif stop_status is True:  # Contains any of the stop tokens
                break  # Stop generating
            else:  # Contains any piece of the stop tokens
                text_buffer = text_to_yield  # Save the buffer


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
    settings: TextGenerationSettings,
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
    input_ids: Tensor,
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


@overload
def _encode(
    tokenizer: ExLlamaTokenizer, text: str, return_mask: bool = False
) -> Tensor:
    ...


@overload
def _encode(
    tokenizer: ExLlamaTokenizer, text: List[str], return_mask: bool = True
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
