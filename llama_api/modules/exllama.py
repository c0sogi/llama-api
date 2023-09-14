"""Wrapper for exllama to generate text completions."""
# flake8: noqa

# if environ.get("XFORMERS") == "1":
#     with logger.log_any_error(
#         "xformers mode is enabled, but xformers is not installed",
#         suppress_exception=True,
#     ):
#         from ..modules.xformers import hijack_attention_forward
#         hijack_attention_forward()

from array import array
from functools import partial
from pathlib import Path
from re import compile
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

from torch import IntTensor, Tensor, cuda, version
from torch.nn.functional import log_softmax

from ..logits.base import BaseLogitProcessor
from ..schemas.api import TextGenerationSettings
from ..schemas.models import ExllamaModel
from ..shared.config import Config
from ..utils.dependency import import_repository
from ..utils.exllama_utils import get_model_path

from ..utils.logger import ApiLogger
from ..utils.system_utils import deallocate_memory
from .base import BaseCompletionGenerator
from .exllama_lora import ExLlamaLora

logger = ApiLogger(__name__)
assert cuda.is_available(), "CUDA must be available to use ExLlama."
with logger.log_any_error("Error importing ExLlama"):
    with import_repository(**Config.repositories["exllama"]):
        from repositories.exllama.generator import ExLlamaGenerator
        from repositories.exllama.model import (
            ExLlama,
            ExLlamaCache,
            ExLlamaConfig,
        )
        from repositories.exllama.tokenizer import ExLlamaTokenizer


class ExllamaCompletionGenerator(BaseCompletionGenerator):
    config: ExLlamaConfig
    model: ExLlama
    cache: ExLlamaCache
    tokenizer: ExLlamaTokenizer
    generator: ExLlamaGenerator
    lora: Optional["ExLlamaLora"] = None
    _byte_pattern = compile(r"<0x([0-9a-fA-F]{2})>")

    @classmethod
    def from_pretrained(
        cls, llm_model: "ExllamaModel"
    ) -> "ExllamaCompletionGenerator":
        model_folder_path = Path(llm_model.model_path_resolved)
        lora_path = model_folder_path / "adapter_model.bin"
        lora_config_path = model_folder_path / "adapter_config.json"
        self = cls(llm_model)

        # Config: Load required parameters
        config = ExLlamaConfig(
            (model_folder_path / "config.json").as_posix()
        )
        config.model_path = get_model_path(model_folder_path)  # type: ignore
        config.max_seq_len = llm_model.max_total_tokens
        config.max_input_len = llm_model.max_total_tokens
        config.max_attention_size = 2048**2
        config.compress_pos_emb = llm_model.compress_pos_emb
        config.gpu_peer_fix = llm_model.gpu_peer_fix
        config.auto_map = llm_model.auto_map
        # Config: Optional parameters for tuning
        config.use_flash_attn_2 = llm_model.use_flash_attn_2
        config.matmul_recons_thd = llm_model.matmul_recons_thd
        config.fused_mlp_thd = llm_model.fused_mlp_thd
        config.sdp_thd = llm_model.sdp_thd
        config.fused_attn = llm_model.fused_attn
        config.matmul_fused_remap = llm_model.matmul_fused_remap
        config.rmsnorm_no_half2 = llm_model.rmsnorm_no_half2
        config.rope_no_half2 = llm_model.rope_no_half2
        config.matmul_no_half2 = llm_model.matmul_no_half2
        config.silu_no_half2 = llm_model.silu_no_half2
        config.concurrent_streams = llm_model.concurrent_streams
        # Config: Optional parameters for NTK RoPE scaling
        if llm_model.alpha_value is not None:
            config.alpha_value = llm_model.alpha_value
            config.calculate_rotary_embedding_base()
            logger.info(
                f"Rotary embedding base has been set to {config.rotary_embedding_base}"
            )
        # Config: For ROCm (AMD GPUs)
        if version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True
        self.config = config

        self.model = ExLlama(self.config)
        if lora_path.exists() and lora_config_path.exists():
            logger.info(f"🦙 LORA model found for {self.model_name}")
            with logger.log_any_error(
                f"🦙 LORA model loading failed for {self.model_name}"
            ):
                self.lora = ExLlamaLora(
                    model=self.model,
                    lora_config_path=lora_config_path.as_posix(),
                    lora_path=lora_path.as_posix(),
                )
            logger.info(f"🦙 LORA model loaded for {self.model_name}")
        self.cache = ExLlamaCache(self.model)
        self.tokenizer = ExLlamaTokenizer(
            (model_folder_path / "tokenizer.model").as_posix()
        )
        self.generator = ExLlamaGenerator(
            model=self.model, tokenizer=self.tokenizer, cache=self.cache
        )
        return self

    def encode(self, text: str) -> List[int]:
        return _encode(self.tokenizer, text).flatten().tolist()

    def decode(self, ids: List[int], **kwargs) -> str:
        return str(self.tokenizer.decode(IntTensor(ids)))

    def __del__(self) -> None:
        self.destruct_model(logger, pytorch=True)

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

            # Apply settings to the generator
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
        eos_token_id = self.tokenizer.eos_token_id  # type: int
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

        initial_len = self.generator.sequence[0].shape[0]  # type: int

        assert settings.max_tokens is not None, "max_tokens must be set"
        for _ in range(settings.max_tokens):
            # If the generator was interrupted, stop the generation
            if self.check_interruption(completion_status):
                return

            # Predict next token id
            try:
                token_id = (
                    _gen_single_token_with_cfg(
                        generator=self.generator,
                        mask=cfg_mask,
                        cfg_alpha=settings.guidance_scale,
                    )
                    if cfg_mask is not None
                    else _gen_single_token_without_cfg(
                        generator=self.generator,
                        initial_len=initial_len,
                        logit_processors=logit_processors,
                    )
                )  # type: int
            except RuntimeError as e:
                if "exceeds dimension size" in str(e):
                    logger.warning(f"Ignoring ExLlama RuntimeError: {e}")
                    return
                raise e
            # Check if the token is a stop token
            if (
                self.check_interruption(completion_status)
                or token_id == eos_token_id
            ):
                return

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
                return  # Stop generating
            else:  # Contains any piece of the stop tokens
                text_buffer = text_to_yield  # Save the buffer


def _apply_settings_to_generator(
    cg: "ExllamaCompletionGenerator",
    settings: TextGenerationSettings,
) -> ExLlamaGenerator:
    """Apply the settings to the generator."""
    # Make sure that the batch size is correct
    required_batch_size = 1 if settings.guidance_scale <= 1 else 2
    cache_batch_size = cg.cache.batch_size  # type: int
    if cache_batch_size != required_batch_size:
        deallocate_memory(cg, "cache", pytorch=True)
        cg.cache = ExLlamaCache(cg.model, batch_size=required_batch_size)
        cg.generator = ExLlamaGenerator(
            model=cg.model, tokenizer=cg.tokenizer, cache=cg.cache
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
        [generator.tokenizer.eos_token_id]
        if settings.ban_eos_token
        else None
    )
    generator.disallow_tokens(disallowed_tokens)
    return generator


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
            for logit_processor in logit_processors:
                logits = logit_processor.with_torch(
                    generator.sequence[0][initial_len:], logits
                )

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
