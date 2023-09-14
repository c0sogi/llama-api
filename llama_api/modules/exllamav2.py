"""Wrapper for exllama to generate text completions."""
# flake8: noqa

from array import array
from pathlib import Path

from random import random
from re import compile
from typing import Iterator, List

from torch import IntTensor, cat, cuda

from ..schemas.api import TextGenerationSettings
from ..schemas.models import ExllamaModel
from ..shared.config import Config
from ..utils.dependency import import_repository
from ..utils.logger import ApiLogger
from .base import BaseCompletionGenerator

logger = ApiLogger(__name__)
assert cuda.is_available(), "CUDA must be available to use ExLlama."
with logger.log_any_error("Error importing ExLlamaV2"):
    with import_repository(**Config.repositories["exllamav2"]):
        from repositories.exllamav2.exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
        )
        from repositories.exllamav2.exllamav2.generator import (
            ExLlamaV2BaseGenerator,
            ExLlamaV2Sampler,
        )


class ExllamaV2CompletionGenerator(BaseCompletionGenerator):
    config: ExLlamaV2Config
    model: ExLlamaV2
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer
    generator: ExLlamaV2BaseGenerator
    _byte_pattern = compile(r"<0x([0-9a-fA-F]{2})>")

    @classmethod
    def from_pretrained(
        cls, llm_model: "ExllamaModel"
    ) -> "ExllamaV2CompletionGenerator":
        model_folder_path = Path(llm_model.model_path_resolved)
        lora_path = model_folder_path / "adapter_model.bin"
        lora_config_path = model_folder_path / "adapter_config.json"
        self = cls(llm_model)

        # Config: Load required parameters
        config = ExLlamaV2Config()
        config.model_dir = model_folder_path.as_posix()
        config.max_seq_len = llm_model.max_total_tokens
        config.max_input_len = llm_model.max_total_tokens
        # Config: Optional parameters for NTK RoPE scaling
        if llm_model.alpha_value is not None:
            config.scale_alpha_value = llm_model.alpha_value
            config.scale_pos_emb = llm_model.compress_pos_emb
            logger.info(
                f"Rotary embedding base has been set to {config.rotary_embedding_base}"
            )
        config.prepare()
        self.config = config

        self.model = ExLlamaV2(config)
        gpu_splits, vram_usage = self.model.load(
            llm_model.auto_map, stats=True
        )  # type: ignore
        logger.debug(
            f"\n- GPU splits: {gpu_splits}"
            f"\n- VRAM usages: {vram_usage} MB"
        )
        self.cache = ExLlamaV2Cache(self.model)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.generator = ExLlamaV2BaseGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )
        if lora_path.exists() and lora_config_path.exists():
            logger.info(
                f"🦙 LORA model found for {self.model_name},"
                "but it is not loaded because ExLlamaV2 does not support LORA yet."
            )
        return self

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).flatten().tolist()

    def decode(self, ids: List[int], **kwargs) -> str:
        return str(self.tokenizer.decode(IntTensor(ids)))

    def __del__(self) -> None:
        self.destruct_model(logger, pytorch=True)

    def generate_text(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[str]:
        with logger.log_any_error():
            # Set up the variables
            IdToPiece = self.tokenizer.tokenizer.IdToPiece
            eos_token_id = self.tokenizer.eos_token_id  # type: int
            completion_status = self.completion_status[
                settings.completion_id
            ]
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
            ) or None

            # Encode the prompt and inject the input ids
            input_ids = self.tokenizer.encode(prompt or " ")
            self.cache.current_seq_len = 0
            self.model.forward(
                input_ids[:, :-1],
                self.generator.cache,
                input_mask=None,
                preprocess_only=True,
            )

            # Make sampler settings
            sampler_settings = ExLlamaV2Sampler.Settings()
            sampler_settings.temperature = settings.temperature or 0.01
            sampler_settings.top_k = settings.top_k
            sampler_settings.top_p = settings.top_p
            sampler_settings.token_repetition_penalty = (
                settings.repeat_penalty
            )
            sampler_settings.token_repetition_range = (
                -1
                if settings.repetition_penalty_range <= 0
                else settings.repetition_penalty_range
            )
            if settings.ban_eos_token:
                sampler_settings.disallow_tokens(
                    self.tokenizer, [self.tokenizer.eos_token_id]
                )
            sampler = ExLlamaV2Sampler.sample

            # Generate text
            assert settings.max_tokens is not None, "max_tokens must be set"
            for _ in range(settings.max_tokens):
                # If the generator was interrupted, stop the generation
                if self.check_interruption(completion_status):
                    return

                # Predict next token id
                try:
                    logits = (
                        self.model.forward(
                            input_ids[:, -1:], self.cache, input_mask=None
                        )
                        .float()  # type: ignore
                        .cpu()
                    )
                    if logit_processors is not None:
                        for logit_processor in logit_processors:
                            logits = logit_processor.with_torch(
                                input_ids, logits
                            )
                    token, _ = sampler(
                        logits, sampler_settings, input_ids, random()
                    )
                    input_ids = cat([input_ids, token], dim=1)
                    token_id = token.item()
                except RuntimeError as e:
                    if "exceeds dimension size" in str(e):
                        logger.warning(
                            f"Ignoring ExLlamaV2 RuntimeError: {e}"
                        )
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
