"""Wrapper for llama_cpp to generate text completions."""
# flake8: noqa
from concurrent.futures import ThreadPoolExecutor
import sys
from array import array
from inspect import signature
from typing import Iterator, List, Optional, Union

from ..mixins.completion import CompletionStatus
from ..schemas.api import (
    ChatCompletionChunk,
    CompletionChunk,
    CompletionLogprobs,
    TextGenerationSettings,
)
from ..schemas.models import LlamaCppModel
from ..shared.config import Config
from ..utils.dependency import import_repository
from ..utils.llama_cpp import build_shared_lib
from ..utils.logger import ApiLogger
from .base import BaseCompletionGenerator

logger = ApiLogger(__name__)
logger.info("🦙 llama-cpp-python repository found!")
with import_repository(**Config.repositories["llama_cpp"]):
    build_shared_lib(logger=logger)
    from repositories.llama_cpp import llama_cpp

StoppingCriteriaList = llama_cpp.StoppingCriteriaList
LogitsProcessorList = llama_cpp.LogitsProcessorList

try:
    LlamaGrammar = llama_cpp.LlamaGrammar
except AttributeError:
    LlamaGrammar = None


class LlamaCppCompletionGenerator(BaseCompletionGenerator):
    generator: Optional[
        Iterator[Union[CompletionChunk, ChatCompletionChunk]]
    ] = None
    client: Optional[llama_cpp.Llama] = None

    def __del__(self) -> None:
        self.destruct_model(logger, pytorch=False)

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
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                client = executor.submit(llama_cpp.Llama, **kwargs).result(
                    timeout=60
                )
        except AssertionError:
            raise MemoryError(
                "Failed to initialize llama.cpp model. "
                "Check if the model path is correct, or if the model is "
                "compatible with the current version of llama.cpp. "
                "e.g. GGML is not supported in newer versions of llama.cpp. "
                "Use GGUF instead."
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
        self = cls(llm_model)
        self.client = client
        return self

    def encode(self, text: str, add_bos: bool = True, **kwargs) -> List[int]:
        assert self.client is not None, "Client is not initialized"
        return self.client.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos
        )

    def decode(self, ids: List[int], **kwargs) -> str:
        assert self.client is not None, "Client is not initialized"
        return self.client.detokenize(ids).decode("utf-8", errors="ignore")

    def generate_text(
        self, prompt: str, settings: TextGenerationSettings
    ) -> Iterator[str]:
        client = self.client
        assert client is not None, "Llama is not initialized"
        self.llm_model.max_total_tokens = client.n_ctx()
        assert client.ctx is not None, "Llama context is not initialized"
        input_ids = array(
            "i",
            client.tokenize(prompt.encode("utf-8"))
            if prompt != ""
            else [client.token_bos()],
        )  # type: array[int]
        yield from self._generate_text(client, input_ids, settings)

    def _generate_text(
        self,
        client: llama_cpp.Llama,
        input_ids: "array[int]",
        settings: TextGenerationSettings,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> Iterator[str]:
        ctx = client.ctx
        assert ctx is not None, "Llama context is not initialized"
        assert isinstance(self.llm_model, LlamaCppModel), "Wrong llm_model"
        verbose = self.llm_model.verbose
        if verbose:
            llama_cpp.llama_reset_timings(ctx)

        # Cache the variables frequently used in the loop
        completion_status = self.completion_status[settings.completion_id]
        logit_processors = (
            LogitsProcessorList(
                [
                    processor.without_torch
                    for processor in self.get_logit_processors(
                        settings=settings, encoder=self.encode
                    )
                ]
            )
            or None
        )  # type: Optional[LogitsProcessorList]
        grammar = (
            LlamaGrammar.from_string(settings.grammar, verbose=verbose)
            if settings.grammar and LlamaGrammar
            else None
        )
        detokenize = client.detokenize
        generated_ids = array("i")  # type: array[int]
        byte_array = bytearray()  # type: bytearray
        eos_token_id = client.token_eos()
        logprobs = settings.logprobs
        text_buffer = ""  # type: str

        if logprobs is not None and client.params.logits_all is False:
            raise ValueError(
                "logprobs is not supported for models "
                "created with logits_all=False"
            )

        if client.cache:
            _load_cache(client, client.cache, input_ids)
        if self.check_interruption(completion_status):
            return
        assert settings.max_tokens is not None, "max_tokens must be set"
        for _, token_id in zip(
            range(settings.max_tokens),
            client.generate(
                input_ids,
                **{
                    key: value
                    for key, value in {
                        **self.llm_model.asdict,
                        **{
                            "temp": settings.temperature,
                            "stopping_criteria": stopping_criteria,
                            "logits_processor": logit_processors,
                            "grammar": grammar,
                        },
                    }.items()
                    # Hacky way to pass arguments safely to older versions of llama-cpp-python
                    if key in signature(client.generate).parameters.keys()
                },
            ),
        ):
            # Check if the token is a stop token
            if (
                self.check_interruption(completion_status)
                or token_id == eos_token_id
            ):
                break

            # Update the generated id
            generated_ids.append(token_id)
            completion_status.generated_tokens += 1

            piece = detokenize([token_id])  # type: bytes
            try:
                # Try to decode the token
                text_to_yield = text_buffer + (byte_array + piece).decode()
                byte_array.clear()
            except UnicodeDecodeError:
                # Multi-byte characters are not decoded correctly if partial
                byte_array.extend(piece)
                continue

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

        # End of the loop
        if verbose:
            llama_cpp.llama_print_timings(ctx)
        if client.cache:
            if verbose:
                print(
                    "Llama._create_completion: cache save", file=sys.stderr
                )
            client.cache[input_ids + generated_ids] = client.save_state()
            print("Llama._create_completion: cache saved", file=sys.stderr)
        return


def _load_cache(
    client: llama_cpp.Llama,
    cache: llama_cpp.BaseLlamaCache,
    ids: "array[int]",
) -> None:
    try:
        cache_item = cache[ids]
        cache_prefix_len = client.longest_token_prefix(
            cache_item.input_ids.tolist(), ids
        )
        eval_prefix_len = client.longest_token_prefix(
            client._input_ids.tolist(), ids
        )
        if cache_prefix_len > eval_prefix_len:
            client.load_state(cache_item)
            if client.verbose:
                print(
                    "Llama._create_completion: cache hit",
                    file=sys.stderr,
                )
    except KeyError:
        if client.verbose:
            print("Llama._create_completion: cache miss", file=sys.stderr)


def _get_log_probs(
    client: llama_cpp.Llama,
    completion_status: "CompletionStatus",
    prompt_tokens: int,
    generated_ids: "array[int]",
    generated_tokens: int,
    logprobs: int,
    token: int,
) -> CompletionLogprobs:
    assert client.ctx is not None, "Llama context is not initialized"
    token_str = client.detokenize([token]).decode("utf-8", errors="ignore")
    text_offset = len(completion_status.input_text) + len(
        completion_status.generated_text
    )
    token_offset = prompt_tokens + generated_tokens
    current_logprobs = client.logits_to_logprobs(
        client.scores[: client.n_tokens, :][token_offset - 1, :].tolist()
    )
    return {
        "tokens": [
            client.detokenize([token]).decode("utf-8", errors="ignore")
        ],
        "text_offset": [text_offset],
        "token_logprobs": [current_logprobs[int(token)]],
        "top_logprobs": [
            {
                **{
                    client.detokenize([i]).decode(
                        "utf-8", errors="ignore"
                    ): logprob
                    for logprob, i in list(
                        sorted(
                            zip(
                                current_logprobs,
                                range(len(current_logprobs)),
                            ),
                            reverse=True,
                        )
                    )[:logprobs]
                },
                token_str: current_logprobs[int(token)],
            }
        ],
    }
