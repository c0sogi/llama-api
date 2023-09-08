from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from multiprocessing import current_process
from os import getpid
from queue import Queue
from threading import Event
from time import time
from typing import Literal  # noqa: F401
from typing import Deque, Iterator, List, Optional, Union

from orjson import OPT_INDENT_2, dumps

from ...mixins.completion import CompletionStatus
from ...modules.base import (
    BaseCompletionGenerator,
    BaseEmbeddingGenerator,
    BaseLLMModel,
)
from ...schemas.api import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    Embedding,
    EmbeddingData,
    EmbeddingUsage,
)
from ...schemas.models import ExllamaModel, LlamaCppModel
from ...utils.concurrency import queue_manager
from ...utils.lazy_imports import LazyImports
from ...utils.logger import ApiLogger, LoggingConfig
from ...utils.model_definition_finder import ModelDefinitions
from ...utils.system import free_memory_of_first_item_from_container

logger = ApiLogger(__name__)
logger.info(f"🔧 {current_process()} is initiated with PID: {getpid()}")
chat_logger = ApiLogger(
    "",
    logging_config=LoggingConfig(
        console_log_level=100,
        file_log_name=f"./logs/{date.today().strftime('%Y-%m-%d')}-chat.log",
        color=False,
    ),
)
lazy = LazyImports()  # lazy-loader of modules
completion_generators: Deque["BaseCompletionGenerator"] = deque(maxlen=1)
embedding_generators: Deque["BaseEmbeddingGenerator"] = deque(maxlen=1)


@dataclass
class EmbeddingStatus:
    started_at: float = field(default_factory=time, init=False)
    state: Literal["done", "interrupted"] = field(default="done", init=False)
    embedding: Optional[Embedding] = None


@contextmanager
def handle_exception():
    try:
        yield
    except RuntimeError as e:
        if "cuda" in str(e).lower():
            raise MemoryError(f"CUDA error: {e}")
        raise e
    except OSError as e:
        if "access violation" in str(e).lower():
            raise MemoryError(f"Access violation: {e}")
        raise e


@contextmanager
def completion_generator_manager(
    body: Union[CreateCompletionRequest, CreateChatCompletionRequest],
    llm_model: BaseLLMModel,
    interrupt_signal: Event,
):
    """Context manager for completion generators."""
    with handle_exception():
        cg = None  # To avoid UnboundLocalError
        try:
            cg = get_completion_generator(llm_model)
            cg.interrupt_signal = interrupt_signal
            cg.acquire_lock()
            yield cg
            log_request_and_response(
                body,
                cg.completion_status[body.completion_id],
            )
        finally:
            if cg is not None:
                cg.release_lock()
                cg.interrupt_signal = None


def get_completion_generator(
    llm_model: BaseLLMModel,
) -> BaseCompletionGenerator:
    """Get a completion generator for the given model.
    If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""

    # Check if the model is an OpenAI model
    with logger.log_any_error(
        f"Error getting a completion generator of {llm_model.model_path}",
    ):
        # Check if the model is defined in LLMModels enum

        # Check if the model is cached. If so, return the cached one.
        for completion_generator in completion_generators:
            if (
                completion_generator.llm_model.model_path
                == llm_model.model_path
            ):
                return completion_generator

        # Before creating new one, deallocate embeddings to free up memory
        if embedding_generators:
            free_memory_of_first_item_from_container(
                embedding_generators, logger=logger
            )

        # Before creating a new completion generator, check memory usage
        if completion_generators.maxlen == len(completion_generators):
            free_memory_of_first_item_from_container(
                completion_generators, logger=logger
            )

        # Create a new completion generator
        if isinstance(llm_model, LlamaCppModel):
            assert not isinstance(
                lazy.LlamaCppCompletionGenerator, Exception
            ), lazy.LlamaCppCompletionGenerator
            to_return = lazy.LlamaCppCompletionGenerator.from_pretrained(
                llm_model
            )
        elif isinstance(llm_model, ExllamaModel):
            assert not isinstance(
                lazy.ExllamaCompletionGenerator, Exception
            ), lazy.ExllamaCompletionGenerator
            to_return = lazy.ExllamaCompletionGenerator.from_pretrained(
                llm_model
            )
        else:
            raise NotImplementedError(
                f"Model {llm_model.model_path} not implemented"
            )

        # Add the new completion generator to the deque cache
        completion_generators.append(to_return)
        return to_return


def get_embedding_generator(
    body: CreateEmbeddingRequest,
) -> BaseEmbeddingGenerator:
    """Get an embedding generator for the given model.
    If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""

    with logger.log_any_error(
        f"Error getting a embedding generator of {body.model}"
    ):
        body.model = body.model.lower()
        for embedding_generator in embedding_generators:
            if embedding_generator.model_name == body.model:
                return embedding_generator

        # Before creating a new completion generator, check memory usage
        if embedding_generators.maxlen == len(embedding_generators):
            free_memory_of_first_item_from_container(
                embedding_generators, logger=logger
            )
        # Before creating a new, deallocate embeddings to free up memory
        if completion_generators:
            free_memory_of_first_item_from_container(
                completion_generators, logger=logger
            )

        if "sentence" in body.model and "encoder" in body.model:
            # Create a new sentence encoder embedding
            assert not isinstance(
                lazy.SentenceEncoderEmbeddingGenerator, Exception
            ), lazy.SentenceEncoderEmbeddingGenerator
            to_return = (
                lazy.SentenceEncoderEmbeddingGenerator.from_pretrained(
                    body.model
                )
            )
        else:
            # Create a new transformer embedding
            assert not isinstance(
                lazy.TransformerEmbeddingGenerator, Exception
            ), lazy.LlamaCppCompletionGenerator
            to_return = lazy.TransformerEmbeddingGenerator.from_pretrained(
                body.model
            )

        # Add the new completion generator to the deque cache
        embedding_generators.append(to_return)
        return to_return


def generate_completion_chunks(
    body: Union[CreateChatCompletionRequest, CreateCompletionRequest],
    llm_model: BaseLLMModel,
    queue: Queue,
    interrupt_signal: Event,
) -> None:
    with queue_manager(queue=queue):
        with completion_generator_manager(
            body, llm_model, interrupt_signal
        ) as cg:
            if isinstance(body, CreateChatCompletionRequest):
                _iterator: Iterator[
                    Union[ChatCompletionChunk, CompletionChunk]
                ] = cg.generate_chat_completion_with_streaming(body)
            elif isinstance(body, CreateCompletionRequest):
                _iterator = cg.generate_completion_with_streaming(body)
            first_response: Union[
                ChatCompletionChunk, CompletionChunk
            ] = next(_iterator)

            def iterator() -> (
                Iterator[Union[ChatCompletionChunk, CompletionChunk]]
            ):
                yield first_response
                for chunk in _iterator:
                    yield chunk

            for chunk in iterator():
                if interrupt_signal.is_set():
                    # If the event is set, the client is disconnected
                    return
                queue.put(chunk)


def generate_completion(
    body: Union[CreateChatCompletionRequest, CreateCompletionRequest],
    llm_model: BaseLLMModel,
    queue: Queue,
    interrupt_signal: Event,
) -> None:
    with queue_manager(queue=queue):
        with completion_generator_manager(
            body, llm_model, interrupt_signal
        ) as cg:
            if isinstance(body, CreateChatCompletionRequest):
                completion: Union[
                    ChatCompletion, Completion
                ] = cg.generate_chat_completion(body)
            elif isinstance(body, CreateCompletionRequest):
                completion = cg.generate_completion(body)
            queue.put(completion)


def generate_embeddings(body: CreateEmbeddingRequest, queue: Queue) -> None:
    embedding_status = EmbeddingStatus()
    with queue_manager(queue=queue):
        try:
            llm_model = ModelDefinitions.get_llm_model_from_request_body(
                body
            )
            if not isinstance(llm_model, LlamaCppModel):
                raise NotImplementedError("Using non-llama-cpp model")
        except Exception:
            # Embedding model from local
            #     "intfloat/e5-large-v2",
            #     "hkunlp/instructor-xl",
            #     "hkunlp/instructor-large",
            #     "intfloat/e5-base-v2",
            #     "intfloat/e5-large",
            embedding_generator: "BaseEmbeddingGenerator" = (
                get_embedding_generator(body)
            )
            embeddings: List[
                List[float]
            ] = embedding_generator.generate_embeddings(
                texts=body.input
                if isinstance(body.input, list)
                else [body.input],
                context_length=512,
                batch=1000,
            )
            embedding = Embedding(
                object="list",
                data=[
                    EmbeddingData(
                        index=embedding_idx,
                        object="embedding",
                        embedding=embedding,
                    )
                    for embedding_idx, embedding in enumerate(embeddings)
                ],
                model=body.model,
                usage=EmbeddingUsage(
                    prompt_tokens=-1,
                    total_tokens=-1,
                ),
            )

        else:
            # Trying to get embedding model from Llama.cpp
            assert getattr(llm_model, "embedding", False), (
                "Model does not support embeddings. "
                "Set `embedding` to True in the LlamaCppModel"
            )
            assert not isinstance(
                lazy.LlamaCppCompletionGenerator, Exception
            ), lazy.LlamaCppCompletionGenerator
            completion_generator = get_completion_generator(llm_model)
            assert isinstance(
                completion_generator, lazy.LlamaCppCompletionGenerator
            ), f"Model {body.model} is not supported for llama.cpp embeddings."
            assert completion_generator.client, "Model not loaded yet."
            embedding = completion_generator.client.create_embedding(
                **body.model_dump(exclude={"user"})
            )
        queue.put(embedding)
        embedding_status.embedding = embedding
        log_request_and_response(body, embedding_status)


def log_request_and_response(
    body: Union[
        CreateChatCompletionRequest,
        CreateCompletionRequest,
        CreateEmbeddingRequest,
    ],
    status: Optional[Union[CompletionStatus, EmbeddingStatus]],
) -> None:
    """Log the request and response of the completion or embedding"""
    # If the status is None, then the request has been interrupted
    if status is None:
        return

    # Measure the elapsed time, and get information about the request
    elapsed_time = time() - status.started_at
    logs: List[str] = [f"elapsed time: {elapsed_time: .1f}s"]
    body_without_prompt = body.model_dump(
        exclude={"prompt", "messages", "input"},
        exclude_defaults=True,
        exclude_unset=True,
        exclude_none=True,
    )

    # Log the embedding status
    if isinstance(status, EmbeddingStatus) and isinstance(
        body, CreateEmbeddingRequest
    ):
        # Embedding usage is the number of characters in the input
        # and the number of chunks in the embedding
        embed_usage = {
            "input_chars": len(body.input),
            "embedding_chunks": len(status.embedding["data"])
            if status.embedding
            else 0,
        }  # type: dict[str, int]
        logs.append(f"embedding chunks: {embed_usage}")
        embed_log = {
            "request": body_without_prompt,
            "input": body.input,
            "embedding": status.embedding,
        }
        logger.info(
            f"🦙 [{status.state} for {body.model}]: ({' | '.join(logs)})"
        )
        return chat_logger.info(
            dumps(embed_log, option=OPT_INDENT_2).decode()
        )
    if not isinstance(status, CompletionStatus):
        return

    # Log the completion status
    tokens = status.generated_tokens
    tokens_per_second = tokens / (elapsed_time or 0.0001)
    logs.append(f"tokens: {tokens}({tokens_per_second: .1f}tok/s)")
    if isinstance(body, CreateChatCompletionRequest):
        # Log the chat completion status
        chat_log = {
            "request": body_without_prompt,
            "chat": [
                body.messages[i].model_dump(exclude_none=True)
                for i in range(len(body.messages))
            ]
            + [{"role": "assistant", "content": status.generated_text}],
        }
    elif isinstance(body, CreateCompletionRequest):
        # Log the text completion status
        chat_log = {
            "request": body_without_prompt,
            "prompt": {
                "user": body.prompt,
                "assistant": status.generated_text,
            },
        }
    else:
        return
    logger.info(f"🦙 [{status.state} for {body.model}]: ({' | '.join(logs)})")
    chat_logger.info(dumps(chat_log, option=OPT_INDENT_2).decode())
