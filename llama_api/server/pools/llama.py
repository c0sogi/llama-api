from collections import deque
from contextlib import contextmanager
from multiprocessing.dummy import current_process
from os import getpid
from queue import Queue
from threading import Event
from typing import Deque, Dict, Iterator, List, Union

import model_definitions

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
from ...utils.logger import ApiLogger
from ...utils.system import free_memory_of_first_item_from_container


logger = ApiLogger(__name__)
logger.info(f"🔧 {current_process()} is initiated with PID: {getpid()}")

lazy = LazyImports()  # lazy-loader of modules
completion_generators: Deque["BaseCompletionGenerator"] = deque(maxlen=1)
embedding_generators: Deque["BaseEmbeddingGenerator"] = deque(maxlen=1)


def init() -> None:
    pass


@contextmanager
def completion_generator_manager(
    body: Union[
        CreateCompletionRequest,
        CreateChatCompletionRequest,
        CreateEmbeddingRequest,
    ],
    interrupt_signal: Event,
):
    """Context manager for completion generators."""
    completion_generator = get_completion_generator(body)
    completion_generator.interrupt_signal = interrupt_signal
    yield completion_generator
    completion_generator.interrupt_signal = None


def get_model_names() -> List[str]:
    return [
        k + f"({v.model_path})"
        for k, v in model_definitions.__dict__.items()
        if isinstance(v, BaseLLMModel)
    ]


def get_model(model_name: str) -> "BaseLLMModel":
    """Get a model from the model_definitions.py file"""
    with logger.log_any_error(
        f"Error getting model: {model_name}", exc_info=None
    ):
        return getattr(model_definitions, model_name)


def get_completion_generator(
    body: Union[
        CreateCompletionRequest,
        CreateChatCompletionRequest,
        CreateEmbeddingRequest,
    ],
) -> "BaseCompletionGenerator":
    """Get a completion generator for the given model.
    If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""

    with logger.log_any_error(
        f"Error getting a completion generator of {body.model}"
    ):
        # Check if the model is an OpenAI model
        openai_replacement_models: Dict[str, str] = getattr(
            model_definitions, "openai_replacement_models", {}
        )
        if body.model in openai_replacement_models:
            body.model = openai_replacement_models[body.model]
            body.is_openai = True

        # Check if the model is defined in LLMModels enum
        llm_model = get_model(body.model)

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
            raise AssertionError(f"Model {body.model} not implemented")

        # Add the new completion generator to the deque cache
        completion_generators.append(to_return)
        return to_return


def get_embedding_generator(
    body: CreateEmbeddingRequest,
) -> "BaseEmbeddingGenerator":
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
            to_return = lazy.SentenceEncoderEmbeddingGenerator.from_pretrained(
                body.model
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
    queue: Queue,
    interrupt_signal: Event,
) -> None:
    with queue_manager(queue=queue):
        with completion_generator_manager(
            body=body, interrupt_signal=interrupt_signal
        ) as cg:
            if isinstance(body, CreateChatCompletionRequest):
                _iterator: Iterator[
                    Union[ChatCompletionChunk, CompletionChunk]
                ] = cg.generate_chat_completion_with_streaming(
                    messages=body.messages,
                    settings=body,
                )
            elif isinstance(body, CreateCompletionRequest):
                _iterator = cg.generate_completion_with_streaming(
                    prompt=body.prompt,
                    settings=body,
                )
            first_response: Union[ChatCompletionChunk, CompletionChunk] = next(
                _iterator
            )

            def iterator() -> (
                Iterator[Union[ChatCompletionChunk, CompletionChunk]]
            ):
                yield first_response
                for chunk in _iterator:
                    yield chunk

            for chunk in iterator():
                if interrupt_signal.is_set():
                    # If the event is set, it means the client has disconnected
                    return
                queue.put(chunk)


def generate_completion(
    body: Union[CreateChatCompletionRequest, CreateCompletionRequest],
    queue: Queue,
    interrupt_signal: Event,
) -> None:
    with queue_manager(queue=queue):
        with completion_generator_manager(
            body=body, interrupt_signal=interrupt_signal
        ) as cg:
            if isinstance(body, CreateChatCompletionRequest):
                completion: Union[
                    ChatCompletion, Completion
                ] = cg.generate_chat_completion(
                    messages=body.messages,
                    settings=body,
                )
            elif isinstance(body, CreateCompletionRequest):
                completion = cg.generate_completion(
                    prompt=body.prompt,
                    settings=body,
                )
            queue.put(completion)


def generate_embeddings(
    body: CreateEmbeddingRequest, queue: Queue, interrupt_signal: Event
) -> None:
    with queue_manager(queue=queue):
        try:
            llm_model = get_model(body.model)
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
            queue.put(
                Embedding(
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
            completion_generator = get_completion_generator(body)
            assert isinstance(
                completion_generator, lazy.LlamaCppCompletionGenerator
            ), f"Model {body.model} is not supported for llama.cpp embeddings."
            assert completion_generator.client, "Model not loaded yet."
            queue.put(
                completion_generator.client.create_embedding,
                **body.model_dump(exclude={"user"}),
            )
