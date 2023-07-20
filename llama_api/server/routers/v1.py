"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""


from collections import deque
from functools import partial
from typing import AsyncGenerator, Iterator, Optional

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Depends, Request
from orjson import dumps
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

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
    ModelList,
)
from ...utils.errors import RouteErrorHandler
from ...utils.logger import ApiLogger
from ...utils.system import free_memory_of_first_item_from_container
from ..imports import (
    ExllamaCompletionGenerator,
    ExllamaModel,
    LlamaCppCompletionGenerator,
    LlamaCppModel,
    SentenceEncoderEmbeddingGenerator,
    TransformerEmbeddingGenerator,
)

logger = ApiLogger(__name__)


router = APIRouter(route_class=RouteErrorHandler)
semaphore = anyio.create_semaphore(1)
completion_generators: deque["BaseCompletionGenerator"] = deque(maxlen=1)
embedding_generators: deque["BaseEmbeddingGenerator"] = deque(maxlen=1)


def get_model(model_name: str) -> "BaseLLMModel":
    """Get a model from the model_definitions.py file"""
    try:
        return getattr(model_definitions, model_name)
    except Exception:
        raise AssertionError(f"Could not find a model: {model_name}")


async def get_semaphore() -> AsyncGenerator[anyio.Semaphore, None]:
    """Get a semaphore for the endpoint. This is to prevent multiple requests
    from creating multiple completion generators at the same time."""
    async with semaphore:
        yield semaphore


def get_completion_generator(
    body: CreateCompletionRequest
    | CreateChatCompletionRequest
    | CreateEmbeddingRequest,
) -> "BaseCompletionGenerator":
    """Get a completion generator for the given model.
    If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""

    try:
        # Check if the model is an OpenAI model
        openai_replacement_models: dict[str, str] = getattr(
            model_definitions, "openai_replacement_models", {}
        )
        if body.model in openai_replacement_models:
            body.model = openai_replacement_models[body.model]
            if not isinstance(body, CreateEmbeddingRequest):
                body.logit_bias = None

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
                embedding_generators,
                min_free_memory_mb=512,
                logger=logger,
            )

        # Before creating a new completion generator, check memory usage
        if completion_generators.maxlen == len(completion_generators):
            free_memory_of_first_item_from_container(
                completion_generators,
                min_free_memory_mb=256,
                logger=logger,
            )

        # Create a new completion generator
        if isinstance(llm_model, LlamaCppModel):
            assert not isinstance(
                LlamaCppCompletionGenerator, str
            ), LlamaCppCompletionGenerator
            to_return = LlamaCppCompletionGenerator.from_pretrained(llm_model)
        elif isinstance(llm_model, ExllamaModel):
            assert not isinstance(
                ExllamaCompletionGenerator, str
            ), ExllamaCompletionGenerator
            to_return = ExllamaCompletionGenerator.from_pretrained(llm_model)
        else:
            raise AssertionError(f"Model {body.model} not implemented")

        # Add the new completion generator to the deque cache
        completion_generators.append(to_return)
        return to_return
    except (AssertionError, OSError, MemoryError) as e:
        raise e
    except Exception as e:
        logger.exception(f"Exception in get_completion_generator: {e}")
        raise AssertionError(f"Could not find a model: {body.model}")


def get_embedding_generator(
    body: CreateEmbeddingRequest,
) -> "BaseEmbeddingGenerator":
    """Get an embedding generator for the given model.
    If the model is not cached, create a new one.
    If the cache is full, delete the oldest completion generator."""
    try:
        body.model = body.model.lower()
        for embedding_generator in embedding_generators:
            if embedding_generator.model_name == body.model:
                return embedding_generator

        # Before creating a new completion generator, check memory usage
        if embedding_generators.maxlen == len(embedding_generators):
            free_memory_of_first_item_from_container(
                embedding_generators,
                min_free_memory_mb=256,
                logger=logger,
            )
        # Before creating a new, deallocate embeddings to free up memory
        if completion_generators:
            free_memory_of_first_item_from_container(
                completion_generators,
                min_free_memory_mb=512,
                logger=logger,
            )

        if "sentence" in body.model and "encoder" in body.model:
            # Create a new sentence encoder embedding
            assert not isinstance(
                SentenceEncoderEmbeddingGenerator, str
            ), SentenceEncoderEmbeddingGenerator
            to_return = SentenceEncoderEmbeddingGenerator.from_pretrained(
                body.model
            )
        else:
            # Create a new transformer embedding
            assert not isinstance(
                TransformerEmbeddingGenerator, str
            ), LlamaCppCompletionGenerator
            to_return = TransformerEmbeddingGenerator.from_pretrained(
                body.model
            )

        # Add the new completion generator to the deque cache
        embedding_generators.append(to_return)
        return to_return
    except (AssertionError, OSError, MemoryError) as e:
        raise e
    except Exception as e:
        logger.exception(f"Exception in get_embedding_generator: {e}")
        raise AssertionError(f"Could not find a model: {body.model}")


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
    is_chat_completion: Optional[bool] = None,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                if is_chat_completion is True:
                    print(
                        chunk["choices"][0]["delta"].get("content", ""),
                        end="",
                        flush=True,
                    )
                elif is_chat_completion is False:
                    print(
                        chunk["choices"][0]["text"],
                        end="",
                        flush=True,
                    )
                await inner_send_chan.send(b"data: " + dumps(chunk) + b"\n\n")
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(b"data: [DONE]\n\n")
        except anyio.get_cancelled_exc_class() as e:
            with anyio.move_on_after(1, shield=True):
                logger.info(
                    f"🦙 Disconnected from client {request.client}",
                )
                raise e
        finally:
            logger.info("\n[🦙 I'm done talking]")


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    semaphore: anyio.Semaphore = Depends(get_semaphore),
):
    logger.info(f"🦙 Chat Completion Settings: {body}\n\n")
    completion_generator = get_completion_generator(body)
    logger.info("\n[🦙 I'm talking now]")
    if body.stream:
        _iterator: Iterator[
            ChatCompletionChunk
        ] = completion_generator.generate_chat_completion_with_streaming(
            messages=body.messages,
            settings=body,
        )
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, _iterator)

        def iterator() -> Iterator[ChatCompletionChunk]:
            yield first_response
            yield from _iterator

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                is_chat_completion=True,
            ),
        )
    else:
        chat_completion: ChatCompletion = await run_in_threadpool(
            completion_generator.generate_chat_completion,
            messages=body.messages,
            settings=body,
        )
        print(chat_completion["choices"][0]["message"]["content"])
        logger.info("\n[🦙 I'm done talking!]")
        return chat_completion


@router.post("/v1/completions")
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    semaphore: anyio.Semaphore = Depends(get_semaphore),
):
    logger.info(f"🦙 Text Completion Settings: {body}\n\n")
    completion_generator = get_completion_generator(body)
    logger.info("\n[🦙 I'm talking now]")
    if body.stream:
        _iterator: Iterator[
            CompletionChunk
        ] = completion_generator.generate_completion_with_streaming(
            prompt=body.prompt,
            settings=body,
        )
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, _iterator)

        def iterator() -> Iterator[CompletionChunk]:
            yield first_response
            yield from _iterator

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                is_chat_completion=False,
            ),
        )
    else:
        completion: Completion = await run_in_threadpool(
            completion_generator.generate_completion,
            prompt=body.prompt,
            settings=body,
        )
        print(completion["choices"][0]["text"])
        logger.info("\n[🦙 I'm done talking!]")
        return completion


@router.post("/v1/embeddings")
async def create_embedding(
    body: CreateEmbeddingRequest,
    semaphore: anyio.Semaphore = Depends(get_semaphore),
) -> Embedding:
    assert body.model is not None, "Model is required"
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
        embeddings: list[list[float]] = await run_in_threadpool(
            embedding_generator.generate_embeddings,
            texts=body.input if isinstance(body.input, list) else [body.input],
            context_length=512,
            batch=1000,
        )

        return {
            "object": "list",
            "data": [
                {
                    "index": embedding_idx,
                    "object": "embedding",
                    "embedding": embedding,
                }
                for embedding_idx, embedding in enumerate(embeddings)
            ],
            "model": body.model,
            "usage": {
                "prompt_tokens": -1,
                "total_tokens": -1,
            },
        }

    else:
        # Trying to get embedding model from Llama.cpp
        assert getattr(llm_model, "embedding", False), (
            "Model does not support embeddings. "
            "Set `embedding` to True in the LlamaCppModel"
        )
        assert not isinstance(
            LlamaCppCompletionGenerator, str
        ), LlamaCppCompletionGenerator
        completion_generator = get_completion_generator(body)
        assert isinstance(
            completion_generator, LlamaCppCompletionGenerator
        ), f"Model {body.model} is not supported for llama.cpp embeddings."

        assert completion_generator.client, "Model is not loaded yet"
        return await run_in_threadpool(
            completion_generator.client.create_embedding,
            **body.dict(exclude={"user"}),
        )


@router.get("/v1/models")
async def get_models() -> ModelList:
    model_names: list[str] = [
        k + f"({v.model_path})"
        for k, v in model_definitions.__dict__.items()
        if isinstance(v, BaseLLMModel)
    ]
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for model_name in model_names
        ],
    }
