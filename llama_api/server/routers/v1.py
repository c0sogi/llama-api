"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""


from asyncio import Task, create_task
from functools import partial
from os import environ
from queue import Queue
from threading import Event
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Iterator,
    Optional,
    Type,
    TypeVar,
)

from anyio import (
    Semaphore,
    create_memory_object_stream,
    create_semaphore,
    get_cancelled_exc_class,
    move_on_after,
)
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from orjson import dumps
from sse_starlette.sse import EventSourceResponse

from ...schemas.api import (
    ChatCompletion,
    Completion,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    Embedding,
    ModelList,
)
from ...utils.concurrency import get_queue_and_event, run_in_processpool
from ...utils.errors import RouteErrorHandler
from ...utils.logger import ApiLogger
from ..pools.llama import (
    generate_completion,
    generate_completion_chunks,
    generate_embeddings,
    get_model_names,
)

MAX_CONCURRENT_REQUESTS: int = int(environ.get("MAX_CONCURRENT_REQUESTS", 1))

logger = ApiLogger(__name__)
router = APIRouter(route_class=RouteErrorHandler)
semaphore = create_semaphore(MAX_CONCURRENT_REQUESTS)
T = TypeVar("T")


def validate_item_type(item: Any, type: Type[T]) -> T:
    """Validate that the item is of the correct type"""
    if isinstance(item, Exception):
        # The producer task has raised an exception
        raise item
    elif not isinstance(item, type):
        # The producer task has returned an invalid response
        raise TypeError(f"Expected type {type}, but got {type(item)} instead")
    return item


async def get_semaphore() -> AsyncGenerator[Semaphore, None]:
    """Get a semaphore for the endpoint. This is to prevent multiple requests
    from creating multiple completion generators at the same time."""
    async with semaphore:
        yield semaphore


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Iterator,
    is_chat_completion: Optional[bool] = None,
):
    """Publish Server-Sent-Events (SSE) to the client"""
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
                    raise get_cancelled_exc_class()()
            await inner_send_chan.send(b"data: [DONE]\n\n")
        except get_cancelled_exc_class() as e:
            with move_on_after(1, shield=True):
                logger.info(
                    f"🦙 Disconnected from client {request.client}",
                )
                raise e
        finally:
            logger.info("\n[🦙 I'm done talking]")


def get_streaming_iterator(
    queue: Queue,
    event: Event,
    first_response: Optional[dict] = None,
    producer_task: Optional[Task] = None,
) -> Iterator[dict]:
    """Get an iterator for the streaming of completion generator"""
    if first_response is not None:
        yield first_response
    try:
        while True:
            gen = queue.get()
            if gen is None:
                # The producer task is done
                break
            yield validate_item_type(gen, type=dict)
    except Exception as e:
        raise e
    finally:
        event.set()
        if producer_task is not None:
            producer_task.cancel()


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
    semaphore: Semaphore = Depends(get_semaphore),
):
    queue, event = get_queue_and_event()
    producer: Callable[
        [CreateChatCompletionRequest | CreateCompletionRequest, Queue, Event],
        None,
    ] = partial(
        generate_completion_chunks if body.stream else generate_completion,
        body=body,
        queue=queue,
        event=event,
    )
    producer_task: Task[None] = create_task(run_in_processpool(producer))
    logger.info(f"🦙 Chat Completion Settings: {body}\n\n")
    logger.info("\n[🦙 I'm talking now]")
    if body.stream:
        # EAFP: It's easier to ask for forgiveness than permission
        first_response: dict = validate_item_type(
            await run_in_threadpool(queue.get), type=dict
        )

        send_chan, recv_chan = create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=get_streaming_iterator(
                    queue=queue,
                    event=event,
                    first_response=first_response,
                    producer_task=producer_task,
                ),
                is_chat_completion=True,
            ),
        )
    else:
        try:
            chat_completion: ChatCompletion = validate_item_type(
                await run_in_threadpool(queue.get), type=dict  # type: ignore
            )
            print(chat_completion["choices"][0]["message"]["content"])
            logger.info("\n[🦙 I'm done talking!]")
            return chat_completion
        finally:
            event.set()
            producer_task.cancel()


@router.post("/v1/completions")
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
    semaphore: Semaphore = Depends(get_semaphore),
):
    queue, event = get_queue_and_event()
    producer: Callable[
        [CreateChatCompletionRequest | CreateCompletionRequest, Queue, Event],
        None,
    ] = partial(
        generate_completion_chunks if body.stream else generate_completion,
        body=body,
        queue=queue,
        event=event,
    )
    producer_task: Task[None] = create_task(run_in_processpool(producer))
    logger.info(f"🦙 Text Completion Settings: {body}\n\n")
    logger.info("\n[🦙 I'm talking now]")
    if body.stream:
        # EAFP: It's easier to ask for forgiveness than permission
        first_response: dict = validate_item_type(
            await run_in_threadpool(queue.get), type=dict
        )

        send_chan, recv_chan = create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=get_streaming_iterator(
                    queue=queue,
                    event=event,
                    first_response=first_response,
                    producer_task=producer_task,
                ),
                is_chat_completion=False,
            ),
        )
    else:
        try:
            completion: Completion = validate_item_type(
                await run_in_threadpool(queue.get), type=dict  # type: ignore
            )
            print(completion["choices"][0]["text"])
            logger.info("\n[🦙 I'm done talking!]")
            return completion
        finally:
            event.set()
            producer_task.cancel()


@router.post("/v1/embeddings")
async def create_embedding(
    body: CreateEmbeddingRequest,
    semaphore: Semaphore = Depends(get_semaphore),
) -> Embedding:
    assert body.model is not None, "Model is required"
    queue, event = get_queue_and_event()
    producer: Callable[
        [CreateEmbeddingRequest, Queue, Event],
        None,
    ] = partial(
        generate_embeddings,
        body=body,
        queue=queue,
        event=event,
    )
    producer_task: Task[None] = create_task(run_in_processpool(producer))
    try:
        return validate_item_type(
            await run_in_processpool(producer), type=list  # type: ignore
        )
    finally:
        event.set()
        producer_task.cancel()


@router.get("/v1/models")
async def get_models() -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
            for model_name in await run_in_processpool(get_model_names)
        ],
    }
