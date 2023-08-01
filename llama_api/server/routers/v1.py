"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""


from asyncio import Task, create_task
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import partial
from os import environ
from queue import Queue
from threading import Event
from time import time
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Iterator,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import TypedDict

from anyio import (
    Semaphore,
    create_memory_object_stream,
    get_cancelled_exc_class,
    move_on_after,
)
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Request
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
from ...utils.concurrency import (
    get_queue_and_event,
    run_in_processpool_with_wix,
)
from ...utils.errors import RouteErrorHandler
from ...utils.logger import ApiLogger
from ..pools.llama import (
    generate_completion,
    generate_completion_chunks,
    generate_embeddings,
    get_model_names,
)

MAX_WORKERS = int(environ.get("MAX_WORKERS", 1))
logger = ApiLogger(__name__)
router = APIRouter(route_class=RouteErrorHandler)
T = TypeVar("T")


class TaskStatus(TypedDict):
    """Completion status"""

    completion_tokens: int
    started_at: float
    interrupted: bool


@dataclass
class WixMetadata:
    key: Optional[str] = None
    semaphore: Semaphore = field(default_factory=lambda: Semaphore(1))


# Worker index (wix) is used to keep track of which worker is currently
# processing a request. This is used to prevent multiple requests from
# creating multiple completion generators at the same time.
wixs: Tuple[WixMetadata] = tuple(WixMetadata() for _ in range(MAX_WORKERS))


def validate_item_type(item: Any, type: Type[T]) -> T:
    """Validate that the item is of the correct type"""
    if isinstance(item, Exception):
        # The producer task has raised an exception
        raise item
    elif not isinstance(item, type):
        # The producer task has returned an invalid response
        raise TypeError(f"Expected type {type}, but got {type(item)} instead")
    return item


@asynccontextmanager
async def get_wix_with_semaphore(
    key: Optional[str] = None,
) -> AsyncGenerator[int, None]:
    if key is None:
        # Find the first available slot
        for wix, wix_metadata in enumerate(wixs):
            if wix_metadata.semaphore.value:
                async with wix_metadata.semaphore:
                    wix_metadata.key = key
                    yield wix
                    return
    else:
        # Get the worker index (wix) for the key
        for wix, wix_metadata in enumerate(wixs):
            if wix_metadata.key == key:
                async with wix_metadata.semaphore:
                    yield wix
                    return

        # If the key is not in the wixs, find the first empty slot
        for wix, wix_metadata in enumerate(wixs):
            if wix_metadata.key is None:
                async with wix_metadata.semaphore:
                    wix_metadata.key = key
                    yield wix
                    return

        # If there are no empty slot, find available slot
        for wix, wix_metadata in enumerate(wixs):
            if wix_metadata.semaphore.value:
                async with wix_metadata.semaphore:
                    wix_metadata.key = key
                    yield wix
                    return

    # If there are no available slot, wait for one to become available
    for wix, wix_metadata in enumerate(wixs):
        async with wix_metadata.semaphore:
            wix_metadata.key = key
            yield wix
            return

    raise LookupError("No available wix")


async def get_event_publisher(
    request: Request,
    body: Union[
        CreateChatCompletionRequest,
        CreateCompletionRequest,
    ],
    inner_send_chan: MemoryObjectSendStream,
    task: "Task[None]",
    event: Event,
    iterator: Iterator,
):
    """Publish Server-Sent-Events (SSE) to the client"""
    with task_manager(body=body, task=task, event=event) as task_status:
        async with inner_send_chan:
            try:
                async for chunk in iterate_in_threadpool(iterator):
                    task_status["completion_tokens"] += 1
                    await inner_send_chan.send(
                        b"data: " + dumps(chunk) + b"\n\n"
                    )
                    if await request.is_disconnected():
                        raise get_cancelled_exc_class()()
                await inner_send_chan.send(b"data: [DONE]\n\n")
            except get_cancelled_exc_class() as e:
                with move_on_after(1, shield=True):
                    task_status["interrupted"] = True
                    raise e


def get_streaming_iterator(
    queue: Queue,
    first_response: Optional[dict] = None,
) -> Iterator[dict]:
    """Get an iterator for the streaming of completion generator"""
    if first_response is not None:
        yield first_response

    while True:
        gen = queue.get()
        if gen is None:
            # The producer task is done
            break
        yield validate_item_type(gen, type=dict)


@contextmanager
def task_manager(
    body: Union[
        CreateChatCompletionRequest,
        CreateCompletionRequest,
        CreateEmbeddingRequest,
    ],
    task: "Task[None]",
    event: Event,
) -> Generator[TaskStatus, None, None]:
    """Start the producer task and cancel it when the client disconnects.
    Also, log the completion status."""
    task_status = TaskStatus(
        completion_tokens=0, started_at=time(), interrupted=False
    )
    try:
        logger.info(f"🦙 Handling request of {body.model}")
        yield task_status
    finally:
        # Cancel the producer task and set event,
        # so the completion task can be stopped
        task.cancel()
        event.set()

        # Log the completion status
        elapsed_time = time() - task_status["started_at"]
        tokens = task_status["completion_tokens"]
        tokens_per_second = tokens / elapsed_time
        basic_message = (
            f"elapsed time: {elapsed_time: .1f}s | "
            f"tokens: {tokens}({tokens_per_second: .1f}tok/s)"
        )
        if task_status["interrupted"]:
            logger.info(f"🦙 [Interrupted!]: {basic_message}")
        else:
            logger.info(f"🦙 [Completed!]: {basic_message}")


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
):
    async with get_wix_with_semaphore(body.model) as wix:
        queue, event = get_queue_and_event()
        producer: Callable[
            [
                Union[CreateChatCompletionRequest, CreateCompletionRequest],
                Queue,
                Event,
            ],
            None,
        ] = partial(
            generate_completion_chunks if body.stream else generate_completion,
            body=body,
            queue=queue,
            event=event,
        )
        producer_task: "Task[None]" = create_task(
            run_in_processpool_with_wix(producer, wix=wix)
        )

        if body.stream:
            send_chan, recv_chan = create_memory_object_stream(10)
            return EventSourceResponse(
                recv_chan,
                data_sender_callable=partial(
                    get_event_publisher,
                    request=request,
                    body=body,
                    inner_send_chan=send_chan,
                    task=producer_task,
                    event=event,
                    iterator=get_streaming_iterator(
                        queue=queue,
                        first_response=validate_item_type(
                            await run_in_threadpool(queue.get), type=dict
                        ),
                    ),
                ),
            )
        else:
            with task_manager(body, producer_task, event) as task_status:
                chat_completion: ChatCompletion = validate_item_type(
                    await run_in_threadpool(queue.get),
                    type=dict,  # type: ignore
                )
                task_status["completion_tokens"] = chat_completion["usage"][
                    "completion_tokens"
                ]
                return chat_completion


@router.post("/v1/completions")
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
):
    async with get_wix_with_semaphore(body.model) as wix:
        queue, event = get_queue_and_event()
        producer: Callable[
            [
                Union[CreateChatCompletionRequest, CreateCompletionRequest],
                Queue,
                Event,
            ],
            None,
        ] = partial(
            generate_completion_chunks if body.stream else generate_completion,
            body=body,
            queue=queue,
            event=event,
        )
        producer_task: "Task[None]" = create_task(
            run_in_processpool_with_wix(producer, wix=wix)
        )
        if body.stream:
            send_chan, recv_chan = create_memory_object_stream(10)
            return EventSourceResponse(
                recv_chan,
                data_sender_callable=partial(
                    get_event_publisher,
                    request=request,
                    body=body,
                    inner_send_chan=send_chan,
                    task=producer_task,
                    event=event,
                    iterator=get_streaming_iterator(
                        queue=queue,
                        first_response=validate_item_type(
                            await run_in_threadpool(queue.get), type=dict
                        ),
                    ),
                ),
            )
        else:
            with task_manager(body, producer_task, event) as task_status:
                completion: Completion = validate_item_type(
                    await run_in_threadpool(queue.get),
                    type=dict,  # type: ignore
                )
                task_status["completion_tokens"] = completion["usage"][
                    "completion_tokens"
                ]
                return completion


@router.post("/v1/embeddings")
async def create_embedding(
    body: CreateEmbeddingRequest,
) -> Embedding:
    assert body.model is not None, "Model is required"
    async with get_wix_with_semaphore(body.model) as wix:
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
        producer_task: "Task[None]" = create_task(
            run_in_processpool_with_wix(producer, wix=wix)
        )
        with task_manager(body, producer_task, event):
            return validate_item_type(
                await run_in_threadpool(queue.get),
                type=dict,  # type: ignore
            )


@router.get("/v1/models")
async def get_models() -> ModelList:
    async with get_wix_with_semaphore() as wix:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "me",
                    "permissions": [],
                }
                for model_name in await run_in_processpool_with_wix(
                    get_model_names,
                    wix=wix,
                )
            ],
        }
