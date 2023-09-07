"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""

from asyncio import (
    FIRST_COMPLETED,
    Task,
    create_task,
    ensure_future,
    sleep,
    wait,
)
from dataclasses import dataclass, field
from functools import partial
from queue import Queue
from random import choice
from threading import Event
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from anyio import (
    Semaphore,
    create_memory_object_stream,
    get_cancelled_exc_class,
)
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
    ModelData,
    ModelList,
)
from ...shared.config import MainCliArgs
from ...utils.concurrency import (
    get_queue_and_event,
    run_in_processpool_with_wix,
)
from ...utils.errors import RouteErrorHandler
from ...utils.logger import ApiLogger
from ...utils.model_definition_finder import ModelDefinitions
from ..pools.llama import (
    generate_completion,
    generate_completion_chunks,
    generate_embeddings,
)

LOOP_TIMEOUT = 30.0
MAX_WORKERS = int(MainCliArgs.max_workers.value or 1)
MAX_SEMAPHORES = int(MainCliArgs.max_semaphores.value or 1)

ChatCompletionContext = Tuple[
    Request, CreateChatCompletionRequest, int, Queue, Event
]
CompletionContext = Tuple[
    Request, CreateCompletionRequest, int, Queue, Event
]
EmbeddingContext = Tuple[Request, CreateEmbeddingRequest, int, Queue, Event]
T = TypeVar("T")

logger = ApiLogger(__name__)
router = APIRouter(prefix="/v1", route_class=RouteErrorHandler)


@dataclass
class WixMetadata:
    """Worker index (wix) metadata"""

    wix: int
    processed_key: Optional[str] = None
    semaphore: Semaphore = field(
        default_factory=lambda: Semaphore(MAX_SEMAPHORES)
    )


class WixHandler:
    """An utility class to handle worker index (wix) metadata.
    Wix is used to keep track of which worker is currently
    processing a request. This is used to prevent multiple requests from
    creating multiple completion generators at the same time."""

    wix_metas: Tuple[WixMetadata, ...] = tuple(
        WixMetadata(wix) for wix in range(MAX_WORKERS)
    )

    @classmethod
    def get_wix_meta(cls, request_key: Optional[str] = None) -> WixMetadata:
        """Get the worker index (wix) metadata for the key"""
        worker_ranks = [
            cls._get_worker_rank(wix_meta, request_key)
            for wix_meta in cls.wix_metas
        ]
        min_rank = min(worker_ranks)

        # Choose a random worker index (wix) with the lowest rank
        candidates = [
            i for i, rank in enumerate(worker_ranks) if rank == min_rank
        ]
        if not candidates:
            raise LookupError("No available wix")
        return cls.wix_metas[choice(candidates)]

    @staticmethod
    def _get_worker_rank(
        meta: WixMetadata, request_key: Optional[str]
    ) -> int:
        """Get the entry rank for the worker index (wix) metadata.
        Lower rank means higher priority of the worker to process the request.
        If the rank is -2, then the worker is processing the same model
        If the rank is -1, then the worker is not processing any model
        If the rank is greater than or equal to 0,
        then the worker is processing a different model"""
        if request_key == meta.processed_key:
            # If the key is the same (worker is processing the same model)
            return -2  # return the highest priority
        if request_key is None or meta.processed_key is None:
            # If not requesting a specific model or worker is not processing
            return -1  # return the second highest priority
        return (
            MAX_SEMAPHORES - meta.semaphore._value
        )  # return the number of slots in use


def validate_item_type(item: Any, type: Type[T]) -> T:
    """Validate that the item is of the correct type"""
    if isinstance(item, Exception):
        # The producer task has raised an exception
        raise item
    elif not isinstance(item, type):
        # The producer task has returned an invalid response
        raise TypeError(
            f"Expected type {type}, but got {type(item)} instead"
        )
    return item


async def get_first_response(
    request: Request, queue: Queue, task: Task
) -> Dict:
    async def check_client_connection():
        while True:
            await sleep(1.0)
            if await request.is_disconnected() or task.cancelled():
                raise get_cancelled_exc_class()()

    done, pending = await wait(
        {
            ensure_future(run_in_threadpool(queue.get)),
            ensure_future(check_client_connection()),
        },
        return_when=FIRST_COMPLETED,
    )
    if pending:
        for t in pending:
            t.cancel()
    if not done:
        raise get_cancelled_exc_class()()

    return validate_item_type(done.pop().result(), type=dict)


async def get_chat_or_text_completion(
    ctx: Union[ChatCompletionContext, CompletionContext]
) -> Union[ChatCompletion, Completion]:
    """Create a chat completion or completion based on the body."""
    task = None
    request, body, wix, queue, event = ctx
    try:
        func = partial(generate_completion, body, queue, event)
        task = create_task(run_in_processpool_with_wix(func, wix=wix))
        completion = await get_first_response(request, queue, task)
        return completion  # type: ignore
    finally:
        event.set()
        if task is not None:
            task.cancel()


async def get_chat_or_text_completion_streaming(
    ctx: Union[ChatCompletionContext, CompletionContext],
) -> EventSourceResponse:
    """Create a chat completion or completion based on the body, and stream"""
    task = None
    request, body, wix, queue, event = ctx
    try:
        func = partial(generate_completion_chunks, body, queue, event)
        task = create_task(run_in_processpool_with_wix(func, wix=wix))
        first_chunk = await get_first_response(request, queue, task)
        send_chan, recv_chan = create_memory_object_stream(10)

        async def get_event_publisher() -> None:
            try:

                def iterator():
                    yield first_chunk
                    while True:
                        gen = queue.get(timeout=LOOP_TIMEOUT)
                        if gen is None:
                            break  # The producer task is done
                        yield validate_item_type(gen, type=dict)

                async for chunk in iterate_in_threadpool(iterator()):
                    await send_chan.send(b"data: " + dumps(chunk) + b"\n\n")
                await send_chan.send(b"data: [DONE]\n\n")
            finally:
                event.set()
                task.cancel()
                send_chan.close()

        return EventSourceResponse(
            recv_chan, data_sender_callable=get_event_publisher
        )

    except Exception:
        event.set()
        if task is not None:
            task.cancel()
        raise


async def get_embedding(
    ctx: EmbeddingContext,
) -> Embedding:
    """Create a chat completion or completion based on the body."""
    task = None
    request, body, wix, queue, _ = ctx
    try:
        func = partial(generate_embeddings, body, queue)
        task = create_task(run_in_processpool_with_wix(func, wix=wix))
        embeddings = await get_first_response(request, queue, task)
        return embeddings  # type: ignore
    finally:
        if task is not None:
            task.cancel()


async def get_chat_completion_context(
    request: Request, body: CreateChatCompletionRequest
) -> AsyncIterator[ChatCompletionContext]:
    interrupt_signal = None
    wix_meta = WixHandler.get_wix_meta(body.model)

    # Acquire the semaphore for the worker index (wix)
    await wix_meta.semaphore.acquire()
    try:
        if await request.is_disconnected():
            # If client is already gone, then ignore the request
            raise get_cancelled_exc_class()()
        # Reserve the worker, it is now processing the request
        wix_meta.processed_key = body.model
        queue, interrupt_signal = get_queue_and_event()
        yield request, body, wix_meta.wix, queue, interrupt_signal
    finally:
        wix_meta.semaphore.release()
        if interrupt_signal is not None:
            interrupt_signal.set()


async def get_completion_context(
    request: Request, body: CreateCompletionRequest
) -> AsyncIterator[CompletionContext]:
    interrupt_signal = None
    wix_meta = WixHandler.get_wix_meta(body.model)

    # Acquire the semaphore for the worker index (wix)
    await wix_meta.semaphore.acquire()
    try:
        if await request.is_disconnected():
            # If client is already gone, then ignore the request
            raise get_cancelled_exc_class()()
        # Reserve the worker, it is now processing the request
        wix_meta.processed_key = body.model
        queue, interrupt_signal = get_queue_and_event()
        yield request, body, wix_meta.wix, queue, interrupt_signal
    finally:
        wix_meta.semaphore.release()
        if interrupt_signal is not None:
            interrupt_signal.set()


async def get_embedding_context(
    request: Request, body: CreateEmbeddingRequest
) -> AsyncIterator[EmbeddingContext]:
    if MainCliArgs.no_embed.value:
        raise PermissionError("Embeddings endpoint is disabled")
    assert body.model is not None, "Model is required"
    interrupt_signal = None
    wix_meta = WixHandler.get_wix_meta(body.model)

    # Acquire the semaphore for the worker index (wix)
    await wix_meta.semaphore.acquire()
    try:
        if await request.is_disconnected():
            # If client is already gone, then ignore the request
            raise get_cancelled_exc_class()()
        # Reserve the worker, it is now processing the request
        wix_meta.processed_key = body.model
        queue, interrupt_signal = get_queue_and_event()
        yield request, body, wix_meta.wix, queue, interrupt_signal
    finally:
        wix_meta.semaphore.release()
        if interrupt_signal is not None:
            interrupt_signal.set()


@router.post("/chat/completions")
async def create_chat_completion(
    ctx: ChatCompletionContext = Depends(get_chat_completion_context),
):
    if ctx[1].stream:
        return await get_chat_or_text_completion_streaming(ctx)
    return await get_chat_or_text_completion(ctx)


@router.post("/v1/engines/copilot-codex/completions")
@router.post("/completions")
async def create_completion(
    ctx: CompletionContext = Depends(get_completion_context),
):
    if ctx[1].stream:
        return await get_chat_or_text_completion_streaming(ctx)
    return await get_chat_or_text_completion(ctx)


@router.post("/embeddings")
async def create_embedding(
    ctx: EmbeddingContext = Depends(get_embedding_context),
) -> Embedding:
    return await get_embedding(ctx)


@router.get("/models")
async def get_models() -> ModelList:
    return ModelList(
        object="list",
        data=[
            ModelData(
                id=model_name,
                object="model",
                owned_by="me",
                permissions=[],
            )
            for model_name in ModelDefinitions.get_model_names()
        ],
    )
