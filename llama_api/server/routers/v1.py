"""V1 Endpoints for Local Llama API
Use same format as OpenAI API"""


from asyncio import Task, create_task, wait_for
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from queue import Queue
from random import choice
from threading import Event
from time import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Literal,
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
    move_on_after,
)
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter, Request
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from orjson import OPT_INDENT_2, dumps
from sse_starlette.sse import EventSourceResponse

from llama_api.shared.config import MainCliArgs

from ...mixins.completion import CompletionStatus
from ...schemas.api import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    Embedding,
    ModelData,
    ModelList,
)
from ...utils.concurrency import (
    get_queue_and_event,
    run_in_processpool_with_wix,
)
from ...utils.errors import RouteErrorHandler
from ...utils.logger import ApiLogger, LoggingConfig
from ..pools.llama import (
    EmbeddingStatus,
    generate_completion,
    generate_completion_chunks,
    generate_embeddings,
    get_model_names,
)

chat_logger = ApiLogger(
    "",
    logging_config=LoggingConfig(
        console_log_level=100, file_log_name="./logs/chat.log", color=False
    ),
)
logger = ApiLogger(__name__)
router = APIRouter(prefix="/v1", route_class=RouteErrorHandler)
max_workers = int(MainCliArgs.max_workers.value or 1)
max_semaphores = int(MainCliArgs.max_semaphores.value or 1)
T = TypeVar("T")


@dataclass
class WixMetadata:
    """Worker index (wix) metadata"""

    wix: int
    processed_key: Optional[str] = None
    semaphore: Semaphore = field(
        default_factory=lambda: Semaphore(max_semaphores)
    )


# Worker index (wix) is used to keep track of which worker is currently
# processing a request. This is used to prevent multiple requests from
# creating multiple completion generators at the same time.
wix_metas: Tuple[WixMetadata] = tuple(
    WixMetadata(wix) for wix in range(max_workers)
)


def get_worker_rank(meta: WixMetadata, request_key: Optional[str]) -> int:
    """Get the entry rank for the worker index (wix) metadata.
    Lower rank means higher priority of the worker to process the request."""
    global max_semaphores
    if request_key == meta.processed_key:
        # If the key is the same (worker is processing the same model)
        return -2  # return the highest priority
    if request_key is None or meta.processed_key is None:
        # If not requesting a specific model or the worker is not processing
        return -1  # return the second highest priority
    return (
        max_semaphores - meta.semaphore.value
    )  # return the number of slots in use


@asynccontextmanager
async def get_wix_with_semaphore(
    request: Request,
    request_key: Optional[str] = None,
) -> AsyncGenerator[int, None]:
    """Get the worker index (wix) for the key and acquire the semaphore"""
    global wix_metas
    worker_ranks = [
        get_worker_rank(wix_meta, request_key) for wix_meta in wix_metas
    ]
    min_rank = min(worker_ranks)
    candidates = [i for i, rank in enumerate(worker_ranks) if rank == min_rank]
    if not candidates:
        raise LookupError("No available wix")
    wix_meta = wix_metas[choice(candidates)]
    async with wix_meta.semaphore:
        if await request.is_disconnected():
            return
        wix_meta.processed_key = request_key
        yield wix_meta.wix


def validate_item_type(item: Any, type: Type[T]) -> T:
    """Validate that the item is of the correct type"""
    if isinstance(item, Exception):
        # The producer task has raised an exception
        raise item
    elif not isinstance(item, type):
        # The producer task has returned an invalid response
        raise TypeError(f"Expected type {type}, but got {type(item)} instead")
    return item


def get_text_from_completion(
    completion: Union[Completion, ChatCompletion]
) -> str:
    """Get the generated text from a completion"""
    if "text" in completion["choices"][0]:
        return completion["choices"][0]["text"]
    return completion["choices"][0]["message"]["content"]


def get_text_from_chunk(
    chunk: Union[CompletionChunk, ChatCompletionChunk]
) -> str:
    """Get the generated text from a completion chunk"""
    if "text" in chunk["choices"][0]:
        return chunk["choices"][0]["text"]
    return chunk["choices"][0]["delta"].get("content", "")


async def get_event_publisher(
    request: Request,
    body: Union[
        CreateChatCompletionRequest,
        CreateCompletionRequest,
    ],
    inner_send_chan: MemoryObjectSendStream[bytes],
    task: "Task[CompletionStatus]",
    interrupt_signal: Event,
    iterator: Iterator[Union[ChatCompletionChunk, CompletionChunk]],
) -> None:
    """Publish Server-Sent-Events (SSE) to the client"""
    is_interrupted = False  # type: bool
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(b"data: " + dumps(chunk) + b"\n\n")
                if await request.is_disconnected():
                    raise get_cancelled_exc_class()()
            await inner_send_chan.send(b"data: [DONE]\n\n")
        except get_cancelled_exc_class():
            is_interrupted = True
            with move_on_after(1, shield=True):
                raise
        finally:
            # Cancel the producer task and set event,
            # so the completion task can be stopped
            interrupt_signal.set()
            state = "Interrupted" if is_interrupted else "Completed"
            try:
                status = await wait_for(task, timeout=3)
                log_request_and_response(body, status, state)
            finally:
                task.cancel()


def get_streaming_iterator(
    queue: Queue,
    first_response: Optional[Dict] = None,
) -> Iterator[Dict]:
    """Get an iterator for the streaming of completion generator"""
    if first_response is not None:
        yield first_response

    while True:
        gen = queue.get()
        if gen is None:
            # The producer task is done
            break
        yield validate_item_type(gen, type=dict)


def log_request_and_response(
    body: Union[
        CreateChatCompletionRequest,
        CreateCompletionRequest,
        CreateEmbeddingRequest,
    ],
    status: Optional[Union[CompletionStatus, EmbeddingStatus]],
    state: Literal["Completed", "Interrupted"],
) -> None:
    """Log the request and response of the completion or embedding"""
    if status is None:
        return
    elapsed_time = time() - status.started_at
    log_messages: List[str] = [f"elapsed time: {elapsed_time: .1f}s"]
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
        embed_usage = {
            "input_chars": len(body.input),
            "embedding_chunks": len(status.embedding["data"])
            if status.embedding
            else 0,
        }
        log_messages.append(f"embedding chunks: {embed_usage}")
        embed_log = {
            "request": body_without_prompt,
            "input": body.input,
            "embedding": status.embedding,
        }
        logger.info(
            f"🦙 [{state} for {body.model}]: ({' | '.join(log_messages)})"
        )
        return chat_logger.info(dumps(embed_log, option=OPT_INDENT_2).decode())
    if not isinstance(status, CompletionStatus):
        return

    # Log the completion status
    tokens = status.generated_tokens
    tokens_per_second = tokens / elapsed_time
    log_messages.append(f"tokens: {tokens}({tokens_per_second: .1f}tok/s)")
    if isinstance(body, CreateChatCompletionRequest):
        chat_log = {
            "request": body_without_prompt,
            "chat": [
                body.messages[i].model_dump(exclude_none=True)
                for i in range(len(body.messages))
            ]
            + [
                {
                    "role": "assistant",
                    "content": status.generated_text,
                }
            ],
        }
    elif isinstance(body, CreateCompletionRequest):
        chat_log = {
            "request": body_without_prompt,
            "prompt": {
                "user": body.prompt,
                "assistant": status.generated_text,
            },
        }
    else:
        return
    logger.info(f"🦙 [{state} for {body.model}]: ({' | '.join(log_messages)})")
    chat_logger.info(dumps(chat_log, option=OPT_INDENT_2).decode())


async def create_chat_completion_or_completion(
    request: Request,
    body: Union[CreateChatCompletionRequest, CreateCompletionRequest],
) -> Union[EventSourceResponse, ChatCompletion, Completion]:
    """Create a chat completion or completion based on the body.
    If the body is a chat completion, then create a chat completion.
    If the body is a completion, then create a completion.
    If streaming is enabled, then return an EventSourceResponse."""
    async with get_wix_with_semaphore(request, body.model) as wix:
        queue, interrupt_signal = get_queue_and_event()
        task: "Task[CompletionStatus]" = create_task(
            run_in_processpool_with_wix(
                partial(
                    generate_completion_chunks
                    if body.stream
                    else generate_completion,
                    body=body,
                    queue=queue,
                    interrupt_signal=interrupt_signal,
                ),
                wix=wix,
            )
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
                    task=task,
                    interrupt_signal=interrupt_signal,
                    iterator=get_streaming_iterator(  # type: ignore
                        queue=queue,
                        first_response=validate_item_type(
                            await run_in_threadpool(queue.get), type=dict
                        ),
                    ),
                ),
            )
        else:
            # Cancel the producer task and set event,
            # so the completion task can be stopped
            try:
                return validate_item_type(
                    await run_in_threadpool(queue.get),
                    type=dict,  # type: ignore
                )
            finally:
                interrupt_signal.set()
                log_request_and_response(body, await task, "Completed")


@router.post("/chat/completions")
async def create_chat_completion(
    request: Request, body: CreateChatCompletionRequest
):
    return await create_chat_completion_or_completion(
        request=request, body=body
    )


@router.post("/completions")
async def create_completion(request: Request, body: CreateCompletionRequest):
    return await create_chat_completion_or_completion(
        request=request, body=body
    )


@router.post("/embeddings")
async def create_embedding(
    request: Request, body: CreateEmbeddingRequest
) -> Embedding:
    if MainCliArgs.no_embed.value:
        raise PermissionError("Embeddings endpoint is disabled")
    assert body.model is not None, "Model is required"
    async with get_wix_with_semaphore(request, body.model) as wix:
        queue, interrupt_signal = get_queue_and_event()
        task: Task["EmbeddingStatus"] = create_task(
            run_in_processpool_with_wix(
                partial(
                    generate_embeddings,
                    body=body,
                    queue=queue,
                ),
                wix=wix,
            )
        )
        try:
            return validate_item_type(
                await run_in_threadpool(queue.get),
                type=dict,  # type: ignore
            )
        finally:
            interrupt_signal.set()
            log_request_and_response(body, await task, "Completed")


@router.get("/models")
async def get_models(request: Request) -> ModelList:
    async with get_wix_with_semaphore(request) as wix:
        return ModelList(
            object="list",
            data=[
                ModelData(
                    id=model_name,
                    object="model",
                    owned_by="me",
                    permissions=[],
                )
                for model_name in await run_in_processpool_with_wix(
                    get_model_names,
                    wix=wix,
                )
            ],
        )
