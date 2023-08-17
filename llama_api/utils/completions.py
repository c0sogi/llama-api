from time import time
from typing import Iterator, Literal, Optional
from uuid import uuid4

from ..schemas.api import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionLogprobs,
    CompletionUsage,
    FunctionCallUnparsed,
)

# ==== CHAT COMPLETION ====#


def make_chat_completion(
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    index: int = 0,
    id: Optional[str] = None,
    role: Optional[Literal["user", "system", "assistant"]] = None,
    created: Optional[int] = None,
    finish_reason: Optional[str] = None,
    user: Optional[str] = None,
    function_name: Optional[str] = None,
    function_args: Optional[str] = None,
) -> ChatCompletion:
    """A helper method to make a chat completion."""
    if id is None:
        id = f"cmpl-{str(uuid4())}"
    if created is None:
        created = int(time())
    if role is None:
        role = "assistant"
    message = ChatCompletionMessage(role=role, content=content)
    if user is not None:
        message["user"] = user
    if function_name is not None:
        function_call = FunctionCallUnparsed(name=function_name)
        if function_args is not None:
            function_call["arguments"] = function_args
        message["function_call"] = function_call
    return ChatCompletion(
        id=id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=index,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_chat_completion_from_json(
    json_data: dict,
    index: int = 0,
) -> ChatCompletion:
    """Make ChatCompletion from json data(dict)"""
    usage = json_data.get("usage")
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
    function_call = json_data["choices"][index]["message"].get("function_call")
    if function_call:
        function_name = function_call.get("name")
        function_arguments = function_call.get("arguments")
    else:
        function_name = None
        function_arguments = None
    return make_chat_completion(
        model=json_data["model"],
        content=json_data["choices"][index]["message"]["content"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        index=index,
        id=json_data.get("id"),
        role=json_data["choices"][index]["message"].get("role"),
        user=json_data["choices"][index]["message"].get("user"),
        created=json_data.get("created"),
        finish_reason=json_data["choices"][index].get("finish_reason"),
        function_name=function_name,
        function_args=function_arguments,
    )


def make_chat_completion_chunk(
    id: str,
    model: str,
    created: Optional[int] = None,
    role: Optional[Literal["assistant"]] = None,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    function_name: Optional[str] = None,
    function_args: Optional[str] = None,
) -> ChatCompletionChunk:
    """A helper method to make a chat completion chunk."""
    if created is None:
        created = int(time())
    delta = ChatCompletionChunkDelta()
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if function_name is not None or function_args is not None:
        function_call = FunctionCallUnparsed()
        if function_name is not None:
            function_call["name"] = function_name
        if function_args is not None:
            function_call["arguments"] = function_args
        delta["function_call"] = function_call
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )


def make_chat_completion_chunk_from_json(
    json_data: dict,
) -> ChatCompletionChunk:
    """Make ChatCompletionChunk from json data(dict)"""
    delta = json_data["choices"][0]["delta"]
    function_call = delta.get("function_call")
    if function_call:
        function_name = function_call.get("name")
        function_arguments = function_call.get("arguments")
    else:
        function_name = None
        function_arguments = None
    return make_chat_completion_chunk(
        id=json_data["id"],
        model=json_data["model"],
        role=delta.get("role"),
        content=delta.get("content"),
        finish_reason=json_data["choices"][0].get("finish_reason"),
        function_name=function_name,
        function_args=function_arguments,
    )


# ==== TEXT COMPLETION ==== #


def make_completion_chunk(
    id: str,
    model: str,
    text: str,
    index: int = 0,
    finish_reason: Optional[str] = None,
    logprobs: Optional[CompletionLogprobs] = None,
    created: Optional[int] = None,
) -> CompletionChunk:
    """A helper method to make a completion chunk."""
    if created is None:
        created = int(time())
    return CompletionChunk(
        id=id,
        object="text_completion",
        created=created,
        model=model,
        choices=[
            CompletionChoice(
                text=text,
                index=index,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ],
    )


def make_completion_chunk_from_json(
    json_data: dict,
) -> CompletionChunk:
    """Make CompletionChunk from json data(dict)"""
    choice = json_data["choices"][0]
    return make_completion_chunk(
        id=json_data["id"],
        model=json_data["model"],
        text=choice["text"],
        index=choice.get("index", 0),
        finish_reason=choice.get("finish_reason"),
        logprobs=choice.get("logprobs"),
        created=json_data.get("created"),
    )


def make_completion(
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    index: int = 0,
    id: Optional[str] = None,
    created: Optional[int] = None,
    finish_reason: Optional[str] = None,
    logprobs: Optional[CompletionLogprobs] = None,
) -> Completion:
    """A helper method to make a completion."""
    if id is None:
        id = f"cmst-{str(uuid4())}"
    if created is None:
        created = int(time())
    return Completion(
        id=id,
        object="text_completion",
        created=created,
        model=model,
        choices=[
            CompletionChoice(
                text=text,
                index=index,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_completion_from_json(
    json_data: dict,
    index: int = 0,
) -> Completion:
    """Make Completion from json data(dict)"""
    usage = json_data.get("usage")
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
    return make_completion(
        id=json_data["id"],
        model=json_data["model"],
        text=json_data["choices"][index]["text"],
        index=index,
        finish_reason=json_data["choices"][index].get("finish_reason"),
        logprobs=json_data["choices"][index].get("logprobs"),
        created=json_data.get("created"),
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )


def convert_text_completion_to_chat(completion: Completion) -> ChatCompletion:
    return ChatCompletion(
        id="chat" + completion["id"],
        object="chat.completion",
        created=completion["created"],
        model=completion["model"],
        choices=[
            ChatCompletionChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                finish_reason=completion["choices"][0]["finish_reason"],
            )
        ],
        usage=completion["usage"],
    )


def convert_text_completion_chunks_to_chat(
    chunks: Iterator[CompletionChunk],
) -> Iterator[ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield ChatCompletionChunk(
                id="chat" + chunk["id"],
                model=chunk["model"],
                created=chunk["created"],
                object="chat.completion.chunk",
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta={"role": "assistant"},
                        finish_reason=None,
                    )
                ],
            )
        yield ChatCompletionChunk(
            id="chat" + chunk["id"],
            model=chunk["model"],
            created=chunk["created"],
            object="chat.completion.chunk",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta={
                        "content": chunk["choices"][0]["text"],
                    }
                    if chunk["choices"][0]["finish_reason"] is None
                    else {},
                    finish_reason=chunk["choices"][0]["finish_reason"],
                )
            ],
        )
