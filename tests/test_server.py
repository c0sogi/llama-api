import re
from asyncio import gather
from pathlib import Path
import sys
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
)
from fastapi.testclient import TestClient

import pytest
from httpx import AsyncClient, Response

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from llama_api.schemas.api import (  # noqa: E402
    ModelList,
    ChatCompletionChoice,
    CompletionChoice,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

EndPoint = Literal["completions", "chat/completions"]

GGML_MODEL: str = "orca-mini-3b.ggmlv3.q4_1.bin"
GGML_PATH: Path = Path(__file__).parents[1] / Path(f"models/ggml/{GGML_MODEL}")

GPTQ_MODEL: str = "orca_mini_7b"
GPTQ_PATH: Path = Path(__file__).parents[1] / Path(f"models/gptq/{GPTQ_MODEL}")

MESSAGES = [{"role": "user", "content": "Hello, there!"}]
PROMPT = "\n".join([f"{m['role']}: {m['content']}" for m in MESSAGES])


async def _arequest_completion(
    app: "FastAPI",
    model_names: List[str] | Tuple[str, ...],
    endpoints: EndPoint | Iterable[EndPoint],
):
    _endpoints: Iterable[str] = (
        [endpoints] * len(model_names)
        if isinstance(endpoints, str)
        else endpoints
    )
    async with AsyncClient(
        app=app, base_url="http://localhost", timeout=None
    ) as client:
        # Get models using the API
        model_resp: ModelList = (await client.get("/v1/models")).json()
        models: list[str] = []
        for model_name in model_names:
            model: Optional[str] = None
            for model_data in model_resp["data"]:
                if model_name in model_data["id"]:
                    model = re.sub(r"\(.*\)", "", model_data["id"]).strip()
                    break
            assert model, f"Model {model_name} not found"
            models.append(model)

        # Submit requests to the API
        tasks: list[Awaitable] = []
        for model, endpoint in zip(models, _endpoints):
            request = {"model": model, "max_tokens": 50}
            request_message = (
                {"messages": MESSAGES}
                if endpoint.startswith("chat")
                else {"prompt": PROMPT}
            )
            tasks.append(
                client.post(
                    f"/v1/{endpoint}",
                    json=request | {"stream": False} | request_message,
                    headers={"Content-Type": "application/json"},
                    timeout=None,
                )
            )

        # Wait for responses
        cmpl_resps: list[Response] = await gather(*tasks)
        results: list[str] = []
        for model, cmpl_resp in zip(models, cmpl_resps):
            assert cmpl_resp.status_code == 200
            choice: CompletionChoice | ChatCompletionChoice = cmpl_resp.json()[
                "choices"
            ][0]
            if "message" in choice:
                results.append(choice["message"]["content"])  # type: ignore
            elif "text" in choice:
                results.append(choice["text"])
            else:
                raise ValueError(f"Unknown response: {cmpl_resp.json()}")
            print(f"Result of {model}:", results[-1], end="\n\n", flush=True)

    assert len(results) == len(models)


def _request_completion(
    app: "FastAPI",
    model_names: List[str] | Tuple[str, ...],
    endpoints: EndPoint | Iterable[EndPoint],
):
    _endpoints: Iterable[str] = (
        [endpoints] * len(model_names)
        if isinstance(endpoints, str)
        else endpoints
    )
    with TestClient(app=app) as client:
        # Get models using the API
        model_resp = (client.get("/v1/models")).json()
        models: list[str] = []
        for model_name in model_names:
            model: Optional[str] = None
            for model_data in model_resp["data"]:
                if model_name in model_data["id"]:
                    model = re.sub(r"\(.*\)", "", model_data["id"]).strip()
                    break
            assert model, f"Model {model_name} not found"
            models.append(model)

        # Submit requests to the API
        cmpl_resps: list[Response] = []
        for model, endpoint in zip(models, _endpoints):
            request = {"model": model, "max_tokens": 50}
            request_message = (
                {"messages": MESSAGES}
                if endpoint.startswith("chat")
                else {"prompt": PROMPT}
            )
            cmpl_resps.append(
                client.post(
                    f"/v1/{endpoint}",
                    json=request | {"stream": False} | request_message,
                    headers={"Content-Type": "application/json"},
                    timeout=None,
                )
            )

        # Wait for responses
        results: list[str] = []
        for model, cmpl_resp in zip(models, cmpl_resps):
            assert cmpl_resp.status_code == 200
            choice: CompletionChoice | ChatCompletionChoice = cmpl_resp.json()[
                "choices"
            ][0]
            if "message" in choice:
                results.append(choice["message"]["content"])  # type: ignore
            elif "text" in choice:
                results.append(choice["text"])
            else:
                raise ValueError(f"Unknown response: {cmpl_resp.json()}")
            print(f"Result of {model}:", results[-1], end="\n\n", flush=True)

    assert len(results) == len(models)


def test_health(app):
    with TestClient(app=app) as client:
        response = client.get(
            "/health",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200


@pytest.mark.skipif(not GGML_PATH.exists(), reason=f"No model in {GGML_PATH}")
def test_llama_cpp(app):
    _request_completion(
        app, model_names=(GGML_MODEL,), endpoints="completions"
    )
    _request_completion(
        app, model_names=(GGML_MODEL,), endpoints="chat/completions"
    )


@pytest.mark.skipif(not GPTQ_PATH.exists(), reason=f"No model in{GPTQ_PATH}")
def test_exllama(app):
    _request_completion(
        app, model_names=(GPTQ_MODEL,), endpoints="completions"
    )
    _request_completion(
        app, model_names=(GPTQ_MODEL,), endpoints="chat/completions"
    )


@pytest.mark.asyncio
async def test_llama_cpp_concurrency(app):
    model_names: Tuple[str, ...] = (GGML_MODEL, GGML_MODEL)
    await _arequest_completion(
        app, model_names=model_names, endpoints="completions"
    )


@pytest.mark.asyncio
async def test_exllama_concurrency(app):
    model_names: Tuple[str, ...] = (GPTQ_MODEL, GPTQ_MODEL)
    await _arequest_completion(
        app, model_names=model_names, endpoints="completions"
    )


@pytest.mark.asyncio
async def test_llama_mixed_concurrency(app):
    model_names: Tuple[str, ...] = (GGML_MODEL, GPTQ_MODEL)
    await _arequest_completion(
        app, model_names=model_names, endpoints="completions"
    )
