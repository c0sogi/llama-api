from asyncio import gather, iscoroutinefunction
from contextlib import ExitStack
from datetime import datetime
from functools import wraps
import importlib
from types import ModuleType
import unittest
from os import environ
from pathlib import Path
from re import compile, sub
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from unittest.mock import MagicMock, patch

from orjson import loads
from llama_api.schemas.api import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    CompletionChoice,
    CompletionChunk,
    ModelList,
)

from llama_api.server.app_settings import create_app_llama_cpp
from llama_api.shared.config import Config
from llama_api.utils.concurrency import _pool
from llama_api.utils.dependency import install_package, is_package_available
from llama_api.utils.system import get_cuda_version

if TYPE_CHECKING:
    from typing import Type  # noqa: F401

    from fastapi.testclient import TestClient  # noqa: F401
    from httpx import AsyncClient, Response  # noqa: F401


EndPoint = Literal["completions", "chat/completions"]


def patch_module(mocking_module: ModuleType):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            patches = []
            for name, attr in mocking_module.__dict__.items():
                # Mock all functions and classes
                if callable(attr) or isinstance(attr, (type,)):
                    patches.append(
                        patch.object(mocking_module, name, MagicMock())
                    )

            with ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)

                if iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

        if iscoroutinefunction(func):
            return async_wrapper
        return func

    return decorator


class TestLlamaAPI(unittest.TestCase):
    ggml_model: str = "orca-mini-3b.ggmlv3.q4_0.bin"
    ggml_path: Path = Config.project_root / Path(f"models/ggml/{ggml_model}")

    gptq_model: str = "orca_mini_7b"
    gptq_path: Path = Config.project_root / Path(f"models/gptq/{gptq_model}")

    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello, there!"}
    ]
    prompt: str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    @classmethod
    def setUpClass(cls):
        if not is_package_available("httpx"):
            install_package("httpx")
        cls.AsyncClient = importlib.import_module(
            "httpx"
        ).AsyncClient  # type: Type[AsyncClient]
        cls.TestClient = importlib.import_module(
            "fastapi.testclient"
        ).TestClient  # type: Type[TestClient]
        cls.app = create_app_llama_cpp()
        environ["LLAMA_API_ARGS"] = '{"MAX_WORKERS": 1}'

    @classmethod
    def tearDownClass(cls):
        if _pool is not None:
            _pool.shutdown(wait=True)

    @property
    def check_ggml(self) -> None:
        if not self.ggml_path.exists():
            self.skipTest(f"No model in {self.ggml_path}")

    @property
    def check_gptq(self) -> None:
        if not self.gptq_path.exists():
            self.skipTest(f"No model in {self.gptq_path}")

    @property
    def check_cuda(self) -> None:
        if not get_cuda_version():
            self.skipTest("CUDA is not available")

    async def arequest_completion(
        self,
        model_names: Union[List[str], Tuple[str, ...]],
        endpoints: Union[EndPoint, Iterable[EndPoint]],
        **kwargs: Any,
    ) -> Tuple[List[List[str]], List[datetime], List[datetime]]:
        async with self.AsyncClient(
            app=self.app, base_url="http://localhost", timeout=None
        ) as client:
            # Get models using the API
            models = await self.get_models(
                client=client, model_names=list(model_names)
            )  # type: List[str]

            # Submit requests to the API and get responses
            return await self.submit_streaming_requests(
                client=client,
                model_and_endpoints=zip(
                    models,
                    (
                        [endpoints] * len(model_names)  # type: ignore
                        if isinstance(endpoints, str)
                        else endpoints
                    ),
                ),
                **kwargs,
            )

    async def get_models(
        self, client: "AsyncClient", model_names: List[str]
    ) -> List[str]:
        # Get models using the API
        model_resp: ModelList = (await client.get("/v1/models")).json()
        models: List[str] = []
        for model_name in model_names:
            model: Optional[str] = None
            for model_data in model_resp["data"]:
                if model_name in model_data["id"]:
                    model = sub(r"\(.*\)", "", model_data["id"]).strip()
                    break
            self.assertTrue(model, f"Model {model_name} not found")
            models.append(str(model))
        return models

    async def submit_streaming_requests(
        self,
        client: "AsyncClient",
        model_and_endpoints: Iterable[Tuple[str, EndPoint]],
        **kwargs: Any,
    ) -> Tuple[List[List[str]], List[datetime], List[datetime]]:
        async def send_request(
            model: str, endpoint: EndPoint
        ) -> Tuple[List[str], datetime, datetime]:
            async with client.stream(
                method="POST",
                url=f"/v1/{endpoint}",
                json=self.union(
                    {"model": model, "max_tokens": 50},
                    {"stream": True},
                    {"messages": self.messages}
                    if endpoint.startswith("chat")
                    else {"prompt": self.prompt},
                    kwargs,
                ),
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                start_at = datetime.now()
                results = []  # type: List[str]
                async for chunk in self.extract_json_from_streaming_response(
                    response
                ):
                    self.assertIn("choices", chunk, "No choices in response")
                    choice = chunk["choices"][0]
                    if "delta" in choice and choice["delta"].get("content"):
                        results.append(choice["delta"]["content"])
                    elif "text" in choice:
                        results.append(choice["text"])
            self.assertGreaterEqual(len(results), 1, "No result in response")
            return results, start_at, datetime.now()

        tasks = [
            send_request(model, endpoint)
            for model, endpoint in model_and_endpoints
        ]
        return tuple(zip(*await gather(*tasks)))  # type: ignore

    def harvest_results(
        self, models: List[str], responses: List["Response"]
    ) -> List[str]:
        results: List[str] = []
        for model, response in zip(models, responses):
            self.assertEqual(response.status_code, 200)
            choice: Union[
                CompletionChoice, ChatCompletionChoice
            ] = response.json()["choices"][0]
            if "message" in choice:
                results.append(choice["message"]["content"])
            elif "text" in choice:
                results.append(choice["text"])
            else:
                raise ValueError(f"Unknown response: {response.json()}")
            print(f"Result of {model}:", results[-1], end="\n\n", flush=True)
        self.assertEqual(len(results), len(models))
        return results

    async def extract_json_from_streaming_response(
        self,
        response: "Response",
    ) -> AsyncIterator[Union[CompletionChunk, ChatCompletionChunk]]:
        """Extract json from streaming `httpx.Response`"""
        regex_finder = compile(rb"data:\s*({.+?})\s*\r?\n\s*\r?\n").finditer
        bytes_buffer = bytearray()
        async for stream in response.aiter_bytes():
            bytes_buffer.extend(stream)
            for match in regex_finder(bytes_buffer):
                try:
                    json_data = loads(match.group(1))
                    yield json_data
                    bytes_buffer.clear()
                except Exception:
                    continue

    @staticmethod
    def union(*dicts: Dict) -> Dict:
        return {k: v for d in dicts for k, v in d.items()}
