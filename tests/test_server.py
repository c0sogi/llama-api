import re
from asyncio import gather
from typing import (
    Awaitable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import unittest
from fastapi.testclient import TestClient
from tests.conftest import TestLlamaAPI

from llama_api.schemas.api import (
    ModelList,
    ChatCompletionChoice,
    CompletionChoice,
)


EndPoint = Literal["completions", "chat/completions"]


class TestServer(TestLlamaAPI, unittest.IsolatedAsyncioTestCase):
    """Test the FastAPI server
    with basic health checks & LLM completion requests (with concurrency)"""

    def test_health(self):
        """Test the health endpoint"""
        with TestClient(app=self.app) as client:
            response = client.get(
                "/health",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 200)

    def test_v1_models(self):
        """Test the v1/models endpoint"""
        with TestClient(app=self.app) as client:
            response = client.get(
                "/v1/models",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 200)

    @unittest.skipIf(
        not TestLlamaAPI.ggml_path.exists(),
        reason=f"No model in {TestLlamaAPI.ggml_path}",
    )
    def test_llama_cpp(self):
        """Test the Llama CPP model completion endpoints"""
        self._request_completion(
            model_names=(self.ggml_model,), endpoints="completions"
        )
        self._request_completion(
            model_names=(self.ggml_model,), endpoints="chat/completions"
        )

    @unittest.skipIf(
        not TestLlamaAPI.gptq_path.exists(),
        reason=f"No model in{TestLlamaAPI.gptq_path}",
    )
    def test_exllama(self):
        """Test the ExLLama model completion endpoints"""
        self._request_completion(
            model_names=(self.gptq_model,), endpoints="completions"
        )
        self._request_completion(
            model_names=(self.gptq_model,), endpoints="chat/completions"
        )

    @unittest.skipIf(
        not TestLlamaAPI.ggml_path.exists(),
        reason=f"No model in {TestLlamaAPI.ggml_path}",
    )
    async def test_llama_cpp_concurrency(self):
        """Test the Llama CPP model completion endpoints with concurrency"""
        model_names: Tuple[str, ...] = (self.ggml_model, self.ggml_model)
        await self._arequest_completion(
            model_names=model_names, endpoints="completions"
        )

    @unittest.skipIf(
        not TestLlamaAPI.gptq_path.exists(),
        reason=f"No model in {TestLlamaAPI.gptq_path}",
    )
    async def test_exllama_concurrency(self):
        """Test the ExLLama model completion endpoints with concurrency"""
        model_names: Tuple[str, ...] = (self.gptq_model, self.gptq_model)
        await self._arequest_completion(
            model_names=model_names, endpoints="completions"
        )

    @unittest.skipIf(
        (not TestLlamaAPI.ggml_path.exists())
        or (not TestLlamaAPI.gptq_path.exists()),
        f"No model in {TestLlamaAPI.ggml_path} or {TestLlamaAPI.gptq_path}",
    )
    async def test_llama_mixed_concurrency(self):
        """Test the Llama CPP & ExLLama model completion endpoints with concurrency"""
        model_names: Tuple[str, ...] = (self.ggml_model, self.gptq_model)
        await self._arequest_completion(
            model_names=model_names, endpoints="completions"
        )

    async def _arequest_completion(
        self,
        model_names: Union[List[str], Tuple[str, ...]],
        endpoints: Union[EndPoint, Iterable[EndPoint]],
    ):
        _endpoints: Iterable[str] = (
            [endpoints] * len(model_names)
            if isinstance(endpoints, str)
            else endpoints
        )
        async with self.AsyncClient(
            app=self.app, base_url="http://localhost", timeout=None
        ) as client:
            # Get models using the API
            model_resp: ModelList = (await client.get("/v1/models")).json()
            models: List[str] = []
            for model_name in model_names:
                model: Optional[str] = None
                for model_data in model_resp["data"]:
                    if model_name in model_data["id"]:
                        model = re.sub(r"\(.*\)", "", model_data["id"]).strip()
                        break
                self.assertTrue(model, f"Model {model_name} not found")
                models.append(str(model))

            # Submit requests to the API
            tasks: List[Awaitable] = []
            for model, endpoint in zip(models, _endpoints):
                request = {"model": model, "max_tokens": 50}
                request_message = (
                    {"messages": self.messages}
                    if endpoint.startswith("chat")
                    else {"prompt": self.prompt}
                )
                tasks.append(
                    client.post(
                        f"/v1/{endpoint}",
                        json=_union(
                            request, {"stream": False}, request_message
                        ),
                        headers={"Content-Type": "application/json"},
                        timeout=None,
                    )
                )

            # Wait for responses
            cmpl_resps: List = await gather(*tasks)
            results: List[str] = []
            for model, cmpl_resp in zip(models, cmpl_resps):
                self.assertEqual(cmpl_resp.status_code, 200)
                choice: Union[
                    CompletionChoice, ChatCompletionChoice
                ] = cmpl_resp.json()["choices"][0]
                if "message" in choice:
                    results.append(choice["message"]["content"])
                elif "text" in choice:
                    results.append(choice["text"])
                else:
                    raise ValueError(f"Unknown response: {cmpl_resp.json()}")
                print(
                    f"Result of {model}:", results[-1], end="\n\n", flush=True
                )

        self.assertEqual(len(results), len(models))

    def _request_completion(
        self,
        model_names: Union[List[str], Tuple[str, ...]],
        endpoints: Union[EndPoint, Iterable[EndPoint]],
    ):
        _endpoints: Iterable[str] = (
            [endpoints] * len(model_names)
            if isinstance(endpoints, str)
            else endpoints
        )
        with TestClient(app=self.app) as client:
            # Get models using the API
            model_resp = (client.get("/v1/models")).json()
            models: List[str] = []
            for model_name in model_names:
                model: Optional[str] = None
                for model_data in model_resp["data"]:
                    if model_name in model_data["id"]:
                        model = re.sub(r"\(.*\)", "", model_data["id"]).strip()
                        break
                self.assertTrue(model, f"Model {model_name} not found")
                models.append(str(model))

            # Submit requests to the API
            cmpl_resps: List = []
            for model, endpoint in zip(models, _endpoints):
                request = {"model": model, "max_tokens": 50}
                request_message = (
                    {"messages": self.messages}
                    if endpoint.startswith("chat")
                    else {"prompt": self.prompt}
                )
                cmpl_resps.append(
                    client.post(
                        f"/v1/{endpoint}",
                        json=_union(
                            request, {"stream": False}, request_message
                        ),
                        headers={"Content-Type": "application/json"},
                        timeout=None,
                    )
                )

            # Wait for responses
            results: List[str] = []
            for model, cmpl_resp in zip(models, cmpl_resps):
                self.assertEqual(cmpl_resp.status_code, 200)
                choice: Union[
                    CompletionChoice, ChatCompletionChoice
                ] = cmpl_resp.json()["choices"][0]
                if "message" in choice:
                    results.append(choice["message"]["content"])
                elif "text" in choice:
                    results.append(choice["text"])
                else:
                    raise ValueError(f"Unknown response: {cmpl_resp.json()}")
                print(
                    f"Result of {model}:", results[-1], end="\n\n", flush=True
                )

        self.assertEqual(len(results), len(models))


def _union(*dicts: Dict) -> Dict:
    return {k: v for d in dicts for k, v in d.items()}
