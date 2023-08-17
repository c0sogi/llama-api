# flake8: noqa
import unittest

from tests.conftest import TestLlamaAPI


class TestServerBasic(TestLlamaAPI):
    """Test the FastAPI server with basic health checks"""

    def test_health(self):
        """Test the health endpoint"""
        with self.TestClient(app=self.app) as client:
            response = client.get(
                "/health",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 200)

    def test_v1_models(self):
        """Test the v1/models endpoint"""
        with self.TestClient(app=self.app) as client:
            response = client.get(
                "/v1/models",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 200)

    def test_import_llama_cpp(self):
        try:
            from llama_api.modules.llama_cpp import (
                LlamaCppCompletionGenerator,  # noqa: F401
            )
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")

    def test_import_exllama(self):
        self.check_cuda
        try:
            from llama_api.modules.exllama import (
                ExllamaCompletionGenerator,  # noqa: F401
            )
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")

    def test_import_sentence_encoder(self):
        try:
            from llama_api.modules.sentence_encoder import (
                SentenceEncoderEmbeddingGenerator,  # noqa: F401
            )
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")

    def test_import_transformer(self):
        try:
            from llama_api.modules.transformer import (
                TransformerEmbeddingGenerator,  # noqa: F401
            )  #
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")


class TestServerAdvanced(TestLlamaAPI, unittest.IsolatedAsyncioTestCase):
    """Test the FastAPI server with advanced completion tests"""

    async def test_llama_cpp(self):
        """Test the Llama CPP model completion endpoints"""
        self.check_ggml
        model_names = (self.ggml_model, self.ggml_model)
        responses, starts, ends = await self.arequest_completion(
            model_names=model_names,
            endpoints=("chat/completions", "completions"),
        )
        start_1, end_1 = starts[0], ends[0]
        print(f"GGML response: {''.join(responses[0])}", flush=True)
        start_2, end_2 = starts[1], ends[1]
        print(f"GGML response: {''.join(responses[1])}", flush=True)

        self.assertTrue(
            end_1 < start_2 or end_2 < start_1,
            f"Synchronous completion failed: {end_1} < {start_2} and {end_2} < {start_1}",
        )

    async def test_exllama(self):
        """Test the ExLLama model completion endpoints"""
        self.check_gptq
        model_names = (self.gptq_model, self.gptq_model)
        responses, starts, ends = await self.arequest_completion(
            model_names=model_names,
            endpoints=("chat/completions", "completions"),
        )
        start_1, end_1 = starts[0], ends[0]
        print(f"GPTQ response: {''.join(responses[0])}", flush=True)
        start_2, end_2 = starts[1], ends[1]
        print(f"GPTQ response: {''.join(responses[1])}", flush=True)

        self.assertTrue(
            end_1 < start_2 or end_2 < start_1,
            f"Synchronous completion failed: {end_1} < {start_2} and {end_2} < {start_1}",
        )

    async def test_llama_mixed_concurrency(self):
        """Test the Llama CPP & ExLLama model completion endpoints
        with concurrency"""
        self.check_ggml
        self.check_gptq
        model_names = (self.ggml_model, self.gptq_model)
        responses, starts, ends = await self.arequest_completion(
            model_names=model_names, endpoints="completions"
        )
        start_1, end_1 = starts[0], ends[0]
        print(f"GGML response: {''.join(responses[0])}", flush=True)
        start_2, end_2 = starts[1], ends[1]
        print(f"GPTQ response: {''.join(responses[1])}", flush=True)

        self.assertTrue(
            start_2 < end_1 or start_1 < end_2,
            f"Asynchronous completion failed: {start_1} < {end_2} and {start_2} < {end_1}",
        )
