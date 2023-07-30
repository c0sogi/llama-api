import importlib
import unittest
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Type  # noqa: F401

from llama_api.server.app_settings import create_app_llama_cpp
from llama_api.shared.config import Config
from llama_api.utils.dependency import install_package, is_package_available
from llama_api.utils.concurrency import _pool

if TYPE_CHECKING:
    from fastapi.testclient import TestClient  # noqa: F401
    from httpx import AsyncClient  # noqa: F401


class TestLlamaAPI(unittest.TestCase):
    ggml_model: str = "orca-mini-3b.ggmlv3.q4_1.bin"
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
        environ["MAX_WORKERS"] = "2"

    @classmethod
    def tearDownClass(cls):
        if _pool is not None:
            _pool.shutdown(wait=True)
