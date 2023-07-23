import re
from typing import Optional
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from llama_api.schemas.api import ModelList

ggml_model: str = "orca-mini-3b.ggmlv3.q4_1.bin"
ggml_path: Path = Path(__file__).parents[1] / Path(f"models/ggml/{ggml_model}")

gptq_model: str = "orca_mini_7b"
gptq_path: Path = Path(__file__).parents[1] / Path(f"models/gptq/{gptq_model}")

messages = [{"role": "user", "content": "Hello, there!"}]
prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def test_health(app):
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200


@pytest.mark.skipif(
    not ggml_path.exists(), reason=f"Test model not found {ggml_path}"
)
def test_llama_cpp(app):
    with TestClient(app) as client:
        response = client.get("/v1/models")
        model_list: ModelList = response.json()
        model_name: Optional[str] = None
        for model_data in model_list["data"]:
            if ggml_model in model_data["id"]:
                model_name = re.sub(r"\(.*\)", "", model_data["id"]).strip()
                break
        assert model_name, f"Model {ggml_model} not found"

        request = {"model": model_name, "max_tokens": 50}
        for endpoint in ("completions", "chat/completions"):
            for stream in (True, False):
                request_message = (
                    {"messages": messages}
                    if endpoint.startswith("chat")
                    else {"prompt": prompt}
                )
                response = client.post(
                    f"/v1/{endpoint}",
                    json=request | {"stream": stream} | request_message,
                    headers={"Content-Type": "application/json"},
                )
                assert response.status_code == 200


@pytest.mark.skipif(
    not gptq_path.exists(), reason=f"Test model not found {gptq_path}"
)
def test_exllama(app):
    with TestClient(app) as client:
        response = client.get("/v1/models")
        model_list: ModelList = response.json()
        model_name: Optional[str] = None
        for model_data in model_list["data"]:
            if gptq_model in model_data["id"]:
                model_name = re.sub(r"\(.*\)", "", model_data["id"]).strip()
                break
        assert model_name, f"Model {gptq_model} not found"

        request = {"model": model_name, "max_tokens": 50}
        for endpoint in ("completions", "chat/completions"):
            for stream in (True, False):
                request_message = (
                    {"messages": messages}
                    if endpoint.startswith("chat")
                    else {"prompt": prompt}
                )
                response = client.post(
                    f"/v1/{endpoint}",
                    json=request | {"stream": stream} | request_message,
                    headers={"Content-Type": "application/json"},
                )
                assert response.status_code == 200
