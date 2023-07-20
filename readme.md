## About this repository
This project aims to provide a simple way to run **LLama.cpp** and **Exllama** models as a OpenAI-like API server.

You can use this server to run the models in your own application, or use it as a standalone API server!

## Before you start

1. **Python 3.11** is required to run the server. You can download it from https://www.python.org/downloads/

2. **CMake** is required to build llama.cpp shared library. You can download it from https://cmake.org/download/

3. **CUDA 11.8** is required to build cuBLAS build of llama.cpp. You can download it from https://developer.nvidia.com/cuda-11-8-0-download-archive


## Where to put the models

> **Note:** The models are not included in this repository. You have to download them from HuggingFace.


### 1. Llama.cpp
The LLama.cpp GGML model must be put here as a `bin file`, in `models/ggml/`.

For example, if you downloaded a q4_0 quantized model from "https://huggingface.co/TheBloke/robin-7B-v2-GGML",
The path of the model has to be `robin-7b.ggmlv3.q4_0.bin`.

### 2. Exllama
The Exllama GPTQ model must be put here as a `folder`, in `models/gptq/`.

For example, if you downloaded 3 files from "https://huggingface.co/TheBloke/orca_mini_7B-GPTQ/tree/main":

- orca-mini-7b-GPTQ-4bit-128g.no-act.order.safetensors
- tokenizer.model
- config.json

Then you need to put them in a folder.
The path of the model has to be the folder name. Let's say, `orca_mini_7b`, which contains the 3 files.

![image](contents/example-models.png)

## Where to define the models
Define llama.cpp & exllama models in `model_definitions.py`. You can define all necessary parameters to load the models there. Refer to the example in the file.

## How to run server

All required packages will be installed automatically.
Simply run the following command:

```bash
python -m main --port 8000
```

Now, you can send a request to the server.

```python
import requests

url = "http://localhost:8000/v1/completions"
payload = {
    "model": "orca_mini_3b",
    "prompt": "Hello, my name is",
    "max_tokens": 30,
    "top_p": 0.9,
    "temperature": 0.9,
    "stop": ["\n"]
}
response = requests.post(url, json=payload)
print(response.json())

# Output:
# {'id': 'cmpl-243b22e4-6215-4833-8960-c1b12b49aa60', 'object': 'text_completion', 'created': 1689857470, 'model': 'D:/llmchat-llama-extension/models/ggml/orca-mini-3b.ggmlv3.q4_1.bin', 'choices': [{'text': " John and I'm excited to share with you how I built a 6-figure online business from scratch! In this video series, I will", 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 6, 'completion_tokens': 30, 'total_tokens': 36}}
```