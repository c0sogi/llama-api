from os import environ

from llama_api.shared.config import BuildSharedLibCliArgs as args
from llama_api.utils.llama_cpp import CPU_ARGS  # Only use CPU
from llama_api.utils.llama_cpp import OPENBLAS_ARGS  # Only use CPU
from llama_api.utils.llama_cpp import CUBLAS_ARGS  # Only use CUBLAS (Nvidia)
from llama_api.utils.llama_cpp import METAL_ARGS  # Only use Metal (MacOS)
from llama_api.utils.llama_cpp import build_shared_lib

BACKENDS = {
    "cpu": CPU_ARGS,
    "openblas": OPENBLAS_ARGS,
    "metal": METAL_ARGS,
    "cublas": CUBLAS_ARGS,
    "cuda": CUBLAS_ARGS,
}

if __name__ == "__main__":
    args.load()
    backend = args.backend.value[0]
    assert backend in BACKENDS, f"Backend `{backend}` is not supported"

    environ["FORCE_CMAKE"] = "1"
    environ["CMAKE_ARGS"] = BACKENDS[backend]
    build_shared_lib()
