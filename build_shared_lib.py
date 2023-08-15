# flake8: noqa

from llama_api.utils.llama_cpp import (
    build_shared_lib,
    CPU_ARGS,  # Only use CPU
    METAL_ARGS,  # Only use Metal (MacOS)
    CUBLAS_ARGS,  # Only use CUBLAS (Nvidia)
)
from os import environ


if __name__ == "__main__":
    environ["FORCE_CMAKE"] = "1"
    environ["CMAKE_ARGS"] = CPU_ARGS  # EDIT THIS LINE TO CHANGE BUILD TYPE !!!
    build_shared_lib()
