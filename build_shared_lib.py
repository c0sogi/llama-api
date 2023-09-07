from argparse import ArgumentParser
from llama_api.utils.llama_cpp import (
    build_shared_lib,
    CPU_ARGS,  # Only use CPU
    METAL_ARGS,  # Only use Metal (MacOS)
    CUBLAS_ARGS,  # Only use CUBLAS (Nvidia)
)
from os import environ

ARGS = {
    "CPU": CPU_ARGS,
    "METAL": METAL_ARGS,
    "CUBLAS": CUBLAS_ARGS,
    "CUDA": CUBLAS_ARGS,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--build_type",
        type=lambda s: str(s).upper(),
        default="CPU",
        choices=["CPU", "METAL", "CUBLAS", "CUDA"],
        help="Build type",
    )

    environ["FORCE_CMAKE"] = "1"
    environ["CMAKE_ARGS"] = ARGS[parser.parse_args().build_type]
    build_shared_lib()
