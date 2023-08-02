import argparse
from llama_api.server.app_settings import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on; default is 8000",
    )
    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of process workers to run; default is 1",
    )
    parser.add_argument(
        "-i",
        "--install-pkgs",
        action="store_true",
        help="Install all required packages before running the server",
    )
    parser.add_argument(
        "--force-cuda",
        action="store_true",
        help=(
            "Force CUDA version of pytorch to be used"
            "when installing pytorch. e.g. torch==2.0.1+cu118"
        ),
    )
    parser.add_argument(
        "--skip-torch-install",
        action="store_true",
        help="Skip installing pytorch, if `install-pkgs` is set",
    )
    parser.add_argument(
        "--skip-tf-install",
        action="store_true",
        help="Skip installing tensorflow, if `install-pkgs` is set",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compiling the shared library of LLaMA C++ code",
    )

    args = parser.parse_args()
    run(
        port=args.port,
        max_workers=args.max_workers,
        install_packages=args.install_pkgs,
        force_cuda=args.force_cuda,
        skip_pytorch_install=args.skip_torch_install,
        skip_tensorflow_install=args.skip_tf_install,
        skip_compile=args.skip_compile,
    )
