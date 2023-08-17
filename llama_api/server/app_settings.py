import argparse
import platform
from contextlib import asynccontextmanager
from os import environ, getpid
from pathlib import Path
from typing import Dict, Literal, Optional

from ..utils.dependency import (
    get_installed_packages,
    get_poetry_executable,
    install_all_dependencies,
    install_package,
    install_pytorch,
    install_tensorflow,
)
from ..utils.llama_cpp import build_shared_lib
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


def set_priority(
    priority: Literal[
        "low", "below_normal", "normal", "above_normal", "high", "realtime"
    ] = "normal",
    pid: Optional[int] = None,
) -> bool:
    """Set The Priority of a Process.  Priority is a string which can be
    'low', 'below_normal', 'normal', 'above_normal', 'high', 'realtime'.
    'normal' is the default.
    Returns True if successful, False if not."""
    if pid is None:
        pid = getpid()
    try:
        import psutil

        if platform.system() == "Windows":
            priorities = {
                "low": psutil.IDLE_PRIORITY_CLASS,
                "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
                "realtime": psutil.REALTIME_PRIORITY_CLASS,
            }
        else:  # Linux and other Unix systems
            priorities = {
                "low": 19,
                "below_normal": 10,
                "normal": 0,
                "above_normal": -5,
                "high": -11,
                "realtime": -20,
            }
        if priority not in priorities:
            logger.warning(f"⚠️ Invalid priority [{priority}]")
            return False

        p = psutil.Process(pid)
        p.nice(priorities[priority])
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to set priority of process [{pid}]: {e}")
        return False


def initialize_before_launch(
    install_packages: bool = False,
    force_cuda: bool = False,
    skip_pytorch_install: bool = False,
    skip_tensorflow_install: bool = False,
    skip_compile: bool = False,
) -> None:
    """Initialize the app"""
    if install_packages:
        # Install all dependencies
        if not skip_compile:
            # Build the shared library of LLaMA C++ code
            build_shared_lib(logger=logger, force_cuda=force_cuda)
        poetry = get_poetry_executable()
        if not poetry.exists():
            # Install poetry
            logger.warning(f"⚠️ Poetry not found: {poetry}")
            install_package("poetry", force=True)
        if not skip_pytorch_install:
            # Install pytorch
            install_pytorch(force_cuda=force_cuda)
        if not skip_tensorflow_install:
            # Install tensorflow
            install_tensorflow()

        # Install all dependencies of our project and other repositories
        project_paths = [Path(".")] + list(Path("repositories").glob("*"))
        install_all_dependencies(project_paths=project_paths)

        # Get current packages installed
        logger.info(f"📦 Installed packages: {get_installed_packages()}")
    if environ.get("LLAMA_API_XFORMERS") == "1":
        install_package("xformers")
    else:
        logger.warning(
            "🏃‍♂️ Skipping package installation... "
            "If any packages are missing, "
            "use `--install-pkgs` option to install them."
        )


@asynccontextmanager
async def lifespan(app):
    from ..utils.concurrency import pool
    from ..utils.logger import ApiLogger

    ApiLogger.cinfo("🦙 LLaMA API server is running")
    yield
    ApiLogger.ccritical("🦙 Shutting down LLaMA API server...")
    pool().kill()


def create_app_llama_cpp():
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware

    from .routers import v1

    new_app = FastAPI(title="🦙 LLaMA API", version="0.0.1", lifespan=lifespan)
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @new_app.get("/health")
    async def health():
        return "ok"

    new_app.include_router(v1.router)
    return new_app


def run(
    port: int,
    install_packages: bool = False,
    force_cuda: bool = False,
    skip_pytorch_install: bool = False,
    skip_tensorflow_install: bool = False,
    skip_compile: bool = False,
    environs: Optional[Dict[str, str]] = None,
) -> None:
    initialize_before_launch(
        install_packages=install_packages,
        force_cuda=force_cuda,
        skip_pytorch_install=skip_pytorch_install,
        skip_tensorflow_install=skip_tensorflow_install,
        skip_compile=skip_compile,
    )

    from uvicorn import Config as UvicornConfig
    from uvicorn import Server as UvicornServer

    if environs:
        environ.update(environs)

    UvicornServer(
        config=UvicornConfig(
            create_app_llama_cpp(),
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
    ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
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

    initialize_before_launch(
        install_packages=args.install_pkgs,
        force_cuda=args.force_cuda,
        skip_pytorch_install=args.skip_torch_install,
        skip_tensorflow_install=args.skip_tf_install,
        skip_compile=args.skip_compile,
    )
