import platform
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from typing import Dict, Optional, Union

from ..shared.config import Config
from ..utils.dependency import (
    get_poetry_executable,
    git_clone,
    install_all_dependencies,
    install_package,
    install_tensorflow,
    install_pytorch,
    is_package_available,
)
from ..utils.llama_cpp import build_shared_lib
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


def set_priority(pid: Optional[int] = None, priority: str = "high"):
    import platform
    from os import getpid

    import psutil

    """Set The Priority of a Process.  Priority is a string which can be
    'low', 'below_normal', 'normal', 'above_normal', 'high', 'realtime'.
    'normal' is the default."""

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

    if pid is None:
        pid = getpid()
    p = psutil.Process(pid)
    p.nice(priorities[priority])


def initialize_before_launch(
    git_and_disk_paths: Optional[Dict[str, Union[str, Path]]] = None,
    install_packages: bool = False,
    force_cuda: bool = False,
    skip_pytorch_install: bool = False,
    skip_tensorflow_install: bool = False,
) -> None:
    """Initialize the app"""

    # Git clone the repositories
    if git_and_disk_paths is not None:
        for git_path, disk_path in git_and_disk_paths.items():
            git_clone(git_path=git_path, disk_path=disk_path)

    if install_packages:
        # Install all dependencies
        poetry = get_poetry_executable()
        if not poetry.exists():
            # Install poetry
            logger.warning(f"⚠️ Poetry not found: {poetry}")
            install_package("poetry", force=True)
        if not skip_pytorch_install or not is_package_available("torch"):
            # Install pytorch
            install_pytorch(force_cuda=force_cuda)
        if not skip_tensorflow_install or not is_package_available(
            "tensorflow"
        ):
            # Install tensorflow
            install_tensorflow()

        # Install all dependencies of our project and other repositories
        project_paths = [Path(".")] + list(Path("repositories").glob("*"))
        install_all_dependencies(project_paths=project_paths)

        # Build the shared library of LLaMA C++ code
        build_shared_lib(logger=logger)
    else:
        logger.warning(
            "🏃‍♂️ Skipping package installation..."
            "If any packages are missing, "
            "use `--install-packages` option to install them."
        )

    # Set the priority of the process
    if platform.system() == "Windows":
        set_priority(priority="high")
    else:
        set_priority(priority="normal")


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
    max_workers: int = 1,
    install_packages: bool = False,
    force_cuda: bool = False,
    skip_pytorch_install: bool = False,
    skip_tensorflow_install: bool = False,
) -> None:
    initialize_before_launch(
        git_and_disk_paths=Config.git_and_disk_paths,
        install_packages=install_packages,
        force_cuda=force_cuda,
        skip_pytorch_install=skip_pytorch_install,
        skip_tensorflow_install=skip_tensorflow_install,
    )

    from uvicorn import Config as UvicornConfig
    from uvicorn import Server as UvicornServer

    environ["MAX_WORKERS"] = str(max_workers)

    UvicornServer(
        config=UvicornConfig(
            create_app_llama_cpp(),
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
    ).run()


if __name__ == "__main__":
    # Git clone the repositories & install the dependencies.
    # This is done before the server is launched to avoid
    # the overhead of doing it when the server is running.
    # Forcing CUDA is necessary when building the docker image
    initialize_before_launch(
        git_and_disk_paths=Config.git_and_disk_paths,
        install_packages=True,
        force_cuda=True,
        skip_pytorch_install=False,
        skip_tensorflow_install=False,
    )
