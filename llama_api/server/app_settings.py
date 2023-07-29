import platform
import subprocess
import sys
from contextlib import asynccontextmanager
from logging import warn
from os import environ
from pathlib import Path
from typing import Dict, Optional, Union

from ..shared.config import Config
from ..utils.dependency import (
    get_poetry_executable,
    git_clone,
    install_all_dependencies,
    install_poetry,
    install_tensorflow,
    install_torch,
    is_package_available,
)
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


def ensure_packages_installed():
    """Install the packages in the requirements.txt file"""
    warn(
        "This function is deprecated. Use `install_dependencies` instead.",
        DeprecationWarning,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--trusted-host",
            "pypi.python.org",
            "-r",
            "requirements.txt",
        ],
        env=environ.copy(),
    )


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
) -> None:
    """Initialize the app"""

    # Git clone the repositories
    if git_and_disk_paths is not None:
        for git_path, disk_path in git_and_disk_paths.items():
            git_clone(git_path=git_path, disk_path=disk_path)

    if install_packages:
        # Install the dependencies
        poetry = get_poetry_executable()
        if not poetry.exists():
            logger.warning(f"⚠️ Poetry not found: {poetry}")
            install_poetry()
        if not is_package_available("torch"):
            install_torch()
        if not is_package_available("tensorflow"):
            install_tensorflow()
        project_paths = [Path(".")] + list(Path("repositories").glob("*"))
        install_all_dependencies(project_paths=project_paths)

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
) -> None:
    initialize_before_launch(
        git_and_disk_paths=Config.git_and_disk_paths,
        install_packages=True,
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
