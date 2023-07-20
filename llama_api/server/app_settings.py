import platform
import subprocess
from contextlib import asynccontextmanager
from typing import Optional


def ensure_packages_installed():
    subprocess.call(
        [
            "pip",
            "install",
            "--trusted-host",
            "pypi.python.org",
            "-r",
            "requirements.txt",
        ]
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


def initialize_before_launch(install_packages: bool = False):
    """Initialize the app"""

    if install_packages:
        ensure_packages_installed()

    if platform.system() == "Windows":
        set_priority(priority="high")
    else:
        set_priority(priority="normal")


@asynccontextmanager
async def lifespan(app):
    from ..utils.logger import ApiLogger

    ApiLogger.ccritical("🦙 LLaMA API server is running")
    yield
    ApiLogger.ccritical("🦙 Shutting down LLaMA API server...")


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


def run(port: int) -> None:
    initialize_before_launch(install_packages=True)

    from uvicorn import Config, Server
    Server(
        config=Config(
            create_app_llama_cpp(),
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
    ).run()
