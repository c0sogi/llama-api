from contextlib import asynccontextmanager
from os import environ, getpid
from pathlib import Path
from random import randint
import sys
from threading import Timer
from typing import Literal, Optional

from ..shared.config import AppSettingsCliArgs, MainCliArgs, Config

from ..utils.dependency import (
    get_installed_packages,
    get_poetry_executable,
    git_clone,
    git_pull,
    run_command,
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

        if sys.platform == "win32":
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


def initialize_before_launch() -> None:
    """Initialize the app"""
    args = MainCliArgs
    install_packages = args.install_pkgs.value or False
    upgrade = args.upgrade.value or False
    force_cuda = args.force_cuda.value or False
    skip_pytorch_install = args.skip_torch_install.value or False
    skip_tensorflow_install = args.skip_tf_install.value or False
    skip_compile = args.skip_compile.value or False
    no_cache_dir = args.no_cache_dir.value or False
    print(
        "Starting Application with CLI args:" + str(environ["LLAMA_API_ARGS"])
    )

    # PIP arguments
    pip_args = []  # type: list[str]
    if no_cache_dir:
        pip_args.append("--no-cache-dir")
    if upgrade:
        pip_args.append("--upgrade")
        # Upgrade pip
        run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            action="upgrad",
            name="pip",
        )

    # Clone all repositories
    for git_clone_args in Config.repositories.values():
        git_clone(**git_clone_args)
        if upgrade:
            git_pull(
                git_clone_args["git_path"], options=["--recurse-submodules"]
            )

    # Install packages
    if install_packages:
        if not skip_compile:
            # Build the shared library of LLaMA C++ code
            build_shared_lib(logger=logger, force_cuda=force_cuda)
        poetry = get_poetry_executable()
        if not poetry.exists():
            # Install poetry
            logger.warning(f"⚠️ Poetry not found: {poetry}")
            install_package("poetry", force=True, args=pip_args)
        if not skip_pytorch_install:
            # Install pytorch
            install_pytorch(force_cuda=force_cuda, args=pip_args)
        if not skip_tensorflow_install:
            # Install tensorflow
            install_tensorflow(args=pip_args)

        # Install all dependencies of our project and other repositories
        project_paths = [Path(".")] + list(Path("repositories").glob("*"))
        install_all_dependencies(project_paths=project_paths, args=pip_args)

        # Get current packages installed
        logger.info(f"📦 Installed packages: {get_installed_packages()}")
    else:
        logger.warning(
            "🏃‍♂️ Skipping package installation... "
            "If any packages are missing, "
            "use `--install-pkgs` option to install them."
        )
    # if MainCliArgs.xformers.value:
    #     install_package("xformers", args=pip_args)


@asynccontextmanager
async def lifespan(app):
    from ..utils.logger import ApiLogger

    ApiLogger.cinfo("🦙 LLaMA API server is running")
    try:
        yield
    finally:
        from ..utils.concurrency import _pool, _manager

        if _manager is not None:
            _manager.shutdown()
        if _pool is not None:
            for wix in _pool.active_workers:
                wix.send_q.close()
                wix.recv_q.close()
                pid = wix.process.pid
                if pid is not None:
                    ApiLogger.cinfo(f"🔧 Worker {wix.process.pid} is stopping")
                    wix.process.kill()
            _pool.join()
        ApiLogger.ccritical("🦙 LLaMA API server is stopped")


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


def run() -> None:
    MainCliArgs.load()
    port = MainCliArgs.port.value
    assert port is not None, "Port is not set"
    if MainCliArgs.force_cuda.value:
        environ["FORCE_CUDA"] = "1"
    initialize_before_launch()

    from uvicorn import Config as UvicornConfig
    from uvicorn import Server as UvicornServer

    if MainCliArgs.tunnel.value:
        install_package("flask-cloudflared")
        from flask_cloudflared import start_cloudflared

        thread = Timer(
            2, start_cloudflared, args=(port, randint(8100, 9000), None, None)
        )
        thread.daemon = True
        thread.start()

    UvicornServer(
        config=UvicornConfig(
            create_app_llama_cpp(),
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
    ).run()


if __name__ == "__main__":
    AppSettingsCliArgs.load()
    initialize_before_launch()
