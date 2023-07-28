from asyncio import AbstractEventLoop, Future, wrap_future
from concurrent.futures import Executor
from contextlib import contextmanager
from multiprocessing.managers import SyncManager
from os import environ
from queue import Queue
from threading import Event
from typing import Callable, Optional, ParamSpec, Tuple, TypeVar

from fastapi.concurrency import run_in_threadpool

from ..server.app_settings import initialize_before_launch
from ..utils.logger import ApiLogger
from ..utils.process_pool import ProcessPool

T = TypeVar("T")
P = ParamSpec("P")

logger = ApiLogger(__name__)
_pool: Optional[ProcessPool] = None
_manager: Optional[SyncManager] = None


def init_process_pool(env_vars: dict[str, str]) -> None:
    """Initialize the process pool,
    and set the environment variables for the child processes"""
    try:
        initialize_before_launch(install_packages=False)
    except Exception:
        pass

    for key, value in env_vars.items():
        environ[key] = value

    cuda_home = environ.get("CUDA_HOME", None)
    cuda_path = environ.get("CUDA_PATH", None)
    if cuda_path is not None and cuda_home is None:
        environ["CUDA_HOME"] = cuda_path
    elif cuda_home is not None and cuda_path is None:
        environ["CUDA_PATH"] = cuda_home


def pool() -> ProcessPool:
    """Get the process pool, and initialize it if it's not initialized yet"""

    global _pool
    if _pool is None:
        logger.critical("Initializing process pool...")
        _pool = ProcessPool(
            max_workers=int(environ.get("MAX_WORKERS", 1)),
            initializer=init_process_pool,
            initargs=(dict(environ),),
        )
    elif not _pool.is_available:
        logger.critical("Process pool died. Reinitializing process pool...")
        _pool = ProcessPool(
            max_workers=int(environ.get("MAX_WORKERS", 1)),
            initializer=init_process_pool,
            initargs=(dict(environ),),
        )
    return _pool


def awake_all_pool_workers() -> None:
    """Awake all the workers in the process pool.
    This is useful when the workers are not awake yet,
    and you want to make sure they are awake before submitting jobs."""

    ppool = pool()
    for wix in range(ppool.max_workers):
        ppool.worker_at_wix(wix)


def run_in_executor(
    loop: AbstractEventLoop,
    executor: Executor,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Future[T]:
    """Run a function in an executor, and return a future"""

    if loop.is_closed:
        raise RuntimeError("Event loop is closed")
    return wrap_future(executor.submit(func, *args, **kwargs), loop=loop)


async def run_in_processpool_with_wix(func: Callable[[], T], wix: int) -> T:
    """Run a function in the process pool, and return the result.
    The function will be run in the worker at the specified worker-index(wix).
    This is useful when you want to run a function in a specific worker, which
    has some specific resources that the other workers don't have."""

    return await run_in_threadpool(pool().run_with_wix, func, wix)


async def run_in_processpool(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run a function in the process pool, and return the result
    This is useful when you want to run a function in any worker,
    and you don't care which worker it is."""

    return await run_in_threadpool(pool().run, func, *args, **kwargs)


def get_queue_and_event() -> Tuple[Queue, Event]:
    global _manager
    if _manager is None:
        _manager = SyncManager()
        _manager.start()
    try:
        return _manager.Queue(), _manager.Event()
    except Exception:
        _manager.shutdown()
        _manager = SyncManager()
        _manager.start()
        return _manager.Queue(), _manager.Event()


@contextmanager
def queue_event_manager(queue: Queue, event: Event):
    try:
        yield queue
    except Exception as e:
        # Put the exception in the queue so that the main process can raise it
        queue.put(e)
    else:
        # Put None in the queue to signal that the iterator is done
        queue.put(None)
