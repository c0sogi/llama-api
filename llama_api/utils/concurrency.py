from asyncio import get_running_loop
from concurrent.futures.process import BrokenProcessPool, ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Event
from typing import Callable, ParamSpec, Tuple, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

pool = ProcessPoolExecutor()
manager = SyncManager()
manager.start()


async def run_sync(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    global pool
    try:
        return await get_running_loop().run_in_executor(
            pool, partial(func, *args, **kwargs)
        )
    except BrokenProcessPool as e:
        pool.shutdown(wait=True)
        pool = ProcessPoolExecutor()
        raise e


async def run_in_processpool(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    return await run_sync(func, *args, **kwargs)


def get_queue_and_event() -> Tuple[Queue, Event]:
    global manager
    try:
        return manager.Queue(), manager.Event()
    except Exception:
        manager.shutdown()
        manager = SyncManager()
        manager.start()
        return manager.Queue(), manager.Event()


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
