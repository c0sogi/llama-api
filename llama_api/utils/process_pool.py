from itertools import islice
from os import kill
import pickle
import queue
from signal import SIGINT
import sys
from concurrent.futures import Future
from functools import partial
from multiprocessing import Process, Queue, cpu_count
from threading import Thread
from time import sleep
from types import TracebackType
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
    Union,
)


class _WrappedWorkerException(Exception):  # type: ignore
    def __init__(
        self,
        exception_str: str,
        exception_cls: Optional[str] = None,
        traceback: Optional[TracebackType] = None,
    ):
        # don't pickle problematic exception classes
        self.exception = JobFailedException(exception_str, exception_cls)
        self.traceback = None


try:
    from tblib import pickling_support  # type: ignore

    class __WrappedWorkerException(Exception):  # noqa: F811
        # We need this since tracebacks aren't pickled
        # by default and therefore lost"""

        def __init__(
            self,
            exception_str: str,
            exception_cls: Optional[str] = None,
            traceback: Optional[TracebackType] = None,
        ):
            # don't pickle problematic exception classes
            self.exception = JobFailedException(exception_str, exception_cls)
            if traceback is None:
                self.traceback: Optional[TracebackType] = sys.exc_info()[2]
            else:
                self.traceback = traceback

    _WrappedWorkerException = pickling_support.install(  # noqa: F811
        __WrappedWorkerException,  # type: ignore
    )
except Exception:
    pass

assert _WrappedWorkerException is not None

SLEEP_TICK: float = (
    0.001  # Duration in seconds used to sleep when waiting for results
)
T = TypeVar("T")
P = ParamSpec("P")
Job = Tuple[Callable[[], Any], Optional[int], Future]


def _get_chunks(
    *iterables: Iterable[Any], chunksize: int
) -> Iterable[Tuple[Any, ...]]:
    """Iterates over zip()ed iterables in chunks."""

    it = zip(*iterables)
    while True:
        chunk = tuple(islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _chunked_fn(fn: Callable[..., T], *args: Tuple[Any, ...]) -> List[T]:
    """Runs a function with the given arguments
    and returns the list of results."""

    return [fn(*arg) for arg in args]


def _process_chunk(
    fn: Callable[..., T], chunk: Iterable[Tuple[Any, ...]]
) -> List[partial[List[T]]]:
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [partial(_chunked_fn, fn, *args) for args in chunk]  # type: ignore


def _create_new_worker(
    initializer: Optional[Callable[..., Any]] = None, initargs: Any = None
) -> "_WorkerHandler":
    # This runs in the main process.
    # We need to create a new process here, because
    # the old one might have died and we can't restart it.
    return _WorkerHandler(initializer=initializer, initargs=initargs)


def _worker_job_loop(
    recv_q: "Queue[bytes]",
    send_q: "Queue[bytes]",
    initializer: Optional[Callable[..., Any]] = None,
    initargs: Any = None,
) -> None:
    # This runs in the worker process, constantly waiting for jobs.

    if initializer is not None:
        # We're running the initializer
        initializer(*initargs)

    while True:
        # We're using pickle to serialize the function
        partialed_func = pickle.loads(send_q.get(block=True))
        try:
            # Try to run the function
            result = partialed_func()
            error = None
        except MemoryError:  # py 3.8 consistent error
            raise WorkerDiedException(
                "Process encountered MemoryError while running job.",
                "MemoryError",
            )
        except Exception as e:
            # If it fails, we need to send the exception back
            error = _WrappedWorkerException(str(e), e.__class__.__name__)
            result = None
        try:
            # We're using pickle to serialize the result
            recv_q.put(pickle.dumps((result, error)))
        except Exception as e:
            # If it fails, we need to send the exception back
            error = _WrappedWorkerException(str(e), e.__class__.__name__)
            recv_q.put(pickle.dumps((None, error)))


class WorkerDiedException(Exception):
    """Raised when getting the result of a job
    where the process died while executing it for any reason."""

    def __init__(self, message: str, code: Optional[Union[str, int]] = None):
        self.code = code
        self.message = message

    def __reduce__(
        self,
    ) -> Tuple[Type[Exception], Tuple[str, Optional[Union[str, int]]]]:
        return (WorkerDiedException, (self.message, self.code))


class JobFailedException(Exception):
    """Raised when a job fails with a normal exception."""

    def __init__(
        self, message: str, original_exception_type: Optional[str] = None
    ):
        self.original_exception_type = original_exception_type
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: "
            f"{self.original_exception_type}({self.message})"
        )

    def __reduce__(self) -> Tuple[Type[Exception], Tuple[str, Optional[str]]]:
        return (
            JobFailedException,
            (self.message, self.original_exception_type),
        )


class ProcessPoolShutDownException(Exception):
    """Raised when submitting jobs to a process pool
    that has been .join()ed or .terminate()d"""


class _WorkerHandler:
    # Runs in the main process

    def __init__(
        self,
        initializer: Optional[Callable[..., Any]] = None,
        initargs: Optional[Any] = None,
    ):
        self.busy_with_future = None  # type: Optional[Future[Any]]
        self.send_q = Queue()  # type: Queue[bytes]
        self.recv_q = Queue()  # type: Queue[bytes]
        self.process = Process(
            target=_worker_job_loop,
            args=(
                self.recv_q,
                self.send_q,
                initializer,
                initargs,
            ),
            daemon=True,
        )
        self.process.start()

    def send(self, job: Job) -> None:
        partialed_func, _, future = job

        # We're keeping track of the future so we can
        # send the result back to it later
        self.busy_with_future = future
        try:
            # We're sending the job to the worker process.
            self.send_q.put(pickle.dumps(partialed_func))
        except Exception as error:
            # If it fails, we need to send the exception back
            self.recv_q.put(
                pickle.dumps(
                    (
                        None,
                        _WrappedWorkerException(
                            str(error),
                            error.__class__.__name__,
                        ),
                    )
                )
            )

    def result(self) -> Optional[Tuple[Any, Exception]]:
        # We're checking if the job is done
        if not self.busy_with_future:
            # This should never happen
            return None
        try:
            # We're waiting for the result to come back
            ret, err = pickle.loads(self.recv_q.get(block=False))
            if err:
                # We're unwrapping the exception
                unwrapped_err = err.exception
                unwrapped_err.__traceback__ = err.traceback
                err = unwrapped_err
            return ret, err
        except queue.Empty:
            if not self.process.is_alive():
                # The process died while running the job
                raise WorkerDiedException(
                    f"{self.process.name} terminated unexpectedly with "
                    f"exit code {self.process.exitcode} while running job.",
                    self.process.exitcode,
                )
            # The job is still running
            return None


class ProcessPool:
    def __init__(
        self,
        max_workers: int = cpu_count(),
        initializer: Optional[Callable[..., Any]] = None,
        initargs: Any = None,
    ):
        """Manages dispatching jobs to processes, checking results,
          sending them to futures and restarting if they die.

        Args:
            worker_class (Class):
                type that will receive the jobs
                in it's `run` method, one instance will be created per process,
                which should initialize itself fully.
            pool_size (int): number of worker processes to use.
        """
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs

        self.shutting_down = False
        self.terminated = False

        self._pool: List[Optional[_WorkerHandler]] = [
            None for _ in range(max_workers)
        ]
        self._job_queue = queue.Queue()  # type: queue.Queue[Optional[Job]]
        self._job_loop = Thread(target=self._job_manager_thread, daemon=True)
        self._job_loop.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False

    def worker_at_wix(self, wix: int) -> _WorkerHandler:
        worker = self._pool[wix]
        if worker is None:
            worker = _create_new_worker(
                initializer=self.initializer, initargs=self.initargs
            )
            self._pool[wix] = worker
        return worker

    @property
    def active_workers(self) -> List[_WorkerHandler]:
        return [worker for worker in self._pool if worker is not None]

    @property
    def is_available(self) -> bool:
        return not (self.terminated or self.shutting_down)

    def join(self) -> None:
        """Waits for jobs to finish and shuts down the pool."""
        self.shutting_down = True
        if self.terminated:
            raise ProcessPoolShutDownException(
                "Can not join a WorkerPool that has been terminated"
            )

        # We're waiting for all jobs to finish
        while not self._job_queue.empty() or any(
            worker.busy_with_future for worker in self.active_workers
        ):
            sleep(SLEEP_TICK)

        # We need to gracefully shut down the workers
        for worker in self.active_workers:
            pid = worker.process.pid
            if pid:
                kill(pid, SIGINT)

        # We're waiting for the workers to shut down
        for worker in self.active_workers:
            worker.process.join()

        # Send sentinel to stop job loop
        self._job_queue.put(None)
        # We're waiting for the scheduler thread to shut down
        self._job_loop.join()
        self.terminated = True
        self.shutting_down = False

    def terminate(self) -> None:
        """Terminates all sub-processes and stops the pool immediately."""
        self.shutting_down = True

        # We're terminating the workers
        for worker in self.active_workers:
            worker.process.terminate()

        # Send sentinel to stop job loop
        self._job_queue.put(None)
        self.terminated = True
        self.shutting_down = False

    def kill(self) -> None:
        """Kills all sub-processes and stops the pool immediately."""
        self.shutting_down = True

        # We're killing the workers
        for worker in self.active_workers:
            worker.process.kill()

        # Send sentinel to stop job loop
        self._job_queue.put(None)
        self.terminated = True
        self.shutting_down = False

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        """Shuts down the pool, waiting for jobs to finish.

        Args:
            wait: If True, will wait for jobs to finish.
                If False, will stop accepting new jobs and return immediately.
            cancel_futures: If True, will cancel all pending futures.
                If False, will wait for pending futures to finish.
        """
        if cancel_futures:
            # We're cancelling all pending futures
            for worker in self.active_workers:
                if worker.busy_with_future:
                    worker.busy_with_future.cancel()
        if wait:
            # Gracefully shut down the pool
            self.join()
        else:
            # Abortively shut down the pool
            self.terminate()

    def _job_manager_thread(self) -> None:
        """Manages dispatching jobs to processes, checking results,
        sending them to futures and restarting if they die"""
        while True:
            busy_procs = []  # type: List[int]
            for wix, worker in enumerate(self._pool):
                if worker is None:
                    continue
                if worker.busy_with_future:
                    # This worker is busy, let's check if it's done
                    try:
                        result = worker.result()
                        if result is None:
                            # still running...
                            busy_procs.append(wix)
                            continue
                        else:
                            # done!
                            result, exc = result
                    except WorkerDiedException as e:
                        # Oh no, the worker died!
                        if not self.terminated:
                            # We're restarting the worker
                            self._pool[wix] = _create_new_worker(
                                initializer=self.initializer,
                                initargs=self.initargs,
                            )
                        result, exc = None, e
                    if exc:
                        # The job failed! Send the exception to the future
                        worker.busy_with_future.set_exception(exc)
                    else:
                        # The job succeeded! Send the result to the future
                        # NOTE: result can be None
                        worker.busy_with_future.set_result(result)
                    worker.busy_with_future = None  # done!

            idle_procs = [
                wix for wix in range(self.max_workers) if wix not in busy_procs
            ]  # type: List[int]
            if not idle_procs:
                # All workers are busy, let's wait for one to become idle
                sleep(SLEEP_TICK)
                continue

            if busy_procs:
                # There are idle workers, but we can't wait for them,
                # because there are still jobs running!
                # We have to check result of busy workers as soon as possible.
                try:
                    # We're getting the next job from the queue
                    job = self._job_queue.get(block=False)
                except queue.Empty:
                    # No jobs in the queue, let's wait for one
                    sleep(SLEEP_TICK)
                    continue
            else:
                # no jobs are running, so we can block
                job = self._job_queue.get(block=True)

            if job is None:
                # We're shutting down
                # because we got None, which is sentinel, from the queue
                return

            # At this point we have an idle worker and a job to run
            wix = job[1]
            if wix is None:
                # We're sending the job to the first idle worker
                self.worker_at_wix(idle_procs[0]).send(job)
            elif wix in idle_procs:
                # We're sending the job to the specified idle worker
                self.worker_at_wix(wix).send(job)
            else:
                # No idle workers with the specified index,
                # so let's put it back in the queue
                self._job_queue.put(job)
                sleep(SLEEP_TICK)

    def submit(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> "Future[T]":
        """Submits job asynchronously, which will eventually call
        the `run` method in worker_class with the arguments given,
        all of which should be picklable.
        """
        if self.terminated or self.shutting_down:
            raise ProcessPoolShutDownException(
                "Worker pool shutting down or terminated, "
                "can not submit new jobs"
            )
        future: "Future[T]" = Future()
        self._job_queue.put((partial(func, *args, **kwargs), None, future))
        return future

    def submit_with_wix(self, func: Callable[[], T], wix: int) -> "Future[T]":
        """Submits job asynchronously, which will eventually call
        the `run` method in worker_class with the arguments given,
        all of which should be picklable.
        """
        if self.terminated or self.shutting_down:
            raise ProcessPoolShutDownException(
                "Worker pool shutting down or terminated, "
                "can not submit new jobs"
            )
        future: "Future[T]" = Future()
        self._job_queue.put((func, wix, future))
        return future

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> Iterable[T]:
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        # Create multiple partial functions so that we can send
        # multiple arguments to map
        chunked_funcs: List[partial[List]] = _process_chunk(
            fn, chunk=_get_chunks(*iterables, chunksize=chunksize)
        )

        # Submit all the partial functions to the pool
        chunked_futures: List[Future[List]] = [
            self.submit(chunked_func) for chunked_func in chunked_funcs
        ]

        # Yield results as they become available.
        return [
            result
            for chunked_future in chunked_futures
            for result in chunked_future.result(timeout=timeout)
        ]

    def run(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Submits job and blocks to wait for result.
        Returns the result or raises any Exception encountered.
          Should typically only be called from a thread.
        """
        return self.submit(func, *args, **kwargs).result()

    def run_with_wix(self, func: Callable[[], T], wix: int) -> T:
        """Submits job and blocks to wait for result.
        Returns the result or raises any Exception encountered.
          Should typically only be called from a thread.
        """
        return self.submit_with_wix(func, wix).result()
