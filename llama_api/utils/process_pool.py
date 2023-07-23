import pickle
import queue
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
    from tblib import pickling_support

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
        while not self._job_queue.empty() or any(
            worker.busy_with_future for worker in self.active_workers
        ):
            sleep(SLEEP_TICK)
        self.terminate()  # could be gentler on the children

    def terminate(self) -> None:
        """Terminates all sub-processes and stops the pool immediately."""
        self.terminated = True
        for worker in self.active_workers:
            worker.process.terminate()
        self._job_queue.put(None)  # in case it's blocking

    def kill(self) -> None:
        """Kills all sub-processes and stops the pool immediately."""
        self.terminated = True
        for worker in self.active_workers:
            worker.process.kill()
        self._job_queue.put(None)  # in case it's blocking

    def _job_manager_thread(self) -> None:
        """Manages dispatching jobs to processes, checking results,
        sending them to futures and restarting if they die"""
        while True:
            busy_procs = []  # type: list[int]
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
            ]  # type: list[int]
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
