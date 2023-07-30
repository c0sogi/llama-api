from concurrent.futures import Future
from contextlib import contextmanager
from functools import partial
from os import getpid
from time import sleep, time
from typing import Tuple

from llama_api.utils.process_pool import ProcessPool
from tests.conftest import TestLlamaAPI


@contextmanager
def process_pool(max_workers: int):
    with ProcessPool(max_workers=max_workers) as executor:
        alive_workers = [False] * max_workers
        while not all(alive_workers):
            for wix in range(executor.max_workers):
                # This will run in the worker
                # at the specified worker-index(wix).
                # We're just checking if the worker is alive.
                alive_workers[wix] = executor.worker_at_wix(wix).is_alive
            print("- Waiting for workers to start...", alive_workers)
            sleep(0.25)  # Wait for the pool to start
        print("- Workers started.")
        yield executor


def simple_job(sleep_time: float) -> Tuple[float, float]:
    """A simple job that sleeps for a given time
    and returns the start and end times."""
    start_time = time()
    print("> Starting at:", start_time, "PID:", getpid())
    sleep(sleep_time)
    end_time = time()
    print("> Ending at", end_time, "PID:", getpid())
    return start_time, end_time


class TestProcessPool(TestLlamaAPI):
    """Test that the process pool works as expected."""

    def test_process_pool(self) -> None:
        """Test the basic functionality of the process pool."""
        # We're recording the start time
        with process_pool(max_workers=2) as executor:
            # Submitting two jobs which will sleep for 1 second each
            f1: Future = executor.submit(simple_job, 1)
            f2: Future = executor.submit(simple_job, 1)
            print("Submitted jobs at", time())

            # Waiting for both jobs to complete
            _, e1 = f1.result()  # This will block until f1 is done
            s2, _ = f2.result()  # This will block until f2 is done

            # Assert that the second job started before the first job ended
            self.assertLess(s2, e1)

    def test_process_pool_with_wix(self) -> None:
        """Test the worker-index-based scheduling functionality
        of the process pool."""
        # We're recording the start time

        with process_pool(max_workers=2) as executor:
            # Submitting two jobs which will sleep for 1 second each
            f1: Future = executor.submit_with_wix(partial(simple_job, 1), wix=0)
            f2: Future = executor.submit_with_wix(partial(simple_job, 1), wix=0)
            print("Submitted jobs at", time())

            # Waiting for both jobs to complete
            _, e1 = f1.result()  # This will block until f1 is done
            s2, _ = f2.result()  # This will block until f2 is done

            # Assert that the second job started before the first job ended
            self.assertGreater(s2, e1)
