from asyncio import gather
from concurrent.futures import Future
from functools import partial
from os import getpid
from time import sleep, time

from llama_api.utils.concurrency import (
    ProcessPool,  # noqa: E402
    awake_all_pool_workers,
    pool,
    run_in_processpool,
    run_in_processpool_with_wix,
)
from tests.conftest import TestLlamaAPI


def simple_job(sleep_time: float) -> float:
    print("> Input sleep time:", sleep_time, "PID:", getpid())
    sleep(sleep_time)
    print("> Job done at", time(), "PID:", getpid())
    return sleep_time


class TestProcessPool(TestLlamaAPI):
    def test_process_pool(self) -> None:
        # We're recording the start time
        start_time = time()

        # Submitting two jobs which will sleep for 1 second each
        f1: Future = self.ppool.submit(simple_job, 1)
        f2: Future = self.ppool.submit(simple_job, 1)
        print("Submitted jobs at", time())

        # Waiting for both jobs to complete
        result1 = f1.result()  # This will block until f1 is done
        result2 = f2.result()  # This will block until f2 is done
        print("Jobs done:", result1, result2)

        # Recording the end time
        end_time = time()
        print("Elapsed time:", end_time - start_time)

        # Total elapsed time should be little more than 1 second,
        # not 2 seconds, because jobs are expected to run in parallel.
        self.assertLessEqual(
            end_time - start_time, 2.0, f"{end_time - start_time}"
        )
        self.assertTrue(result1 == result2 == 1)

    def test_process_pool_with_wix(self) -> None:
        # We're recording the start time
        start_time = time()

        # Submitting two jobs which will sleep for 1 second each
        f1: Future = self.ppool.submit_with_wix(partial(simple_job, 1), wix=1)
        f2: Future = self.ppool.submit_with_wix(partial(simple_job, 1), wix=1)
        print("Submitted jobs")

        # Waiting for both jobs to complete
        result1 = f1.result()  # This will block until f1 is done
        result2 = f2.result()  # This will block until f2 is done
        print("Jobs done:", result1, result2)

        # Recording the end time
        end_time = time()
        print("Elapsed time:", end_time - start_time)

        # Total elapsed time should be little more than 2 seconds,
        # not 1 second, because jobs are expected to run in serial.
        self.assertGreaterEqual(
            end_time - start_time, 2.0, f"{end_time - start_time}"
        )
        self.assertTrue(result1 == result2 == 1)

    async def test_process_pool_async(self):
        try:
            awake_all_pool_workers()
            sleep(2)  # Give the pool some time to start up

            start_time = time()
            print("Submitting jobs...")
            result_1, result_2 = await gather(
                run_in_processpool(simple_job, 1),
                run_in_processpool(simple_job, 1),  # type: ignore
            )
            end_time = time()
            print("Elapsed time:", end_time - start_time)

            # Total elapsed time should be little more than 1 second,
            # not 2 seconds, because jobs are expected to run in parallel.
            self.assertLessEqual(
                end_time - start_time, 2.0, f"{end_time - start_time}"
            )
            self.assertTrue(result_1 == result_2 == 1)
        finally:
            _pool = pool()
            _pool.join()

    async def test_process_pool_with_wix_async(self):
        try:
            awake_all_pool_workers()
            sleep(2)  # Give the pool some time to start up

            start_time = time()
            print("Submitting jobs...")
            result_1, result_2 = await gather(
                run_in_processpool_with_wix(partial(simple_job, 1), wix=1),
                run_in_processpool_with_wix(partial(simple_job, 1), wix=1),
            )
            end_time = time()
            print("Elapsed time:", end_time - start_time)

            # Total elapsed time should be little more than 2 seconds,
            # not 1 second, because jobs are expected to run in serial.
            self.assertGreaterEqual(
                end_time - start_time, 2.0, f"{end_time - start_time}"
            )
            self.assertTrue(result_1 == result_2 == 1)
        finally:
            _pool = pool()
            _pool.join()

    def test_process_pool_graceful_shutdown(self):
        timeout = None
        with ProcessPool(max_workers=2) as executor:
            f1 = executor.submit(simple_job, 1)
            f2 = executor.submit(simple_job, 1)
            print("Submitted jobs")

            # Wait for the jobs to start
            result1 = f1.result(timeout=timeout)
            result2 = f2.result(timeout=timeout)
            print("Jobs done:", result1, result2)

    def test_process_pool_map(self):
        timeout = 3.0
        with ProcessPool(max_workers=4) as executor:
            results = iter(
                executor.map(
                    simple_job,
                    [1, 1, 1, 1],
                    chunksize=2,
                    timeout=timeout,
                )
            )
            print("Submitted jobs")

            # Wait for the jobs to start
            result1 = next(results)
            result2 = next(results)
            print("Jobs done:", result1, result2)
