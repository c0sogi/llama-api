from pathlib import Path
import sys
import pytest


sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from llama_api.utils.concurrency import pool  # noqa: E402


@pytest.fixture(scope="session")
def app():
    from llama_api.server.app_settings import (
        create_app_llama_cpp,
    )

    try:
        yield create_app_llama_cpp()
    finally:
        pool().shutdown(wait=False)


@pytest.fixture(scope="session")
def ppool():
    from llama_api.utils.process_pool import (
        ProcessPool,
    )

    with ProcessPool(max_workers=2) as pool:
        for wix in range(pool.max_workers):
            pool.worker_at_wix(wix)

        yield pool
