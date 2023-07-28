import sys
from pathlib import Path

import pytest

# Insert the root directory into the path, so we can import the package
_root_dir = Path(__file__).parent.parent
sys.path.insert(0, _root_dir.as_posix())


@pytest.fixture(scope="session")
def app():
    from llama_api.server.app_settings import create_app_llama_cpp
    from llama_api.utils.concurrency import pool

    try:
        yield create_app_llama_cpp()
    finally:
        pool().shutdown(wait=False)


@pytest.fixture(scope="session")
def ppool():
    from llama_api.utils.process_pool import ProcessPool

    with ProcessPool(max_workers=2) as pool:
        for wix in range(pool.max_workers):
            pool.worker_at_wix(wix)

        yield pool


@pytest.fixture(scope="session")
def root_dir():
    return _root_dir
