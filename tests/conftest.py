from pathlib import Path
import sys
import pytest

sys.path.insert(0, Path(__file__).parent.parent.as_posix())


@pytest.fixture(scope="session")
def app():
    from llama_api.server.app_settings import (
        create_app_llama_cpp,
    )

    return create_app_llama_cpp()


@pytest.fixture(scope="session")
def ppool():
    from llama_api.utils.process_pool import (
        ProcessPool,
    )

    pool = ProcessPool(max_workers=2)
    for wix in range(pool.max_workers):
        pool.worker_at_wix(wix)

    yield pool
    pool.kill()
