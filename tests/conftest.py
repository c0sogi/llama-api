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
