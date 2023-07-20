from os import getcwd
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


class RelativeImport:
    def __init__(self, path: str):
        self.import_path = Path(path)

    def __enter__(self):
        sys.path.insert(0, str(self.import_path))

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path.remove(str(self.import_path))


@contextmanager
def suppress_import_error():
    try:
        yield
    except ImportError as e:
        logger.warning(f"ImportError: {e}")


def resolve_model_path_to_posix(
    model_path: str, default_relative_directory: Optional[str] = None
) -> str:
    """Resolve a model path to a POSIX path, relative to the BASE_DIR."""
    cwd = getcwd()
    path = Path(model_path)
    parent_directory: Path = (
        Path(cwd) / Path(default_relative_directory)
        if default_relative_directory is not None
        else Path(cwd)
        if Path.cwd() == path.parent.resolve()
        else path.parent.resolve()
    )
    filename: str = path.name
    return (parent_directory / filename).as_posix()
