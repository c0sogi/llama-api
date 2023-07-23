from os import getcwd
from subprocess import run
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


@contextmanager
def import_repository(git_path: str, disk_path: str):
    if not Path(disk_path).exists():
        run(["git", "clone", git_path, disk_path])
    sys.path.insert(0, str(disk_path))
    yield
    sys.path.remove(str(disk_path))


@contextmanager
def relative_import(path: str):
    sys.path.insert(0, str(path))
    yield
    sys.path.remove(str(path))


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
