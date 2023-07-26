from contextlib import contextmanager
from os import chdir, environ, getcwd
from shutil import copy
import subprocess
import sys
from logging import Logger, getLogger
from pathlib import Path
from typing import Optional

# You can set the CMAKE_ARGS environment variable to change the cmake args.
# cuBLAS is default to ON,
# but if it fails to build, fall back to the default settings (CPU only)
CMAKE_ARGS: str = "-DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON"

LIB_BASE_NAME: str = "llama"
REPOSITORY_FOLDER: str = "repositories"
PROJECT_GIT_URL: str = "https://github.com/abetlen/llama-cpp-python.git"
PROJECT_NAME: str = "llama_cpp"
MODULE_NAME: str = "llama_cpp"
VENDOR_GIT_URL: str = "https://github.com/ggerganov/llama.cpp.git"
VENDOR_NAME: str = "llama.cpp"

REPOSITORY_PATH: Path = Path(REPOSITORY_FOLDER).resolve()
PROJECT_PATH: Path = REPOSITORY_PATH / Path(PROJECT_NAME)
MODULE_PATH: Path = PROJECT_PATH / Path(MODULE_NAME)
VENDOR_PATH: Path = PROJECT_PATH / Path("vendor") / Path(VENDOR_NAME)


GIT_CLONES = {
    PROJECT_PATH: [
        "git",
        "clone",
        # "--recurse-submodules",
        PROJECT_GIT_URL,
        PROJECT_NAME,
    ],
    VENDOR_PATH: [
        "git",
        "clone",
        VENDOR_GIT_URL,
        VENDOR_NAME,
    ],
}


@contextmanager
def _temporary_change_cwd(path):
    # Change the current working directory to `path` and then change it back
    prev_cwd = getcwd()
    chdir(path)
    try:
        yield
    finally:
        chdir(prev_cwd)


def _git_clone() -> None:
    # Clone the git repos if they don't exist
    for clone_path, clone_command in GIT_CLONES.items():
        if not clone_path.exists() or not any(clone_path.iterdir()):
            cwd = clone_path.parent
            cwd.mkdir(exist_ok=True)
            subprocess.run(clone_command, cwd=cwd)


def _get_libs() -> list[str]:
    # Determine the libs based on the platform
    if sys.platform.startswith("linux"):
        return [
            f"lib{LIB_BASE_NAME}.so",
        ]
    elif sys.platform == "darwin":
        return [
            f"lib{LIB_BASE_NAME}.so",
            f"lib{LIB_BASE_NAME}.dylib",
        ]
    elif sys.platform == "win32":
        return [
            f"{LIB_BASE_NAME}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")


def _get_lib_paths(base_path: Path) -> list[Path]:
    # Determine the lib paths based on the platform
    return [base_path / lib for lib in _get_libs()]


def _copy_skbuild_libs_to_target(
    cmake_dir: Path, target_dir: Path
) -> list[Path]:
    # Copy the built libs to the target folder
    source_libs: Optional[list[Path]] = None
    for dir in (cmake_dir / "_skbuild").glob("*"):
        if dir.is_dir():
            print(f"~~~ Checking {dir}")
            source_libs = [
                source_lib
                for source_lib in (dir / "cmake-install" / MODULE_NAME).glob(
                    "*"
                )
                if source_lib.name in _get_libs()
            ]
            if source_libs:
                print(f"~~~ Found {source_libs}")
                break
    assert source_libs is not None, "Could not find build libs"

    for source_lib in source_libs:
        copy(source_lib, target_dir)
    return source_libs


def build_shared_lib(
    logger: Optional[Logger] = None,
    force_cmake: bool = bool(environ.get("FORCE_CMAKE", False)),
) -> None:
    """Build the shared library for llama.cpp"""

    if logger is None:
        logger = getLogger(__name__)
        logger.setLevel("INFO")

    # Git clone llama-cpp-python and llama.cpp
    _git_clone()

    # Build the libs if they don't exist or if `force_cmake` is True
    if force_cmake or not any(
        lib_path.exists() for lib_path in _get_lib_paths(MODULE_PATH)
    ):
        target_dir = MODULE_PATH

        # Build the libs
        with _temporary_change_cwd(PROJECT_PATH):
            # Try to build the lib with cmake
            environ["FORCE_CMAKE"] = "1"
            if environ.get("CMAKE_ARGS") is None:
                environ["CMAKE_ARGS"] = CMAKE_ARGS

            logger.critical(
                f"🦙 Building llama.cpp libs with {environ['CMAKE_ARGS']}"
            )
            subprocess.run([sys.executable, "-m", "pip", "install", "."])

        # Move the built libs to the target folder
        source_libs = _copy_skbuild_libs_to_target(
            cmake_dir=PROJECT_PATH, target_dir=target_dir
        )
        logger.info(f"🦙 llama.cpp libs built with `{environ['CMAKE_ARGS']}`")
        for source_lib in source_libs:
            logger.info(f"~~~ Moved {source_lib.name} to {target_dir}")
        return


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, root_path.as_posix())

    from llama_api.utils.logger import ApiLogger

    build_shared_lib(force_cmake=True, logger=ApiLogger(__name__))
