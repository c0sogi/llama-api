import shutil
import subprocess
import sys
from contextlib import contextmanager
from logging import Logger, getLogger
from os import chdir, environ, getcwd
from pathlib import Path
from typing import List, Optional, Union

from ..utils.dependency import install_package, run_command

# You can set the CMAKE_ARGS environment variable to change the cmake args.
# cuBLAS is default to ON,
# but if it fails to build, fall back to the default settings (CPU only)
CMAKE_ARGS: str = "-DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON"

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


def _get_libs() -> List[str]:
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


def _get_lib_paths(base_path: Path) -> List[Path]:
    # Determine the lib paths based on the platform
    return [base_path / lib for lib in _get_libs()]


def _copy_libs_to_target(cmake_dir: Path, target_dir: Path) -> None:
    # Copy the built libs to the target folder
    for lib_name in _get_libs():
        lib = cmake_dir / "build" / "bin" / "Release" / lib_name
        if lib.exists():
            print(f"~~~ Found shared library: {lib}")
            shutil.copy(lib, target_dir)
        else:
            print(f"~~~ Library {lib_name} not found")


def _cmake(
    cmake_dir: Path, cmake_args: Union[str, List[str]], target_dir: Path
) -> None:
    # Run cmake to build the shared lib
    env = environ.copy()
    build_dir = cmake_dir / "build"
    if isinstance(cmake_args, str):
        cmake_args = cmake_args.split(" ")
    if "-DBUILD_SHARED_LIBS=ON" not in cmake_args:
        cmake_args.append("-DBUILD_SHARED_LIBS=ON")
    if build_dir.exists():
        # If the build folder exists, delete it
        shutil.rmtree(build_dir)

    # Create the build folder
    build_dir.mkdir(exist_ok=True)

    # Check if cmake is installed
    if not run_command(
        ["cmake"], action="check", name="cmake", env=env, verbose=False
    ):
        # If cmake is not installed, try to install it
        install_package("cmake", force=True)

    # Build the shared lib
    run_command(
        ["cmake", *cmake_args, ".."],
        action="build",
        name="llama.cpp shared lib",
        cwd=build_dir,
        env=env,
    )
    run_command(
        ["cmake", "--build", ".", "--config", "Release"],
        action="build",
        name="llama.cpp shared lib",
        cwd=build_dir,
        env=env,
    )

    # Copy the built libs to the target folder
    _copy_libs_to_target(cmake_dir=cmake_dir, target_dir=target_dir)


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
        # Build the libs
        # Try to build the lib with cmake
        cmake_dir = VENDOR_PATH
        cmake_args = environ.get("CMAKE_ARGS", CMAKE_ARGS)
        _cmake(
            cmake_dir=cmake_dir, cmake_args=cmake_args, target_dir=MODULE_PATH
        )
        return
