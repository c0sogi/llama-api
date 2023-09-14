import shutil
import subprocess
import sys
from logging import Logger, getLogger
from os import environ
from pathlib import Path
from typing import List, Optional, Union

from ..shared.config import MainCliArgs
from .dependency import install_package, run_command
from .system_utils import get_cuda_version

# You can set the CMAKE_ARGS environment variable to change the cmake args.
# cuBLAS is default to ON if CUDA is installed.
# CPU inference is default if CUDA is not installed.
METAL_ARGS = "-DBUILD_SHARED_LIBS=ON -DLLAMA_METAL=ON"
CUBLAS_ARGS = "-DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON"
CPU_ARGS = "-DBUILD_SHARED_LIBS=ON"
OPENBLAS_ARGS = (
    "-DBUILD_SHARED_LIBS=ON -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
)

if sys.platform == "darwin":
    CMAKE_ARGS: str = METAL_ARGS
elif get_cuda_version() is None:
    CMAKE_ARGS: str = CPU_ARGS
else:
    CMAKE_ARGS: str = CUBLAS_ARGS

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
        "--recurse-submodules",
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


def _git_clone_if_not_exists() -> None:
    # Clone the git repos if they don't exist
    for clone_path, clone_command in GIT_CLONES.items():
        if not clone_path.exists() or not any(clone_path.iterdir()):
            cwd = clone_path.parent
            cwd.mkdir(exist_ok=True)
            subprocess.run(clone_command, cwd=cwd)


def _get_libs() -> List[str]:
    # Determine the libs based on the platform
    if "linux" in sys.platform:
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


def _copy_make_libs_to_target(make_dir: Path, target_dir: Path) -> None:
    # Copy the built libs to the target folder
    for lib_name in _get_libs():
        lib = make_dir / lib_name
        if lib.exists():
            print(f"~~~ Found shared library: {lib}")
            shutil.copy(lib, target_dir)
        else:
            print(f"~~~ Library {lib_name} not found")


def _copy_cmake_libs_to_target(cmake_dir: Path, target_dir: Path) -> None:
    # Copy the built libs to the target folder
    for lib_name in _get_libs():
        lib = cmake_dir / "build" / "bin" / "Release" / lib_name
        if lib.exists():
            print(f"~~~ Found shared library: {lib}")
            shutil.copy(lib, target_dir)
        else:
            print(f"~~~ Library {lib_name} not found")


def _get_cmake_args(cmake_args: Union[str, List[str]]) -> List[str]:
    if isinstance(cmake_args, str):
        cmake_args = cmake_args.split(" ")
    if "-DBUILD_SHARED_LIBS=ON" not in cmake_args:
        cmake_args.append("-DBUILD_SHARED_LIBS=ON")
    return cmake_args


def _cmake_args_to_make_args(cmake_args: List[str]) -> List[str]:
    # initialize an empty list to store the converted parts
    result: List[str] = []
    # loop through each part
    for cmake_arg in cmake_args:
        # capitalize all letters
        cmake_arg = cmake_arg.upper()

        # skip the `BUILD_SHARED_LIBS` flag
        if "BUILD_SHARED_LIBS" in cmake_arg:
            continue

        # replace `ON` with `1` and `OFF` with `0`
        cmake_arg = cmake_arg.replace("=ON", "=1").replace("=OFF", "=0")

        # remove the `-D` flag
        if cmake_arg.startswith("-D"):
            cmake_arg = cmake_arg[2:]

        # append the converted part to the result list
        result.append(cmake_arg)
    return result


def _make(make_dir: Path, make_args: List[str], target_dir: Path) -> None:
    # Run make to build the shared lib

    # Build the shared lib
    run_command(
        ["make", "clean"],
        action="clean",
        name="llama.cpp shared lib",
        cwd=make_dir,
    )
    for lib in _get_libs():
        run_command(
            ["make", *make_args, lib],
            action="build",
            name="llama.cpp shared lib",
            cwd=make_dir,
        )

    # Copy the built libs to the target folder
    _copy_make_libs_to_target(make_dir=make_dir, target_dir=target_dir)


def _cmake(cmake_dir: Path, cmake_args: List[str], target_dir: Path) -> None:
    # Run cmake to build the shared lib
    build_dir = cmake_dir / "build"
    if build_dir.exists():
        # If the build folder exists, delete it
        shutil.rmtree(build_dir)

    # Create the build folder
    build_dir.mkdir(exist_ok=True)

    # Check if cmake is installed
    result = run_command(
        ["cmake"], action="check", name="cmake", verbose=False
    )
    if result is None or result.returncode != 0:
        # If cmake is not installed, try to install it
        install_package("cmake", force=True)

    # Build the shared lib
    run_command(
        ["cmake", *cmake_args, ".."],
        action="configur",
        name="llama.cpp shared lib",
        cwd=build_dir,
    )
    run_command(
        ["cmake", "--build", ".", "--config", "Release"],
        action="build",
        name="llama.cpp shared lib",
        cwd=build_dir,
    )

    # Copy the built libs to the target folder
    _copy_cmake_libs_to_target(cmake_dir=cmake_dir, target_dir=target_dir)


def build_shared_lib(
    logger: Optional[Logger] = None, force_cuda: bool = False
) -> None:
    """Build the shared library for llama.cpp"""
    global CMAKE_ARGS
    if force_cuda or bool(
        environ.get("FORCE_CUDA", MainCliArgs.force_cuda.value)
    ):
        assert get_cuda_version() is not None, "CUDA is not available"
        CMAKE_ARGS = CUBLAS_ARGS

    if logger is None:
        logger = getLogger(__name__)
        logger.setLevel("INFO")

    # Git clone llama-cpp-python and llama.cpp
    _git_clone_if_not_exists()

    # Build the libs if they don't exist or if `force_cmake` is True
    if bool(environ.get("FORCE_CMAKE", False)) or not any(
        lib_path.exists() for lib_path in _get_lib_paths(MODULE_PATH)
    ):
        # Build the libs
        # Try to build the lib with cmake
        cmake_dir = VENDOR_PATH
        cmake_args_str = environ.get("CMAKE_ARGS", CMAKE_ARGS)
        if sys.platform == "win32":
            _cmake(
                cmake_dir=cmake_dir,
                cmake_args=_get_cmake_args(cmake_args_str),
                target_dir=MODULE_PATH,
            )
        else:
            _make(
                make_dir=cmake_dir,
                make_args=_cmake_args_to_make_args(
                    _get_cmake_args(cmake_args_str)
                ),
                target_dir=MODULE_PATH,
            )
        return
