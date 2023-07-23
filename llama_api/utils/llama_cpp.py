import subprocess
import sys
from logging import Logger, getLogger
from pathlib import Path
from tempfile import mkstemp
from typing import Optional

LIB_BASE_NAME: str = "llama"
REPOSITORY_FOLDER: str = "repositories"
PROJECT_GIT_URL: str = "https://github.com/abetlen/llama-cpp-python.git"
PROJECT_NAME: str = "llama_cpp"
MODULE_NAME: str = "llama_cpp"
VENDOR_GIT_URL: str = "https://github.com/ggerganov/llama.cpp.git"
VENDOR_NAME: str = "llama.cpp"
CMAKE_CONFIG: str = "Release"
SCRIPT_FILE_NAME: str = "build-llama-cpp"
CMAKE_ARGS: dict[str, str] = {
    "cublas": "-DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON",
    "default": "-DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=ON",
}

REPOSITORY_PATH: Path = Path(REPOSITORY_FOLDER).resolve()
PROJECT_PATH: Path = REPOSITORY_PATH / Path(PROJECT_NAME)
MODULE_PATH: Path = PROJECT_PATH / Path(MODULE_NAME)
VENDOR_PATH: Path = PROJECT_PATH / Path("vendor") / Path(VENDOR_NAME)
BUILD_OUTPUT_PATH: Path = (
    VENDOR_PATH / Path("build") / Path("bin") / Path(CMAKE_CONFIG)
)

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

WINDOWS_BUILD_SCRIPT = r"""
rmdir /s /q build
mkdir build
cd build
cmake {cmake_args} ..
cmake --build . --config {cmake_config}
cd ../../../../..
"""

UNIX_BUILD_SCRIPT = r"""#!/bin/bash
rm -rf build
mkdir build
cd build
cmake {cmake_args} ..
cmake --build . --config {cmake_config}
cd ../../../../..
"""


def _git_clone() -> None:
    for clone_path, clone_command in GIT_CLONES.items():
        if not clone_path.exists() or not any(clone_path.iterdir()):
            cwd = clone_path.parent
            cwd.mkdir(exist_ok=True)
            subprocess.run(clone_command, cwd=cwd)


def _get_lib_paths(base_path: Path) -> list[Path]:
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        return [
            base_path / f"lib{LIB_BASE_NAME}.so",
        ]
    elif sys.platform == "darwin":
        return [
            base_path / f"lib{LIB_BASE_NAME}.so",
            base_path / f"lib{LIB_BASE_NAME}.dylib",
        ]
    elif sys.platform == "win32":
        return [
            base_path / f"{LIB_BASE_NAME}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")


def _get_script_content(cmake_args: str) -> str:
    cmd = "copy" if sys.platform == "win32" else "cp"
    content = (
        WINDOWS_BUILD_SCRIPT if sys.platform == "win32" else UNIX_BUILD_SCRIPT
    ) + "".join(
        [
            f'\n{cmd} "{lib}" "{MODULE_PATH}"'
            for lib in _get_lib_paths(BUILD_OUTPUT_PATH)
        ]
    )
    return content.format(
        vendor_path=VENDOR_PATH,
        cmake_args=cmake_args,
        cmake_config=CMAKE_CONFIG,
    )


def _cmake_args_to_make_args(cmake_args: str) -> list[str]:
    # initialize an empty list to store the converted parts
    result: list[str] = []
    # loop through each part
    for cmake_arg in cmake_args.split():
        # remove the `-D` flag
        cmake_arg = cmake_arg.removeprefix("-D")
        # replace '=Off' with '=0' and '=ON' with '=1'
        cmake_arg = cmake_arg.replace("=Off", "=0").replace("=ON", "=1")
        # append the converted part to the result list
        result.append(cmake_arg)
    return result


def build_shared_lib(logger: Optional[Logger] = None) -> None:
    """
    Ensure that the llama.cpp DLL exists.
    You need cmake and Visual Studio 2019 to build llama.cpp.
    You can download cmake here: https://cmake.org/download/
    """

    if logger is None:
        logger = getLogger(__name__)
        logger.setLevel("INFO")

    _git_clone()
    if not any(lib_path.exists() for lib_path in _get_lib_paths(MODULE_PATH)):
        logger.critical("🦙 llama.cpp lib not found, building it...")
        files: list[str] = []

        ext: str = "bat" if sys.platform == "win32" else "sh"
        for cmake_args in CMAKE_ARGS.values():
            if sys.platform == "darwin" and "cublas" in cmake_args.lower():
                # Skip cublas on macOS
                logger.warning(
                    "🦙 cuBLAS is not supported on macOS, skipping this..."
                )
                continue

            file_descriptor, file_name = mkstemp(suffix=f".{ext}", text=True)
            with open(file_descriptor, "w") as file:
                file.write(_get_script_content(cmake_args))
            if sys.platform != "win32":
                subprocess.run(["chmod", "755", file_name])
            files.append(file_name)

        VENDOR_PATH.mkdir(exist_ok=True)
        for file in files:
            try:
                # Try to build with cublas.
                logger.critical(f"🦙 Trying to build llama.cpp lib: {file}")
                subprocess.run([file], cwd=VENDOR_PATH, check=True)
                logger.critical("🦙 llama.cpp lib successfully built!")
                return
            except subprocess.CalledProcessError:
                logger.critical("🦙 Could not build llama.cpp lib!")
        raise RuntimeError("🦙 Could not build llama.cpp lib!")


if __name__ == "__main__":
    build_shared_lib()
