from importlib.util import find_spec
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from re import compile
from subprocess import DEVNULL, call, check_call, run
from tempfile import mkstemp
from typing import List, Optional, Union
from urllib.request import urlopen

from ..shared.config import Config
from ..utils.logger import ApiLogger
from ..utils.system import get_cuda_version

logger = ApiLogger(__name__)


def is_package_available(package: str) -> bool:
    return True if find_spec(package) else False


def git_clone(git_path: str, disk_path: Union[Path, str]) -> None:
    """Clone a git repository to a disk path."""
    if not Path(disk_path).exists():
        # Clone the repository
        check_call(["git", "clone", git_path, disk_path])


def get_poetry_executable() -> Path:
    """Construct the path to the poetry executable within the virtual environment
    Check the operating system to determine the correct location"""
    if os.name == "nt":  # Windows
        return Path(sys.prefix) / "Scripts" / "poetry.exe"
    else:  # Linux or Mac
        return Path(sys.prefix) / "bin" / "poetry"


def get_proper_torch_cuda_version(
    cuda_version: str,
    source: str = Config.torch_source,
    fallback_cuda_version: str = "11.8",
) -> str:
    """Helper function that returns the proper CUDA version of torch."""
    if cuda_version == fallback_cuda_version:
        return fallback_cuda_version
    elif check_if_torch_cuda_version_available(
        cuda_version=cuda_version, source=source
    ):
        return cuda_version
    else:
        return fallback_cuda_version


def check_if_torch_cuda_version_available(
    cuda_version: str = Config.torch_version,
    source: str = Config.torch_source,
) -> bool:
    """Helper function that checks if the CUDA version of torch is available"""
    try:
        # Determine the version of python, CUDA, and platform
        cuda_version = (f'cu{cuda_version.replace(".", "")}').encode()  # type: ignore
        python_version = (
            f"cp{sys.version_info.major}{sys.version_info.minor}"
        ).encode()
        platform = ("win" if sys.platform == "win32" else "linux").encode()

        # Check if the CUDA version of torch is available
        for line in urlopen(source).read().splitlines():
            if (
                cuda_version in line
                and python_version in line
                and platform in line
            ):
                return True
        return False
    except Exception:
        return False


def parse_requirements(
    requirements: str,
    excludes: Optional[List[str]] = None,
    include_version: bool = True,
) -> List[str]:
    """Parse requirements from a string.
    Args:
        requirements: The string of requirements.txt to parse.
        excludes: A list of packages to exclude.
        include_version: Whether to include the version in the parsed requirements.
    Returns:
        A list of parsed requirements.
    """
    # Define the regular expression pattern
    pattern = compile(r"([a-zA-Z0-9_\-\+]+)(==|>=|<=|~=|>|<|!=|===)([0-9\.]+)")

    # Use finditer to get all matches in the string
    return [
        match.group() if include_version else match.group(1)
        for match in pattern.finditer(requirements)
        if not excludes or match.group(1) not in excludes
    ]


def convert_toml_to_requirements_with_poetry(toml_path: Path) -> None:
    """Convert dependencies from pyproject.toml to requirements.txt."""
    try:
        if toml_path.exists():
            # Convert dependencies from pyproject.toml to requirements.txt
            call(
                [
                    get_poetry_executable(),
                    "export",
                    "-f",
                    "requirements.txt",
                    "--output",
                    "requirements.txt",
                    "--without-hashes",
                ],
                cwd=toml_path.parent,
            )
    except Exception:
        pass


# def parse_dependencies(
#     parse_dir: Path,
#     excludes: Optional[List[str]] = None,
#     include_version: bool = True,
# ) -> List[str]:
#     """Parse dependencies from pyproject.toml or requirements.txt.
#     Then, convert the dependencies to a list of strings."""
#     try:
#         if (parse_dir / "pyproject.toml").exists():
#             # Convert dependencies from pyproject.toml to requirements.txt
#             call(
#                 [
#                     get_poetry_executable(),
#                     "export",
#                     "-f",
#                     "requirements.txt",
#                     "--output",
#                     "requirements.txt",
#                     "--without-hashes",
#                 ],
#                 cwd=parse_dir,
#             )
#     except Exception:
#         pass
#     try:
#         with open(parse_dir / "requirements.txt", "r") as file:
#             return [
#                 dep
#                 for dep in parse_requirements(
#                     file.read(),
#                     excludes=excludes,
#                     include_version=include_version,
#                 )
#             ]
#     except Exception:
#         return []


@contextmanager
def import_repository(git_path: str, disk_path: str):
    """
    Import a repository from git. The repository will be cloned to disk_path.
    The dependencies will be installed from pyproject.toml or requirements.txt.
    """

    # Clone the repository
    git_clone(git_path=git_path, disk_path=disk_path)

    # Add the repository to the path so that it can be imported
    sys.path.insert(0, str(disk_path))
    yield
    sys.path.remove(str(disk_path))


def install_poetry():
    """Install poetry."""
    logger.info("📦 Installing poetry...")
    check_call(
        [sys.executable, "-m", "pip", "install", "poetry"], stdout=DEVNULL
    )
    logger.info("✅ Poetry installed.")


def install_torch(
    torch_version: str = Config.torch_version,
    cuda_version: Optional[str] = Config.cuda_version,
    source: Optional[
        str
    ] = Config.torch_source,  # https://download.pytorch.org/whl/torch_stable.html
) -> bool:
    """Try to install Pytorch.
    If CUDA is available, install the CUDA version of torch.
    Else, install the CPU version of torch.
    Args:
        torch_version (str): The version of torch. Defaults to Config.torch_version.
        cuda_version (str): The version of CUDA. Defaults to Config.cuda_version.
        source (Optional[str]): The source to install torch from.
            Defaults to Config.torch_source.
    Returns:
        bool: True if CUDA is installed, False otherwise."""
    pip_install = [sys.executable, "-m", "pip", "install"]
    fallback_cuda_version = Config.cuda_version
    # If a source is specified, and if CUDA is available,
    # install the CUDA version of torch
    if source and get_cuda_version():
        # Check if the CUDA version of torch is available.
        # If not, fallback to `Config.cuda_version`.
        cuda_version = (
            get_proper_torch_cuda_version(
                cuda_version=cuda_version,
                fallback_cuda_version=fallback_cuda_version,
            )
            if cuda_version is not None
            else fallback_cuda_version
        )
        pip_install.append(
            f'torch{torch_version}+cu{cuda_version.replace(".", "")}'
        )
        # If a source is specified, add it to the pip install command
        pip_install += ["-f", source]
    elif source:
        # If a source is specified, but CUDA is not available,
        # install the CPU version of torch
        pip_install.append(f"torch{torch_version}+cpu")
        # If a source is specified, add it to the pip install command
        pip_install += ["-f", source]
    else:
        # If source is not specified, install the canonical version of torch
        pip_install.append(f"torch{torch_version}")

    # Install torch
    logger.info(f"📦 Installing PyTorch with command: {' '.join(pip_install)}")
    check_call(pip_install)
    logger.info("✅ PyTorch installed.")
    return cuda_version is not None


def install_tensorflow(
    tensorflow_version: str = Config.tensorflow_version,
    source: Optional[str] = None,
) -> None:
    """Try to install TensorFlow.

    Args:
        tensorflow_version (str): The version of TensorFlow.
          Defaults to Config.tensorflow_version.
        source (Optional[str]): The source to install TensorFlow from.
          If not specified, TensorFlow will be installed from PyPi.
    """
    pip_install = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"tensorflow{tensorflow_version}",
    ]

    # If a source is specified, add it to the pip install command
    if source:
        pip_install += ["-f", source]

    # Install TensorFlow
    logger.info(
        f"📦 Installing TensorFlow with command: {' '.join(pip_install)}"
    )
    check_call(pip_install)
    logger.info("✅ TensorFlow installed.")


def install_all_dependencies(
    project_paths: Optional[Union[List[Path], List[str]]] = None,
) -> None:
    """Install every dependencies."""
    pip_install = [sys.executable, "-m", "pip", "install", "-r"]
    for project_path in project_paths or []:
        project_path = Path(project_path).resolve()
        logger.info(f"📦 Installing dependencies for {project_path}...")
        convert_toml_to_requirements_with_poetry(
            project_path / "pyproject.toml"
        )
        requirements_path = project_path / "requirements.txt"
        if not requirements_path.exists():
            logger.warning(
                f"⚠️ Could not find requirements.txt in {project_path}."
            )
            continue
        result = run(
            pip_install + [requirements_path],
            text=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )

        if result.stderr or result.returncode != 0:
            logger.error(
                "❌ Error installing dependencies: "
                + (result.stdout or "")
                + (result.stderr or "")
            )
        else:
            logger.info("✅ Dependencies installed!")


def remove_all_dependencies():
    """Remove all dependencies.
    To be used when cleaning up the environment."""
    logger.critical("Removing all dependencies...")
    # Create a temporary file
    fd, temp_path = mkstemp()

    try:
        # Step 1: List out all installed packages
        with open(temp_path, "w") as temp_file:
            check_call(
                [sys.executable, "-m", "pip", "freeze"], stdout=temp_file
            )

        # Step 2: Uninstall all packages listed in the temp file
        with open(temp_path, "r") as temp_file:
            packages = [line.strip() for line in temp_file if "-e" not in line]

        for package in packages:
            # The "--yes" option automatically confirms the uninstallation
            check_call(
                [sys.executable, "-m", "pip", "uninstall", "--yes", package]
            )
    finally:
        # Close the file descriptor and remove the temporary file
        os.close(fd)
        os.remove(temp_path)
        logger.critical("All dependencies removed.")
