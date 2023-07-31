import os
import sys
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from platform import mac_ver
from re import compile
from subprocess import PIPE, check_call, run
from tempfile import mkstemp
from typing import List, Optional, Union
from urllib.request import urlopen

from ..shared.config import Config
from ..utils.logger import ApiLogger
from ..utils.system import get_cuda_version

logger = ApiLogger(__name__)


def run_command(
    command: List[str],
    action: str,
    name: str,
    try_emoji: str = "📦",
    success_emoji: str = "✅",
    failure_emoji: str = "❌",
    verbose: bool = True,
    **kwargs,
) -> bool:
    """Run a command and log the result.
    Return True if the command was successful, False otherwise."""
    try:
        if verbose:
            logger.info(
                f"{try_emoji} {action}ing {name} with command: "
                f"{' '.join(command)}"
            )
        result = run(command, text=True, stdout=PIPE, stderr=PIPE, **kwargs)
        if result.returncode != 0:
            if verbose:
                logger.error(
                    f"{failure_emoji} Failed to {action} {name}:\n"
                    + (result.stdout or "")
                    + (result.stderr or "")
                )
            return False
        else:
            if verbose:
                logger.info(f"{success_emoji} Successfully {name} {action}ed.")
            return True
    except Exception as e:
        if verbose:
            logger.error(
                f"{failure_emoji} Failed to {action} {name}:\n" + str(e)
            )
        return False


def is_package_available(package: str) -> bool:
    return True if find_spec(package) else False


def git_clone(git_path: str, disk_path: Union[Path, str]) -> Optional[bool]:
    """Clone a git repository to a disk path."""
    if not Path(disk_path).exists():
        return run_command(
            ["git", "clone", git_path, str(disk_path)],
            action="clone",
            name=f"{git_path} to {disk_path}",
            try_emoji="📥",
        )
    return None


def get_mac_major_version_string():
    # platform.mac_ver() returns a tuple ('10.16', ('', '', ''), 'x86_64')
    # Split the version string on '.' and take the first two components
    major = mac_ver()[0].split(".")[0]

    # Join the components with '_' and prepend 'macosx_'
    return "macosx_" + major


def get_installed_packages() -> List[str]:
    """Return a list of installed packages"""
    return [
        package.split("==")[0]
        for package in run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stdout=PIPE,
            stderr=PIPE,
        )
        .stdout.strip()
        .split("\n")
    ]


def get_poetry_executable() -> Path:
    """Construct the path to the poetry executable
    within the virtual environment.
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
    elif check_if_torch_version_available(
        version=f'cu{cuda_version.replace(".", "")}', source=source
    ):
        return cuda_version
    else:
        return fallback_cuda_version


def check_if_torch_version_available(
    version: str = Config.torch_version,
    source: str = Config.torch_source,
) -> bool:
    """Helper function that checks if the version of torch is available"""
    try:
        # Determine the version of python, CUDA, and platform
        canonical_version = compile(r"([0-9\.]+)").search(version)
        if not canonical_version:
            return False
        package_ver = canonical_version.group().encode()
        python_ver = (
            f"cp{sys.version_info.major}{sys.version_info.minor}"
        ).encode()
        if "win32" in sys.platform:
            platform = "win_amd64".encode()
        elif "linux" in sys.platform:
            platform = "linux_x86_64".encode()
        elif "darwin" in sys.platform:
            platform = get_mac_major_version_string().encode()
        else:
            return False

        # Check if the CUDA version of torch is available
        for line in urlopen(source).read().splitlines():
            if package_ver in line and python_ver in line and platform in line:
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
        include_version:
            Whether to include the version in the parsed requirements.
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


def convert_toml_to_requirements_with_poetry(
    toml_path: Path,
    *args,
    format: str = "requirements.txt",
    output: str = "requirements.txt",
) -> Optional[bool]:
    """Convert dependencies from pyproject.toml to requirements.txt."""
    poetry = str(get_poetry_executable())
    command = [poetry, "export", "-f", format, "--output", output, *args]
    try:
        if toml_path.exists():
            # Convert dependencies from pyproject.toml to requirements.txt
            return run_command(
                command,
                action="convert",
                name="pyproject.toml to requirements.txt",
                try_emoji="🔄",
                cwd=toml_path.parent,
            )
    except Exception:
        return None


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


def install_package(package: str, force: bool = False) -> bool:
    """Install a package with pip."""
    if not force and is_package_available(package):
        return True
    return run_command(
        [sys.executable, "-m", "pip", "install", package],
        action="install",
        name=package,
    )


def install_poetry() -> bool:
    """Install poetry."""
    logger.info("📦 Installing poetry...")
    return run_command(
        [sys.executable, "-m", "pip", "install", "poetry"],
        action="install",
        name="poetry",
    )


def install_pytorch(
    torch_version: str = Config.torch_version,
    cuda_version: Optional[str] = Config.cuda_version,
    source: Optional[str] = Config.torch_source,
    force_cuda: bool = False,
) -> bool:
    """Try to install Pytorch.
    If CUDA is available, install the CUDA version of torch.
    Else, install the CPU version of torch.
    Args:
        torch_version (str): The version of torch.
          Defaults to Config.torch_version.
        cuda_version (str): The version of CUDA.
          Defaults to Config.cuda_version.
        source (Optional[str]): The source to install torch from.
            Defaults to Config.torch_source.
        force_cuda (bool): Whether to force install the CUDA version of torch.
    Returns:
        bool: True if Pytorch is installed successfully, else False."""
    pip_install = [sys.executable, "-m", "pip", "install"]
    fallback_cuda_version = Config.cuda_version
    # If a source is specified, and if CUDA is available,
    # install the CUDA version of torch
    if force_cuda or (source and get_cuda_version()):
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

        if check_if_torch_version_available(
            version=f"{torch_version}+cpu",
            source=source,
        ):
            # install the CPU version of torch if available
            pip_install.append(f"torch{torch_version}+cpu")
        else:
            # else, install the canonical version of torch
            pip_install.append(f"torch{torch_version}")

        # If a source is specified, add it to the pip install command
        pip_install += ["-f", source]
    else:
        # If source is not specified, install the canonical version of torch
        pip_install.append(f"torch{torch_version}")

    # Install torch
    return run_command(pip_install, action="install", name="PyTorch")


def install_tensorflow(
    tensorflow_version: str = Config.tensorflow_version,
    source: Optional[str] = None,
) -> bool:
    """Try to install TensorFlow.

    Args:
        tensorflow_version (str): The version of TensorFlow.
          Defaults to Config.tensorflow_version.
        source (Optional[str]): The source to install TensorFlow from.
          If not specified, TensorFlow will be installed from PyPi.
    Returns:
        bool: True if TensorFlow is installed successfully, else False.
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
    return run_command(pip_install, action="install", name="TensorFlow")


def install_all_dependencies(
    project_paths: Optional[Union[List[Path], List[str]]] = None,
) -> Optional[bool]:
    """Install every dependencies."""
    pip_install = [sys.executable, "-m", "pip", "install", "-r"]
    result = True
    for project_path in project_paths or []:
        project_path = Path(project_path).resolve()
        logger.info(f"📦 Installing dependencies for {project_path}...")
        convert_toml_to_requirements_with_poetry(
            project_path / "pyproject.toml", "--without-hashes"
        )
        requirements_path = project_path / "requirements.txt"
        if not requirements_path.exists():
            logger.warning(
                f"⚠️ Could not find requirements.txt in {project_path}."
            )
            continue
        result &= run_command(
            pip_install + [requirements_path.as_posix()],
            action="install",
            name="dependencies",
        )
    return result


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
