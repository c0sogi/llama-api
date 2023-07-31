import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ..shared.config import Config


class VirtualEnvironment:
    def __init__(self, venv_path: Union[Path, str]) -> None:
        self.venv_path = Path(venv_path).resolve()
        self.env_for_venv = Config.env_for_venv
        self._executable: Optional[Path] = None
        self._env: Optional[Dict[str, str]] = None

    def remove(self) -> int:
        """Remove the venv if it exists.
        If successful, return 0. Otherwise, return 1."""
        try:
            if self.venv_path.exists():
                shutil.rmtree(self.venv_path)
            return 0
        except OSError:
            return 1

    def create(self) -> int:
        """Create a virtual environment.
        If successful, return 0. Otherwise, return non-zero exit code"""
        assert (
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                    "virtualenv",
                ],
                stdout=subprocess.DEVNULL,
            )
            == 0
        ), "Failed to install virtualenv."
        return subprocess.check_call(
            [sys.executable, "-m", "virtualenv", self.venv_path.as_posix()],
            stdout=subprocess.DEVNULL,
        )

    def recreate(self) -> int:
        """Remove and create a virtual environment"""
        self.remove()
        return self.create()

    def get_settings(self) -> Tuple[Path, Dict[str, str]]:
        """Return the path and the environment variables.
        These will be used to run commands in the virtual environment."""

        # Create the virtual environment if it does not exist.
        if not self.venv_path.exists():
            self.create()

        # The name of the Python executable may vary across platforms.
        python_executable = (
            "python.exe" if sys.platform == "win32" else "python"
        )

        # The name of the Python executable and the directory
        # it resides in may vary across platforms.
        if sys.platform == "win32":
            python_executable = "python.exe"
            executable_directory = "Scripts"
        else:
            python_executable = "python"
            executable_directory = "bin"

        venv_python_path = (
            self.venv_path / executable_directory / python_executable
        )

        # Verify if the path is correct.
        if not venv_python_path.exists():
            raise FileNotFoundError(f"{venv_python_path} does not exist.")

        # Create the environment variables.
        # Copy only the environment variables that are needed.
        # This is for security reasons.
        env = {
            "PATH": venv_python_path.parent.as_posix(),
            "VIRTUAL_ENV": venv_python_path.parent.parent.as_posix(),
        }
        for var in self.env_for_venv:
            if var in os.environ:
                env[var] = os.environ[var]

        # Check if the virtual environment is correct.
        check_command = [venv_python_path, "-c", "import sys; sys.executable"]
        exit_code = subprocess.check_call(
            check_command, env=env, stdout=subprocess.DEVNULL
        )
        assert (
            exit_code == 0
        ), "The virtual environment is not configured correctly."

        # Return the path and the environment variables.
        return venv_python_path, env

    def pip(self, *commands: str, stdout: Optional[int] = None) -> int:
        """Run a pip command in the virtual environment.
        Return the exit code."""
        original_env = os.environ.copy()
        executable, env = self.get_settings()
        original_env.update(env)
        return subprocess.check_call(
            [executable.as_posix(), "-m", "pip", *commands],
            env=original_env,
            stdout=stdout,
        )

    def run_script(
        self, script_path: Union[Path, str]
    ) -> subprocess.CompletedProcess[str]:
        """Run a python script in the virtual environment.
        Return the completed process object.
        This contains the returncode, stdout, and stderr."""
        executable, env = self.get_settings()
        return subprocess.run(
            [executable.as_posix(), Path(script_path).as_posix()],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
        )

    @property
    def executable(self) -> Path:
        if self._executable is None:
            self._executable = self.get_settings()[0]
        return self._executable

    @property
    def env(self) -> Dict[str, str]:
        if self._env is None:
            self._env = self.get_settings()[1]
        return self._env
