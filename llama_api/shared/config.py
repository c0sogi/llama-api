from pathlib import Path
from typing import Dict, Tuple, Union


class Config:
    """Configuration for the project"""

    project_root: Path = Path(__file__).parent.parent.parent
    env_for_venv: Tuple[str, ...] = ("SYSTEMROOT", "CUDA_HOME", "CUDA_PATH")
    cuda_version: str = "11.8"
    torch_version: str = "==2.0.1"
    torch_source: str = "https://download.pytorch.org/whl/torch_stable.html"
    tensorflow_version: str = "==2.13.0"
    git_and_disk_paths: Dict[str, Union[Path, str]] = {
        "https://github.com/abetlen/llama-cpp-python": "repositories/llama_cpp",
        "https://github.com/turboderp/exllama": "repositories/exllama",
    }
