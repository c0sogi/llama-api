from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

try:
    from typing_extensions import TypedDict


except ImportError:
    from typing import TypedDict  # When dependencies aren't installed yet


class GitCloneArgs(TypedDict):
    git_path: str
    disk_path: str
    options: Optional[List[str]]


class Config:
    """Configuration for the project"""

    project_root: Path = Path(__file__).parent.parent.parent
    env_for_venv: Tuple[str, ...] = ("SYSTEMROOT", "CUDA_HOME", "CUDA_PATH")
    cuda_version: str = "11.8"
    torch_version: str = "==2.0.1"
    torch_source: str = "https://download.pytorch.org/whl/torch_stable.html"
    tensorflow_version: str = "==2.13.0"
    ggml_quanitzation_preferences_order: List[str] = [
        "q4_K_M",
        "q4_K_S",
        "q4_1",
        "q4_0",
        "q5_K_S",
        "q5_1",
        "q5_0",
        "q3_K_L",
        "q3_K_M",
        "q3_K_S",
        "q2_K",
        "q6_K",
        "q8_0",
    ]
    repositories: Dict[Literal["exllama", "llama_cpp"], GitCloneArgs] = {
        "exllama": GitCloneArgs(
            git_path="https://github.com/turboderp/exllama",
            disk_path="repositories/exllama",
            options=["--recurse-submodules"],
        ),
        "llama_cpp": GitCloneArgs(
            git_path="https://github.com/abetlen/llama-cpp-python",
            disk_path="repositories/llama_cpp",
            options=None,
        ),
    }
