from pathlib import Path
from typing import List, Tuple


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
