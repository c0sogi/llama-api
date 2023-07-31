from pathlib import Path
from typing import Optional

from ..shared.config import Config
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


def resolve_model_path_to_posix(
    model_path: str, default_relative_directory: Optional[str] = None
) -> str:
    """Resolve a model path to a POSIX path."""
    path = Path(model_path)
    cwd = Path.cwd()

    # Try to find the model in all possible scenarios
    parent_dir_candidates = [
        cwd,
        Config.project_root,
        Config.project_root / "models",
        Config.project_root / "models" / "ggml",
        Config.project_root / "models" / "gptq",
        Config.project_root / "llama_api",
    ]
    if default_relative_directory is not None:
        parent_dir_candidates.insert(0, cwd / Path(default_relative_directory))
    for parent_dir in parent_dir_candidates:
        if (parent_dir / model_path).exists():
            logger.info(f"`{path.name}` found in {parent_dir}")
            return (parent_dir / model_path).resolve().as_posix()
        logger.warning(f"`{path.name}` not found in {parent_dir}")
    logger.error(f"`{path.name}` not found in any of the above directories")
    return model_path
