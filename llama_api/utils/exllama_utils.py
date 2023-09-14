from pathlib import Path
from re import compile
from typing import List, Union

from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


def get_model_path(model_folder_path: Path) -> Union[str, List[str]]:
    # Find the model checkpoint file and remove numbers from file names
    remove_numbers_pattern = compile(r"\d+")
    grouped_by_base_name = {}  # type: dict[str, list[Path]]
    for model_file in (
        list(model_folder_path.glob("*.safetensors"))
        or list(model_folder_path.glob("*.pt"))
        or list(model_folder_path.glob("*.bin"))
    ):
        grouped_by_base_name.setdefault(
            remove_numbers_pattern.sub("", model_file.name), []
        ).append(model_file)

    # Choose the group with maximum files
    # having the same base name after removing numbers
    max_group = max(grouped_by_base_name.values(), key=len, default=[])
    if len(max_group) == 1:
        # If there is only one file in the group,
        # use the largest file among all groups with a single file
        return max(
            (
                group[0]
                for group in grouped_by_base_name.values()
                if len(group) == 1
            ),
            key=lambda x: x.stat().st_size,
        ).as_posix()
    elif len(max_group) > 1:
        # If there are multiple files in the group,
        # use all of them as the model path
        return [model_file.as_posix() for model_file in max_group]
    else:
        # If there is no file in the group, raise an error
        raise FileNotFoundError(
            f"No model has been found in {model_folder_path}."
        )
