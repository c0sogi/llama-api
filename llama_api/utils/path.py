import orjson
from pathlib import Path
from re import compile
from typing import List, Literal, Optional


from ..shared.config import Config
from ..utils.huggingface_downloader import (
    Classification,
    HuggingfaceDownloader,
)
from ..utils.logger import ApiLogger


logger = ApiLogger(__name__)


class HuggingfaceResolver(HuggingfaceDownloader):
    """Resolve the local path of a model from Huggingface."""

    def __init__(
        self,
        model_path: str,
        branch: str = "main",
        threads: int = 1,
        base_folder: Optional[str] = None,
        clean: bool = False,
        check: bool = False,
        text_only: bool = False,
        start_from_scratch: bool = False,
    ) -> None:
        super().__init__(
            model_path,
            branch,
            threads,
            base_folder,
            clean,
            check,
            text_only,
            start_from_scratch,
        )

        # Change the base folder
        download_dir = self.model_path
        if "." in download_dir.name:
            # This is not a directory, but a file.
            # We need directory to download the model.
            download_dir = download_dir.parent
        self.base_folder = download_dir

    @property
    def model_type(self) -> Literal["ggml", "gptq"]:
        """Get the model type: ggml or gptq."""
        classifications: List[Classification] = self.hf_info["classifications"]
        if "ggml" in classifications:
            return "ggml"
        elif (
            "safetensors" in classifications
            or "pytorch" in classifications
            or "pt" in classifications
        ):
            return "gptq"
        else:
            raise ValueError(
                "Supported models: [ggml, safetensors, pytorch, pt]"
            )

    @property
    def model_path(self) -> Path:
        """Get the local path when downloading a model from Huggingface."""
        if self.model_type == "ggml":
            # Get the GGML model path (actually, it can be GGUF)
            for file_name in self.preferred_ggml_files:
                path = (
                    Config.project_root
                    / "models"
                    / self.model_type
                    / file_name
                )
                if path.exists():
                    return path.resolve()

            return path  # type: ignore
        else:  # model_type == "gptq"
            # Get the GPTQ model path (actually, it can be pytorch)
            return (
                Config.project_root
                / "models"
                / self.model_type
                / self.proper_folder_name
            ).resolve()

    @property
    def proper_folder_name(self) -> str:
        """Get a folder name with alphanumeric and underscores only."""
        return compile(r"\W").sub("_", self.model).lower()

    @property
    def preferred_ggml_files(self) -> List[str]:
        """Get the preferred GGML file to download.
        Quanitzation preferences are considered."""

        # Get the GGML file names from the Huggingface info
        ggml_file_names = [
            file_name
            for file_name in self.hf_info["file_names"]
            if self.is_ggml(file_name)
        ]
        if not ggml_file_names:
            raise FileNotFoundError("No GGML file found.")

        # Sort the GGML files by the preferences
        # Return the most preferred GGML file, or the first one if none of the
        # preferences are found
        prefs = Config.ggml_quanitzation_preferences_order
        prefs = [pref.lower() for pref in prefs]
        return sorted(
            ggml_file_names,
            key=lambda ggml_file: next(
                (
                    prefs.index(pref)
                    for pref in prefs
                    if pref in ggml_file.lower()
                ),
                len(prefs),
            ),
        )

    def resolve(self) -> str:
        """Resolve the local path of a model from Huggingface."""
        model_path = self.model_path
        if model_path.exists():
            # The model is already downloaded, return the path
            logger.info(f"`{model_path.name}` found in {model_path.parent}")
            return model_path.as_posix()

        # The model is not downloaded, download it
        if self.model_type == "ggml":
            link = next(
                (
                    link
                    for link in self.hf_info["links"]
                    if any(ggml in link for ggml in self.preferred_ggml_files)
                ),
                None,
            )
            assert link is not None, "No GGML file found."
            links = [link]  # Get only the preferred GGML file
        else:  # model_type == "gptq"
            links = self.hf_info["links"]  # Get all the links available
        self.download_model_files(links=links)
        if model_path.exists():
            logger.info(f"`{model_path.name}` found in {model_path.parent}")
            return model_path.as_posix()

        # The model is not downloaded, and the download failed
        raise FileNotFoundError(
            f"`{model_path.name}` not found in {model_path.parent}"
        )


def resolve_model_path_to_posix(
    model_path: str, default_relative_directory: Optional[str] = None
) -> str:
    """Resolve a model path to a POSIX path."""
    with logger.log_any_error("Error resolving model path"):
        path = Path(model_path)
        if path.is_absolute():
            # The path is already absolute
            if path.exists():
                logger.info(f"`{path.name}` found in {path.parent}")
                return path.resolve().as_posix()
            raise FileNotFoundError(
                f"`{path.name}` not found in {path.parent}"
            )

        parent_dir_candidates = [
            Config.project_root / "models",
            Config.project_root / "llama_api",
            Config.project_root,
            Path.cwd(),
        ]

        if default_relative_directory is not None:
            # Add the default relative directory to the list of candidates
            parent_dir_candidates.insert(
                0, Path.cwd() / Path(default_relative_directory)
            )

        # Try to find the model in all possible scenarios
        for parent_dir in parent_dir_candidates:
            if (parent_dir / model_path).exists():
                logger.info(f"`{path.name}` found in {parent_dir}")
                return (parent_dir / model_path).resolve().as_posix()

        if model_path.count("/") != 1:
            raise FileNotFoundError(
                f"`{model_path}` not found in any of the following "
                f"directories: {parent_dir_candidates}"
            )
        # Try to resolve the model path from Huggingface
        return HuggingfaceResolver(model_path).resolve()


def resolve_model_path_to_posix_with_cache(
    model_path: str,
    default_relative_directory: Optional[str] = None,
) -> str:
    """Resolve a model path to a POSIX path, with caching."""
    from filelock import FileLock, Timeout

    cache_file = Path(".temp/model_paths.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(
            cache_file.with_suffix(".lock"), timeout=10
        ):  # Set a timeout if necessary
            # Read the cache
            try:
                with open(cache_file, "r") as f:
                    cache = orjson.loads(f.read())
                    assert isinstance(cache, dict)
            except Exception:
                cache = {}

            resolved = cache.get(model_path)
            if not (isinstance(resolved, str) or resolved is None):
                raise TypeError(
                    f"Invalid cache entry for model path `{model_path}`: "
                    f"{resolved}"
                )
            if not resolved:
                resolved = resolve_model_path_to_posix(
                    model_path, default_relative_directory
                )
                cache[model_path] = resolved

                # Update the cache file
                with logger.log_any_error("Error writing model path cache"):
                    with open(cache_file, "w") as f:
                        f.write(orjson.dumps(cache).decode())
            return resolved
    except (Timeout, TypeError) as e:
        logger.warning(
            "Error acquiring lock for model path cache"
            + str(cache_file.with_suffix(".lock"))
            + f": {e}"
        )
        return resolve_model_path_to_posix(
            model_path, default_relative_directory
        )


def path_resolver(
    model_path: str, default_relative_directory: Optional[str] = None
) -> str:
    """Resolve a model path to a POSIX path, with caching if possible."""
    try:
        return resolve_model_path_to_posix_with_cache(
            model_path, default_relative_directory
        )
    except ImportError:
        return resolve_model_path_to_posix(
            model_path, default_relative_directory
        )
