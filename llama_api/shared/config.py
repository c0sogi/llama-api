import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from ..utils.cli import CliArg, CliArgHelper, CliArgList

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class GitCloneArgs(TypedDict):
    git_path: str
    disk_path: str
    options: Optional[List[str]]


class AppSettingsCliArgs(CliArgHelper):
    __description__ = (
        "Settings for the server, and installation of dependencies"
    )

    install_pkgs: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="i",
        help="Install all required packages before running the server",
    )
    force_cuda: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="c",
        help="Force CUDA version of pytorch to be used "
        "when installing pytorch. e.g. torch==2.0.1+cu118",
    )
    skip_torch_install: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="-no-torch",
        help="Skip installing pytorch, if `install-pkgs` is set",
    )
    skip_tf_install: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="-no-tf",
        help="Skip installing tensorflow, if `install-pkgs` is set",
    )
    skip_compile: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="-no-compile",
        help="Skip compiling the shared library of LLaMA C++ code",
    )
    no_cache_dir: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="-no-cache",
        help="Disable caching of pip installs, if `install-pkgs` is set",
    )
    upgrade: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="u",
        help="Upgrade all packages and repositories before running the server",
    )


class MainCliArgs(AppSettingsCliArgs):
    __description__ = (
        "Main CLI arguments for the server, including app settings"
    )
    port: CliArg[int] = CliArg(
        type=int,
        short_option="p",
        help="Port to run the server on; default is 8000",
        default=8000,
    )
    max_workers: CliArg[int] = CliArg(
        type=int,
        short_option="w",
        help="Maximum number of process workers to run; default is 1",
        default=1,
    )
    max_semaphores: CliArg[int] = CliArg(
        type=int,
        short_option="s",
        help="Maximum number of process semaphores to permit; default is 1",
        default=1,
    )
    max_tokens_limit: CliArg[int] = CliArg(
        type=int,
        short_option="l",
        help=(
            "Set the maximum number of tokens to `max_tokens`. "
            "This is needed to limit the number of tokens generated."
            "Default is None, which means no limit."
        ),
        default=None,
    )
    api_key: CliArg[str] = CliArg(
        type=str,
        short_option="k",
        help="API key to use for the server",
        default=None,
    )
    no_embed: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        help="Disable embeddings endpoint",
    )
    tunnel: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="t",
        help="Tunnel the server through cloudflared",
    )
    # xformers: CliArg[bool] = CliArg(
    #     type=bool,
    #     action="store_true",
    #     short_option="x",
    #     help="Apply xformers' memory-efficient optimizations",
    # )


class ModelDownloaderCliArgs(CliArgHelper):
    __description__ = "Download models from HuggingFace"
    model: CliArgList[str] = CliArgList(
        type=str,
        n_args="+",
        help="The model you'd like to download. e.g. facebook/opt-1.3b",
    )
    branch: CliArg[str] = CliArg(
        type=str,
        default="main",
        help="Name of the Git branch to download from.",
    )
    threads: CliArg[int] = CliArg(
        type=int,
        default=1,
        help="Number of files to download simultaneously.",
    )
    text_only: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        help="Only download text files (txt/json).",
    )
    output: CliArg[str] = CliArg(
        type=str,
        default=None,
        help="The folder where the model should be saved.",
    )
    clean: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        help="Does not resume the previous download.",
    )
    check: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        help="Validates the checksums of model files.",
    )
    start_from_scratch: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        help="Starts the download from scratch.",
    )


class LogParserCliArgs(CliArgHelper):
    __description__ = "Process chat and debug logs."

    min_output_length: CliArg[int] = CliArg(
        type=int, default=30, help="Minimum length for the output."
    )
    chat_log_file_path: CliArg[str] = CliArg(
        type=str,
        default="logs/chat.log",
        help="Path to the chat log file.",
    )
    debug_log_file_path: CliArg[str] = CliArg(
        type=str,
        default="logs/debug.log",
        help="Path to the debug log file.",
    )
    ignore_messages_less_than: CliArg[int] = CliArg(
        type=int, default=2, help="Ignore messages shorter than this length."
    )
    output_path: CliArg[str] = CliArg(
        type=str,
        default="./logs/chat.csv",
        help="Path to save the extracted chats as CSV.",
    )


class BuildSharedLibCliArgs(CliArgHelper):
    __description__ = "Process chat and debug logs."

    backend: CliArgList[str] = CliArgList(
        type=lambda s: str(s).lower(),
        choices=["cuda", "cpu", "metal", "cublas", "openblas"],
        help="The backend to use for building the shared library.",
    )


class Config:
    """Configuration for the project"""

    project_root: Path = Path(__file__).parent.parent.parent
    env_for_venv: Tuple[str, ...] = ("SYSTEMROOT", "CUDA_HOME", "CUDA_PATH")
    cuda_version: str = "11.8"
    torch_version: str = "==2.0.1"
    torch_source: str = "https://download.pytorch.org/whl/torch_stable.html"
    tensorflow_version: str = "==2.13.0"
    trained_tokens: int = 4096
    ggml_quanitzation_preferences_order: List[str] = [
        "q4_k_m",
        "q4_k_s",
        "q4_1",
        "q4_0",
        "q5_k_s",
        "q5_1",
        "q5_0",
        "q3_k_l",
        "q3_k_m",
        "q3_k_s",
        "q2_k",
        "q6_k",
        "q8_0",
    ]
    repositories: Dict[Literal["exllama", "llama_cpp"], GitCloneArgs] = {
        "exllama": GitCloneArgs(
            git_path="https://github.com/turboderp/exllama",
            disk_path="repositories/exllama",
            options=None,
        ),
        "llama_cpp": GitCloneArgs(
            git_path="https://github.com/abetlen/llama-cpp-python",
            disk_path="repositories/llama_cpp",
            options=["--recurse-submodules"],
        ),
    }
