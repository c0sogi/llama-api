import argparse
from dataclasses import dataclass, field
import json
from os import environ
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

try:
    from typing_extensions import TypedDict


except ImportError:
    from typing import TypedDict  # When dependencies aren't installed yet


T = TypeVar("T", bound=Union[str, int, float, bool])


@dataclass
class CliArg(Generic[T]):
    type: Callable[[Any], T]
    help: str = ""
    short_option: Optional[str] = None
    action: Optional[str] = None
    default: Optional[T] = None
    value: Optional[T] = field(init=False)  # ensure it's set in __post_init__

    def __post_init__(self):
        self.value = self.default


class CliArgHelper:
    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for cli_key, cli_arg in cls.iterate_over_cli_args():
            args = []  # type: List[str]
            if cli_arg.short_option:
                args.append(f"-{cli_arg.short_option.replace('_', '-')}")
            args.append(f"--{cli_key.replace('_', '-')}")
            kwargs = {}
            if cli_arg.help:
                kwargs["help"] = cli_arg.help
            if cli_arg.default is not None:
                kwargs["default"] = cli_arg.default
            if cli_arg.action:
                kwargs["action"] = cli_arg.action
            else:
                kwargs["type"] = cli_arg.type
            parser.add_argument(*args, **kwargs)
        return parser

    @classmethod
    def load(cls) -> None:
        cls.load_from_namespace(cls.get_parser().parse_args())

    @classmethod
    def load_from_namespace(
        cls, args: argparse.Namespace, environ_key: str = "LLAMA_API_ARGS"
    ) -> None:
        cli_args = {
            cli_key: cli_arg
            for cli_key, cli_arg in cls.iterate_over_cli_args()
        }
        for cli_key, cli_arg in cli_args.items():
            cli_arg_value = getattr(args, cli_key, None)
            if cli_arg_value is not None:
                cli_arg.value = cli_arg.type(cli_arg_value)
        environ[environ_key] = json.dumps(
            {
                cli_key.upper(): cli_arg.value
                for cli_key, cli_arg in cli_args.items()
            }
        )

    @classmethod
    def load_from_environ(cls, environ_key: str = "LLAMA_API_ARGS") -> None:
        json_str = environ.get(environ_key)
        assert (
            json_str is not None
        ), f"Environment variable {environ_key} not found"
        cli_args = {
            cli_key: cli_arg
            for cli_key, cli_arg in cls.iterate_over_cli_args()
        }  # type: Dict[str, CliArg]
        cli_arg_values = json.loads(json_str)  # type: Dict[str, Any]
        for cli_key, cli_value in cli_arg_values.items():
            cli_key = cli_key.lower()
            if cli_key in cli_args and cli_value is not None:
                cli_arg = cli_args[cli_key]
                cli_arg.value = cli_arg.type(cli_value)

    @classmethod
    def iterate_over_cli_args(cls) -> Iterable[Tuple[str, CliArg]]:
        for _cls in cls.__mro__:
            for attr_name, attr_value in vars(_cls).items():
                if isinstance(attr_value, CliArg):
                    yield attr_name, attr_value


class AppSettingsCliArgs(CliArgHelper):
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
    upgrade_pkgs: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="u",
        help="Upgrade all packages before running the server",
    )


class MainCliArgs(AppSettingsCliArgs):
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
    api_key: CliArg[str] = CliArg(
        type=str,
        short_option="k",
        help="API key to use for the server",
        default=None,
    )
    xformers: CliArg[bool] = CliArg(
        type=bool,
        action="store_true",
        short_option="x",
        help="Apply xformers' memory-efficient optimizations",
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
            options=["recurse-submodules"],
        ),
        "llama_cpp": GitCloneArgs(
            git_path="https://github.com/abetlen/llama-cpp-python",
            disk_path="repositories/llama_cpp",
            options=None,
        ),
    }
