import argparse
import json
from dataclasses import dataclass, field
from os import environ
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T", bound=Union[str, int, float, bool])
NArgs = Union[int, Literal["*", "+", "?"]]
DEFAULT_ENVIRON_KEY = "LLAMA_API_ARGS"
DEFAULT_ENVIRON_KEY_PREFIX = "LLAMA_API_"


@dataclass
class CliArg(Generic[T]):
    type: Callable[[Any], T]
    help: str = ""
    short_option: Optional[str] = None
    action: Optional[str] = None
    default: Optional[T] = None
    # The following fields are automatically set
    value: Optional[T] = field(init=False)
    is_positional: bool = field(init=False, default=False)
    is_list: bool = field(init=False, default=False)
    n_args: Optional[NArgs] = field(init=False, default=None)

    def __post_init__(self):
        self.value = self.default


@dataclass
class CliArgList(CliArg[T]):
    n_args: NArgs = 1
    # The following fields are automatically set
    short_option: Optional[str] = field(init=False, default=None)
    default: List[T] = field(init=False, default_factory=list)
    value: List[T] = field(init=False)
    is_positional: bool = field(init=False, default=True)
    is_list: bool = field(init=False, default=True)


class CliArgHelper:
    """Helper class for loading CLI arguments from environment variables
    or a namespace of CLI arguments"""

    __description__: Optional[str] = None

    @classmethod
    def load(
        cls,
        environ_key: str = DEFAULT_ENVIRON_KEY,
        environ_key_prefix: str = DEFAULT_ENVIRON_KEY_PREFIX,
    ) -> None:
        """Load CLI arguments from environment variables and CLI arguments"""
        cls.load_from_namespace(cls.get_parser().parse_args())
        cls.load_from_environ(
            environ_key=environ_key, environ_key_prefix=environ_key_prefix
        )

    @classmethod
    def load_from_namespace(
        cls,
        args: argparse.Namespace,
        environ_key: Optional[str] = DEFAULT_ENVIRON_KEY,
    ) -> None:
        """Load CLI arguments from a namespace,
        and set an environment variable with the CLI arguments as JSON"""
        # Get all defined CLI arguments within the class
        cli_args = {
            cli_key: cli_arg
            for cli_key, cli_arg in cls.iterate_over_cli_args()
        }

        # Parse the CLI arguments and set the value of the CLI argument
        # if it's not None, otherwise keep the default value
        for cli_key, cli_arg in cli_args.items():
            cls.assign_value(
                cli_arg=cli_arg, value=getattr(args, cli_key, None)
            )

        # Set an environment variable with the CLI arguments as JSON,
        # if an environment variable key is provided
        if environ_key is not None:
            environ[environ_key] = json.dumps(
                {
                    cli_key.upper(): cli_arg.value
                    for cli_key, cli_arg in cli_args.items()
                }
            )

    @classmethod
    def load_from_environ(
        cls,
        environ_key: str = DEFAULT_ENVIRON_KEY,
        environ_key_prefix: Optional[str] = DEFAULT_ENVIRON_KEY_PREFIX,
    ) -> None:
        """Load JSON CLI arguments from an environment variable.
        If an environment variable key prefix is provided,
        load CLI arguments from environment variables which start with
        the prefix."""
        json_str = environ.get(environ_key)
        assert (
            json_str is not None
        ), f"Environment variable {environ_key} not found"
        # Get all defined CLI arguments within the class
        cli_args = {
            cli_key: cli_arg
            for cli_key, cli_arg in cls.iterate_over_cli_args()
        }  # type: dict[str, CliArg]

        # Parse the CLI arguments from the JSON string
        # and set the value of the CLI argument if it's not None,
        # otherwise keep the default value
        cli_arg_values = json.loads(json_str)  # type: dict[str, Any]
        for cli_key, value in cli_arg_values.items():
            cli_key = cli_key.lower()
            if cli_key in cli_args:
                cls.assign_value(cli_arg=cli_args[cli_key], value=value)

        # Parse the CLI arguments from environment variables,
        # which start with the prefix
        if environ_key_prefix is None:
            return
        environ_key_prefix = environ_key_prefix.lower()
        prefix_length = len(environ_key_prefix)
        for key, value in environ.items():
            key = key.lower()
            if not key.startswith(environ_key_prefix):
                continue
            key = key[prefix_length:]
            if key not in cli_args:
                continue
            cli_arg = cli_args[key]
            if not isinstance(cli_arg, CliArg):
                continue
            cls.assign_value(cli_arg=cli_arg, value=value)

    @classmethod
    def iterate_over_cli_args(cls) -> Iterable[Tuple[str, CliArg]]:
        """Get all CLI arguments defined in the class,
        including inherited classes. Yields a tuple of
        (attribute name, CliArg)"""
        for _cls in cls.__mro__:
            for attr_name, attr_value in vars(_cls).items():
                if isinstance(attr_value, CliArg):
                    yield attr_name, attr_value

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """Return an argument parser with all CLI arguments"""
        arg_parser = argparse.ArgumentParser(description=cls.__description__)
        for cli_key, cli_arg in cls.iterate_over_cli_args():
            args = []  # type: List[str]
            if cli_arg.is_positional:
                args.append(cli_key.replace("_", "-"))
            else:
                args.append(f"--{cli_key.replace('_', '-')}")
            if cli_arg.short_option:
                args.append(f"-{cli_arg.short_option.replace('_', '-')}")
            kwargs = {}
            if cli_arg.action:
                kwargs["action"] = cli_arg.action
            else:
                kwargs["type"] = cli_arg.type
            if cli_arg.help:
                kwargs["help"] = cli_arg.help
            if cli_arg.n_args is not None:
                kwargs["nargs"] = cli_arg.n_args
            arg_parser.add_argument(*args, **kwargs)
        return arg_parser

    @staticmethod
    def assign_value(
        cli_arg: Union[CliArg[T], CliArgList[T]], value: Any
    ) -> None:
        """Assign a value to a CLI argument"""
        if value is None:
            return
        if isinstance(cli_arg, CliArgList):
            cli_arg.value = [cli_arg.type(v) for v in value]
        else:
            cli_arg.value = cli_arg.type(value)
