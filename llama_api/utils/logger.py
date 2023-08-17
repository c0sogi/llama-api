"""Logger module for the API"""
# flake8: noqa
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, Optional, Union

from .colorama import Fore, Style


@dataclass
class LoggingConfig:
    logger_level: int = logging.DEBUG
    console_log_level: int = logging.INFO
    file_log_level: Optional[int] = logging.DEBUG
    file_log_name: Optional[str] = "./logs/debug.log"
    logging_format: str = "[%(asctime)s] %(name)s:%(levelname)s - %(message)s"
    color: bool = True


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define color codes
        self.colors = {
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
        }

    def format(self, record: logging.LogRecord):
        # Apply color to the entire log message
        prefix = self.colors.get(record.levelname, Fore.WHITE + Style.BRIGHT)
        message = super().format(record)
        return f"{prefix}{message}{Style.RESET_ALL}"


class ApiLogger(logging.Logger):
    _instances: Dict[str, "ApiLogger"] = {}

    def __new__(
        cls, name: str, logging_config: LoggingConfig = LoggingConfig()
    ) -> "ApiLogger":
        """Singleton pattern for ApiLogger class"""
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(
        self, name: str, logging_config: LoggingConfig = LoggingConfig()
    ) -> None:
        super().__init__(name=name, level=logging_config.logger_level)
        formatter = (
            ColoredFormatter(logging_config.logging_format)
            if logging_config.color
            else logging.Formatter(logging_config.logging_format)
        )

        console = logging.StreamHandler()
        console.setLevel(logging_config.console_log_level)
        console.setFormatter(formatter)

        if (
            logging_config.file_log_name is not None
            and logging_config.file_log_level is not None
        ):
            Path(logging_config.file_log_name).parent.mkdir(
                parents=True, exist_ok=True
            )
            file_handler = logging.FileHandler(
                filename=logging_config.file_log_name,
                mode="a",
                encoding="utf-8",
            )
            file_handler.setLevel(logging_config.file_log_level)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

        self.addHandler(console)

    @classmethod
    def cinfo(cls, msg: object, *args, **kwargs) -> None:
        if cls.__name__ not in cls._instances:
            cls(cls.__name__)
        super(
            ApiLogger,
            cls._instances[cls.__name__],
        ).info(msg, *args, **kwargs)

    @classmethod
    def cdebug(cls, msg: object, *args, **kwargs) -> None:
        if cls.__name__ not in cls._instances:
            cls(cls.__name__)
        super(ApiLogger, cls._instances[cls.__name__]).debug(
            msg, *args, **kwargs
        )

    @classmethod
    def cwarning(cls, msg: object, *args, **kwargs) -> None:
        if cls.__name__ not in cls._instances:
            cls(cls.__name__)
        super(ApiLogger, cls._instances[cls.__name__]).warning(
            msg, *args, **kwargs
        )

    @classmethod
    def cerror(cls, msg: object, *args, **kwargs) -> None:
        if cls.__name__ not in cls._instances:
            cls(cls.__name__)
        super(ApiLogger, cls._instances[cls.__name__]).error(
            msg, *args, **kwargs
        )

    @classmethod
    def cexception(cls, msg: object, *args, **kwargs) -> None:
        if cls.__name__ not in cls._instances:
            cls(cls.__name__)
        super(ApiLogger, cls._instances[cls.__name__]).exception(
            msg, *args, **kwargs
        )

    @classmethod
    def ccritical(cls, msg: object, *args, **kwargs) -> None:
        if cls.__name__ not in cls._instances:
            cls(cls.__name__)
        super(ApiLogger, cls._instances[cls.__name__]).critical(
            msg, *args, **kwargs
        )

    @contextmanager
    def log_any_error(
        self,
        msg: Optional[object] = None,
        level: int = logging.ERROR,
        exc_info: Optional[Union[bool, Exception]] = True,
        suppress_exception: bool = False,
        on_error: Optional[Callable[[Exception], None]] = None,
        *args,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        A context manager to automatically log exceptions that occur within its context.

        Args:
            msg (Optional[object], default=None): An optional message to be prepended to the exception message in the log.
            level (int, default=logging.ERROR): The logging level at which the exception should be logged. Default is ERROR.
            exc_info (logging._ExcInfoType, default=True): If set to True, exception information will be added to the log. Otherwise, only the exception message will be logged.
            suppress_exception (bool, default=False): If True, the exception will be suppressed (not re-raised). If False, the exception will be re-raised after logging.
            on_error (Optional[Callable[[Exception], None]], default=None): A callback function that will be invoked with the exception as its argument if one occurs.
            *args: Variable length argument list passed to the logging function.
            **kwargs: Arbitrary keyword arguments passed to the logging function.

        Usage:
            with logger.log_any_error(msg="An error occurred", level=logging.WARNING, on_error=my_callback_function):
                potentially_faulty_function()

        Notes:
            - If a custom message is provided using the 'msg' parameter, it will be prepended to the actual exception message in the log.
            - If 'on_error' is provided, it will be executed with the caught exception as its argument. This can be used for custom handling or notification mechanisms.
        """

        try:
            yield
        except Exception as e:
            self.log(
                level,
                f"{msg}: {e}" if msg else e,
                *args,
                **kwargs,
                exc_info=exc_info,
            )
            if on_error:
                on_error(e)
            if not suppress_exception:
                raise
