"""A module for lazy imports of modules.
The modules are only imported when they are used. This is useful because
importing those modules costs expensive resources."""


from functools import wraps
from typing import Callable, Set, TypeVar, Union

from .logger import ApiLogger

T = TypeVar("T")
logger = ApiLogger(__name__)
logged_modules: Set[str] = set()


def try_import(module_name: str):
    """A decorator for attempting to import a module.
    Returns the function's result if the module is imported successfully.
    Otherwise, returns the exception.
    If the module has been imported before, logger will be suppressed.
    Otherwise, logger will be used to log the import attempt and result."""

    def decorator(
        func: Callable[..., T]
    ) -> Callable[..., Union[T, Exception]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Exception]:
            # Only log and attempt import
            # if the module hasn't been loaded successfully yet
            if module_name not in logged_modules:
                try:
                    logger.info(f"🦙 Attempting to import {module_name}...")
                    result = func(*args, **kwargs)
                    logger.info(f"🦙 Successfully imported {module_name}!")
                    return result
                except Exception as e:
                    logger.exception(f"🦙 Error importing {module_name}: {e}")
                    return e
                finally:
                    # Add the module to the `logged_modules` set
                    # to prevent further logs
                    logged_modules.add(module_name)
            else:
                # If the module has been loaded before,
                #  just return the function's result
                return func(*args, **kwargs)

        return wrapper

    return decorator


class LazyImports:
    """A class for lazy imports of modules."""

    @property
    @try_import("llama_cpp")
    def LlamaCppCompletionGenerator(self):
        from ..modules.llama_cpp import LlamaCppCompletionGenerator

        return LlamaCppCompletionGenerator

    @property
    @try_import("exllama")
    def ExllamaCompletionGenerator(self):
        from ..modules.exllama import ExllamaCompletionGenerator

        return ExllamaCompletionGenerator

    @property
    @try_import("exllamav2")
    def ExllamaV2CompletionGenerator(self):
        from ..modules.exllamav2 import ExllamaV2CompletionGenerator

        return ExllamaV2CompletionGenerator

    @property
    @try_import("transformer")
    def TransformerEmbeddingGenerator(self):
        from ..modules.transformer import TransformerEmbeddingGenerator

        return TransformerEmbeddingGenerator

    @property
    @try_import("sentence_encoder")
    def SentenceEncoderEmbeddingGenerator(self):
        from ..modules.sentence_encoder import (
            SentenceEncoderEmbeddingGenerator,
        )

        return SentenceEncoderEmbeddingGenerator
