from importlib import import_module, reload
from os import environ
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union

from orjson import loads

from ..schemas.api import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
)
from ..schemas.models import BaseLLMModel, ExllamaModel, LlamaCppModel
from .logger import ApiLogger

logger = ApiLogger(__name__)


class ModelDefinitions:
    modules: Dict[str, ModuleType] = {}
    last_modified: Dict[str, float] = {}
    no_model_definitions_warned: bool = False

    MODULE_GLOB_PATTERN = "*model*def*.py"
    ENVIRON_KEY_PATTERN = ("model", "def")
    LLAMA_CPP_KEYS = set(
        [
            "llama.cpp",
            "llama_cpp",
            "llama-cpp",
            "llamacpp",
            "llama_cpp_model",
            "llamacppmodel",
            "ggml",
            "gguf",
        ]
    )
    EXLLAMA_KEYS = set(
        [
            "exllama",
            "ex-llama",
            "ex_llama",
            "ex_llama_model",
            "ex-llama-model",
            "exllamamodel",
            "gptq",
        ]
    )

    @classmethod
    def get_llm_model_from_request_body(
        cls,
        body: Union[
            CreateCompletionRequest,
            CreateChatCompletionRequest,
            CreateEmbeddingRequest,
        ],
        refresh_modules: bool = True,
    ) -> BaseLLMModel:
        if refresh_modules or not cls.last_modified:
            cls._refresh_modules()
        model_name, llm_model, is_openai = cls._resolve_model_name(
            body.model, *cls._collect_from_environs()
        )
        if llm_model is None:
            model_name, llm_model, is_openai = cls._resolve_model_name(
                body.model, *cls._collect_from_modules()
            )
        if llm_model is None:
            raise ValueError(f"Model path does not exist: {body.model}")
        body.model = model_name
        body.is_openai = is_openai
        return llm_model

    @classmethod
    def get_llm_model_names(cls) -> List[str]:
        """Get the names of all the LLaMA models"""
        cls._refresh_modules()
        return list(
            {
                **cls._collect_from_modules()[0],
                **cls._collect_from_environs()[0],
            }.keys()
        )

    @classmethod
    def _load_or_reload_module(cls, path: Path) -> None:
        module_name = path.stem
        if module_name == "__init__":
            return

        current_time = path.stat().st_mtime
        if cls._module_is_modified(module_name, current_time):
            try:
                existing_module = cls.modules.get(module_name)
                cls.modules[module_name] = (
                    reload(existing_module)
                    if existing_module
                    else import_module(module_name)
                )
                cls.last_modified[module_name] = current_time
            except Exception as e:
                logger.error(
                    f"Failed to load or reload module {module_name}: {e}"
                )

    @classmethod
    def _module_is_modified(
        cls, module_name: str, current_time: float
    ) -> bool:
        return (
            module_name not in cls.last_modified
            or cls.last_modified[module_name] != current_time
        )

    @classmethod
    def _collect_from_modules(
        cls,
    ) -> Tuple[Dict[str, BaseLLMModel], Dict[str, str]]:
        model_definitions, openai_replacement_models = {}, {}
        for module in cls.modules.values():
            for key, value in module.__dict__.items():
                if isinstance(value, BaseLLMModel):
                    model_definitions[key.lower()] = value
                elif isinstance(value, dict) and "openai" in key.lower():
                    openai_replacement_models.update(
                        {k.lower(): v.lower() for k, v in value.items()}
                    )
        return model_definitions, openai_replacement_models

    @classmethod
    def _collect_from_environs(
        cls,
    ) -> Tuple[Dict[str, BaseLLMModel], Dict[str, str]]:
        model_definitions = openai_replacement_models = None

        for key, value in environ.items():
            key = key.lower()
            if (
                model_definitions is None
                and all(k in key for k in cls.ENVIRON_KEY_PATTERN)
                and value.startswith("{")
                and value.endswith("}")
            ):
                model_definitions = dict(loads(value))
            if (
                openai_replacement_models is None
                and "openai" in key
                and value.startswith("{")
                and value.endswith("}")
            ):
                openai_replacement_models = {
                    k.lower(): v.lower() for k, v in loads(value).items()
                }

        llm_models = {}  # type: Dict[str, BaseLLMModel]
        if model_definitions is not None:
            for key, value in model_definitions.items():
                if isinstance(value, dict) and "type" in value:
                    type = value.pop("type")
                    if type.lower() in cls.LLAMA_CPP_KEYS:
                        llm_models[key] = LlamaCppModel(**value)
                    elif type.lower() in cls.EXLLAMA_KEYS:
                        llm_models[key] = ExllamaModel(**value)
                    else:
                        raise ValueError(
                            f"Unknown model type: {value['type']}"
                        )
        return llm_models, openai_replacement_models or {}

    @classmethod
    def _resolve_model_name(
        cls,
        raw_model_name: str,
        model_definitions: Dict[str, BaseLLMModel],
        openai_replacement_models: Dict[str, str],
    ) -> Tuple[str, Optional[BaseLLMModel], bool]:
        model_name = raw_model_name.lower()
        if model_name in openai_replacement_models:
            return (
                openai_replacement_models[model_name],
                model_definitions.get(openai_replacement_models[model_name]),
                True,
            )
        return model_name, model_definitions.get(model_name), False

    @classmethod
    def _refresh_modules(cls) -> None:
        model_definition_paths = []  # type: List[Path]

        for path in Path(".").glob(cls.MODULE_GLOB_PATTERN):
            if path.stem == "model_definitions":
                model_definition_paths.insert(0, path)
            else:
                model_definition_paths.append(path)

        # Print warning if no model definitions found
        if not model_definition_paths and not cls.no_model_definitions_warned:
            logger.error(
                "No model definition files found. Please make sure "
                "there is at least one file matching "
                f"the pattern {cls.MODULE_GLOB_PATTERN}."
            )
            cls.no_model_definitions_warned = True

        # Load model_definitions.py first and then the rest
        for path in model_definition_paths:
            cls._load_or_reload_module(path)