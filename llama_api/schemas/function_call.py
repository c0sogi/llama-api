"""Helper classes for wrapping functions in OpenAI's API"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from ..schemas.api import FunctionParameter, FunctionSchema

# The types that can be used in JSON
JsonTypes = Union[int, float, str, bool, dict, list, None]

ParamType = TypeVar("ParamType", bound=JsonTypes)
ReturnType = TypeVar("ReturnType")


@dataclass
class FunctionCallParameter(Generic[ParamType]):
    """A class for wrapping function parameters in OpenAI's API"""

    name: str
    type: Type[ParamType]
    description: Optional[str] = None
    enum: Optional[List[ParamType]] = None

    def to_dict(self) -> Dict[str, FunctionParameter]:
        """Returns a dictionary representation of the parameter"""
        parameter_property: FunctionParameter = {
            "type": self._get_json_type(self.type)
        }  # type: ignore
        if self.description:
            parameter_property["description"] = self.description
        if self.enum:
            parameter_property["enum"] = self.enum  # type: ignore
        return {self.name: parameter_property}

    @staticmethod
    def _get_json_type(python_type: Type[JsonTypes]) -> str:
        """Returns the JSON type for a given python type"""
        if python_type is int:
            return "integer"
        elif python_type is float:
            return "number"
        elif python_type is str:
            return "string"
        elif python_type is bool:
            return "boolean"
        elif python_type is dict:
            return "object"
        elif python_type is list:
            return "array"
        elif python_type is type(None) or python_type is None:
            return "null"
        else:
            raise ValueError(
                f"Invalid type {python_type} for JSON. "
                f"Permitted types are {JsonTypes}"
            )


@dataclass
class FunctionCall:
    """A class for wrapping functions in OpenAI's API"""

    name: str
    parameters: List[FunctionCallParameter[Any]]
    description: Optional[str] = None
    required: Optional[List[str]] = None

    def to_dict(self) -> FunctionSchema:
        """Returns a dictionary representation of the function"""
        function_property: FunctionSchema = FunctionSchema(
            name=self.name,
            parameters={
                "type": "object",
                "properties": {
                    param.name: param.to_dict()[param.name]
                    for param in self.parameters
                },
                "required": [
                    param.name
                    for param in self.parameters
                    if param.name in (self.required or [])
                ],
            },
        )
        if self.description:
            function_property["description"] = self.description
        return function_property
