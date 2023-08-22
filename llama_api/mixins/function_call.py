"""Helper classes for wrapping functions in OpenAI's API"""
# flake8: noqa
import json
from inspect import signature
from re import Pattern, compile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

from typing_extensions import Annotated, get_args, get_origin

from ..schemas.api import (
    ChatCompletion,
    ChatCompletionChunk,
    CreateChatCompletionRequest,
)
from ..schemas.function_call import (
    FunctionCall,
    FunctionCallParameter,
    JsonTypes,
)

# whitespace is constrained to a single space char
# to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE: str = "([ \t\n])?"

PRIMITIVE_RULES: Dict[str, str] = {
    "boolean": '("true" | "false") space',
    "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    "integer": '("-"? ([0-9] | [1-9] [0-9]*)) space',
    "string": r""" "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space """,
    "null": '"null" space',
}

INVALID_RULE_CHARS_RE: "Pattern[str]" = compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE: "Pattern[str]" = compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES: Dict[str, str] = {
    "\r": "\\r",
    "\n": "\\n",
    '"': '\\"',
}

# Type aliases
SchemaType = Literal[
    "boolean", "number", "integer", "string", "null", "object", "array"
]
SchemaKey = Literal[
    "type",
    "oneOf",
    "anyOf",
    "const",
    "enum",
    "properties",
    "items",
    "required",
]


def _get_type_and_optional(t: Type) -> Tuple[Type, bool]:
    """Returns the type and whether it's an Optional type.
    This is useful when Type can be Union and you want to know if it's an Optional type.
    """
    # Optional[str] is equivalent to Union[str, None], so check if it's a Union type.
    if get_origin(t) in (type(Union), Union):
        args = get_args(t)  # type: Tuple[Type, ...]
        # If there's a None type in the Union, it's an Optional type.
        optional = type(None) in args
        # Return the first argument that isn't None.
        first_arg = next(arg for arg in args if arg is not type(None))
        return first_arg, optional
    else:
        # If it's not a Union type, it's not an Optional type.
        return t, False


class FunctionCallMixin:
    """Contains helper functions converting JSON schemas to BNF grammars
    Reference: https://github.com/ggerganov/llama.cpp/pull/1887"""

    _prop_order: Dict[str, int]
    _rules: Dict[str, str]

    def invoke_function_call(
        self, request: CreateChatCompletionRequest
    ) -> ChatCompletion:
        """Invoke the function call while chat completion"""
        raise NotImplementedError(
            "function call is not implemented for this model"
        )

    def invoke_function_call_streaming(
        self, request: CreateChatCompletionRequest
    ) -> Iterator[ChatCompletionChunk]:
        """Invoke the function call while chat completion, streaming the results"""
        raise NotImplementedError(
            "function call is not implemented for this model"
        )

    @classmethod
    def from_json_schema(
        cls,
        schema: Union[Dict[SchemaKey, Any], str],
        prop_order: Optional[Dict[str, int]] = None,
    ) -> str:
        """Parse a JSON schema into a BNF grammar"""
        if isinstance(schema, str):
            schema = json.loads(schema)
            assert isinstance(schema, dict), "schema must be valid JSON"
        self = cls()
        self._prop_order = prop_order or {}
        self._rules = {"space": SPACE_RULE}
        self._visit(schema, "")
        return self._format_grammar()

    @classmethod
    @overload
    def from_function_calls(
        cls,
        function_calls: FunctionCall,
        prop_order: Optional[Dict[str, int]] = None,
    ) -> str:
        ...

    @classmethod
    @overload
    def from_function_calls(
        cls,
        function_calls: Iterable[FunctionCall],
        prop_order: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        ...

    @classmethod
    def from_function_calls(
        cls,
        function_calls: Union[FunctionCall, Iterable[FunctionCall]],
        prop_order: Optional[Dict[str, int]] = None,
    ) -> Union[str, List[str]]:
        """Parse a FunctionCall object into a BNF grammar"""
        if isinstance(function_calls, Iterable):
            return_as_list = True
            function_calls = list(function_calls)
        else:
            return_as_list = False
            function_calls = [function_calls]

        bnfs = []  # type: List[str]
        for function_call in function_calls:
            self = cls()
            self._prop_order = prop_order or {}
            self._rules = {"space": SPACE_RULE}
            parameters = function_call.to_dict().get("parameters")
            assert parameters is not None, "function call must have parameters"
            self._visit(dict(parameters), "")
            bnfs.append(self._format_grammar())
        return bnfs if return_as_list else bnfs[0]

    @classmethod
    @overload
    def from_functions(
        cls,
        functions: Callable,
        prop_order: Optional[Dict[str, int]] = None,
    ) -> str:
        ...

    @classmethod
    @overload
    def from_functions(
        cls,
        functions: Iterable[Callable],
        prop_order: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        ...

    @classmethod
    def from_functions(
        cls,
        functions: Union[Callable, Iterable[Callable]],
        prop_order: Optional[Dict[str, int]] = None,
    ) -> Union[str, List[str]]:
        """Parse a function into a BNF grammar"""
        if isinstance(functions, Iterable):
            return_as_list = True
            functions = list(functions)
        else:
            return_as_list = False
            functions = [functions]

        function_calls = []  # type: List[FunctionCall]
        json_types = get_args(JsonTypes)
        line_break_pattern = compile(r"\n\s*")

        for function in functions:
            function_call_params = []  # type: List[FunctionCallParameter]
            required = []  # type: List[str]
            for name, param in signature(function).parameters.items():
                annotation = param.annotation
                description = ""  # type: str
                enum = []  # type: List[Any]

                if get_origin(annotation) is Annotated:
                    # If the annotation is an Annotated type,
                    # we need to parse the metadata
                    _param_args = get_args(param.annotation)
                    _param_type = _param_args[0]

                    for metadata in _param_args[1:]:
                        if isinstance(metadata, str):
                            # If the metadata is a string, it's the description
                            description += metadata
                        elif isinstance(metadata, Iterable):
                            # If the metadata is an iterable, it's the enum
                            enum.extend(metadata)

                else:
                    _param_type = annotation
                param_type, optional = _get_type_and_optional(_param_type)
                if not optional:
                    required.append(name)
                if param_type not in json_types:
                    continue
                function_call_params.append(
                    FunctionCallParameter(
                        name=name,
                        type=param_type,
                        description=description or None,
                        enum=enum or None,
                    )
                )
                function_calls.append(
                    FunctionCall(
                        name=function.__name__,
                        description=line_break_pattern.sub(
                            " ", function.__doc__
                        )
                        if function.__doc__
                        else None,
                        parameters=function_call_params,
                        required=required or None,
                    )
                )
        return FunctionCallMixin.from_function_calls(
            function_calls if return_as_list else function_calls[0],
            prop_order,
        )

    def _format_literal(self, literal: Any) -> str:
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)) or "",
            json.dumps(literal),
        )
        return f'"{escaped}"'

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub("-", name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f"{esc_name}{i}" in self._rules:
                i += 1
            key = f"{esc_name}{i}"
        self._rules[key] = rule
        return key

    def _visit(self, schema: Dict[SchemaKey, Any], name: str) -> str:
        schema_type: SchemaType = schema[
            "type"
        ]  # The "type" key is always present
        rule_name: str = name or "root"  # root rule is always named "root"

        if "oneOf" in schema or "anyOf" in schema:
            # This is a union type
            rule = " | ".join(
                (
                    self._visit(alt_schema, f'{name}{"-" if name else ""}{i}')
                    for i, alt_schema in enumerate(
                        schema.get("oneOf") or schema["anyOf"]
                    )
                )
            )
            return self._add_rule(rule_name, rule)

        elif "const" in schema:
            # This is a literal
            return self._add_rule(
                rule_name, self._format_literal(schema["const"])
            )

        elif "enum" in schema:
            # This is a set of literals
            rule = " | ".join(
                (self._format_literal(v) for v in schema["enum"])
            )
            return self._add_rule(rule_name, rule)

        elif schema_type == "object" and "properties" in schema:
            required_properties = set(
                schema.get("required", schema["properties"].keys())
            )
            if not required_properties:
                raise ValueError(
                    "Object schema must have at least one required property if `required` is specified"
                )
            prop_order = self._prop_order
            prop_pairs = sorted(
                schema["properties"].items(),
                key=lambda kv: (prop_order.get(kv[0], len(prop_order)), kv[0]),
            )

            rule_parts = []  # type: List[str]
            optional_rule_parts = []  # type: List[str]
            first_property = True  # type: bool

            for prop_name, prop_schema in prop_pairs:
                prop_rule_name = self._visit(
                    prop_schema, f'{name}{"-" if name else ""}{prop_name}'
                )
                prop_str = rf'{self._format_literal(prop_name)} space ":" space {prop_rule_name}'

                if prop_name in required_properties:
                    if not first_property:
                        prop_str = rf'"," space {prop_str}'
                    rule_parts.append(prop_str)
                    first_property = False
                else:
                    optional_rule_parts.append(prop_str)

            for i, optional_str in enumerate(optional_rule_parts):
                if i == 0 and not rule_parts:
                    # if no required properties
                    combined_str = rf"({optional_str})?"
                else:
                    combined_str = rf'("," space {optional_str})?'
                rule_parts.append(combined_str)

            # Combine rules
            rule = '"{" space ' + " ".join(rule_parts) + ' "}" space'
            return self._add_rule(rule_name, rule)

        elif schema_type == "array" and "items" in schema:
            # TODO `prefixItems` keyword
            item_rule_name = self._visit(
                schema["items"], f'{name}{"-" if name else ""}item'
            )
            rule = f'"[" space ({item_rule_name} ("," space {item_rule_name})*)? "]" space'
            return self._add_rule(rule_name, rule)

        else:
            assert (
                schema_type in PRIMITIVE_RULES
            ), f"Unrecognized schema: {schema}"
            return self._add_rule(
                "root" if rule_name == "root" else schema_type,
                PRIMITIVE_RULES[schema_type],
            )

    def _format_grammar(self):
        return "\n".join(
            (f"{name} ::= {rule}" for name, rule in self._rules.items())
        )


if __name__ == "__main__":
    from repositories.llama_cpp.llama_cpp import LlamaGrammar, Llama

    # Define a python function and parse it into a grammar
    def get_current_weather(
        location: Annotated[
            str,
            "The location to get the current weather for",
        ],
        unit: Annotated[
            str,
            "The unit of temperature to return",
            ["fahrenheit", "celsius"],
        ],
        source: Annotated[
            Optional[str],
            "The source of the weather information",
            ["openweathermap", "weatherapi"],
        ] = "openweathermap",
    ):
        """Get the current weather in a given location"""

    model_path = r"models\ggml\orca-mini-3b.ggmlv3.q4_0.bin"
    grammar: str = FunctionCallMixin.from_functions(get_current_weather)
    # print(f"Grammar:\n{grammar}")

    json_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {
                "type": "string",
                "enum": ["fahrenheit", "celsius"],
            },
            "source": {
                "type": "string",
                "enum": ["openweathermap", "weatherapi"],
            },
        },
        "required": ["location", "unit"],
    }  # type: Dict[SchemaKey, Any]
    grammar = FunctionCallMixin.from_json_schema(json_schema)
    print(f"Grammar:\n{grammar}")

    llama_grammar = LlamaGrammar.from_string(grammar, verbose=False)
    llm = Llama(model_path)
    for city in (
        "London",
        "Paris",
        "New York",
        "Berlin",
        "Tokyo",
        "Sydney",
        "Moscow",
        "Beijing",
        "Cairo",
        "Rome",
    ):
        output = llm(prompt=f"### User: What is the weather in {city} today? ### Assistant:", grammar=llama_grammar)["choices"][0]["text"]  # type: ignore
        print(json.loads(output))

    # Output:
    # { "location": "London", "source": "openweathermap","unit" : "celsius"}
