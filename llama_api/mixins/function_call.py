"""Helper classes for wrapping functions in OpenAI's API"""
# flake8: noqa
import json
from inspect import signature
from re import DOTALL, Pattern, compile
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

from orjson import JSONDecodeError, loads

from ..schemas.api import (
    APIChatMessage,
    CreateChatCompletionRequest,
    FunctionCompletion,
    FunctionCompletionChunk,
    FunctionSchema,
)
from ..schemas.function_call import (
    FunctionCall,
    FunctionCallParameter,
    JsonTypes,
)

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

    _space_rule: str = "([ \t\n])?"
    _primitive_rules: Dict[str, str] = {
        "boolean": '("true" | "false") space',
        "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
        "integer": '("-"? ([0-9] | [1-9] [0-9]*)) space',
        "string": r""" "\"" (
            [^"\\] |
            "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
        )* "\"" space """,
        "null": '"null" space',
    }
    _grammar_literal_escapes: Dict[str, str] = {
        "\r": "\\r",
        "\n": "\\n",
        '"': '\\"',
    }
    _invalid_rule_chars_re: "Pattern[str]" = compile(r"[^a-zA-Z0-9-]+")
    _grammar_literal_escape_re: "Pattern[str]" = compile(r'[\r\n"]')
    _prop_order: Dict[str, int]
    _rules: Dict[str, str]
    _line_break_re: "Pattern[str]" = compile(r"\n\s*")
    _function_call_re: "Pattern[str]" = compile(
        r'\{\s*"name"\s*:\s*"((?:[^"]|\\")*)"\s*,\s*"arguments"\s*:\s*(\{.*\})\s*\}\s*',
        flags=DOTALL,
    )
    _function_name_re: "Pattern[str]" = compile(
        r'\{\s*"name"\s*:\s*"((?:[^"]|\\")*)', flags=DOTALL
    )
    _function_arguments_re: "Pattern[str]" = compile(r'"arguments"\s*:\s*{.*')

    def generate_function_call(
        self, generated_text: str
    ) -> FunctionCompletion:
        function_call_match = self._function_call_re.search(generated_text)
        assert (
            function_call_match is not None
        ), f"Invalid function call: {generated_text}"
        return {
            "name": function_call_match.group(1),
            "arguments": function_call_match.group(2),
        }

    def generate_function_call_streaming(
        self, text_generator: Iterator[str]
    ) -> Iterator[FunctionCompletionChunk]:
        function_call = last_function_name = ""
        begin_function_arguments = False
        for token in text_generator:
            function_call += token
            name_match = self._function_name_re.search(function_call)
            arguments_match = self._function_arguments_re.search(function_call)
            if name_match is not None and arguments_match is None:
                current_function_name = name_match.group(1)
                if current_function_name == last_function_name:
                    continue
                last_function_name = current_function_name
                function_chunk = {
                    "name": token,
                }  # type: FunctionCompletionChunk
            elif name_match is not None:
                if not begin_function_arguments:
                    begin_function_arguments = True
                    token = token[token.rfind("{") :]
                try:
                    loads(function_call)
                    token = token[: token.rfind("}")]
                    if not token:
                        continue
                except JSONDecodeError:
                    pass
                function_chunk = {"arguments": token}
            else:
                continue
            yield function_chunk

    @classmethod
    def from_json_schema(
        cls,
        schema: Union[Dict[SchemaKey, Any], str],
        prop_order: Optional[Dict[str, int]] = None,
    ) -> str:
        """Parse a JSON schema into a BNF grammar."""
        if isinstance(schema, str):
            schema = json.loads(schema)
            assert isinstance(schema, dict), "schema must be valid JSON"
        self = cls()
        self._prop_order = prop_order or {}
        self._rules = {"space": self._space_rule}
        self._visit(schema, "root")
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
            self._rules = {"space": self._space_rule}
            parameters = function_call.to_dict().get("parameters")
            assert parameters is not None, "function call must have parameters"
            self._visit(dict(parameters), "root")
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
        line_break_pattern = cls._line_break_re

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
        return cls.from_function_calls(
            function_calls if return_as_list else function_calls[0],
            prop_order,
        )

    def _format_literal(self, literal: Any) -> str:
        escaped = self._grammar_literal_escape_re.sub(
            lambda m: self._grammar_literal_escapes.get(m.group(0)) or "",
            json.dumps(literal),
        )
        return f'"{escaped}"'

    def _add_rule(self, name, rule):
        esc_name = self._invalid_rule_chars_re.sub("-", name)
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
                schema_type in self._primitive_rules
            ), f"Unrecognized schema: {schema}"
            return self._add_rule(
                "root" if rule_name == "root" else schema_type,
                self._primitive_rules[schema_type],
            )

    def _format_grammar(self):
        return "\n".join(
            (f"{name} ::= {rule}" for name, rule in self._rules.items())
        )

    def accept_function_call(
        self, request: CreateChatCompletionRequest
    ) -> None:
        """Accept the function call request"""
        functions: List[FunctionSchema] = request.functions or []
        function_call: List[FunctionSchema] = []

        if request.functions:
            function_call = functions

        if request.function_call:
            if not functions:
                raise ValueError("No functions specified")
            if not isinstance(request.function_call, (str, dict)):
                raise ValueError(
                    "function_call must be "
                    "a string or a dictionary of parameters"
                )
            if isinstance(request.function_call, dict):
                if not (
                    function_call := [
                        function
                        for function in functions
                        if function["name"] == request.function_call["name"]
                    ]
                ):
                    raise ValueError(
                        f"Function {request.function_call['name']} not found"
                    )
            elif request.function_call == "none":
                function_call.clear()
            elif request.function_call == "auto":
                pass
            else:
                if not (
                    function_call := [
                        function
                        for function in functions
                        if function["name"] == request.function_call
                    ]
                ):
                    raise ValueError(
                        f"Function {request.function_call} not found"
                    )
        if functions:
            for function in functions:
                request.messages.insert(
                    0,
                    APIChatMessage(
                        role="function",
                        content=self.format_function_into_prompt(function),
                    ),
                )
        if function_call:
            request.grammar = self.from_json_schema(
                {
                    "type": "oneOf",
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "const",
                                    "const": function["name"],
                                },
                                "arguments": function["parameters"],
                            },
                            "required": ["name", "arguments"],
                        }
                        for function in function_call
                    ],
                },
                prop_order={"name": 0, "arguments": 1},
            )

    @staticmethod
    def format_function_into_prompt(function: FunctionSchema) -> str:
        """Format a function into a prompt."""
        # Get function name and description
        description = function.get("description", "None")

        # Start building pseudo-function string
        pseudo_function = f"\nFunction {function.get('name', 'Unknown')}()\n\tDescription: {description}\n"

        # Parse parameters
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        required_params = parameters.get("required", properties.keys())

        pseudo_function += "\tParameters:\n"
        for param, details in properties.items():
            param_type = details.get("type", "Unknown")

            # Check if parameter is required
            pseudo_function += (
                (f"\t\t{param} (REQUIRED, Type: {param_type})")
                if param in required_params
                else f"\t\t{param} (Type: {param_type})"
            )

            # Add description if available
            if "description" in details:
                pseudo_function += f" - {details['description']}"

            # Add enum values if available
            if "enum" in details:
                pseudo_function += (
                    f" [Enum: {', '.join([str(e) for e in details['enum']])}]"
                )

            pseudo_function += "\n"

        return pseudo_function


if __name__ == "__main__":
    from repositories.llama_cpp.llama_cpp import Llama, LlamaGrammar

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
