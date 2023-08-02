from pathlib import Path
from re import Match, Pattern, compile
from typing import Callable, Coroutine, Dict, Optional, Tuple, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from typing_extensions import TypedDict

from ..schemas.api import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
)
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


class ErrorResponse(TypedDict):
    """OpenAI style error response"""

    message: str
    type: str
    param: Optional[str]
    code: Optional[str]


class ErrorResponseFormatters:
    """Collection of formatters for error responses.

    Args:
        request (Union[CreateCompletionRequest, CreateChatCompletionRequest]):
            Request body
        match (Match[str]): Match object from regex pattern

    Returns:
        Tuple[int, ErrorResponse]: Status code and error response
    """

    @staticmethod
    def context_length_exceeded(
        request: Union[
            "CreateCompletionRequest", "CreateChatCompletionRequest"
        ],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for context length exceeded error"""

        context_window = int(match.group(2))
        prompt_tokens = int(match.group(1))
        completion_tokens = request.max_tokens
        if hasattr(request, "messages"):
            # Chat completion
            param = "messages"
            message = (
                "This model's maximum context length is {} tokens. "
                "However, you requested {} tokens "
                "({} in the messages, {} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        else:
            # Text completion
            param = "prompt"
            message = (
                "This model's maximum context length is {} tokens, "
                "however you requested {} tokens "
                "({} in your prompt; {} for the completion). "
                "Please reduce your prompt; or completion length."
            )
        return 400, ErrorResponse(
            message=message.format(
                context_window,
                completion_tokens + prompt_tokens,
                prompt_tokens,
                completion_tokens,
            ),
            type="invalid_request_error",
            param=param,
            code="context_length_exceeded",
        )

    @staticmethod
    def model_not_found(
        request: Union[
            "CreateCompletionRequest", "CreateChatCompletionRequest"
        ],
        match,  # type: Match[str]
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for model_not_found error"""

        model_path = str(match.group(1))
        message = f"The model `{model_path}` does not exist"
        return 400, ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=None,
            code="model_not_found",
        )


class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    # key: regex pattern for original error message from llama_cpp
    # value: formatter function
    pattern_and_formatters: Dict[
        Pattern,
        Callable[
            [
                Union[
                    "CreateCompletionRequest", "CreateChatCompletionRequest"
                ],
                "Match[str]",
            ],
            Tuple[int, ErrorResponse],
        ],
    ] = {
        compile(
            r"Requested tokens \((\d+)\) exceed context window of (\d+)"
        ): ErrorResponseFormatters.context_length_exceeded,
        compile(
            r"Model path does not exist: (.+)"
        ): ErrorResponseFormatters.model_not_found,
    }

    def error_message_wrapper(
        self,
        error: Exception,
        body: Optional[
            Union[
                "CreateChatCompletionRequest",
                "CreateCompletionRequest",
                "CreateEmbeddingRequest",
            ]
        ] = None,
    ) -> Tuple[int, ErrorResponse]:
        """Wraps error message in OpenAI style error response"""

        if body is not None and isinstance(
            body,
            (
                CreateCompletionRequest,
                CreateChatCompletionRequest,
            ),
        ):
            # When text completion or chat completion
            for pattern, callback in self.pattern_and_formatters.items():
                match = pattern.search(str(error))
                if match is not None:
                    return callback(body, match)

        # Wrap other errors as internal server error
        return 500, ErrorResponse(
            message=str(error),
            type="internal_server_error",
            param=f"traceback:: {self.parse_trackback(error)}",
            code=type(error).__name__,
        )

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[None, None, Response]]:
        return self.custom_route_handler

    async def custom_route_handler(self, request: Request) -> Response:
        """Defines custom route handler that catches exceptions and formats
        in OpenAI style error response"""
        try:
            return await super().get_route_handler()(request)
        except Exception as error:
            json_body = await request.json()
            try:
                if "messages" in json_body and "prompt" not in json_body:
                    # Chat completion
                    body: Optional[
                        Union[
                            CreateChatCompletionRequest,
                            CreateCompletionRequest,
                            CreateEmbeddingRequest,
                        ]
                    ] = CreateChatCompletionRequest(**json_body)
                elif "prompt" in json_body and "messages" not in json_body:
                    # Text completion
                    body = CreateCompletionRequest(**json_body)
                else:
                    # Embedding
                    body = CreateEmbeddingRequest(**json_body)
            except Exception:
                # Invalid request body
                body = None

            # Get proper error message from the exception
            (
                status_code,
                error_message,
            ) = self.error_message_wrapper(error=error, body=body)
            return JSONResponse(
                {"error": error_message},
                status_code=status_code,
            )

    def parse_trackback(self, exception: Exception) -> str:
        """Parses traceback information from the exception"""
        if (
            exception.__traceback__ is not None
            and exception.__traceback__.tb_next is not None
        ):
            # Get previous traceback from the exception
            traceback = exception.__traceback__.tb_next

            # Get filename, function name, and line number
            try:
                co_filename = Path(traceback.tb_frame.f_code.co_filename).name
            except Exception:
                co_filename = "UNKNOWN"
            co_name = traceback.tb_frame.f_code.co_name
            lineno = traceback.tb_lineno
            return f"Error in {co_filename} at line {lineno} in {co_name}"

        # If traceback is not available, return UNKNOWN
        return "UNKNOWN"
