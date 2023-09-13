from functools import cached_property
from pathlib import Path
from re import Match, Pattern, compile
from typing import Callable, Coroutine, Dict, Optional, Tuple, Union

from anyio import get_cancelled_exc_class
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.types import Receive, Scope, Send
from typing_extensions import TypedDict

from ..schemas.api import (
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
)
from ..shared.config import MainCliArgs
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


class EmptyResponse(Response):
    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Do nothing"""


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
        match,  # type: Match[str]
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
                (completion_tokens or 0) + prompt_tokens,
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

    api_key: Optional[str] = MainCliArgs.api_key.value or None

    @cached_property
    def authorization(self) -> Optional[str]:
        """API key for authentication"""
        if self.api_key is None:
            return None
        return f"Bearer {self.api_key}"

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
            param=f"traceback:: {parse_traceback(error)}",
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
            if self.authorization is not None:
                # Check API key
                authorization = request.headers.get(
                    "Authorization",
                    request.query_params.get("authorization", None),
                )  # type: Optional[str]
                if not authorization or not authorization.startswith(
                    "Bearer "
                ):
                    error_response = ErrorResponse(
                        message=(
                            (
                                "You didn't provide an API key. "
                                "You need to provide your API key in "
                                "an Authorization header using Bearer auth "
                                "(i.e. Authorization: Bearer YOUR_KEY)."
                            )
                        ),
                        type="invalid_request_error",
                        param=None,
                        code=None,
                    )
                    return JSONResponse(
                        {"error": error_response},
                        status_code=401,
                    )
                if authorization.lower() != self.authorization.lower():
                    api_key = authorization[len("Bearer ") :]  # noqa: E203
                    error_response = ErrorResponse(
                        message=(
                            "Incorrect API key provided: "
                            + mask_secret(api_key, 8, 4)
                        ),
                        type="invalid_request_error",
                        param=None,
                        code="invalid_api_key",
                    )
                    return JSONResponse(
                        {"error": error_response},
                        status_code=401,
                    )
            return await super().get_route_handler()(request)
        except get_cancelled_exc_class():
            # Client has disconnected
            return EmptyResponse()
        except Exception as error:
            if request.method != "GET":
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
                    elif (
                        "prompt" in json_body and "messages" not in json_body
                    ):
                        # Text completion
                        body = CreateCompletionRequest(**json_body)
                    else:
                        # Embedding
                        body = CreateEmbeddingRequest(**json_body)
                except Exception:
                    # Invalid request body
                    body = None
            else:
                body = None

            # Get proper error message from the exception
            (
                status_code,
                error_message,
            ) = self.error_message_wrapper(error=error, body=body)
            client = request.client.host if request.client else "UNKNOWN"
            logger.error(
                f'"{client} → {request.url.path}": {error_message["message"]}'
            )
            return JSONResponse(
                {"error": error_message},
                status_code=status_code,
            )


def parse_traceback(exception: Exception) -> str:
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


def mask_secret(api_key: str, n_start: int, n_end: int) -> str:
    length = len(api_key)
    if length <= n_start + n_end:
        return api_key
    else:
        return (
            api_key[:n_start]
            + "*" * (length - n_start - n_end)
            + api_key[-n_end:]
        )
