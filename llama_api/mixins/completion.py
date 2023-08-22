from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Dict, Literal, Optional

from ..schemas.api import CompletionLogprobs, TextGenerationSettings


@dataclass
class CompletionStatus:
    # These fields are automatically set
    started_at: float = field(default_factory=time, init=False)

    # These fields are set by `accept_settings` method.
    input_text: str = field(default="", init=False)
    input_tokens: int = field(default=0, init=False)

    # These fields are set by `generate_text` method.
    generated_text: str = field(default="", init=False)
    generated_tokens: int = field(default=0, init=False)
    logprobs: Optional[CompletionLogprobs] = field(default=None, init=False)


class CompletionMixin:
    """A mixin for modules that support completion generation."""

    _completion_status: Optional["defaultdict[str, CompletionStatus]"] = None

    @property
    def completion_status(self) -> Dict[str, CompletionStatus]:
        """Get the completion status.
        key: completion_id
        value: CompletionStatus"""
        if self._completion_status is None:
            self._completion_status = defaultdict(CompletionStatus)
        return self._completion_status

    def get_finish_reason(
        self,
        settings: TextGenerationSettings,
    ) -> Literal["length", "stop"]:
        """Get the finish reason for the completion."""
        return (
            "length"
            if self.completion_status[settings.completion_id].generated_tokens
            >= settings.max_tokens
            else "stop"
        )
