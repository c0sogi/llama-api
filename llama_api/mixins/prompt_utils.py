from typing import List, Optional, Set, Tuple

from ..schemas.api import (
    APIChatMessage,
    CreateChatCompletionRequest,
    TextGenerationSettings,
)
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


class PromptUtilsMixin:
    _stop_set: Optional[Set[str]] = None
    _stop_piece_set: Optional[Set[str]] = None

    def convert_messages_into_prompt(
        self,
        body: CreateChatCompletionRequest,
        instruction_template: Optional[str] = None,
    ) -> str:  # noqa: F821
        """A helper method to convert list of messages into one text prompt.
        Save the stop tokens in the settings object for later use."""

        prompt, stops = self._convert_messages_into_prompt(
            body.messages, instruction_template
        )
        if isinstance(body.stop, str):
            body.stop = stops + [body.stop]
        elif isinstance(body.stop, list):
            body.stop = stops + body.stop
        else:
            body.stop = stops
        return prompt

    def _convert_messages_into_prompt(
        self,
        messages: List[APIChatMessage],
        instruction_template: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        stops = set()
        role_formats = None

        if instruction_template:
            try:
                import yaml

                instruct_template = yaml.safe_load(
                    open(
                        f"instruction-templates/{instruction_template}.yaml",
                        "r",
                    )
                )

                turn_template = instruct_template["turn_template"]
                bot_start = turn_template.find("<|bot|>")  # type: int
                bot_message_template = (
                    turn_template[bot_start:]
                    .replace("<|bot-message|>", "{message}")
                    .replace("<|bot|>", instruct_template.get("bot", ""))
                )  # type: str

                if "alpaca" in instruction_template.lower():
                    stops.add("\n###")
                elif instruct_template["user"]:
                    # WizardLM and some others have no user prompt.
                    stops.add(instruct_template["user"])
                    stops.add("\n" + instruct_template["user"])

                logger.debug(
                    f"Loaded instruction role format: {instruction_template}"
                )
                role_formats = {
                    "user": (
                        turn_template[:bot_start]
                        .replace("<|user-message|>", "{message}")
                        .replace("<|user|>", instruct_template.get("user", ""))
                    ),
                    "assistant": bot_message_template,
                    "system": "{message}",
                    "function": "{message}",
                    "context": instruct_template.get("context", ""),
                    "prompt": bot_message_template[
                        : bot_message_template.find("{message}")
                    ].rstrip(" "),
                }
            except Exception as e:
                stops.add("\nUser:")
                stops.add("\nUser: ")
                logger.error(
                    "Exception: When loading "
                    f"instruction-templates/{instruction_template}.yaml: {e}\n"
                    "Loaded default instruction-following template for model."
                )

        else:
            stops.add("\nUser:")
            stops.add("\nUser: ")

        system_prompts = []  # type: List[str]
        chat_histories = []  # type: List[str]
        if role_formats is None:
            role_formats = {
                "user": "User: {message}\n",
                "assistant": "Assistant: {message}\n",
                "system": "{message}",
                "function": "{message}",
                "context": "You are a helpful assistant.",
                "prompt": "Assistant:",
            }
        for message in messages:
            msg = role_formats[message.role].format(message=message.content)
            system_prompts.append(msg) if message.role in (
                "system",
                "function",
            ) else chat_histories.append(msg)

        return (
            self._ensure_line_break("\n".join(system_prompts))
            + self._ensure_line_break(
                (
                    role_formats["system"].format(
                        message=role_formats["context"]
                    )
                    if role_formats["context"]
                    else ""
                )
            )
            + "".join(chat_histories)
            + role_formats["prompt"]
        ), list(stops)

    def build_stops_from_settings(
        self, settings: TextGenerationSettings
    ) -> None:
        """Pre-calculate sets for stops and the pieces of stops,
        to speed up the stop checking process."""
        if isinstance(settings.stop, str):
            stops = [settings.stop]  # type: List[str]
        elif isinstance(settings.stop, list):
            stops = settings.stop
        else:
            stops = []
        self._stop_set = set(stops)
        self._stop_piece_set = {
            stop[:prefix_idx]
            for stop in stops
            for prefix_idx in range(1, len(stop))
        }

    def stop_checker(self, text_piece: str) -> Optional[bool]:
        """Optimized stop checker for text completion.
        Returns False if the text piece ends with any piece of stop.
        Returns True if the text piece contains any stop.
        Returns None if the text piece does not contain any piece of stop."""
        if any(
            text_piece.endswith(stop_piece)
            for stop_piece in self._stop_piece_set or ()
        ):
            return False
        if any(stop in text_piece for stop in self._stop_set or ()):
            return True
        return None

    @staticmethod
    def _ensure_line_break(msg: str) -> str:
        return msg if msg.endswith("\n") else msg + "\n" if msg else ""

    @staticmethod
    def raise_for_token_limit(prompt_tokens: int, context_window: int) -> None:
        """A helper method to raise an error if the number of tokens
        requested for completion exceeds the context window."""
        if prompt_tokens >= context_window:
            raise ValueError(
                f"Requested tokens ({prompt_tokens}) exceed "
                f"context window of {context_window}"
            )
