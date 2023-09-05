from typing import Optional, Set

from ..schemas.api import (
    CreateChatCompletionRequest,
    TextGenerationSettings,
)
from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


class PromptUtilsMixin:
    _stop_set: Optional[Set[str]] = None
    _stop_piece_set: Optional[Set[str]] = None
    _role_formats_and_stops = (
        {}
    )  # type: dict[str, tuple[dict[str, str], set[str]]]
    _default_role_formats = {
        "user": "User: {message}\n",
        "assistant": "Assistant: {message}\n",
        "system": "{message}",
        "function": "{message}",
        "context": "You are a helpful assistant.",
        "prompt": "Assistant:",
    }  # type: dict[str, str]
    _default_stops = {
        "User:",
        " User: ",
        "\nUser:",
        "\nUser: ",
    }  # type: set[str]

    def convert_messages_into_prompt(
        self,
        body: CreateChatCompletionRequest,
        instruction_template: Optional[str] = None,
    ) -> str:  # noqa: F821
        """A helper method to convert list of messages into one text prompt.
        Save the stop tokens in the settings object for later use."""

        if instruction_template:
            self.build_role_formats(instruction_template)
            role_formats, stops = self._role_formats_and_stops.get(
                instruction_template,
                (
                    self._default_role_formats,
                    self._default_stops,
                ),
            )
        else:
            role_formats, stops = (
                self._default_role_formats,
                self._default_stops,
            )
        system_prompts = []  # type: list[str]
        chat_histories = []  # type: list[str]
        for message in body.messages:
            msg = role_formats[message.role].format(message=message.content)
            system_prompts.append(msg) if message.role in (
                "system",
                "function",
            ) else chat_histories.append(msg)

        if isinstance(body.stop, str):
            body.stop = list(stops.union({body.stop}))
        elif isinstance(body.stop, list):
            body.stop = list(stops.union(body.stop))
        else:
            body.stop = list(stops)
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
        )

    def build_role_formats(self, instruction_template: str) -> None:
        if instruction_template in self._role_formats_and_stops:
            return
        try:
            import yaml

            template, stops = (
                yaml.safe_load(
                    open(
                        f"instruction-templates/{instruction_template}.yaml",
                        "r",
                    )
                ),
                set(),
            )

            logger.info(
                f"Loaded instruction role format: {instruction_template}"
            )

            turn_template = template["turn_template"]
            bot_start = turn_template.find("<|bot|>")  # type: int
            bot_message_template = (
                turn_template[bot_start:]
                .replace("<|bot-message|>", "{message}")
                .replace("<|bot|>", template.get("bot", ""))
            )  # type: str

            if "alpaca" in instruction_template.lower():
                stops.add("\n###")
            elif template["user"]:
                # WizardLM and some others have no user prompt.
                stops.add(template["user"])
                stops.add("\n" + template["user"])
            self._role_formats_and_stops[instruction_template] = (
                {
                    "user": (
                        turn_template[:bot_start]
                        .replace("<|user-message|>", "{message}")
                        .replace("<|user|>", template.get("user", ""))
                    ),
                    "assistant": bot_message_template,
                    "system": "{message}",
                    "function": "{message}",
                    "context": template.get("context", ""),
                    "prompt": bot_message_template[
                        : bot_message_template.find("{message}")
                    ].rstrip(" "),
                },
                stops,
            )

        except Exception as e:
            logger.error(
                "Exception: When loading "
                f"instruction-templates/{instruction_template}.yaml: {e}\n"
                "Loaded default instruction-following template for model."
            )

    def build_stops_from_settings(
        self, settings: TextGenerationSettings
    ) -> None:
        """Pre-calculate sets for stops and the pieces of stops,
        to speed up the stop checking process."""
        if isinstance(settings.stop, str):
            stops = [settings.stop]  # type: list[str]
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
    def raise_for_token_limit(
        prompt_tokens: int, context_window: int
    ) -> None:
        """A helper method to raise an error if the number of tokens
        requested for completion exceeds the context window."""
        if prompt_tokens >= context_window:
            raise ValueError(
                f"Requested tokens ({prompt_tokens}) exceed "
                f"context window of {context_window}"
            )
