from typing import List, Optional, Set

from ..schemas.api import APIChatMessage, TextGenerationSettings


def _get_stop_strings(*roles: str) -> List[str]:
    """A helper method to generate stop strings for a given set of roles.
    Stop strings are required to stop text completion API from generating
    text that does not belong to the current chat turn.
    e.g. The common stop string is "### USER:",
    which can prevent ai from generating user's message itself."""

    prompt_stop = set()
    for role in roles:
        avoids = (
            f"### {role}:",
            f"###{role}:",
        )
        prompt_stop.update(
            avoids,
            map(str.capitalize, avoids),
            map(str.upper, avoids),
            map(str.lower, avoids),
        )
    return list(prompt_stop)


class PromptUtilsMixin:
    _stop_set: Optional[Set[str]] = None
    _stop_piece_set: Optional[Set[str]] = None

    @staticmethod
    def convert_messages_into_prompt(
        messages: List[APIChatMessage], settings: TextGenerationSettings
    ) -> str:
        """A helper method to convert list of messages into one text prompt.
        Save the stop tokens in the settings object for later use."""

        stops = _get_stop_strings(
            *set(message.role for message in messages)
        )  # type: List[str]
        if isinstance(settings.stop, str):
            settings.stop = stops + [settings.stop]
        elif isinstance(settings.stop, list):
            settings.stop = stops + settings.stop
        else:
            settings.stop = stops
        return (
            " ".join(
                [
                    f"### {message.role.upper()}: {message.content}"
                    for message in messages
                ]
            )
            + " ### ASSISTANT: "
        )

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
    def raise_for_token_limit(prompt_tokens: int, context_window: int) -> None:
        """A helper method to raise an error if the number of tokens
        requested for completion exceeds the context window."""
        if prompt_tokens >= context_window:
            raise ValueError(
                f"Requested tokens ({prompt_tokens}) exceed "
                f"context window of {context_window}"
            )
