import argparse
import codecs
import csv
from functools import partial
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class ChatLog:
    """A single chat log."""

    prompt: Optional[str] = None
    chat: Optional[List[Dict[str, str]]] = None
    request: Optional[Dict[str, Any]] = None
    input: Optional[str] = None
    embedding: Optional[dict] = None


class ChatLogParser:
    """Parse the chat log file and return a summary of the results."""

    logger_header_regex: str = (
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] :"
        r"(?:DEBUG|INFO|WARNING|CRITICAL|ERROR) - "
    )

    def __init__(self, content: str) -> None:
        self.log_content: str = content
        logs = []  # type: List[ChatLog]
        for result in [
            json.loads(match)
            for match in re.findall(
                self.logger_header_regex
                + r"(.*?)(?="
                + self.logger_header_regex
                + "|$)",
                self.log_content,
                flags=re.DOTALL,
            )
        ]:
            if "chat" in result:
                assert isinstance(result["chat"], list)
                result["chat"] = [
                    m if isinstance(m, dict) else json.loads(m)
                    for m in result["chat"]
                ]
            logs.append(ChatLog(**result))
        self.logs = logs

    @classmethod
    def load_from_file(cls, log_path: PathLike) -> "ChatLogParser":
        """Load the log from a file and return a LogParser instance."""
        with open(log_path, "r", encoding="utf-8") as file:
            return cls(file.read())

    def extract_chats(
        self,
        input_checker: Optional[Callable[[str], bool]] = None,
        output_checker: Optional[Callable[[str], bool]] = None,
        ignore_messages_less_than: Optional[int] = 2,
        csv_output_path: Optional[PathLike] = None,
    ) -> List[Tuple[str, str, str]]:
        """Extract instruction, input, and output pairs from the chat log."""
        results = []  # type: List[Tuple[str, str, str]]

        if input_checker is None:
            input_checker = lambda _: True  # noqa: E731
        if output_checker is None:
            output_checker = lambda _: True  # noqa: E731
        for messages in (
            log.chat for log in self.logs if log.chat is not None
        ):
            if (
                ignore_messages_less_than is not None
                and len(messages) < ignore_messages_less_than
            ):
                print(f"[Warning!] chat length is less than 2: {messages}")
                continue

            # Extracting instruction
            last_user_prompt, last_assistant_prompt = next(
                (
                    (
                        message["content"].strip()
                        for message in reversed(messages)
                        if message["role"] == "user"
                    )
                ),
                None,
            ), next(
                (
                    (
                        message["content"].strip()
                        for message in reversed(messages)
                        if message["role"] == "assistant"
                    )
                ),
                None,
            )
            if last_user_prompt is None:
                print(f"[Warning!] No user role in chat: {messages}")
                continue
            if last_assistant_prompt is None:
                print(f"[Warning!] No assistant role in chat: {messages}")
                continue

            # Check input and output
            if input_checker(last_user_prompt) and output_checker(
                last_assistant_prompt
            ):
                results.append(
                    (
                        "\n".join(
                            message["content"].strip()
                            for message in messages
                            if message["role"] == "system"
                        ),
                        last_user_prompt,
                        last_assistant_prompt,
                    )
                )
        if csv_output_path:
            with codecs.open(
                str(csv_output_path), "w", encoding="utf-8-sig"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(("Instruction", "Input", "Output"))
                writer.writerows(results)
        return results


class DebugLogParser:
    """Parse the debug.log file and return a summary of the results."""

    tokens_pattern: re.Pattern = re.compile(
        r"tokens: (\d+)\(\s*([\d.]+)tok/s\)"
    )
    prompt_truncated_pattern: re.Pattern = re.compile(
        r"Prompt is too long, truncating (\d+) tokens."
    )
    status_pattern: re.Pattern = re.compile(r"🦙 \[(.+?) for (.+?)\]:")

    def __init__(self, content: str) -> None:
        self.log_content: str = content

    @classmethod
    def load_from_file(cls, log_path: PathLike) -> "DebugLogParser":
        """Load the log from a file and return a LogParser instance."""
        with open(log_path, "r", encoding="utf-8") as file:
            return cls(file.read())

    def count_entries(self) -> Dict[str, Dict[str, int]]:
        """Count the number of completed and interrupted requests for each
        model."""
        counts = defaultdict(lambda: defaultdict(int))
        for m in self.status_pattern.findall(self.log_content):
            status, model = m
            counts[model][status] += 1
        return {k: dict(v) for k, v in counts.items()}

    def parse_tokens_and_speed(self) -> Tuple[int, float]:
        """Parse the total number of tokens and average speed from the log."""
        matches = self.tokens_pattern.findall(self.log_content)
        total_tokens = sum(int(token) for token, _ in matches)
        average_speed = (
            sum(float(speed) for _, speed in matches) / len(matches)
            if matches
            else 0
        )

        return total_tokens, average_speed

    def count_truncated_tokens(self) -> Tuple[int, int]:
        """Count the number of truncated lines and tokens due to the prompt
        too long."""
        matches = self.prompt_truncated_pattern.findall(self.log_content)
        truncated_lines_count = len(matches)
        total_truncated_tokens = sum(int(token) for token in matches)
        return truncated_lines_count, total_truncated_tokens

    def get_summary(self) -> Dict[str, Any]:
        """Get the summary of the log."""
        total_tokens, average_speed = self.parse_tokens_and_speed()
        (
            truncated_lines_count,
            total_truncated_tokens,
        ) = self.count_truncated_tokens()
        return dict(
            model_counts=self.count_entries(),
            total_tokens=total_tokens,
            average_speed=average_speed,
            truncated_lines_count=truncated_lines_count,
            total_truncated_tokens=total_truncated_tokens,
        )


def output_checker(output_str: str, min_output_length: int = 0) -> bool:
    output_str = output_str.strip()

    # Check if the length is at least min_output_length
    if len(output_str) < min_output_length:
        return False

    # Check if the output is composed only of digits
    if output_str.isdigit():
        return False

    return True


def parse_logs(
    chat_log_file_path: str,
    debug_log_file_path: str,
    output_path: str,
    min_output_length: int,
    ignore_messages_less_than: int,
) -> None:
    """Parse the chat and debug logs and save the results as CSV."""
    if chat_log_file_path:
        if not Path(chat_log_file_path).exists():
            raise FileNotFoundError(f"File not found: {chat_log_file_path}")
        # Load log from file and process
        chat_log_parser = ChatLogParser.load_from_file(chat_log_file_path)

        # 채팅 데이터를 저장할 리스트
        num_chats = 0
        for num_chats, (instruction, input, output) in enumerate(
            chat_log_parser.extract_chats(
                ignore_messages_less_than=ignore_messages_less_than,
                output_checker=partial(
                    output_checker, min_output_length=min_output_length
                ),
                csv_output_path=output_path,
            ),
            start=1,
        ):
            print(f"Instruction: {instruction[:30]}")
            print(f"Input: {input[:30]}")
            print(f"Output: {output[:30]}")
            print("=" * 50)
        print(f"Number of chats: {num_chats}")

    if debug_log_file_path:
        if not Path(debug_log_file_path).exists():
            raise FileNotFoundError(f"File not found: {debug_log_file_path}")
        # Load log from file and process
        debug_log_parser = DebugLogParser.load_from_file(debug_log_file_path)
        print(debug_log_parser.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process chat and debug logs."
    )

    parser.add_argument(
        "--min-output-length",
        "-m",
        type=int,
        default=30,
        help="Minimum length for the output.",
    )
    parser.add_argument(
        "--chat-log-file-path",
        "-c",
        type=str,
        default="logs/chat.log",
        help="Path to the chat log file.",
    )
    parser.add_argument(
        "--debug-log-file-path",
        "-d",
        type=str,
        default="logs/debug.log",
        help="Path to the debug log file.",
    )
    parser.add_argument(
        "--ignore-messages-less-than",
        "-i",
        type=int,
        default=2,
        help="Ignore messages shorter than this length.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./logs/chat.csv",
        help="Path to save the extracted chats as CSV.",
    )

    # Use the argparse results to parse the logs
    args = parser.parse_args()
    parse_logs(
        chat_log_file_path=args.chat_log_file_path,
        debug_log_file_path=args.debug_log_file_path,
        output_path=args.output_path,
        min_output_length=args.min_output_length,
        ignore_messages_less_than=args.ignore_messages_less_than,
    )
