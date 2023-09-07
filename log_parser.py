from llama_api.utils.log_parser import parse_logs
from llama_api.shared.config import LogParserCliArgs as args

if __name__ == "__main__":
    args.load()
    parse_logs(
        chat_log_file_path=args.chat_log_file_path.value or "logs/chat.log",
        debug_log_file_path=args.debug_log_file_path.value
        or "logs/debug.log",
        output_path=args.output_path.value or "./logs/chat.csv",
        min_output_length=args.min_output_length.value or 30,
        ignore_messages_less_than=args.ignore_messages_less_than.value or 2,
    )
