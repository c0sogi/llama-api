import argparse
from llama_api.server.app_settings import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of process workers to run",
    )

    args = parser.parse_args()
    run(
        port=args.port,
        max_workers=args.max_workers,
    )
