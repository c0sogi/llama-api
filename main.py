import argparse
from llama_api.server.app_settings import run


if __name__ != "__mp_main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to run the server on",
    )

    args = parser.parse_args()
    run(port=args.port)
