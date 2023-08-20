from llama_api.server.app_settings import run
from llama_api.shared.config import MainCliArgs


if __name__ == "__main__":
    MainCliArgs.load()
    run()
