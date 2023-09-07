"""Helper script to download models from Huggingface repository."""
from llama_api.utils.huggingface_downloader import HuggingfaceDownloader
from llama_api.shared.config import ModelDownloaderCliArgs

if __name__ == "__main__":
    ModelDownloaderCliArgs.load()
    assert ModelDownloaderCliArgs.model.value, "Model is required"
    for model in ModelDownloaderCliArgs.model.value:
        try:
            print(f"Downloading model `{model}`...")
            HuggingfaceDownloader.from_repository(
                model=model,
                branch=ModelDownloaderCliArgs.branch.value or "main",
                base_folder=ModelDownloaderCliArgs.output.value,
                clean=ModelDownloaderCliArgs.clean.value or False,
                check=ModelDownloaderCliArgs.check.value or False,
                text_only=ModelDownloaderCliArgs.text_only.value or False,
                threads=ModelDownloaderCliArgs.threads.value or 1,
                start_from_scratch=ModelDownloaderCliArgs.start_from_scratch.value  # noqa: E501
                or False,
            )
        except Exception as e:
            print(f"Failed to download model `{model}`: {e}")
            continue
