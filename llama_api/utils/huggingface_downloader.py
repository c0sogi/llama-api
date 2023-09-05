"""
Downloads models from Hugging Face to models/username_modelname.
"""

import argparse
import base64
import datetime
import hashlib
import json
from os import environ
from pathlib import Path
from re import Pattern, compile
from typing import Dict, List, Literal, Optional

from requests import HTTPError, Response, Session

from ..utils.logger import ApiLogger

try:
    from typing_extensions import TypedDict


except ImportError:
    print("Failed to import typing_extensions, using TypedDict from typing")
    from typing import TypedDict  # When dependencies aren't installed yet
try:
    from tqdm import tqdm
    from tqdm.contrib.concurrent import thread_map

except ImportError:
    tqdm = thread_map = None

logger = ApiLogger(__name__)


Classification = Literal["text", "pytorch", "safetensors", "ggml", "pt"]


class HuggingfaceModelInfo(TypedDict):
    links: List[str]
    file_names: List[str]
    sha256: List[List[str]]
    is_lora: bool
    classifications: List[Classification]


class HuggingfaceDownloader:
    _base_url: str = "https://huggingface.co"
    _branch_pattern: Pattern = compile(r"^[a-zA-Z0-9._-]+$")
    _pytorch_pattern: Pattern = compile(r"(pytorch|adapter)_model.*\.bin")
    _safetensors_pattern: Pattern = compile(r".*\.safetensors")
    _pt_pattern: Pattern = compile(r".*\.pt")
    _ggml_pattern: Pattern = compile(r".*\.(bin|gguf)")
    _tokenizer_pattern: Pattern = compile(r"(tokenizer|ice).*\.model")
    _text_pattern: Pattern = compile(r".*\.(txt|json|py|md)")

    def __init__(
        self,
        model: str,
        branch: str = "main",
        threads: int = 1,
        base_folder: Optional[str] = None,
        clean: bool = False,
        check: bool = False,
        text_only: bool = False,
        start_from_scratch: bool = False,
    ) -> None:
        self.session = Session()
        user: Optional[str] = environ.get("HF_USER")
        password: Optional[str] = environ.get("HF_PASS")
        if user and password:
            self.session.auth = (user, password)

        # Cleaning up the model/branch names
        try:
            self._model = model
            self._branch = branch
            self.threads = threads
            self.base_folder = (
                Path(base_folder.lower()) if base_folder else None
            )
            self.clean = clean
            self.check = check
            self.text_only = text_only
            self.start_from_scratch = start_from_scratch
            self.progress_bar = None

            # Getting the model info from Huggingface
            self.hf_info: HuggingfaceModelInfo = (
                self._get_model_info_from_huggingface()
            )
        except ValueError as err_branch:
            logger.error(err_branch)
            raise
        except HTTPError as err_http:
            logger.error(err_http)
            raise

    @property
    def model(self) -> str:
        if self._model.endswith("/"):
            return self._model.lower()[:-1]
        return self._model.lower()

    @property
    def branch(self) -> str:
        if not self._branch_pattern.match(self._branch or "main"):
            raise ValueError(
                "Invalid branch name. Only alphanumeric characters, "
                "period, underscore and dash are allowed."
            )
        return self._branch.lower()

    @property
    def output_folder(self) -> Path:
        if self.base_folder is None:
            return (
                Path("models")
                if not self.hf_info["is_lora"]
                else Path("loras")
            )
        return self.base_folder

    @classmethod
    def from_repository(
        cls,
        model: str,  # e.g. "facebook/opt-1.3b"
        branch: str = "main",
        threads: int = 1,
        base_folder: Optional[str] = None,
        clean: bool = False,
        check: bool = False,
        text_only: bool = False,
        start_from_scratch: bool = False,
    ) -> "HuggingfaceDownloader":
        self = cls(
            model=model,
            branch=branch,
            threads=threads,
            base_folder=base_folder,
            clean=clean,
            check=check,
            text_only=text_only,
            start_from_scratch=start_from_scratch,
        )

        # Getting the output folder
        logger.info(
            "Links:"
            + "".join([f"\n- {link}" for link in self.hf_info["links"]])
            + "\n"
            "SHA256:"
            + "".join(
                [
                    f"\n- {fname}: {fhash}"
                    for fname, fhash in self.hf_info["sha256"]
                ]
            )
            + "\n"
            f"Is LoRA: {self.hf_info['is_lora']}\n"
            f"Output folder: {self.output_folder}"
        )

        if self.check:
            # Check previously downloaded files
            self.check_model_files_by_sha256()
        else:
            # Download files
            self.download_model_files()
        return self

    def download_model_files(
        self, links: Optional[List[str]] = None
    ) -> None:
        # Creating the folder and writing the metadata
        output_folder: Path = self.output_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        metadata: str = (
            f"- url: {self._base_url}/{self.model}\n"
            f"- branch: {self.branch}\n"
            f'- download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'  # noqa: E501
        )
        logger.info(f"Downloading with metadata:\n{metadata}")

        sha256_str: str = "\n".join(
            [f"    {item[1]} {item[0]}" for item in self.hf_info["sha256"]]
        )
        if sha256_str:
            metadata += f"sha256sum:\n{sha256_str}"

        metadata += "\n"
        (output_folder / "huggingface-metadata.txt").write_text(metadata)

        # Downloading the files
        logger.info(f"Downloading the model to {output_folder}...")
        self._start_download_threads(links=links or self.hf_info["links"])

    def get_single_file(self, url: str) -> None:
        file_name = Path(url.rsplit("/", 1)[1])
        output_path = self.output_folder / file_name
        headers: Dict[str, str] = {}
        mode: str = "wb"
        if output_path.exists() and not self.start_from_scratch:
            # Check if the file has already been downloaded completely
            response: Response = self.session.get(
                url, stream=True, timeout=20
            )
            total_size: int = int(response.headers.get("content-length", 0))
            if output_path.stat().st_size >= total_size:
                logger.info(f"{file_name} already exists. Skipping...")
                return

            # Otherwise, resume the download from where it left off
            logger.info(
                f"Resuming download of {file_name} "
                f"from {output_path.stat().st_size / 1024**2} MB "
                f"to {total_size / 1024**2} MB"
            )
            headers = {"Range": f"bytes={output_path.stat().st_size}-"}
            mode = "ab"

        with self.session.get(
            url, stream=True, headers=headers, timeout=20
        ) as response:
            # Do not continue the download if the request was unsuccessful
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size: int = 1024 * 1024  # 1MB
            with open(output_path, mode) as f:
                t = (
                    tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        bar_format=(
                            "{l_bar}{bar}| "
                            "{n_fmt:6}/{total_fmt:6} {rate_fmt:6}"
                        ),
                    )
                    if tqdm is not None
                    else None
                )
                count: int = 0
                for data in response.iter_content(block_size):
                    if t is not None:
                        t.update(len(data))
                    f.write(data)
                    if total_size != 0 and self.progress_bar is not None:
                        count += len(data)
                        self.progress_bar(
                            float(count) / float(total_size),
                            f"Downloading {file_name}",
                        )
                # tqdm 객체가 있으면 close 메서드를 호출합니다.
                if t is not None:
                    t.close()

    def check_model_files_by_sha256(self) -> bool:
        # Validate the checksums
        is_validated: bool = True
        output_folder: Path = self.output_folder
        for single_sha256 in self.hf_info["sha256"]:
            fname, fhash = single_sha256
            fpath = output_folder / Path(fname)

            if not fpath.exists():
                logger.info(f"The following file is missing: {fpath}")
                is_validated = False
                continue

            with open(output_folder / Path(fname), "rb") as f:
                real_hash = hashlib.sha256(f.read()).hexdigest()
                if real_hash != fhash:
                    logger.info(f"Checksum failed: {fname}  {fhash}")
                    is_validated = False
                else:
                    logger.info(f"Checksum validated: {fname}  {fhash}")

        if is_validated:
            logger.info("[+] Validated checksums of all model files!")
        else:
            logger.error(
                "[-] Invalid checksums. Rerun downloader with the --clean flag"
            )
        return is_validated

    def is_ggml(self, file_name: str) -> bool:
        return (
            self._ggml_pattern.match(file_name) is not None
            and self._pytorch_pattern.match(file_name) is None
        )

    def _start_download_threads(self, links: List[str]) -> None:
        if links is None:
            links = self.hf_info["links"]
        if thread_map is not None:
            thread_map(
                lambda url: self.get_single_file(url),
                links,
                max_workers=min(self.threads, len(links)),
                disable=True,
            )
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(
                max_workers=min(self.threads, len(links))
            ) as executor:
                [
                    future.result()
                    for future in as_completed(
                        [
                            executor.submit(self.get_single_file, url)
                            for url in links
                        ]
                    )
                ]

    def _get_model_info_from_huggingface(self) -> HuggingfaceModelInfo:
        model, branch = self.model, self.branch
        page: str = f"/api/models/{model}/tree/{branch}"
        cursor: bytes = b""

        links: List[str] = []
        file_names: List[str] = []
        sha256: List[List[str]] = []
        classifications: List[Classification] = []
        has_pytorch: bool = False
        has_pt: bool = False
        has_ggml: bool = False
        has_safetensors: bool = False
        is_lora: bool = False
        while True:
            url: str = f"{self._base_url}{page}" + (
                f"?cursor={cursor.decode()}" if cursor else ""
            )
            response: Response = self.session.get(url, timeout=20)
            response.raise_for_status()
            content: bytes = response.content

            json_decoded: dict = json.loads(content)
            if not json_decoded:
                break

            for json_idx in range(len(json_decoded)):
                file_name: str = json_decoded[json_idx]["path"]
                file_names.append(file_name)
                if file_name.endswith(
                    ("adapter_config.json", "adapter_model.bin")
                ):
                    is_lora = True

                (
                    is_pytorch,
                    is_safetensors,
                    is_pt,
                    is_possibly_ggml,
                    is_tokenizer,
                    is_text,
                ) = (
                    self._pytorch_pattern.match(file_name),
                    self._safetensors_pattern.match(file_name),
                    self._pt_pattern.match(file_name),
                    self._ggml_pattern.match(file_name),
                    self._tokenizer_pattern.match(file_name),
                    self._text_pattern.match(file_name),
                )

                if is_text is None:
                    is_text = is_tokenizer
                if any(
                    (
                        is_pytorch,
                        is_safetensors,
                        is_pt,
                        is_possibly_ggml,
                        is_tokenizer,
                        is_text,
                    )
                ):
                    if "lfs" in json_decoded[json_idx]:
                        sha256.append(
                            [file_name, json_decoded[json_idx]["lfs"]["oid"]]
                        )

                    if is_text:
                        links.append(
                            f"https://huggingface.co/{model}/resolve/{branch}/{file_name}"  # noqa: E501
                        )
                        classifications.append("text")
                        continue

                    if not self.text_only:
                        links.append(
                            f"https://huggingface.co/{model}/resolve/{branch}/{file_name}"  # noqa: E501
                        )
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append("safetensors")
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append("pytorch")
                        elif is_pt:
                            has_pt = True
                            classifications.append("pt")
                        elif is_possibly_ggml and not is_pytorch:
                            has_ggml = True  # noqa: F841
                            classifications.append("ggml")

            cursor = base64.b64encode(
                (
                    base64.b64encode(
                        f'{{"file_name":"{json_decoded[-1]["path"]}"}}'.encode()  # noqa: E501
                    )
                    + b":50"
                )
            ).replace(b"=", b"%3D")

        # If both pytorch and safetensors are available,
        # download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for json_idx in range(len(classifications) - 1, -1, -1):
                if classifications[json_idx] in ("pytorch", "pt"):
                    links.pop(json_idx)
        return HuggingfaceModelInfo(
            links=links,
            file_names=file_names,
            sha256=sha256,
            is_lora=is_lora,
            classifications=classifications,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        default=None,
        nargs="?",
        help="The model you'd like to download. e.g. facebook/opt-1.3b",
    )

    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Name of the Git branch to download from.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of files to download simultaneously.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only download text files (txt/json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The folder where the model should be saved.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Does not resume the previous download.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validates the checksums of model files.",
    )
    args = parser.parse_args()

    if args.model is None:
        parser.error(
            "Error: Please specify the model you'd like to download "
            "(e.g. 'python download-model.py facebook/opt-1.3b')."
        )

    HuggingfaceDownloader.from_repository(
        model=args.model,
        branch=args.branch,
        threads=args.threads,
        base_folder=args.output,
        clean=args.clean,
        check=args.check,
        text_only=args.text_only,
    )
