from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Literal, Optional

from ..modules.base import BaseLLMModel
from ..shared.config import MainCliArgs
from ..utils.path import path_resolver


@dataclass
class LlamaCppModel(BaseLLMModel):
    """Llama.cpp model that can be loaded from local path."""

    n_parts: int = field(
        default=-1,
        metadata={
            "description": "Number of parts to split the model into. If -1, "
            "the number of parts is automatically determined."
        },
    )
    n_gpu_layers: int = field(
        default=30,
        metadata={
            "description": "Number of layers to keep on the GPU. "
            "If 0, all layers are kept on the GPU."
        },
    )
    seed: int = field(
        default=-1,
        metadata={"description": "Seed. If -1, a random seed is used."},
    )
    f16_kv: bool = field(
        default=True,
        metadata={"description": "Use half-precision for key/value cache."},
    )
    logits_all: bool = field(
        default=False,
        metadata={
            "description": "Return logits for all tokens, "
            "not just the last token."
        },
    )
    vocab_only: bool = field(
        default=False,
        metadata={"description": "Only load the vocabulary, no weights."},
    )
    use_mlock: bool = field(
        default=True,
        metadata={"description": "Force system to keep model in RAM."},
    )
    n_batch: int = field(
        default=512,
        metadata={
            "description": "Number of tokens to process in parallel. "
            "Should be a number between 1 and n_ctx."
        },
    )
    last_n_tokens_size: int = field(
        default=64,
        metadata={
            "description": "The number of tokens to look back "
            "when applying the repeat_penalty."
        },
    )
    use_mmap: bool = True  # Whether to use memory mapping for the model.
    cache: bool = (
        False  # The size of the cache in bytes. Only used if cache is True.
    )
    verbose: bool = True  # Whether to echo the prompt.
    echo: bool = True  # Compatibility of verbose.
    lora_base: Optional[str] = None  # The path to the Llama LoRA base model.
    lora_path: Optional[
        str
    ] = None  # The path to the Llama LoRA. If None, no LoRa is loaded.
    cache_type: Optional[Literal["disk", "ram"]] = "ram"
    cache_size: Optional[int] = (
        2 << 30
    )  # The size of the cache in bytes. Only used if cache is True.
    n_threads: Optional[int] = field(
        default=None,
        metadata={
            "description": "Number of threads to use. "
            "If None, the number of threads is automatically determined."
        },
    )
    low_vram: bool = False  # Whether to use less VRAM.
    embedding: bool = False  # Whether to use the embedding layer.

    # Refer: https://github.com/ggerganov/llama.cpp/pull/2054
    rope_freq_base: float = 10000.0  # I use 26000 for n_ctx=4096.
    rope_freq_scale: float = 1.0  # Generally, 2048 / n_ctx.
    n_gqa: Optional[int] = None  # TEMPORARY: Set to 8 for Llama2 70B
    rms_norm_eps: Optional[float] = None  # TEMPORARY
    mul_mat_q: Optional[bool] = None  # TEMPORARY

    def __post_init__(self) -> None:
        """Calculate the rope_freq_base based on the n_ctx.
        Assume that the trained token length is 4096."""
        if self.rope_freq_base == 10000.0:
            self.rope_freq_base = self.calculate_rope_freq()
        if self.rope_freq_scale == 1.0:
            self.rope_freq_scale = self.calculate_rope_scale()

    @cached_property
    def model_path_resolved(self) -> str:
        return path_resolver(
            self.model_path,
            default_model_directory=MainCliArgs.model_dir.value,
        )


@dataclass
class ExllamaModel(BaseLLMModel):
    """Exllama model that can be loaded from local path."""

    version: Literal[1, 2] = field(
        default=1,
        metadata={
            "description": "Version of the exllama model. "
            "Currently version 1 and 2 are supported."
        },
    )

    compress_pos_emb: float = field(
        default=1.0,
        metadata={
            "description": "Increase to compress positional embeddings "
            "applied to sequence. This is useful when you want to "
            "extend context window size. e.g. If you want to extend context "
            "window size from 2048 to 4096, set this to 2.0."
        },
    )
    alpha_value: Optional[float] = field(
        default=None,
        metadata={
            "description": "Positional embeddings alpha factor for "
            "NTK RoPE scaling. Use either this or compress_pos_emb, "
            "not both at the same time."
        },
    )
    gpu_peer_fix: bool = field(
        default=False,
        metadata={
            "description": "Apparently Torch can have problems transferring "
            "tensors directly 1 GPU to another. Enable this to use system "
            "RAM as a buffer for GPU to GPU transfers."
        },
    )
    auto_map: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "List of floats with memory allocation in GB, "
            "per CUDA device, overrides device_map."
        },
    )

    # Optional parameters for tuning
    use_flash_attn_2: bool = False
    matmul_recons_thd: int = 8
    fused_mlp_thd: int = 2
    sdp_thd: int = 8
    fused_attn: bool = True
    matmul_fused_remap: bool = False
    rmsnorm_no_half2: bool = False
    rope_no_half2: bool = False
    matmul_no_half2: bool = False
    silu_no_half2: bool = False
    concurrent_streams: bool = False

    def __post_init__(self) -> None:
        """Calculate the rope_freq_base based on the n_ctx.
        Assume that the trained token length is 4096."""
        if self.alpha_value is None:
            self.alpha_value = self.calculate_rope_alpha()
        if self.compress_pos_emb == 1.0:
            self.compress_pos_emb = self.calculate_rope_compress_ratio()

    @cached_property
    def model_path_resolved(self) -> str:
        return path_resolver(
            self.model_path,
            default_model_directory=MainCliArgs.model_dir.value,
        )


@dataclass
class ReverseProxyModel(BaseLLMModel):
    """A model that can be directed to other API.
    Ignore all the parameters except model path!
    The model path is the base URL(host) of the API."""

    model_path: str = (
        "https://api.openai.com"  # The base URL(host) of the API.
    )
    max_total_tokens: int = field(init=False, default=-1)
    instruction_template: Optional[str] = field(init=False, default=None)
    auto_truncate: bool = field(init=False, default=False)
