# flake8: noqa
import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import transformers.models.llama.modeling_llama
from xformers.ops import memory_efficient_attention, LowerTriangularMask
from torch import Tensor, cat, finfo, float32, matmul, softmax, tensor

from ..utils.logger import ApiLogger

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaAttention


logger = ApiLogger(__name__)


def hijack_attention_forward():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = _forward
    logger.info(f"Replaced attention forward with {__name__.split('.')[-1]}")


def _forward(
    self: "LlamaAttention",
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_value: Optional[Tuple[Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
    # COPY: oobabooga/text-generation-webui/modules/llama_attn_hijack.py
    logger.info(f"Using {__name__.split('.')[-1]}")
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    (
        query_states,
        key_states,
    ) = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = cat([past_key_value[0], key_states], dim=2)
        value_states = cat([past_key_value[1], value_states], dim=2)  # type: ignore

    past_key_value = (key_states, value_states) if use_cache else None  # type: ignore

    # We only apply xformers optimizations if we don't need to output the whole attention matrix
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=None
            )
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=LowerTriangularMask(),
            )
        attn_weights = None
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, tensor(finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = softmax(attn_weights, dim=-1, dtype=float32).to(
            query_states.dtype
        )
        attn_output = matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

    return (
        self.o_proj(attn_output.reshape(bsz, q_len, self.hidden_size)),
        attn_weights,
        past_key_value,
    )
