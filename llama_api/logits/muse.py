# flake8: noqa
from typing import TYPE_CHECKING, List, Tuple

from .base import BaseLogitProcessor

if TYPE_CHECKING:
    import torch as pytorch


class MuseLogitProcessor(BaseLogitProcessor):
    """Performs dampening of the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        damp (`float`, *optional*, defaults to 0.98):
            How much less likely should the top_k most likely tokens be made. If set to 0, they become impossible.
    """

    def __init__(
        self,
        top_k: int = 3,
        damp: float = 0.9,
        damp_initial: float = 1.0,
        damp_ramp_tokens: int = 32,
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                "`top_k` has to be a strictly positive integer, "
                f"but is {top_k}"
            )

        self.top_k = max(top_k, min_tokens_to_keep)
        self.damp = damp
        self.damp_initial = damp_initial
        self.damp_ramp_tokens = damp_ramp_tokens
        self.token_num = 0

    def with_torch(
        self, input_ids: "pytorch.Tensor", scores: "pytorch.Tensor"
    ) -> "pytorch.Tensor":
        import torch

        top_k_safety = min(self.top_k, scores.size(-1))  # Safety check
        linear_damp = self.linear_damp
        topk_values, topk_indices = torch.topk(
            scores, top_k_safety, dim=-1
        )  # Specify the dimension
        self.token_num += 1
        return scores.scatter_(-1, topk_indices, topk_values * linear_damp)

    def without_torch(
        self, input_ids: List[int], scores: List[float]
    ) -> List[float]:
        top_k_safety = min(self.top_k, len(scores))  # Safety check
        linear_damp = self.linear_damp
        topk_values_indices = sorted(
            range(len(scores)), key=lambda x: scores[x], reverse=True
        )[:top_k_safety]
        self.token_num += 1
        return [
            score * linear_damp if idx in topk_values_indices else score
            for idx, score in enumerate(scores)
        ]

    @property
    def linear_damp(self) -> float:
        ratio = (
            1.0
            if self.damp_ramp_tokens == 0
            else min(self.token_num / self.damp_ramp_tokens, 1.0)
        )
        return (
            self.damp_initial + ratio * (self.damp - self.damp_initial)
            if ratio < 1.0
            else self.damp
        )
