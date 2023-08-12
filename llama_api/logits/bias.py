from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional

from .base import BaseLogitProcessor

if TYPE_CHECKING:
    import torch as pytorch


class LogitBiasProcessor(BaseLogitProcessor):
    """Create a logit bias processor to bias the logit scores."""

    def __init__(
        self,
        logit_bias: Dict[str, float],
        logit_bias_type: Optional[Literal["input_ids", "tokens"]],
        encoder: Callable[[str], List[int]],
    ):
        if logit_bias_type is None:
            logit_bias_type = "input_ids"

        to_bias = {}  # type: Dict[int, float]
        if logit_bias_type == "input_ids":
            for input_id_string, score in logit_bias.items():
                to_bias[int(input_id_string)] = score

        elif logit_bias_type == "tokens":
            for token, score in logit_bias.items():
                for input_id in encoder(token):
                    to_bias[input_id] = score

        self._to_bias = to_bias
        self._bias_tensor = None

    def _get_bias_tensor(self, scores: "pytorch.Tensor") -> "pytorch.Tensor":
        if self._bias_tensor is None:
            import torch

            self._bias_tensor = torch.zeros(
                scores.shape[-1], dtype=scores.dtype, device=scores.device
            )
            for idx, value in self._to_bias.items():
                self._bias_tensor[idx] = value

        return self._bias_tensor

    def with_torch(
        self, input_ids: "pytorch.Tensor", scores: "pytorch.Tensor"
    ) -> "pytorch.Tensor":
        return scores + self._get_bias_tensor(scores)

    def without_torch(
        self, input_ids: List[int], scores: List[float]
    ) -> List[float]:
        for id, biased_score in self._to_bias.items():
            scores[id] += biased_score
        return scores
