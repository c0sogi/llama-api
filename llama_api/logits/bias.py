from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
)

from ..utils.logger import ApiLogger
from .base import BaseLogitProcessor

if TYPE_CHECKING:
    import torch as pytorch

logger = ApiLogger(__name__)

try:
    import tiktoken

    openai_decoder = tiktoken.get_encoding("cl100k_base").decode
except Exception as e:
    logger.warning(
        "Could not load tiktoken, which is required for OpenAI GPT models. "
        f"Please `pip install tiktoken` to use the OpenAI encoder: {e}"
    )
    openai_decoder: Optional[Callable[[List[int]], str]] = None


class LogitBiasProcessor(BaseLogitProcessor):
    """Create a logit bias processor to bias the logit scores."""

    def __init__(
        self,
        logit_bias: Dict[str, float],
        encoder: Callable[[str], List[int]],
        is_openai: bool = False,
    ):
        """Create a logit bias processor to bias the logit scores."""

        global openai_decoder

        biases = {}  # type: Dict[int, float]
        for id_or_token, bias in logit_bias.items():
            is_digit = id_or_token.isdigit()

            if is_digit and is_openai and openai_decoder is not None:
                # If we have an OpenAI id, we need to convert it to a token
                # and then encode the token to get the ids
                for id in encoder(openai_decoder([int(id_or_token)])):
                    if abs(bias) > abs(biases.get(id, 0.0)):
                        biases[id] = bias
            elif is_digit:
                # If we have a digit, we can just use it directly
                biases[int(id_or_token)] = bias
            else:
                # Otherwise, we need to encode the token and use the ids
                for id in encoder(id_or_token):
                    if abs(bias) > abs(biases.get(id, 0.0)):
                        biases[id] = bias

        self._biases = biases
        self._bias_tensor = None

    def _get_bias_tensor(self, scores: "pytorch.Tensor") -> "pytorch.Tensor":
        if self._bias_tensor is None:
            import torch

            self._bias_tensor = torch.zeros(
                scores.shape[-1], dtype=scores.dtype, device=scores.device
            )
            for id, bias in self._biases.items():
                self._bias_tensor[id] = bias

        return self._bias_tensor

    def with_torch(
        self, input_ids: "pytorch.Tensor", scores: "pytorch.Tensor"
    ) -> "pytorch.Tensor":
        return scores + self._get_bias_tensor(scores)

    def without_torch(
        self, input_ids: List[int], scores: List[float]
    ) -> List[float]:
        for id, bias in self._biases.items():
            scores[id] += bias
        return scores
