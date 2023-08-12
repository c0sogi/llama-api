from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch as pytorch


class BaseLogitProcessor(ABC):
    @abstractmethod
    def with_torch(
        self, input_ids: "pytorch.Tensor", scores: "pytorch.Tensor"
    ) -> "pytorch.Tensor":
        """Process logits with PyTorch tensors."""

    @abstractmethod
    def without_torch(
        self, input_ids: List[int], scores: List[float]
    ) -> List[float]:
        """Process logits with Python lists."""
