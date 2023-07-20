from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import tensorflow_hub as hub

from ..utils.logger import ApiLogger
from .base import BaseEmbeddingGenerator

if TYPE_CHECKING:
    from tensorflow.python.framework.ops import Tensor

logger = ApiLogger(__name__)


class SentenceEncoderEmbeddingGenerator(BaseEmbeddingGenerator):
    """Generate embeddings using a sentence encoder model,
    automatically downloading the model from https://tfhub.dev/"""

    base_url: str = "https://tfhub.dev/google/"
    model: Optional[Callable[[list[str]], "Tensor"]] = None
    _model_name: Optional[str] = None

    def __del__(self) -> None:
        if self.model is not None:
            getattr(self.model, "__del__", lambda: None)()
            del self.model
            self.model = None
            logger.info("🗑️ SentenceEncoderEmbedding deleted!")

    @classmethod
    def from_pretrained(
        cls, model_name: str
    ) -> "SentenceEncoderEmbeddingGenerator":
        self = cls()
        self._model_name = model_name
        url = f"{self.base_url.rstrip('/')}/{model_name.lstrip('/')}"
        self.model = hub.load(url)  # type: ignore
        logger.info(f"🤖 TFHub {model_name} loaded!")
        return self

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 100,
        **kwargs,
    ) -> list[list[float]]:
        assert self.model is not None, "Please load the model first."
        embeddings: list["Tensor"] = []
        for batch_idx_start in range(0, len(texts), batch_size):
            batch_idx_end = batch_idx_start + batch_size
            batch_texts = texts[batch_idx_start:batch_idx_end]
            embeddings.append(self.model(batch_texts))
        return np.vstack(embeddings).tolist()

    @property
    def model_name(self) -> str:
        return self._model_name or self.__class__.__name__
