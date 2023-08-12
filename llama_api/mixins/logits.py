from typing import Callable, List

from ..logits.base import BaseLogitProcessor
from ..logits.bias import LogitBiasProcessor
from ..logits.muse import MuseLogitProcessor
from ..schemas.api import TextGenerationSettings


class LogitsMixin:
    @staticmethod
    def get_logit_processors(
        settings: TextGenerationSettings, encoder: Callable[[str], List[int]]
    ) -> List[BaseLogitProcessor]:
        logit_processors: List[BaseLogitProcessor] = []
        if settings.muse:
            logit_processors.append(
                MuseLogitProcessor(
                    top_k=3,
                    damp=0.9,
                    damp_initial=1.0,
                    damp_ramp_tokens=32,
                    min_tokens_to_keep=1,
                )
            )
        if settings.logit_bias is not None:
            logit_processors.insert(
                0,
                LogitBiasProcessor(
                    logit_bias=settings.logit_bias,
                    logit_bias_type=settings.logit_bias_type,
                    encoder=encoder,
                ),
            )
        return logit_processors
