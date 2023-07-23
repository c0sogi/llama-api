from ..utils.logger import ApiLogger

logger = ApiLogger(__name__)


# Importing llama.cpp
try:
    from ..modules.llama_cpp import LlamaCppCompletionGenerator, LlamaCppModel

    logger.info("🦙 Successfully imported llama.cpp module!")
except Exception as e:
    logger.error("Llama.cpp import error: " + str(e))
    LlamaCppCompletionGenerator = LlamaCppModel = str(
        e
    )  # Import error message


# Importing exllama
try:
    from ..modules.exllama import ExllamaCompletionGenerator, ExllamaModel

    logger.info("🦙 Successfully imported exllama module!")
except Exception as e:
    logger.exception("Exllama package import error: " + str(e))
    ExllamaCompletionGenerator = ExllamaModel = str(e)  # Import error message


# Importing embeddings (Pytorch + Transformer)
try:
    from ..modules.transformer import TransformerEmbeddingGenerator

    logger.info(
        "🦙 Successfully imported embeddings(Pytorch + Transformer) module!"
    )
except Exception as e:
    logger.error("Transformer embedding import error: " + str(e))
    TransformerEmbeddingGenerator = str(e)  # Import error message


# Importing embeddings (Tensorflow + Sentence Encoder)
try:
    from ..modules.sentence_encoder import SentenceEncoderEmbeddingGenerator

    logger.info("🦙 Successfully imported embeddings(Sentence Encoder) module!")
except Exception as e:
    logger.error("Sentence Encoder embedding import error: " + str(e))
    SentenceEncoderEmbeddingGenerator = str(e)  # Import error message


__all__ = [
    "LlamaCppCompletionGenerator",
    "LlamaCppModel",
    "ExllamaCompletionGenerator",
    "ExllamaModel",
    "TransformerEmbeddingGenerator",
    "SentenceEncoderEmbeddingGenerator",
]
