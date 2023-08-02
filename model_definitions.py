from typing import Dict
from llama_api.schemas.models import ExllamaModel, LlamaCppModel

# ================== LLaMA.cpp models ================== #
orca_mini_3b = LlamaCppModel(
    model_path="orca-mini-3b.ggmlv3.q4_1.bin",  # model_path here
    max_total_tokens=4096,
    rope_freq_base=26000,
    rope_freq_scale=0.5,
)
llama2_13b_chat = LlamaCppModel(
    model_path="llama-2-13b-chat.ggmlv3.q4_K_M.bin",
    max_total_tokens=4096,
)
stable_beluga_7b = LlamaCppModel(
    model_path="TheBloke/StableBeluga-7B-GGML",
    max_total_tokens=4096,
)


# ================== ExLLaMa models ================== #
orca_mini_7b = ExllamaModel(
    model_path="orca_mini_7b",  # model_path here
    max_total_tokens=4096,
    compress_pos_emb=2.0,
)
stable_beluga_13b = ExllamaModel(
    model_path="stable_beluga_13b",  # model_path here
    max_total_tokens=4096,
)
nous_hermes_llama_2_13b = ExllamaModel(
    model_path="nous_hermes_llama_2_13b",  # model_path here
    max_total_tokens=8192,
    compress_pos_emb=2.0,
)
mythologic_mini_7b = ExllamaModel(
    model_path="TheBloke/MythoLogic-Mini-7B-GPTQ",  # model_path here
    max_total_tokens=8192,
    compress_pos_emb=2.0,
)


# Define a mapping from OpenAI model names to LLaMA models.
# e.g. If you request API model "gpt-3.5-turbo",
# the API will load the LLaMA model "orca_mini_3b"
openai_replacement_models: Dict[str, str] = {
    "gpt-3.5-turbo": "stable_beluga_13b",
    "gpt-4": "nous_hermes_llama_2_13b",
}
