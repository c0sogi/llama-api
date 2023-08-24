from typing import Dict
from llama_api.schemas.models import ExllamaModel, LlamaCppModel

# ================== LLaMA.cpp models ================== #
orca_mini_3b = LlamaCppModel(
    model_path="orca-mini-3b.ggmlv3.q4_0.bin",  # model_path here
    max_total_tokens=4096,
    rope_freq_base=26000,
    rope_freq_scale=0.5,
)
mythomax_l2_13b_ggml = LlamaCppModel(
    model_path="TheBloke/MythoMax-L2-13B-GGML",
    max_total_tokens=4096,
    n_gpu_layers=100,
    mul_mat_q=True,
    n_batch=4096,
)
frankensteins_monster_13b_ggml = LlamaCppModel(
    model_path="Blackroot_FrankensteinsMonster-13B-ggml-Q3_K_M.bin",
    max_total_tokens=4096,
    n_gpu_layers=100,
    mul_mat_q=True,
    n_batch=4096,
)


# ================== ExLLaMa models ================== #
orca_mini_7b = ExllamaModel(
    model_path="orca_mini_7b",  # model_path here
    max_total_tokens=4096,
    compress_pos_emb=2.0,
)
# stable_beluga_13b = ExllamaModel(
#     model_path="stable_beluga_13b",  # model_path here
#     max_total_tokens=4096,
# )
chronos_hermes_13b_v2 = ExllamaModel(
    model_path="Austism/chronos-hermes-13b-v2-GPTQ",  # model_path here
    max_total_tokens=4096,
)
# mythologic_mini_7b = ExllamaModel(
#     model_path="TheBloke/MythoLogic-Mini-7B-GPTQ",  # model_path here
#     max_total_tokens=8192,
#     compress_pos_emb=2.0,
# )
mythomax_l2_13b_gptq = ExllamaModel(
    model_path="TheBloke/MythoMax-L2-13B-GPTQ",
    max_total_tokens=4096,
)

# Define a mapping from OpenAI model names to LLaMA models.
# e.g. If you request API model "gpt-3.5-turbo",
# the API will load the LLaMA model "orca_mini_3b"
openai_replacement_models: Dict[str, str] = {
    "gpt-3.5-turbo": "mythomax_l2_13b_ggml",
    "gpt-4": "mythomax_l2_13b_gptq",
}
