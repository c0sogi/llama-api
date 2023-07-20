from llama_api.common.templates import ChatTurnTemplates, DescriptionTemplates
from llama_api.modules.base import UserChatRoles
from llama_api.utils.path import suppress_import_error

with suppress_import_error():
    from llama_api.modules.llama_cpp import LlamaCppModel, LlamaCppTokenizer

    # ================== LLaMA.cpp models ================== #
    orca_mini_3b = LlamaCppModel(
        model_path="orca-mini-3b.ggmlv3.q4_1.bin",  # model_path here
        max_total_tokens=4096,
        rope_freq_base=26000,
        rope_freq_scale=0.5,
        tokenizer=LlamaCppTokenizer(
            "psmathur/orca_mini_3b"
        ),  # Huggingface repo here (the repo must contain `tokenizer.model`)
        prefix_template=DescriptionTemplates.USER_AI__DEFAULT,
        chat_turn_prompt=ChatTurnTemplates.ROLE_CONTENT_2,
        user_chat_roles=UserChatRoles(
            user="User",
            ai="Response",
            system="System",
        ),
    )

with suppress_import_error():
    from llama_api.modules.exllama import ExllamaModel, ExllamaTokenizer

    # ================== ExLLaMa models ================== #
    orca_mini_7b = ExllamaModel(
        model_path="orca_mini_7b",  # model_path here
        max_total_tokens=4096,
        compress_pos_emb=2.0,
        tokenizer=ExllamaTokenizer("orca_mini_7b"),  # model_path here
        prefix_template=DescriptionTemplates.USER_AI__DEFAULT,
        chat_turn_prompt=ChatTurnTemplates.ROLE_CONTENT_2,
        user_chat_roles=UserChatRoles(
            user="User",
            ai="Response",
            system="System",
        ),
    )
