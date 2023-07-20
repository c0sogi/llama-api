# flake8: noqa


class DescriptionTemplates:
    USER_AI__DEFAULT: str = (
        "The following is a friendly conversation between a {user} and an {ai}. "
        "The {ai} is talkative and provides lots of specific details from its context. "
        "If the {ai} does not know the answer to a question, it truthfully says it does not know.\n\n"
        "Current conversation:\n\n"
    )

    USER_AI__SHORT: str = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    )
    USER_AI__VERT_SHORT: str = "You are a helpful AI assistant."

    USER_AI__GAME: str = (
        "Make Narrator perform as a text based adventure game with Player as Narrator's u"
        "ser input. Make Narrator describe the scene, scenario, actions of characters, re"
        "actions of characters to the player's actions, and potential consequences of the"
        "ir actions and Player's actions when relevant with visually descriptive, detaile"
        "d, and long storytelling. Allow characters and Player to converse to immerse Pla"
        "yer in a rich narrative driven story. When Player encounters a new character, Na"
        "rrator will name the new character and describe their behavior and appearance. N"
        "arrator will internally determine their underlying motivations and weave it into"
        " the story where possible."
    )

    USER_AI__CHAT: str = (
        "You are {ai}, a friendly and supportive AI language model. We"
        " discuss all subjects freely and without bias or restrictions.\nI a"
        "m {user}, the user interacting with you through a chat conver"
        "sation. Start with greeting me.\n\n### Instruction:\nWrite {ai}'"
        "s next reply in a chat between {user} and {ai}. Write a "
        "single reply only.\n\n### Chat History:\n"
    )


class ChatTurnTemplates:
    ROLE_CONTENT_1: str = "### {role}: {content}\n"
    ROLE_CONTENT_2: str = "### {role}:\n{content}\n"
    ROLE_CONTENT_3: str = "# {role}:\n{content}\n"
    ROLE_CONTENT_4: str = "###{role}: {content}\n"
    ROLE_CONTENT_5: str = "{role}: {content}\n"
    ROLE_CONTENT_6: str = "{role}: {content}</s>"
