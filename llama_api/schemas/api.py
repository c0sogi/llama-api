# flake8: noqa

from sys import version_info
from typing import Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import Field
from pydantic.main import BaseModel
from typing_extensions import TypedDict

# If python version >= 3.11, use the built-in NotRequired type.
# Otherwise, import it from typing_extensi
if version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

JsonTypes = Union[int, float, str, bool, dict, list, None]


class FunctionCompletion(TypedDict):
    name: str
    arguments: NotRequired[str]


class FunctionCompletionChunk(TypedDict):
    name: NotRequired[str]
    arguments: NotRequired[str]


class FunctionParameter(TypedDict):
    type: str
    description: NotRequired[str]
    enum: NotRequired[List[JsonTypes]]


class FunctionParameters(TypedDict):
    type: Literal["object"]
    properties: Dict[str, FunctionParameter]
    required: NotRequired[List[str]]


class FunctionSchema(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: FunctionParameters


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(TypedDict):
    index: int
    object: str
    embedding: List[float]


class Embedding(TypedDict):
    object: Literal["list"]
    model: str
    data: List[EmbeddingData]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]


class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionMessage(TypedDict):
    role: Literal["assistant", "user", "system"]
    content: Optional[str]
    user: NotRequired[str]
    function_call: NotRequired[FunctionCompletion]


class ChatCompletionChoice(TypedDict):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str]


class ChatCompletion(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


class ChatCompletionChunkDelta(TypedDict):
    role: NotRequired[Literal["assistant"]]
    content: NotRequired[str]
    function_call: NotRequired[FunctionCompletionChunk]


class ChatCompletionChunkChoice(TypedDict):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str]


class ChatCompletionChunk(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionChunkChoice]


class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]


class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


class APIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        default="user",
        description="The role of the messages author. One of system, user, assistant, or function",
    )
    content: str = Field(
        default="",
        description=(
            "The contents of the message. content is required for all messages, "
            "and may be null for assistant messages with function calls."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the author of this message. name is required if role is function, "
            "and it should be the name of the function whose response is in the content. "
            "May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters."
        ),
    )
    function_call: Optional[FunctionSchema] = Field(
        default=None,
        description="The name and arguments of a function that should be called, as generated by the model.",
    )

    class Config:
        frozen = True


class TextGenerationSettings(BaseModel):
    completion_id: str = Field(
        default_factory=lambda: f"cmpl-{str(uuid4())}",
        description="The unique ID of the text generation",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description=(
            "Adjust the randomness of the generated text."
            "Temperature is a hyperparameter that controls the randomness of the generated te"
            "xt. It affects the probability distribution of the model's output tokens. A high"
            "er temperature (e.g., 1.5) makes the output more random and creative, while a lo"
            "wer temperature (e.g., 0.5) makes the output more focused, deterministic, and co"
            "nservative. The default value is 0.8, which provides a balance between randomnes"
            "s and determinism. At the extreme, a temperature of 0 will always pick the most "
            "likely next token, leading to identical outputs in each run."
        ),
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Limit the next token selection to a subset of tokens with a cumulative probabili"
            "ty above a threshold P. Top-p sampling, also known as nucleus sampling, "
            "is another text generation method that selects the next token from a subset of t"
            "okens that together have a cumulative probability of at least p. This method pro"
            "vides a balance between diversity and quality by considering both the probabilit"
            "ies of tokens and the number of tokens to sample from. A higher value for top_p "
            "(e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) wil"
            "l generate more focused and conservative text."
        ),
    )
    typical_p: float = Field(
        default=0.0,
        description="Locally typical sampling threshold, 0.0 to disable typical sampling",
    )
    logprobs: Optional[int] = Field(
        default=None,
        description="The number of logprobs to return. If None, no logprobs are returned.",
    )
    echo: bool = Field(
        default=False,
        description="If True, the input is echoed back in the output.",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on their existing frequency in the tex"
            "t so far, decreasing the model's likelihood to repeat the same line verbatim."
        ),
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Positive values penalize new tokens based on whether they appear in the text so far, increasing "
            "the model's likelihood to talk about new topics."
        ),
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        description=(
            "A penalty applied to each token that is already generated. This helps prevent th"
            "e model from repeating itself. Repeat penalty is a hyperparameter used t"
            "o penalize the repetition of token sequences during text generation. It helps pr"
            "event the model from generating repetitive or monotonous text. A higher value (e"
            ".g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0."
            "9) will be more lenient."
        ),
    )
    repetition_penalty_range: int = Field(
        default=0,
        ge=0,
        description=(
            "The number of most recent tokens to consider for repetition penalty. 0 makes all tokens be used."
        ),
    )
    top_k: int = Field(
        default=40,
        ge=0,
        description=(
            "Limit the next token selection to the K most probable tokens. Top-k samp"
            "ling is a text generation method that selects the next token only from the top k"
            " most likely tokens predicted by the model. It helps reduce the risk of generati"
            "ng low-probability or nonsensical tokens, but it may also limit the diversity of"
            " the output. A higher value for top_k (e.g., 100) will consider more tokens and "
            "lead to more diverse text, while a lower value (e.g., 10) will focus on the most"
            " probable tokens and generate more conservative text."
        ),
    )
    tfs_z: float = Field(
        default=1.0,
        description="Modify probability distribution to carefully cut off least likely tokens",
    )
    mirostat_mode: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
    )
    mirostat_tau: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description=(
            "Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, "
            "larger values produce more diverse and less coherent text"
        ),
    )
    mirostat_eta: float = Field(
        default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "A dictionary of logit bias values to use for each token. The keys are the tokens"
            " and the values are the bias values. The bias values are added to the logits of "
            "the model to influence the next token probabilities. For example, a bias value o"
            "f 5.0 will make the model 10 times more likely to select that token than it woul"
            "d be otherwise. A bias value of -5.0 will make the model 10 times less likely to"
            " select that token than it would be otherwise. The bias values are added to the "
            "logits of the model to influence."
        ),
    )
    ban_eos_token: bool = Field(
        default=False,
        description="If True, the EOS token is banned from being generated.",
    )
    muse: bool = Field(
        default=False,
        description="Use Muse logit processor (experimental). "
        "Muse logit processor performs dampening of the k highest probability elements.",
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=1.0,
        description="The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`. "
        "Higher guidance scale encourages the model to generate samples that are more closely linked to the input "
        "prompt, usually at the expense of poorer quality",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="The negative prompt for classifier free guidance (CFG). "
        "The negative prompt is used to encourage the model not to generate samples that are too similar to the "
        "negative prompt. CFG is enabled by setting `guidance_scale > 1`.",
    )
    is_openai: bool = Field(
        default=False,
        description="If True, the model is regarded as an OpenAI model.",
    )
    grammar: Optional[str] = Field(
        default=None,
        description="The BNF grammar to use for the model. Only used for llama.cpp models.",
    )

    def __init__(self, **kwargs):
        super().__init__(
            **{k: v for k, v in kwargs.items() if v is not None}
        )


class CreateEmbeddingRequest(BaseModel):
    model: str = Field(description="The model to use for embedding.")
    input: Union[str, List[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None, description="Not in use.")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama_7b",
                "input": "The food was delicious and the waiter...",
            },
        }


class CreateCompletionRequest(TextGenerationSettings):
    model: str = Field(
        default=..., description="The model to use for completion."
    )
    prompt: str = Field(
        default="", description="The prompt to use for completion."
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama_7b",
                "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                "stop": ["\n", "###"],
            }
        }


class CreateChatCompletionRequest(TextGenerationSettings):
    completion_id: str = Field(
        default_factory=lambda: f"chatcmpl-{str(uuid4())}",
        description="The unique ID of the chat generation",
    )
    model: str = Field(
        default=..., description="The model to use for completion."
    )
    messages: List[APIChatMessage] = Field(
        default=[],
        description="A list of messages to generate completions for.",
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response."
    )
    functions: Optional[List[FunctionSchema]] = Field(
        default=None, description="The functions to invoke."
    )
    function_call: Optional[
        Union[FunctionCompletion, str, Literal["auto", "none"]]
    ] = Field(default=None, description="The function call to invoke.")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama_7b",
                "messages": [
                    APIChatMessage(
                        role="system", content="You are a helpful assistant."
                    ),
                    APIChatMessage(
                        role="user", content="What is the capital of France?"
                    ),
                ],
            }
        }
