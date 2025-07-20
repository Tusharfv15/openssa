"""
============================
GEMINI LANGUAGE MODELS (LMs)
============================
"""


from __future__ import annotations

from dataclasses import dataclass, field
import json
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

from loguru import logger
from openai import OpenAI
from llama_index.embeddings.openai.base import OpenAIEmbedding, OpenAIEmbeddingMode, OpenAIEmbeddingModelType
from llama_index.llms.openai.base import OpenAI as LlamaIndexOpenAILM

from .base import BaseLM
from .config import LMConfig

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from .base import LMChatHist


@dataclass
class GeminiLM(BaseLM):
    """Gemini LM using OpenAI-compatible interface."""

    client: OpenAI = field(init=False)

    def __post_init__(self):
        """Initialize OpenAI client for Gemini."""
        self.client: OpenAI = OpenAI(api_key=self.api_key, base_url=self.api_base)

    @classmethod
    def from_defaults(cls) -> GeminiLM:
        """Get Gemini LM instance with default parameters."""
        return cls(
            model=LMConfig.GEMINI_DEFAULT_MODEL, 
            api_key=LMConfig.GEMINI_API_KEY, 
            api_base=LMConfig.GEMINI_API_URL
        )

    def call(self, messages: LMChatHist, **kwargs) -> ChatCompletion:
        """Call Gemini LM API and return response object."""
        # Remove seed parameter as Gemini doesn't support it
        kwargs.pop('seed', None)
        
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=kwargs.pop('temperature', LMConfig.DEFAULT_TEMPERATURE),
            **kwargs
        )

    def get_response(self, prompt: str, history: LMChatHist | None = None, json_format: bool = False, **kwargs) -> str:
        """Call Gemini LM API and return response content."""
        messages: LMChatHist = history or []
        messages.append({'role': 'user', 'content': prompt})

        if json_format:
            kwargs['response_format'] = {'type': 'json_object'}

            while True:
                try:
                    return json.loads(response := self.call(messages, **kwargs).choices[0].message.content)
                except json.decoder.JSONDecodeError:
                    logger.debug(f'INVALID JSON, TO BE RETRIED:\n{response}')

        return self.call(messages, **kwargs).choices[0].message.content


def default_llama_index_gemini_embed_model() -> OpenAIEmbedding:
    """Default Gemini embedding model using OpenAI interface."""
    return OpenAIEmbedding(
        mode=OpenAIEmbeddingMode.SIMILARITY_MODE, 
        model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE,
        embed_batch_size=100, 
        dimensions=3072, 
        additional_kwargs=None,
        api_key=LMConfig.GEMINI_API_KEY, 
        api_base=LMConfig.GEMINI_API_URL, 
        api_version=None,
        max_retries=10, 
        timeout=60,
        reuse_client=True, 
        callback_manager=None, 
        default_headers=None, 
        http_client=None,
        num_workers=cpu_count()
    )


def default_llama_index_gemini_lm(name: str = LMConfig.GEMINI_DEFAULT_MODEL, /) -> LlamaIndexOpenAILM:
    """Default Gemini LLM using OpenAI interface."""
    return LlamaIndexOpenAILM(
        model=name,
        temperature=LMConfig.DEFAULT_TEMPERATURE,
        max_tokens=None,
        additional_kwargs={},  # Remove seed for Gemini compatibility
        max_retries=3, 
        timeout=60, 
        reuse_client=True,
        api_key=LMConfig.GEMINI_API_KEY, 
        api_base=LMConfig.GEMINI_API_URL, 
        api_version=None,
        callback_manager=None, 
        default_headers=None, 
        http_client=None, 
        async_http_client=None,
        system_prompt=None, 
        messages_to_prompt=None, 
        completion_to_prompt=None,
        output_parser=None
    )