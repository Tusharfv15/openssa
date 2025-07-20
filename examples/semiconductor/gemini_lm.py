from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import os

from openssa.core.util.lm.gemini import GeminiLM


DEFAULT_MODEL = 'gemini-2.0-flash'
DEFAULT_API_KEY = os.environ.get('GEMINI_API_KEY', 'your-gemini-api-key')
DEFAULT_API_BASE = 'https://generativelanguage.googleapis.com/v1beta/openai'


@dataclass
class SemiconductorGeminiLM(GeminiLM):
    """Gemini LM for Semiconductor project."""

    @classmethod
    def from_defaults(cls) -> SemiconductorGeminiLM:
        """Get default Gemini LM instance for semiconductor project."""
        return cls(model=DEFAULT_MODEL, api_key=DEFAULT_API_KEY, api_base=DEFAULT_API_BASE)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('question')
    args = arg_parser.parse_args()

    print(SemiconductorGeminiLM.from_defaults().get_response(prompt=args.question))