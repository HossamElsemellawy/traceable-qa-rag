"""
qarag (evaluation subset) - the parts of the qarag package that the answer
evaluation needs:

    config.py   settings, read from .env
    llm.py      chat calls to OpenRouter through LiteLLM, JSON parsing,
                usage and cost meter

The full package (indexer, retrieval, answering) is in the indexer program.
"""

__version__ = "0.4.0"
