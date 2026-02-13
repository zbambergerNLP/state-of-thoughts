"""Language model module for DSPy.

This module provides GenerativeLocalVLLM, ScoringLocalVLLM, and related classes
for using local language models with DSPy.
"""

# Local imports
from lm.generative_local_lm import (
    ChatCompletionResponse,
    Choice,
    GenerativeLocalVLLM,
)
from lm.lm_constants import (
    CHAT_TEMPLATE_FORMATS,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    ENABLE_THINKING,
    RESULT_CHOICES,
    TASK_TYPES,
)
from lm.lm_utils import DictAccessMixin, Message, Usage
from lm.scoring_local_lm import (
    RerankResponse,
    RerankResult,
    ScoringLocalVLLM,
)

__all__ = [
    "CHAT_TEMPLATE_FORMATS",
    "ChatCompletionResponse",
    "Choice",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "DEFAULT_MAX_MODEL_LEN",
    "DEFAULT_SEED",
    "DEFAULT_TEMPERATURE",
    "DictAccessMixin",
    "ENABLE_THINKING",
    "GenerativeLocalVLLM",
    "Message",
    "RerankResponse",
    "RerankResult",
    "RESULT_CHOICES",
    "ScoringLocalVLLM",
    "TASK_TYPES",
    "Usage",
]
