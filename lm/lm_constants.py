"""Constants specific to local language models."""

# Standard library imports
from typing import Literal

# Type aliases for backward compatibility
TASK_TYPES = Literal["auto", "generate", "score", "embedding", "embed", "classify", "reward"]


CHAT_TEMPLATE_FORMATS = Literal["auto", "string", "openai"]

# Parameters for initializing vLLM models
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_SEED: int = 42
DEFAULT_MAX_MODEL_LEN: int = 16384
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.9

# Result dictionary key
RESULT_CHOICES: str = "choices"

# Parameters for chat template kwargs
ENABLE_THINKING: str = "enable_thinking"

