"""Common utilities for language model wrappers.

This module provides shared functionality used by both GenerativeLocalVLLM and
ScoringLocalVLLM classes, including base classes, mixins, and utility functions.

Note: Response classes (ChatCompletionResponse, RerankResponse, etc.) are defined
in their respective module files (generative_local_lm.py and scoring_local_lm.py)
since they are specific to each LM type.
"""

# Standard library imports
import gc
import logging
from typing import Any, Literal

# Third-party imports
import torch
from pydantic import BaseModel, Field
from vllm import LLM, distributed

logger = logging.getLogger(__name__)


class DictAccessMixin:
    """Mixin for dictionary-style access to BaseModel classes.

    Provides __getitem__, get, and __contains__ methods to allow
    Pydantic models to be accessed like dictionaries.
    """

    def __getitem__(self, key: str) -> Any:
        """Get attribute by key.

        Args:
            key: The attribute name to retrieve.

        Returns:
            The attribute value.
        """
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute by key with default.

        Args:
            key: The attribute name to retrieve.
            default: Default value if attribute doesn't exist.

        Returns:
            The attribute value or default.
        """
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists.

        Args:
            key: The attribute name to check.

        Returns:
            True if attribute exists, False otherwise.
        """
        return hasattr(self, key)


class Message(BaseModel, DictAccessMixin):
	"""Message structure for chat completions."""

	role: Literal["assistant", "user", "system", "tool"] = Field(
		..., description="Role: 'assistant', 'user', 'system', or 'tool'."
	)
	content: str = Field(..., description="Content of the message")


class Usage(BaseModel, DictAccessMixin):
	"""Token usage statistics."""

	prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt")
	completion_tokens: int = Field(default=0, description="Number of tokens in the completion")
	total_tokens: int = Field(default=0, description="Total number of tokens")


def extract_model_name(model_path: str) -> str:
    """Extract the model name from a model path.

    Args:
        model_path: Full model path (e.g., "Qwen/Qwen3-30B-A3B-Instruct").

    Returns:
        The last component of the path (e.g., "Qwen3-30B-A3B-Instruct").
    """
    return model_path.split("/")[-1] if "/" in model_path else model_path


def kill_vllm_model(model: LLM) -> None:
    """Free all model resources and memory.

    Performs garbage collection, destroys distributed environments,
    and clears CUDA cache.

    Args:
        model: The vLLM LLM instance to clean up.
    """
    gc.collect()
    try:
        distributed.parallel_state.destroy_model_parallel()
    except Exception:
        distributed.destroy_model_parallel()
    distributed.destroy_distributed_environment()
    torch.cuda.empty_cache()
    model.llm_engine.__del__()


def sleep_vllm_model(model: LLM, sleep_level: int) -> None:
    """Put the vLLM engine to sleep.

    The engine should not process any requests during sleep.

    Args:
        model: The vLLM LLM instance.
        sleep_level: The sleep level (1 or 2).
            Level 1: Offloads model weights and discards KV cache.
            Level 2: Discards both model weights and KV cache.
    """
    model.sleep(level=sleep_level)


def wake_up_vllm_model(model: LLM, tags: list[str] | None) -> None:
    """Wake up the vLLM engine from sleep mode.

    Args:
        model: The vLLM LLM instance.
        tags: Optional list of tags to reallocate memory for.
            Values must be in ("weights", "kv_cache").
            If None, all memory is reallocated.
    """
    model.wake_up(tags=tags)


def count_tokens(tokenizer: Any, text: str) -> int:
    """Return the number of tokens in text.

    Args:
        tokenizer: The tokenizer to use.
        text: The text to tokenize.

    Returns:
        The number of tokens in the text.
    """
    return len(tokenizer(text))

