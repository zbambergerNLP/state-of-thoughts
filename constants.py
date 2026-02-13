"""
Centralized constants for the dspy_reasoning project.

Core constants that are used across multiple modules.
Module-specific constants are stored in their respective modules:
- adapter/adapter_constants.py: Adapter-specific constants
- predict/controller/controller_constants.py: Controller-specific constants
- predict/tree_of_thoughts/tree_parameters.py: Tree of Thoughts parameters
- tree/tree_constants.py: Tree and ToT-specific constants
- signatures/field_constants.py: Signature field constants
- experiments/argument_generation/arg_gen_constants.py: Argument generation constants
- lm/lm_constants.py: Language model constants (FinishReason, TaskType)
"""
# Standard library imports
import enum
import logging
from pathlib import Path

# ============================================================================
# Enums
# ============================================================================


class GPU(enum.StrEnum):
	"""Enum for GPU types."""

	# NVIDIA GPUs
	A100_40_GB = "A100_40GB"
	A100_80_GB = "A100_80GB"
	H100_80_GB = "H100_80GB"

	def __str__(self) -> str:
		return self.value


class OpenSourceModel(enum.StrEnum):
	"""Enum for open source language models."""

	##############
	### Qwen3  ###
	##############

	# 4B Models
	QWEN_3_4B_INSTRUCT_2507 = "Qwen3-4B-Instruct-2507"
	QWEN_3_4B_INSTRUCT_2507_FP8 = "Qwen3-4B-Instruct-2507-FP8"
	QWEN_3_VL_4B_INSTRUCT_FP8 = "Qwen3-VL-4B-Instruct-FP8"
	QWEN_3_4B_THINKING_2507 = "Qwen3-4B-Thinking-2507"
	QWEN_3_4B_THINKING_2507_FP8 = "Qwen3-4B-Thinking-2507-FP8"

	# 8B Models
	QWEN_3_8B_INSTRUCT = "Qwen3-8B"
	QWEN_3_VL_8B_INSTRUCT = "Qwen3-VL-8B-Instruct"
	QWEN_3_VL_8B_INSTRUCT_FP8 = "Qwen3-VL-8B-Instruct-FP8"

	# 30B Models
	QWEN_3_30B_A3B_INSTRUCT_2507 = "Qwen3-30B-A3B-Instruct-2507"
	QWEN_3_30B_A3B_INSTRUCT_2507_FP8 = "Qwen3-30B-A3B-Instruct-2507-FP8"
	QWEN_3_30B_A3B_THINKING_2507_FP8 = "Qwen3-30B-A3B-Thinking-2507-FP8"
	QWEN_3_VL_30B_A3B_INSTRUCT_FP8 = "Qwen3-VL-30B-A3B-Instruct-FP8"

	# 80B Models
	QWEN_3_NEXT_80B_A3B_INSTRUCT = "Qwen3-Next-80B-A3B-Instruct"
	QWEN_3_NEXT_80B_A3B_INSTRUCT_FP8 = "Qwen3-Next-80B-A3B-Instruct-FP8"

	# 235B Models
	QWEN_3_235B_A22B_INSTRUCT_2507 = "Qwen3-235B-A22B-Instruct-2507"
	QWEN_3_235B_A22B_INSTRUCT_2507_FP8 = "Qwen3-235B-A22B-Instruct-2507-FP8"

	###############
	### Mistral ###
	###############

	MINISTRAL_3_3B_INSTRUCT_2512 = "Ministral-3-3B-Instruct-2512"
	MINISTRAL_3_3B_REASONING_2512 = "Ministral-3-3B-Reasoning-2512"
	MINISTRAL_3_8B_INSTRUCT_2512 = "Ministral-3-8B-Instruct-2512"
	MINISTRAL_3_8B_REASONING_2512 = "Ministral-3-8B-Reasoning-2512"
	MINISTRAL_3_14B_INSTRUCT_2512 = "Ministral-3-14B-Instruct-2512"
	MINISTRAL_3_14B_REASONING_2512 = "Ministral-3-14B-Reasoning-2512"

	# Google (Gemma) models
	GEMMA_3_4B_IT = "gemma-3-4b-it"
	GEMMA_3_27B_IT = "gemma-3-27b-it"

	# Nvidia (Nemotron) models
	NEMOTRON_3_NANO_30B_A3B_FP8 = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
	NEMOTRON_3_NANO_30B_A3B_BF16 = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

	# Qwen3 Reranker Models
	QWEN_3_RERANKER_4B = "Qwen3-Reranker-4B"  # Requires a T4 or better to run
	QWEN_3_RERANKER_8B = "Qwen3-Reranker-8B"  # Requires an A100 or better to run




	def __str__(self) -> str:
		return self.value


class ModelProvider(enum.StrEnum):
	"""
	Enum for open source language model providers.
	We support the providers of the open source models listed in OpenSourceModel.
	"""

	UNSLOTH = "unsloth"
	META_LLAMA = "meta-llama"
	QWEN = "Qwen"
	GOOGLE = "google"

	def __str__(self) -> str:
		return self.value


class CandidateGenerationMethod(enum.StrEnum):
	"""
	Methods for generating candidate reasoning steps.

	SINGLE_CANDIDATE_CALLS: Instructs the model to generate a single output per call and derives
		multiple candidates by repeatedly calling the model. This method depends on
		temperature-induced diversity, meaning it relies on relatively high generator temperature
		to produce different outputs over multiple calls.
	MULTI_CANDIDATE_CALL: Directly instructs the model to generate multiple outputs in a single
		call. This method leverages the model's ability to follow instructions for producing
		diverse reasoning steps without depending on temperature-based sampling.
	"""

	SINGLE_CANDIDATE_CALLS = "single_candidate_calls"
	MULTI_CANDIDATE_CALL = "multi_candidate_call"


class Verbosity(enum.StrEnum):
	"""Enum for verbosity levels."""
	DEBUG = "debug"
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"


# Mapping from Verbosity enum to logging levels
VERBOSITY_TO_LOGGING_LEVEL: dict[Verbosity, int] = {
	Verbosity.DEBUG: logging.DEBUG,
	Verbosity.INFO: logging.INFO,
	Verbosity.WARNING: logging.WARNING,
	Verbosity.ERROR: logging.ERROR,
}

# ============================================================================
# Path Constants
# ============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
