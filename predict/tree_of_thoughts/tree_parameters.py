"""
Tree of Thoughts Parameters

This module provides parameter classes for configuring Tree of Thoughts search behavior.
"""

# Standard library imports
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

# Local imports
from constants import CandidateGenerationMethod
from lm import DEFAULT_TEMPERATURE


class NodeSelectionStrategy(StrEnum):
	"""Strategy for selecting nodes during tree search."""

	GREEDY = "greedy"
	SAMPLE = "sample"


@dataclass
class TreeOfThoughtsParameters:
	"""
	Parameters for controlling Tree of Thoughts search behavior.

	This class replaces the old utils.search_parameters.TreeOfThoughtsParameters
	with a simplified, modern implementation focused on local vLLM models.
	"""

	##############################
	### Tree search parameters ###
	##############################

	depth: int = 3
	# Maximum depth of the reasoning tree (number of reasoning steps).

	n_samples_generation: int = 3
	# Number of candidate reasoning steps to generate at each node.

	n_samples_judge: int = 1
	# Number of evaluation samples to use when scoring candidates.

	n_final_responses_per_trajectory: int | None = None
	# Number of final responses to generate per frontier node in the final layer.
	# If None, falls back to n_samples_generation.

	top_k: int = 2
	# Number of top-scoring candidates to keep at each layer.

	top_k_first: int | None = None
	# Optional override for top_k at the first layer only. If None, uses top_k.

	num_final_candidates: int = 1
	# Number of final candidates to return from the tree search.
	# TODO[P2]: We should include an easy option to return all final candidates.

	# Search behavior
	do_pruning: bool = True
	# Whether to prune low-scoring candidates during search.

	do_early_stopping: bool = False
	# Whether to allow early stopping based on controller decisions.

	use_self_consistency: bool = False
	# Whether to use self-consistency for final answer selection.

	candidate_generation_method: CandidateGenerationMethod = (
		CandidateGenerationMethod.SINGLE_CANDIDATE_CALLS
	)
	# Method for generating multiple candidates (single calls vs multi-candidate).

	node_selection_strategy: NodeSelectionStrategy = NodeSelectionStrategy.GREEDY
	# Strategy for selecting which nodes to expand.


	#################################
	### Judge-specific parameters ###
	#################################

	judge_max_tokens: int = 15_000
	# Maximum number of tokens to generate for evaluator/judge calls.

	# Evaluator (judge) sampling parameters
	judge_temperature: float = 0.7
	# Temperature for evaluating candidate quality.

	judge_top_p: float = 1.0
	# vLLM nucleus sampling top_p for evaluator (0, 1].

	judge_top_k: int = -1
	# vLLM sampling top_k for evaluator (int, can be -1 for all tokens).

	judge_min_p: float = 0.0
	# vLLM min_p for evaluator (>=0).

	judge_use_beam_search: bool = False
	# Whether to use beam search instead of sampling for evaluator.

	evaluator_type: Literal["generator", "reranker"] = "reranker"
	# Which evaluator implementation to use (generator-based vs reranker/scoring-based).

	###############################################
	### Controller-specific sampling parameters ###
	###############################################

	controller_max_tokens: int = 15_000
	# Maximum number of tokens to generate for controller calls.

	controller_temperature: float | None = None
	# Temperature override specifically for controller. If None, uses generator_temperature.

	controller_top_p: float = 1.0
	# Cumulative probability for controller nucleus sampling (0-1].

	controller_top_k: int = -1
	# Number of top tokens for controller to consider (int, can be -1 for all tokens).

	controller_min_p: float = 0.0
	# Minimum probability for controller tokens, relative to most likely token [0-1].

	controller_use_beam_search: bool = False
	# Whether to use beam search instead of sampling for controller decisions.

	##############################################
	### Generator-specific sampling parameters ###
	##############################################

	generator_temperature: float = DEFAULT_TEMPERATURE
	# Temperature for generating reasoning step candidates.

	generator_max_tokens: int = 15_000
	# Maximum number of tokens to generate for generator calls.

	generator_top_p: float = 1.0
	# Cumulative probability for generator nucleus sampling (0-1].

	generator_top_k: int = -1
	# Number of top tokens for generator to consider (int, can be -1 for all tokens).

	generator_min_p: float = 0.0
	# Minimum probability for generator tokens, relative to most likely token [0-1].

	generator_use_beam_search: bool = False
	# Whether to use beam search instead of sampling for generator decisions.

	def __post_init__(self) -> None:
		"""Validate parameter values after initialization."""
		# Auto-enable do_pruning when top_k implies pruning will occur
		if self.top_k < self.n_samples_generation:
			self.do_pruning = True

		assert self.depth > 0, "depth must be greater than 0"
		assert self.n_samples_generation > 0, (
			"n_samples_generation must be greater than 0"
		)
		if self.n_final_responses_per_trajectory is not None:
			assert self.n_final_responses_per_trajectory > 0, (
				"n_final_responses_per_trajectory must be greater than 0 when provided"
			)
		assert self.n_samples_judge > 0, "n_samples_judge must be greater than 0"
		assert self.top_k > 0, "top_k must be greater than 0"
		assert self.num_final_candidates > 0, (
			"num_final_candidates must be greater than 0"
		)
		assert 0 <= self.generator_temperature <= 2, (
			"generator_temperature must be between 0 and 2"
		)
		assert self.generator_max_tokens > 0, "generator_max_tokens must be greater than 0"
		assert self.controller_max_tokens > 0, "controller_max_tokens must be greater than 0"
		assert self.judge_max_tokens > 0, "judge_max_tokens must be greater than 0"

		assert 0 <= self.judge_temperature <= 2, (
			"judge_temperature must be between 0 and 2"
		)
		assert 0 < self.judge_top_p <= 1, "judge_top_p must be in (0, 1]"
		assert self.judge_top_k == -1 or self.judge_top_k > 0, (
			"judge_top_k must be -1 (all tokens) or > 0"
		)
		assert 0 <= self.judge_min_p <= 1, "judge_min_p must be in [0, 1]"

		if self.do_pruning:
			assert (
				self.num_final_candidates <= self.top_k * self.n_samples_generation
			), (
				f"num_final_candidates ({self.num_final_candidates}) must be <= "
				f"top_k * n_samples_generation ({self.top_k * self.n_samples_generation})"
			)

		# When top_k_first is set, the constraint for layer 1 is n_samples >= top_k_first,
		# and for layer 2+ it's n_samples * top_k_first >= top_k (checked below).
		# When top_k_first is None, we need n_samples >= top_k for all layers.
		if self.top_k_first is None:
			assert self.n_samples_generation >= self.top_k, (
				"n_samples_generation must be >= top_k (when top_k_first is not set)"
			)

		if self.top_k_first is not None:
			assert self.top_k_first > 0, "`top_k_first` must be greater than 0"
			assert self.n_samples_generation >= self.top_k_first, (
				f"`n_samples_generation` must be >= `top_k_first`, "
				f"but got {self.n_samples_generation} < {self.top_k_first}"
			)
			# Ensure enough candidates at layer 2 to select top_k
			assert self.n_samples_generation * self.top_k_first >= self.top_k, (
				f"`n_samples_generation * top_k_first` must be >= `top_k` to ensure "
				f"enough candidates at layer 2, but got "
				f"{self.n_samples_generation} * {self.top_k_first} = "
				f"{self.n_samples_generation * self.top_k_first} < {self.top_k}"
			)

		if self.controller_temperature is not None:
			assert 0 <= self.controller_temperature <= 2, (
				"controller_temperature must be between 0 and 2"
			)

		assert 0 < self.controller_top_p <= 1, "controller_top_p must be in (0, 1]"
		assert self.controller_top_k == -1 or self.controller_top_k > 0, (
			f"controller_top_k must be -1 (all tokens) or > 0. Got {self.controller_top_k}"
		)
		assert 0 <= self.controller_min_p <= 1, "controller_min_p must be in [0, 1]"

		assert 0 < self.generator_top_p <= 1, "generator_top_p must be in (0, 1]"
		assert self.generator_top_k == -1 or self.generator_top_k > 0, (
			f"generator_top_k must be -1 (all tokens) or > 0. Got {self.generator_top_k}"
		)
		assert 0 <= self.generator_min_p <= 1, "generator_min_p must be in [0, 1]"
