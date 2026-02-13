"""
Programmatic evaluator for Tree-of-Thoughts without LLM judges.

This module defines a programmatic evaluator that accepts Python scoring functions for
PRM/ORM evaluation while preserving the same interface as existing evaluators.
"""

# Standard library imports
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

# Third-party imports
import dspy

# Local imports
from adapter.adapter_constants import FIELD_HEADER_PATTERN
from misc_utils import ExecutionError
from signatures import ReasoningSignature, ensure_reasoning_signature
from tree import EvaluationResult, JudgeEvaluation, State

logger = logging.getLogger(__name__)

ProgrammaticScore = float
ProgrammaticScoreBounds = tuple[float, float]
ProgrammaticScorer = Callable[["ProgrammaticEvaluationInput"], ProgrammaticScore]


@dataclass(frozen=True)
class ProgrammaticEvaluationInput:
	"""
	Container for inputs passed to programmatic scoring functions.

	Contains only the extracted text candidates from reasoning/output and input context.
	Reasoning text is extracted from <thinking>...</thinking> tags.
	Output text is extracted from <answer>...</answer> tags.

	Attributes:
		candidates: List of text strings extracted from reasoning or output for evaluation.
			For PRM: contains reasoning text extracted from <thinking>...</thinking>.
			For ORM: contains output text extracted from <answer>...</answer>.
			Can contain both if evaluating intermediate and final outputs together.
		inputs: Input fields for the task (e.g., instruction, prompt).
		evaluation_type: The evaluation type ("process" for PRM or "outcome" for ORM).
	"""

	candidates: list[str]
	inputs: dict[str, Any]
	evaluation_type: Literal["process", "outcome"]


class TreeOfThoughtProgrammaticEvaluator(dspy.Module):
	"""
	Programmatic evaluator for Tree-of-Thoughts that uses Python scoring functions.

	This evaluator supports PRM (process) and ORM (outcome) scoring without requiring an LLM.
	"""

	def __init__(
		self,
		generator_signature: type[ReasoningSignature],
		prm_scorer: ProgrammaticScorer | None = None,
		orm_scorer: ProgrammaticScorer | None = None,
		consider_reasoning_in_final_eval: bool = False,
		prm_score_bounds: ProgrammaticScoreBounds = (0.0, 1.0),
		orm_score_bounds: ProgrammaticScoreBounds = (0.0, 1.0),
		prm_score_name: str = "score",
		orm_score_name: str = "score",
		verbosity: Literal["debug", "info", "warning", "error"] = "warning",
	) -> None:
		"""
		Initialize the programmatic evaluator.

		Args:
			generator_signature: The task signature used to extract field metadata.
			prm_scorer: Callable that returns a PRM score for a ProgrammaticEvaluationInput.
			orm_scorer: Callable that returns an ORM score for a ProgrammaticEvaluationInput.
			consider_reasoning_in_final_eval: Whether ORM scoring should include reasoning.
			prm_score_bounds: (lower, upper) bounds for normalizing PRM scores to [0, 1].
			orm_score_bounds: (lower, upper) bounds for normalizing ORM scores to [0, 1].
			prm_score_name: Name used in PRM raw/normalized score dictionaries.
			orm_score_name: Name used in ORM raw/normalized score dictionaries.
			verbosity: Logging verbosity for this evaluator.
		"""
		super().__init__()
		self.generator_signature = ensure_reasoning_signature(generator_signature)
		self.prm_scorer = prm_scorer
		self.orm_scorer = orm_scorer
		self.consider_reasoning_in_final_eval = consider_reasoning_in_final_eval
		self._verbosity = verbosity
		self.prm_score_bounds = prm_score_bounds
		self.orm_score_bounds = orm_score_bounds
		self.prm_score_name = prm_score_name
		self.orm_score_name = orm_score_name

		logging_level = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
		}.get(verbosity, logging.WARNING)
		logger.setLevel(logging_level)

		# TODO[P4]: Support optional processing of generator input fields for
		#  programmatic evaluation of intermediate/final outputs.
		if not self.generator_signature.reasoning_fields:
			raise ValueError("Generator signature must have at least one reasoning field")
		self.reasoning_field_name = list(self.generator_signature.reasoning_fields.keys())[0]

	@property
	def verbosity(self) -> Literal["debug", "info", "warning", "error"]:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Literal["debug", "info", "warning", "error"]) -> None:
		"""Set verbosity and update logger."""
		self._verbosity = verbosity
		logging_level = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
		}.get(verbosity, logging.WARNING)
		logger.setLevel(logging_level)

	@staticmethod
	def _normalize_score(
		score: float, lower_bound: float, upper_bound: float
	) -> float:
		"""
		Normalize a score to the range [0, 1].

		Args:
			score: The raw score from the programmatic scorer.
			lower_bound: Minimum possible score.
			upper_bound: Maximum possible score.

		Returns:
			The normalized score in [0, 1].
		"""
		assert lower_bound < upper_bound, (
			"Score bounds must satisfy lower_bound < upper_bound, "
			f"got ({lower_bound}, {upper_bound})."
		)
		if not (lower_bound <= score <= upper_bound):
			raise ValueError(
				f"Score {score} outside bounds [{lower_bound}, {upper_bound}]."
			)
		return (score - lower_bound) / (upper_bound - lower_bound)

	@staticmethod
	def _ensure_finite_score(score: float, evaluation_type: str) -> None:
		"""
		Validate that a score is finite and numeric.

		Args:
			score: Score returned by the scorer.
			evaluation_type: The evaluation type for error context.
		"""
		if not math.isfinite(score):
			raise ValueError(f"{evaluation_type} score must be finite, got {score}.")

	def _extract_text_from_xml_tag(self, text: str, tag_name: str) -> str:
		"""
		Extract text content between XML tags (e.g., <thinking>...</thinking>).

		Handles both closed tags (<thinking>...</thinking>) and unclosed tags
		(<thinking>... at end of string, which occurs for intermediate reasoning steps).

		Args:
			text: The full text containing XML tags.
			tag_name: The name of the XML tag (e.g., "thinking", "answer").

		Returns:
			The text content between the opening and closing tags, or from opening tag
			to end of string if no closing tag found, or empty string if tag not found.
		"""
		# First try to find closed tag
		pattern_closed = rf"<{tag_name}>(.*?)</{tag_name}>"
		match = re.search(pattern_closed, text, re.DOTALL)
		if match:
			return match.group(1).strip()

		# If no closing tag, extract from opening tag to end of string
		pattern_unclosed = rf"<{tag_name}>(.*)$"
		match = re.search(pattern_unclosed, text, re.DOTALL)
		if match:
			return match.group(1).strip()

		return ""

	def _extract_reasoning_text(self, state: State) -> str:
		"""
		Extract reasoning text from the state's reasoning steps.

		Extracts reasoning field values directly from state.reasoning using the
		reasoning field name from the generator signature. This is more reliable
		than parsing the XML representation.

		Args:
			state: The state containing reasoning steps.

		Returns:
			Single string containing the extracted reasoning text.
		"""
		if not state.reasoning:
			return ""

		step_contents: list[str] = []
		empty_steps: list[tuple[int, str | None]] = []
		for step_index, step in enumerate(state.reasoning):
			# Extract the reasoning field value specifically
			if self.reasoning_field_name not in step:
				error_obj = step.get("error")
				error_msg = None
				if error_obj and isinstance(error_obj, ExecutionError) and error_obj.has_error():
					error_msg = error_obj.error_message or f"{error_obj.error_type} error"
				empty_steps.append((step_index, error_msg))
				continue

			reasoning_value = step[self.reasoning_field_name]
			if reasoning_value is None:
				error_obj = step.get("error")
				error_msg = None
				if error_obj and isinstance(error_obj, ExecutionError) and error_obj.has_error():
					error_msg = error_obj.error_message or f"{error_obj.error_type} error"
				empty_steps.append((step_index, error_msg))
				continue

			# Convert to string and strip whitespace
			reasoning_text = str(reasoning_value).strip()

			# If reasoning is empty, check if there's an error we can use
			if not reasoning_text:
				error_obj = step.get("error")
				if error_obj and isinstance(error_obj, ExecutionError) and error_obj.has_error():
					# Use error message as reasoning text if available
					error_msg = error_obj.error_message or "Generation/parsing error occurred"
					reasoning_text = error_msg
				else:
					error_msg = None
					if error_obj and isinstance(error_obj, ExecutionError) and error_obj.has_error():
						error_msg = error_obj.error_message or f"{error_obj.error_type} error"
					empty_steps.append((step_index, error_msg))

			if reasoning_text:
				step_contents.append(reasoning_text)

		# Log debug info if there were empty steps (but we still extracted some content)
		if empty_steps and step_contents:
			error_details = [
				f"step {idx}" + (f" (error: {err})" if err else "")
				for idx, err in empty_steps
			]
			logger.debug(
				f"Skipped {len(empty_steps)} empty reasoning step(s): {', '.join(error_details)}. "
				f"Extracted content from {len(step_contents)} step(s)."
			)

		# Return the entire reasoning trajectory as a single concatenated string
		# Join steps with double newline to preserve step boundaries
		result = "\n\n".join(step_contents).strip()
		return result

	def _extract_output_text(self, state: State) -> str:
		"""
		Extract output text from <answer>...</answer> tags in the state.

		Uses the state's model_output_so_far() to reconstruct the XML structure,
		then extracts only the content between <answer> and </answer> tags,
		excluding XML/header formatting.

		Args:
			state: The state containing output fields.

		Returns:
			Single string containing the extracted output text (without XML tags
			or header formatting).
		"""
		if not state.output:
			return ""

		# Reconstruct the XML structure to extract from <answer> tags
		model_output = state.model_output_so_far()
		answer_text = self._extract_text_from_xml_tag(model_output, "answer")

		if not answer_text:
			return ""

		# Strip header formatting (like ## result, ## response, etc.)
		lines = answer_text.splitlines()
		cleaned_lines = []
		for line in lines:
			line_stripped = line.strip()
			# Skip header lines
			if FIELD_HEADER_PATTERN.match(line_stripped):
				continue
			# Remove header patterns that appear within lines (but keep the content)
			line_cleaned = FIELD_HEADER_PATTERN.sub("", line).strip()
			if line_cleaned:
				cleaned_lines.append(line_cleaned)

		return "\n".join(cleaned_lines).strip()

	def _build_eval_input(
		self, state: State, evaluation_type: Literal["process", "outcome"]
	) -> ProgrammaticEvaluationInput:
		"""
		Construct the ProgrammaticEvaluationInput for a state.

		Extracts text from <thinking>...</thinking> for reasoning and <answer>...</answer>
		for output, then passes only the extracted text candidates to the scorer.

		Args:
			state: The state to evaluate.
			evaluation_type: The evaluation type ("process" or "outcome").

		Returns:
			ProgrammaticEvaluationInput containing extracted text candidates and inputs.

		Raises:
			ValueError: If PRM evaluation has no reasoning or ORM evaluation has no output.
		"""
		inputs: dict[str, Any] = dict(state.input)
		candidates: list[str] = []

		if evaluation_type == "process":
			# For PRM, extract reasoning text from <thinking>...</thinking>
			reasoning_text = self._extract_reasoning_text(state=state)
			if not reasoning_text.strip():
				raise ValueError("PRM evaluation requires at least one reasoning step.")
			candidates.append(reasoning_text)
		else:
			# For ORM, extract output text from <answer>...</answer>
			output_text = self._extract_output_text(state=state)
			if not output_text.strip():
				raise ValueError("ORM evaluation requires a non-empty final output.")
			candidates.append(output_text)

			# Optionally include reasoning if consider_reasoning_in_final_eval is True
			if self.consider_reasoning_in_final_eval:
				reasoning_text = self._extract_reasoning_text(state=state)
				if reasoning_text.strip():
					candidates.append(reasoning_text)

		logger.debug(
			f"Built {evaluation_type} evaluation input: "
			f"{len(candidates)} candidate(s) with lengths {[len(c) for c in candidates]}, "
			f"input fields: {list(inputs.keys())}"
		)

		return ProgrammaticEvaluationInput(
			candidates=candidates,
			inputs=inputs,
			evaluation_type=evaluation_type,
		)

	def _score_state(
		self, state: State, evaluation_type: Literal["process", "outcome"]
	) -> EvaluationResult:
		"""
		Score a single state and return an EvaluationResult.

		Args:
			state: The state to score.
			evaluation_type: The evaluation type ("process" or "outcome").

		Returns:
			EvaluationResult containing normalized score and judge metadata.
		"""
		eval_input = self._build_eval_input(state=state, evaluation_type=evaluation_type)
		if evaluation_type == "process":
			scorer = self.prm_scorer
			score_bounds = self.prm_score_bounds
			score_name = self.prm_score_name
		else:
			scorer = self.orm_scorer
			score_bounds = self.orm_score_bounds
			score_name = self.orm_score_name

		if scorer is None:
			raise ValueError(
				f"No scorer provided for {evaluation_type} evaluation in programmatic evaluator."
			)

		logger.debug(
			f"Calling {evaluation_type.upper()} scorer: "
			f"{len(eval_input.candidates)} candidate(s), "
			f"input keys: {list(eval_input.inputs.keys())}"
		)

		raw_score = float(scorer(eval_input))
		self._ensure_finite_score(raw_score, evaluation_type)
		normalized = self._normalize_score(raw_score, *score_bounds)

		logger.debug(
			f"{evaluation_type.upper()} score: raw={raw_score:.3f}, normalized={normalized:.3f}"
		)

		judge = JudgeEvaluation(
			raw_scores={score_name: raw_score},
			normalized_scores={score_name: normalized},
		)
		return EvaluationResult(score=normalized, judge_evaluations=[judge])

	def forward(
		self,
		states: State | list[State],
		demos: list[dict[str, Any]] | None = None,
		demos_prm: list[dict[str, Any]] | None = None,
		demos_orm: list[dict[str, Any]] | None = None,
		**kwargs: Any,
	) -> list[list[EvaluationResult]]:
		"""
		Evaluate states using programmatic PRM/ORM scoring.

		Args:
			states: Single state or list of states to evaluate.
			demos: Unused, accepted for API compatibility with the LLM evaluator.
			demos_prm: Unused, accepted for API compatibility with the LLM evaluator.
			demos_orm: Unused, accepted for API compatibility with the LLM evaluator.
			**kwargs: Additional keyword arguments (e.g., LLM sampling params)
				accepted for API compatibility with the LLM evaluator and ignored.

		Returns:
			List of evaluation result lists, one per state.
		"""
		state_list = states if isinstance(states, list) else [states]
		num_states = len(state_list)

		results: list[list[EvaluationResult]] = []
		for idx, state in enumerate(state_list):
			try:
				result = self._evaluate_single_state(idx, state, num_states)
			except Exception:
				logger.warning(
					f"Programmatic evaluation failed for state {idx + 1}/{num_states}",
					exc_info=True,
				)
				result = EvaluationResult(
					score=0.0,
					judge_evaluations=[
						JudgeEvaluation(
							raw_scores={"default": 0.0},
							normalized_scores={"default": 0.0},
						)
					],
				)
			# Wrap in a single-element list to match the list[list[EvaluationResult]]
			# return type shared with the generative evaluator, which supports
			# ensemble judging (multiple judges per state via n_samples_evaluator).
			# The programmatic evaluator always produces one result per state.
			results.append([result])
		return results

	def _evaluate_single_state(
		self,
		state_idx: int,
		state: "State",
		num_states: int,
	) -> EvaluationResult:
		"""Evaluate a single state, returning an EvaluationResult.

		Args:
			state_idx: Zero-based index of the state in the batch.
			state: The state to evaluate, containing reasoning steps and optional output.
			num_states: Total number of states in the batch (used for logging).

		Returns:
			An EvaluationResult with a score and per-judge evaluations.
		"""
		# TODO[P1]: Update error-checking logic here once error-handling
		#  refactor is complete (errors will be exceptions, not dict entries).
		logger.debug(
			f"Evaluating state {state_idx + 1}/{num_states}: "
			f"{len(state.reasoning)} reasoning step(s), "
			f"has_output={bool(state.output) and any(k != 'error' for k in state.output.keys())}"
		)

		# TODO[P1]: Adjust error-handling logic once refactor is complete.
		# Check if output exists and has non-error keys
		has_output = (
			bool(state.output)
			and len(state.output) > 0
			and any(k != "error" for k in state.output.keys())
		)

		if has_output:
			evaluation_type: Literal["process", "outcome"] = "outcome"
			return self._score_state(state=state, evaluation_type=evaluation_type)

		# Check if reasoning can be extracted (PRM evaluation)
		reasoning_text = self._extract_reasoning_text(state=state)

		if reasoning_text.strip():
			evaluation_type: Literal["process", "outcome"] = "process"
			return self._score_state(state=state, evaluation_type=evaluation_type)

		# TODO[P1]: Refactor error-handling logic once errors are returned as
		#  exceptions rather than embedded in the output dict.
		# No reasoning and no output: return default low score to prune this state
		error_messages: list[str] = []

		# Check reasoning steps for errors
		for step_index, step in enumerate(state.reasoning):
			error_obj = step.get("error")
			if error_obj and isinstance(error_obj, ExecutionError) and error_obj.has_error():
				error_msg = error_obj.error_message or f"{error_obj.error_type} error"
				error_messages.append(f"s{step_index}:{error_msg}")

		# Check output for errors
		if state.output:
			error_obj = state.output.get("error")
			if error_obj and isinstance(error_obj, ExecutionError) and error_obj.has_error():
				error_msg = error_obj.error_message or f"{error_obj.error_type} error"
				error_messages.append(f"out:{error_msg}")

		if error_messages:
			logger.debug(f"Can't extract reasoning. Got errors: {', '.join(error_messages)}")
		else:
			node_content = state.model_output_so_far()
			content_preview = node_content[:300] if node_content else "(empty)"
			logger.debug(f"Can't extract reasoning/output. Got {content_preview}")

		return EvaluationResult(
			score=0.0,
			judge_evaluations=[
				JudgeEvaluation(
					raw_scores={"default": 0.0}, normalized_scores={"default": 0.0}
				)
			],
		)
