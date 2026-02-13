"""
Reranker-based evaluator for Tree-of-Thoughts.

This module implements PRM (process) and ORM (outcome) evaluation using a Qwen3-style
reranker model via `ScoringLocalVLLM` and `LocalVLLMScoringAdapter`.

Unlike the generative-judge evaluator, reranker evaluation is deterministic, so it
always uses single-judge semantics (one score per state).
"""

# Standard library imports
import logging
from typing import Any, Literal

# Third-party imports
import dspy

# Local imports
from adapter.vllm_scoring_adapter import LocalVLLMScoringAdapter
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.scoring_local_lm import ScoringLocalVLLM
from signatures import ReasoningSignature, ensure_reasoning_signature
from tree import EvaluationResult, JudgeEvaluation, State

logger = logging.getLogger(__name__)

class TreeOfThoughtRerankerEvaluator(dspy.Module):
	"""
	Reranker-based evaluator for Tree-of-Thoughts that supports PRM and ORM scoring.

	- PRM: score the current reasoning trajectory (with focus on the latest step)
	- ORM: score the final output (optionally conditioned on the reasoning trajectory)

	Scores are returned as probabilities in [0, 1] (affinity for "yes" vs "no") and
	are directly used as ToT node scores.
	"""

	def __init__(
		self,
		generator_signature: type[ReasoningSignature],
		evaluator_signature: type[dspy.Signature] | str | None = None,
		evaluator_signature_prm: type[dspy.Signature] | str | None = None,
		evaluator_signature_orm: type[dspy.Signature] | str | None = None,
		consider_reasoning_in_final_eval: bool = False,
		include_reasoning_in_document_for_output: bool = False,
		verbosity: Verbosity = Verbosity.WARNING,
	) -> None:
		"""
		Initialize the reranker evaluator.

		Args:
			generator_signature: The task signature for ToT generation. This is used to
				provide task instructions and input/output field metadata to the scoring adapter.
			consider_reasoning_in_final_eval: Whether ORM scoring should condition on the
				reasoning trajectory in addition to the final output.
			include_reasoning_in_document_for_output: Whether to include the reasoning trajectory in
				the Document (preceding the output) for OUTPUT scoring. If False, reasoning (if any)
				is included in the Query instead. Output is always in the Document.
			verbosity: Logging verbosity.
		"""
		super().__init__()
		self.generator_signature = ensure_reasoning_signature(generator_signature)
		self._evaluator_signature = evaluator_signature
		self._evaluator_signature_prm = evaluator_signature_prm
		self._evaluator_signature_orm = evaluator_signature_orm
		self.consider_reasoning_in_final_eval = consider_reasoning_in_final_eval
		self.include_reasoning_in_document_for_output = include_reasoning_in_document_for_output
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])
		self.scoring_adapter = LocalVLLMScoringAdapter(verbosity=verbosity)
		self.lm: ScoringLocalVLLM | None = None
		self.prm_rubric = self._resolve_prm_rubric()
		self.orm_rubric = self._resolve_orm_rubric()

	def _resolve_prm_rubric(self) -> dspy.Signature | None:
		"""Returns the ReasoningSignature for PRM scoring (i.e., the rubric for PRM scoring)."""
		if self._evaluator_signature_prm is not None:
			return dspy.ensure_signature(self._evaluator_signature_prm)
		if self._evaluator_signature is not None:
			return dspy.ensure_signature(self._evaluator_signature)
		return None

	def _resolve_orm_rubric(self) -> dspy.Signature | None:
		"""Returns the ReasoningSignature for ORM scoring (i.e., the rubric for ORM scoring)."""
		if self._evaluator_signature_orm is not None:
			return dspy.ensure_signature(self._evaluator_signature_orm)
		if self._evaluator_signature is not None:
			return dspy.ensure_signature(self._evaluator_signature)
		return None

	@staticmethod
	def _instruction_target_names(
		scoring_target: Literal["reasoning", "output"],
		consider_reasoning_in_output: bool,
		is_reasoning_empty: bool,
	) -> tuple[str, str]:
		"""Return (label_description, candidate_noun) for instruction copy."""
		if scoring_target == "reasoning":
			return "partial solution for the task given the inputs", "reasoning trajectory"
		# scoring_target == "output"
		label_description = (
			"final output for the task given the inputs and reasoning"
			if consider_reasoning_in_output and not is_reasoning_empty
			else "final output for the task given the inputs"
		)
		return label_description, "output"

	@staticmethod
	def _objective_line(scoring_target: Literal["reasoning", "output"]) -> str:
		"""Return the first-line objective for the judge."""
		if scoring_target == "reasoning":
			return "Your objective is to judge a reasoning trajectory towards solving a user-assigned task."
		return "Your objective is to judge an output for a user-assigned task."

	def _heading_lines(
		self,
		scoring_target: Literal["reasoning", "output"],
		consider_reasoning_in_output: bool,
		is_reasoning_empty: bool,
	) -> list[str]:
		"""Return lines describing where inputs/reasoning/output are located by heading."""
		lines: list[str] = [
			"You will find the inputs for this task under the \"# Inputs\" header in the Query."
		]
		include_reasoning_heading = (
			scoring_target == "reasoning"
			or (
				scoring_target == "output"
				and consider_reasoning_in_output
				and not is_reasoning_empty
			)
		)
		if include_reasoning_heading:
			lines.append(
				"You will find the intermediate reasoning trajectory towards solving this problem "
				"under the \"# Reasoning\" header in the Document."
			)
		if scoring_target == "output":
			lines.append(
				"You will find the output under consideration under the \"# Output\" heading in the "
				"Document."
			)
		return lines

	@staticmethod
	def _candidate_noun(scoring_target: Literal["reasoning", "output"]) -> str:
		if scoring_target == "reasoning":
			return "reasoning trajectory"
		return "output"

	def _base_instruction_text(
		self,
		scoring_target: Literal["reasoning", "output"],
		rubric: dspy.Signature | None,
		consider_reasoning_in_output: bool,
		is_reasoning_empty: bool,
		label_description: str,
		candidate_noun: str,
	) -> str:
		"""Build base instructions (excluding rubric bullets)."""
		assessment_instruction = (
			f"Judge whether the provided {candidate_noun} is a strong {label_description}."
			if rubric is None
			else (
				f"Judge whether the provided {candidate_noun} is a strong {label_description}, "
				"using the rubric below."
			)
		)
		lines: list[str] = [
			self._objective_line(scoring_target=scoring_target),
			"",
			"The user provided the following task:",
			f"\"{self.generator_signature.instructions.strip()}\"",
			"",
		]
		include_reasoning_interest = not (
			(scoring_target == "output" and is_reasoning_empty)
			or (scoring_target == "output" and not consider_reasoning_in_output)
		)
		if include_reasoning_interest:
			lines.append(
				"Since this is a reasoning task, we are interested not only in the final output, "
				"but also in the reasoning process that leads to it."
			)
		lines.extend(
			self._heading_lines(
				scoring_target=scoring_target,
				consider_reasoning_in_output=consider_reasoning_in_output,
				is_reasoning_empty=is_reasoning_empty,
			)
		)
		lines.extend(["", assessment_instruction])
		return "\n".join(lines).strip()

	@staticmethod
	def _rubric_section_text(rubric: dspy.Signature) -> str:
		"""Build rubric section text with bullet points."""
		rubric_lines: list[str] = ["Rubric:"]
		for field_name, field_info in rubric.output_fields.items():
			description = getattr(field_info, "description", None) or getattr(field_info, "desc", None)
			if description is None:
				json_schema_extra = getattr(field_info, "json_schema_extra", None)
				if isinstance(json_schema_extra, dict):
					description = json_schema_extra.get("desc")

			if isinstance(description, str) and description.strip():
				rubric_lines.append(f"- {field_name}: {description.strip()}")
			else:
				rubric_lines.append(f"- {field_name}: {field_name}")
		return "\n".join(rubric_lines).strip()

	def build_scoring_instructions(
		self,
		scoring_target: Literal["reasoning", "output"],
		rubric: dspy.Signature | None,
		consider_reasoning_in_output: bool,
		is_reasoning_empty: bool,
	) -> str:
		"""Builds the scoring instructions for the given scoring target and rubric.

		Args:
			scoring_target: The scoring target (REASONING, OUTPUT, or ACTION).
			rubric: The rubric for the scoring target.
			consider_reasoning_in_output: Whether to consider reasoning in the output.

		Returns:
			The scoring instructions for the given scoring target and rubric.
		"""
		assert scoring_target in {"reasoning", "output"}, (
			"Reranker evaluator only supports PRM/ORM scoring targets (REASONING or OUTPUT). "
			f"Got: {scoring_target}"
		)
		label_description, candidate_noun = self._instruction_target_names(
			scoring_target=scoring_target,
			consider_reasoning_in_output=consider_reasoning_in_output,
			is_reasoning_empty=is_reasoning_empty,
		)
		base_instructions = self._base_instruction_text(
			scoring_target=scoring_target,
			rubric=rubric,
			consider_reasoning_in_output=consider_reasoning_in_output,
			is_reasoning_empty=is_reasoning_empty,
			label_description=label_description,
			candidate_noun=candidate_noun,
		)

		if rubric is None:
			return base_instructions
		rubric_section = self._rubric_section_text(rubric)

		parts: list[str] = [base_instructions, rubric_section]
		return "\n\n".join(parts).strip()

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set verbosity and propagate it to the scoring adapter."""
		self._verbosity = verbosity
		self.scoring_adapter.verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

	def set_lm(self, lm: ScoringLocalVLLM) -> None:
		"""Set the reranker LM used for scoring."""
		assert isinstance(lm, ScoringLocalVLLM), "lm must be a ScoringLocalVLLM instance"
		self.lm = lm

	def get_lm(self) -> ScoringLocalVLLM:
		"""Return the reranker LM, raising if it is not set."""
		if self.lm is None:
			raise ValueError("Language model has not been set. Call set_lm() first.")
		return self.lm

	@staticmethod
	def _thinking_only_from_state(state: State) -> str:
		"""
		Return a thinking-only string from a state.

		This strips any `<answer>...</answer>` block so the caller can decide whether to
		condition on reasoning for ORM evaluation.

		Args:
			state: The ToT state.

		Returns:
			A string containing `<thinking>...</thinking>` (or empty string if no reasoning).
		"""
		full = state.model_output_so_far().strip()
		if not full:
			return ""
		idx = full.find("<answer>")
		if idx < 0:
			return full
		return full[:idx].rstrip()

	@staticmethod
	def _first_content_line_from_reasoning_candidate(reasoning_candidate: str) -> str:
		"""
		Extract the first natural-language content line from a rendered reasoning candidate.

		`State.model_output_so_far()` wraps reasoning in XML-ish tags (e.g. `<thinking>`,
		`<step>`) and field headers (e.g. `## reasoning_step`). For heuristics we want the
		first actual content line.
		"""
		for raw in (reasoning_candidate or "").splitlines():
			line = raw.strip()
			if not line:
				continue
			if line.startswith("<") and line.endswith(">"):
				continue
			if line.startswith("## "):
				continue
			return line
		return ""

	def _score_single_state_prm(self, state: State, reranker_lm: ScoringLocalVLLM) -> float:
		"""
		Score a single state using PRM-style reasoning scoring.

		Args:
			state: State with at least one reasoning step.
			reranker_lm: The scoring LM.

		Returns:
			A score in [0, 1].
		"""
		if not state.reasoning:
			raise ValueError("PRM evaluation requires at least one reasoning step.")

		inputs = dict(state.input)
		reasoning_candidate = state.model_output_so_far()
		scoring_instructions = self.build_scoring_instructions(
			scoring_target="reasoning",
			rubric=self.prm_rubric,
			consider_reasoning_in_output=True,
			is_reasoning_empty=True,
		)

		responses = self.scoring_adapter(
			instructions=scoring_instructions,
			lm=reranker_lm,
			inputs=inputs,
			scoring_target="reasoning",
			reasoning_candidates=[reasoning_candidate],
		)

		assert responses and responses[0].results, "No reranker results returned for PRM scoring"
		return float(responses[0].results[0].relevance_score)

	def _score_batch_prm(self, states: list[State], reranker_lm: ScoringLocalVLLM) -> list[float]:
		"""
		Score a batch of states using PRM-style reasoning scoring in a single vLLM call.

		Args:
			states: States with at least one reasoning step each.
			reranker_lm: The scoring LM.

		Returns:
			List of scores in [0, 1], aligned to input order.
		"""
		if not states:
			return []

		reasoning_candidates: list[str] = []
		for state in states:
			if not state.reasoning:
				raise ValueError("PRM evaluation requires at least one reasoning step.")
			reasoning_candidates.append(state.model_output_so_far())

		scoring_instructions = self.build_scoring_instructions(
			scoring_target="reasoning",
			rubric=self.prm_rubric,
			consider_reasoning_in_output=True,
			is_reasoning_empty=True,
		)
		responses = self.scoring_adapter(
			instructions=scoring_instructions,
			lm=reranker_lm,
			inputs=dict(states[0].input),
			scoring_target="reasoning",
			reasoning_candidates=reasoning_candidates,
		)
		assert len(responses) == len(states), (
			f"Expected {len(states)} rerank responses but got {len(responses)}"
		)
		scores: list[float] = []
		for resp in responses:
			assert resp.results, "No reranker results returned for PRM scoring"
			scores.append(float(resp.results[0].relevance_score))
		return scores

	def _score_single_state_orm(self, state: State, reranker_lm: ScoringLocalVLLM) -> float:
		"""
		Score a single state using ORM-style final-output scoring.

		Args:
			state: State with a final output.
			reranker_lm: The scoring LM.

		Returns:
			A score in [0, 1].
		"""
		if not state.output:
			raise ValueError("ORM evaluation requires a non-empty final output.")

		inputs = dict(state.input)
		reasoning_str = (
			self._thinking_only_from_state(state)
			if self.consider_reasoning_in_final_eval
			else ""
		)
		include_reasoning_in_document_for_output = (
			self.include_reasoning_in_document_for_output
			or (self.consider_reasoning_in_final_eval and bool(reasoning_str.strip()))
		)
		scoring_instructions = self.build_scoring_instructions(
			scoring_target="output",
			rubric=self.orm_rubric,
			consider_reasoning_in_output=self.consider_reasoning_in_final_eval,
			is_reasoning_empty=not bool(reasoning_str.strip()),
		)

		output_candidate = {k: v for k, v in dict(state.output).items() if k != "error"}
		responses = self.scoring_adapter(
			instructions=scoring_instructions,
			lm=reranker_lm,
			inputs=inputs,
			scoring_target="output",
			reasoning_candidates=[reasoning_str],
			output_candidates=[output_candidate],
			include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
		)

		assert responses and responses[0].results, "No reranker results returned for ORM scoring"
		return float(responses[0].results[0].relevance_score)

	def _score_batch_orm(self, states: list[State], reranker_lm: ScoringLocalVLLM) -> list[float]:
		"""
		Score a batch of states using ORM-style final-output scoring in a single vLLM call.

		Args:
			states: States with non-empty output each.
			reranker_lm: The scoring LM.

		Returns:
			List of scores in [0, 1], aligned to input order.
		"""
		if not states:
			return []

		reasoning_candidates: list[str] = []
		output_candidates: list[dict[str, Any]] = []
		include_reasoning_in_document_for_output = self.include_reasoning_in_document_for_output
		for state in states:
			if not state.output:
				raise ValueError("ORM evaluation requires a non-empty final output.")
			reasoning_candidates.append(
				self._thinking_only_from_state(state)
				if self.consider_reasoning_in_final_eval
				else ""
			)
			output_candidates.append({k: v for k, v in dict(state.output).items() if k != "error"})

		if self.consider_reasoning_in_final_eval and any(rc.strip() for rc in reasoning_candidates):
			include_reasoning_in_document_for_output = True

		# Pre-build per-query instructions, then score the entire batch in one adapter call.
		inputs = dict(states[0].input)
		assert all(inputs == state.input for state in states[1:]), "All states must have the same input"
		instructions_per_query: list[str] = []
		for reasoning_str in reasoning_candidates:
			is_reasoning_empty = not bool(reasoning_str.strip())
			consider_reasoning_in_output = self.consider_reasoning_in_final_eval
			instructions_per_query.append(
				self.build_scoring_instructions(
					scoring_target="output",
					rubric=self.orm_rubric,
					consider_reasoning_in_output=consider_reasoning_in_output,
					is_reasoning_empty=(is_reasoning_empty or not consider_reasoning_in_output),
				)
			)

		responses = self.scoring_adapter(
			instructions=instructions_per_query,
			lm=reranker_lm,
			inputs=inputs,
			scoring_target="output",
			reasoning_candidates=reasoning_candidates,
			output_candidates=output_candidates,
			include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
		)
		assert len(responses) == len(states), (
			f"Expected {len(states)} rerank responses but got {len(responses)}"
		)
		scores: list[float] = []
		for resp in responses:
			assert resp.results, "No reranker results returned for ORM scoring"
			scores.append(float(resp.results[0].relevance_score))
		return scores

	@staticmethod
	def _as_evaluation_result(score: float) -> EvaluationResult:
		"""
		Wrap a reranker score into the standard EvaluationResult container.

		Args:
			score: Reranker relevance score in [0, 1].

		Returns:
			EvaluationResult with a single JudgeEvaluation.
		"""
		score_f = float(score)
		if not (0.0 <= score_f <= 1.0):
			raise ValueError(f"Reranker score must be in [0, 1], got {score_f}")
		judge = JudgeEvaluation(
			raw_scores={"relevance": score_f},
			normalized_scores={"relevance": score_f},
		)
		return EvaluationResult(score=score_f, judge_evaluations=[judge])

	def forward(
		self,
		states: State | list[State],
		n_samples_evaluator: int = 1,
		**kwargs: Any,  # noqa: ARG002
	) -> list[list[EvaluationResult]]:
		"""
		Evaluate a state or list of states using deterministic reranker scoring.

		Notes:
			- `n_samples_evaluator` is accepted for API compatibility but reranker evaluation is
			  deterministic; single-judge semantics are always used.
			- Sampling-related kwargs (temperature, max_tokens, etc.) are ignored for reranking.

		Args:
			states: A State or list of States.
			n_samples_evaluator: Ignored beyond validating it is >= 1 (compatibility).
			**kwargs: Ignored for reranker evaluation.

		Returns:
			List[list[EvaluationResult]]: One list per input state, always containing exactly one
			EvaluationResult.
		"""
		if n_samples_evaluator < 1:
			raise ValueError("n_samples_evaluator must be >= 1")

		states_list = states if isinstance(states, list) else [states]
		if len(states_list) > 1:
			for state in states_list:
				assert state.input == states_list[0].input, (
					"All states must have the same input (keys and values) when evaluating in batch. "
					f"Expected input={states_list[0].input}, got input={state.input}."
				)
		reranker_lm = self.get_lm()

		prm_indices: list[int] = []
		prm_states: list[State] = []
		orm_indices: list[int] = []
		orm_states: list[State] = []
		for i, state in enumerate(states_list):
			if state.output:
				orm_indices.append(i)
				orm_states.append(state)
			else:
				prm_indices.append(i)
				prm_states.append(state)

		final: list[list[EvaluationResult] | None] = [None] * len(states_list)
		# TODO[P3]: Add support for batching PRM and ORM calls simultaneously.
		if prm_states:
			prm_scores = self._score_batch_prm(states=prm_states, reranker_lm=reranker_lm)
			for idx, score in zip(prm_indices, prm_scores, strict=True):
				final[idx] = [self._as_evaluation_result(score)]
		if orm_states:
			orm_scores = self._score_batch_orm(states=orm_states, reranker_lm=reranker_lm)
			for idx, score in zip(orm_indices, orm_scores, strict=True):
				final[idx] = [self._as_evaluation_result(score)]

		assert all(x is not None for x in final), "Internal error: missing evaluation result"
		return final  # pyright: ignore[reportReturnType]

