"""
TreeOfThoughtEvaluator: Simplified evaluator for Tree-of-Thought reasoning.

This module implements the TreeOfThoughtEvaluator class for Tree-of-Thought reasoning.
Uses three specialized LocalPredict instances with auto-batching for efficient evaluation.
"""

# Standard library imports
import copy
import logging
from typing import Annotated, Any, Literal

# Third-party imports
import annotated_types
import dspy
import numpy as np
import pydantic
from dspy.adapters.utils import get_annotation_name
from dspy.primitives.prediction import Prediction
from vllm import SamplingParams

# Local imports
from lm.lm_constants import ENABLE_THINKING
from misc_utils import ExecutionError, format_list_of_fields
from predict.local_predict import LocalPredict
from signatures import (
	ReasoningSignature,
	ensure_reasoning_signature,
)
from tree import (
	EvaluationResult,
	JudgeEvaluation,
	State,
)

logger = logging.getLogger(__name__)


def _format_rubric_dimensions_for_prompt(rubric: dspy.Signature) -> str:
	"""
	Return a model-facing rubric section built only from output-field names + descriptions.

	Args:
		rubric: The DSPy signature for rubric-based scoring.

	Returns:
		A string containing the formatted rubric.
	"""
	lines: list[str] = ["Rubric:"]
	for field_name, field_info in rubric.output_fields.items():
		description = getattr(field_info, "description", None)
		if description is None:
			json_schema_extra = getattr(field_info, "json_schema_extra", None)
			if isinstance(json_schema_extra, dict):
				description = json_schema_extra.get("desc")
		if isinstance(description, str) and description.strip():
			lines.append(f"- {field_name}: {description.strip()}")
		else:
			lines.append(f"- {field_name}")
	return "\n".join(lines).strip()


def _format_numeric_rubric_for_prompt(
	rubric: dspy.Signature,
	numeric_fields: list[str],
	dimension_bounds: dict[str, tuple[float, float]],
) -> str:
	"""
	Return rubric bullets for numeric score dimensions, including type + bounds.

	Each bullet uses:
	`- <field_name>: <description> (a/an <type> between <lower> and <upper>).`
	"""
	lines: list[str] = []
	for field_name in numeric_fields:
		field_info = rubric.output_fields.get(field_name)
		description: str | None = None
		if field_info is not None:
			description = getattr(field_info, "description", None)
			if description is None:
				json_schema_extra = getattr(field_info, "json_schema_extra", None)
				if isinstance(json_schema_extra, dict):
					description = json_schema_extra.get("desc")
		description_str = (
			description.strip().rstrip(".")
			if isinstance(description, str) and description.strip()
			else field_name
		)

		lower, upper = dimension_bounds[field_name]
		annotation = rubric.output_fields[field_name].annotation
		annotation_name = get_annotation_name(annotation).lower()
		if "float" in annotation_name:
			type_name = "float"
		elif "int" in annotation_name:
			type_name = "int"
		else:
			type_name = "number"
		article = "an" if type_name[:1] in {"a", "e", "i", "o", "u"} else "a"
		lines.append(
			f"- {field_name}: {description_str} ({article} {type_name} between {lower} and {upper})."
		)

	return "\n".join(lines).strip()


class TreeOfThoughtEvaluator(dspy.Module):
	"""
	A module for evaluating reasoning steps and final outputs in Tree-of-Thought reasoning.

	Uses three specialized LocalPredict instances for different evaluation contexts:
	- First Step: Evaluates initial reasoning step quality and promise
	- Subsequent Step: Evaluates reasoning progression and coherence
	- Final Step: Evaluates final response quality and completeness

	Supports efficient batch processing and state management for Tree-of-Thought workflows.
	"""

	# TODO[P2][Till]: Add support for voting across multiple evaluations
	# Potential mixed approach: scores for preselection, then 'rank' to select top k from preselected
	# TODO[P3][Till]: Create custom adapter to support both PRM and ORM calls in a single batch (similar to generator)
	def __init__(
		self,
		generator_signature: type[ReasoningSignature],
		evaluator_signature: type[ReasoningSignature] | None = None,
		evaluator_signature_prm: type[ReasoningSignature] | None = None,
		evaluator_signature_orm: type[ReasoningSignature] | None = None,
		consider_reasoning_in_final_eval: bool = False,
		verbosity: Literal["debug", "info", "warning", "error"] = "warning",
	) -> None:
		"""
		Initialize the TreeOfThoughtEvaluator.

		Parameters:
		    generator_signature (ReasoningSignature): The signature of the generator being evaluated.
		    evaluator_signature (ReasoningSignature | None): Default custom evaluator signature (rubric)
		        defining evaluation criteria and score dimensions for both PRM and ORM evaluation.
		        If None, uses default signatures (soundness+promise for PRM, quality for ORM).
		        This signature should contain only output fields (score dimensions). Use the
		        rubric_weight parameter in OutputField() to specify dimension weights.
		    evaluator_signature_prm (ReasoningSignature | None): Optional custom evaluator signature (rubric)
		        specifically for PRM (Process Reward Model) evaluation. If provided, overrides
		        evaluator_signature for PRM only. If None, PRM uses evaluator_signature (or default).
		        Use rubric_weight in OutputField() to specify dimension weights.
		    evaluator_signature_orm (ReasoningSignature | None): Optional custom evaluator signature (rubric)
		        specifically for ORM (Outcome Reward Model) evaluation. If provided, overrides
		        evaluator_signature for ORM only. If None, ORM uses evaluator_signature (or default).
		        Use rubric_weight in OutputField() to specify dimension weights.
		    consider_reasoning_in_final_eval (bool): Whether to include reasoning in outcome evaluation.
		    verbosity (str): Verbosity level for logging ("debug", "info", "warning", "error").

		Note:
		    - Demos must be provided explicitly via the demos parameter in forward()/call.
		      If no demos are provided, zero-shot evaluation is used.
		    - Dimension weights are specified directly in the signature using the rubric_weight parameter:
		        soundness: float = dspy.OutputField(desc="...", ge=0.0, le=10.0, rubric_weight=0.7)
		        promise: float = dspy.OutputField(desc="...", ge=0.0, le=10.0, rubric_weight=0.3)
		      If no rubric_weight is specified, equal weights are used for all dimensions.
		"""
		super().__init__()

		self.generator_signature = ensure_reasoning_signature(generator_signature)
		self.consider_reasoning_in_final_eval = consider_reasoning_in_final_eval
		self._verbosity = verbosity

		# Set logger level based on verbosity
		logging_level = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
		}.get(verbosity, logging.WARNING)
		logger.setLevel(logging_level)

		# Extract field information
		if not self.generator_signature.reasoning_fields:
			raise ValueError("Generator signature must have at least one reasoning field")
		self.generator_input_field_names = list(self.generator_signature.input_fields.keys())
		self.generator_output_field_names = list(self.generator_signature.output_fields.keys())
		self.reasoning_field_name = list(self.generator_signature.reasoning_fields.keys())[0]
		self.reasoning_field = self.generator_signature.reasoning_fields.get(self.reasoning_field_name)

		# Process evaluator signatures (custom or default)
		# Determine PRM rubric: evaluator_signature_prm takes precedence, then evaluator_signature, then default
		# Type annotation: rubrics are always dspy.Signature instances after normalization
		self.prm_rubric: dspy.Signature
		self.orm_rubric: dspy.Signature

		if evaluator_signature_prm is not None:
			self.prm_rubric = dspy.ensure_signature(evaluator_signature_prm)
			self.is_custom_prm_signature = True
		elif evaluator_signature is not None:
			# Backward compatibility: use evaluator_signature for PRM if no specific PRM signature provided
			self.prm_rubric = dspy.ensure_signature(evaluator_signature)
			self.is_custom_prm_signature = True
		else:
			# Use default PRM rubric (soundness + promise)
			self.prm_rubric = self._create_default_prm_rubric()
			self.is_custom_prm_signature = False

		# Determine ORM rubric: evaluator_signature_orm → evaluator_signature → default
		if evaluator_signature_orm is not None:
			self.orm_rubric = dspy.ensure_signature(evaluator_signature_orm)
			self.is_custom_orm_signature = True
		elif evaluator_signature is not None:
			# Backward compatibility: use evaluator_signature for ORM
			self.orm_rubric = dspy.ensure_signature(evaluator_signature)
			self.is_custom_orm_signature = True
		else:
			# Use default ORM rubric (quality)
			self.orm_rubric = self._create_default_orm_rubric()
			self.is_custom_orm_signature = False

		# Process output fields separately for PRM and ORM rubrics
		# Extract and store field metadata for PRM (including normalized weights)
		(
			self.prm_numeric_score_fields,
			self.prm_dimension_bounds,
			self.prm_dimension_weights,
		) = self._process_evaluator_output_fields(self.prm_rubric)

		# Extract and store field metadata for ORM (including normalized weights)
		(
			self.orm_numeric_score_fields,
			self.orm_dimension_bounds,
			self.orm_dimension_weights,
		) = self._process_evaluator_output_fields(self.orm_rubric)

		# Initialize PRM and ORM evaluator predictors
		self.process_evaluator = LocalPredict(
			signature=self._create_process_evaluator_signature(),
			verbose=verbosity,
		)
		self.outcome_evaluator = LocalPredict(
			signature=self._create_outcome_evaluator_signature(),
			verbose=verbosity,
		)

	@property
	def verbosity(self) -> Literal["debug", "info", "warning", "error"]:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Literal["debug", "info", "warning", "error"]) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logging_level = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
		}.get(verbosity, logging.WARNING)
		logger.setLevel(logging_level)

	def _create_default_prm_rubric(self) -> dspy.Signature:
		"""
		Create the default PRM (Process Reward Model) rubric with soundness and promise scores.

		PRM evaluates reasoning trajectories, so we need dual scores:
		- soundness: backward-looking correctness (is this step logically valid?)
		- promise: forward-looking trajectory quality (is this leading toward a solution?)

		Returns:
		    dspy.Signature: A signature with soundness and promise (both 0-10 float) output fields.
		"""
		signature_fields = {}

		# Add soundness output field with bounds metadata
		signature_fields["soundness"] = (
			Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)],
			dspy.OutputField(desc="Logical validity, factual accuracy, and coherence with prior steps")
		)

		# Add promise output field with bounds metadata
		signature_fields["promise"] = (
			Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)],
			dspy.OutputField(desc="Likelihood of the reasoning to lead to a strong final answer")
		)

		instructions = (
			"Score the reasoning step on two dimensions: soundness and promise. "
			"Soundness refers to correctness of the reasoning so far, while promise refers to the "
			"likelihood that the reasoning will lead to a strong final answer. "
			"Return only numeric floats in the exact format below, and nothing else:\n\n"
			"## soundness\n<NUMBER>\n\n## promise\n<NUMBER>"
		)

		return dspy.Signature(signature_fields, instructions)

	def _create_default_orm_rubric(self) -> dspy.Signature:
		"""
		Create the default ORM (Outcome Reward Model) rubric with single unified quality score.

		ORM evaluates final solutions, where we assess overall correctness and quality
		with a single holistic score rather than separate soundness/promise dimensions.

		Returns:
		    dspy.Signature: A signature with quality (0-10 scale) output field.
		"""
		signature_fields = {}

		# Add quality output field with bounds metadata
		signature_fields["quality"] = (
			Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)],
			dspy.OutputField(
				desc=(
					"The overall quality of the final answer. Judge the quality based on correctness "
					"(accurate, logically sound, and follows the user's instructions), "
					"completeness (all required elements present), "
					"and clarity (well-structured with appropriate tone and style). "
					"Higher scores indicate strong performance across these dimensions. "
					"Any flaws along these dimensions should be penalized."
				),
			),
		)

		instructions = (
			"Score the final output with a single quality metric (i.e., correctness, completeness, clarity, etc...). "
			"Return only a numeric float in the exact format below, and nothing else:\n\n"
			"## quality\n<NUMBER>"
		)

		return dspy.Signature(signature_fields, instructions)

	def _process_evaluator_output_fields(
		self,
		signature: type[dspy.Signature],
		default_lower_bound: float = 0.0,
		default_upper_bound: float = 10.0,
	) -> tuple[list[str], dict[str, tuple[float, float]], dict[str, float]]:
		"""
		Process evaluator output fields to identify, validate, and extract metadata.

		This method does a single pass through output fields to:
		1. Identify numeric score fields
		2. Validate at least one numeric output exists (i.e., at least 1 numeric metric).
		3. Extract bounds from metadata (or use defaults if not specified)
		4. Extract and validate rubric_weight from field metadata
		5. Generate equal weights if no rubric_weight specified

		Parameters:
		    signature (dspy.Signature): The evaluator signature to process.
		    default_lower_bound (float): Default lower bound if not specified in metadata.
		    default_upper_bound (float): Default upper bound if not specified in metadata.

		Returns:
		    Tuple[List[str], Dict[str, Tuple[int, int]], Dict[str, float]]:
		        - List of numeric score field names
		        - Dictionary mapping numeric field names to (lower, upper) bounds
		        - Dictionary mapping numeric field names to normalized weights (sum to 1.0)

		Raises:
			AssertionError: If evaluator signature does not have at least one numeric (int or float)
				score field, or if rubric_weight is specified for some but not all numeric fields.

		"""
		numeric_score_fields: list[str] = []
		dimension_bounds: dict[str, tuple[float, float]] = {}
		dimension_weights: dict[str, float] = {}

		for name, field in signature.output_fields.items():
			field: pydantic.fields.FieldInfo
			field_type = field.annotation

			if field_type in (float, int):
				numeric_score_fields.append(name)

				# Extract bounds from metadata, or use defaults if not present
				ge_bound = next(
					(meta.ge for meta in field.metadata if hasattr(meta, "ge")),
					default_lower_bound,
				)
				le_bound = next(
					(meta.le for meta in field.metadata if hasattr(meta, "le")),
					default_upper_bound,
				)
				dimension_bounds[name] = (ge_bound, le_bound)
				if ge_bound >= le_bound:
					raise ValueError(
						"Lower bound must be less than upper bound, "
						f"but got ge={ge_bound} and le={le_bound} for field '{name}'."
					)

				# Extract rubric_weight from json_schema_extra if present
				if hasattr(field, "json_schema_extra") and field.json_schema_extra:
					rubric_weight = field.json_schema_extra.get("rubric_weight")
					if rubric_weight is not None:
						dimension_weights[name] = float(rubric_weight)

			elif field_type is str:
				raise ValueError(
					"Evaluator signatures must be numeric-only; string output fields are not "
					f"supported. Found string output field '{name}'."
				)

		# Validate required fields exist
		if not numeric_score_fields:
			raise ValueError(
				"Evaluator signatures must be numeric-only and must include at least one numeric output "
				"field (float or int) for scoring."
			)
		# Process rubric_weights: either all fields have them, or none do
		if dimension_weights:
			# Some fields have rubric_weight - validate ALL fields have it
			missing_weights = set(numeric_score_fields) - set(dimension_weights.keys())
			if missing_weights:
				raise ValueError(
					f"If rubric_weight is specified for any dimension, it must be specified for all. "
					f"Missing rubric_weight for: {missing_weights}"
				)
			# Validate all weights are positive
			if not all(w > 0 for w in dimension_weights.values()):
				raise ValueError(
					f"All rubric_weight values must be positive. Given weights: {dimension_weights}"
				)
			# Normalize weights to sum to 1.0
			total = sum(dimension_weights.values())
			dimension_weights = {k: v / total for k, v in dimension_weights.items()}
		else:
			# No rubric_weight specified - use equal weights
			n = len(numeric_score_fields)
			dimension_weights = dict.fromkeys(numeric_score_fields, 1.0 / n)

		return numeric_score_fields, dimension_bounds, dimension_weights

	def _create_process_evaluator_signature(self) -> dspy.Signature:
		"""
		Create signature for PRM (Process Reward Model) - evaluating reasoning step quality.

		Uses the PRM rubric to construct a PRM-specific signature by adding
		generator input fields and reasoning_trajectory field. Default PRM rubric uses
		soundness + promise dual scores.

		Returns:
		    dspy.Signature: PRM evaluation signature with inputs and score dimensions from PRM rubric.
		"""
		generator_inputs = format_list_of_fields(self.generator_input_field_names)
		generator_outputs = format_list_of_fields(self.generator_output_field_names)

		prm_rubric_text = _format_numeric_rubric_for_prompt(
			rubric=self.prm_rubric,
			numeric_fields=self.prm_numeric_score_fields,
			dimension_bounds=self.prm_dimension_bounds,
		)
		prm_instructions = f"""
Judge the quality of reasoning steps for a problem-solving process.
The task requires producing {generator_outputs} given {generator_inputs}.
Reasoning steps towards producing {generator_outputs} are provided in `reasoning_steps`.

Since you are evaluating intermediate reasoning, don't score based on completeness, and instead score based on the rubric items below:
{prm_rubric_text}
""".strip()

		# Create signature fields dictionary
		signature_fields = {}

		# Add base input fields from generator
		for name, field in self.generator_signature.input_fields.items():
			field_type = (
				field.annotation
				if hasattr(field, "annotation") and field.annotation
				else str
			)
			signature_fields[name] = (field_type, field)

		# Extract type and description from generator's reasoning field for more precise specification
		reasoning_element_type = self.reasoning_field.annotation or str
		# Use the reasoning field name as the description
		reasoning_description = self.reasoning_field_name
		signature_fields["reasoning_steps"] = (
			list[reasoning_element_type],
			dspy.InputField(
				desc=f"List of '{reasoning_description}'s to evaluate toward producing {generator_outputs}",
				annotation=list[reasoning_element_type],
			),
		)

		# Add output fields from PRM rubric
		for name, field in self.prm_rubric.output_fields.items():
			field_type = field.annotation
			signature_fields[name] = (field_type, field)

		return dspy.Signature(signature_fields, prm_instructions)

	def _create_outcome_evaluator_signature(self) -> dspy.Signature:
		"""
		Create signature for ORM (Outcome Reward Model) - evaluating final solution quality.

		Uses the ORM rubric to construct an ORM-specific signature by adding
		generator input fields, generator output fields, and optionally reasoning_trajectory field.
		Default ORM rubric uses a single unified quality score.

		Returns:
		    dspy.Signature: ORM evaluation signature with inputs and score dimensions from ORM rubric.
		"""
		generator_inputs = format_list_of_fields(self.generator_input_field_names)
		generator_outputs = format_list_of_fields(self.generator_output_field_names)

		orm_evaluation_target = (
			"response"
			if not self.consider_reasoning_in_final_eval
			else "response and the reasoning steps that led to it"
		)

		orm_rubric_text = _format_numeric_rubric_for_prompt(
			rubric=self.orm_rubric,
			numeric_fields=self.orm_numeric_score_fields,
			dimension_bounds=self.orm_dimension_bounds,
		)
		reasoning_context_note = (
			"\n\nThe reasoning steps that led to this solution are stored in `reasoning_steps` for context."
			if self.consider_reasoning_in_final_eval
			else ""
		)
		orm_instructions = f"""
Judge the quality of a response for the provided task.
The task requires producing {generator_outputs} given {generator_inputs}.

Evaluate the {orm_evaluation_target} using the rubric items below and assign numeric scores to each:
{orm_rubric_text}{reasoning_context_note}
""".strip()

		# Create signature fields dictionary
		signature_fields = {}

		# Add base input fields from generator
		for name, field in self.generator_signature.input_fields.items():
			field_type = (
				field.annotation
				if hasattr(field, "annotation") and field.annotation
				else str
			)
			signature_fields[name] = (field_type, field)

		# Add generator output fields as inputs for ORM evaluation
		for field_name, field_info in self.generator_signature.output_fields.items():
			field_type = (
				field_info.annotation
				if hasattr(field_info, "annotation") and field_info.annotation
				else str
			)
			# Preserve the original field description from the generator
			original_desc = field_info.description
			if original_desc is None:
				# Fall back to json_schema_extra['desc'] if description is None
				json_schema_extra = getattr(field_info, "json_schema_extra", None)
				if isinstance(json_schema_extra, dict):
					original_desc = json_schema_extra.get("desc")
			signature_fields[field_name] = (field_type, dspy.InputField(desc=original_desc, annotation=field_type))

		# Add reasoning steps if configured (as optional field)
		if self.consider_reasoning_in_final_eval:
			# Extract type and description from generator's reasoning field for more precise specification
			reasoning_element_type = self.reasoning_field.annotation or str
			signature_fields["reasoning_steps"] = (
				list[reasoning_element_type],
				dspy.InputField(
					desc=f"List of '{self.reasoning_field_name}'s that led to the final {generator_outputs}",
					annotation=list[reasoning_element_type],
					default=[],
				),
			)

		# Add output fields from ORM rubric
		for name, field in self.orm_rubric.output_fields.items():
			field_type = field.annotation
			signature_fields[name] = (field_type, field)

		return dspy.Signature(signature_fields, orm_instructions)

	def _normalize_score(
		self, score: float, lower_bound: float, upper_bound: float
	) -> float:
		"""
		Normalize a score to the range [0, 1].

		Parameters:
		    score (float): The score to normalize.
		    lower_bound (float): The lower bound of the score range.
		    upper_bound (float): The upper bound of the score range.

		Returns:
		    float: The normalized score in [0, 1].
		"""
		return (score - lower_bound) / (upper_bound - lower_bound)

	def _consolidate_scores(
		self,
		predictions: list[Prediction],
		rubric: dspy.Signature,
		n_samples_judge: int,
	) -> list[EvaluationResult]:
		"""
		Consolidate scores across multiple judges and dimensions for each prediction.

		When using multiple dimensions or multiple judges, this method:
		1. Extracts scores for each dimension across all judges
		2. Computes average score per dimension
		3. Normalizes scores using dimension bounds
		4. Computes a total score as the weighted average across all dimensions (using dimension_weights)
		5. Preserves individual judge evaluations with both raw and normalized scores

		Parameters:
		    predictions (List[Prediction]): Predictions from the evaluator.
		    rubric (dspy.Signature): The rubric signature (PRM or ORM) used for this evaluation.

		Returns:
		    List[EvaluationResult]: List of consolidated evaluation results.
		        Each EvaluationResult contains:
		        - score: consolidated total score (weighted average across dimensions and judges)
		        - judge_evaluations: list of JudgeEvaluation instances with raw and normalized scores
		"""
		output_results = []

		# Use cached field metadata based on rubric type
		if rubric == self.prm_rubric:
			numeric_score_fields = self.prm_numeric_score_fields
			dimension_bounds = self.prm_dimension_bounds
			rubric_dimension_weights = self.prm_dimension_weights
		elif rubric == self.orm_rubric:
			numeric_score_fields = self.orm_numeric_score_fields
			dimension_bounds = self.orm_dimension_bounds
			rubric_dimension_weights = self.orm_dimension_weights
		else:
			# Fallback: recompute from rubric (should not happen in normal usage)
			numeric_score_fields, dimension_bounds, rubric_dimension_weights = (
				self._process_evaluator_output_fields(rubric)
			)
			n = len(numeric_score_fields)
			rubric_dimension_weights = dict.fromkeys(numeric_score_fields, 1.0 / n)

		for prediction in predictions:
			completions = prediction.completions
			score_dimension_names = list(numeric_score_fields)

			# Required: all numeric score fields must be present.
			dimension_scores_by_dim: dict[str, list[Any]] = {}
			for dim in score_dimension_names:
				dim_scores = completions[dim]
				if not isinstance(dim_scores, list):	# If using only a single judge.
					dim_scores = [dim_scores]
				dimension_scores_by_dim[dim] = dim_scores

			# Candidate judge indices are those that:
			# - exist for every required dimension
			# - were not marked as failed parses by the adapter
			#
			# If parsing succeeded, DSPy guarantees all required output fields were assigned.
			# All completions include an ExecutionError object; check if error_type is None (success)
			errors = completions["error"]
			errors: list[ExecutionError] = errors if isinstance(errors, list) else [errors]
			failed_flags = [err.has_error() for err in errors]

			candidate_judge_indices: list[int] = [
				i for i in range(n_samples_judge)
				if all(i < len(dimension_scores_by_dim[dim]) for dim in score_dimension_names)
				and not failed_flags[i]
			]

			valid_judge_scores: list[float] = []
			valid_judge_evaluations: list[JudgeEvaluation] = []
			for judge_idx in candidate_judge_indices:
				try:
					raw_scores: dict[str, float] = {}
					normalized_scores: dict[str, float] = {}
					for dim in score_dimension_names:
						raw_score = float(dimension_scores_by_dim[dim][judge_idx])
						lower, upper = dimension_bounds[dim]
						if not (lower <= raw_score <= upper):
							raise ValueError(
								f"Score for dimension '{dim}' is out of bounds: {raw_score} "
								f"(expected range: [{lower}, {upper}])"
							)
						raw_scores[dim] = raw_score
						normalized_scores[dim] = self._normalize_score(raw_score, lower, upper)
					weighted_sum = sum(
						normalized_scores[dim] * rubric_dimension_weights[dim]
						for dim in score_dimension_names
					)
					valid_judge_scores.append(float(weighted_sum))
					valid_judge_evaluations.append(
						JudgeEvaluation(
							raw_scores=raw_scores,
							normalized_scores=normalized_scores,
						)
					)
				except Exception as e:
					logger.warning(f"Dropped failed judge evaluation (judge_idx={judge_idx}): {e}")
					continue

			# If all judges failed for this prediction:
			if not valid_judge_scores:
				logger.warning(
					"All judges failed to produce valid evaluator outputs for a prediction; "
					"assigning score=0.0."
				)
				output_results.append(EvaluationResult(score=0.0, judge_evaluations=[]))
				continue

			output_results.append(
				EvaluationResult(
					score=float(np.mean(valid_judge_scores)),
					judge_evaluations=valid_judge_evaluations,
				)
			)

		return output_results

	def _state_to_evaluator_input(
		self, state: State, evaluation_type: Literal["process", "outcome"]
	) -> dict[str, Any]:
		"""
		Convert the state to input for the evaluator.

		Parameters:
		    state (State): The state to convert.
		    evaluation_type (str): The evaluation type ("process" or "outcome").

		Returns:
		    Dict[str, Any]: The input dictionary for the evaluator.
		"""
		# Start with base input fields
		evaluator_input = copy.deepcopy(state.input)

		# Get reasoning steps - use reasoning field name from generator signature
		existing_reasoning_trajectory = self._extract_reasoning_values(
			reasoning_steps=state.reasoning,
			controller_output_trajectory=state.controller_output_trajectory,
		)
		if evaluation_type == "process":
			# PRM requires at least one reasoning step to evaluate
			if not existing_reasoning_trajectory:
				raise ValueError(
					f"PRM evaluation requires at least one reasoning step. "
					f"Looking for reasoning field '{self.reasoning_field_name}' in state.reasoning, "
					f"but no reasoning steps were found. "
					f"State input: {state.input}, State output: {state.output}"
				)
			evaluator_input["reasoning_steps"] = existing_reasoning_trajectory

		elif evaluation_type == "outcome":
			# ORM: evaluate final solution quality
			# Don't include metadata keys (e.g. ExecutionError) as part of the candidate output.
			for k, v in state.output.items():
				if k == "error":
					continue
				evaluator_input[k] = v
			# TODO[P1]: Refactor error-handling so that the adapter returns either a
			#  dictionary or an exception object, rather than embedding errors in the dict.
			if not any(k != "error" for k in state.output):
				logger.warning(
					"Output contained only an error key and no scorable fields. "
					f"State input: {state.input}"
				)
			if self.consider_reasoning_in_final_eval:
				evaluator_input["reasoning_steps"] = existing_reasoning_trajectory

		return evaluator_input

	def _extract_reasoning_values(
		self,
		reasoning_steps: list[dict[str, Any]],
		controller_output_trajectory: list[Any] | None = None,
	) -> list[Any]:
		values: list[str] = []
		controller_output_trajectory = controller_output_trajectory or []
		for i, step in enumerate(reasoning_steps):
			assert self.reasoning_field_name in step, (
				f"Missing reasoning field '{self.reasoning_field_name}' in step: {step}"
			)
			parts: list[str] = []
			if i < len(controller_output_trajectory):
				internal_reasoning = getattr(
					controller_output_trajectory[i],
					"internal_reasoning",
					"",
				)
				internal_reasoning = str(internal_reasoning).strip()
				if internal_reasoning:
					parts.append(f"internal_reasoning: {internal_reasoning}")
			step_value = str(step[self.reasoning_field_name]).strip()
			if parts:
				values.append("\n".join([*parts, step_value]).strip())
			else:
				values.append(step_value)
		return values

	def _get_evaluator_for_type(self, evaluation_type: Literal["process", "outcome"]) -> LocalPredict:
		"""Get the appropriate evaluator for the given type."""
		if evaluation_type == "process":
			return self.process_evaluator
		elif evaluation_type == "outcome":
			return self.outcome_evaluator
		else:
			raise ValueError(f"Unknown evaluation type: {evaluation_type}")


	def forward(
		self,
		states: State | list[State],
		n_samples_evaluator: int = 1,
		evaluator_temperature: float | None = None,
		evaluator_top_p: float | None = None,
		evaluator_top_k: int | None = None,
		evaluator_min_p: float | None = None,
		evaluator_use_beam_search: bool = False,
		evaluator_max_tokens: int | None = None,
		demos: list[dict[str, Any]] | None = None,
		demos_prm: list[dict[str, Any]] | None = None,
		demos_orm: list[dict[str, Any]] | None = None,
		**kwargs: Any,
	) -> list[list[EvaluationResult]]:
		"""
		Forward method that automatically evaluates states using PRM/ORM based on completion.

		PRM (Process Reward Model): Used for states without outputs (evaluates reasoning quality)
		ORM (Outcome Reward Model): Used for states with outputs (evaluates solution quality)

		Parameters:
		    states (Union[State, List[State]]): Single state or list of states to evaluate.
		    n_samples_evaluator (int): Number of generations per state (supports future voting).
		    evaluator_temperature (float | None): Temperature for judge sampling.
		    evaluator_top_p (float | None): Top-p (nucleus) sampling parameter.
		    evaluator_top_k (int | None): Top-k sampling parameter.
		    evaluator_min_p (float | None): Min-p sampling parameter.
		    evaluator_use_beam_search (bool): Whether to use beam search.
		    evaluator_max_tokens (int | None): Maximum tokens per evaluation.
		    demos (Optional[List[Dict[str, Any]]]): Default demo examples for both PRM and ORM evaluation.
		        If None, uses default demos based on signature type for each evaluation type.
		        To use no demos for both, pass an empty list [].
		    demos_prm (Optional[List[Dict[str, Any]]]): Demo examples specifically for PRM (process) evaluation.
		        If provided, overrides `demos` for PRM evaluation only. If None, PRM uses `demos` (or defaults).
		    demos_orm (Optional[List[Dict[str, Any]]]): Demo examples specifically for ORM (outcome) evaluation.
		        If provided, overrides `demos` for ORM evaluation only. If None, ORM uses `demos` (or defaults).

		Returns:
		    List[List[EvaluationResult]]: List of evaluation results for each state
		        - For single state: [results_for_state]
		        - For multiple states: [results_for_state1, results_for_state2, ...]
		        Each results_for_state contains EvaluationResult instances with scores and feedback.
		"""
		states = [states] if not isinstance(states, list) else states

		# Build vLLM sampling config: forward args take priority, kwargs are fallback.
		# If a forward arg is left as its default, allow kwargs to override it.
		config: dict[str, Any] = {
			"n": n_samples_evaluator,
			"temperature": evaluator_temperature,
			"max_tokens": evaluator_max_tokens,
			"top_p": evaluator_top_p,
			"top_k": evaluator_top_k,
			"min_p": evaluator_min_p,
			"use_beam_search": evaluator_use_beam_search,
		}
		# Only allow known/intentional keys to flow into LocalPredict config.
		allowed_extra_config_keys = set(SamplingParams.__annotations__) | {
			"chat_template",
			"chat_template_content_format",
			"chat_template_kwargs",
			"tools",
			"use_tqdm",
		}
		for k, v in kwargs.items():
			if k not in allowed_extra_config_keys:
				continue
			if (k not in config) or (config[k] is None):
				config[k] = v

		# Divide states into PRM and ORM groups based on completion status
		process_states_info = []  # (index, state) pairs for PRM evaluation
		outcome_states_info = []  # (index, state) pairs for ORM evaluation

		for i, state in enumerate(states):
			evaluation_type = "outcome" if state.output else "process"
			if evaluation_type == "process":
				process_states_info.append((i, state))
			else:
				outcome_states_info.append((i, state))

		# Initialize results in original order
		final_results: list[list[EvaluationResult] | None] = [None] * len(states)
		chat_template_kwargs = dict(config.get("chat_template_kwargs") or {})
		chat_template_kwargs.setdefault(ENABLE_THINKING, False)
		config["chat_template_kwargs"] = chat_template_kwargs

		# Process PRM batch if any
		if process_states_info:
			process_states = [info[1] for info in process_states_info]

			# Convert states to evaluator inputs
			evaluator_inputs = [
				self._state_to_evaluator_input(state, "process")
				for state in process_states
			]

			# Create batched kwargs
			field_names = list(evaluator_inputs[0].keys())
			evaluator_inputs_batched_kwargs = {}
			for field_name in field_names:
				evaluator_inputs_batched_kwargs[field_name] = [
					input_dict[field_name] for input_dict in evaluator_inputs
				]

			# Run PRM evaluator with appropriate demos
			# Use demos_prm if provided, otherwise fall back to demos
			process_demos = demos_prm if demos_prm is not None else demos
			if process_demos is not None:
				logger.info(f"Using {len(process_demos)} user-provided demos for PRM evaluation")
			else:
				logger.info("Using zero-shot evaluation for PRM (no demos provided)")
			process_evaluator = self._get_evaluator_for_type("process")
			process_predictions = process_evaluator(
				config=config,
				demos=process_demos,
				**evaluator_inputs_batched_kwargs,
			)
			process_output_dicts = self._consolidate_scores(
				process_predictions,
				self.prm_rubric,
				n_samples_judge=n_samples_evaluator,
			)

			# Group and assign results
			evaluations_per_state = (
				len(process_output_dicts) // len(process_states)
				if process_states
				else 0
			)

			for i, (original_idx, _) in enumerate(process_states_info):
				start_idx = i * evaluations_per_state
				end_idx = start_idx + evaluations_per_state
				state_evaluations = process_output_dicts[start_idx:end_idx]
				final_results[original_idx] = state_evaluations

		# Process ORM batch if any
		if outcome_states_info:
			outcome_states = [info[1] for info in outcome_states_info]

			# Convert states to evaluator inputs
			outcome_batch_inputs = [
				self._state_to_evaluator_input(state, "outcome")
				for state in outcome_states
			]

			# Create batched kwargs
			field_names = list(outcome_batch_inputs[0].keys())
			outcome_batched_kwargs = {}
			for field_name in field_names:
				outcome_batched_kwargs[field_name] = [
					input_dict[field_name] for input_dict in outcome_batch_inputs
				]

			# Run ORM evaluator with appropriate demos
			# Use demos_orm if provided, otherwise fall back to demos
			outcome_demos = demos_orm if demos_orm is not None else demos
			if outcome_demos is not None:
				logger.info(f"Using {len(outcome_demos)} user-provided demos for ORM evaluation")
			else:
				logger.info("Using zero-shot evaluation for ORM (no demos provided)")
			outcome_evaluator = self._get_evaluator_for_type("outcome")
			outcome_predictions = outcome_evaluator(
				config=config,
				demos=outcome_demos,
				**outcome_batched_kwargs,
			)
			outcome_output_dicts = self._consolidate_scores(
				outcome_predictions,
				self.orm_rubric,
				n_samples_judge=n_samples_evaluator,
			)

			# Group and assign results
			evaluations_per_state = (
				len(outcome_output_dicts) // len(outcome_states)
				if outcome_states
				else 0
			)

			for i, (original_idx, _) in enumerate(outcome_states_info):
				start_idx = i * evaluations_per_state
				end_idx = start_idx + evaluations_per_state
				state_evaluations = outcome_output_dicts[start_idx:end_idx]
				final_results[original_idx] = state_evaluations

		return final_results # pyright: ignore[reportReturnType]
