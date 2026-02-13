"""
Tests for the TreeOfThoughtEvaluator module.

Expected usage:
```bash
pytest predict/test_evaluator.py -vv
```
"""

# Standard library imports
import logging
import os
from typing import Annotated, Any, Literal
from unittest.mock import Mock

# Third-party imports
import annotated_types
import dspy
import pytest
import torch
from dspy.adapters.utils import get_annotation_name

# Local imports
from constants import OpenSourceModel
from lm.generative_local_lm import GenerativeLocalVLLM
from misc_utils import ExecutionError
from predict.evaluator.evaluator import TreeOfThoughtEvaluator
from predict.evaluator.evaluator_demos import ORM_DEMOS, PRM_DEMOS
from predict.local_predict import LocalPredict
from signatures import (
	ArgumentEvaluatorMultiDimensional,
	GenerateArgumentWithReasoning,
	InputField,
	OutputField,
	ReasoningField,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from tree import EvaluationResult, JudgeEvaluation, State
from utilities_for_tests import MockPredict

logger = logging.getLogger(__name__)


# =========================== #
# Test Signatures and Helpers #
# =========================== #


class SimpleReasoningSignatureForTests(ReasoningSignature):
	"""Simple reasoning signature used across evaluator unit tests."""
	question: str = InputField(desc="The question to answer")
	reasoning_step: str = ReasoningField(desc="A reasoning step toward the answer")
	answer: str = OutputField(desc="The final answer")


class PRMEvaluatorSignatureForTests(dspy.Signature):
	"""Minimal PRM evaluator signature for mocked LocalPredict calls."""
	question: str = InputField(desc="Question")
	reasoning_steps: list[str] = InputField(desc="Reasoning steps to evaluate")
	soundness: Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)] = OutputField(
		desc="Soundness score on a 0-10 scale"
	)
	promise: Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)] = OutputField(
		desc="Promise score on a 0-10 scale"
	)


class ORMEvaluatorSignatureForTests(dspy.Signature):
	"""Minimal ORM evaluator signature for mocked LocalPredict calls."""
	question: str = InputField(desc="Question")
	answer: str = InputField(desc="Proposed answer to evaluate")
	quality: Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)] = OutputField(
		desc="Quality score on a 0-10 scale"
	)


class PRMRubricSingleDimensionForTests(dspy.Signature):
	"""Single-dimension PRM rubric used to validate signature building behavior."""
	quality: Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(10.0)] = OutputField(
		desc="Overall reasoning-step quality"
	)


def _field_summaries(fields: dict[str, Any]) -> dict[str, dict[str, str | None]]:
	"""Summarize DSPy fields for stable equality checks in unit tests."""
	summaries: dict[str, dict[str, str | None]] = {}
	for name, field in fields.items():
		json_schema_extra = getattr(field, "json_schema_extra", None)
		desc = json_schema_extra.get("desc") if isinstance(json_schema_extra, dict) else None
		summaries[name] = {
			"annotation": get_annotation_name(field.annotation),
			"desc": desc,
		}
	return summaries


def build_evaluator_with_mocked_predictors(
	generator_signature: type[ReasoningSignature],
	prm_responses: list[list[list[str]]] | None,
	orm_responses: list[list[list[str]]] | None,
	verbosity: Literal["debug", "info", "warning", "error"] = "info",
) -> TreeOfThoughtEvaluator:
	"""Create a TreeOfThoughtEvaluator with MockPredict PRM/ORM predictors."""
	evaluator = TreeOfThoughtEvaluator(generator_signature=generator_signature, verbosity=verbosity)
	# Only override evaluators we will actually use in a given test case.
	# This keeps tests explicit and avoids forcing "unused" responses.
	if prm_responses is not None:
		evaluator.process_evaluator = MockPredict(
			responses=prm_responses, signature=PRMEvaluatorSignatureForTests
		)
	if orm_responses is not None:
		evaluator.outcome_evaluator = MockPredict(
			responses=orm_responses, signature=ORMEvaluatorSignatureForTests
		)
	return evaluator


# =============================================================================
# GPU Skip Markers
# =============================================================================

# Check if one or more GPUs are available
if torch.cuda.is_available():
	_has_gpu = True
	# Use real torch - don't mock
else:
	_has_gpu = False

# Skip GPU tests if no GPU is available
pytestmark_gpu = pytest.mark.skipif(
	not _has_gpu,
	reason="GPU tests require GPU access",
)



@pytest.fixture
def simple_reasoning_signature():
	"""Create a simple reasoning signature for testing."""
	return SimpleReasoningSignatureForTests


@pytest.fixture
def sample_states():
	"""Create sample states for testing."""
	states = []

	# State with first reasoning step
	state1 = State(
		input={"question": "What is the capital of France?"},
		reasoning=[
			{"reasoning_step": "What is France, I don't remember."},
			{"reasoning_step": "Let me review my geography knowledge."},
		],
		output={},
	)
	states.append(state1)

	# State with multiple reasoning steps
	state2 = State(
		input={"question": "What is the capital of France?"},
		reasoning=[
			{"reasoning_step": "I need to recall my knowledge about European capitals."},
			{"reasoning_step": "France is a country in Western Europe."},
		],
		output={},
	)
	states.append(state2)

	# State with final output
	state3 = State(
		input={"question": "What is the capital of France?"},
		reasoning=[{"reasoning_step": "The capital of France is Paris."}],
		output={"answer": "Paris"},
	)
	states.append(state3)

	return states


class TestTreeOfThoughtEvaluator:
	"""Test cases for TreeOfThoughtEvaluator class."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"generator_signature",
			"expected_reasoning_field_name",
			"expected_process_type",
			"expected_outcome_type",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				SolveMathProblemWithReasoning,  # generator_signature
				"math_operation", 				# expected_reasoning_field_name
				LocalPredict,  					# expected_process_type
				LocalPredict,  					# expected_outcome_type
				None,  							# expected_exception
				id="default_init_math",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  # generator_signature
				"claim",  						# expected_reasoning_field_name
				LocalPredict,  					# expected_process_type
				LocalPredict,  					# expected_outcome_type
				None,  							# expected_exception
				id="default_init_argument",
			),
		],
	)
	def test_initialization(
		self,
		generator_signature: type[ReasoningSignature],
		expected_reasoning_field_name: str,
		expected_process_type: type[LocalPredict],
		expected_outcome_type: type[LocalPredict],
		expected_exception: type[BaseException] | None,
	):
		"""Test that evaluator initializes correctly with required parameters."""
		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(generator_signature=generator_signature)
			return

		evaluator = TreeOfThoughtEvaluator(generator_signature=generator_signature)

		assert evaluator.reasoning_field_name == expected_reasoning_field_name
		assert isinstance(evaluator.process_evaluator, expected_process_type)
		assert isinstance(evaluator.outcome_evaluator, expected_outcome_type)

	@pytest.mark.parametrize(
		# Parameter names
		[
			"generator_signature",
			"evaluator_signature_prm",
			"expected_prm_instructions",
			"expected_prm_input_fields",
			"expected_prm_output_fields",
			"expected_orm_instructions",
			"expected_orm_input_fields",
			"expected_orm_output_fields",
		],
		# Parameter values
		[
			pytest.param(
				SimpleReasoningSignatureForTests,  			# generator_signature
				None,  										# evaluator_signature_prm
				(											# expected_prm_instructions
					"""
Judge the quality of reasoning steps for a problem-solving process.
The task requires producing `answer` given `question`.
Reasoning steps towards producing `answer` are provided in `reasoning_steps`.

Since you are evaluating intermediate reasoning, don't score based on completeness, and instead score based on the rubric items below:
- soundness: Logical validity, factual accuracy, and coherence with prior steps (a float between 0.0 and 10.0).
- promise: Likelihood of the reasoning to lead to a strong final answer (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_prm_input_fields
					"question": {
						"annotation": "str",
						"desc": "The question to answer",
					},
					"reasoning_steps": {
						"annotation": "list[str]",
						"desc": "List of 'reasoning_step's to evaluate toward producing `answer`",
					},
				},
				{  											# expected_prm_output_fields
					"soundness": {
						"annotation": "float",
						"desc": "Logical validity, factual accuracy, and coherence with prior steps",
					},
					"promise": {
						"annotation": "float",
						"desc": "Likelihood of the reasoning to lead to a strong final answer",
					},
				},
				(											# expected_orm_instructions
					"""
Judge the quality of a response for the provided task.
The task requires producing `answer` given `question`.

Evaluate the response using the rubric items below and assign numeric scores to each:
- quality: The overall quality of the final answer. Judge the quality based on correctness (accurate, logically sound, and follows the user's instructions), completeness (all required elements present), and clarity (well-structured with appropriate tone and style). Higher scores indicate strong performance across these dimensions. Any flaws along these dimensions should be penalized (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_orm_input_fields
					"question": {
						"annotation": "str",
						"desc": "The question to answer",
					},
					"answer": {
						"annotation": "str",
						"desc": "The final answer",
					},
				},
				{  											# expected_orm_output_fields
					"quality": {
						"annotation": "float",
						"desc": (
							"The overall quality of the final answer. Judge the quality based on correctness "
							"(accurate, logically sound, and follows the user's instructions), completeness "
							"(all required elements present), and clarity (well-structured with appropriate tone "
							"and style). Higher scores indicate strong performance across these dimensions. "
							"Any flaws along these dimensions should be penalized."
						),
					},
				},
				id="default_rubrics_simple_signature",
			),
			pytest.param(
				SimpleReasoningSignatureForTests,  			# generator_signature
				PRMRubricSingleDimensionForTests,  			# evaluator_signature_prm
				(											# expected_prm_instructions
					"""
Judge the quality of reasoning steps for a problem-solving process.
The task requires producing `answer` given `question`.
Reasoning steps towards producing `answer` are provided in `reasoning_steps`.

Since you are evaluating intermediate reasoning, don't score based on completeness, and instead score based on the rubric items below:
- quality: Overall reasoning-step quality (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_prm_input_fields
					"question": {
						"annotation": "str",
						"desc": "The question to answer",
					},
					"reasoning_steps": {
						"annotation": "list[str]",
						"desc": "List of 'reasoning_step's to evaluate toward producing `answer`",
					},
				},
				{  											# expected_prm_output_fields
					"quality": {
						"annotation": "float",
						"desc": "Overall reasoning-step quality",
					},
				},
				(											# expected_orm_instructions
					"""
Judge the quality of a response for the provided task.
The task requires producing `answer` given `question`.

Evaluate the response using the rubric items below and assign numeric scores to each:
- quality: The overall quality of the final answer. Judge the quality based on correctness (accurate, logically sound, and follows the user's instructions), completeness (all required elements present), and clarity (well-structured with appropriate tone and style). Higher scores indicate strong performance across these dimensions. Any flaws along these dimensions should be penalized (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_orm_input_fields
					"question": {
						"annotation": "str",
						"desc": "The question to answer",
					},
					"answer": {
						"annotation": "str",
						"desc": "The final answer",
					},
				},
				{  											# expected_orm_output_fields
					"quality": {
						"annotation": "float",
						"desc": (
							"The overall quality of the final answer. Judge the quality based on correctness "
							"(accurate, logically sound, and follows the user's instructions), completeness "
							"(all required elements present), and clarity (well-structured with appropriate tone "
							"and style). Higher scores indicate strong performance across these dimensions. "
							"Any flaws along these dimensions should be penalized."
						),
					},
				},
				id="custom_prm_rubric_simple_signature",
			),
			pytest.param(
				SolveMathProblemWithReasoning,  			# generator_signature
				None,  										# evaluator_signature_prm
				(											# expected_prm_instructions
					"""
Judge the quality of reasoning steps for a problem-solving process.
The task requires producing `answer` given `math_problem`.
Reasoning steps towards producing `answer` are provided in `reasoning_steps`.

Since you are evaluating intermediate reasoning, don't score based on completeness, and instead score based on the rubric items below:
- soundness: Logical validity, factual accuracy, and coherence with prior steps (a float between 0.0 and 10.0).
- promise: Likelihood of the reasoning to lead to a strong final answer (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_prm_input_fields
					"math_problem": {
						"annotation": "str",
						"desc": "The math problem to solve",
					},
					"reasoning_steps": {
						"annotation": "list[str]",
						"desc": "List of 'math_operation's to evaluate toward producing `answer`",
					},
				},
				{  											# expected_prm_output_fields
					"soundness": {
						"annotation": "float",
						"desc": "Logical validity, factual accuracy, and coherence with prior steps",
					},
					"promise": {
						"annotation": "float",
						"desc": "Likelihood of the reasoning to lead to a strong final answer",
					},
				},
				(											# expected_orm_instructions
					"""
Judge the quality of a response for the provided task.
The task requires producing `answer` given `math_problem`.

Evaluate the response using the rubric items below and assign numeric scores to each:
- quality: The overall quality of the final answer. Judge the quality based on correctness (accurate, logically sound, and follows the user's instructions), completeness (all required elements present), and clarity (well-structured with appropriate tone and style). Higher scores indicate strong performance across these dimensions. Any flaws along these dimensions should be penalized (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_orm_input_fields
					"math_problem": {
						"annotation": "str",
						"desc": "The math problem to solve",
					},
					"answer": {
						"annotation": "str",
						"desc": "The answer to the math problem",
					},
				},
				{  											# expected_orm_output_fields
					"quality": {
						"annotation": "float",
						"desc": (
							"The overall quality of the final answer. Judge the quality based on correctness "
							"(accurate, logically sound, and follows the user's instructions), completeness "
							"(all required elements present), and clarity (well-structured with appropriate tone "
							"and style). Higher scores indicate strong performance across these dimensions. "
							"Any flaws along these dimensions should be penalized."
						),
					},
				},
				id="default_rubrics_math_signature",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# generator_signature
				None,  										# evaluator_signature_prm
				(											# expected_prm_instructions
					"""
Judge the quality of reasoning steps for a problem-solving process.
The task requires producing `argument` given `topic` and `stance`.
Reasoning steps towards producing `argument` are provided in `reasoning_steps`.

Since you are evaluating intermediate reasoning, don't score based on completeness, and instead score based on the rubric items below:
- soundness: Logical validity, factual accuracy, and coherence with prior steps (a float between 0.0 and 10.0).
- promise: Likelihood of the reasoning to lead to a strong final answer (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_prm_input_fields
					"topic": {
						"annotation": "str",
						"desc": "The topic to generate an argument about",
					},
					"stance": {
						"annotation": "Literal['PRO', 'ANTI']",
						"desc": "The stance to take on the topic",
					},
					"reasoning_steps": {
						"annotation": "list[str]",
						"desc": "List of 'claim's to evaluate toward producing `argument`",
					},
				},
				{  											# expected_prm_output_fields
					"soundness": {
						"annotation": "float",
						"desc": "Logical validity, factual accuracy, and coherence with prior steps",
					},
					"promise": {
						"annotation": "float",
						"desc": "Likelihood of the reasoning to lead to a strong final answer",
					},
				},
				(											# expected_orm_instructions
					"""
Judge the quality of a response for the provided task.
The task requires producing `argument` given `topic` and `stance`.

Evaluate the response using the rubric items below and assign numeric scores to each:
- quality: The overall quality of the final answer. Judge the quality based on correctness (accurate, logically sound, and follows the user's instructions), completeness (all required elements present), and clarity (well-structured with appropriate tone and style). Higher scores indicate strong performance across these dimensions. Any flaws along these dimensions should be penalized (a float between 0.0 and 10.0).
""".strip()
				),
				{  											# expected_orm_input_fields
					"topic": {
						"annotation": "str",
						"desc": "The topic to generate an argument about",
					},
					"stance": {
						"annotation": "Literal['PRO', 'ANTI']",
						"desc": "The stance to take on the topic",
					},
					"argument": {
						"annotation": "str",
						"desc": "The generated argument",
					},
				},
				{  											# expected_orm_output_fields
					"quality": {
						"annotation": "float",
						"desc": (
							"The overall quality of the final answer. Judge the quality based on correctness (accurate, "
							"logically sound, and follows the user's instructions), completeness (all required elements "
							"present), and clarity (well-structured with appropriate tone and style). Higher scores indicate "
							"strong performance across these dimensions. Any flaws along these dimensions should be penalized."
						),
					},
				},
				id="default_rubrics_argument_signature",
			),
		],
	)
	def test_create_evaluator_signatures_instructions_and_fields(
		self,
		generator_signature: type[ReasoningSignature],
		evaluator_signature_prm: type[dspy.Signature] | None,
		expected_prm_instructions: str,
		expected_prm_input_fields: dict[str, dict[str, str | None]],
		expected_prm_output_fields: dict[str, dict[str, str | None]],
		expected_orm_instructions: str,
		expected_orm_input_fields: dict[str, dict[str, str | None]],
		expected_orm_output_fields: dict[str, dict[str, str | None]],
	) -> None:
		"""Test that both PRM and ORM evaluator signatures are created with correct instructions and fields."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=generator_signature,
			evaluator_signature_prm=evaluator_signature_prm,
		)

		# Test PRM signature
		prm_sig = evaluator._create_process_evaluator_signature()
		assert prm_sig.instructions == expected_prm_instructions
		assert _field_summaries(prm_sig.input_fields) == expected_prm_input_fields
		assert _field_summaries(prm_sig.output_fields) == expected_prm_output_fields

		# Test ORM signature
		orm_sig = evaluator._create_outcome_evaluator_signature()
		assert orm_sig.instructions == expected_orm_instructions
		assert _field_summaries(orm_sig.input_fields) == expected_orm_input_fields
		assert _field_summaries(orm_sig.output_fields) == expected_orm_output_fields

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"n_samples_evaluator",
			"evaluator_temperature",
			"prm_responses",
			"orm_responses",
			"expected_results",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				[								# states
					State(
						# PRM Only: no output => evaluated by process_evaluator
						input={"question": "Test question"},
						reasoning=[{"reasoning_step": "First reasoning step"}],
						output={},
					),
				],
				3,  							# n_samples_evaluator
				0.7,  							# evaluator_temperature
				[								# prm_responses (1 thread, 3 samples)
					[
						[
							"## soundness\n0\n\n## promise\n5",
							"## soundness\n5\n\n## promise\n5",
							"## soundness\n5\n\n## promise\n10",
						]
					]
				],
				None,  							# orm_responses (unused)
				[								# expected_results
					[
						# score = mean([0.25, 0.50, 0.75]) where judge=(soundness_norm+promise_norm)/2
						EvaluationResult(
							score=0.5,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 0.0, "promise": 5.0},
									normalized_scores={"soundness": 0.0, "promise": 0.5},
								),
								JudgeEvaluation(
									raw_scores={"soundness": 5.0, "promise": 5.0},
									normalized_scores={"soundness": 0.5, "promise": 0.5},
								),
								JudgeEvaluation(
									raw_scores={"soundness": 5.0, "promise": 10.0},
									normalized_scores={"soundness": 0.5, "promise": 1.0},
								),
							],
						)
					]
				],
				None,  							# expected_exception
				id="process_single_state",
			),
			pytest.param(
				[								# states
					State(
						# PRM: no output => evaluated by process_evaluator
						input={"question": "Q1"},
						reasoning=[{"reasoning_step": "Step 1"}, {"reasoning_step": "Step 2"}],
						output={},
					),
					State(
						# PRM: no output => evaluated by process_evaluator
						input={"question": "Q2"},
						reasoning=[
							{"reasoning_step": "Step A"},
							{"reasoning_step": "Step B"},
							{"reasoning_step": "Step C"},
						],
						output={},
					),
				],
				2,  							# n_samples_evaluator
				0.7,  							# evaluator_temperature
				[								# prm_responses (2 threads, 2 samples)
					[
						[
							"## soundness\n0\n\n## promise\n5",
							"## soundness\n5\n\n## promise\n5",
						],
						[
							"## soundness\n5\n\n## promise\n10",
							"## soundness\n10\n\n## promise\n10",
						],
					]
				],
				None,  							# orm_responses (unused)
				[								# expected_results (order must match `states`)
					[
						# score = mean([0.25, 0.50]) where judge=(soundness_norm+promise_norm)/2
						EvaluationResult(
							score=0.375,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 0.0, "promise": 5.0},
									normalized_scores={"soundness": 0.0, "promise": 0.5},
								),
								JudgeEvaluation(
									raw_scores={"soundness": 5.0, "promise": 5.0},
									normalized_scores={"soundness": 0.5, "promise": 0.5},
								),
							],
						)
					],
					[
						# score = mean([0.75, 1.00]) where judge=(soundness_norm+promise_norm)/2
						EvaluationResult(
							score=0.875,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 5.0, "promise": 10.0},
									normalized_scores={"soundness": 0.5, "promise": 1.0},
								),
								JudgeEvaluation(
									raw_scores={"soundness": 10.0, "promise": 10.0},
									normalized_scores={"soundness": 1.0, "promise": 1.0},
								),
							],
						)
					],
				],
				None,  							# expected_exception
				id="process_multiple_states",
			),
			pytest.param(
				[								# states
					State(
						# ORM: has output => evaluated by outcome_evaluator
						input={"question": "Final test"},
						reasoning=[
							{"reasoning_step": "Reasoning 1"},
							{"reasoning_step": "Reasoning 2"},
						],
						output={"answer": "Final answer"},
					),
				],
				3,  							# n_samples_evaluator
				0.7, 							# evaluator_temperature
				None,  							# prm_responses (unused)
				[								# orm_responses (1 thread, 3 samples)
					[
						[
							"## quality\n0",
							"## quality\n5",
							"## quality\n10",
						]
					]
				],
				[								# expected_results
					[
						# score = mean([0.00, 0.50, 1.00]) where judge=quality_norm
						EvaluationResult(
							score=0.5,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"quality": 0.0}, normalized_scores={"quality": 0.0}
								),
								JudgeEvaluation(
									raw_scores={"quality": 5.0}, normalized_scores={"quality": 0.5}
								),
								JudgeEvaluation(
									raw_scores={"quality": 10.0}, normalized_scores={"quality": 1.0}
								),
							],
						)
					]
				],
				None,  							# expected_exception
				id="outcome_with_outputs",
			),
			pytest.param(
				[								# states
					State(
						# PRM: no output => evaluated by process_evaluator
						input={"question": "Mixed Q1"},
						reasoning=[{"reasoning_step": "Step 1"}],
						output={},
					),
					State(
						# ORM: has output => evaluated by outcome_evaluator
						input={"question": "Mixed Q2"},
						reasoning=[{"reasoning_step": "Step 1"}],
						output={"answer": "A"},
					),
				],
				2,  							# n_samples_evaluator
				0.7,  							# evaluator_temperature
				[								# prm_responses (1 thread, 2 samples)
					[
						[
							"## soundness\n0\n\n## promise\n5",
							"## soundness\n5\n\n## promise\n10",
						]
					]
				],
				[								# orm_responses (1 thread, 2 samples)
					[
						[
							"## quality\n10",
							"## quality\n0",
						]
					]
				],
				[								# expected_results (order must match `states`)
					[
						# score = mean([0.25, 0.75]) where judge=(soundness_norm+promise_norm)/2
						EvaluationResult(
							score=0.5,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 0.0, "promise": 5.0},
									normalized_scores={"soundness": 0.0, "promise": 0.5},
								),
								JudgeEvaluation(
									raw_scores={"soundness": 5.0, "promise": 10.0},
									normalized_scores={"soundness": 0.5, "promise": 1.0},
								),
							],
						)
					],
					[
						# score = mean([1.00, 0.00]) where judge=quality_norm
						EvaluationResult(
							score=0.5,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"quality": 10.0},
									normalized_scores={"quality": 1.0},
								),
								JudgeEvaluation(
									raw_scores={"quality": 0.0},
									normalized_scores={"quality": 0.0},
								),
							],
						)
					],
				],
				None,  							# expected_exception
				id="mixed_prm_and_orm_in_one_call",
			),
		],
	)
	def test_forward_evaluation_types(
		self,
		states: list[State],
		n_samples_evaluator: int,
		evaluator_temperature: float,
		prm_responses: list[list[list[str]]] | None,
		orm_responses: list[list[list[str]]] | None,
		expected_results: list[list[EvaluationResult]],
		expected_exception: type[BaseException] | None,
	):
		"""Test forward method for different evaluation types and state configurations.

		Args:
			states: A list of State objects to evaluate.
			n_samples_evaluator: The number of unique evaluations for each score of each state.
			evaluator_temperature: The temperature to use for each call to the language model
				to produce a score for a given state.
			prm_responses: A list of lists of lists of strings representing the PRM responses.
				None means no PRM responses were provided.
			orm_responses: A list of lists of lists of strings representing the ORM responses.
				None means no ORM responses were provided.
			expected_results: A list of lists of EvaluationResult objects representing the expected
				results. None means no expected results were provided (i.e., the test is expected
				to fail with an exception).
			expected_exception: The exception to expect if the evaluation fails. None means no
				exception is expected (i.e., the test is expected to pass).
		"""
		evaluator = build_evaluator_with_mocked_predictors(
			generator_signature=SimpleReasoningSignatureForTests,
			prm_responses=prm_responses,
			orm_responses=orm_responses,
		)

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				evaluator(
					states=states,
					n_samples_evaluator=n_samples_evaluator,
					evaluator_temperature=evaluator_temperature,
				)
			return

		evaluation_results = evaluator(
			states=states,
			n_samples_evaluator=n_samples_evaluator,
			evaluator_temperature=evaluator_temperature,
		)
		assert evaluation_results == expected_results

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state_index",
			"evaluation_type",
			"expected_question_key",
			"expected_reasoning_key",
			"expected_reasoning_steps",
			"expected_answer_value",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				0,  # state_index
				"process",  						# evaluation_type
				"question",  						# expected_question_key
				"reasoning_steps",  				# expected_reasoning_key
				[									# expected_reasoning_steps
					"What is France, I don't remember.",
					"Let me review my geography knowledge.",
				],  								# expected_reasoning_steps
				None,  								# expected_answer_value
				None,  								# expected_exception
				id="process_single_state",
			),
			pytest.param(
				1,  								# state_index
				"process",  						# evaluation_type
				"question",  						# expected_question_key
				"reasoning_steps",  				# expected_reasoning_key
				[									# expected_reasoning_steps
					"I need to recall my knowledge about European capitals.",
					"France is a country in Western Europe.",
				],  								# expected_reasoning_steps
				None,  								# expected_answer_value
				None,  								# expected_exception
				id="process_multi_step",
			),
			pytest.param(
				2, 									# state_index
				"outcome",  						# evaluation_type
				"question",  						# expected_question_key
				"reasoning_steps",  				# expected_reasoning_key
				[									# expected_reasoning_steps
					"The capital of France is Paris."
				],
				"Paris", 		 					# expected_answer_value
				None,  								# expected_exception
				id="outcome_with_answer",
			),
		],
	)
	def test_state_to_evaluator_input_conversion(
		self,
		simple_reasoning_signature: type[ReasoningSignature],
		sample_states: list[State],
		state_index: int,
		evaluation_type: Literal["process", "outcome"],
		expected_question_key: str,
		expected_reasoning_key: str,
		expected_reasoning_steps: list[str],
		expected_answer_value: str | None,
		expected_exception: type[BaseException] | None,
	):
		"""Test state to evaluator input conversion for different evaluation types."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			verbosity="info",
		)

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				evaluator._state_to_evaluator_input(
					sample_states[state_index], evaluation_type
				)
			return

		evaluator_input = evaluator._state_to_evaluator_input(
			sample_states[state_index], evaluation_type
		)
		assert expected_question_key in evaluator_input
		if evaluation_type == "outcome":
			assert "answer" in evaluator_input
			assert evaluator_input["answer"] == expected_answer_value
			# Reasoning steps are only included for outcome eval when configured.
			if evaluator.consider_reasoning_in_final_eval:
				assert expected_reasoning_key in evaluator_input
				assert evaluator_input[expected_reasoning_key] == expected_reasoning_steps
			else:
				assert expected_reasoning_key not in evaluator_input
		else:
			assert expected_reasoning_key in evaluator_input
			assert evaluator_input[expected_reasoning_key] == expected_reasoning_steps

	@pytest.mark.parametrize(
		# Parameter names
		[
			"evaluator_signature",
			"completions_data",
			"n_samples_judge",
			"expected_results",
			"expected_exception",
		],
		# Parameter values
		[
			# Multi-dimension success cases with different bounds
			pytest.param(
				dspy.Signature(  						# evaluator_signature
					{
						"quality": (
							Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
							dspy.OutputField(desc="Quality 1-7"),
						),
						"clarity": (
							Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(7.0)],
							dspy.OutputField(desc="Clarity 0-7"),
						),
					},
					"Multi-dimension evaluation",
				),
				{										# completions_data
					"quality": [3.0, 5.0], "clarity": [4.0, 6.0]
				},
				2,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						score=17/28,  # mean(19/42, 32/42) = 51/84 = 17/28
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"quality": 3.0, "clarity": 4.0},
								normalized_scores={"quality": 1/3, "clarity": 4/7},
							),
							JudgeEvaluation(
								raw_scores={"quality": 5.0, "clarity": 6.0},
								normalized_scores={"quality": 2/3, "clarity": 6/7},
							),
						],
					)
				],
				None,  									# expected_exception
				id="multi_dim_quality_clarity_two_judges",
			),
			pytest.param(
				dspy.Signature(  						# evaluator_signature
					{
						"correctness": (
							Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
							dspy.OutputField(desc="Correctness 1-7"),
						),
						"clarity": (
							Annotated[float, annotated_types.Ge(1.0), annotated_types.Le(7.0)],
							dspy.OutputField(desc="Clarity 1-7"),
						),
					},
					"Multi-dimension evaluation",
				),
				{										# completions_data
					"correctness": [2.0, 2.0, 3.0],
					"clarity": [3.0, 6.0, 1.0],
				},
				3,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						# Score calculation (equal weights 0.5, 0.5 since no rubric_weight):
						# J1: 0.5*((2-1)/6) + 0.5*((3-1)/6) = 0.5*(1/6) + 0.5*(2/6) = 0.25
						# J2: 0.5*((2-1)/6) + 0.5*((6-1)/6) = 0.5*(1/6) + 0.5*(5/6) = 0.5
						# J3: 0.5*((3-1)/6) + 0.5*((1-1)/6) = 0.5*(2/6) + 0.5*(0/6) = 1/6
						# mean(0.25, 0.5, 1/6) = (0.25 + 0.5 + 1/6) / 3
						score=(0.25 + 0.5 + 1/6) / 3,
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"correctness": 2.0, "clarity": 3.0},
								normalized_scores={"correctness": 1/6, "clarity": 1/3},
							),
							JudgeEvaluation(
								raw_scores={"correctness": 2.0, "clarity": 6.0},
								normalized_scores={"correctness": 1/6, "clarity": 5/6},
							),
							JudgeEvaluation(
								raw_scores={"correctness": 3.0, "clarity": 1.0},
								normalized_scores={"correctness": 1/3, "clarity": 0.0},
							),
						],
					)
				],
				None,  									# expected_exception
				id="multi_dim_three_judges",
			),
			# Multi-judge error handling with default PRM rubric
			pytest.param(
				None,  									# evaluator_signature (use default)
				{										# completions_data
					"soundness": [9.0, "<invalid>", 7.0],
					"promise": [8.0, 6.0, 7.0],
				},
				3,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						score=0.775,  # mean(0.85, 0.7) - judge 1 invalid
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"soundness": 9.0, "promise": 8.0},
								normalized_scores={"soundness": 0.9, "promise": 0.8},
							),
							JudgeEvaluation(
								raw_scores={"soundness": 7.0, "promise": 7.0},
								normalized_scores={"soundness": 0.7, "promise": 0.7},
							),
						],
					)
				],
				None,  									# expected_exception
				id="drops_invalid_judge_value",
			),
			pytest.param(
				None,  									# evaluator_signature (use default)
				{										# completions_data
					"soundness": [9.0, 8.0],
					"promise": [7.0, 6.0],
				},
				2,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						score=0.75,  # mean(0.8, 0.7)
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"soundness": 9.0, "promise": 7.0},
								normalized_scores={"soundness": 0.9, "promise": 0.7},
							),
							JudgeEvaluation(
								raw_scores={"soundness": 8.0, "promise": 6.0},
								normalized_scores={"soundness": 0.8, "promise": 0.6},
							),
						],
					)
				],
				None,  									# expected_exception
				id="dict_like_two_judges",
			),
			pytest.param(
				None,  									# evaluator_signature (use default)
				{										# completions_data
					"soundness": [9.0, 8.0, 7.0],
					"promise": [7.0, 6.0],
				},
				3,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						score=0.75,  # mean(0.8, 0.7) - judge 2 excluded (missing promise)
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"soundness": 9.0, "promise": 7.0},
								normalized_scores={"soundness": 0.9, "promise": 0.7},
							),
							JudgeEvaluation(
								raw_scores={"soundness": 8.0, "promise": 6.0},
								normalized_scores={"soundness": 0.8, "promise": 0.6},
							),
						],
					)
				],
				None,  									# expected_exception
				id="mismatched_lengths_truncates",
			),
			# Weighted consolidation cases
			pytest.param(
				dspy.Signature(  						# evaluator_signature
					{
						"soundness": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
							OutputField(desc="Soundness 1-7", rubric_weight=0.7),
						),
						"promise": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
							OutputField(desc="Promise 1-7", rubric_weight=0.3),
						),
					},
					"Weighted PRM",
				),
				{"soundness": 6.0, "promise": 4.0},  	# completions_data
				1,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						# Score calculation (custom weights 0.7, 0.3 from rubric_weight):
						# J1: 0.7*((6-1)/6) + 0.3*((4-1)/6)
						score=0.7 * ((6.0 - 1.0) / 6.0) + 0.3 * ((4.0 - 1.0) / 6.0),
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"soundness": 6.0, "promise": 4.0},
								normalized_scores={"soundness": 5/6, "promise": 0.5},
							),
						],
					)
				],
				None,  									# expected_exception
				id="single_judge_weighted_0.7_0.3",
			),
			pytest.param(
				dspy.Signature(  						# evaluator_signature
					{
						"soundness": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
							OutputField(desc="Soundness 1-7", rubric_weight=0.6),
						),
						"promise": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
							OutputField(desc="Promise 1-7", rubric_weight=0.4),
						),
					},
					"Weighted PRM",
				),
				{"soundness": [5.0, 7.0], "promise": [3.0, 5.0]},  # completions_data
				2,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						score=(
							(0.6 * ((5.0 - 1.0) / 6.0) + 0.4 * ((3.0 - 1.0) / 6.0))
							+ (0.6 * ((7.0 - 1.0) / 6.0) + 0.4 * ((5.0 - 1.0) / 6.0))
						) / 2,
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"soundness": 5.0, "promise": 3.0},
								normalized_scores={"soundness": 2/3, "promise": 1/3},
							),
							JudgeEvaluation(
								raw_scores={"soundness": 7.0, "promise": 5.0},
								normalized_scores={"soundness": 1.0, "promise": 2/3},
							),
						],
					)
				],
				None,  									# expected_exception
				id="two_judges_weighted_0.6_0.4",
			),
			pytest.param(
				dspy.Signature(  						# evaluator_signature
					{
						"correctness": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(5)],
							OutputField(desc="Correctness 1-5", rubric_weight=0.5),
						),
						"clarity": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(5)],
							OutputField(desc="Clarity 1-5", rubric_weight=0.3),
						),
						"efficiency": (
							Annotated[int, annotated_types.Ge(1), annotated_types.Le(5)],
							OutputField(desc="Efficiency 1-5", rubric_weight=0.2),
						),
					},
					"Three-dimension weighted",
				),
				{"correctness": 4.0, "clarity": 3.0, "efficiency": 5.0},  # completions_data
				1,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						# Score calculation: 0.5*0.75 + 0.3*0.5 + 0.2*1.0 = 0.375 + 0.15 + 0.2 = 0.725
					score=0.725,
						judge_evaluations=[
							JudgeEvaluation(
								raw_scores={"correctness": 4.0, "clarity": 3.0, "efficiency": 5.0},
								normalized_scores={"correctness": 0.75, "clarity": 0.5, "efficiency": 1.0},
							),
						],
					)
				],
				None,  									# expected_exception
				id="three_dimensions_weighted_0.5_0.3_0.2",
			),
			# Error cases - single judge
			pytest.param(
				None,  									# evaluator_signature (use default)
				{"soundness": 11.0, "promise": 5.0},  	# completions_data
				1,  									# n_samples_judge
				[										# expected_results
					EvaluationResult(
						score=0.0,	# score=0.0 because all judges failed to produce valid outputs
						judge_evaluations=[],
					)
				],
				None,									# expected_exception
				id="single_judge_out_of_bounds",
			),
			pytest.param(
				None,  									# evaluator_signature (use default)
				{"promise": 5.0},  						# completions_data
				1,  									# n_samples_judge
				None,  									# expected_results
				KeyError,  								# expected_exception
				id="missing_required_dimension",
			),
		],
	)
	def test_consolidate_scores(
		self,
		simple_reasoning_signature: dspy.Signature,
		evaluator_signature: dspy.Signature | None,
		completions_data: dict[str, Any],
		n_samples_judge: int,
		expected_results: list[EvaluationResult] | None,
		expected_exception: type[BaseException] | None,
	):
		"""Comprehensive test for _consolidate_scores with multi-dimension, multi-judge, and error cases."""
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=evaluator_signature,
		)
		mock_pred = Mock()
		# Add ExecutionError objects to completions (successful completions have error_type=None)
		completions_with_errors = dict(completions_data)
		# TODO[P3]: Add test-cases where some judges fail to produce valid evaluator outputs (i.e.,
		# include an 'error' -- either generation or parsing -- in the completions_data).
		if "error" not in completions_with_errors:
			# Create error objects for each completion
			first_field_value = next(iter(completions_data.values()))
			if isinstance(first_field_value, list):
				n_completions = len(first_field_value)
				completions_with_errors["error"] = [ExecutionError() for _ in range(n_completions)]
			else:
				completions_with_errors["error"] = ExecutionError()
		mock_pred.completions = completions_with_errors

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				evaluator._consolidate_scores([mock_pred], evaluator.prm_rubric, n_samples_judge=n_samples_judge)
		else:
			results = evaluator._consolidate_scores([mock_pred], evaluator.prm_rubric, n_samples_judge=n_samples_judge)
			assert results == expected_results

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"n_samples_evaluator",
			"evaluator_temperature",
			"prm_responses",
			"orm_responses",
			"expected_results",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				[		# states
					State(
						input={"question": "Batch Q1"},
						reasoning=[{"reasoning_step": "Step 1"}],
						output={},
					),
				],
				1,  	# n_samples_evaluator
				0.7,  	# evaluator_temperature
				[		# prm_responses (1 thread, 1 sample)
					[["## soundness\n0\n\n## promise\n5"]]
				],
				None,  	# orm_responses (unused)
				[		# expected_results
					[
						EvaluationResult(
							score=0.25,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 0.0, "promise": 5.0},
									normalized_scores={"soundness": 0.0, "promise": 0.5},
								)
							],
						)
					]
				],
				None,  	# expected_exception
				id="single_state",
			),
			pytest.param(
				[		# states
					State(
						input={"question": "Batch Q1"},
						reasoning=[{"reasoning_step": "Step 1"}],
						output={},
					),
					State(
						input={"question": "Batch Q2"},
						reasoning=[{"reasoning_step": "Step 1"}],
						output={},
					),
				],
				1,  	# n_samples_evaluator
				0.7,  	# evaluator_temperature
				[		# prm_responses (2 threads, 1 sample)
					[
						["## soundness\n0\n\n## promise\n5"],
						["## soundness\n5\n\n## promise\n5"],
					]
				],
				None,  	# orm_responses (unused)
				[
					[
						# score = mean([0.25]) where judge=(soundness_norm+promise_norm)/2
						# score = mean([0.25]) where judge=(soundness_norm+promise_norm)/2
						EvaluationResult(
							score=0.25,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 0.0, "promise": 5.0},
									normalized_scores={"soundness": 0.0, "promise": 0.5},
								)
							],
						)
					],
					[
						# score = mean([0.50]) where judge=(soundness_norm+promise_norm)/2
						EvaluationResult(
							score=0.5,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"soundness": 5.0, "promise": 5.0},
									normalized_scores={"soundness": 0.5, "promise": 0.5},
								)
							],
						)
					],
				],
				None,  # expected_exception
				id="two_states",
			),
		],
	)
	def test_batch_processing_efficiency(
		self,
		simple_reasoning_signature: type[ReasoningSignature],
		states: list[State],
		n_samples_evaluator: int,
		evaluator_temperature: float,
		prm_responses: list[list[list[str]]] | None,
		orm_responses: list[list[list[str]]] | None,
		expected_results: list[list[EvaluationResult]],
		expected_exception: type[BaseException] | None,
	):
		"""Test that batch processing works correctly with multiple states."""
		evaluator = build_evaluator_with_mocked_predictors(
			generator_signature=simple_reasoning_signature,
			prm_responses=prm_responses,
			orm_responses=orm_responses,
		)

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				evaluator(
					states=states,
					n_samples_evaluator=n_samples_evaluator,
					evaluator_temperature=evaluator_temperature,
				)
			return

		# Process multiple states at once (evaluation type is auto-detected)
		evaluation_results = evaluator(
			states=states,
			n_samples_evaluator=n_samples_evaluator,
			evaluator_temperature=evaluator_temperature,
		)
		assert evaluation_results == expected_results

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"n_samples_evaluator",
			"evaluator_temperature",
			"prm_responses",
			"orm_responses",
			"expected_results",
			"expected_exception",
			"expected_message",
		],
		# Parameter values
		[
			pytest.param(
				State(
					input={"question": "What is the capital of France?"},
					reasoning=[],
					output={},
				),  			# state (PRM: empty reasoning, no output)
				1,  			# n_samples_evaluator
				0.7,  			# evaluator_temperature
				None,  			# prm_responses (unused)
				None,  			# orm_responses (unused)
				None,  			# expected_results (unused)
				ValueError,  	# expected_exception
				"PRM evaluation requires at least one reasoning step",  # expected_message
				id="prm_empty_reasoning_raises",
			),
			pytest.param(
				State(			# state (ORM: empty reasoning, has output)
					input={"question": "What is the capital of France?"},
					reasoning=[],
					output={"answer": "Paris"},
				),
				1,  			# n_samples_evaluator
				0.7,  			# evaluator_temperature
				None,  			# prm_responses (unused)
				[				# orm_responses (1 thread, 1 sample)
					[["## quality\n5"]]
				],
				[				# expected_results
					[
						# score = mean([0.50]) where judge=quality_norm
						EvaluationResult(
							score=0.5,
							judge_evaluations=[
								JudgeEvaluation(
									raw_scores={"quality": 5.0},
									normalized_scores={"quality": 0.5},
								)
							],
						)
					]
				],
				None,  			# expected_exception
				None,  			# expected_message
				id="orm_empty_reasoning_allows",
			),
		],
	)
	def test_empty_reasoning_behavior(
		self,
		simple_reasoning_signature: type[ReasoningSignature],
		state: State,
		n_samples_evaluator: int,
		evaluator_temperature: float,
		prm_responses: list[list[list[str]]] | None,
		orm_responses: list[list[list[str]]] | None,
		expected_results: list[list[EvaluationResult]] | None,
		expected_exception: type[BaseException] | None,
		expected_message: str | None,
	):
		"""Test PRM vs ORM behavior for empty reasoning steps.

		Args:
			simple_reasoning_signature: The signature of the reasoning model to use for the evaluator.
			state: The State object to evaluate.
			n_samples_evaluator: The number of unique evaluations for each score of the state.
			evaluator_temperature: The temperature to use for each call to the language model
				to produce a score for the state.
			prm_responses: A list of lists of lists of strings representing the PRM responses.
				None means no PRM responses were provided.
			orm_responses: A list of lists of lists of strings representing the ORM responses.
				None means no ORM responses were provided.
			expected_results: A list of lists of EvaluationResult objects representing the expected
				results. None means no expected results were provided (i.e., the test is expected
				to fail with an exception).
			expected_exception: The exception to expect if the evaluation fails. None means no exception
				is expected (i.e., the test is expected to pass).
			expected_message: The message to expect if the evaluation fails. None means no message
				is expected (i.e., the test is expected to pass).
			None means no message is expected (i.e., the test is expected to pass).
		"""
		evaluator = build_evaluator_with_mocked_predictors(
			generator_signature=simple_reasoning_signature,
			prm_responses=prm_responses,
			orm_responses=orm_responses,
		)

		if expected_exception is not None:
			with pytest.raises(expected_exception) as exc_info:
				evaluator(
					states=state,
					n_samples_evaluator=n_samples_evaluator,
					evaluator_temperature=evaluator_temperature,
				)
			assert expected_message in str(exc_info.value)
		else:
			evaluation_results = evaluator(
				states=state,
				n_samples_evaluator=n_samples_evaluator,
				evaluator_temperature=evaluator_temperature,
			)
			assert expected_results is not None
			assert evaluation_results == expected_results


class TestCustomEvaluatorSignatures:
	"""Test cases for custom evaluator signature functionality."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"lower_bound",
			"upper_bound",
			"expected_bounds",
			"expected_numeric_fields",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				0.0,  						# lower_bound
				10.0,  						# upper_bound
				{"quality": (0.0, 10.0)},  	# expected_bounds
				["quality"],  				# expected_numeric_fields
				None,  						# expected_exception
				id="single_dimension_default_bounds",
			),
			pytest.param(
				1.0,  						# lower_bound
				7.0,  						# upper_bound
				{"quality": (1.0, 7.0)},  	# expected_bounds
				["quality"],  				# expected_numeric_fields
				None,  						# expected_exception
				id="single_dimension_custom_bounds",
			),
		],
	)
	def test_custom_single_dimension_signature(
		self,
		simple_reasoning_signature: dspy.Signature,
		lower_bound: float,
		upper_bound: float,
		expected_bounds: dict[str, tuple[float, float]],
		expected_numeric_fields: list[str],
		expected_exception: type[BaseException] | None,
	):
		"""Test evaluator with custom single-dimension signature (numeric-only).

		Args:
			simple_reasoning_signature: The signature of the reasoning model to use for the evaluator.
			lower_bound: The lower bound of the quality score.
			upper_bound: The upper bound of the quality score.
			expected_bounds: A dictionary mapping the field name to a tuple of the lower and upper bounds.
			expected_numeric_fields: A list of the field names that are numeric.
			expected_exception: The exception to expect if the evaluation fails. None means no exception
				is expected (i.e., the test is expected to pass).
		"""
		# Create custom evaluator signature with different bounds
		signature_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(lower_bound), annotated_types.Le(upper_bound)],
				dspy.OutputField(desc="Quality score"),
			),
		}
		custom_sig = dspy.Signature(signature_fields, "Evaluate quality on 0-10 scale")

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(
					generator_signature=simple_reasoning_signature,
					evaluator_signature=custom_sig,
				)
			return

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify field extraction (since custom_sig is used for both PRM and ORM)
		assert evaluator.prm_numeric_score_fields == expected_numeric_fields
		assert evaluator.prm_dimension_bounds == expected_bounds
		# ORM uses same signature
		assert evaluator.orm_numeric_score_fields == expected_numeric_fields
		assert evaluator.orm_dimension_bounds == expected_bounds

	@pytest.mark.parametrize(
		# Parameter names
		[
			"dimension_defs",
			"expected_numeric_fields",
			"expected_bounds",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				[					# dimension_defs
					("correctness", 1.0, 7.0), ("clarity", 1.0, 5.0), ("completeness", 1.0, 3.0),
				],
				[					# expected_numeric_fields
					"correctness", "clarity", "completeness",
				],
				{					# expected_bounds
					"correctness": (1.0, 7.0), "clarity": (1.0, 5.0), "completeness": (1.0, 3.0),
				},
				None,  				# expected_exception
				id="multi_dimension_bounds",
			),
			pytest.param(
				[					# dimension_defs
					("correctness", 0.0, 10.0), ("clarity", 0.0, 10.0),
				],
				[					# expected_numeric_fields
					"correctness", "clarity",
				],
				{					# expected_bounds
					"correctness": (0.0, 10.0), "clarity": (0.0, 10.0),
				},
				None,  				# expected_exception
				id="multi_dimension_two_fields",
			),
		],
	)
	def test_custom_multi_dimension_signature(
		self,
		simple_reasoning_signature: dspy.Signature,
		dimension_defs: list[tuple[str, float, float]],
		expected_numeric_fields: list[str],
		expected_bounds: dict[str, tuple[float, float]],
		expected_exception: type[BaseException] | None,
	):
		"""Test evaluator with custom multi-dimension signature (numeric-only)."""
		# Create multi-dimension evaluator signature
		signature_fields = {}
		for field_name, lower_bound, upper_bound in dimension_defs:
			signature_fields[field_name] = (
				Annotated[float, annotated_types.Ge(lower_bound), annotated_types.Le(upper_bound)],
				dspy.OutputField(desc=f"{field_name} bounds"),
			)
		custom_sig = dspy.Signature(signature_fields, "Evaluate on multiple dimensions")

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(
					generator_signature=simple_reasoning_signature,
					evaluator_signature=custom_sig,
				)
			return

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify fields for both PRM and ORM (same custom signature used for both)
		assert evaluator.prm_numeric_score_fields == expected_numeric_fields
		assert evaluator.prm_dimension_bounds == expected_bounds
		assert evaluator.orm_numeric_score_fields == expected_numeric_fields
		assert evaluator.orm_dimension_bounds == expected_bounds

	@pytest.mark.parametrize(
		# Parameter names
		[
			"field_name",
			"expected_bounds",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				"score",  					# field_name
				{"score": (0.0, 10.0)},  	# expected_bounds
				None,  						# expected_exception
				id="default_bounds_added",
			),
			pytest.param(
				"quality",  				# field_name
				{"quality": (0.0, 10.0)},  	# expected_bounds
				None, 						 # expected_exception
				id="default_bounds_alt_field",
			),
		],
	)
	def test_custom_signature_without_bounds(
		self,
		simple_reasoning_signature: dspy.Signature,
		field_name: str,
		expected_bounds: dict[str, tuple[float, float]],
		expected_exception: type[BaseException] | None,
	):
		"""Test that default bounds are added when not specified."""
		# Create signature without bounds metadata
		signature_fields = {
			field_name: (float, dspy.OutputField(desc="Quality score")),
		}
		custom_sig = dspy.Signature(signature_fields, "Custom evaluation")

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(
					generator_signature=simple_reasoning_signature,
					evaluator_signature=custom_sig,
				)
			return

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature=custom_sig,
		)

		# Verify default bounds were added (0.0, 10.0) - default scale
		assert evaluator.prm_dimension_bounds == expected_bounds
		assert evaluator.orm_dimension_bounds == expected_bounds


	@pytest.mark.parametrize(
		# Parameter names
		[
			"expected_message_substr",
			"expected_missing_substr",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				"must be specified for all",  	# expected_message_substr
				"Missing rubric_weight for",  	# expected_missing_substr
				ValueError,  					# expected_exception
				id="partial_weights",
			),
			pytest.param(
				"must be specified for all",  	# expected_message_substr
				"Missing rubric_weight for",  	# expected_missing_substr
				ValueError,  					# expected_exception
				id="partial_weights_repeat",
			),
		],
	)
	def test_partial_weights_error(
		self,
		simple_reasoning_signature: dspy.Signature,
		expected_message_substr: str,
		expected_missing_substr: str,
		expected_exception: type[BaseException],
	):
		"""Test that specifying rubric_weight for only some fields raises an error."""
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=0.7),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7"),
			),
		}
		partial_weighted_sig = dspy.Signature(signature_fields, "Partial weights")

		with pytest.raises(expected_exception) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature_prm=partial_weighted_sig,
			)
		assert expected_message_substr in str(exc_info.value)
		assert expected_missing_substr in str(exc_info.value)

	@pytest.mark.parametrize(
		# Parameter names
		[
			"expected_message_substr",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				"must be positive",  	# expected_message_substr
				ValueError,  			# expected_exception
				id="negative_weights",
			),
			pytest.param(
				"must be positive",  	# expected_message_substr
				ValueError,  			# expected_exception
				id="negative_weights_repeat",
			),
		],
	)
	def test_negative_weights(
		self,
		simple_reasoning_signature: dspy.Signature,
		expected_message_substr: str,
		expected_exception: type[BaseException],
	):
		"""Test that negative rubric_weight values raise an error."""
		signature_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Soundness 1-7", rubric_weight=-0.5),
			),
			"promise": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(7)],
				OutputField(desc="Promise 1-7", rubric_weight=1.5),
			),
		}
		negative_weighted_sig = dspy.Signature(signature_fields, "Negative weights")

		with pytest.raises(expected_exception) as exc_info:
			TreeOfThoughtEvaluator(
				generator_signature=simple_reasoning_signature,
				evaluator_signature_prm=negative_weighted_sig,
			)
		assert expected_message_substr in str(exc_info.value)


class TestSymmetricAPI:
	"""Tests for the fully symmetric API (evaluator_signature_prm/orm with rubric_weight, demos_prm/orm)."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"expected_prm_fields",
			"expected_prm_bounds",
			"expected_orm_fields",
			"expected_orm_bounds",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				["soundness", "foresight"],  					# expected_prm_fields
				{"soundness": (1, 10), "foresight": (1, 10)},  	# expected_prm_bounds
				["quality"],  									# expected_orm_fields
				{"quality": (0.0, 1.0)},  						# expected_orm_bounds
				None,  											# expected_exception
				id="separate_prm_orm",
			),
		],
	)
	def test_separate_prm_orm_signatures(
		self,
		simple_reasoning_signature: type[ReasoningSignature],
		expected_prm_fields: list[str],
		expected_prm_bounds: dict[str, tuple[float, float]],
		expected_orm_fields: list[str],
		expected_orm_bounds: dict[str, tuple[float, float]],
		expected_exception: type[BaseException] | None,
	):
		"""Test that providing separate PRM and ORM signatures works correctly."""
		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(generator_signature=simple_reasoning_signature)
			return
		# Create different signatures for PRM and ORM
		prm_sig_fields = {
			"soundness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(10)],
				dspy.OutputField(desc="Soundness 1-10"),
			),
			"foresight": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(10)],
				dspy.OutputField(desc="Foresight 1-10"),
			),
		}
		prm_sig = dspy.Signature(prm_sig_fields, "Custom PRM rubric")

		orm_sig_fields = {
			"quality": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(1.0)],
				dspy.OutputField(desc="Quality 0-1"),
			),
		}
		orm_sig = dspy.Signature(orm_sig_fields, "Custom ORM rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=prm_sig,
			evaluator_signature_orm=orm_sig,
		)

		# Verify PRM uses prm_sig
		assert evaluator.prm_rubric == prm_sig
		assert evaluator.prm_numeric_score_fields == expected_prm_fields
		assert evaluator.prm_dimension_bounds == expected_prm_bounds

		# Verify ORM uses orm_sig
		assert evaluator.orm_rubric == orm_sig
		assert evaluator.orm_numeric_score_fields == expected_orm_fields
		assert evaluator.orm_dimension_bounds == expected_orm_bounds


	@pytest.mark.parametrize(
		# Parameter names
		[
			"use_base_demos",
			"use_prm_override",
			"use_orm_override",
			"expected_prm_source",
			"expected_orm_source",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				True,  		# use_base_demos
				False,  	# use_prm_override
				False,  	# use_orm_override
				"base",  	# expected_prm_source
				"base",  	# expected_orm_source
				None,  		# expected_exception
				id="base_demos_only",
			),
			pytest.param(
				True,  		# use_base_demos
				True,  		# use_prm_override
				False,  	# use_orm_override
				"prm",  	# expected_prm_source
				"base",  	# expected_orm_source
				None,  		# expected_exception
				id="prm_override",
			),
			pytest.param(
				True,  		# use_base_demos
				False,  	# use_prm_override
				True,  		# use_orm_override
				"base",  	# expected_prm_source
				"orm",  	# expected_orm_source
				None,  		# expected_exception
				id="orm_override",
			),
			pytest.param(
				True,  		# use_base_demos
				True,  		# use_prm_override
				True,  		# use_orm_override
				"prm",  	# expected_prm_source
				"orm",  	# expected_orm_source
				None,  		# expected_exception
				id="both_overrides",
			),
			pytest.param(
				False,  	# use_base_demos
				False,  	# use_prm_override
				False,  	# use_orm_override
				"none",  	# expected_prm_source
				"none",  	# expected_orm_source
				None,  		# expected_exception
				id="zero_shot",
			),
		],
	)
	def test_demos_parameter_precedence(
		self,
		simple_reasoning_signature: type[ReasoningSignature],
		use_base_demos: bool,
		use_prm_override: bool,
		use_orm_override: bool,
		expected_prm_source: str,
		expected_orm_source: str,
		expected_exception: type[BaseException] | None,
	):
		"""Test that demos_prm and demos_orm take precedence over base demos parameter.

		Args:
			simple_reasoning_signature: The signature to use for the evaluator.
			use_base_demos: Whether to use the base demos.
			use_prm_override: Whether to use the PRM override demos.
			use_orm_override: Whether to use the ORM override demos.
			expected_prm_source: The expected PRM source.
			expected_orm_source: The expected ORM source.
			expected_exception: The expected exception.
		"""
		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(generator_signature=simple_reasoning_signature)
			return
		# Create evaluator
		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
		)

		# Create test demos
		base_demos = [{"input": {"question": "Base"}, "output": {"soundness": 5, "promise": 5}}]
		prm_demos = PRM_DEMOS
		orm_demos = ORM_DEMOS

		# Create test states
		prm_state = State(
			input={"question": "Test PRM"},
			reasoning=[{"reasoning_step": "Step 1"}],
			output={},
		)
		orm_state = State(
			input={"question": "Test ORM"},
			reasoning=[{"reasoning_step": "Step 1"}],
			output={"answer": "Test answer"},
		)

		# Track which demos were actually used by capturing them in closures
		captured_prm_demos = []
		captured_orm_demos = []

		def mock_prm_call(config=None, demos=None, **kwargs):
			captured_prm_demos.append(demos)
			predictions = []
			batch_size = len(list(kwargs.values())[0]) if kwargs else 1
			for i in range(batch_size):
				mock_prediction = Mock()
				mock_prediction.completions = {
					"soundness": 3.0 + (i * 0.1),
					"promise": 2.0 + (i * 0.1),
					"error": ExecutionError(),  # Successful completion
				}
				predictions.append(mock_prediction)
			return predictions

		def mock_orm_call(config=None, demos=None, **kwargs):
			captured_orm_demos.append(demos)
			predictions = []
			batch_size = len(list(kwargs.values())[0]) if kwargs else 1
			for i in range(batch_size):
				mock_prediction = Mock()
				mock_prediction.completions = {
					"quality": 4.0 + (i * 0.1),
					"error": ExecutionError(),  # Successful completion
				}
				predictions.append(mock_prediction)
			return predictions

		evaluator.process_evaluator = Mock(side_effect=mock_prm_call)
		evaluator.outcome_evaluator = Mock(side_effect=mock_orm_call)

		captured_prm_demos.clear()
		captured_orm_demos.clear()

		base_arg = base_demos if use_base_demos else None
		prm_arg = prm_demos if use_prm_override else None
		orm_arg = orm_demos if use_orm_override else None

		call_kwargs = {}
		if use_base_demos:
			call_kwargs["demos"] = base_arg
		if use_prm_override:
			call_kwargs["demos_prm"] = prm_arg
		if use_orm_override:
			call_kwargs["demos_orm"] = orm_arg

		evaluator(states=[prm_state, orm_state], **call_kwargs)

		expected_prm = {
			"base": base_demos,
			"prm": prm_demos,
			"none": None,
		}[expected_prm_source]
		expected_orm = {
			"base": base_demos,
			"orm": orm_demos,
			"none": None,
		}[expected_orm_source]

		assert captured_prm_demos == [expected_prm]
		assert captured_orm_demos == [expected_orm]

	@pytest.mark.parametrize(
		# Parameter names
		[
			"expected_prm_weights",
			"expected_orm_weights",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				{"correctness": 1.0},  	# expected_prm_weights
				{"accuracy": 1.0}, 		# expected_orm_weights
				None,  					# expected_exception
				id="full_overrides",
			),
		],
	)
	def test_full_symmetric_overrides(
		self,
		simple_reasoning_signature: type[ReasoningSignature],
		expected_prm_weights: dict[str, float],
		expected_orm_weights: dict[str, float],
		expected_exception: type[BaseException] | None,
	):
		"""Test using all symmetric override parameters together with field-level weights."""
		if expected_exception is not None:
			with pytest.raises(expected_exception):
				TreeOfThoughtEvaluator(generator_signature=simple_reasoning_signature)
			return
		# Create custom signatures with rubric_weight in fields
		prm_sig_fields = {
			"correctness": (
				Annotated[int, annotated_types.Ge(1), annotated_types.Le(5)],
				OutputField(desc="Correctness 1-5", rubric_weight=1.0),
			),
		}
		prm_sig = dspy.Signature(prm_sig_fields, "PRM rubric")

		orm_sig_fields = {
			"accuracy": (
				Annotated[float, annotated_types.Ge(0.0), annotated_types.Le(100.0)],
				OutputField(desc="Accuracy 0-100", rubric_weight=1.0),
			),
		}
		orm_sig = dspy.Signature(orm_sig_fields, "ORM rubric")

		evaluator = TreeOfThoughtEvaluator(
			generator_signature=simple_reasoning_signature,
			evaluator_signature_prm=prm_sig,
			evaluator_signature_orm=orm_sig,
		)

		# Verify all overrides are applied correctly
		assert evaluator.prm_rubric == prm_sig
		assert evaluator.orm_rubric == orm_sig
		assert evaluator.prm_dimension_weights == expected_prm_weights
		assert evaluator.orm_dimension_weights == expected_orm_weights
		assert evaluator.is_custom_prm_signature is True
		assert evaluator.is_custom_orm_signature is True



# =============================================================================
# Shared GPU Model Fixture
# =============================================================================

@pytest.fixture(scope="module")
def shared_gpu_model():
	"""Shared GenerativeLocalVLLM fixture for all GPU integration tests.

	This fixture loads a model once and shares it across all GPU test classes
	to avoid loading multiple models and running out of GPU memory.
	"""
	if not torch.cuda.is_available():
		pytest.skip("GPU not available")

	base_path = "/projects/BSTEWART/model_storage"
	model_name = OpenSourceModel.QWEN_3_30B_A3B_INSTRUCT_2507.value
	model_path = os.path.join(base_path, model_name)
	lm = None
	try:
		logger.info(f"Initializing shared GPU model from: {model_path}")
		lm = GenerativeLocalVLLM(
			model=model_path,
			tensor_parallel_size=1,
			dtype="auto",
			gpu_memory_utilization=0.9,
			max_model_len=16_384,
			enforce_eager=True,
			verbosity="info",
		)
		logger.info("Shared GPU model initialized successfully")
		dspy.settings.configure(lm=lm)
		yield lm
	except Exception as e:
		logger.error(f"Failed to load GPU model {model_path}: {e}")
		# Re-raise the exception so tests fail with clear error messages
		# rather than being skipped silently
		raise
	finally:
		# Cleanup after all GPU tests complete
		if lm is not None:
			logger.info("Cleaning up shared GPU model...")
			lm.kill()


@pytestmark_gpu
class TestEvaluatorIntegration:
	"""Integration tests for the evaluator using real models (requires GPU)."""

	@pytest.fixture
	def local_lm(self, shared_gpu_model):
		"""Use the shared GPU model fixture."""
		return shared_gpu_model

	@pytest.fixture
	def evaluator(self, local_lm):
		"""Create an evaluator instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtEvaluator(
			generator_signature=SolveMathProblemWithReasoning,
			consider_reasoning_in_final_eval=True,
			verbosity="info",
		)

	@pytest.mark.parametrize(
		# Parameter names
		[
			"better_state",
			"worse_state",
			"_test_type",
			"_description",
			"reasoning_analysis",
			"expected_is_better",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				State(							# better_state (complete algebraic reasoning)
					input={"math_problem": "Solve 5x - 12 = 18"},
					reasoning=[
						{"math_operation": "I need to isolate x by moving constants to one side."},
						{"math_operation": "Adding 12 to both sides: 5x - 12 + 12 = 18 + 12"},
						{"math_operation": "This gives me: 5x = 30"},
						{"math_operation": "Dividing both sides by 5: x = 6"},
						{"math_operation": "Check: 5*6 - 12 = 30 - 12 = 18, so the solution is consistent."},
					],
					output={},
				),
				State(							# worse_state (incorrect algebra and self-contradiction)
					input={"math_problem": "Solve 5x - 12 = 18"},
					reasoning=[
						{"math_operation": "I need to solve for x."},
						{"math_operation": "Move 12 to the right: 5x = 18 - 12 = 6"},
						{"math_operation": "Divide by 5: x = 1.2"},
						{"math_operation": "Check: 5*1.2 - 12 = -6, so that seems off but I'll keep it."},
					],
					output={},
				),
				"PRM",  						# _test_type
				(								# _description
					"Complete vs incomplete algebraic steps",
				),
				(								# reasoning_analysis
					"Better shows correct steps and verifies the solution, worse applies a wrong sign and admits an inconsistent check",
				),
				True,  							# expected_is_better
				None,  # expected_exception
				id="prm_algebraic_steps",
			),
			pytest.param(
				State(							# better_state (correct formula)
					input={"math_problem": "Find the area of a triangle with base 8 and height 6"},
					reasoning=[
						{"math_operation": "The formula for triangle area is A = (1/2) * base * height"},
						{"math_operation": "Substituting values: A = (1/2) * 8 * 6"},
						{"math_operation": "A = (1/2) * 48 = 24 square units"},
					],
					output={},
				),
				State(							# worse_state (missing 1/2 factor)
					input={"math_problem": "Find the area of a triangle with base 8 and height 6"},
					reasoning=[
						{"math_operation": "Triangle area is base times height"},
						{"math_operation": "So area = 8 * 6 = 48"},
					],
					output={},
				),
				"PRM",  						# _test_type
				(								# _description
					"Correct vs incorrect formula application",
				),
				(								# reasoning_analysis
					"Better uses correct triangle formula, worse omits the 1/2 factor",
				),
				True,  							# expected_is_better
				None,  							# expected_exception
				id="prm_triangle_area",
			),
			pytest.param(
				State(							# better_state (systematic algebra)
					input={"math_problem": "A number increased by 15% becomes 92. Find the original number."},
					reasoning=[
						{"math_operation": "Let the original number be x"},
						{"math_operation": "After 15% increase: x + 0.15x = 92"},
						{"math_operation": "Combining like terms: 1.15x = 92"},
					],
					output={},
				),
				State(							# worse_state (flawed reverse calculation)
					input={"math_problem": "A number increased by 15% becomes 92. Find the original number."},
					reasoning=[
						{"math_operation": "92 minus 15% should give the original number"},
						{"math_operation": "15% of 92 is about 14, so 92 - 14 = 78"},
					],
					output={},
				),
				"PRM",  						# _test_type
				(								# _description
					"Systematic vs unsystematic percentage problem approach",
				),
				(								# reasoning_analysis
					"Better uses proper algebraic setup, worse uses flawed reverse calculation",
				),
				True,  							# expected_is_better
				None,  							# expected_exception
				id="prm_percentage_problem",
			),
			pytest.param(
				State(							# better_state (detailed calculus reasoning)
					input={"math_problem": "Find the derivative of f(x) = x³ - 4x^2 + 7"},
					reasoning=[
						{"math_operation": "I'll apply the power rule to each term: d/dx(x^n) = nx^(n-1)"},
						{"math_operation": "For x³: derivative is 3x^2"},
						{"math_operation": "For -4x^2: derivative is -8x"},
						{"math_operation": "For constant 7: derivative is 0"},
						{"math_operation": "Therefore: f'(x) = 3x^2 - 8x"},
						{"math_operation": "Quick check: derivative has degree 2, which matches expectations for a cubic."},
					],
					output={},
				),
				State(							# worse_state (incorrect derivative and rule misuse)
					input={"math_problem": "Find the derivative of f(x) = x^3 - 4x^2 + 7"},
					reasoning=[
						{"math_operation": "Using power rule"},
						{"math_operation": "f'(x) = 3x^3 - 8x + 7"},
						{"math_operation": "Constants stay the same, so +7 remains in the derivative."},
					],
					output={},
				),
				"PRM",  							# _test_type
				(								# _description
					"Detailed vs superficial calculus explanation",
				),
				(								# reasoning_analysis
					"Better applies the power rule correctly, worse keeps the constant and misapplies exponents",
				),
				True,  							# expected_is_better
				None,  							# expected_exception
				id="prm_calculus_detail",
			),
		],
	)
	def test_prm_semantic_validation(
		self,
		evaluator: TreeOfThoughtEvaluator,
		better_state: State,
		worse_state: State,
		_test_type: str,
		_description: str,
		reasoning_analysis: str,
		expected_is_better: bool | None,
		expected_exception: type[BaseException] | None,
	):
		"""Test PRM semantic validation (better vs worse reasoning)."""
		if expected_exception is not None:
			with pytest.raises(expected_exception):
				evaluator(
					states=better_state,
					n_samples_evaluator=1,
					evaluator_temperature=0.7,
					evaluator_max_tokens=256,
					demos=PRM_DEMOS,
				)
			return

		try:
			better_result = evaluator(
				states=better_state,
				n_samples_evaluator=1,
				evaluator_temperature=0.7,
				evaluator_max_tokens=256,
				demos=PRM_DEMOS,
			)
			worse_result = evaluator(
				states=worse_state,
				n_samples_evaluator=1,
				evaluator_temperature=0.7,
				evaluator_max_tokens=256,
				demos=PRM_DEMOS,
			)

			# Extract scores from results
			better_eval = better_result[0][0] if better_result and better_result[0] else None
			worse_eval = worse_result[0][0] if worse_result and worse_result[0] else None

			assert isinstance(better_eval, EvaluationResult), "Better result is not an EvaluationResult"
			assert isinstance(worse_eval, EvaluationResult), "Worse result is not an EvaluationResult"

			better_score = better_eval.score
			worse_score = worse_eval.score

			if expected_is_better:
				assert better_score > worse_score, (
					f"Expected better score ({better_score}) > worse score ({worse_score})."
					f"\tAnalysis: {reasoning_analysis}"
				)
			else:
				assert better_score < worse_score, (
					f"Expected better score ({better_score}) < worse score ({worse_score})."
					f"\tAnalysis: {reasoning_analysis}"
				)
		except Exception as e:
			pytest.fail(f"PRM validation failed: {e}")

	@pytest.mark.parametrize(
		# Parameter names
		[
			"better_state",
			"worse_state",
			"_test_type",
			"_description",
			"reasoning_analysis",
			"expected_is_better",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				State(						# better_state (correct decimal)
					input={"math_problem": "Convert 3/8 to a decimal"},
					reasoning=[
						{"math_operation": "To convert fraction to decimal, I divide numerator by denominator"},
						{"math_operation": "3 ÷ 8 = 0.375"},
					],
					output={"answer": "3/8 = 0.375"},
				),
				State(						# worse_state (rounded incorrectly)
					input={"math_problem": "Convert 3/8 to a decimal"},
					reasoning=[
						{"math_operation": "To convert fraction to decimal, I divide numerator by denominator"},
						{"math_operation": "3 ÷ 8 = 0.38"},
					],
					output={"answer": "3/8 = 0.38"},
				),
				"ORM",  					# _test_type
				(							# _description
					"Correct vs incorrect decimal conversion",
				),
				(							# reasoning_analysis
					"Better gives correct answer 0.375, worse has calculation error giving 0.38",
				),
				True,  						# expected_is_better
				None,  						# expected_exception
				id="orm_decimal_conversion",
			),
			pytest.param(
				State(						# better_state (correct factual answer)
					input={"math_problem": "Who won the 2012 Presidential Election in the US?"},
					reasoning=[
						{"math_operation": "The 2012 election was between Barack Obama and Mitt Romney"},
						{"math_operation": "Barack Obama was the incumbent president running for re-election"},
					],
					output={"answer": "Barack Obama"},
				),
				State(						# worse_state (incorrect factual answer)
					input={"math_problem": "Who won the 2012 Presidential Election in the US?"},
					reasoning=[
						{"math_operation": "The 2012 election was between Barack Obama and Mitt Romney"},
						{"math_operation": "Mitt Romney challenged the incumbent"},
					],
					output={"answer": "Mitt Romney"},
				),
				"ORM",  					# _test_type
				(							# _description
					"Correct vs incorrect factual answer",
				),
				(							# reasoning_analysis
					"Better gives correct winner (Obama), worse gives incorrect winner (Romney)",
				),
				True,  						# expected_is_better
				None,  						# expected_exception
				id="orm_factual_answer",
			),
			pytest.param(
				State(						# better_state (fraction result)
					input={
						"math_problem": "A pizza is cut into 8 equal slices. If 3 slices are eaten, what fraction remains?"
					},
					reasoning=[
						{"math_operation": "Total slices = 8, eaten = 3, remaining = 8 - 3 = 5"},
						{"math_operation": "Fraction remaining = 5/8"},
					],
					output={"answer": "5/8 of the pizza remains"},
				),
				State(						# worse_state (missing fraction)
					input={
						"math_problem": "A pizza is cut into 8 equal slices. If 3 slices are eaten, what fraction remains?"
					},
					reasoning=[{"math_operation": "8 - 3 = 5 slices remain"}],
					output={"answer": "5 slices remain"},
				),
				"ORM",  					# _test_type
				(							# _description
					"Complete vs incomplete fraction solution",
				),
				(							# reasoning_analysis
					"Better gives proper fraction answer, worse only gives count without fraction",
				),
				True,  						# expected_is_better
				None,  						# expected_exception
				id="orm_fraction_solution",
			),
			pytest.param(
				State(						# better_state (systematic justification)
					input={"math_problem": "Is 17 a prime number?"},
					reasoning=[
						{"math_operation": "A prime number has exactly two factors: 1 and itself"},
						{"math_operation": "I need to check if 17 has any factors other than 1 and 17"},
						{"math_operation": "Testing divisors up to √17 ≈ 4.1: 2, 3, 4"},
						{"math_operation": "17 ÷ 2 = 8.5 (not divisible), 17 ÷ 3 = 5.67 (not divisible), 17 ÷ 4 = 4.25 (not divisible)"},
					],
					output={"answer": "Yes, 17 is prime because it has no divisors other than 1 and 17"},
				),
				State(						# worse_state (minimal justification)
					input={"math_problem": "Is 17 a prime number?"},
					reasoning=[{"math_operation": "17 is not divisible by small numbers like 2, 3"}],
					output={"answer": "Yes, 17 is prime"},
				),
				"ORM",  					# _test_type
				(							# _description
					"Well-justified vs poorly justified prime check",
				),
				(							# reasoning_analysis
					"Better shows systematic checking method, worse gives minimal justification",
				),
				True,  						# expected_is_better
				None,  						# expected_exception
				id="orm_prime_check",
			),
			pytest.param(
				State(						# better_state (shows work)
					input={"math_problem": "Find the slope of the line through points (2, 5) and (7, 15)"},
					reasoning=[
						{"math_operation": "Using slope formula: m = (y₂ - y₁)/(x₂ - x₁)"},
						{"math_operation": "Points: (2, 5) and (7, 15), so x₁=2, y₁=5, x₂=7, y₂=15"},
						{"math_operation": "m = (15 - 5)/(7 - 2) = 10/5 = 2"},
					],
					output={"answer": "The slope is 2"},
				),
				State(						# worse_state (no reasoning)
					input={"math_problem": "Find the slope of the line through points (2, 5) and (7, 15)"},
					reasoning=[],
					output={"answer": "2"},
				),
				"ORM",  					# _test_type
				(							# _description
					"Appropriately detailed vs too brief slope calculation",
				),
				(							# reasoning_analysis
					"Better shows formula and substitution, worse gives bare answer with no work",
				),
				True,  						# expected_is_better
				None,  						# expected_exception
				id="orm_slope_detail",
			),
			pytest.param(
				State(						# better_state (includes units)
					input={
						"math_problem": (
							"A car travels 150 miles in 3 hours. What is its average speed in miles per "
							"hour (mph)? Provide the answer including units."
						)
					},
					reasoning=[
						{"math_operation": "Average speed = total distance / total time"},
						{"math_operation": "Speed = 150 miles / 3 hours = 50 miles per hour"},
					],
					output={"answer": "The average speed is 50 mph"},
				),
				State(						# worse_state (wrong unit)
					input={
						"math_problem": (
							"A car travels 150 miles in 3 hours. What is its average speed in miles per "
							"hour (mph)? Provide the answer including units."
						)
					},
					reasoning=[
						{"math_operation": "Speed = distance / time"},
						{"math_operation": "Speed = 150 / 3 = 50"},
					],
					output={"answer": "The average speed is 50 km/h"},
				),
				"ORM",  					# _test_type
				(							# _description
					"Correct units vs incorrect units in speed calculation",
				),
				(							# reasoning_analysis
					"Better includes mph as requested, worse gives an incorrect unit (km/h)",
				),
				True,  						# expected_is_better
				None,  						# expected_exception
				id="orm_speed_units",
			),
		],
	)
	def test_orm_semantic_validation(
		self,
		evaluator: TreeOfThoughtEvaluator,
		better_state: State,
		worse_state: State,
		_test_type: str,
		_description: str,
		reasoning_analysis: str,
		expected_is_better: bool | None,
		expected_exception: type[BaseException] | None,
	):
		"""Test ORM semantic validation (better vs worse solution).

		Args:
			evaluator: The evaluator to use.
			better_state: The better state to evaluate.
			worse_state: The worse state to evaluate.
			_test_type: The type of test.
			_description: The description of the test.
			reasoning_analysis: The reasoning analysis of the test.
			expected_is_better: Whether the better state is expected to be better.
			expected_exception: The exception to expect.
		"""
		try:
			if expected_exception is not None:
				with pytest.raises(expected_exception):
					evaluator(
						states=better_state,
						n_samples_evaluator=1,
						evaluator_temperature=0.7,
						evaluator_max_tokens=256,
						demos=ORM_DEMOS,
					)
				return
			better_result = evaluator(
				states=better_state,
				n_samples_evaluator=1,
				evaluator_temperature=0.7,
				evaluator_max_tokens=256,
				demos=ORM_DEMOS,
			)
			worse_result = evaluator(
				states=worse_state,
				n_samples_evaluator=1,
				evaluator_temperature=0.7,
				evaluator_max_tokens=256,
				demos=ORM_DEMOS,
			)

			# Extract scores from results
			better_eval = better_result[0][0] if better_result and better_result[0] else None
			worse_eval = worse_result[0][0] if worse_result and worse_result[0] else None

			assert isinstance(better_eval, EvaluationResult), "Better result is not an EvaluationResult"
			assert isinstance(worse_eval, EvaluationResult), "Worse result is not an EvaluationResult"

			better_score = better_eval.score
			worse_score = worse_eval.score

			if expected_is_better:
				assert better_score > worse_score, (
					f"Expected better score ({better_score}) > worse score ({worse_score})."
					f"\tAnalysis: {reasoning_analysis}"
				)
			else:
				assert better_score < worse_score, (
					f"Expected better score ({better_score}) < worse score ({worse_score})."
					f"\tAnalysis: {reasoning_analysis}"
				)
		except Exception as e:
			pytest.fail(f"ORM validation failed: {e}")

	@pytest.mark.parametrize(
		# Parameter names
		[
			"argument_state",
			"expected_result_count",
			"expected_exception",
		],
		# Parameter values
		[
			pytest.param(
				State(
					input={"topic": "Renewable energy", "stance": "PRO"},
					reasoning=[
						{"claim": "Renewable energy sources like solar and wind are becoming increasingly cost-competitive with fossil fuels."},
					],
					output={},
				),  # argument_state (process-style reasoning only)
				1,  # expected_result_count
				None,  # expected_exception
				id="argument_process_only",
			),
			pytest.param(
				State(
					input={"topic": "Universal basic income is a good idea", "stance": "PRO"},
					reasoning=[
							{"claim": "Universal basic income provides economic security in an era of automation."},
							{"claim": "It simplifies welfare systems and reduces administrative costs."},
							{"claim": "Pilot programs have shown positive effects on mental health and employment."},
						],
					output={
						"argument": "Universal basic income (UBI) offers a pragmatic solution to economic challenges posed by automation and job displacement. By providing a financial safety net, UBI enables individuals to pursue education, entrepreneurship, or caregiving without the fear of destitution. Evidence from pilot programs in Finland and Kenya demonstrates that UBI recipients experience reduced stress, improved health outcomes, and sustained employment rates. Furthermore, UBI streamlines bureaucratic welfare systems, reducing overhead costs while ensuring no one falls through the cracks."
					},
				),  # argument_state (outcome-style full argument)
				1,  # expected_result_count
				None,  # expected_exception
				id="argument_outcome_full",
			),
			pytest.param(
				State(
					input={"topic": "Remote work policies", "stance": "PRO"},
					reasoning=[
							{"claim": "Remote work increases employee productivity and satisfaction."},
							{"claim": "It reduces commuting time and environmental impact."},
							{"claim": "Companies save on office space and overhead costs."},
						],
					output={
						"argument": "Remote work policies represent a win-win for both employers and employees. Studies consistently show that remote workers are more productive, with fewer distractions and greater autonomy over their schedules. Employees benefit from eliminating lengthy commutes, gaining hours for personal pursuits and reducing their carbon footprint. Meanwhile, companies can reduce real estate costs and access a global talent pool unrestricted by geography."
					},
				),  # argument_state (outcome-style alternate topic)
				1,  # expected_result_count
				None,  # expected_exception
				id="argument_outcome_remote_work",
			),
		],
	)
	def test_argument_evaluator_execution(
		self,
		local_lm,
		argument_state: State,
		expected_result_count: int,
		expected_exception: type[BaseException] | None,
	):
		"""Test execution of argument evaluators."""
		dspy.settings.configure(lm=local_lm)

		argument_evaluator = TreeOfThoughtEvaluator(
			generator_signature=GenerateArgumentWithReasoning,
			evaluator_signature=ArgumentEvaluatorMultiDimensional,
			consider_reasoning_in_final_eval=True,
			verbosity="info",
		)

		if expected_exception is not None:
			with pytest.raises(expected_exception):
				argument_evaluator(
					states=[argument_state],
					n_samples_evaluator=1,
					evaluator_temperature=0.7,
				)
			return

		try:
			results = argument_evaluator(
				states=[argument_state],
				n_samples_evaluator=1,
				evaluator_temperature=0.7,
				evaluator_max_tokens=256,
			)
			assert results is not None
			assert len(results) == expected_result_count
			assert isinstance(results[0][0], EvaluationResult)
		except Exception as e:
			pytest.fail(f"Argument evaluator execution failed: {e}")


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
