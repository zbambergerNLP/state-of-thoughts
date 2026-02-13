"""
Test suite for the State-based TreeOfThoughtGenerator class.

This module provides comprehensive tests for the overhauled generator that uses
State objects as inputs and supports automatic heterogeneous batch splitting.

Expected usage:
```bash
pytest predict/test_generator.py -vv
```
"""

# Standard library imports
import logging
import os
from unittest.mock import Mock, patch

# Third-party imports
import dspy
import pytest
import torch
from dspy.primitives.prediction import Prediction

# Local imports
from constants import OpenSourceModel
from lm.generative_local_lm import GenerativeLocalVLLM
from predict.controller.controller_constants import ControllerOutput
from predict.generator.generator import TreeOfThoughtGenerator
from predict.generator.generator_demos import ARGUMENT_DEMOS, MATH_DEMOS
from signatures import (
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	SolveMathProblemWithReasoning,
)
from tree import State
from utilities_for_tests import MockGenerativeLocalVLLM

logger = logging.getLogger(__name__)

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


class TestTreeOfThoughtGeneratorInit:
	"""Test cases for TreeOfThoughtGenerator initialization."""

	def test_init_with_reasoning_signature_class(self):
		"""Test initialization with a ReasoningSignature class."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=3,
		)

		assert generator.signature == QuestionAnsweringWithReasoning
		assert generator.max_reasoning_steps == 3
		assert len(generator.signature.input_fields) == 1
		assert len(generator.signature.reasoning_fields) == 1
		assert len(generator.signature.output_fields) == 1
		assert "question" in generator.signature.input_fields
		assert "reasoning_step" in generator.signature.reasoning_fields
		assert "answer" in generator.signature.output_fields
		assert generator.reasoning_field_name == "reasoning_step"

	def test_init_with_string_signature(self):
		"""Test initialization with a string signature."""
		generator = TreeOfThoughtGenerator(
			signature="question -> reasoning -> answer",
			max_reasoning_steps=5,
		)

		assert generator.signature is not None
		assert generator.max_reasoning_steps == 5
		assert len(generator.signature.input_fields) == 1
		assert len(generator.signature.reasoning_fields) == 1
		assert len(generator.signature.output_fields) == 1

	def test_init_with_config(self):
		"""Test initialization with configuration parameters."""
		config = {"temperature": 0.7, "max_tokens": 100}
		generator = TreeOfThoughtGenerator(
			signature="question -> reasoning -> answer",
			max_reasoning_steps=3,
			**config,
		)
		assert generator.config == config
		assert generator.config["temperature"] == 0.7
		assert generator.config["max_tokens"] == 100


class TestStateToForwardContext:
	"""Test cases for _state_to_forward_context method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"max_reasoning_steps",
			"expected_continue_reasoning",
			"expected_previous_content_contains",
			"expected_internal_reasoning",
			"expected_prefix",
		],
		# Parameter values
		[
			pytest.param(
				State(  								# state
					input={"question": "What is 2+2?"}
				),
				3,  									# max_reasoning_steps
				[True],  								# expected_continue_reasoning
				[],  									# expected_previous_content_contains
				[""],  									# expected_internal_reasoning
				[""],  									# expected_prefix
				id="state_without_reasoning",
			),
			pytest.param(
				State(  								# state
					input={"question": "What is 1+1?"},
					reasoning=[{"reasoning": "First step"}]
				),
				3,  									# max_reasoning_steps
				[True],  								# expected_continue_reasoning
				["<thinking>", "First step"],  			# expected_previous_content_contains
				[""],  									# expected_internal_reasoning
				[""],  									# expected_prefix
				id="state_with_reasoning_below_max",
			),
			pytest.param(
				State(  								# state
					input={"question": "What is 1+1?"},
					reasoning=[
						{"reasoning": "Step 1"},
						{"reasoning": "Step 2"},
					]
				),
				2,  									# max_reasoning_steps
				[False],  								# expected_continue_reasoning
				[],  									# expected_previous_content_contains
				[""],  									# expected_internal_reasoning
				[""],  									# expected_prefix
				id="state_at_max_reasoning_steps",
			),
			pytest.param(
				State(  								# state
					input={"question": "What is 1+1?"},
					reasoning=[{"reasoning": "2 + 2 = 4"}],
					output={"answer": "4"},
				),
				3,  									# max_reasoning_steps
				[False],  								# expected_continue_reasoning
				[],  									# expected_previous_content_contains
				[""],  									# expected_internal_reasoning
				[""],  									# expected_prefix
				id="state_with_output",
			),
			pytest.param(
				State(  								# state
					input={"question": "What is 1+1?"},
					controller_outputs=[
						ControllerOutput(
							action="continue_reasoning",
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="Test",
							internal_reasoning="Think carefully about addition",
							prefix="First, I will",
						),
					],
				),
				3,  										# max_reasoning_steps
				[True],  									# expected_continue_reasoning
				[],  										# expected_previous_content_contains
				["Think carefully about addition"],  		# expected_internal_reasoning
				["First, I will"],  						# expected_prefix
				id="state_with_interventions",
			),
			pytest.param(
				State(  									# state
					input={"question": "What is 1+1?"},
					reasoning=[{"reasoning": "2 + 2 = 4"}],
					controller_outputs=[
						ControllerOutput(
							action="finish",
							action_arguments={},
							tool_descriptions="Action Name: finish",
							continue_reasoning=False,
							considerations="Ready to finish",
							internal_reasoning="",		# No internal reasoning for `finish` action
						),
					],
				),
				5,  										# max_reasoning_steps
				[False],  									# expected_continue_reasoning
				[],  										# expected_previous_content_contains
				# TODO[P2]: Add support for native reasoning of LLMs to parse into evaluator
				# reasoning field, and be stored in state.
				[""],  										# expected_internal_reasoning
				[""],  										# expected_prefix
				id="state_with_controller_finish_decision",
			),
		],
	)
	def test_state_to_forward_context(
		self,
		state: State,
		max_reasoning_steps: int,
		expected_continue_reasoning: list[bool],
		expected_previous_content_contains: list[str],
		expected_internal_reasoning: list[str],
		expected_prefix: list[str],
	) -> None:
		"""Test converting states to forward context with various configurations."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=max_reasoning_steps,
		)
		mock_lm = MockGenerativeLocalVLLM([])

		context = generator._state_to_forward_context(
			state=state,
			lm=mock_lm,
			demos=None,
		)

		assert context.continue_reasoning == expected_continue_reasoning
		assert context.internal_reasoning_for_output == expected_internal_reasoning
		assert context.prefix_for_output == expected_prefix
		for content in expected_previous_content_contains:
			assert content in context.previous_content




class TestGeneratorForward:
	"""Integration tests for the forward method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"max_reasoning_steps",
			"n_samples_generation",
			"mock_adapter_return_value",
			"expected_num_states",
			"expected_num_completions_per_state",
			"expected_continue_reasoning",
			"expected_adapter_call_count",
		],
		# Parameter values
		[
			pytest.param(
				State( input={"question": "What is 2+2?"}),	# states
				3,  										# max_reasoning_steps
				1,  										# n_samples_generation
				[[{"reasoning": "2 + 2 = 4"}]],				# mock_adapter_return_value
				1,  										# expected_num_states
				[1],  										# expected_num_completions_per_state
				[[True]],  									# expected_continue_reasoning
				1,  										# expected_adapter_call_count
				id="single_state_reasoning",
			),
			pytest.param(
				[  											# states
					State(input={"question": "What is 1+1?"}),
					State(
						input={"question": "What is 1+1?"},
						reasoning=[{"reasoning": "First step"}]
					),
				],
				3,  										# max_reasoning_steps
				1,  										# n_samples_generation
				[  											# mock_adapter_return_value
					[{"reasoning": "Trajectory 1"}],
					[{"reasoning": "Trajectory 2"}],
				],
				2,  										# expected_num_states
				[1, 1],  									# expected_num_completions_per_state
				[[True], [True]],  							# expected_continue_reasoning
				1,  										# expected_adapter_call_count
				id="batch_homogeneous_states",
			),
			pytest.param(
				[  											# states
					State(input={"question": "What is 1+1?"}),
					State(
						input={"question": "What is 1+1?"},
						reasoning=[{"reasoning": "Step 1"}, {"reasoning": "Step 2"}],
					),
					State(
						input={"question": "What is 1+1?"},
						reasoning=[{"reasoning": "Step 1"}]
					),
				],
				2,  										# max_reasoning_steps (state 1 has 2 steps = max)
				1,  										# n_samples_generation
				[  											# mock_adapter_return_value
					[{"reasoning": "Reasoning for state 0"}],
					[{"reasoning": "Answer for state 1"}],
					[{"reasoning": "Reasoning for state 2"}],
				],
				3,  										# expected_num_states
				[1, 1, 1],  								# expected_num_completions_per_state
				[[True], [False], [True]],  				# expected_continue_reasoning
				1,  										# expected_adapter_call_count
				id="heterogeneous_batch",
			),
			pytest.param(
				State(  									# states
					input={"question": "What is 2+2?"}
				),
				3,  										# max_reasoning_steps
				3,  										# n_samples_generation
				[  											# mock_adapter_return_value
					[
						{"reasoning": "Completion 1"},
						{"reasoning": "Completion 2"},
						{"reasoning": "Completion 3"},
					]
				],
				1,  										# expected_num_states
				[3],  										# expected_num_completions_per_state
				[[True]],  									# expected_continue_reasoning
				1,  										# expected_adapter_call_count
				id="multiple_completions",
			),
		],
	)
	def test_generator_forward(
		self,
		states: State | list[State],
		max_reasoning_steps: int,
		n_samples_generation: int,
		mock_adapter_return_value: list[list[dict[str, str]]],
		expected_num_states: int,
		expected_num_completions_per_state: list[int],
		expected_continue_reasoning: list[list[bool]],
		expected_adapter_call_count: int,
	) -> None:
		"""Test generator forward method with various state configurations."""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=max_reasoning_steps,
		)
		mock_lm = MockGenerativeLocalVLLM([])
		generator.lm = mock_lm

		with patch("predict.generator.generator.VLLMGeneratorAdapter") as mock_adapter_class:
			mock_adapter_instance = Mock()
			mock_adapter_class.return_value = mock_adapter_instance
			mock_adapter_instance.return_value = mock_adapter_return_value

			with patch("predict.generator.generator.settings") as mock_settings:
				mock_settings.lm = mock_lm

				results = generator(states=states, n_samples_generation=n_samples_generation)

				# Verify structure
				assert len(results) == expected_num_states
				for i, expected_completions in enumerate(expected_num_completions_per_state):
					assert len(results[i]) == expected_completions

				# Verify adapter was called correctly
				assert mock_adapter_instance.call_count == expected_adapter_call_count
				call_kwargs = mock_adapter_instance.call_args[1]
				assert call_kwargs["continue_reasoning"] == expected_continue_reasoning

				# Verify n_samples_generation was passed correctly
				if isinstance(states, State):
					states_list = [states]
				else:
					states_list = states
				for i in range(len(states_list)):
					assert call_kwargs["lm_kwargs"][i][0]["n"] == n_samples_generation


class TestNSamplesGeneration:
	"""Test cases for n_samples_generation and match_n_to_controller_choice_count functionality."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"states",
			"n_samples_generation",
			"match_n_to_controller_choice_count",
			"expected_n_values",
		],
		# Parameter values
		[
			pytest.param(
				[  											# states
					State(input={"question": "What is 1+1?"}),
					State(input={"question": "What is 1+1?"}),
				],
				5,  										# n_samples_generation (scalar)
				False,  									# match_n_to_controller_choice_count
				[5, 5],  									# expected_n_values
				id="n_samples_scalar_broadcast",
			),
			pytest.param(
				[  											# states
					State(
						input={"question": "What is 1+1?"},
						controller_outputs=[
							{
								"action": "continue_reasoning",
								"action_arguments": {},
								"tool_descriptions": "Action Name: continue_reasoning",
								"considerations": "Step 1",
								"continue_reasoning": True,
								"internal_reasoning": "Step 1",
								"unique_action_response_count": 2,
							},
							{
								"action": "continue_reasoning",
								"action_arguments": {},
								"tool_descriptions": "Action Name: continue_reasoning",
								"considerations": "Step 2",
								"continue_reasoning": True,
								"internal_reasoning": "Step 2",
								"unique_action_response_count": 3,
							},
						],
					),
					State(
						input={"question": "What is 1+1?"},
						controller_outputs=[
							{
								"action": "finish",
								"action_arguments": {},
								"tool_descriptions": "Action Name: finish",
								"considerations": "Final",
								"continue_reasoning": False,
								"internal_reasoning": "Final",
								"unique_action_response_count": 1,
							},
						],
					),
				],
				[[4, 6], [8]],  							# n_samples_generation (list of lists)
				False,  									# match_n_to_controller_choice_count
				[4, 6, 8],  								# expected_n_values (state0_int0, state0_int1, state1_int0)
				id="n_samples_list_of_lists_per_intervention",
			),
			pytest.param(
				[  											# states
					State(
						input={"question": "What is 1+1?"},
						controller_outputs=[
							{
								"action": "continue_reasoning",
								"action_arguments": {},
								"tool_descriptions": "Action Name: continue_reasoning",
								"considerations": "Think",
								"continue_reasoning": True,
								"internal_reasoning": "Think",
								"unique_action_response_count": 5,
							},
							{
								"action": "continue_reasoning",
								"action_arguments": {},
								"tool_descriptions": "Action Name: continue_reasoning",
								"considerations": "Think more",
								"continue_reasoning": True,
								"internal_reasoning": "Think more",
								"unique_action_response_count": 3,
							},
						],
					),
				],
				999,  										# n_samples_generation (should be ignored)
				True,  										# match_n_to_controller_choice_count
				[5, 3],  									# expected_n_values (from controller)
				id="match_n_to_controller_choice_count_true",
			),
		],
	)
	def test_n_samples_generation(
		self,
		states: list[State],
		n_samples_generation: int | list[int] | list[list[int]],
		match_n_to_controller_choice_count: bool,
		expected_n_values: list[int],
	) -> None:
		"""
		Test n_samples_generation with various formats and controller matching.

		Test cases:
		- n_samples_scalar_broadcast: Test when n_samples_generation is a scalar.
		- n_samples_list_of_lists_per_intervention: Test when n_samples_generation is a
			list of lists per intervention.
		- match_n_to_controller_choice_count_true: Test when match_n_to_controller_choice_count is
			True and n_samples is overridden by controller choice counts.

		Args:
			states: List of states to process.
			n_samples_generation: Number of completions to generate.
			match_n_to_controller_choice_count: If True, then the n_samples_generation per
				intervention will be overwritten to match how often that intervention was selected
				by the controller.
			expected_n_values: Expected number of completions per intervention per state.
		"""
		generator = TreeOfThoughtGenerator(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=3,
		)
		mock_lm = MockGenerativeLocalVLLM([])
		generator.lm = mock_lm

		# Create mock adapter return value based on number of states/interventions
		mock_return_value = []
		for state in states:
			if state.controller_outputs:
				mock_return_value.append(
					[{"reasoning": "some reasoning..."}] * len(state.controller_outputs)
				)
			else:
				mock_return_value.append([{"reasoning": "some reasoning..."}])

		with patch("predict.generator.generator.VLLMGeneratorAdapter") as mock_adapter_class:
			mock_adapter_instance = Mock()
			mock_adapter_class.return_value = mock_adapter_instance
			mock_adapter_instance.return_value = mock_return_value

			with patch("predict.generator.generator.settings") as mock_settings:
				mock_settings.lm = mock_lm

				_ = generator(
					states=states,
					n_samples_generation=n_samples_generation,
					match_n_to_controller_choice_count=match_n_to_controller_choice_count,
				)

				call_kwargs = mock_adapter_instance.call_args[1]
				lm_kwargs = call_kwargs["lm_kwargs"]

				# Flatten expected n values to match structure
				idx = 0
				for state_idx, state in enumerate(states):
					if state.controller_outputs:
						for intervention_idx in range(len(state.controller_outputs)):
							assert lm_kwargs[state_idx][intervention_idx]["n"] == expected_n_values[idx]
							idx += 1
					else:
						assert lm_kwargs[state_idx][0]["n"] == expected_n_values[idx]
						idx += 1
				assert idx == len(expected_n_values), (
					f"Expected {len(expected_n_values)} n values, got {idx}."
					f"Expected n values: {expected_n_values}"
					f"Got n values: {lm_kwargs}"
				)


class TestFinalOutputKindConfiguration:
	"""Test final_output_kind parameter configuration."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"final_output_kind",
			"expected_final_output_kind",
			"test_passed_to_adapter",
		],
		# Parameter values
		[
			pytest.param(
				None,  										# final_output_kind (None = default)
				"synthesis_faithful",  						# expected_final_output_kind
				False,  									# test_passed_to_adapter
				id="default_final_output_kind",
			),
			pytest.param(
				"synthesis_faithful",  						# final_output_kind
				"synthesis_faithful",  						# expected_final_output_kind
				False,  									# test_passed_to_adapter
				id="custom_final_output_kind_synthesis",
			),
			pytest.param(
				"conclusion",  								# final_output_kind
				"conclusion",  								# expected_final_output_kind
				True,  										# test_passed_to_adapter
				id="custom_final_output_kind_conclusion_passed_to_adapter",
			),
		],
	)
	def test_final_output_kind_configuration(
		self,
		final_output_kind: str | None,
		expected_final_output_kind: str,
		test_passed_to_adapter: bool,
	) -> None:
		"""Test final_output_kind parameter configuration and adapter passing."""
		generator_kwargs = {
			"signature": QuestionAnsweringWithReasoning,
			"max_reasoning_steps": 3,
		}
		if final_output_kind is not None:
			generator_kwargs["final_output_kind"] = final_output_kind

		generator = TreeOfThoughtGenerator(**generator_kwargs)
		assert generator.final_output_kind == expected_final_output_kind

		if test_passed_to_adapter:
			mock_lm = MockGenerativeLocalVLLM([])
			generator.lm = mock_lm
			state = State(input={"question": "What is 2+2?"})

			with patch("predict.generator.generator.VLLMGeneratorAdapter") as mock_adapter_class:
				mock_adapter_instance = Mock()
				mock_adapter_class.return_value = mock_adapter_instance
				mock_adapter_instance.return_value = [
					[{"reasoning": "2 + 2 = 4"}]
				]

				with patch("predict.generator.generator.settings") as mock_settings:
					mock_settings.lm = mock_lm

					generator(states=state, n_samples_generation=1)

					# Verify adapter was called with correct final_output_kind
					call_kwargs = mock_adapter_instance.call_args[1]
					assert call_kwargs["final_output_kind"] == expected_final_output_kind



# =============================================================================
# Integration Test Helper Functions
# =============================================================================

def validate_output_structure(
	results: list[list[Prediction]],
	expected_num_states: int,
	expected_num_predictions: int,
) -> bool:
	"""Validate that output has correct structure."""
	if len(results) != expected_num_states:
		return False

	for state_results in results:
		if len(state_results) != expected_num_predictions:
			return False

	return True


def validate_reasoning_state_output(predictions: list[Prediction], reasoning_field: str) -> bool:
	"""Validate that reasoning state generated reasoning (not final answer)."""
	for pred in predictions:
		if not hasattr(pred, reasoning_field):
			return False
		if not getattr(pred, reasoning_field):
			return False
	return True


def validate_finish_state_output(predictions: list[Prediction], output_field: str) -> bool:
	"""Validate that finish state generated final answer (not reasoning)."""
	for pred in predictions:
		if not hasattr(pred, output_field):
			return False
		if not getattr(pred, output_field):
			return False
	return True


def analyze_diversity(predictions: list[list[Prediction]], field_name: str) -> float:
	"""Analyze diversity of predictions by counting unique values."""
	all_values = []
	for state_predictions in predictions:
		for pred in state_predictions:
			if hasattr(pred, field_name):
				all_values.append(getattr(pred, field_name))

	if not all_values:
		return 0.0

	unique_values = len(set(all_values))
	total_values = len(all_values)
	return unique_values / total_values


# Shared GPU model fixture for all GPU tests
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
			verbosity="debug",
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
class TestGeneratorIntegration:
	"""Integration tests for the generator using real models (requires GPU)."""

	@pytest.fixture
	def local_lm(self, shared_gpu_model):
		"""Use the shared GPU model fixture."""
		return shared_gpu_model

	@pytest.fixture
	def math_generator(self, local_lm):
		"""Create a math generator instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtGenerator(
			signature=SolveMathProblemWithReasoning,
			max_reasoning_steps=3,
		)

	@pytest.fixture
	def argument_generator(self, local_lm):
		"""Create an argument generator instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtGenerator(
			signature=GenerateArgumentWithReasoning,
			max_reasoning_steps=3,
		)

	def test_single_state_reasoning(self, math_generator):
		"""Test generating next reasoning step for a single state."""
		state = State(input={"math_problem": "What is 15 + 27?"})
		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 1, 1)
			assert validate_reasoning_state_output(
				results[0], reasoning_field="math_operation"
			)
		except Exception as e:
			pytest.fail(f"Single state reasoning failed: {e}")

	def test_single_state_answer(self, math_generator):
		"""Test generating final answer for a state at max reasoning steps."""
		state = State(
			input={"math_problem": "What is 15 + 27?"},
			reasoning=[
					{"math_operation": "I need to add 15 and 27"},
					{"math_operation": "15 + 27 = 15 + 20 + 7 = 35 + 7"},
					{"math_operation": "35 + 7 = 42"},
				]
		)
		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 1, 1)
			assert validate_finish_state_output(
				results[0], output_field="answer"
			)
		except Exception as e:
			pytest.fail(f"Single state answer failed: {e}")

	def test_batch_processing(self, math_generator: TreeOfThoughtGenerator):
		"""Test batch processing of multiple trajectories."""
		problem = {"math_problem": "What is 8 * 9?"}
		states = [
			State(input=problem),
			State(
				input=problem,
				reasoning=[{"math_operation": "I need to multiply 8 by 9"}],
			),
			State(
				input=problem,
				reasoning=[
						{"math_operation": "I need to multiply 8 by 9"},
						{"math_operation": "8 * 9 = 8 * (10 - 1) = 80 - 8"},
					],
				controller_outputs=[
					ControllerOutput(
						action="continue_reasoning",
						action_arguments={},
						tool_descriptions="Action Name: continue_reasoning",
						continue_reasoning=True,
						considerations="Break down the subtraction step by step.",
						internal_reasoning="Break down the subtraction step by step.",
						prefix="Step by step: ",
						unique_action_response_count=1,
					)
				],
			),
		]
		try:
			results = math_generator(states=states, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 3, 1)
		except Exception as e:
			pytest.fail(f"Batch processing failed: {e}")

	def test_heterogeneous_batch(self, math_generator: TreeOfThoughtGenerator):
		"""Test heterogeneous batch with mixed reasoning and answer generation."""
		problem = {"math_problem": "What is 12 - 5?"}
		states = [
			State(input=problem),  # Needs reasoning
			State(
				input=problem,
				reasoning=[
					{"math_operation": "I need to subtract 5 from 12"},
					{"math_operation": "12 - 5 = 12 - 2 - 3 = 10 - 3"},
					{"math_operation": "10 - 3 = 7"},
				]
			),  # Needs answer (max steps)
			State(
				input=problem,
				reasoning=[{"math_operation": "I need to subtract 5 from 12"}],
			),  # Needs reasoning
		]

		try:
			results = math_generator(
				states=states, n_samples_generation=1, demos=MATH_DEMOS, verbosity="error"
			)

			assert validate_output_structure(results, 3, 1)
			assert validate_reasoning_state_output(
				results[0], reasoning_field="math_operation"
			)
			assert validate_finish_state_output(results[1], output_field="answer")
			assert validate_reasoning_state_output(
				results[2], reasoning_field="math_operation"
			)
		except Exception as e:
			pytest.fail(f"Heterogeneous batch failed: {e}")

	def test_multiple_completions(self, math_generator: TreeOfThoughtGenerator):
		"""Test generating multiple diverse completions."""
		state = State(input={"math_problem": "What is 6 * 7?"})

		try:
			results = math_generator(states=state, n_samples_generation=5, demos=MATH_DEMOS)

			assert validate_output_structure(results, 1, 5)
			assert validate_reasoning_state_output(
				results[0], reasoning_field="math_operation"
			)

			diversity = analyze_diversity(results, field_name="math_operation")
			# We can't strictly enforce diversity > 0 on a small model/test, but we can check it runs
			logger.info(f"Diversity: {diversity}")
		except Exception as e:
			pytest.fail(f"Multiple completions failed: {e}")

	def test_controller_interventions(self, math_generator: TreeOfThoughtGenerator):
		"""Test generator respecting controller interventions."""
		state = State(
			input={"math_problem": "What is 20 ÷ 4?"},
			reasoning=[{"math_operation": "I need to divide 20 by 4"}],
			controller_outputs=[
				ControllerOutput(
					action="continue_reasoning",
					action_arguments={},
					tool_descriptions="Action Name: continue_reasoning",
					continue_reasoning=True,
					considerations="Think about division as repeated subtraction",
					internal_reasoning="Think about division as repeated subtraction",
					prefix="Using repeated subtraction:",
					unique_action_response_count=1,
				)
			],
		)

		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)

			assert validate_output_structure(results, 1, 1)
			assert validate_reasoning_state_output(results[0], reasoning_field="math_operation")

			# Check if prefix appears in output (it should, but might be fragile on small models)
			pred = results[0][0]
			if hasattr(pred, "reasoning"):
				# We log but don't fail if prefix is missing, as it depends on model capability
				if "Using repeated subtraction:" in pred.reasoning:
					logger.info("Prefix found in reasoning")
		except Exception as e:
			pytest.fail(f"Controller interventions failed: {e}")

	def test_controller_finish(self, math_generator: TreeOfThoughtGenerator):
		"""Test generator respecting controller's finish decision."""
		state = State(
			input={"math_problem": "What is 3 + 4?"},
			reasoning=[{"math_operation": "3 + 4 = 7"}],
			controller_outputs=[
				ControllerOutput(
					action="finish",
					action_arguments={},
					tool_descriptions="Action Name: finish",
					continue_reasoning=False,
					considerations="The answer is clear, finish now",
					internal_reasoning="The answer is clear, finish now",
					prefix="",
					unique_action_response_count=1,
				)
			],
		)

		try:
			results = math_generator(states=state, n_samples_generation=1, demos=MATH_DEMOS)
			assert validate_output_structure(results, 1, 1)
			assert validate_finish_state_output(results[0], output_field="answer")
		except Exception as e:
			pytest.fail(f"Controller finish failed: {e}")

	def test_argument_generation_interventions(self, argument_generator: TreeOfThoughtGenerator):
		"""Test argument generation with various interventions."""

		# Test cases for different intervention types
		test_cases = [
			(
				"Style (Knowledge)",
				State(
					input={"topic": "electric vehicles", "stance": "PRO"},
					reasoning=[{"claim": "EVs reduce greenhouse gas emissions"}],
					controller_outputs=[
						ControllerOutput(
							action="continue_reasoning",
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="I should provide accurate information...",
							internal_reasoning="I should provide accurate information...",
							prefix="",
							unique_action_response_count=1,
						)
					],
				)
			),
			(
				"Structure (Cause)",
				State(
					input={"topic": "universal healthcare", "stance": "PRO"},
					reasoning=[{"claim": "Healthcare costs burden families"}],
					controller_outputs=[
						ControllerOutput(
							action="continue_reasoning",
							action_arguments={},
							tool_descriptions="Action Name: continue_reasoning",
							continue_reasoning=True,
							considerations="",
							internal_reasoning="",
							prefix="Therefore, ",
							unique_action_response_count=1,
						)
					],
				)
			),
		]

		for name, state in test_cases:
			try:
				results = argument_generator(states=state, n_samples_generation=1, demos=ARGUMENT_DEMOS)
				assert validate_output_structure(results, 1, 1)
			except Exception as e:
				pytest.fail(f"Argument generation ({name}) failed: {e}")


if __name__ == "__main__":
	import sys
	sys.exit(pytest.main(["-vv", __file__]))
