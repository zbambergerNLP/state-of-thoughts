"""
Tests for the controller_reranker module.

Expected usage:
```bash
pytest predict/test_controller_reranker.py -vv
```
"""

# Standard library imports
import json
import logging
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

# Third-party imports
import dspy
import pytest
import torch

# Local imports
from constants import OpenSourceModel
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.controller.controller_constants import (
	DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	ControllerOutput,
)
from predict.controller.controller_reranker import (
	TreeOfThoughtsControllerReranker,
	build_action_scoring_instructions,
)
from predict.controller.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	ControllerPrediction,
	ReasoningIntervention,
	return_action_if_single_option,
)
from signatures import (
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from signatures.example_signatures import (
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
)
from tree import State
from utilities_for_tests import MockScoringLocalVLLM

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
	[
		"task_instructions",
		"is_reasoning_empty",
		"num_actions_remaining",
		"expected_instructions",
	],
	[
		pytest.param(
			"Answer the provided question.",	# task_instructions
			False,								# is_reasoning_empty
			3,									# num_actions_remaining
			(									# expected_instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Answer the provided question."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
""".strip()
			),
			id="early_stopping_enabled_with_reasoning",
		),
		pytest.param(
			"Answer the provided question.",	# task_instructions
			True,								# is_reasoning_empty
			1,									# num_actions_remaining
			(									# expected_instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Answer the provided question."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 1 action remaining before you must return a final answer.
""".strip()
			),
			id="early_stopping_enabled_reasoning_empty",
		),
		pytest.param(
			"Answer the provided question.",	# task_instructions
			False,								# is_reasoning_empty
			2,									# num_actions_remaining
			(									# expected_instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Answer the provided question."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 2 actions remaining before you must return a final answer.
""".strip()
			),
			id="early_stopping_disabled_with_reasoning",
		),
		pytest.param(
			"Answer the provided question.",	# task_instructions
			True,								# is_reasoning_empty
			3,									# num_actions_remaining
			(									# expected_instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Answer the provided question."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
""".strip()
			),
			id="early_stopping_disabled_reasoning_empty",
		),
	],
)
def test_build_action_scoring_instructions(
	task_instructions: str,
	is_reasoning_empty: bool,
	num_actions_remaining: int,
	expected_instructions: str,
) -> None:
	"""
	Test that controller reranker instructions match expected formatting and conditionality.

	In particular, the instructions should:
	- Quote the user task.
	- Mention the "# Reasoning" heading only when reasoning exists in the query.
	- Mention optional early-stopping capability only when enabled.
	"""
	assert (
		build_action_scoring_instructions(
			task_instructions=task_instructions,
			is_reasoning_empty=is_reasoning_empty,
			num_actions_remaining=num_actions_remaining,
		)
		== expected_instructions
	)


# =============================================================================
# Test Fixtures for Action Space JSONs (matching test_controller.py style)
# =============================================================================



@pytest.fixture(scope="module")
def reranker_model_config() -> dict:
	"""
	Create the model configuration for reranker VLLM initialization.

	Returns:
		Dictionary with ScoringLocalVLLM initialization parameters for reranker model
	"""
	# Configuration settings
	model_directory = "/projects/BSTEWART/model_storage"
	# Use the same model constant as in test_tree_of_thoughts.py
	model_name = OpenSourceModel.QWEN_3_RERANKER_8B.value

	# Construct full model path (models are stored directly in model_directory)
	full_model_path = os.path.join(model_directory, model_name)

	if not os.path.exists(full_model_path):
		pytest.skip(f"Model not found at {full_model_path}")

	# Create configuration dictionary for reranker model
	config = {
		"model": full_model_path,
		"tensor_parallel_size": 1,
		"gpu_memory_utilization": 0.9,
		"max_model_len": 16_384,
		"trust_remote_code": True,
		"enable_prefix_caching": True,
		"verbosity": "info",
	}
	logger.info(f"Reranker model configuration created for: {config['model']}")
	return config


@pytest.fixture
def temp_action_space_styles(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for styles with 3 options.

	For reranker controller, 3 styles = 3 tools (each with no params).

	Args:
		tmp_path: Pytest fixture providing temporary directory path.

	Returns:
		Path to the created temporary JSON file.
	"""
	styles_json = {
		"name": "style",
		"definition": "Force the next reasoning step to adopt a specific rhetorical style.",
		"choices": {
			"Figurative Language": {
				"definition": "Use metaphor, simile, or analogy.",
				"internal_reasoning": (
					"I should employ non-literal comparison. "
				),
			},
			"Statistical & Data-Driven": {
				"definition": "Present numerical data or statistics.",
				"internal_reasoning": (
					"I should use numbers and data. "
				),
				"prefix": "The data shows:",
			},
			"Casual Slang": {
				"definition": "Use informal, slang-heavy language.",
				"internal_reasoning": (
					"I should strictly use casual slang like 'yo', 'dude', 'vibes', avoiding all formal or structural logic. "
				),
				"prefix": "Yo, check it:",
			},
		},
	}
	styles_file = tmp_path / "styles.json"
	styles_file.write_text(json.dumps(styles_json))
	return styles_file


@pytest.fixture
def temp_action_space_structures(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for structures with 3 options.

	For reranker controller, 3 structures = 3 tools (each with no params).

	Args:
		tmp_path: Pytest fixture providing temporary directory path.

	Returns:
		Path to the created temporary JSON file.
	"""
	structures_json = {
		"name": "structure",
		"definition": "Control the argumentative structure of the next reasoning step.",
		"choices": {
			"Causal Reasoning": {
				"definition": "State causes and effects.",
				"internal_reasoning": (
					"I should explain cause and effect. "
				),
				"prefix": "Therefore,",
			},
			"Contrast": {
				"definition": "Present contrasting viewpoints.",
				"internal_reasoning": (
					"I should present a contrasting view. "
				),
				"prefix": "However,",
			},
			"Tangential Remark": {
				"definition": "Make a completely unrelated, random remark.",
				"internal_reasoning": (
					"I should change the subject to something totally irrelevant like the weather or food. "
				),
				"prefix": "Speaking of bananas,",
			},
		},
	}
	structures_file = tmp_path / "structures.json"
	structures_file.write_text(json.dumps(structures_json))
	return structures_file


# =============================================================================
# Helper Functions for Controller Creation
# =============================================================================


def create_dummy_controller_output(
	action: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	action_arguments: dict[str, Any] | None = None,
	tool_descriptions: str = "Dummy tool description",
	considerations: str = "Dummy considerations",
	continue_reasoning: bool = True,
	internal_reasoning: str = "Dummy internal reasoning",
	prefix: str = "Dummy prefix",
) -> ControllerOutput:
	"""Create a dummy ControllerOutput for testing."""
	return ControllerOutput(
		action=action,
		action_arguments=action_arguments or {},
		tool_descriptions=tool_descriptions,
		considerations=considerations,
		continue_reasoning=continue_reasoning,
		internal_reasoning=internal_reasoning,
		prefix=prefix,
	)


def create_controller_reranker_with_mocked_lm(
	rerank_responses: list[list[float]] | None = None,
	signature: type[ReasoningSignature] = QuestionAnsweringWithReasoning,
	max_reasoning_steps: int = 5,
	action_space_paths: list[str | Path] | None = None,
	early_stopping_enabled: bool = True,
	forced_choice_function: Callable | None = None,
) -> TreeOfThoughtsControllerReranker:
	"""Create a controller reranker instance with mocked ScoringLocalVLLM.

	Args:
		rerank_responses: Mock scores in vLLM format: list[list[float]] = layers[scores].
		signature: The reasoning signature to use.
		max_reasoning_steps: Maximum reasoning steps.
		action_space_paths: Paths to action space JSON files.
		early_stopping_enabled: Whether to enable early stopping.
		forced_choice_function: Custom forced choice function.

	Returns:
		TreeOfThoughtsControllerReranker instance with LM set.
	"""
	mock_lm = MockScoringLocalVLLM(rerank_responses=rerank_responses)

	kwargs = {
		"signature": signature,
		"max_reasoning_steps": max_reasoning_steps,
		"action_space_paths": action_space_paths,
		"early_stopping_enabled": early_stopping_enabled,
	}
	if forced_choice_function is not None:
		kwargs["forced_choice_function"] = forced_choice_function

	controller = TreeOfThoughtsControllerReranker(**kwargs)
	controller.set_lm(mock_lm)
	return controller


def create_state(
	question: str = "What is 2+2?",
	reasoning_steps: list[str] | None = None,
	signature: type[ReasoningSignature] = QuestionAnsweringWithReasoning,
) -> State:
	"""Create a State for testing.

	Args:
		question: The question/input for the state.
		reasoning_steps: List of reasoning steps (defaults to empty).
		signature: Signature to derive reasoning field name from.

	Returns:
		State instance.
	"""
	reasoning_field_name = list(signature.reasoning_fields.keys())[0]
	reasoning_steps = reasoning_steps or []
	reasoning_dicts = [
		{reasoning_field_name: step} for step in reasoning_steps
	]
	return State(
		input={"question": question}, # Changed to "question" for compatibility with QuestionAnsweringWithReasoning
		reasoning=reasoning_dicts,
		controller_output_trajectory=[create_dummy_controller_output() for _ in reasoning_steps],
	)


# =============================================================================
# Test Classes
# =============================================================================


class TestControllerRerankerInitialization:
	"""Comprehensive initialization tests for TreeOfThoughtsControllerReranker."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"signature",
			"max_reasoning_steps",
			"action_space_keys",
			"early_stopping_enabled",
			"expected_num_tools",
			"expected_has_finish",
			"expected_tools_details",
		],
		# Parameter values
		[
			pytest.param(
				QuestionAnsweringWithReasoning,		# signature
				5,									# max_reasoning_steps
				None,								# action_space_keys (None = default)
				True,								# early_stopping_enabled
				2,									# expected_num_tools (continue + finish)
				True,								# expected_has_finish
				[
					{
						"name": "continue_reasoning",
						"desc": DEFAULT_TOOL.desc,
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="default_tools_with_early_stopping",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				5,
				None,								# action_space_keys
				False,								# early_stopping_enabled
				1,									# expected_num_tools (continue only)
				False,								# expected_has_finish
				[
					{
						"name": "continue_reasoning",
						"desc": DEFAULT_TOOL.desc,
						"args": {},
					},
				],
				id="default_tools_without_early_stopping",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				10,									# max_reasoning_steps
				None,
				True,
				2,									# expected_num_tools
				True,
				[
					{
						"name": "continue_reasoning",
						"desc": DEFAULT_TOOL.desc,
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="math_signature_default_tools",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				5,
				["styles"],							# 3 styles (Figurative, Statistical, Slang)
				True,
				4,									# 3 styles + finish
				True,
				[
					{
						"name": "figurative_language",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
""".strip(),
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="single_dimension_styles_only",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				5,
				["structures"],						# 3 structures (Causal, Contrast, Tangential)
				True,
				4,									# 3 structures + finish
				True,
				[
					{
						"name": "causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="single_dimension_structures_only",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				5,
				["styles", "structures"],			# Both dimensions
				True,
				10,									# 3*3=9 + finish = 10
				True,
				[
					{
						"name": "figurative_language_causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "figurative_language_contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "figurative_language_tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven_causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven_contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven_tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang_causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang_contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang_tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="two_dimensions_combined",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				5,
				["styles", "structures"],			# Both dimensions
				False,								# No early stopping
				9,									# 3×3=9 combinations (no finish)
				False,
				[
					{
						"name": "figurative_language_causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "figurative_language_contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "figurative_language_tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use metaphor, simile, or analogy.
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven_causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven_contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "statistical_data_driven_tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present numerical data or statistics.
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang_causal_reasoning",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
* State causes and effects.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang_contrast",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
* Present contrasting viewpoints.
""".strip(),
						"args": {},
					},
					{
						"name": "casual_slang_tangential_remark",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Use informal, slang-heavy language.
* Make a completely unrelated, random remark.
""".strip(),
						"args": {},
					},
				],
				id="two_dimensions_no_early_stopping",
			),
			# Edge cases: max_reasoning_steps boundaries
			pytest.param(
				QuestionAnsweringWithReasoning,
				1,									# Minimum reasonable value
				None,
				True,
				2,
				True,
				[
					{
						"name": "continue_reasoning",
						"desc": DEFAULT_TOOL.desc,
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="edge_case_min_reasoning_steps",
			),
			pytest.param(
				QuestionAnsweringWithReasoning,
				100,								# Very large value
				None,
				True,
				2,									# Action space None -> 2 tools (continue reasoning, finish)
				True,
				[
					{
						"name": "continue_reasoning",
						"desc": DEFAULT_TOOL.desc,
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="edge_case_max_reasoning_steps_large",
			),
			# Edge cases: dimension order shouldn't matter
			pytest.param(
				QuestionAnsweringWithReasoning,
				5,
				["structures", "styles"],			# Reversed order
				True,
				10,									# 3*3=9 + finish = 10
				True,
				[
					{
						"name": "causal_reasoning_figurative_language",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* State causes and effects.
* Use metaphor, simile, or analogy.
""".strip(),
						"args": {},
					},
					{
						"name": "causal_reasoning_statistical_data_driven",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* State causes and effects.
* Present numerical data or statistics.
""".strip(),
						"args": {},
					},
					{
						"name": "causal_reasoning_casual_slang",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* State causes and effects.
* Use informal, slang-heavy language.
""".strip(),
						"args": {},
					},
					{
						"name": "contrast_figurative_language",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present contrasting viewpoints.
* Use metaphor, simile, or analogy.
""".strip(),
						"args": {},
					},
					{
						"name": "contrast_statistical_data_driven",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present contrasting viewpoints.
* Present numerical data or statistics.
""".strip(),
						"args": {},
					},
					{
						"name": "contrast_casual_slang",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Present contrasting viewpoints.
* Use informal, slang-heavy language.
""".strip(),
						"args": {},
					},
					{
						"name": "tangential_remark_figurative_language",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Make a completely unrelated, random remark.
* Use metaphor, simile, or analogy.
""".strip(),
						"args": {},
					},
					{
						"name": "tangential_remark_statistical_data_driven",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Make a completely unrelated, random remark.
* Present numerical data or statistics.
""".strip(),
						"args": {},
					},
					{
						"name": "tangential_remark_casual_slang",
						"desc": """
Perform the following actions in the upcoming reasoning step:
* Make a completely unrelated, random remark.
* Use informal, slang-heavy language.
""".strip(),
						"args": {},
					},
					{
						"name": "finish",
						"desc": FINISH_TOOL.desc,
						"args": {},
					},
				],
				id="edge_case_dimension_order_reversed",
			),
		],
	)
	def test_initialization(
		self,
		temp_action_space_styles: Path,
		temp_action_space_structures: Path,
		signature: type[ReasoningSignature],
		max_reasoning_steps: int,
		action_space_keys: list[str] | None,
		early_stopping_enabled: bool,
		expected_num_tools: int,
		expected_has_finish: bool,
		expected_tools_details: list[dict[str, Any]],
	) -> None:
		"""Test TreeOfThoughtsControllerReranker initialization with various configurations.

		Validates that the controller correctly creates tools from action spaces:
		- For reranker, each tool is a complete choice combination (no params)
		- Tool count = Cartesian product of all dimensions + optional finish tool
		- Checks tool names, descriptions, and arguments against expected details.
		"""
		# Convert action_space_keys to actual paths
		action_space_paths = None
		if action_space_keys is not None:
			action_space_paths = []
			for key in action_space_keys:
				if key == "styles":
					action_space_paths.append(temp_action_space_styles)
				elif key == "structures":
					action_space_paths.append(temp_action_space_structures)

		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,
			signature=signature,
			max_reasoning_steps=max_reasoning_steps,
			action_space_paths=action_space_paths,
			early_stopping_enabled=early_stopping_enabled,
		)

		# Verify tool count
		assert len(controller.tools) == expected_num_tools, (
			f"Expected {expected_num_tools} tools, got {len(controller.tools)}. "
			f"Tools: {list(controller.tools.keys())}"
		)

		# Verify finish tool presence
		has_finish = "finish" in controller.tools
		assert has_finish == expected_has_finish, (
			f"Expected has_finish={expected_has_finish}, got {has_finish}"
		)

		# Verify detailed tool properties (EXACT match, not subset)
		expected_tool_names_in_order = [t["name"] for t in expected_tools_details]
		assert list(controller.tools.keys()) == expected_tool_names_in_order, (
			"Tool ordering mismatch.\n"
			f"Expected: {expected_tool_names_in_order}\n"
			f"Actual: {list(controller.tools.keys())}"
		)
		assert len(controller.tools) == len(expected_tools_details), (
			f"Expected {len(expected_tools_details)} tools, got {len(controller.tools)}."
		)

		for expected_tool in expected_tools_details:
			tool_name = expected_tool["name"]
			assert tool_name in controller.tools, f"Tool {tool_name} not found in controller.tools"
			actual_tool = controller.tools[tool_name]

			# Check full description match
			assert actual_tool.desc == expected_tool["desc"], (
				f"Tool {tool_name} description mismatch.\n"
				f"Expected:\n{repr(expected_tool['desc'])}\n"
				f"Actual:\n{repr(actual_tool.desc)}"
			)

			# Check arguments are exactly as expected (usually empty for reranker)
			assert actual_tool.args == expected_tool.get("args", {}), (
				f"Tool {tool_name} args mismatch.\n"
				f"Expected: {expected_tool.get('args', {})}\n"
				f"Actual: {actual_tool.args}"
			)

	@pytest.mark.parametrize(
		[
			"action_space_keys",
			"expected_tool_names",
		],
		[
			pytest.param(
				["styles"],
				{
					"figurative_language",
					"statistical_data_driven",
					"casual_slang",
					"finish",
				},
				id="styles_tool_names",
			),
			pytest.param(
				["structures"],
				{
					"causal_reasoning",
					"contrast",
					"tangential_remark",
					"finish",
				},
				id="structures_tool_names",
			),
			pytest.param(
				["styles", "structures"],
				{
					"figurative_language_causal_reasoning",
					"figurative_language_contrast",
					"figurative_language_tangential_remark",
					"statistical_data_driven_causal_reasoning",
					"statistical_data_driven_contrast",
					"statistical_data_driven_tangential_remark",
					"casual_slang_causal_reasoning",
					"casual_slang_contrast",
					"casual_slang_tangential_remark",
					"finish",
				},
				id="combined_tool_names",
			),
		],
	)
	def test_tool_names_from_action_spaces(
		self,
		temp_action_space_styles: Path,
		temp_action_space_structures: Path,
		action_space_keys: list[str],
		expected_tool_names: set[str],
	) -> None:
		"""Verify that tool names are correctly generated from action space choices.

		Tool names are underscore-joined, sanitized choice names.

		Args:
			temp_action_space_styles: Fixture providing styles JSON path.
			temp_action_space_structures: Fixture providing structures JSON path.
			action_space_keys: Names of action space fixtures to use.
			expected_tool_names: Expected set of tool names.
		"""
		# Convert keys to paths
		action_space_paths = []
		for key in action_space_keys:
			if key == "styles":
				action_space_paths.append(temp_action_space_styles)
			elif key == "structures":
				action_space_paths.append(temp_action_space_structures)

		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,
			action_space_paths=action_space_paths,
			early_stopping_enabled=True,
		)

		actual_tool_names = set(controller.tools.keys())
		assert actual_tool_names == expected_tool_names, (
			"Tool set mismatch.\n"
			f"Expected: {expected_tool_names}\n"
			f"Actual: {actual_tool_names}"
		)

	def test_tools_have_no_arguments(
		self,
		temp_action_space_styles: Path,
		temp_action_space_structures: Path,
	) -> None:
		"""Verify that reranker tools take no arguments.

		Unlike generative controller tools that accept parameters,
		reranker tools represent complete choice combinations.
		"""
		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,
			action_space_paths=[temp_action_space_styles, temp_action_space_structures],
			early_stopping_enabled=True,
		)

		# All action_metadata should have empty argument dicts
		for tool_name, arguments in controller.action_metadata:
			assert arguments == {}, (
				f"Tool {tool_name} should have empty arguments, got {arguments}"
			)


class TestControllerRerankerForward:
	"""Comprehensive forward method tests for TreeOfThoughtsControllerReranker."""

	@pytest.mark.parametrize(
		[
			"state_question",
			"state_reasoning",
			"action_space_keys",
			"mock_scores",
			"n_samples_generation",
			"expected_top_action",
		],
		[
			pytest.param(
				"What is 2+2?",					# State question
				[],								# Empty reasoning list
				None,							# Default tools (continue + finish)
				[0.8, 0.2],						# continue=0.8, finish=0.2
				1,								# Number of samples to generate
				"continue_reasoning",			# Expected top action
				id="continue_higher_score",
			),
			pytest.param(
				"Calculate the final answer.",	# State question
				["Step 1", "Step 2", "Step 3"],	# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.3, 0.9],						# continue=0.3, finish=0.9
				1,								# Number of samples to generate
				"finish",						# Expected top action
				id="finish_higher_score",
			),
			pytest.param(
				"Explain climate change.",		# State question
				["Initial thought"],			# Reasoning steps
				["styles"],						# 3 styles + finish = 4 tools
				[0.9, 0.5, 0.4, 0.3],			# figurative=0.9, statistical=0.5, slang=0.4, finish=0.3
				1,								# Number of samples to generate
				"figurative_language",			# Expected top action
				id="style_tool_highest",
			),
			pytest.param(
				"Analyze the data.",			# State question
				["Looking at numbers"],			# Reasoning steps
				["styles"],						# 3 styles + finish = 4 tools
				[0.2, 0.8, 0.4, 0.6],			# figurative=0.2, statistical=0.8, slang=0.4, finish=0.6
				1,								# Number of samples to generate
				"statistical_data_driven",		# Expected top action
				id="statistical_style_highest",
			),
			pytest.param(
				"Build an argument.",			# State question
				["Thesis statement"],			# Reasoning steps
				["styles", "structures"],		# 3x3=9 combinations + finish = 10 tools
				[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.2], # 9th combo highest before finish
				1,								# Number of samples to generate
				"casual_slang_tangential_remark", # 9th combo (index 8) receives high score
				id="combined_dimension_highest",
			),
			# Edge cases: Empty reasoning
			pytest.param(
				"Start fresh.",					# State question
				[],								# Empty reasoning list
				None,							# Default tools (continue + finish)
				[0.6, 0.4],						# continue=0.6, finish=0.4
				1,								# Number of samples to generate
				"continue_reasoning",			# Expected top action
				id="edge_case_empty_reasoning",
			),
			# Edge cases: Very close scores (tie-breaking)
			pytest.param(
				"Close call.",					# State question
				["Thinking"],					# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.501, 0.499],					# Very close scores
				1,								# Number of samples to generate
				"continue_reasoning",			# Expected top action
				id="edge_case_very_close_scores",
			),
			# Edge cases: Equal scores (first wins in sort)
			pytest.param(
				"Tied scores.",					# State question
				["Even"],						# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.5, 0.5],						# Exactly equal
				1,								# Number of samples to generate
				"continue_reasoning",			# Expected top action
				id="edge_case_equal_scores",
			),
			# Edge cases: Very long reasoning
			pytest.param(
				"Complex problem.",				# State question
				[f"Step {i}" for i in range(20)],	# 20 reasoning steps
				None,							# Default tools (continue + finish)
				[0.2, 0.95],					# High finish score
				1,								# Number of samples to generate
				"finish",						# Expected top action
				id="edge_case_long_reasoning_chain",
			),
			# Edge cases: Score extremes
			pytest.param(
				"Extreme scores.",				# State question
				["Test"],						# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.0, 1.0],						# Min and max scores
				1,								# Number of samples to generate
				"finish",						# Expected top action
				id="edge_case_extreme_scores_zero_and_one",
			),
			# Edge cases: All very low scores
			pytest.param(
				"Low confidence.",				# State question
				["Uncertain"],					# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.01, 0.02],					# All scores very low
				1,								# Number of samples to generate
				"finish",						# Expected top action
				id="edge_case_all_low_scores",
			),
			# Edge cases: Special characters in input
			pytest.param(
				"What about émojis 🎉 and spëcial chars?",		# State question
				["Step with 'quotes' and \"double quotes\""],	# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.7, 0.3],						# Scores for continue and finish
				1,								# Number of samples to generate
				"continue_reasoning",			# Expected top action
				id="edge_case_special_characters",
			),
			# Edge cases: Very long input question
			pytest.param(
				"This is a very long question " * 50,	# ~250 words
				["Processing"],						# Reasoning steps
				None,							# Default tools (continue + finish)
				[0.6, 0.4],						# Scores for continue and finish
				1,								# Number of samples to generate
				"continue_reasoning",			# Expected top action
				id="edge_case_very_long_input",
			),
		],
	)
	def test_forward_selects_highest_scoring_action(
		self,
		temp_action_space_styles: Path,
		temp_action_space_structures: Path,
		state_question: str,
		state_reasoning: list[str],
		action_space_keys: list[str] | None,
		mock_scores: list[float],
		n_samples_generation: int,
		expected_top_action: str,
	) -> None:
		"""Test that forward method selects the highest-scoring action.

		Args:
			temp_action_space_styles: Fixture providing styles JSON path.
			temp_action_space_structures: Fixture providing structures JSON path.
			state_question: Input question for the state.
			state_reasoning: Reasoning steps in the state.
			action_space_keys: Names of action space fixtures to use.
			mock_scores: Mock scores for each action (ordered by tool creation).
			n_samples_generation: Number of samples to generate.
			expected_top_action: Expected top-scoring action name.
		"""
		# Convert keys to paths
		action_space_paths = None
		if action_space_keys is not None:
			action_space_paths = []
			for key in action_space_keys:
				if key == "styles":
					action_space_paths.append(temp_action_space_styles)
				elif key == "structures":
					action_space_paths.append(temp_action_space_structures)

		# Create mock rerank responses in vLLM format: layers[scores]
		rerank_responses = [mock_scores]

		# Contract: if no actions remain, the controller should not be called. We enforce this
		# by ensuring `max_reasoning_steps` always exceeds the current reasoning length in tests.
		max_reasoning_steps = len(state_reasoning) + 1
		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=rerank_responses,
			action_space_paths=action_space_paths,
			max_reasoning_steps=max_reasoning_steps,
			early_stopping_enabled=True,
		)

		state = create_state(
			question=state_question,
			reasoning_steps=state_reasoning,
		)

		result = controller.forward(
			states=state,
			n_samples_generation=n_samples_generation,
		)

		# Verify result structure
		assert len(result) == 1, "Should have 1 state output"
		assert len(result[0]) >= 1, "Should have at least 1 prediction"

		# Verify top action
		top_prediction = result[0][0]
		assert isinstance(top_prediction, ControllerPrediction)
		assert top_prediction.tool.name == expected_top_action, (
			f"Expected top action {expected_top_action}, got {top_prediction.tool.name}"
		)

	@pytest.mark.parametrize(
		[
			"n_samples_generation",
			"mock_scores",
			"expected_num_predictions",
		],
		[
			pytest.param(
				1,					# Number of samples to generate
				[0.8, 0.2],			# Mock scores for each action
				1,					# Expected number of predictions
				id="single_sample",
			),
			pytest.param(
				2,					# Number of samples to generate
				[0.8, 0.2],			# Mock scores for each action
				2,					# Expected number of predictions
				id="two_samples",
			),
			pytest.param(
				5,					# Number of samples to generate
				[0.8, 0.2],			# Only 2 tools
				2,					# Capped at number of tools
				id="samples_capped_at_tool_count",
			),
			# Edge cases: n_samples edge values
			pytest.param(
				10,					# Much larger than tool count
				[0.8, 0.2],			# Mock scores for each action
				2,					# Expected number of predictions
				id="edge_case_n_samples_much_larger",
			),
			pytest.param(
				3,					# Exactly one more than tools
				[0.8, 0.2],			# Mock scores for each action
				2,					# Expected number of predictions
				id="edge_case_n_samples_one_more_than_tools",
			),
		],
	)
	def test_forward_returns_n_samples(
		self,
		n_samples_generation: int,
		mock_scores: list[float],
		expected_num_predictions: int,
	) -> None:
		"""Test that forward returns up to n_samples_generation predictions.

		Args:
			n_samples_generation: Number of samples requested.
			mock_scores: Mock scores for each action.
			expected_num_predictions: Expected number of predictions returned.
		"""
		rerank_responses = [mock_scores]

		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=rerank_responses,
			action_space_paths=None,
			early_stopping_enabled=True,
		)

		state = create_state()

		result = controller.forward(
			states=state,
			n_samples_generation=n_samples_generation,
		)

		assert len(result) == 1
		assert len(result[0]) == expected_num_predictions, (
			f"Expected {expected_num_predictions} predictions, got {len(result[0])}"
		)


class TestControllerRerankerBatchProcessing:
	"""Test batch processing of multiple states."""

	@pytest.mark.parametrize(
		[
			"num_states",
			"mock_scores_per_state",
		],
		[
			pytest.param(
				2,
				[[0.8, 0.2], [0.3, 0.9]],
				id="two_states_different_outcomes",
			),
			pytest.param(
				3,
				[[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]],
				id="three_states_varied_scores",
			),
			# Edge cases
			pytest.param(
				1,									# Single state in a list
				[[0.7, 0.3]],
				id="edge_case_single_state_in_list",
			),
			pytest.param(
				5,									# Larger batch
				[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
				id="edge_case_five_states",
			),
			pytest.param(
				2,
				[[0.5, 0.5], [0.5, 0.5]],			# Identical scores for all states
				id="edge_case_identical_scores_all_states",
			),
		],
	)
	def test_batch_forward(
		self,
		num_states: int,
		mock_scores_per_state: list[list[float]],
	) -> None:
		"""Test forward with multiple states processes each correctly.

		Args:
			num_states: Number of states to process.
			mock_scores_per_state: Mock scores for each state.
		"""
		# For batch processing, each state needs its own scoring layer
		# The MockScoringLocalVLLM expects list[list[float]] where each inner list
		# is scores for one call to score()
		rerank_responses = mock_scores_per_state  # Each state gets its own layer

		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=rerank_responses,
			action_space_paths=None,
			early_stopping_enabled=True,
		)

		states = [
			create_state(question=f"Question {i}")
			for i in range(num_states)
		]

		result = controller.forward(states=states, n_samples_generation=1)

		assert len(result) == num_states, (
			f"Expected {num_states} state outputs, got {len(result)}"
		)

		for state_result in result:
			assert len(state_result) >= 1
			assert isinstance(state_result[0], ControllerPrediction)


class TestForcedChoiceScenarios:
	"""Test forced choice functionality in the reranker controller."""

	@pytest.mark.parametrize(
		[
			"available_tools",
			"expected_action",
		],
		[
			pytest.param(
				{"continue_reasoning": DEFAULT_TOOL},
				"continue_reasoning",
				id="forced_continue_single_option",
			),
			pytest.param(
				{"finish": FINISH_TOOL},
				"finish",
				id="forced_finish_single_option",
			),
			pytest.param(
				{
					"continue_reasoning": DEFAULT_TOOL,
					"finish": FINISH_TOOL,
				},
				None,	# No forced choice when multiple options
				id="no_forced_choice_multiple_options",
			),
		],
	)
	def test_return_action_if_single_option(
		self,
		available_tools: dict[str, dspy.Tool],
		expected_action: str | None,
	) -> None:
		"""Test the default forced choice function.

		Args:
			available_tools: Dictionary of available tools.
			expected_action: Expected action or None if no forced choice.
		"""
		state = create_state()

		result = return_action_if_single_option(available_tools, state)

		if expected_action is None:
			assert result is None
		else:
			assert result is not None
			assert len(result) >= 1
			action_name, arguments, considerations = result[0]
			assert action_name == expected_action
			assert arguments == {}
			assert isinstance(considerations, str)

	def test_forced_choice_bypasses_scoring(
		self,
		temp_action_space_styles: Path,
	) -> None:
		"""Verify that forced choice bypasses the scoring model entirely.

		When only one tool is available, the forced choice function should
		return immediately without calling the reranker.
		"""
		# Create controller but don't provide rerank_responses
		# If scoring were called, it would fail
		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,		# Would fail if scoring is called
			action_space_paths=None,
			early_stopping_enabled=False,	# Only DEFAULT_TOOL (continue)
		)

		# With only continue tool, forced choice should trigger
		state = create_state()

		result = controller.forward(states=state, n_samples_generation=1)

		assert len(result) == 1
		assert len(result[0]) == 1
		assert result[0][0].tool.name == "continue_reasoning"


class TestScoreAggregation:
	"""Test score aggregation functionality."""

	def test_scores_sorted_descending(self) -> None:
		"""Verify that action scores are sorted in descending order."""
		mock_scores = [0.3, 0.9, 0.5]	# Unsorted
		rerank_responses = [mock_scores]

		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=rerank_responses,
			action_space_paths=None,
			early_stopping_enabled=True,
		)

		# Manually add an extra tool to have 3 options
		def dummy_tool_func() -> ReasoningIntervention:
			return ReasoningIntervention(continue_reasoning=True)

		controller.tools["extra_tool"] = dspy.Tool(
			name="extra_tool", func=dummy_tool_func, desc="Extra tool"
		)
		controller._enumerate_all_action_candidates()	# Re-enumerate

		state = create_state()
		action_scores = controller._score_actions(state, controller.get_lm())

		# Verify sorted descending by score
		scores_only = [score for _, _, score in action_scores]
		assert scores_only == sorted(scores_only, reverse=True), (
			f"Scores should be sorted descending, got: {scores_only}"
		)



class TestErrorHandling:
	"""Test error handling scenarios."""

	def test_empty_tools_raises_error(self) -> None:
		"""Verify ValueError when no action candidates are available."""
		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,
			action_space_paths=None,
			early_stopping_enabled=True,
		)

		# Empty the tools dict
		controller.tools = {}
		controller._enumerate_all_action_candidates()

		state = create_state()

		with pytest.raises(ValueError, match="No action candidates to score"):
			controller._score_actions(state, controller.get_lm())

	def test_lm_not_set_raises_error(self) -> None:
		"""Verify error when LM is not set before forward."""
		controller = TreeOfThoughtsControllerReranker(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=5,
		)
		# Don't call set_lm()

		state = create_state()

		with pytest.raises(ValueError, match="Language model has not been set"):
			controller(states=state)


class TestToolExecution:
	"""Test tool execution and intervention creation."""

	def test_action_space_tool_returns_intervention(
		self,
		temp_action_space_styles: Path,
	) -> None:
		"""Verify that action space tools return correct ReasoningIntervention.

		For reranker, tools are pre-built with fixed interventions.
		"""
		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,
			action_space_paths=[temp_action_space_styles],
			early_stopping_enabled=True,
		)

		# Get a style tool and execute it
		figurative_tool = controller.tools.get("figurative_language")
		assert figurative_tool is not None

		# Execute tool (takes no arguments)
		intervention = figurative_tool.func()

		assert isinstance(intervention, ReasoningIntervention)
		assert intervention.continue_reasoning is True
		assert "non-literal comparison" in intervention.internal_reasoning

	def test_finish_tool_returns_false_continue(self) -> None:
		"""Verify that FINISH tool returns continue_reasoning=False."""
		controller = create_controller_reranker_with_mocked_lm(
			rerank_responses=None,
			action_space_paths=None,
			early_stopping_enabled=True,
		)

		finish_tool = controller.tools["finish"]
		intervention = finish_tool.func()

		assert isinstance(intervention, ReasoningIntervention)
		assert intervention.continue_reasoning is False

	def test_finish_tool_default_description(self) -> None:
		"""Verify that finish tool uses default description when not customized."""
		controller = TreeOfThoughtsControllerReranker(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=5,
			action_space_paths=None,
			early_stopping_enabled=True,
			finish_tool_description=None,
		)

		finish_tool = controller.tools["finish"]
		expected_desc = (
			"Signals that the reasoning so far is sufficient for producing a "
			"high-quality response for the task.\n"
			"If selected, the next step will involve generating the final output "
			"rather than reasoning further."
		)
		assert finish_tool.desc == expected_desc

	def test_finish_tool_custom_description(self) -> None:
		"""Verify that finish tool uses custom description when provided."""
		custom_desc = (
			"Only finish if you have verified all constraints. "
			"Requires at least 3 reasoning steps."
		)
		controller = TreeOfThoughtsControllerReranker(
			signature=QuestionAnsweringWithReasoning,
			max_reasoning_steps=5,
			action_space_paths=None,
			early_stopping_enabled=True,
			finish_tool_description=custom_desc,
		)

		finish_tool = controller.tools["finish"]
		assert finish_tool.desc == custom_desc
		# Verify it still returns the correct intervention
		intervention = finish_tool.func()
		assert isinstance(intervention, ReasoningIntervention)
		assert intervention.continue_reasoning is False


# =============================================================================
# Semantic Integration Tests (requires GPU)
# =============================================================================


@pytest.fixture(scope="module")
def scoring_lm(reranker_model_config: dict) -> Generator[ScoringLocalVLLM, Any, Any]:
	"""Create real ScoringLocalVLLM for integration tests.

	This fixture is only used when GPU is available.
	"""
	try:
		lm = ScoringLocalVLLM(**reranker_model_config)
		yield lm
	except Exception as e:
		pytest.skip(f"Failed to initialize scoring model (skipping integration test): {e}")

class TestControllerRerankerSemanticIntegration:
	"""Semantic integration tests verifying reranker preferences match human common sense.

	These tests validate that the reranker model scores actions appropriately:
	- Statistical reasoning should favor statistical/data-driven tools
	- Complete arguments should favor finish over continue
	- Incomplete arguments should favor continue over finish

	Note: These tests require a GPU and will be skipped on CPU-only systems.
	"""

	@pytest.mark.parametrize(
		"state_input, state_reasoning, action_space_keys, expected_choice_to_make, expected_choice_to_avoid, rationale",
		[
			# Early stopping: Complete argument should favor FINISH
			pytest.param(
				{							# State input
					"topic": "renewable energy is worth significant government investment",
                    "stance": "PRO",
				},
				[							# State reasoning
					"Renewable energy reduces carbon emissions.",
					"Studies show solar costs dropped 89% since 2010.",
					"Wind power now employs 1.2 million workers globally.",
					"Therefore, switching to renewables is both an environmental and economic necessity.",
				],
				None,  						# Default continue vs finish only
				(
					"finish"
				),
				(							# Explicitly avoid CONTINUE
					"continue_reasoning"
				),
				"Complete argument with claim, evidence, and conclusion should finish",
				id="complete_argument_should_finish",
			),
			# Early stopping: Incomplete argument should favor CONTINUE
			pytest.param(
				{							# State input
					"topic": "We should prioritize investments in education at the expense of increased taxes",
                    "stance": "PRO",
				},
				[ 							# State reasoning
					"I should think about this more before giving an answer -- I want to find some sources to back up the claims I want to make.",
				],
				None,						# Use default tools
				"continue_reasoning",		# Expected: Continue (incomplete thought)
				"finish",					# Avoid: Finish (premature)
				"Reasoning is missing details, should continue",
				id="incomplete_argument_should_continue",
			),
			# Style preference: Statistical content should favor statistical style
			pytest.param(
				{							# State input
					"topic": "Advanced calculus should be a mandatory requirement for high school graduation.",
                    "stance": "PRO",
				},
				[							# State reasoning
					"Let's dive into the statistics about the job prospects of high schooler's who took advanced calculus.",
				],
				["styles"],  				# Style choices only
				"statistical_data_driven",	# Expected choice to make
				"figurative_language",		# Expected choice to avoid
				"Math context requires data/stats style over figurative",
				id="statistical_context_prefers_statistical_style",
			),
			# Style preference: Figurative content should vary style
			pytest.param(
				{ 							# State input
					"topic": "Public companies must be required to undergo rigorous annual financial audits.",
                    "stance": "PRO",
				},
				[ 							# State reasoning
					"The audit requires precise accounting of every cent. Let's look at the raw numbers.",
				],
				["styles"],					# Style choices only
				"statistical_data_driven",  # Expected choice to make
				"figurative_language",      # Expected choice to avoid
				"Audit context explicitly demands numbers over metaphors",
				id="figurative_context_transitions_to_statistical",
			),
			# Structure preference: Claim needs evidence
			pytest.param(
				{ 							# State input
					"topic": "We should reduce plastic usage in our country",
                    "stance": "PRO",
				},
				[
					"Some people say plastic is a necessary evil -- that it is too convenient and cost effective to give up.",
				],
				["structures"],				# Structure choices only
				"contrast", 				# Expected choice to use `contrast` ("however") as next prefix
				"causal_reasoning", 		# Expected choice to avoid `causal reasoning` ("therefore") as next prefix
				"We expect the next prefix to be contrast since the first claim opposes the specified stance",
				id="claim_needs_supporting_structure",
			),
			# Structure preference: Clear cut Rebuttal (However)
			# "Note that" preamble with misleading info needs contrast
			pytest.param(
				{ 							# State input
					"topic": "The earth is flat.",
                    "stance": "ANTI",
				},
				[ 							# State reasoning
					"Note that from a local perspective on the ground, the horizon appears to be a flat line.",
				],
				["structures"],				# Structure choices only
				"contrast",  				# Must pivot to reality (Round)
				"causal_reasoning", 		# "Therefore it is flat" would be wrong
				"Misleading 'Note that' premise requires 'However' transition to correct it",
				id="structure_preference_contrast_for_rebuttal",
			),
			# Structure preference: Physical Causal Chain (Therefore)
			# "Cause" preamble needs "Effect" conclusion
			pytest.param(
				{ 							# State input
					"topic": "The laws of physics dictate that unsupported objects must fall.",
                    "stance": "PRO",
				},
				[ 							# State reasoning
					"For instance, if I let go of the ball from the roof,",
				],
				["structures"],				# Structure choices only
				"causal_reasoning", 		# "Therefore it falls"
				"Tangential Remark", 		# "Speaking of bananas..." (Totally irrelevant, should be ranked last)
				"Physical action (drop) requires 'Therefore' effect, not random tangent",
				id="structure_preference_causal_for_physical_effect",
			),
		],
	)
	def test_semantic_validation(
		self,
		scoring_lm,
		temp_action_space_styles: Path,
		temp_action_space_structures: Path,
		state_input: dict[str, str],
		state_reasoning: list[str],
		action_space_keys: list[str] | None,
		expected_choice_to_make: str | None,
		expected_choice_to_avoid: str | None,
		rationale: str,
	) -> None:
		"""Validate that reranker preferences align with human common sense.

		These tests verify the reranker makes semantically appropriate decisions
		based on the reasoning context by comparing scores of specific actions.

		Args:
			scoring_lm: Real ScoringLocalVLLM fixture.
			temp_action_space_styles: Styles JSON fixture.
			temp_action_space_structures: Structures JSON fixture.
			state_input: Input dictionary for the state.
			state_reasoning: List of reasoning steps.
			action_space_keys: Which action spaces to use.
			expected_choice_to_make: Action name expected to have higher score.
			expected_choice_to_avoid: Action name expected to have lower score.
			rationale: Why this preference is semantically correct.
		"""
		# Convert action_space_keys to paths
		action_space_paths = None
		if action_space_keys is not None:
			action_space_paths = []
			for key in action_space_keys:
				if key == "styles":
					action_space_paths.append(temp_action_space_styles)
				elif key == "structures":
					action_space_paths.append(temp_action_space_structures)

		# Create controller with real LM (ArgumentGeneration signature)
		controller = TreeOfThoughtsControllerReranker(
			signature=GenerateArgumentWithReasoning,
			max_reasoning_steps=5,
			action_space_paths=action_space_paths,
			early_stopping_enabled=True,
		)
		controller.set_lm(scoring_lm)

		# Create state
		reasoning_field_name = list(GenerateArgumentWithReasoning.reasoning_fields.keys())[0]
		reasoning_dicts = [
			{reasoning_field_name: step} for step in state_reasoning
		]
		state = State(
			input=state_input,
			reasoning=reasoning_dicts,
			controller_output_trajectory=[create_dummy_controller_output() for _ in state_reasoning],
		)

		# Get top 20 predictions (should cover all tools in small action spaces)
		# We use forward() directly as requested
		n_samples = 20
		prediction_lists = controller.forward(states=state, n_samples_generation=n_samples)

		assert len(prediction_lists) == 1
		predictions = prediction_lists[0]

		# Extract ordered tool names from predictions
		ranked_tools = [p.tool.name for p in predictions]

		# Log ranks for debugging
		logger.info(f"\n{'='*60}")
		logger.info(f"Test: {rationale}")
		logger.info(f"Input: {state_input}")
		logger.info(f"Ranked tools: {ranked_tools}")

		# Validation Logic
		# 1. If we have both make and avoid, assert make is ranked higher (lower index) than avoid
		if expected_choice_to_make and expected_choice_to_avoid:
			if expected_choice_to_make in ranked_tools and expected_choice_to_avoid in ranked_tools:
				rank_make = ranked_tools.index(expected_choice_to_make)
				rank_avoid = ranked_tools.index(expected_choice_to_avoid)

				logger.info(f"Comparing ranks: {expected_choice_to_make} (Rank #{rank_make}, Score {predictions[rank_make].score}) vs {expected_choice_to_avoid} (Rank #{rank_avoid}, Score {predictions[rank_avoid].score})")

				assert rank_make < rank_avoid, (
					f"Semantic validation failed: {rationale}\n"
					f"Expected '{expected_choice_to_make}' to be ranked higher than '{expected_choice_to_avoid}'\n"
					f"Actual ranks: #{rank_make} vs #{rank_avoid}\n"
					f"Scores: {expected_choice_to_make}={predictions[rank_make].score:.4f}, {expected_choice_to_avoid}={predictions[rank_avoid].score:.4f}"
				)
			elif expected_choice_to_make in ranked_tools:
				# Make is present, Avoid is not (meaning Avoid was ranked so low it dropped off, or didn't exist)
				# This is a pass (Make >> Avoid)
				logger.info(f"Pass: {expected_choice_to_make} found at #{ranked_tools.index(expected_choice_to_make)}, {expected_choice_to_avoid} not in top {n_samples}")
			elif expected_choice_to_avoid in ranked_tools:
				# Avoid is present, Make is not -> Fail
				raise AssertionError(
					f"Semantic validation failed: {rationale}\n"
					f"Expected '{expected_choice_to_make}' > '{expected_choice_to_avoid}'\n"
					f"Actual: '{expected_choice_to_avoid}' found at #{ranked_tools.index(expected_choice_to_avoid)}, '{expected_choice_to_make}' not in top {n_samples}"
				)
			else:
				# Neither found? This implies N is too small or tools are broken
				raise AssertionError(f"Neither expected choice found in top {n_samples} tools: {ranked_tools}")

		elif expected_choice_to_make:
			# Only expected make provided
			if expected_choice_to_make in ranked_tools:
				rank = ranked_tools.index(expected_choice_to_make)
				logger.info(f"Pass: {expected_choice_to_make} found at #{rank}")
			else:
				logger.warning(
					f"Warning for '{rationale}': "
					f"Expected '{expected_choice_to_make}' not found in top {n_samples}."
				)

		logger.info(f"✓ Passed: {rationale}")


if __name__ == "__main__":
	gpu_available = torch.cuda.is_available()
	if not gpu_available:
		pytest.main([__file__, "-vv"])
	else:
		pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
