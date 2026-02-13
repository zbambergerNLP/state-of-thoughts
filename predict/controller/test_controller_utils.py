"""
Tests for the controller_utils module.

Expected usage:
```bash
pytest predict/test_controller_utils.py -vv
```
"""

# Standard library imports
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Third-party imports
import dspy
import pytest

# Local imports
from predict.controller.controller_constants import ReasoningIntervention
from predict.controller.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	ActionSpaceConfig,
	create_literal_from_dict,
	create_reasoning_intervention_from_choices,
	execute_tool_safely,
	load_action_space_json,
	remove_duplicate_actions_with_counts,
	return_action_if_single_option,
	sanitize_param_name,
)
from signatures import QuestionAnsweringWithReasoning
from tree import State

# =============================================================================
# Test Fixtures for Mock Action Space JSONs
# =============================================================================


@pytest.fixture
def temp_action_space_styles(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for style action space with 3 options.

	Args:
		tmp_path: Pytest fixture providing temporary directory path.

	Returns:
		Path to the created temporary JSON file.
	"""
	styles_json = {
		"name": "style",
		"definition": (
			"Force the next reasoning step to adopt a specific rhetorical style."
		),
		"choices": {
			"Figurative Language": {
				"definition": "Use metaphor, simile, or analogy.",
				"internal_reasoning": (
					"I should employ non-literal comparison. "
				),
				"prefix": "Imagine this:",
			},
			"Statistical & Data-Driven": {
				"definition": "Present numerical data or statistics.",
				"internal_reasoning": (
					"I should use numbers and data. "
				),
				"prefix": "The data shows:",
			},
			"Formal & Academic": {
				"definition": "Use formal, scholarly language.",
				"internal_reasoning": (
					"I should use academic tone. "
				),
				"prefix": "It can be argued that",
			},
		},
	}
	styles_file = tmp_path / "styles.json"
	styles_file.write_text(json.dumps(styles_json))
	return styles_file


@pytest.fixture
def temp_action_space_structures(tmp_path: Path) -> Path:
	"""Create a temporary JSON file for structure action space with 2 options.

	Args:
		tmp_path: Pytest fixture providing temporary directory path.

	Returns:
		Path to the created temporary JSON file.
	"""
	structures_json = {
		"name": "structure",
		"definition": (
			"Control the argumentative structure of the next reasoning step."
		),
		"choices": {
			"Cause & Effect": {
				"definition": "Explain causal relationships.",
				"internal_reasoning": (
					"I should explain cause and effect. "
				),
				"prefix": "As a result,",
			},
			"Contrast": {
				"definition": "Present contrasting viewpoints.",
				"internal_reasoning": (
					"I should present a contrasting view. "
				),
				"prefix": "On the other hand,",
			},
		},
	}
	structures_file = tmp_path / "structures.json"
	structures_file.write_text(json.dumps(structures_json))
	return structures_file


# =============================================================================
# Test Classes
# =============================================================================


class TestReturnActionIfSingleOption:
	"""Test cases for return_action_if_single_option function."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"available_tools",
			"expected_action",
			"expected_arguments",
			"expected_considerations_contains",
		],
		# Parameter values
		[
			pytest.param(
				{							# available_tools
					"continue_reasoning": DEFAULT_TOOL,
				},
				(							# expected_action
					"continue_reasoning"
				),
				{},							# expected_arguments
				"only available action",	# expected_considerations_contains
				id="forced_continue_reasoning",
			),
			pytest.param(
				{							# available_tools
					"finish": FINISH_TOOL,
				},
				(							# expected_action
					"finish"
				),
				{},							# expected_arguments
				"only available action",	# expected_considerations_contains
				id="forced_finish",
			),
			pytest.param(
				{							# available_tools
					"continue_reasoning": DEFAULT_TOOL,
					"finish": FINISH_TOOL,
				},
				None,						# expected_action
				None,						# expected_arguments
				None,						# expected_considerations_contains
				id="no_forced_choice_multiple_tools",
			),
		],
	)
	def test_return_action_if_single_option(
		self,
		available_tools: dict[str, dspy.Tool],
		expected_action: str | None,
		expected_arguments: dict[str, Any] | None,
		expected_considerations_contains: str | None,
	) -> None:
		"""Test return_action_if_single_option function with various tool configurations.

		Validates that the function correctly identifies when only one tool is available
		and returns the appropriate forced action, or returns None when multiple tools exist.

		Args:
			available_tools: Dictionary mapping action names to Tool objects.
			expected_action: Expected action name if forced, None if not forced.
			expected_arguments: Expected arguments dict for the action.
			expected_considerations_contains: String that should appear in considerations.
		"""
		# Create state directly in the test
		reasoning_field_name = list(QuestionAnsweringWithReasoning.reasoning_fields.keys())[0]
		state = State(
			input={"question": "What is 2+2?"},
			reasoning=[
					{reasoning_field_name: "First, I need to think about addition."}
				]
		)

		result = return_action_if_single_option(available_tools, state)

		if expected_action is None:
			assert result is None, "Should return None when multiple tools available"
		else:
			assert result is not None, "Should return action when only one tool available"
			assert isinstance(result, list), "Result should be a list"
			assert len(result) == 1, "Result should contain exactly one action"
			action_name, action_arguments, considerations = result[0]
			assert action_name == expected_action, f"Expected action {expected_action}, got {action_name}"
			assert action_arguments == expected_arguments, f"Expected arguments {expected_arguments}"
			if expected_considerations_contains is not None:
				assert expected_considerations_contains.lower() in considerations.lower(), \
					f"Expected '{expected_considerations_contains}' in considerations"


class TestExecuteToolSafely:
	"""Test cases for execute_tool_safely function."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"tool_func_logic",
			"tool_args",
			"expected_continue",
			"expected_error_contains",
		],
		# Parameter values
		[
			pytest.param(
				# tool_func_logic
				lambda **kwargs: {
					"continue_reasoning": True,
					"result": "success",
					"internal_reasoning": "Test reasoning",
					"prefix": "Test prefix",
				},
				{},							# tool_args
				True,						# expected_continue
				"",							# expected_error_contains
				id="success_simple",
			),
			pytest.param(
				# tool_func_logic
				lambda style, **kwargs: {
					"continue_reasoning": True,
					"style": style,
					"internal_reasoning": f"Using {style}",
					"prefix": "",
				},
				{},							# tool_args
				False,						# expected_continue
				"missing",					# expected_error_contains
				id="missing_required_arg",
			),
			pytest.param(
				# tool_func_logic
				lambda style, **kwargs: {
					"continue_reasoning": True,
					"style": style,
					"internal_reasoning": "",
					"prefix": "",
				},
				None,						# tool_args
				False,						# expected_continue
				"failed",					# expected_error_contains
				id="none_args",
			),
			pytest.param(
				# tool_func_logic
				lambda style="default", **kwargs: {
					"continue_reasoning": True,
					"style": style,
					"internal_reasoning": "",
					"prefix": "",
				},
				{"other": "value"},			# tool_args
				True,						# expected_continue
				"",							# expected_error_contains
				id="partial_args_with_default",
			),
			pytest.param(
				# tool_func_logic
				lambda **kwargs: ReasoningIntervention(continue_reasoning=True),
				{},							# tool_args
				True,						# expected_continue
				"",							# expected_error_contains
				id="returns_reasoning_intervention_object",
			),
		],
	)
	def test_execute_tool_safely(
		self,
		tool_func_logic: Callable[[dict[str, Any]], ReasoningIntervention],
		tool_args: dict[str, Any] | None,
		expected_continue: bool,
		expected_error_contains: str,
	) -> None:
		"""Test execute_tool_safely with various scenarios.

		Validates proper error handling for missing parameters, None arguments,
		and successful tool execution. Ensures that failures return appropriate
		error messages and set continue_reasoning to False.

		Args:
			tool_func_logic: Lambda function defining tool behavior.
			tool_args: Arguments to pass to the tool.
			expected_continue: Expected value of continue_reasoning in intervention.
			expected_error_contains: String expected in error message (empty if no error).
		"""
		tool = dspy.Tool(name="test_tool", func=tool_func_logic)
		intervention, error = execute_tool_safely(tool, tool_args)

		assert intervention.continue_reasoning is expected_continue, (
			f"Expected continue_reasoning={expected_continue}, "
			f"got {intervention.continue_reasoning}"
		)

		if expected_error_contains:
			assert (
				expected_error_contains.lower() in error.lower() or
				"missing" in str(error).lower() or
				"type" in str(error).lower()
			), f"Expected error containing '{expected_error_contains}', got: {error}"
		else:
			assert error == "", f"Expected no error, but got: {error}"


class TestRemoveDuplicateActionsWithCounts:
	"""Test cases for remove_duplicate_actions_with_counts function."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"input_dicts",
			"expected_combinations",
		],
		# Parameter values
		[
			pytest.param(
				# input_dicts
				[
					{
						"action": "continue_reasoning",
						"action_arguments": {},
						"considerations": "First",
					},
					{
						"action": "continue_reasoning",
						"action_arguments": {},
						"considerations": "Second",
					},
				],
				# expected_combinations: list of (tool, params, count) tuples
				[("continue_reasoning", {}, 2)],
				id="duplicate_continue_reasoning",
			),
			pytest.param(
				# input_dicts
				[
					{
						"action": "continue_reasoning",
						"action_arguments": {},
						"considerations": "First",
					},
					{
						"action": "finish",
						"action_arguments": {},
						"considerations": "Different action",
					},
				],
				# expected_combinations
				[
					("continue_reasoning", {}, 1),
					("finish", {}, 1),
				],
				id="no_duplicates",
			),
			pytest.param(
				# input_dicts
				[
					{
						"action": "select_style",
						"action_arguments": {"style": "A"},
						"considerations": "1",
					},
					{
						"action": "select_style",
						"action_arguments": {"style": "B"},
						"considerations": "2",
					},
					{
						"action": "select_style",
						"action_arguments": {"style": "A"},
						"considerations": "3",
					},
				],
				# expected_combinations
				[
					("select_style", {"style": "A"}, 2),
					("select_style", {"style": "B"}, 1),
				],
				id="mixed_duplicates",
			),
			pytest.param(
				# input_dicts
				[
					{
						"action": "select_style_structure",
						"action_arguments": {"style": "Formal", "structure": "Cause"},
						"considerations": "First combo",
					},
					{
						"action": "select_style_structure",
						"action_arguments": {"style": "Formal", "structure": "Contrast"},
						"considerations": "Different structure",
					},
					{
						"action": "select_style_structure",
						"action_arguments": {"style": "Formal", "structure": "Cause"},
						"considerations": "Duplicate combo",
					},
					{
						"action": "select_style_structure",
						"action_arguments": {"style": "Formal", "structure": "Cause"},
						"considerations": "Another duplicate",
					},
				],
				# expected_combinations
				[
					("select_style_structure", {"style": "Formal", "structure": "Cause"}, 3),
					("select_style_structure", {"style": "Formal", "structure": "Contrast"}, 1),
				],
				id="complex_tool_with_multiple_params",
			),
		],
	)
	def test_remove_duplicate_actions_with_counts(
		self,
		input_dicts: list[dict[str, Any]],
		expected_combinations: list[tuple[str, dict[str, Any], int]],
	) -> None:
		"""Test remove_duplicate_actions_with_counts function.

		Validates that duplicate actions (same action + arguments) are properly
		identified and counted, with the first occurrence retained and subsequent
		duplicates incrementing the count.

		Args:
			input_dicts: List of action dictionaries to process.
			expected_combinations: List of tuples (tool_name, params, expected_count)
				representing each unique tool+params combination and how many times
				it should appear.
		"""
		result = remove_duplicate_actions_with_counts(input_dicts)

		# Verify we have the expected number of unique combinations
		assert len(result) == len(expected_combinations), \
			f"Expected {len(expected_combinations)} unique combinations, got {len(result)}"

		# Build a mapping of (action, arguments) to result dict for easier comparison
		result_map = {}
		for item in result:
			action = item["action"]
			args = item["action_arguments"]
			# Convert args dict to frozenset of items for hashable key
			args_key = frozenset(args.items()) if args else frozenset()
			result_map[(action, args_key)] = item

		# Verify each expected combination
		for expected_tool, expected_params, expected_count in expected_combinations:
			# Create hashable key from expected params
			params_key = frozenset(expected_params.items()) if expected_params else frozenset()
			key = (expected_tool, params_key)

			assert key in result_map, \
				f"Expected combination (tool={expected_tool}, params={expected_params}) not found in results"

			actual_count = result_map[key]["unique_action_response_count"]
			assert actual_count == expected_count, (
				f"For combination (tool={expected_tool}, params={expected_params}): "
				f"expected count={expected_count}, got {actual_count}"
			)


class TestSanitizeParamName:
	"""Test cases for sanitize_param_name function."""

	@pytest.mark.parametrize(
		"input_name, expected_output",
		[
			("SimpleName", "simplename"),
			("Name With Spaces", "name_with_spaces"),
			("Name-With-Dashes", "name_with_dashes"),
			("Name(With)Parens", "namewithparens"),
			("Name123", "name123"),
			("Name___Underscores", "name_underscores"),
			("  Spaces  ", "spaces"),
			("__LeadingUnderscore", "leadingunderscore"),
		],
	)
	def test_sanitize_param_name(self, input_name: str, expected_output: str) -> None:
		"""Test sanitization of parameter names.

		Validates that parameter names are properly sanitized by converting to lowercase,
		replacing spaces and dashes with underscores, removing parentheses and leading
		underscores, and collapsing multiple underscores.

		Args:
			input_name: Raw parameter name to sanitize.
			expected_output: Expected sanitized parameter name.
		"""
		result = sanitize_param_name(input_name)
		assert result == expected_output, \
			f"Expected '{expected_output}', got '{result}'"


class TestCreateLiteralFromDict:
	"""Test cases for create_literal_from_dict function."""

	def test_create_literal_from_dict(self) -> None:
		"""Test creation of Literal type from dictionary keys.

		Validates that a Literal type is correctly created from dictionary keys,
		which is used for type-safe parameter validation in action spaces.
		"""
		options = {"OptionA": 1, "OptionB": 2}
		literal_type = create_literal_from_dict(options)

		# In runtime, Literal[...] is a special form. We can check __args__.
		assert hasattr(literal_type, "__args__"), \
			"Literal type should have __args__ attribute"
		assert set(literal_type.__args__) == {"OptionA", "OptionB"}, \
			f"Expected Literal['OptionA', 'OptionB'], got {literal_type.__args__}"


class TestLoadActionSpaceJson:
	"""Test cases for load_action_space_json function."""

	def test_load_valid_json(self, tmp_path: Path) -> None:
		"""Test loading a valid action space JSON file.

		Validates that a properly formatted JSON file is correctly parsed into
		an ActionSpaceConfig object with all fields populated.

		Args:
			tmp_path: Pytest fixture providing temporary directory path.
		"""
		data = {
			"name": "My Dimension",
			"definition": "A test dimension.",
			"choices": {
				"ChoiceA": {"definition": "Def A"},
				"ChoiceB": {"definition": "Def B"},
			},
		}
		json_file = tmp_path / "action_space.json"
		json_file.write_text(json.dumps(data))

		config = load_action_space_json(json_file)

		assert config.name == "my_dimension", \
			f"Expected name 'my_dimension', got '{config.name}'"
		assert config.definition == "A test dimension.", \
			f"Expected definition 'A test dimension.', got '{config.definition}'"
		assert "ChoiceA" in config.choices, "ChoiceA should be in choices"
		assert "ChoiceB" in config.choices, "ChoiceB should be in choices"

	def test_load_missing_file(self) -> None:
		"""Test loading a non-existent file raises FileNotFoundError.

		Validates that attempting to load a file that doesn't exist properly
		raises FileNotFoundError with appropriate error message.
		"""
		with pytest.raises(FileNotFoundError):
			load_action_space_json("non_existent_file.json")

	@pytest.mark.parametrize(
		"missing_key",
		[
			"name",
			"definition",
			"choices",
		],
	)
	def test_load_invalid_json(self, tmp_path: Path, missing_key: str) -> None:
		"""Test loading invalid JSON raises ValueError.

		Validates that JSON files missing required keys raise ValueError
		with clear error messages indicating which key is missing.

		Args:
			tmp_path: Pytest fixture providing temporary directory path.
			missing_key: The required key to omit from the JSON.
		"""
		data = {
			"name": "Name",
			"definition": "Def",
			"choices": {},
		}
		del data[missing_key]

		json_file = tmp_path / "invalid.json"
		json_file.write_text(json.dumps(data))

		with pytest.raises(ValueError, match=f"missing '{missing_key}'"):
			load_action_space_json(json_file)

	def test_load_with_mock_styles_fixture(self, temp_action_space_styles: Path) -> None:
		"""Test loading using mock styles fixture.

		Validates that the mock styles fixture is properly formatted and can be
		loaded successfully with all expected choices.

		Args:
			temp_action_space_styles: Fixture providing temporary styles JSON file.
		"""
		config = load_action_space_json(temp_action_space_styles)

		assert config.name == "style", f"Expected name 'style', got '{config.name}'"
		assert len(config.choices) == 3, f"Expected 3 style choices, got {len(config.choices)}"
		assert {"Figurative Language", "Statistical & Data-Driven", "Formal & Academic"} == set(config.choices)

	def test_load_with_mock_structures_fixture(self, temp_action_space_structures: Path) -> None:
		"""Test loading using mock structures fixture.

		Validates that the mock structures fixture is properly formatted and can be
		loaded successfully with all expected choices.

		Args:
			temp_action_space_structures: Fixture providing temporary structures JSON file.
		"""
		config = load_action_space_json(temp_action_space_structures)

		assert config.name == "structure", f"Expected name 'structure', got '{config.name}'"
		assert len(config.choices) == 2, f"Expected 2 structure choices, got {len(config.choices)}"
		assert {"Cause & Effect", "Contrast"} == set(config.choices)


class TestCreateReasoningInterventionFromChoices:
	"""Test cases for create_reasoning_intervention_from_choices function."""

	@pytest.fixture
	def mock_config(self) -> ActionSpaceConfig:
		"""Create a mock ActionSpaceConfig for testing.

		Returns:
			ActionSpaceConfig with style dimension and two choices.
		"""
		return ActionSpaceConfig(
			name="style",
			definition="Style dimension",
			choices={
				"Formal": {
					"internal_reasoning": "Result: Formal.",
					"prefix": "Therefore,",
				},
				"Informal": {
					"internal_reasoning": "Result: Informal.",
				},
			},
		)

	def test_valid_choice(self, mock_config: ActionSpaceConfig) -> None:
		"""Test creating intervention with valid choice.

		Validates that a valid choice is correctly processed into a
		ReasoningIntervention with appropriate internal reasoning and prefix.

		Args:
			mock_config: Fixture providing mock ActionSpaceConfig.
		"""
		intervention = create_reasoning_intervention_from_choices(
			[mock_config], {"style": "Formal"}
		)

		assert intervention.continue_reasoning is True, \
			"continue_reasoning should be True for valid choice"
		assert intervention.internal_reasoning == "Result: Formal.", \
			f"Expected 'Result: Formal.', got '{intervention.internal_reasoning}'"
		assert intervention.prefix == "Therefore,", \
			f"Expected 'Therefore,', got '{intervention.prefix}'"

	def test_missing_choice(self, mock_config: ActionSpaceConfig) -> None:
		"""Test error when choice is missing for a dimension.

		Validates that ValueError is raised when a required dimension choice
		is not provided in the chosen_values dictionary.

		Args:
			mock_config: Fixture providing mock ActionSpaceConfig.
		"""
		with pytest.raises(ValueError, match="Missing required choice"):
			create_reasoning_intervention_from_choices([mock_config], {})

	def test_invalid_choice(self, mock_config: ActionSpaceConfig) -> None:
		"""Test error when choice is invalid.

		Validates that ValueError is raised when an invalid choice value
		is provided for a dimension.

		Args:
			mock_config: Fixture providing mock ActionSpaceConfig.
		"""
		with pytest.raises(ValueError, match="Unknown choice"):
			create_reasoning_intervention_from_choices(
				[mock_config], {"style": "InvalidOption"}
			)

	def test_multiple_configs(self, mock_config: ActionSpaceConfig) -> None:
		"""Test combining choices from multiple configs.

		Validates that internal reasoning from multiple action space dimensions
		is properly combined with space separators.

		Args:
			mock_config: Fixture providing mock ActionSpaceConfig.
		"""
		config2 = ActionSpaceConfig(
			name="tone",
			definition="Tone dimension",
			choices={
				"Serious": {
					"internal_reasoning": "So serious.",
				}
			},
		)

		intervention = create_reasoning_intervention_from_choices(
			[mock_config, config2],
			{"style": "Formal", "tone": "Serious"}
		)

		# Should combine internal reasoning with space separator
		assert "Result: Formal." in intervention.internal_reasoning, \
			"Should contain Formal style reasoning"
		assert "So serious." in intervention.internal_reasoning, \
			"Should contain Serious tone reasoning"

	def test_with_mock_styles_fixture(self, temp_action_space_styles: Path) -> None:
		"""Test creating intervention from mock styles fixture.

		Validates that styles loaded from the fixture can be used to create
		interventions with proper internal reasoning and prefixes.

		Args:
			temp_action_space_styles: Fixture providing temporary styles JSON file.
		"""
		config = load_action_space_json(temp_action_space_styles)

		intervention = create_reasoning_intervention_from_choices(
			[config], {"style": "Statistical & Data-Driven"}
		)

		assert intervention.continue_reasoning is True
		assert "I should use numbers and data." in intervention.internal_reasoning
		assert intervention.prefix == "The data shows:"

	def test_with_multiple_mock_fixtures(
		self,
		temp_action_space_styles: Path,
		temp_action_space_structures: Path
	) -> None:
		"""Test creating intervention from multiple mock fixtures.

		Validates that multiple action spaces can be combined to create
		interventions with combined internal reasoning.

		Args:
			temp_action_space_styles: Fixture providing temporary styles JSON file.
			temp_action_space_structures: Fixture providing temporary structures JSON file.
		"""
		style_config = load_action_space_json(temp_action_space_styles)
		structure_config = load_action_space_json(temp_action_space_structures)

		intervention = create_reasoning_intervention_from_choices(
			[style_config, structure_config],
			{"style": "Formal & Academic", "structure": "Contrast"}
		)

		assert intervention.continue_reasoning is True
		assert "I should use academic tone." in intervention.internal_reasoning
		assert "I should present a contrasting view." in intervention.internal_reasoning
		# Both prefixes are concatenated
		assert "It can be argued that" in intervention.prefix
		assert "On the other hand," in intervention.prefix

if __name__ == "__main__":
	pytest.main([__file__])
