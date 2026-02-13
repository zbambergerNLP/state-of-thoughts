# Tests for field.py functionality including InputField, OutputField, ReasoningField and related utilities.

# Standard library imports

# Third-party imports
from typing import Any

import pydantic
import pytest
from dspy import infer_prefix

# Local imports
from signatures.field import (
	InputField,
	OldInputField,
	OldOutputField,
	OldReasoningField,
	OutputField,
	ReasoningField,
	_translate_pydantic_field_constraints,
	move_kwargs,
	new_to_old_field,
)
from tree.tree_constants import ReasoningState


@pytest.mark.parametrize(
	"kwargs, expected",
	[
		({"gt": 5}, "greater than 5"),
		(
			{"ge": 10, "le": 20},
			"greater than or equal to 10 and less than or equal to 20",
		),
		(
			{"min_length": 1, "max_length": 100},
			"minimum length 1 and maximum length 100",
		),
		({"multiple_of": 3}, "a multiple of 3"),
		({}, ""),
		({"description": "test", "default": None}, ""),
		(
			{
				"gt": 0,
				"lt": 100,
				"multiple_of": 5,
				"description": "A number",
				"default": 25,
			},
			"greater than 0, less than 100, and a multiple of 5",
		),
	],
)
def test_translate_pydantic_field_constraints(
	kwargs: dict[str, int | str | None],
	expected: str,
) -> None:
	"""
	Test translation of pydantic constraints to human-readable format.

	Args:
	            kwargs (dict): Pydantic field constraints.
	    expected (str): Expected human-readable constraints string.
	"""
	result = _translate_pydantic_field_constraints(**kwargs)
	assert result == expected


@pytest.mark.parametrize(
	"kwargs1, kwargs2, expected_consistency_checks",
	[
		(
			{"min_length": 5, "gt": 0, "max_length": 100, "le": 50},
			{"le": 50, "max_length": 100, "gt": 0, "min_length": 5},
			[
				"greater than 0",
				"less than or equal to 50",
				"minimum length 5",
				"maximum length 100",
			],
		),
		(
			{"gt": 10, "le": 20},
			{"le": 20, "gt": 10},
			["greater than 10", "less than or equal to 20"],
		),
		(
			{"min_length": 1, "max_length": 50},
			{"max_length": 50, "min_length": 1},
			["minimum length 1", "maximum length 50"],
		),
		(
			{"multiple_of": 3, "gt": 0},
			{"gt": 0, "multiple_of": 3},
			["greater than 0", "a multiple of 3"],
		),
	],
)
def test_constraint_order_consistency(
	kwargs1: dict[str, int | str | None],
	kwargs2: dict[str, int | str | None],
	expected_consistency_checks: list[str],
) -> None:
	"""
	    Test that constraints maintain consistent order regardless of input order.

	Ensures that the order of constraints does not affect the output, and that
	the expected consistency checks are present in the result.

	Args:
	            kwargs1 (dict): First set of pydantic field constraints.
	            kwargs2 (dict): Second set of pydantic field constraints.
	            expected_consistency_checks (list): Expected list of consistency checks.
	"""
	result1 = _translate_pydantic_field_constraints(**kwargs1)
	result2 = _translate_pydantic_field_constraints(**kwargs2)
	# Results should be identical regardless of input order
	assert result1 == result2
	for check in expected_consistency_checks:
		assert check in result1


def test_constraint_parsing_for_display() -> None:
	"""Test constraint parsing for display in adapter-style formatting."""
	# This mimics how constraints are used in adapter/utils.py
	field = InputField(
		desc="Test field with constraints", gt=0, le=100, min_length=5, max_length=50
	)
	constraints = field.json_schema_extra.get("constraints", "")
	# Test that constraints can be easily formatted for display (now bulletized)
	if constraints:
		# Parse constraints properly considering "and" syntax
		# Replace ", and " with ", " to normalize, then split on ", "
		normalized_constraints = constraints.replace(", and ", ", ")
		individual_constraints = [
			c.strip() for c in normalized_constraints.split(",") if c.strip()
		]
		expected_constraints = [
			"greater than 0",
			"less than or equal to 100",
			"minimum length 5",
			"maximum length 50",
		]
		assert individual_constraints == expected_constraints


@pytest.mark.parametrize(
	"kwargs, expected_result",
	[
		(
			{
				"desc": "Test description",
				"prefix": "Input:",
				"default": "default_value",
				"min_length": 5,
			},
			{
				"default": "default_value",
				"min_length": 5,
				"json_schema_extra": {
					"desc": "Test description",
					"prefix": "Input:",
					"constraints": "minimum length 5",
				},
			},
		),
		(
			{
				"description": "Pydantic description",
				"default": "test",
			},
			{
				"default": "test",
				"description": "Pydantic description",
				"json_schema_extra": {
					"desc": "Pydantic description",
				},
			},
		),
		(
			{
				"desc": "DSPy description",
				"description": "Pydantic description",
				"default": "test",
			},
			{
				"default": "test",
				"description": "Pydantic description",
				"json_schema_extra": {
					"desc": "DSPy description",
				},
			},
		),
		(
			{
				"desc": "Test field",
				"gt": 0,
				"le": 100,
			},
			{
				"gt": 0,
				"le": 100,
				"json_schema_extra": {
					"desc": "Test field",
					"constraints": "greater than 0 and less than or equal to 100",
				},
			},
		),
		(
			{
				"desc": "Description",
				"prefix": "Prefix:",
				"format": "json",
				"parser": str,
				"__dspy_field_type": ReasoningState.INPUT,
			},
			{
				"json_schema_extra": {
					"desc": "Description",
					"prefix": "Prefix:",
					"format": "json",
					"parser": str,
					"__dspy_field_type": ReasoningState.INPUT,
				},
			},
		),
	],
)
def test_move_kwargs(kwargs: dict[str, Any], expected_result: dict[str, Any]) -> None:
	"""
	Test that the move_kwargs function separates DSPy and Pydantic arguments.

	Args:
	            kwargs (dict): Keyword arguments to be moved to pydantic.Field.
	            expected_result (dict): Expected result after moving arguments.
	"""
	result = move_kwargs(**kwargs)

	# Check that all expected keys are present and have expected values
	for key, expected_value in expected_result.items():
		assert key in result
		assert result[key] == expected_value


@pytest.mark.parametrize(
	"field_func, field_type, kwargs, expected_field_type, expected_attributes",
	[
		(
			InputField,
			ReasoningState.INPUT,
			{"desc": "Test input", "prefix": "Input:", "default": "test"},
			ReasoningState.INPUT,
			{"desc": "Test input", "prefix": "Input:"},
		),
		(
			OutputField,
			ReasoningState.OUTPUT,
			{"desc": "Test output", "format": "json"},
			ReasoningState.OUTPUT,
			{"desc": "Test output", "format": "json"},
		),
		(
			ReasoningField,
			ReasoningState.REASONING,
			{"desc": "Test reasoning", "prefix": "Reasoning:"},
			ReasoningState.REASONING,
			{"desc": "Test reasoning", "prefix": "Reasoning:"},
		),
	],
)
def test_field_creation(
	field_func: Any,
	field_type: str,
	kwargs: dict,
	expected_field_type: str,
	expected_attributes: dict[str, Any],
) -> None:
	"""Test creation of different field types."""
	field = field_func(**kwargs)
	assert isinstance(field, pydantic.fields.FieldInfo)

	json_extra = field.json_schema_extra
	if not isinstance(json_extra, dict):
		json_extra = {}

	assert json_extra["__dspy_field_type"] == expected_field_type

	for attr_name, expected_value in expected_attributes.items():
		assert json_extra[attr_name] == expected_value


def test_field_with_constraints() -> None:
	"""Test field creation with pydantic constraints."""
	field = InputField(
		desc="Constrained field",
		min_length=5,
		max_length=100,
		gt=0,
	)
	json_extra = field.json_schema_extra
	assert (
		json_extra["constraints"]
		== "greater than 0, minimum length 5, and maximum length 100"
	)


@pytest.mark.parametrize(
	"new_field_func, old_field_type, field_kwargs, expected_attributes",
	[
		(
			InputField,
			OldInputField,
			{"desc": "Test input", "prefix": "Input:", "format": "text"},
			{"desc": "Test input", "prefix": "Input:", "format": "text"},
		),
		(
			OutputField,
			OldOutputField,
			{"desc": "Test output", "prefix": "Output:"},
			{"desc": "Test output", "prefix": "Output:"},
		),
		(
			ReasoningField,
			OldReasoningField,
			{"desc": "Test reasoning", "prefix": "Reasoning:"},
			{"desc": "Test reasoning", "prefix": "Reasoning:"},
		),
	],
)
def test_new_to_old_field_conversion(
	new_field_func, old_field_type, field_kwargs: dict, expected_attributes: dict
) -> None:
	"""Test conversion from new pydantic fields to old field format."""
	field = new_field_func(**field_kwargs)
	old_field = new_to_old_field(field)

	assert isinstance(old_field, old_field_type)

	for attr_name, expected_value in expected_attributes.items():
		assert getattr(old_field, attr_name) == expected_value


def test_conversion_with_none_values() -> None:
	"""Test conversion when some values are None."""
	field = InputField(desc="Test")
	old_field = new_to_old_field(field)

	assert old_field.desc == "Test"
	assert old_field.prefix is None
	assert old_field.format is None


def test_unknown_field_type_raises_error() -> None:
	"""Test that unknown field types raise ValueError."""
	# Create a mock field with unknown type
	field = pydantic.Field()
	field.json_schema_extra = {"__dspy_field_type": "unknown"}

	with pytest.raises(ValueError, match="Unknown field type: unknown"):
		new_to_old_field(field)


@pytest.mark.parametrize(
	"old_field_class, init_kwargs, expected_attributes",
	[
		(
			OldInputField,
			{"prefix": "Input:", "desc": "Test description", "format": "json"},
			{"prefix": "Input:", "desc": "Test description", "format": "json"},
		),
		(
			OldOutputField,
			{"prefix": "Output:", "desc": "Test output"},
			{"prefix": "Output:", "desc": "Test output", "format": None},
		),
		(
			OldReasoningField,
			{"desc": "Reasoning step"},
			{"desc": "Reasoning step", "prefix": None, "format": None},
		),
	],
)
def test_old_field_initialization(
	old_field_class, init_kwargs: dict, expected_attributes: dict
) -> None:
	"""Test initialization of old field classes."""
	field = old_field_class(**init_kwargs)

	for attr_name, expected_value in expected_attributes.items():
		assert getattr(field, attr_name) == expected_value


@pytest.mark.parametrize(
	# Parameter names
	[
		"old_field_class",
		"init_kwargs",
		"finalize_key",
		"finalize_type",
		"expected_prefix",
		"expected_desc",
	],
	# Parameter values
	[
		pytest.param(
			OldInputField,									# old_field_class
			{"desc": "Test"},								# init_kwargs
			"test_key",  									# finalize_key
			ReasoningState.INPUT,  										# finalize_type
			"Test Key:",  									# expected_prefix
			"Test",  										# expected_desc
			id="test_input_field_finalize_with_key",
		),
		pytest.param(
			OldInputField,									# old_field_class
			{"prefix": "Custom:", "desc": "Test"},			# init_kwargs
			"test_key",  									# finalize_key
			ReasoningState.INPUT,  										# finalize_type
			"Custom:",  									# expected_prefix
			"Test",  										# expected_desc
			id="test_input_field_finalize_with_prefix",
		),
		pytest.param(
			OldInputField,									# old_field_class
			{"prefix": "Input:"},							# init_kwargs
			"test_key",  									# finalize_key
			ReasoningState.INPUT,  										# finalize_type
			"Input:",  										# expected_prefix
			"${test_key}",  								# expected_desc
			id="test_input_field_finalize_with_key",
		),
	],
)
def test_field_finalize(
	old_field_class,
	init_kwargs: dict,
	finalize_key: str,
	finalize_type: str,
	expected_prefix: str,
	expected_desc: str,
) -> None:
	"""Test field finalization behavior."""
	field = old_field_class(**init_kwargs)
	field.finalize(finalize_key, infer_prefix(finalize_key))

	assert field.prefix == expected_prefix
	assert field.desc == expected_desc


def test_field_equality() -> None:
	"""Test field equality comparison."""
	field1 = OldInputField(prefix="Input:", desc="Test")
	field2 = OldInputField(prefix="Input:", desc="Test")
	field3 = OldInputField(prefix="Different:", desc="Test")

	assert field1 == field2
	assert field1 != field3


def test_field_repr() -> None:
	"""Test field string representation."""
	field = OldInputField(prefix="Input:", desc="Test description")
	repr_str = repr(field)

	assert "OldInputField" in repr_str
	assert "prefix=Input:" in repr_str
	assert "desc=Test description" in repr_str


@pytest.mark.parametrize(
	"field_func, old_field_type, field_kwargs, expected_old_attributes, expected_constraints",
	[
		(
			InputField,
			OldInputField,
			{
				"desc": "User input",
				"prefix": "User:",
				"min_length": 1,
				"max_length": 500,
				"default": "",
			},
			{"desc": "User input", "prefix": "User:"},
			["minimum length 1", "maximum length 500"],
		),
		(
			ReasoningField,
			OldReasoningField,
			{
				"desc": "Step-by-step reasoning",
				"prefix": "Reasoning:",
				"format": "structured",
			},
			{
				"desc": "Step-by-step reasoning",
				"prefix": "Reasoning:",
				"format": "structured",
			},
			[],
		),
	],
)
def test_complete_workflow(
	field_func,
	old_field_type,
	field_kwargs: dict,
	expected_old_attributes: dict,
	expected_constraints: list,
) -> None:
	"""Test complete workflow from field creation to old field conversion."""
	# Create new field
	new_field = field_func(**field_kwargs)

	# Convert to old field
	old_field = new_to_old_field(new_field)

	# Verify the conversion
	assert isinstance(old_field, old_field_type)

	for attr_name, expected_value in expected_old_attributes.items():
		assert getattr(old_field, attr_name) == expected_value

	# Verify constraints if any
	if expected_constraints:
		json_extra = new_field.json_schema_extra
		assert "constraints" in json_extra
		for constraint in expected_constraints:
			assert constraint in json_extra["constraints"]


@pytest.mark.parametrize(
	"field_func, expected_type, field_type_name",
	[
		(InputField, OldInputField, ReasoningState.INPUT),
		(OutputField, OldOutputField, ReasoningState.OUTPUT),
		(ReasoningField, OldReasoningField, ReasoningState.REASONING),
	],
)
def test_field_type_consistency(
	field_func, expected_type, field_type_name: str
) -> None:
	"""Test that field types are consistent across creation and conversion."""
	# Create field
	field = field_func(desc="Test field")

	# Check field type in json_schema_extra
	assert field.json_schema_extra["__dspy_field_type"] == field_type_name

	# Convert and check type
	old_field = new_to_old_field(field)
	assert isinstance(old_field, expected_type)


def test_field_with_all_parameters() -> None:
	"""Test field creation with all possible parameters."""
	field = ReasoningField(
		desc="Complex reasoning field",
		prefix="Think:",
		format="json",
		parser=str,
		default="Start thinking...",
		min_length=10,
		max_length=1000,
		gt=0,
	)

	# Verify DSPy parameters are in json_schema_extra
	json_extra = field.json_schema_extra
	assert json_extra["desc"] == "Complex reasoning field"
	assert json_extra["prefix"] == "Think:"
	assert json_extra["format"] == "json"
	assert json_extra["parser"] is str
	assert json_extra["__dspy_field_type"] == "reasoning"

	# Verify constraints
	assert "constraints" in json_extra
	assert "greater than 0" in json_extra["constraints"]
	assert "minimum length 10" in json_extra["constraints"]
	assert "maximum length 1000" in json_extra["constraints"]

	# Verify conversion works
	old_field = new_to_old_field(field)
	assert isinstance(old_field, OldReasoningField)
	assert old_field.desc == "Complex reasoning field"
	assert old_field.prefix == "Think:"
	assert old_field.format == "json"


def test_constraint_display_format_matching_adapter() -> None:
	"""Test that constraint display format matches what's expected in adapter/utils.py."""
	field = InputField(
		desc="Test input with constraints", gt=0, le=100, min_length=5, max_length=200
	)

	# This mimics how constraints are displayed in adapter/utils.py get_field_description_string
	constraints = field.json_schema_extra.get("constraints", "")
	display_format = ""
	if constraints and isinstance(constraints, str):
		# Parse constraints properly considering "and" syntax
		# Replace ", and " with ", " to normalize, then split on ", "
		normalized_constraints = constraints.replace(", and ", ", ")
		individual_constraints = [
			c.strip() for c in normalized_constraints.split(",") if c.strip()
		]
		if individual_constraints:
			display_format = "\n\tConstraints:"
			for constraint in individual_constraints:
				display_format += f"\n\t\t* {constraint}"

	expected_display = (
		"\n\tConstraints:"
		"\n\t\t* greater than 0"
		"\n\t\t* less than or equal to 100"
		"\n\t\t* minimum length 5"
		"\n\t\t* maximum length 200"
	)
	assert display_format == expected_display


def test_reasoning_field_with_constraints_for_adapter() -> None:
	"""Test ReasoningField with constraints formatted for adapter display."""
	field = ReasoningField(
		desc="Reasoning step with length constraints",
		prefix="Step:",
		min_length=10,
		max_length=500,
	)

	# Verify the field has the right type and constraints
	assert field.json_schema_extra["__dspy_field_type"] == "reasoning"
	assert field.json_schema_extra["desc"] == "Reasoning step with length constraints"
	assert field.json_schema_extra["prefix"] == "Step:"

	constraints = field.json_schema_extra.get("constraints", "")
	assert constraints == "minimum length 10 and maximum length 500"


@pytest.mark.parametrize(
	"field_func, field_type, expected_constraints",
	[
		(
			InputField,
			ReasoningState.INPUT,
			"greater than or equal to 1 and less than or equal to 10",
		),
		(OutputField, ReasoningState.OUTPUT, "minimum length 1 and maximum length 100"),
		(
			ReasoningField,
			"reasoning",
			"greater than 0, maximum length 1000, and a multiple of 5",
		),
	],
)
def test_multiple_field_types_with_constraints(
	field_func, field_type: str, expected_constraints: str
) -> None:
	"""Test different field types with various constraints for adapter compatibility."""
	if field_type == ReasoningState.INPUT:
		field = field_func(desc="Input with range constraints", ge=1, le=10)
	elif field_type == ReasoningState.OUTPUT:
		field = field_func(
			desc="Output with length constraints", min_length=1, max_length=100
		)
	else:  # reasoning
		field = field_func(
			desc="Reasoning with multiple constraints",
			gt=0,
			multiple_of=5,
			max_length=1000,
		)

	constraints = field.json_schema_extra.get("constraints", "")
	assert constraints == expected_constraints


@pytest.mark.parametrize(
	"field_type, field_func, type_name",
	[
		(ReasoningState.INPUT, InputField, ReasoningState.INPUT),
		(ReasoningState.OUTPUT, OutputField, ReasoningState.OUTPUT),
		(ReasoningState.REASONING, ReasoningField, ReasoningState.REASONING),
	],
)
def test_field_types_constraint_consistency(
	field_type: str, field_func, type_name: str
) -> None:
	"""Test that all field types handle constraints consistently for adapter use."""
	field = field_func(
		desc=f"Test {field_type} field", min_length=10, max_length=100, gt=5
	)
	# All field types should have consistent constraint formatting
	constraints = field.json_schema_extra.get("constraints", "")
	assert "greater than 5" in constraints
	assert "minimum length 10" in constraints
	assert "maximum length 100" in constraints
	# Field type should be correct
	assert field.json_schema_extra["__dspy_field_type"] == type_name


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
