"""
Tests for the utility functions in misc_utils.py.
"""

from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

import pytest

from misc_utils import (
	ExecutionError,
	_is_instance_of_type,
	_is_optional_type,
	format_list_of_fields,
	parse_base_signature,
	stringify_without_metadata,
	strip_metadata_from_obj,
	serialize_object,
)


@pytest.mark.parametrize(
	["input_fields", "expected_output"],
	[
		pytest.param(["field1", "field2"], "`field1` and `field2`", id="two_fields"),
		pytest.param(["field1"], "`field1`", id="one_field"),
		pytest.param(["a", "b", "c", "d"], "`a`, `b`, `c`, and `d`", id="multi_fields"),
	],
)
def test_format_list_of_fields(input_fields: list[str], expected_output: str) -> None:
	"""Test the format_list_of_fields function with various input cases."""
	assert format_list_of_fields(input_fields) == expected_output


def test_format_list_of_fields_empty() -> None:
	"""Test the format_list_of_fields function with an empty list raises AssertionError."""
	with pytest.raises(AssertionError):
		format_list_of_fields([])


@pytest.mark.parametrize(
	["input_field_names", "output_field_names", "expected_output"],
	[
		pytest.param(
			["input1", "input2"],
			["output1"],
			("`input1` and `input2`", "`output1`"),
			id="multi_input_single_output"
		),
		pytest.param(
			["input1"],
			["output1", "output2"],
			("`input1`", "`output1` and `output2`"),
			id="single_input_multi_output"
		),
		pytest.param(
			["input1", "input2", "input3"],
			["output1", "output2"],
			("`input1`, `input2`, and `input3`", "`output1` and `output2`"),
			id="triple_input_double_output"
		),
	],
)
def test_parse_base_signature(
	input_field_names: list[str],
	output_field_names: list[str],
	expected_output: tuple[str, str],
) -> None:
	"""Test the parse_base_signature function with various field combinations."""
	assert parse_base_signature(input_field_names, output_field_names) == expected_output


@pytest.mark.parametrize(
	["value", "type_hint", "expected"],
	[
		pytest.param("hello", str, True, id="str_match"),
		pytest.param(42, int, True, id="int_match"),
		pytest.param(3.14, float, True, id="float_match"),
		pytest.param(True, bool, True, id="bool_match"),
		pytest.param("hello", int, False, id="str_int_mismatch"),
		pytest.param(["a", "b"], list[str], True, id="list_str_match"),
		pytest.param([1, 2], list[str], False, id="list_int_str_mismatch"),
		pytest.param({"a": 1}, dict[str, int], True, id="dict_match"),
		pytest.param("hello", str | int, True, id="union_match"),
		pytest.param(None, Optional[str], True, id="optional_none_match"),
	],
)
def test_is_instance_of_type(value: Any, type_hint: Any, expected: bool) -> None:
	"""Test the _is_instance_of_type helper function."""
	assert _is_instance_of_type(value, type_hint) is expected


@pytest.mark.parametrize(
	["type_hint", "expected"],
	[
		pytest.param(str, False, id="str_not_optional"),
		pytest.param(Optional[str], True, id="optional_str"),
		pytest.param(str | None, True, id="union_none"),
		pytest.param(str | int, False, id="union_no_none"),
	],
)
def test_is_optional_type(type_hint: Any, expected: bool) -> None:
	"""Test the _is_optional_type helper function."""
	assert _is_optional_type(type_hint) is expected


@pytest.mark.parametrize(
	["error_type", "raw_output", "error_message", "expected_has_error"],
	[
		pytest.param(None, None, None, False, id="no_error"),
		pytest.param("generation", None, "Context length exceeded", True, id="generation_error"),
		pytest.param("parsing", "invalid", "Parse failed", True, id="parsing_error"),
	],
)
def test_execution_error(
	error_type: str | None,
	raw_output: str,
	error_message: str | None,
	expected_has_error: bool,
) -> None:
	"""Test ExecutionError model methods."""
	error = ExecutionError(
		error_type=error_type,
		raw_output=raw_output,
		error_message=error_message,
	)
	assert error.has_error() == expected_has_error


@pytest.mark.parametrize(
	["obj", "expected_str"],
	[
		pytest.param(
			{"answer": "Hello", "error": ExecutionError()},
			"Hello",
			id="strip_error_single_field"
		),
		pytest.param(
			[
				{"reasoning_step": "A", "error": ExecutionError()},
				{"reasoning_step": "B", "error": ExecutionError()},
			],
			"[\n\tA,\n\tB,\n]",
			id="strip_error_nested_list"
		),
	],
)
def test_stringify_without_metadata(obj: Any, expected_str: str) -> None:
	"""Test stringify_without_metadata recursiveness."""
	assert stringify_without_metadata(obj) == expected_str


def test_serialize_object() -> None:
	"""Test serialize_object with diverse types."""
	@dataclass
	class Point:
		x: int
		y: int
		
	class Color(Enum):
		RED = "red"
		BLUE = "blue"
		
	class SimpleObj:
		def __init__(self):
			self.a = 1
			self.b = "2"
			
	dt = datetime(2023, 1, 1, 12, 0, 0)
	path = Path("/tmp/test")
	
	input_data = {
		"dataclass": Point(10, 20),
		"enum": Color.RED,
		"datetime": dt,
		"path": path,
		"obj": SimpleObj(),
		"nested": {
			"list": [Point(1, 1)],
			"tuple": (path,)
		}
	}
	
	sanitized = serialize_object(input_data)
	
	assert sanitized["dataclass"] == {"x": 10, "y": 20}
	assert sanitized["enum"] == "Color.RED"
	assert sanitized["datetime"] == dt.isoformat()
	assert sanitized["path"] == str(path)
	assert sanitized["obj"] == {"a": 1, "b": "2"}
	assert sanitized["nested"]["list"] == [{"x": 1, "y": 1}]
	assert sanitized["nested"]["tuple"] == [str(path)]
