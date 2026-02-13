"""
Unit tests for converting `dspy.Tool` definitions into OpenAI-compatible tool schemas.
"""

# Standard library imports
import enum
from typing import Any, Literal, Optional, Union

# Third-party imports
import dspy
import pytest

# Local imports
from adapter.tool_schema import format_dspy_tool_as_openai_tool


class _Color(enum.StrEnum):
	"""Example enum for testing schema `enum` emission."""

	RED = "red"
	BLUE = "blue"


class _Level(enum.IntEnum):
	"""Example int enum for testing integer enum emission."""

	LOW = 1
	HIGH = 2


class _CustomType:
	"""Example custom type to ensure unknown annotations fall back to empty schema fragments."""


class TestFormatDspyToolAsOpenAITool:
	"""Test cases for `format_dspy_tool_as_openai_tool`."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"arg_types",
			"expected_properties",
			"expected_required",
		],
		# Parameter values
		[
			pytest.param(
				{  # arg_types
					"color": _Color,
				},
				{  # expected_properties
					"color": {"type": "string", "enum": ["red", "blue"]},
				},
				["color"],  # expected_required
				id="enum_strenum_emits_string_enum",
			),
			pytest.param(
				{  # arg_types
					"level": Literal[1, 2, 3],
				},
				{  # expected_properties
					"level": {"type": "integer", "enum": [1, 2, 3]},
				},
				["level"],  # expected_required
				id="literal_int_emits_integer_enum",
			),
			pytest.param(
				{  # arg_types
					"mode": Literal["a", "b"],
				},
				{  # expected_properties
					"mode": {"type": "string", "enum": ["a", "b"]},
				},
				["mode"],  # expected_required
				id="literal_str_emits_string_enum",
			),
		],
	)
	def test_emits_enum_schemas(
		self,
		arg_types: dict[str, Any],
		expected_properties: dict[str, dict[str, Any]],
		expected_required: list[str],
	) -> None:
		"""Ensure enums are emitted as JSON Schema `enum` in tool parameters.

		Args:
			arg_types: Mapping of tool arg name -> Python annotation.
			expected_properties: Expected JSON Schema fragments for each tool arg.
			expected_required: Expected required list produced by `format_dspy_tool_as_openai_tool`.
		"""

		def tool_func(**kwargs: Any) -> None:  # pragma: no cover - not executed
			return None

		args = {k: f"Choose {k}." for k in arg_types.keys()}
		tool = dspy.Tool(
			name="test_tool",
			func=tool_func,
			desc="Test tool",
			args=args,
			arg_types=arg_types,
		)

		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "test_tool", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."
		assert isinstance(params["properties"], dict), "Tool parameters properties must be a dict."
		assert isinstance(params["required"], list), "Tool parameters required must be a list."

		props = schema["function"]["parameters"]["properties"]
		for arg_name, expected in expected_properties.items():
			assert arg_name in props, f"Expected '{arg_name}' in tool schema properties."
			for key, value in expected.items():
				assert props[arg_name][key] == value, (
					f"Property mismatch for '{arg_name}.{key}': "
					f"expected {value!r}, got {props[arg_name].get(key)!r}."
				)

		assert schema["function"]["parameters"]["required"] == expected_required, (
			f"Required mismatch: expected {expected_required!r}, "
			f"got {schema['function']['parameters']['required']!r}."
		)


	def test_defaulted_param_is_not_required(self) -> None:
		"""Ensure tool params with signature defaults are not included in `required`.

		This verifies that `format_dspy_tool_as_openai_tool` consults the tool function
		signature to infer requiredness, not just `tool.args`.
		"""

		def tool_func(
			mode: Literal["a", "b"] = "a",
		) -> None:  # pragma: no cover - not executed
			return None

		tool = dspy.Tool(
			name="test_tool_defaults",
			func=tool_func,
			desc="Test tool with defaults",
			args={"mode": "Choose mode."},
			arg_types={"mode": Literal["a", "b"]},
		)

		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "test_tool_defaults", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."
		assert isinstance(params["properties"], dict), "Tool parameters properties must be a dict."
		assert isinstance(params["required"], list), "Tool parameters required must be a list."

		assert schema["function"]["parameters"]["required"] == [], (
			"Defaulted parameters should not be required."
		)

	@pytest.mark.parametrize(
		[
			"arg_types",
			"expected_properties",
		],
		[
			pytest.param(
				# arg_types
				{
					"query": str,
					"limit": int,
					"temperature": float,
					"safe": bool,
				},
				# expected_properties
				{
					"query": {"type": "string"},
					"limit": {"type": "integer"},
					"temperature": {"type": "number"},
					"safe": {"type": "boolean"},
				},
				id="primitives_mixed_types",
			),
			pytest.param(
				# arg_types
				{
					"ids": list[int],
					"tags": set[str],
					"pairs": tuple[str],
				},
				# expected_properties
				{
					"ids": {"type": "array", "items": {"type": "integer"}},
					"tags": {"type": "array", "items": {"type": "string"}},
					"pairs": {"type": "array", "items": {"type": "string"}},
				},
				id="containers_arrays",
			),
			pytest.param(
				# arg_types
				{
					"maybe_id": Optional[int],
					"union_val": Union[int, str],
				},
				# expected_properties
				{
					"maybe_id": {
						"anyOf": [
							{"type": "integer"},
							{"type": "null"},
						],
					},
					"union_val": {
						"anyOf": [
							{"type": "integer"},
							{"type": "string"},
						],
					},
				},
				id="union_and_optional_anyof",
			),
			pytest.param(
				# arg_types
				{"metadata": dict[str, Any]},
				# expected_properties
				{"metadata": {"type": "object"}},
				id="dict_is_object",
			),
			pytest.param(
				# arg_types
				{"mixed_literal": Literal[1, "a"]},
				# expected_properties
				{"mixed_literal": {"enum": [1, "a"]}},
				id="literal_mixed_types_has_no_single_type",
			),
			pytest.param(
				# arg_types
				{"level": _Level},
				# expected_properties
				{"level": {"type": "integer", "enum": [1, 2]}},
				id="enum_int_emits_integer_enum",
			),
			pytest.param(
				# arg_types
				{"custom": _CustomType},
				# expected_properties
				{"custom": {}},
				id="unknown_type_falls_back_to_empty_schema",
			),
		],
	)
	def test_emits_multi_arg_varying_types(
		self,
		arg_types: dict[str, Any],
		expected_properties: dict[str, dict[str, Any]],
	) -> None:
		"""Cover multi-input tools spanning supported schema conversions.

		Args:
			arg_types: Mapping of tool arg name -> Python annotation.
			expected_properties: Expected JSON Schema fragments for each tool arg.
		"""

		def tool_func(**kwargs: Any) -> None:  # pragma: no cover - not executed
			return None

		args = {k: f"Describe {k}." for k in arg_types.keys()}
		tool = dspy.Tool(
			name="multi_arg_tool",
			func=tool_func,
			desc="Multi-arg tool",
			args=args,
			arg_types=arg_types,
		)
		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "multi_arg_tool", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."
		assert isinstance(params["properties"], dict), "Tool parameters properties must be a dict."
		assert isinstance(params["required"], list), "Tool parameters required must be a list."

		props = schema["function"]["parameters"]["properties"]
		for arg_name, expected in expected_properties.items():
			assert arg_name in props, f"Expected '{arg_name}' in tool schema properties."
			assert props[arg_name]["description"] == f"Describe {arg_name}.", (
				f"Expected description for '{arg_name}' to be present."
			)
			for k, v in expected.items():
				assert props[arg_name][k] == v, (
					f"Property mismatch for '{arg_name}.{k}': expected {v!r}, "
					f"got {props[arg_name].get(k)!r}."
				)

	@pytest.mark.parametrize(
		["args", "arg_types", "expected_required"],
		[
			pytest.param(
				# args
				{"a": "A", "b": "B", "c": "C"},
				# arg_types
				{
					"a": int,
					"b": Optional[str],
					"c": Literal["x", "y"],
				},
				# expected_required
				["a", "c"],
				id="optional_is_not_required",
			),
			pytest.param(
				# args
				{"x": "X", "y": "Y"},
				# arg_types
				{
					"x": int,
					"y": int,
				},
				# expected_required
				["x"],
				id="defaults_in_signature_not_required",
			),
			pytest.param(
				# args
				{"sig_missing": "Sig missing param"},
				# arg_types
				{"sig_missing": str},
				# expected_required
				["sig_missing"],
				id="arg_not_in_signature_is_required",
			),
		],
	)
	def test_required_inference(
		self,
		args: dict[str, str],
		arg_types: dict[str, Any],
		expected_required: list[str],
	) -> None:
		"""Ensure required fields reflect signature defaults and Optional annotations.

		Args:
			args: Tool arg descriptions (controls which args appear in schema).
			arg_types: Tool arg type annotations used for schema conversion.
			expected_required: Expected `required` list in JSON Schema.
		"""

		def tool_func(x: int, y: int = 0) -> int:  # pragma: no cover - not executed
			return x + y

		tool = dspy.Tool(
			name="required_tool",
			func=tool_func,
			desc="Required inference tool",
			args=args,
			arg_types=arg_types,
		)
		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "required_tool", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."
		assert isinstance(params["properties"], dict), "Tool parameters properties must be a dict."
		assert isinstance(params["required"], list), "Tool parameters required must be a list."

		assert schema["function"]["parameters"]["required"] == expected_required, (
			f"Required mismatch: expected {expected_required!r}, "
			f"got {schema['function']['parameters']['required']!r}."
		)

	def test_ignores_return_type_single_output(self) -> None:
		"""Ensure return annotations (single output) do not alter OpenAI tool schema.

		OpenAI tool schemas describe input parameters only; any return type annotation is
		ignored by `format_dspy_tool_as_openai_tool`.
		"""

		def tool_func(x: int, y: int) -> int:  # pragma: no cover - not executed
			return x + y

		tool = dspy.Tool(
			name="return_int_tool",
			func=tool_func,
			desc="Return int tool",
			args={"x": "X", "y": "Y"},
			arg_types={"x": int, "y": int},
		)
		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "return_int_tool", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."

		assert "returns" not in schema["function"], (
			"Return type should not be represented in the OpenAI tool schema."
		)

	def test_ignores_return_type_multi_output(self) -> None:
		"""Ensure multi-output return annotations do not alter input JSON Schema."""

		def tool_func(x: int) -> tuple[int, str]:  # pragma: no cover - not executed
			return x, str(x)

		tool = dspy.Tool(
			name="return_tuple_tool",
			func=tool_func,
			desc="Return tuple tool",
			args={"x": "X"},
			arg_types={"x": int},
		)
		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "return_tuple_tool", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."

		assert schema["function"]["parameters"]["properties"]["x"]["type"] == "integer", (
			"Input parameter schema should be unchanged by return type annotations."
		)

	def test_only_declared_args_are_emitted(self) -> None:
		"""Ensure `tool.args` keys control which parameters appear in the schema.

		This test ensures that extra entries in `tool.arg_types` (or extra parameters in
		the function signature) do not appear unless they are declared in `tool.args`.
		"""

		def tool_func(a: int, b: int, c: int) -> None:  # pragma: no cover - not executed
			return None

		tool = dspy.Tool(
			name="declared_args_only",
			func=tool_func,
			desc="Declared args only",
			args={"a": "A", "c": "C"},
			arg_types={"a": int, "b": int, "c": int},
		)
		schema = format_dspy_tool_as_openai_tool(tool)
		assert schema["type"] == "function", "Tool schema must have type='function'."
		assert schema["function"]["name"] == "declared_args_only", "Tool name must be preserved."
		assert "description" in schema["function"], "Tool schema must include description."

		params = schema["function"]["parameters"]
		assert params["type"] == "object", "Tool parameters schema must be an object."
		assert params["additionalProperties"] is False, "Tool parameters must forbid extra keys."

		props = schema["function"]["parameters"]["properties"]
		assert set(props.keys()) == {"a", "c"}, (
			f"Expected only declared args {{'a','c'}}, got {set(props.keys())!r}."
		)

if __name__ == "__main__":
	pytest.main([__file__, "-vv"])

