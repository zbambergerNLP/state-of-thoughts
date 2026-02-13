"""
Helpers for converting `dspy.Tool` definitions into OpenAI-compatible tool schemas.

This module focuses on producing JSON Schema that is accepted by OpenAI-style tool calling,
including support for discrete-choice parameters via JSON Schema `enum`.
"""

# Standard library imports
import enum
import inspect
import types
from typing import Any, Literal, Union, get_args, get_origin

# Third-party imports
import dspy

JsonSchema = dict[str, Any]


def _json_schema_for_annotation(annotation: Any) -> JsonSchema:
	"""
	Convert a Python type annotation into a JSON Schema fragment.

	Supports:
	- Primitive types: str, int, float, bool
	- `typing.Literal[...]` for discrete-choice (enum-like) parameters
	- `enum.Enum` (including `enum.StrEnum`) for discrete-choice parameters
	- Containers: list[T], tuple[T], set[T]
	- Unions via `anyOf` (including Optional[T])

    Examples:
        _json_schema_for_annotation(str) -> {"type": "string"}
        _json_schema_for_annotation(int) -> {"type": "integer"}
        _json_schema_for_annotation(float) -> {"type": "number"}
        _json_schema_for_annotation(bool) -> {"type": "boolean"}
        _json_schema_for_annotation(Literal[1, 2, 3]) -> {"type": "integer", "enum": [1, 2, 3]}
        _json_schema_for_annotation(_Color) -> {"type": "string", "enum": ["red", "blue"]}
        _json_schema_for_annotation(list[int]) -> {"type": "array", "items": {"type": "integer"}}

	Args:
		annotation: The Python type annotation to convert.

	Returns:
		A JSON Schema fragment suitable for use under `properties`.
	"""
	if annotation is None:
		return {}

	if annotation is type(None):  # noqa: E721
		return {"type": "null"}

	if annotation is Any:
		return {}

	origin = get_origin(annotation)
	args = get_args(annotation)

	# typing.Literal -> enum
	if origin is Literal:
		values = list(args)
		schema: JsonSchema = {"enum": values}
		value_types = {type(v) for v in values if v is not None}
		if len(value_types) == 1:
			(t,) = tuple(value_types)
			if t is str:
				schema["type"] = "string"
			elif t is int:
				schema["type"] = "integer"
			elif t is float:
				schema["type"] = "number"
			elif t is bool:
				schema["type"] = "boolean"
		return schema

	# enum.Enum -> enum
	if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
		values = [member.value for member in annotation]  # type: ignore[misc]
		schema = {"enum": values}
		value_types = {type(v) for v in values if v is not None}
		if len(value_types) == 1:
			(t,) = tuple(value_types)
			if t is str:
				schema["type"] = "string"
			elif t is int:
				schema["type"] = "integer"
			elif t is float:
				schema["type"] = "number"
			elif t is bool:
				schema["type"] = "boolean"
		return schema

	# Union / Optional -> anyOf
	if origin in (Union, types.UnionType):
		return {"anyOf": [_json_schema_for_annotation(a) for a in args]}

	# Containers
	if origin in (list, tuple, set):
		item_schema = _json_schema_for_annotation(args[0]) if args else {}
		return {"type": "array", "items": item_schema}

	# Dict-like
	if origin is dict:
		return {"type": "object"}

	# Primitives
	if annotation is str:
		return {"type": "string"}
	if annotation is int:
		return {"type": "integer"}
	if annotation is float:
		return {"type": "number"}
	if annotation is bool:
		return {"type": "boolean"}

	# Fallback: unknown/custom object
	return {}


def _is_optional_annotation(annotation: Any) -> bool:
	"""
	Check if an annotation is Optional-like (i.e., Union[..., None]).

	Args:
		annotation: The type annotation to inspect.

	Returns:
		True if `None` is allowed by the annotation, else False.
	"""
	origin = get_origin(annotation)
	args = get_args(annotation)
	if origin not in (Union, types.UnionType):
		return False
	return any(a is type(None) for a in args)  # noqa: E721


def format_dspy_tool_as_openai_tool(tool: dspy.Tool) -> dict[str, Any]:
	"""
	Format a `dspy.Tool` into an OpenAI-compatible tool spec with JSON Schema parameters.

	This is similar in spirit to DSPy's `format_as_litellm_function_call`, but supports
	OpenAI-style JSON Schema features like `enum` for discrete-choice parameters.

	Args:
		tool: The DSPy tool to format.

	Returns:
		An OpenAI-compatible tool spec of the form:
		{
			"type": "function",
			"function": {
				"name": ...,
				"description": ...,
				"parameters": {
					"type": "object",
					"properties": {...},
					"required": [...],
					"additionalProperties": false
				}
			}
		}
	"""
	arg_descs: dict[str, str] = getattr(tool, "args", {}) or {}
	arg_types: dict[str, Any] = getattr(tool, "arg_types", {}) or {}

	sig = None
	try:
		sig = inspect.signature(tool.func)
	except Exception:  # pragma: no cover - best effort
		sig = None

	properties: dict[str, Any] = {}
	required: list[str] = []

	for arg_name, arg_desc in arg_descs.items():
		annotation = arg_types.get(arg_name, Any)
		prop_schema = _json_schema_for_annotation(annotation)
		prop_schema = {"description": arg_desc, **prop_schema} if arg_desc else prop_schema
		properties[arg_name] = prop_schema

		is_required = True
		if sig is not None and arg_name in sig.parameters:
			param = sig.parameters[arg_name]
			if param.default is not inspect._empty:
				is_required = False
		if _is_optional_annotation(annotation):
			is_required = False
		if is_required:
			required.append(arg_name)

	function_block = {
		"name": getattr(tool, "name", ""),
		"description": getattr(tool, "desc", "") or "",
		"parameters": {
			"type": "object",
			"properties": properties,
			"required": required,
			"additionalProperties": False,
		},
	}
	return {"type": "function", "function": function_block}


