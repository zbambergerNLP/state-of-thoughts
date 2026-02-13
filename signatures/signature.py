"""A reasoning-based `Signature` class for DSPy (with reasoning field support).

You typically subclass a Signature class, like this:

    class MySignature(dspy.Signature):
        input: str = InputField(desc="...")
        output: int = OutputField(desc="...")

For reasoning-based signatures (i.e., `ReasoningSignature`), you can include
intermediate reasoning fields:

    class ReasoningSignature(ReasoningSignature):
        question: str = InputField(desc="...")
        reasoning_step: str = ReasoningField(desc="...")
        answer: str = OutputField(desc="...")

You can create signature types using string notation with arrows:
- Single arrow for direct input->output: "input1, input2 -> output1, output2"
- Double arrow for reasoning flow: "input1, input2 -> reasoning_step -> output1, output2"

These string-based signatures can also be "hard-typed" as follows:
- Single arrow for direct input->output: "input1: Type1, input2: Type2 -> output1: Type3, output2: Type4"
- Double arrow for reasoning flow: "input1: Type1, input2: Type2 -> reasoning_step: Type3 -> output1: Type4, output2: Type5"

Examples:
    # Direct signature
    ReasoningSignature("question: str, context: str -> answer: str")

    # Reasoning signature
    ReasoningSignature("question: str -> reasoning_step: str -> answer: str")

    # With instructions
    ReasoningSignature("input: Type1 -> reasoning_step: Type2 -> output: Type3", "Think step by step")

For programmatic creation, use the make_reasoning_signature function which provides more control.
If you are not sure if your input is a string representation or a signature class,
you can use the ensure_reasoning_signature function.

For compatibility with the legacy dsp format, you can use the signature_to_template function.
"""

import ast
import inspect
import types
import typing
from copy import deepcopy
from typing import Any

from dspy.signatures.signature import _parse_type_node, infer_prefix
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from misc_utils import parse_base_signature, parse_reasoning_signature
from signatures.field import InputField, OutputField, ReasoningField
from tree.tree_constants import ReasoningState

DEFAULT_REASONING_FIELD = ReasoningField(
	desc="A single reasoning step towards solving the provided task."
)


def _convert_fields_to_signature_format(
	fields: dict[str, FieldInfo],
) -> dict[str, tuple[type, FieldInfo]]:
	"""
	Convert Pydantic model fields to the format expected by make_reasoning_signature.

	This function takes a dictionary of field names to Pydantic FieldInfo objects
	and converts them into a dictionary where each key is the field name and the value
	is a tuple containing the field type and the FieldInfo object.

	Args:
	    fields: A dictionary mapping field names to Pydantic FieldInfo objects.

	Returns:
	    A dictionary mapping field names to tuples of (field type, FieldInfo). If
	    a field does not have an explicit type annotation, it defaults to `str`.
	"""
	result: dict[str, tuple[type, FieldInfo]] = {}
	for name, field in fields.items():
		field_type = field.annotation if field.annotation is not None else str
		result[name] = (field_type, field)
	return result


def _default_instructions(cls) -> str:
	"""Generate default instructions for a Signature class based on its input, reasoning, and output fields.

	This function creates a standard instruction template that describes what the signature
	expects as input and what it should produce as output (and perhaps intermediate reasoning steps).
	This default instruction is used when no explicit instructions are provided for a signature class.

	Args:
	    cls: A ReasoningSignature class with input_fields and output_fields properties (and perhaps
	            reasoning_fields).

	Returns:
	    A string containing default instructions describing the input-to-output transformation.
	"""
	input_fields = cls.input_fields
	reasoning_fields = cls.reasoning_fields
	output_fields = cls.output_fields

	assert len(input_fields) > 0, "Signature must have at least one input field."
	assert len(output_fields) > 0, "Signature must have at least one output field."
	if reasoning_fields:
		formatted_inputs, formatted_reasoning_fields, formatted_outputs = (
			parse_reasoning_signature(
				input_field_names=list(input_fields.keys()),
				reasoning_field_names=list(reasoning_fields.keys()),
				output_field_names=list(output_fields.keys()),
			)
		)
		return f"Given the fields {formatted_inputs}, produce a sequence of {formatted_reasoning_fields}, and finally produce the fields {formatted_outputs}."
	else:
		formatted_inputs, formatted_outputs = parse_base_signature(
			input_field_names=list(input_fields.keys()),
			output_field_names=list(output_fields.keys()),
		)
		return f"Given the fields {formatted_inputs}, produce the fields {formatted_outputs}."


class ReasoningSignatureMeta(type(BaseModel)):
	def __call__(cls, *args, **kwargs):
		"""
		Custom __call__ method to handle the creation of Signature instances.

		If the class is Signature, it creates a new Signature class with the provided
		arguments and keyword arguments. This allows for dynamic creation of Signature
		instances at runtime.

		Args:
			*args: Positional arguments to pass to the Signature class.
			**kwargs: Keyword arguments to pass to the Signature class.

		Returns:
			A new Signature class instance or a dynamically created Signature class.
		"""
		if cls is ReasoningSignature:
			# We don't create an actual Signature instance, instead, we create a new Signature class.
			custom_types = kwargs.pop("custom_types", None)
			if custom_types is None and args and isinstance(args[0], str):
				custom_types = cls._detect_custom_types_from_caller(args[0])
			return make_reasoning_signature(*args, custom_types=custom_types, **kwargs)
		return super().__call__(*args, **kwargs)

	def __new__(
		mcs: type["ReasoningSignatureMeta"],
		signature_name: str,
		bases: tuple[type, ...],
		namespace: dict[str, Any],
		**kwargs: Any,
	) -> type["ReasoningSignature"]:
		"""
		Custom __new__ method to create a new Signature class with the specified fields and instructions.

		This method handles the ordering of fields, setting default types, and ensuring
		that all fields are declared with InputField, OutputField, or ReasoningField.
		        It also sets the class docstring to the provided instructions or defaults to a basic description.

		Args:
		                mcs: The metaclass itself.
		                signature_name: The name of the signature class to create.
		                bases: The base classes for the signature class.
		                namespace: The namespace containing the class attributes and methods.
		                **kwargs: Additional keyword arguments to pass to the class creation.

		        Returns:
		                A new Signature class with the specified fields and instructions.
		"""
		# At this point, the orders have been swapped already.
		field_order = [
			name for name, value in namespace.items() if isinstance(value, FieldInfo)
		]
		raw_annotations = namespace.get("__annotations__", {})
		for name, field in namespace.items():
			if not isinstance(field, FieldInfo):
				continue  # Don't add types to non-field attributes
			if not name.startswith("__") and name not in raw_annotations:
				raw_annotations[name] = str
		ordered_annotations = {
			name: raw_annotations[name]
			for name in field_order
			if name in raw_annotations
		}
		ordered_annotations.update(
			{k: v for k, v in raw_annotations.items() if k not in ordered_annotations}
		)
		namespace["__annotations__"] = ordered_annotations

		# Let Pydantic do its thing
		cls = super().__new__(mcs, signature_name, bases, namespace, **kwargs)  # type: ignore

		# If we don't have instructions, it might be because we are a derived generic type.
		# In that case, we should inherit the instructions from the base class.
		if cls.__doc__ is None:
			for base in bases:
				if isinstance(base, ReasoningSignatureMeta):
					doc = getattr(base, "__doc__", "")
					if doc != "":
						cls.__doc__ = doc

		# Ensure all fields are declared with InputField or OutputField first
		cls._validate_fields()

		# The more likely case is that the user has just not given us a type.
		# In that case, we should default to the input/output format.
		# But only do this if the class actually has fields
		if cls.__doc__ is None and hasattr(cls, "model_fields") and cls.model_fields:
			cls.__doc__ = _default_instructions(cls)

		# Ensure all fields have a prefix
		for name, field in cls.__dict__.get("model_fields", {}).items():
			if hasattr(field, "json_schema_extra") and field.json_schema_extra:
				if "prefix" not in field.json_schema_extra:
					field.json_schema_extra["prefix"] = infer_prefix(name) + ":"
				if "desc" not in field.json_schema_extra:
					field.json_schema_extra["desc"] = f"${{{name}}}"
		return cls

	def _validate_fields(cls):
		model_fields = getattr(cls, "model_fields", {})
		for name, field in model_fields.items():
			if hasattr(field, "json_schema_extra") and field.json_schema_extra:
				extra = field.json_schema_extra or {}
				field_type = extra.get("__dspy_field_type")
				if field_type in ["input", "reasoning", "output"]:
					continue  # Valid DSPy field

			# If we get here, it's not a properly configured DSPy field
			raise TypeError(
				f"Field `{name}` in `{cls.__name__}` must be declared with InputField, OutputField, or ReasoningField, but "
				f"field `{name}` has `field.json_schema_extra={getattr(field, 'json_schema_extra', None)}`",
			)

	@property
	def instructions(cls) -> str:
		doc = inspect.cleandoc(getattr(cls, "__doc__", ""))
		if doc and doc != "Base signature class with no specific instructions.":
			return doc
		# If we have the placeholder message or no doc, regenerate instructions
		return _default_instructions(cls)

	@instructions.setter
	def instructions(cls, instructions: str) -> None:
		cls.__doc__ = instructions

	@property
	def input_fields(cls) -> dict[str, FieldInfo]:
		return cls._get_fields_with_type(ReasoningState.INPUT)

	@property
	def output_fields(cls) -> dict[str, FieldInfo]:
		return cls._get_fields_with_type(ReasoningState.OUTPUT)

	@property
	def reasoning_fields(cls) -> dict[str, FieldInfo]:
		return cls._get_fields_with_type(ReasoningState.REASONING)

	@property
	def fields(cls) -> dict[str, FieldInfo]:
		# Make sure to give input fields before reasoning fields before output fields
		return {**cls.input_fields, **cls.reasoning_fields, **cls.output_fields}

	@property
	def signature(cls) -> str:
		"""The string representation of the signature."""
		input_fields = ", ".join(cls.input_fields.keys())
		reasoning_fields = ", ".join(cls.reasoning_fields.keys())
		output_fields = ", ".join(cls.output_fields.keys())

		if reasoning_fields:
			return f"{input_fields} -> {reasoning_fields} -> {output_fields}"
		else:
			return f"{input_fields} -> {output_fields}"

	def _get_fields_with_type(cls, field_type: str) -> dict[str, FieldInfo]:
		model_fields = getattr(cls, "model_fields", {})
		result = {}
		for k, v in model_fields.items():
			if hasattr(v, "json_schema_extra") and v.json_schema_extra:
				if v.json_schema_extra.get("__dspy_field_type") == field_type:
					result[k] = v
		return result

	def __repr__(cls):
		"""Output a representation of the signature.

		Uses the form:
		Signature(question, context -> answer
		    question: str = InputField(desc="..."),
		    context: List[str] = InputField(desc="..."),
		    answer: int = OutputField(desc="..."),
		).
		"""
		field_reprs = []
		for name, field in cls.fields.items():
			field_reprs.append(f"{name} = Field({field})")
		field_repr = "\n\t".join(field_reprs)
		return f"{cls.__name__}({cls.signature}\n    instructions={cls.instructions!r}\n    {field_repr}\n)"


class ReasoningSignature(BaseModel, metaclass=ReasoningSignatureMeta):
	# Note: Don't put a docstring here, as it will become the default instructions
	# for any signature that doesn't define it's own instructions.

	@classmethod
	def with_instructions(cls, instructions: str) -> type["ReasoningSignature"]:
		return make_reasoning_signature(
			_convert_fields_to_signature_format(cls.fields), instructions
		)

	@classmethod
	def with_updated_fields(
		cls, name: str, type_: type | None = None, **kwargs: Any
	) -> type["ReasoningSignature"]:
		"""Create a new Signature class with the updated field information.

		Returns a new Signature class with the field, name, updated
		with fields[name].json_schema_extra[key] = value.

		Args:
		    name: The name of the field to update.
		    type_: The new type of the field.
		    **kwargs: The new values for the field.

		Returns:
		    A new Signature class (not an instance) with the updated field information.
		"""
		fields_copy = deepcopy(cls.fields)
		if type_ is not None:
			fields_copy[name].annotation = type_

		new_signature = make_reasoning_signature(
			_convert_fields_to_signature_format(fields_copy), cls.instructions
		)
		# Apply the requested updates directly on the newly created class so that
		# downstream callers (and reprs/tests) observe the latest metadata.
		if kwargs:
			field = new_signature.model_fields[name]
			existing_extra = dict(field.json_schema_extra or {})
			field.json_schema_extra = {**existing_extra, **kwargs}
			if "desc" in kwargs:
				field.description = kwargs["desc"]
		return new_signature

	@classmethod
	def prepend(
		cls, name: str, field: FieldInfo, type_: type | None = None
	) -> type["ReasoningSignature"]:
		return cls.insert(0, name, field, type_)

	@classmethod
	def append(
		cls, name: str, field: FieldInfo, type_: type | None = None
	) -> type["ReasoningSignature"]:
		return cls.insert(-1, name, field, type_)

	@classmethod
	def delete(cls, name: str) -> type["ReasoningSignature"]:
		fields = dict(cls.fields)
		fields.pop(name, None)
		return make_reasoning_signature(
			_convert_fields_to_signature_format(fields), cls.instructions
		)

	@classmethod
	def insert(
		cls, index: int, name: str, field: FieldInfo, type_: type | None = None
	) -> type["ReasoningSignature"]:
		# It's possible to set the type as annotation=type in pydantic.Field(...)
		# But this may be annoying for users, so we allow them to pass the type
		if type_ is None:
			type_ = field.annotation
		if type_ is None:
			type_ = str

		input_fields = list(cls.input_fields.items())
		reasoning_fields = list(cls.reasoning_fields.items())
		output_fields = list(cls.output_fields.items())

		# Choose the list to insert into based on the field type
		extra = field.json_schema_extra
		if callable(extra) or extra is None:
			extra = {}
		field_type = extra.get("__dspy_field_type")
		if field_type == "input":
			lst = input_fields
		elif field_type == "reasoning":
			lst = reasoning_fields
		else:  # output
			lst = output_fields

		# We support negative insert indices
		if index < 0:
			index += len(lst) + 1
		if index < 0 or index > len(lst):
			raise ValueError(
				f"Invalid index to insert: {index}, index must be in the range of [{len(lst) - 1}, {len(lst)}] for "
				f"{field_type} fields, but received: {index}.",
			)
		lst.insert(index, (name, field))

		# Reconstruct all fields
		new_fields = {}
		for field_name, field_obj in input_fields:
			new_fields[field_name] = field_obj
		for field_name, field_obj in reasoning_fields:
			new_fields[field_name] = field_obj
		for field_name, field_obj in output_fields:
			new_fields[field_name] = field_obj

		return make_reasoning_signature(
			_convert_fields_to_signature_format(new_fields), cls.instructions
		)

	@classmethod
	def equals(cls, other) -> bool:
		"""Compare the JSON schema of two Signature classes."""
		if not isinstance(other, type) or not issubclass(other, BaseModel):
			return False

		# Check if other is also a Signature class with the same attributes
		if not issubclass(other, ReasoningSignature):
			return False

		# Create instances to access model fields properly
		try:
			cls_instructions = getattr(cls, "__doc__", "") or ""
			other_instructions = getattr(other, "__doc__", "") or ""
			if cls_instructions != other_instructions:
				return False

			# Compare model fields using model_fields
			cls_fields = cls.model_fields
			other_fields = other.model_fields

			for name in cls_fields.keys() | other_fields.keys():
				if name not in other_fields or name not in cls_fields:
					return False
				cls_extra = cls_fields[name].json_schema_extra or {}
				other_extra = other_fields[name].json_schema_extra or {}
				if cls_extra != other_extra:
					return False
			return True
		except AttributeError:
			return False

	@classmethod
	def dump_state(cls):
		state = {"instructions": cls.instructions, "fields": []}
		for field_name in cls.fields:
			field = cls.fields[field_name]
			extra = field.json_schema_extra
			if callable(extra):
				extra = {}
			elif extra is None:
				extra = {}
			state["fields"].append(
				{
					"prefix": extra.get("prefix", ""),
					"description": extra.get("desc", ""),
				}
			)
		return state

	@classmethod
	def load_state(cls, state):
		fields_copy = deepcopy(cls.fields)
		signature_copy = make_reasoning_signature(
			_convert_fields_to_signature_format(fields_copy),
			instructions=state.get("instructions", cls.instructions),
		)
		signature_copy.instructions = state["instructions"]
		for field, saved_field in zip(
			fields_copy.values(), state["fields"], strict=False
		):
			extra = field.json_schema_extra or {}
			if callable(extra):
				extra = {}
			elif extra is None:
				extra = {}
			extra["prefix"] = saved_field.get("prefix")
			extra["desc"] = saved_field.get("description")
			field.json_schema_extra = extra
		return signature_copy

		# For now, just return a copy since proper state loading is complex
		return cls


def ensure_reasoning_signature(
	signature: str | type[ReasoningSignature], instructions: str | None = None
) -> type[ReasoningSignature]:
	"""Ensure that the input is a ReasoningSignature class, converting if necessary.

	This is a convenience function that accepts either a string representation of a signature
	or an existing ReasoningSignature class, and returns a ReasoningSignature class. This is
	useful when you're not sure whether you have a string or a signature class but need to
	work with a signature class.

	Args:
	    signature: Either a string signature (e.g., "input -> output" or "input -> reasoning -> output"),
	        a ReasoningSignature class, or a dspy.Signature class.
	    instructions: Optional instructions to use if creating from a string.
	        Cannot be specified if signature is already a ReasoningSignature class.

	Returns:
	    A ReasoningSignature class.

	Raises:
	    ValueError: If instructions are provided when signature is already a ReasoningSignature class.

	Examples:
	    ```
	    # Convert string to signature
	    sig1 = ensure_reasoning_signature("question -> answer")

	    # Pass through existing signature
	    sig2 = ensure_reasoning_signature(ExistingSignature)

	    # Handle None case
	    sig3 = ensure_reasoning_signature(None)  # Returns None

	    # With instructions for string signature
	    sig4 = ensure_reasoning_signature("input -> output", "Think carefully.")
	    ```
	"""
	assert signature is not None, "Signature cannot be None."
	if isinstance(signature, str):
		return make_reasoning_signature(signature, instructions)
	if instructions is not None:
		raise ValueError(
			"Don't specify instructions when initializing with a ReasoningSignature"
		)
	return signature


def make_reasoning_signature(
	signature: str | dict[str, tuple[type, FieldInfo]],
	instructions: str | None = None,
	signature_name: str = "StringReasoningSignature",
	custom_types: dict[str, type] | None = None,
) -> type[ReasoningSignature]:
	"""Create a new Signature subclass with the specified fields and instructions.

	This function supports both traditional input->output signatures and reasoning-based
	signatures with intermediate reasoning fields. Use single arrow notation for direct
	input->output flow, or double arrow notation for input->reasoning->output flow.

	Args:
	    signature: Either a string in the format:
	        - "input1, input2 -> output1, output2" (direct flow)
	        - "input1, input2 -> reasoning -> output1, output2" (reasoning flow)
	        or a dictionary mapping field names to tuples of (type, FieldInfo).
	    instructions: Optional string containing instructions/prompt for the signature.
	        If not provided, defaults to a basic description of inputs and outputs.
	    signature_name: Optional string to name the generated Signature subclass.
	        Defaults to "StringSignature".
	    custom_types: Optional dictionary mapping type names to their actual type objects.
	        Useful for resolving custom types that aren't built-ins or in the typing module.

	Returns:
	    A new signature class with the specified fields and instructions.

	Examples:

	```
	# Traditional input->output signature
	sig1 = make_reasoning_signature("question, context -> answer")

	# Reasoning-based signature with intermediate reasoning
	sig2 = make_reasoning_signature("question -> reasoning -> answer")

	# Multiple inputs and reasoning fields
	sig3 = make_reasoning_signature("context, question -> analysis, reasoning -> answer")

	# Using dictionary format with reasoning fields
	sig4 = make_reasoning_signature({
	    "question": (str, InputField()),
	    "reasoning": (str, ReasoningField()),
	    "answer": (str, OutputField())
	})

	# Using custom types
	class MyType:
	    pass

	sig5 = make_reasoning_signature("input: MyType -> reasoning -> output", custom_types={"MyType": MyType})
	```
	"""
	# Prepare the names dictionary for type resolution
	names = None
	if custom_types:
		names = dict(typing.__dict__)
		names.update(custom_types)

	fields = (
		_parse_signature(signature, names) if isinstance(signature, str) else signature
	)

	# Validate the fields, this is important because we sometimes forget the
	# slightly unintuitive syntax with tuples of (type, Field)
	fixed_fields = {}
	for name, type_field in fields.items():
		if not isinstance(name, str):
			raise ValueError(f"Field names must be strings, but received: {name}.")
		if isinstance(type_field, FieldInfo):
			type_ = type_field.annotation
			field = type_field
		else:
			if not isinstance(type_field, tuple):
				raise ValueError(
					f"Field values must be tuples, but received: {type_field}."
				)
			type_, field = type_field
		# It might be better to be explicit about the type, but it currently would break
		# program of thought and teleprompters, so we just silently default to string.
		if type_ is None:
			type_ = str
		if not isinstance(
			type_,
			type
			| getattr(typing, "_GenericAlias", type(None))
			| types.GenericAlias
			| typing._SpecialForm,
		):
			raise ValueError(
				f"Field types must be types, but received: {type_} of type {type(type_)}."
			)
		if not isinstance(field, FieldInfo):
			raise ValueError(
				f"Field values must be Field instances, but received: {field}."
			)
		fixed_fields[name] = (type_, field)

	# Default prompt when no instructions are provided
	if instructions is None:
		# Create a temporary signature to generate default instructions
		temp_sig = create_model(
			"TempReasoningSignature",
			__base__=ReasoningSignature,
			__doc__="",
			**fixed_fields,
		)
		instructions = _default_instructions(temp_sig)

	return create_model(
		signature_name,
		__base__=ReasoningSignature,
		__doc__=instructions,
		**fixed_fields,
	)


def _parse_signature(
	signature: str, names: dict[str, type] | None = None
) -> dict[str, tuple[type, FieldInfo]]:
	"""Parse a signature string into field definitions supporting reasoning flow.

	This function parses signature strings with either single arrow (input -> output)
	or double arrow (input -> reasoning -> output) notation. Fields are automatically
	assigned the appropriate field types based on their position in the signature.

	Args:
	    signature: A string representing the signature in one of these formats:
	        - "input1, input2 -> output1, output2" (single arrow: direct flow)
	        - "input1, input2 -> reasoning -> output1, output2" (double arrow: reasoning flow)
	    names: Optional dictionary for resolving custom type names.

	Returns:
	    Dictionary mapping field names to tuples of (type, FieldInfo) where:
	    - Input fields get InputField() instances
	    - Reasoning fields get ReasoningField() instances
	    - Output fields get OutputField() instances

	Raises:
	    ValueError: If signature format is invalid (wrong number of arrows, empty parts, etc.)

	Examples:
	    ```
	    # Single arrow - direct input to output
	    fields = _parse_signature("question -> answer")
	    # Returns: {"question": (str, InputField()), "answer": (str, OutputField())}

	    # Double arrow - input through reasoning to output
	    fields = _parse_signature("question -> reasoning -> answer")
	    # Returns: {"question": (str, InputField()), "reasoning": (str, ReasoningField()), "answer": (str, OutputField())}
	    ```
	"""
	arrow_count = signature.count("->")

	if arrow_count == 1:
		# Single arrow format: input -> output
		inputs_str, outputs_str = signature.split("->")
		inputs_str = inputs_str.strip()
		outputs_str = outputs_str.strip()

		# Check for empty parts
		if not inputs_str:
			raise ValueError(
				f"Invalid signature format: '{signature}', input part cannot be empty."
			)
		if not outputs_str:
			raise ValueError(
				f"Invalid signature format: '{signature}', output part cannot be empty."
			)

		fields = {}
		for field_name, field_type in _parse_field_string(inputs_str, names):
			fields[field_name] = (field_type, InputField())
		for field_name, field_type in _parse_field_string(outputs_str, names):
			fields[field_name] = (field_type, OutputField())

		return fields

	elif arrow_count == 2:
		# Two arrow format: input -> reasoning -> output
		parts = signature.split("->")
		if len(parts) != 3:
			raise ValueError(
				f"Invalid two-arrow signature format: '{signature}', expected 'input -> reasoning -> output'."
			)

		inputs_str, reasoning_str, outputs_str = [part.strip() for part in parts]

		# Check for empty parts
		if not inputs_str:
			raise ValueError(
				f"Invalid signature format: '{signature}', input part cannot be empty."
			)
		if not reasoning_str:
			raise ValueError(
				f"Invalid signature format: '{signature}', reasoning part cannot be empty."
			)
		if not outputs_str:
			raise ValueError(
				f"Invalid signature format: '{signature}', output part cannot be empty."
			)

		fields = {}
		for field_name, field_type in _parse_field_string(inputs_str, names):
			fields[field_name] = (field_type, InputField())
		for field_name, field_type in _parse_field_string(reasoning_str, names):
			fields[field_name] = (field_type, ReasoningField())
		for field_name, field_type in _parse_field_string(outputs_str, names):
			fields[field_name] = (field_type, OutputField())

		return fields

	else:
		raise ValueError(
			f"Invalid signature format: '{signature}', must contain exactly one '->' (input -> output) or two '->' (input -> reasoning -> output)."
		)


def _parse_field_string(
	field_string: str, names: dict[str, type] | None = None
) -> list[tuple[str, type]]:
	"""Extract field names and types from a comma-separated field string.

	This function parses field strings like "x: int, y: str" into a list of
	(field_name, field_type) tuples. It uses Python's AST to safely parse
	the field definitions and resolve type annotations.

	Args:
	    field_string: A comma-separated string of field definitions.
	        Examples: "x", "x: int", "x: int, y: str", "field1: List[str], field2: Optional[int]"
	    names: Optional dictionary for resolving custom type names.

	Returns:
	    List of tuples where each tuple is (field_name, field_type).
	    If no type annotation is provided, defaults to str.

	Examples:
	    ```
	    _parse_field_string("x") -> [("x", str)]
	    _parse_field_string("x: int, y: str") -> [("x", int), ("y", str)]
	    _parse_field_string("items: List[str]") -> [("items", list)]  # simplified
	    ```

	Note:
	    This function utilizes Python's AST to parse the fields and types safely.
	    Empty or whitespace-only strings return an empty list.
	"""

	# Handle empty or whitespace-only strings
	if not field_string or field_string.strip() == "":
		return []

	try:
		parsed = ast.parse(f"def f({field_string}): pass")
		func_def = parsed.body[0]
		if isinstance(func_def, ast.FunctionDef) and hasattr(func_def.args, "args"):
			args = func_def.args.args
		else:
			return []
	except (AttributeError, SyntaxError):
		# Handle case where AST parsing fails or structure is unexpected
		return []
	field_names = [arg.arg for arg in args]
	types = [
		str if arg.annotation is None else _parse_type_node(arg.annotation, names)
		for arg in args
	]
	return list(zip(field_names, types, strict=False))
