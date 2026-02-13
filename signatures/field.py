"""
Field classes for DSPy signatures.

Since we are supporting ReasoningSignatures, we need to extend the field classes to support the
reasoning field type.

Moreover, we add support for rubric_weight for evaluator dimensions (i.e., assigning importance
weights to the dimensions [fields] of evaluator signatures).
"""

# Third-party imports
import pydantic
from dspy.signatures.field import DSPY_FIELD_ARG_NAMES

# Extend DSPy field arguments to include rubric_weight for evaluator dimensions
DSPY_FIELD_ARG_NAMES = list(DSPY_FIELD_ARG_NAMES) + ["rubric_weight"]

PYDANTIC_CONSTRAINT_MAP = {
	"gt": "greater than ",
	"ge": "greater than or equal to ",
	"lt": "less than ",
	"le": "less than or equal to ",
	"min_length": "minimum length ",
	"max_length": "maximum length ",
	"multiple_of": "a multiple of ",
	"allow_inf_nan": "allows 'inf', '-inf', and 'nan' values ",
}



def move_kwargs(**kwargs):
	"""
	Move keyword arguments to pydantic.Field while preserving DSPy-specific metadata.

	NOTE: We overrided this function to allow for a revised constraint representation.

	Pydantic doesn't allow arbitrary arguments to be given to fields,
	but asks that
	> any extra data you want to add to the JSON schema should be passed
	> as a dictionary to the json_schema_extra keyword argument.
	See: https://docs.pydantic.dev/2.6/migration/#changes-to-pydanticfield

	Args:
		**kwargs: Keyword arguments to be moved to pydantic.Field.

	Returns:
		A dictionary of keyword arguments for initializing a `pydantic.Field`.
	"""
	pydantic_kwargs = {}
	json_schema_extra = {}
	for k, v in kwargs.items():
		if k in DSPY_FIELD_ARG_NAMES:
			json_schema_extra[k] = v
		else:
			pydantic_kwargs[k] = v
	if (
		"description" in kwargs
		and "desc" not in json_schema_extra
	):
		json_schema_extra["desc"] = kwargs["description"]
	constraints = _translate_pydantic_field_constraints(**kwargs)
	if constraints:
		json_schema_extra["constraints"] = constraints
	pydantic_kwargs["json_schema_extra"] = json_schema_extra
	return pydantic_kwargs


def _translate_pydantic_field_constraints(**kwargs) -> str:
	"""
	Extracts Pydantic constraints and translates them into human-readable format.

	Args:
		**kwargs: Keyword arguments that may include Pydantic constraints like `gt`, `ge`, `lt`, etc...

	Returns:
		A string that describes the constraints in a human-readable format. If no constraints are provided,
		returns an empty string.
	"""

	constraints = []
	# Iterate through the constraint map to ensure consistent order
	for key in PYDANTIC_CONSTRAINT_MAP:
		if key in kwargs:
			value = kwargs[key]
			constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")
	# Join constraints with proper English grammar
	if len(constraints) == 0:
		return ""
	elif len(constraints) == 1:
		return constraints[0]
	elif len(constraints) == 2:
		return f"{constraints[0]} and {constraints[1]}"
	else:
		# For more than 2 constraints: "constraint1, constraint2, and constraint3"
		return ", ".join(constraints[:-1]) + f", and {constraints[-1]}"


def InputField(**kwargs):  # noqa: N802
	"""
	Create a pydantic field for the input of a DSPy signature.

	This function is a wrapper around pydantic.Field that adds additional metadata
	to indicate that the field is an input field in a DSPy signature.

	Args:
		**kwargs: Additional keyword arguments to pass to pydantic.Field.
			These can include standard pydantic field arguments like `default`, `alias`, etc.,
			as well as DSPy-specific arguments like `desc`, `prefix`, and `format`.

	Returns:
		A pydantic Field object with additional metadata indicating it is an input field.
	"""
	return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def ReasoningField(**kwargs):  # noqa: N802
	"""
	Create a pydantic field for reasoning steps in a DSPy signature.

	This function is a wrapper around pydantic.Field that adds additional metadata
	to indicate that the field is used for reasoning steps in a DSPy signature.

	Args:
		**kwargs: Additional keyword arguments to pass to pydantic.Field.
			These can include standard pydantic field arguments like `default`, `alias`, etc.,
			as well as DSPy-specific arguments like `desc`, `prefix`, and `format`.

	Returns:
		A pydantic Field object with additional metadata indicating it is a reasoning field.
	"""
	return pydantic.Field(
		**move_kwargs(**kwargs, __dspy_field_type="reasoning")
	)


def OutputField(**kwargs):  # noqa: N802
	"""
	Create a pydantic field for the output of a DSPy signature.

	This function is a wrapper around pydantic.Field that adds additional metadata
	to indicate that the field is an output field in a DSPy signature.

	Args:
		**kwargs: Additional keyword arguments to pass to pydantic.Field.
			These can include standard pydantic field arguments like `default`, `alias`, etc.,
			as well as DSPy-specific arguments like `desc`, `prefix`, and `format`.

	Returns:
		A pydantic Field object with additional metadata indicating it is an output field.
	"""
	return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))


def new_to_old_field(field):
	"""Convert new pydantic Field to old field format."""
	field_type = field.json_schema_extra["__dspy_field_type"]

	if field_type == "input":
		return OldInputField(
			prefix=field.json_schema_extra.get("prefix"),
			desc=field.json_schema_extra.get("desc"),
			format=field.json_schema_extra.get("format"),
		)
	elif field_type == "reasoning":
		return OldReasoningField(
			prefix=field.json_schema_extra.get("prefix"),
			desc=field.json_schema_extra.get("desc"),
			format=field.json_schema_extra.get("format"),
		)
	elif field_type == "output":
		return OldOutputField(
			prefix=field.json_schema_extra.get("prefix"),
			desc=field.json_schema_extra.get("desc"),
			format=field.json_schema_extra.get("format"),
		)
	else:
		raise ValueError(f"Unknown field type: {field_type}")


class OldField:
	"""A more ergonomic datatype that infers prefix and desc if omitted."""

	def __init__(self, *, prefix=None, desc=None, input, format=None):
		self.prefix = prefix  # This can be None initially and set later
		self.desc = desc
		self.format = format

	def finalize(self, key, inferred_prefix):
		"""Set the prefix if it's not provided explicitly."""
		if self.prefix is None:
			self.prefix = inferred_prefix + ":"

		if self.desc is None:
			self.desc = f"${{{key}}}"

	def __repr__(self):
		return f"{self.__class__.__name__}(prefix={self.prefix}, desc={self.desc})"

	def __eq__(self, __value: object) -> bool:
		return self.__dict__ == __value.__dict__


class OldInputField(OldField):
	def __init__(self, *, prefix=None, desc=None, format=None):
		super().__init__(prefix=prefix, desc=desc, input=True, format=format)


class OldOutputField(OldField):
	def __init__(self, *, prefix=None, desc=None, format=None):
		super().__init__(prefix=prefix, desc=desc, input=False, format=format)


class OldReasoningField(OldField):
	def __init__(self, *, prefix=None, desc=None, format=None):
		super().__init__(prefix=prefix, desc=desc, input=False, format=format)
