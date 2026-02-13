# Standard library imports
from typing import Literal

# Third-party imports
from pydantic import BaseModel, Field, ValidationInfo, field_validator

# Local imports
from signatures.field_constants import DEFAULT_REASONING_FIELD_NAME

# Type alias for granularity of response length constraints
GranularityType = Literal["word", "sentence", "paragraph"]


class ResponseLength(BaseModel):
	"""A class to represent the constraints for the length of a text response."""

	granularity: GranularityType = Field(
		description="The unit of measurement for the response length. Default is `'sentence'`.",
		default="sentence",
	)
	bounds: tuple[int | None, int | None] = Field(
		description="""
The bounds for the response length, corresponding to `minimum` and `maximum` respectively.
When applied to a paragraph granularity, for example, the output must have between `minimum`
and `maximum` paragraphs. If `minimum` is `None`, there is no lower bound. If `maximum` is `None`,
there is no upper bound. Default is `(None, None)`.
""".strip(),
		default=(None, None),
	)

	@field_validator("bounds")
	@classmethod
	def validate_bounds_difference(
		cls,
		bounds: tuple[int | None, int | None],
		info: ValidationInfo,
	) -> tuple[int | None, int | None]:
		"""
		Validates bounds constraints:
		- Lower bound must be greater than 0
		- Upper bound must be greater or equal to lower bound
		- If bounds are different, the difference must meet minimum requirements per granularity

		Parameters:
			bounds (Tuple[Optional[int], Optional[int]]): The lower and upper bounds to validate.
			info (ValidationInfo): Validation information containing the granularity type

		Returns:
			bounds (Tuple[int, int]): The lower and upper bounds to validate
		"""
		min_length, max_length = bounds
		assert min_length is None or min_length > 0, (
			f"Lower bound must be either `None` or greater than 0, but got {min_length}"
		)
		assert max_length is None or (
			max_length >= min_length if min_length is not None else max_length > 0
		), (
			f"Upper bound ({max_length}) must be either `None`, or greater than or equal to lower bound ({min_length}).\n"
			"If the lower bound is `None`, the upper bound must be greater than 0."
		)
		return bounds

	def __str__(self) -> str:
		"""Create a string representation of the ResponseLength object."""
		# After validation, self.bounds is always Tuple[Optional[int], Optional[int]]
		min_b, max_b = self.bounds
		granularity_val = self.granularity
		if min_b is not None and max_b is not None:
			if min_b == max_b:
				s_suffix = "" if min_b == 1 else "s"
				return f"Exactly {min_b} {granularity_val}{s_suffix}"
			else:
				return f"Between {min_b} and {max_b} {granularity_val}s"
		elif min_b is not None and max_b is None:  # At least min_b
			s_suffix = "" if min_b == 1 else "s"
			return f"At least {min_b} {granularity_val}{s_suffix}"
		elif min_b is None and max_b is not None:  # At most max_b
			s_suffix = "" if max_b == 1 else "s"
			return f"At most {max_b} {granularity_val}{s_suffix}"
		else:  # min_b is None and max_b is None
			return f"Any number of {granularity_val}s"


def format_thought_length_instruction(
	use_internal_reasoning_for_thought_generation: bool = False,
	thought_length: ResponseLength | None = None,
	reasoning_field_name: str = DEFAULT_REASONING_FIELD_NAME,
) -> str:
	"""
	Format the thought length instruction for the system message.

	NOTE: If `thought_length` is `None`, no instruction will be added.

	Args:
	    use_internal_reasoning_for_thought_generation: Whether to use chain-of-thought for
			generating thoughts.
	    thought_length: The thought length constraints
	    reasoning_field_name: Name of the reasoning field

	Returns:
	    A string that contains the thought length instruction.
	"""
	if thought_length is None:
		return ""
	thought_length_string = str(thought_length)
	modified_thought_length_string = (
		thought_length_string[0].lower() + thought_length_string[1:]
	)
	thought_length_instruction = (
		f"Each `{reasoning_field_name}` should be {modified_thought_length_string}."
	)
	if use_internal_reasoning_for_thought_generation and thought_length is not None:
		granularity_value = thought_length.granularity
		thought_length_instruction += f" NOTE: This {granularity_value} limit does not include internal reasoning."
	return thought_length_instruction


def format_response_length_instruction(
	response_length: ResponseLength | None = None,
) -> str:
	"""
	Format the response length instruction for the system message.

	Args:
	    response_length: The response length constraints

	Returns:
	    A string that contains the response length instruction.
	"""
	if response_length is None:
		return ""
	response_length_string = str(response_length)
	modified_response_length_string = (
		response_length_string[0].lower() + response_length_string[1:]
	)
	response_length_instruction = (
		f"Your final answer should be {modified_response_length_string}."
	)
	return response_length_instruction
