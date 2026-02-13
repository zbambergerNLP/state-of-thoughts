# Tests utility functions in the adapter/utils.py library.

# Standard library imports
from typing import Literal

# Third-party imports
import pytest

# Local imports
from adapter.prompts import (
	FINAL_OUTPUT_CONCLUSION,
	FINAL_OUTPUT_SYNTHESIS_FAITHFUL,
	FINAL_OUTPUT_SYNTHESIS_RESTRUCTURED,
	FINAL_OUTPUT_SYNTHESIS_STRICT,
)
from adapter.utils import (
	format_field_description,
	get_field_description_string,
	get_final_output_description,
	normalize_numeric_field_value,
)
from signatures import (
	InputField,
	OutputField,
	ReasoningField,
	ReasoningSignature,
	ensure_reasoning_signature,
)


class SolveMathProblem(ReasoningSignature):
	"""
	Solve the provided math problem and return its answer.
	"""

	math_problem: str = InputField(desc="The math problem to solve")
	answer: str = OutputField(desc="The answer to the math problem")


class GenerateArgument(ReasoningSignature):
	"""
	Generate an argument which takes the provided stance towards the provided topic.
	"""

	topic: str = InputField(desc="The topic to generate an argument about")
	stance: Literal["PRO", "ANTI"] = InputField(desc="The stance to take on the topic")
	argument: str = OutputField(desc="The generated argument")


class ValidatedInputSignature(ReasoningSignature):
	"""
	Signature with constrained input fields to test bulletized constraint display.
	"""

	user_input: str = InputField(
		desc="User input with validation constraints",
		min_length=5,
		max_length=200,
		gt=0,
	)
	response: str = OutputField(desc="Generated response")


class MultiConstraintSignature(ReasoningSignature):
	"""
	Signature with multiple constrained fields.
	"""

	number_input: int = InputField(
		desc="A number with range constraints", ge=1, le=100, multiple_of=5
	)
	text_input: str = InputField(
		desc="Text with length constraints", min_length=10, max_length=500
	)
	result: str = OutputField(
		desc="Result with constraints", min_length=1, max_length=1000
	)


class MultiOutputConstraintsSignature(ReasoningSignature):
	"""
	Signature with multiple constrained output fields to test bulletized constraint display.
	"""

	task: str = InputField(desc="The task to complete")
	summary: str = OutputField(
		desc="A summary of the task completion", min_length=10, max_length=200
	)
	score: int = OutputField(desc="A numerical score for the task", ge=0, le=100)
	details: str = OutputField(
		desc="Detailed explanation with length constraints",
		min_length=50,
		max_length=1000,
		gt=0,
	)


class MathWithReasoningSignature(ReasoningSignature):
	"""
	Solve a math problem by performing a series of math operations (e.g., addition, subtraction, etc.)
	"""

	math_problem: str = InputField(desc="The math problem to solve")
	math_operation: str = ReasoningField(
		desc="A single math operation to perform as part of the solution to the problem",
		min_length=20,
		max_length=500,
	)
	answer: str = OutputField(desc="The final numerical answer")


@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"expected_output_string",
	],
	# Parameter values
	[
		pytest.param(
			ensure_reasoning_signature("input_1: int -> output_1: str"),
			"""
Your input is:
`input_1` (int)

Your goal is to produce the following output:
`output_1` (str)
""".strip(),
			id="input_1_int_output_1_str",
		),
		pytest.param(
			ensure_reasoning_signature(
				"input_1: int, input_2: str -> output_1: str, output_2: float"
			),
			"""
Your inputs will be:
1. `input_1` (int)
2. `input_2` (str)

Your goal is to produce the following outputs:
1. `output_1` (str)
2. `output_2` (float)
""".strip(),
			id="input_1_int_input_2_str_output_1_str_output_2_float",
		),
		pytest.param(
			ensure_reasoning_signature(
				"input_1: int, input_2: str, input_3: bool -> output_1: str",
			),
			"""
Your inputs will be:
1. `input_1` (int)
2. `input_2` (str)
3. `input_3` (bool)

Your goal is to produce the following output:
`output_1` (str)
""".strip(),
			id="input_1_int_input_2_str_input_3_bool_output_1_str",
		),
		pytest.param(
			ensure_reasoning_signature("input_4: float -> output_2: bool"),
			"""
Your input is:
`input_4` (float)

Your goal is to produce the following output:
`output_2` (bool)
""".strip(),
			id="input_4_float_output_2_bool",
		),
		pytest.param(
			ensure_reasoning_signature(
				"input_5: str -> output_3: str",
				instructions="This is a test signature with instructions.",
			),
			"""
Your input is:
`input_5` (str)

Your goal is to produce the following output:
`output_3` (str)
""".strip(),
			id="input_5_str_output_3_str",
		),
	],
)
def test_format_field_description(
	signature: type[ReasoningSignature], expected_output_string: str
) -> None:
	"""
	Test the format_field_description function with a sample signature.
	"""
	formatted_description = format_field_description(signature)
	assert formatted_description == expected_output_string


@pytest.mark.parametrize(
	"raw_value, expected",
	[
		pytest.param("**9.2**", "9.2", id="markdown_wrapped_float"),
		pytest.param("9.2/10", "9.2", id="fraction_like_suffix"),
		pytest.param("Score:\t-1e-3", "-1e-3", id="scientific_notation"),
	],
)
def test_normalize_numeric_field_value(raw_value: str, expected: str) -> None:
	"""
	Test normalize_numeric_field_value extracts the first numeric token.

	This covers common model behaviors like markdown wrappers and numeric values embedded in
	other text.
	"""
	assert normalize_numeric_field_value(raw_value) == expected


def test_format_field_description_math_problem() -> None:
	"""
	Test the format_field_description function with the SolveMathProblem signature.
	"""
	expected_output = """
Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem
""".strip()
	assert format_field_description(SolveMathProblem) == expected_output


def test_format_field_description_generate_argument() -> None:
	"""
	Test the format_field_description function with the GenerateArgument signature.
	"""
	expected_output = """
Your inputs will be:
1. `topic` (str): The topic to generate an argument about
2. `stance` (Literal['PRO', 'ANTI']): The stance to take on the topic

Your goal is to produce the following output:
`argument` (str): The generated argument
""".strip()
	assert format_field_description(GenerateArgument) == expected_output


def test_format_field_description_with_bulletized_constraints() -> None:
	"""
	Test that field descriptions with constraints display in bulletized format.
	"""
	expected_output = """
Your input is:
`user_input` (str): User input with validation constraints
	Constraints:
		* greater than 0
		* minimum length 5
		* maximum length 200

Your goal is to produce the following output:
`response` (str): Generated response
""".strip()
	assert format_field_description(ValidatedInputSignature) == expected_output


def test_format_field_description_multiple_constraints() -> None:
	"""
	Test field description formatting with multiple constrained fields.
	"""
	expected_output = """
Your inputs will be:
1. `number_input` (int): A number with range constraints
	Constraints:
		* greater than or equal to 1
		* less than or equal to 100
		* a multiple of 5
2. `text_input` (str): Text with length constraints
	Constraints:
		* minimum length 10
		* maximum length 500

Your goal is to produce the following output:
`result` (str): Result with constraints
	Constraints:
		* minimum length 1
		* maximum length 1000
""".strip()
	assert format_field_description(MultiConstraintSignature) == expected_output


def test_format_field_description_multi_output_constraints() -> None:
	"""
	Test format_field_description with a signature that has multiple constrained output fields.
	"""
	expected_output = """
Your input is:
`task` (str): The task to complete

Your goal is to produce the following outputs:
1. `summary` (str): A summary of the task completion
	Constraints:
		* minimum length 10
		* maximum length 200
2. `score` (int): A numerical score for the task
	Constraints:
		* greater than or equal to 0
		* less than or equal to 100
3. `details` (str): Detailed explanation with length constraints
	Constraints:
		* greater than 0
		* minimum length 50
		* maximum length 1000
""".strip()
	assert format_field_description(MultiOutputConstraintsSignature) == expected_output


def test_format_field_description_math_with_reasoning() -> None:
	"""
	Test the format_field_description function with the MathWithReasoningSignature.

	This test verifies that reasoning fields are properly handled when formatting field descriptions.
	Note that reasoning fields are not shown to the user in the instructions, as they're internal
	to the reasoning process.
	"""
	expected_output = """
Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The final numerical answer
""".strip()
	assert format_field_description(MathWithReasoningSignature) == expected_output


def test_get_reasoning_fields() -> None:
	"""
	Test that reasoning fields are correctly identified and accessible.

	This test verifies that a ReasoningSignature correctly identifies and exposes
	its reasoning fields, which represent individual reasoning steps in the reasoning process.
	"""
	reasoning_fields = MathWithReasoningSignature.reasoning_fields
	assert len(reasoning_fields) == 1
	assert "math_operation" in reasoning_fields

	# Just verify that we can identify this as a reasoning field
	assert (
		MathWithReasoningSignature.reasoning_fields["math_operation"]
		is not None
	)


def test_reasoning_signature_structure() -> None:
	"""
	Test that a ReasoningSignature correctly structures its fields.

	This test verifies that ReasoningSignature maintains a specific field ordering
	(input -> reasoning -> output) and correctly categorizes each field type.
	"""
	# Verify field ordering matches declaration order
	field_names = list(MathWithReasoningSignature.fields.keys())
	assert field_names == ["math_problem", "math_operation", "answer"]

	# Verify field categorization
	assert list(MathWithReasoningSignature.input_fields.keys()) == ["math_problem"]
	assert list(MathWithReasoningSignature.reasoning_fields.keys()) == ["math_operation"]
	assert list(MathWithReasoningSignature.output_fields.keys()) == ["answer"]

	# Verify signature representation
	assert (
		MathWithReasoningSignature.signature
		== "math_problem -> math_operation -> answer"
	)


@pytest.mark.parametrize(
	"signature_str, expected_input, expected_reasoning, expected_output",
	[
		pytest.param(
			"question: str -> reasoning: str -> answer: str",
			["question"],
			["reasoning"],
			["answer"],
			id="question_reasoning_answer",
		),
		pytest.param(
			"context: str, query: str -> analysis: str -> result: str, confidence: float",
			["context", "query"],
			["analysis"],
			["result", "confidence"],
			id="context_query_analysis_result_confidence",
		),
		pytest.param(
			"premise: str, hypothesis: str -> step1: str, step2: str -> conclusion: str",
			["premise", "hypothesis"],
			["step1", "step2"],
			["conclusion"],
			id="premise_hypothesis_steps_conclusion",
		),
	],
)
def test_ensure_reasoning_signature(
	signature_str: str,
	expected_input: list[str],
	expected_reasoning: list[str],
	expected_output: list[str],
) -> None:
	"""
	Test that ensure_reasoning_signature correctly parses string signatures with reasoning steps.
	"""
	signature = ensure_reasoning_signature(signature_str)
	assert signature is not None
	assert list(signature.input_fields.keys()) == expected_input
	assert list(signature.reasoning_fields.keys()) == expected_reasoning
	assert list(signature.output_fields.keys()) == expected_output


def test_get_field_description_string_with_constraints() -> None:
	"""
	Test get_field_description_string function with constrained fields.
	"""
	# Test input fields with constraints
	input_fields_output = get_field_description_string(
		ValidatedInputSignature.input_fields
	)
	expected_input = """
`user_input` (str): User input with validation constraints
	Constraints:
		* greater than 0
		* minimum length 5
		* maximum length 200
""".strip()
	assert input_fields_output == expected_input


def test_get_field_description_string_multiple_fields_with_constraints() -> None:
	"""
	Test get_field_description_string with multiple constrained fields.
	"""
	input_fields_output = get_field_description_string(
		MultiConstraintSignature.input_fields
	)
	expected_input = """
1. `number_input` (int): A number with range constraints
	Constraints:
		* greater than or equal to 1
		* less than or equal to 100
		* a multiple of 5
2. `text_input` (str): Text with length constraints
	Constraints:
		* minimum length 10
		* maximum length 500
""".strip()
	assert input_fields_output == expected_input


def test_get_field_description_string_no_constraints() -> None:
	"""
	Test that fields without constraints don't show constraint section.
	"""
	output_fields_output = get_field_description_string(SolveMathProblem.output_fields)
	expected_output = "`answer` (str): The answer to the math problem"
	assert output_fields_output == expected_output
	# Ensure no constraint formatting appears
	assert "Constraints:" not in output_fields_output


def test_get_field_description_string_multiple_output_constraints() -> None:
	"""
	Test get_field_description_string with multiple constrained output fields.
	"""
	output_fields_output = get_field_description_string(
		MultiOutputConstraintsSignature.output_fields
	)
	expected_output = """
1. `summary` (str): A summary of the task completion
	Constraints:
		* minimum length 10
		* maximum length 200
2. `score` (int): A numerical score for the task
	Constraints:
		* greater than or equal to 0
		* less than or equal to 100
3. `details` (str): Detailed explanation with length constraints
	Constraints:
		* greater than 0
		* minimum length 50
		* maximum length 1000
""".strip()
	assert output_fields_output == expected_output


@pytest.mark.parametrize(
	"final_output_kind, expected_description",
	[
		pytest.param(
			"synthesis_strict",						# final_output_kind
			FINAL_OUTPUT_SYNTHESIS_STRICT,			# expected_description
			id="synthesis_strict",
		),
		pytest.param(
			"synthesis_faithful",					# final_output_kind
			FINAL_OUTPUT_SYNTHESIS_FAITHFUL,		# expected_description
			id="synthesis_faithful",
		),
		pytest.param(
			"synthesis_restructured",				# final_output_kind
			FINAL_OUTPUT_SYNTHESIS_RESTRUCTURED,	# expected_description
			id="synthesis_restructured",
		),
		pytest.param(
			"conclusion",							# final_output_kind
			FINAL_OUTPUT_CONCLUSION,				# expected_description
			id="conclusion",
		),
	],
)
def test_get_final_output_description_valid_kinds(
	final_output_kind: Literal[
		"synthesis_strict", "synthesis_faithful", "synthesis_restructured", "conclusion"
	],
	expected_description: str
) -> None:
	"""
	Test get_final_output_description returns correct prompt for each final_output_kind.

	This test verifies that the function correctly maps each final_output_kind value
	to its corresponding prompt string from adapter.prompts.

	Parameters:
	    final_output_kind: The final_output_kind value to test.
	    expected_description: The expected prompt string that should be returned.
	"""
	actual_description = get_final_output_description(final_output_kind)
	assert actual_description == expected_description
	assert isinstance(actual_description, str)
	assert len(actual_description) > 0


def test_get_final_output_description_invalid_kind() -> None:
	"""
	Test get_final_output_description raises ValueError for invalid final_output_kind.

	This test verifies that the function properly handles edge cases by raising
	a ValueError with an appropriate error message when given an unrecognized value.
	We use an integer that doesn't correspond to any valid final_output_kind value.
	"""
	# Create an invalid value by using a raw integer that's not in the enum
	invalid_value = 999
	with pytest.raises(ValueError) as exc_info:
		get_final_output_description(invalid_value)  # type: ignore

	# Verify the error message contains information about the invalid value
	assert "Unknown final_output_kind" in str(exc_info.value)
	assert "999" in str(exc_info.value)


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
