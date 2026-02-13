"""Tests functionality related to constraints over language model outputs."""

# Third-party imports
import pytest

# Local imports
from adapter.constraints import (
	GranularityType,
	ResponseLength,
	format_response_length_instruction,
	format_thought_length_instruction,
)
from signatures.field_constants import DEFAULT_REASONING_FIELD_NAME


@pytest.mark.parametrize(
	# Parameter names
	[
		"granularity",
		"bounds",
		"expected",
	],
	# Parameter values
	[
		pytest.param(
			"word",
			(10, 20),
			"Between 10 and 20 words",
			id="between_10_and_20_words",
		),
		pytest.param(
			"sentence",
			(1, 5),
			"Between 1 and 5 sentences",
			id="between_1_and_5_sentences",
		),
		pytest.param(
			"word",
			(5, 5),
			"Exactly 5 words",
			id="exactly_5_words",
		),
		pytest.param(
			"sentence",
			(1, 1),
			"Exactly 1 sentence",
			id="exactly_1_sentence",
		),
		pytest.param(
			"word",
			(1, 10),
			"Between 1 and 10 words",
			id="between_1_and_10_words",
		),
		pytest.param(
			"paragraph",
			(3, 5),
			"Between 3 and 5 paragraphs",
			id="between_3_and_5_paragraphs",
		),
		pytest.param(
			"word",
			(None, None),
			"Any number of words",
			id="any_number_of_words",
		),
		pytest.param(
			"sentence",
			(None, None),
			"Any number of sentences",
			id="any_number_of_sentences",
		),
		pytest.param(
			"paragraph",
			(None, None),
			"Any number of paragraphs",
			id="any_number_of_paragraphs",
		),
	],
)
def test_response_length_str(
	granularity: GranularityType,
	bounds: tuple[int | None, int | None],
	expected: str,
) -> None:
	"""
	Test the string representation of ResponseLength.

	Args:
		granularity: The granularity type (`word`, `sentence`, or `paragraph`)
		bounds: A tuple representing the bounds (minimum, maximum) of the response length.
			If `None`, there is no bound for that side of the range.
		expected: The expected string representation of the response length.
	"""
	response_length = ResponseLength(granularity=granularity, bounds=bounds)
	assert str(response_length) == expected


def test_fails_if_lower_bound_is_zero_or_negative() -> None:
	"""
	Test that ResponseLength raises an error if the lower bound is zero or negative.
	"""
	with pytest.raises(ValueError):
		ResponseLength(granularity="word", bounds=(0, 10))

	with pytest.raises(ValueError):
		ResponseLength(granularity="sentence", bounds=(-5, 10))


def test_fails_if_upper_bound_is_negative() -> None:
	"""
	Test that ResponseLength raises an error if the upper bound is negative.
	"""
	with pytest.raises(ValueError):
		ResponseLength(granularity="word", bounds=(5, -10))


def test_fails_if_upper_bound_is_less_than_lower_bound() -> None:
	"""
	Test that ResponseLength raises an error if the upper bound is less than the lower bound.
	"""
	with pytest.raises(ValueError):
		ResponseLength(granularity="word", bounds=(10, 5))


@pytest.mark.parametrize(
	# Parameter names
	"response_length, expected",
	# Parameter values
	[
		pytest.param(
			ResponseLength(granularity="word", bounds=(10, 20)),
			"Your final answer should be between 10 and 20 words.",
			id="between_10_and_20_words",
		),
		pytest.param(
			ResponseLength(granularity="sentence", bounds=(1, 5)),
			"Your final answer should be between 1 and 5 sentences.",
			id="between_1_and_5_sentences",
		),
		pytest.param(
			ResponseLength(granularity="paragraph", bounds=(3, 5)),
			"Your final answer should be between 3 and 5 paragraphs.",
			id="between_3_and_5_paragraphs",
		),
		pytest.param(
			ResponseLength(granularity="word", bounds=(5, 5)),
			"Your final answer should be exactly 5 words.",
			id="exactly_5_words",
		),
		pytest.param(
			ResponseLength(granularity="sentence", bounds=(1, 1)),
			"Your final answer should be exactly 1 sentence.",
			id="exactly_1_sentence",
		),
		pytest.param(
			ResponseLength(granularity="word", bounds=(1, 10)),
			"Your final answer should be between 1 and 10 words.",
			id="between_1_and_10_words",
		),
		pytest.param(None, "", id="no_response_length"),
	],
)
def test_format_response_length_instruction(
	response_length: ResponseLength,
	expected: str,
) -> None:
	"""
	Test the format_response_length_instruction function.

	Args:
	    use_internal_reasoning_for_response_generation: Whether to use chain-of-thought for
			generating the response.
	    response_length: The response length constraints
	    expected: The expected formatted instruction string
	"""
	result = format_response_length_instruction(response_length=response_length)
	assert result == expected


@pytest.mark.parametrize(
	# Parameter names
	[
		"use_internal_reasoning_for_thought_generation",
		"thought_length",
		"reasoning_step_name",
		"expected",
	],
	# Parameter values
	[
		pytest.param(
			True,
			ResponseLength(granularity="word", bounds=(10, 20)),
			DEFAULT_REASONING_FIELD_NAME,
			"Each `reasoning_step` should be between 10 and 20 words. NOTE: This word limit does not include internal reasoning.",
		),
		pytest.param(
			False,
			ResponseLength(granularity="sentence", bounds=(1, 5)),
			DEFAULT_REASONING_FIELD_NAME,
			"Each `reasoning_step` should be between 1 and 5 sentences.",
		),
		pytest.param(
			True,
			ResponseLength(granularity="paragraph", bounds=(3, 5)),
			DEFAULT_REASONING_FIELD_NAME,
			"Each `reasoning_step` should be between 3 and 5 paragraphs. NOTE: This paragraph limit does not include internal reasoning.",
		),
		pytest.param(
			False,
			ResponseLength(granularity="word", bounds=(5, 5)),
			DEFAULT_REASONING_FIELD_NAME,
			"Each `reasoning_step` should be exactly 5 words.",
		),
		pytest.param(
			True,
			ResponseLength(granularity="sentence", bounds=(1, 1)),
			DEFAULT_REASONING_FIELD_NAME,
			"Each `reasoning_step` should be exactly 1 sentence. NOTE: This sentence limit does not include internal reasoning.",
		),
		pytest.param(
			False,
			ResponseLength(granularity="word", bounds=(1, 10)),
			"solution_step",
			"Each `solution_step` should be between 1 and 10 words.",
		),
		pytest.param(
			False,
			None,
			DEFAULT_REASONING_FIELD_NAME,
			"",
			id="no_thought_length",
		),
	],
)
def test_format_thought_length_instruction(
	use_internal_reasoning_for_thought_generation: bool,
	thought_length: ResponseLength | None,
	reasoning_step_name: str,
	expected: str,
) -> None:
	"""
	Test the format_thought_length_instruction function.

	Args:
	    use_internal_reasoning_for_thought_generation: Whether to use chain-of-thought for thought generation
	    thought_length: The thought length constraints
	    expected: The expected formatted instruction string
	"""
	result = format_thought_length_instruction(
		use_internal_reasoning_for_thought_generation,
		thought_length,
		reasoning_step_name,
	)
	assert result == expected


if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
