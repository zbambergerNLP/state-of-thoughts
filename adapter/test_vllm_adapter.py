"""
Tests for the LocalVLLMAdapter class functionality.

Expected usage:
```bash
pytest adapter/test_vllm_adapter.py -vv
```
"""

# Standard library imports
import json
import re
from typing import Any

# Third-party imports
import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Local imports
from adapter.constraints import ResponseLength
from adapter.vllm_adapter import FIELD_HEADER_PATTERN, LocalVLLMAdapter
from misc_utils import ExecutionError
from signatures import (
	AnalyzeTextWithReasoning,
	GenerateArgumentWithReasoning,
	SolveMathProblemWithReasoning,
)
from utilities_for_tests import MockGenerativeLocalVLLM


@pytest.fixture
def adapter() -> LocalVLLMAdapter:
	"""Creates a LocalVLLMAdapter instance for testing."""
	return LocalVLLMAdapter()


class _SoundnessPromiseFloat(dspy.Signature):
	"""Minimal signature to test soundness/promise header parsing for float scores."""

	soundness: float = dspy.OutputField(desc="Soundness score.", ge=0.0, le=10.0)
	promise: float = dspy.OutputField(desc="Promise score.", ge=0.0, le=10.0)


def test_adapter_initialization() -> None:
	"""Tests that the adapter can be initialized properly."""
	adapter = LocalVLLMAdapter()
	assert adapter is not None
	assert hasattr(adapter, "format")
	assert hasattr(adapter, "parse")


@pytest.mark.parametrize(
	# Parmeter names
	[
		"inputs",
		"main_request",
		"expected_result",
	],
	# Parameter values
	[
		pytest.param(
			{"math_problem": "What is 5 + 3?"},				# inputs
			False,  										# main_request
			(  												# expected_result
				"""
## math_problem
What is 5 + 3?
""".strip()
			),
			id="math_problem_not_main_request",
		),
		pytest.param(
			{"topic": "Climate change", "stance": "PRO"},	# inputs
			False,  										# main_request
			(  												# expected_result
				"""
## topic
Climate change

## stance
PRO
""".strip()
			),
			id="argument_generation_not_main_request",
		),
		pytest.param(
			{"math_problem": "What is 5 + 3?"},				# inputs
			True,  											# main_request
			(  												# expected_result
			"""
## math_problem
What is 5 + 3?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip()
			),
			id="math_problem_main_request",
		),
	],
)
def test_format_user_message_content(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any],
	main_request: bool,
	expected_result: str,
) -> None:
	"""
	Validates that `format_user_message_content` correctly formats user messages.

	Args:
	    adapter: The LocalVLLMAdapter instance.
	    inputs: The inputs to format.
	    main_request: Whether this is the main request (final user message in the list of messages) or
	        a user input as part of in-context examples (demos, which are earlier in the list of messages).
	    expected_result: The expected formatted user message content.
	"""
	signature = (
		SolveMathProblemWithReasoning if "math_problem" in inputs else GenerateArgumentWithReasoning
	)
	result = adapter.format_user_message_content(
		signature=signature, inputs=inputs, main_request=main_request
	)
	assert result == expected_result


@pytest.mark.parametrize(
	# Parmeter names
	[
		"prefix",
		"suffix",
		"expected_result",
	],
	# Parameter values
	[
		pytest.param(
			"Please solve this:",  	# prefix
			"Thank you!", 	 		# Suffix
			(  						# expected_result
				"""
Please solve this:

## math_problem
What is 5 + 3?

Thank you!
""".strip()
			),
			id="with_prefix_and_suffix",
		),
		pytest.param(
			"",  					# prefix
			"",  					# suffix
			(  						# expected_result
			"""
## math_problem
What is 5 + 3?
""".strip()
			),
			id="no_prefix_no_suffix",
		),
		pytest.param(
			"Custom prefix:", 		# prefix
			"",  					# suffix
			(  						# expected_result
				"""
Custom prefix:

## math_problem
What is 5 + 3?
""".strip()
			),
			id="only_prefix",
		),
	],
)
def test_format_user_message_with_prefix_suffix(
	adapter: LocalVLLMAdapter,
	prefix: str,
	suffix: str,
	expected_result: str,
) -> None:
	"""
	Valides that format_user_message_content correctly formats user messages with prefixes and suffixes.

	Args:
	    adapter: The LocalVLLMAdapter instance.
	    prefix: The prefix to add to the user message.
	    suffix: The suffix to add to the user message.
	    expected_result: The expected formatted user message content.
	"""
	inputs = {"math_problem": "What is 5 + 3?"}
	result = adapter.format_user_message_content(
		signature=SolveMathProblemWithReasoning,
		inputs=inputs,
		prefix=prefix,
		suffix=suffix,
		main_request=False,
	)
	assert result == expected_result


@pytest.mark.parametrize(
	"signature, expected_result",
	[
		pytest.param(
			SolveMathProblemWithReasoning,
			"Respond with the corresponding output fields, starting with the field `## answer`",
			id="math_problem_signature",
		),
		pytest.param(
			AnalyzeTextWithReasoning,
			"Respond with the corresponding output fields, starting with the field `## summary`, then `## sentiment`, then `## keywords` (must be formatted as a valid Python list[str])",
			id="analyze_text_signature",
		),
	],
)
def test_user_message_output_requirements(
	adapter: LocalVLLMAdapter,
	signature: type[dspy.Signature],
	expected_result: str,
) -> None:
	"""
	Test output requirements for various signatures.

	Output requirements are a component of the user message that specifies how the model should format its response.

	Args:
		adapter: The LocalVLLMAdapter instance.
		signature: The signature to test.
		expected_result: The expected output requirements string.
	"""
	result = adapter.user_message_output_requirements(signature)
	assert result == expected_result


@pytest.mark.parametrize(
	# Parmeter names
	[
		"signature",
		"demo",
		"expected_result",
	],
	# Parameter values
	[
		pytest.param(
			SolveMathProblemWithReasoning,		# signature
			{									# demo
				"input": {"math_problem": "What is 2 + 2?"},
				"output": {"answer": "4"},
			},
			(  									# expected_result
			"""
## answer

4
""".strip()
			),
			id="math_problem_demo",
		),
		pytest.param(
			AnalyzeTextWithReasoning,			# signature
			{									# demo
				"input": {
					"input_text": "I love this product! It works perfectly."
				},
				"output": {
					"summary": "Customer expresses satisfaction",
					"sentiment": "positive",
					"keywords": ["love", "perfectly"],
				},
			},
			(  									# expected_result
			"""
## summary

Customer expresses satisfaction

## sentiment

positive

## keywords

["love", "perfectly"]
""".strip()
			),
			id="analyze_text_demo",
		),
	],
)
def test_format_demo_assistant_message(
	adapter: LocalVLLMAdapter,
	signature: type[dspy.Signature],
	demo: dict[str, Any],
	expected_result: str,
) -> None:
	"""
	Test formatting of assistant message for demos.

	Args:
		adapter: The LocalVLLMAdapter instance.
		signature: The signature to use for formatting.
		demo: The demo data to format. This consists of a single message by the assistant,
			which includes the *output* fields of the demo (whereas a presumed user message
			that preceded it contains the *input* fields).
		expected_result: The expected formatted assistant message content.
	"""
	result = adapter.format_demo_assistant_message(signature=signature, demo=demo)
	assert result == expected_result


@pytest.mark.parametrize(
	# Parmeter names
	[
		"invalid_demo",
		"error_message",
	],
	# Parameter values
	[
		pytest.param(
			{							 			# invalid_demo: missing output
				"input": {"math_problem": "2+2"}
			},
			"missing one or more required keys",	# error_message
			id="missing_output",
		),
		pytest.param(
			{							 			# invalid_demo: missing input field
				"input": {},
				"output": {"answer": "4"},
			},
			"input is missing required fields",		# error_message
			id="missing_input_field",
		),
		pytest.param(
			{							 			# invalid_demo: missing required output field
				"input": {"math_problem": "2+2"},
				"output": {},
			},
			"output is missing required fields",	# error_message
			id="missing_output_field",
		),
	],
)
def test_format_demos_validation_errors(
	adapter: LocalVLLMAdapter,
	invalid_demo: dict[str, Any],
	error_message: str,
) -> None:
	"""
	Test that format_demos validates demo structure properly.

	We assume that the `format_demos` method will raise an `AssertionError` if the
	provided demos do not conform to the expected structure.

	Args:
		adapter: The LocalVLLMAdapter instance.
		invalid_demo: A demo that is intentionally invalid to trigger validation errors.
		error_message: The expected error message when validation fails.
	"""
	with pytest.raises(AssertionError, match=error_message):
		adapter.format_demos(
			signature=SolveMathProblemWithReasoning, demos=[invalid_demo]
		)


@pytest.mark.parametrize(
	# Parmeter names
	"num_demos",
	# Parameter values
	[
		pytest.param(1, id="single_demo"),  	# num_demos
		pytest.param(2, id="two_demos"),  		# num_demos
		pytest.param(5, id="five_demos"),  		# num_demos
	],
)
def test_format_demos_message_count(
	adapter: LocalVLLMAdapter,
	num_demos: int,
) -> None:
	"""
	Test that format_demos returns correct number of messages.

	Args:
		adapter: The LocalVLLMAdapter instance.
		num_demos: The number of demos to format.
		expected_message_count: The expected number of messages in the formatted output.
	"""
	demos = []
	for i in range(num_demos):
		demos.append(
			{
				"input": {"math_problem": f"What is {i} + {i}?"},
				"output": {"answer": str(i + i)},
			}
		)
	result = adapter.format_demos(signature=SolveMathProblemWithReasoning, demos=demos)
	assert len(result) == num_demos * 2  # Each demo has 1 user and 1 assistant message
	for i, msg in enumerate(result):  # Check role alternation
		expected_role = "user" if i % 2 == 0 else "assistant"
		assert msg["role"] == expected_role


@pytest.mark.parametrize(
	# Parmeter names
	["completion", "expected_result"],
	# Parameter values
	[
		pytest.param(
			(								# completion
				"""
## answer
4
""".strip()
			),
			{"answer": "4"},				# expected_result
			id="simple_answer",
		),
		pytest.param(
			(								# completion
				"""
## answer
The answer is 42
""".strip()
			),
			{"answer": "The answer is 42"},	# expected_result
			id="verbose_answer",
		),
		pytest.param(
			(								# completion
				"""
## summary
Customer is satisfied

## sentiment
positive

## keywords
["love", "great"]
""".strip()
			),
			{								# expected_result
				"summary": "Customer is satisfied",
				"sentiment": "positive",
				"keywords": ["love", "great"],
			},
			id="multi_field_response",
		),
	],
)
def test_parse_valid_completions(
	adapter: LocalVLLMAdapter,
	completion: str,
	expected_result: dict[str, Any],
) -> None:
	"""
	Test parsing various valid completion formats.

	Args:
		adapter: The LocalVLLMAdapter instance.
		completion: The completion string to parse.
		expected_result: The expected parsed completion in the form of a dictionary.
			This dictionary maps output field names to their corresponding values (parsed
			from the completion string).
	"""
	signature = (
		SolveMathProblemWithReasoning
		if "answer" in expected_result
		else AnalyzeTextWithReasoning
	)
	result = adapter.parse(signature, completion, sampling_params=SamplingParams())
	assert result == expected_result


@pytest.mark.parametrize(
	# Parmeter names
	"json_data",
	# Parameter values
	[
		pytest.param(
			{"answer": "8"},					# json_data
			id="simple_json",
		),
		pytest.param(
			{"answer": "The solution is 42"},	# json_data
			id="verbose_json",
		),
	],
)
def test_parse_json_completions(
	adapter: LocalVLLMAdapter,
	json_data: dict[str, Any],
) -> None:
	"""
	Test parsing JSON-formatted completions.

	LLMs may return completions in JSON format, which should be parsed correctly
	by the adapter when vLLM structured outputs are enabled for the request. These
	completions are expected to be in a dictionary format.

	Args:
		adapter: The LocalVLLMAdapter instance.
		json_data: A dictionary representing the JSON completion data.
	"""
	json_completion = json.dumps(json_data)
	sampling_params = SamplingParams(
		structured_outputs=StructuredOutputsParams(json_object=True)
	)
	result = adapter.parse(
		SolveMathProblemWithReasoning,
		json_completion,
		sampling_params=sampling_params,
	)
	assert result == json_data


@pytest.mark.parametrize(
	# Parmeter names
	[
		"completion",
		"should_parse_successfully",
	],
	# Parameter values
	[
		pytest.param(
			(								# completion
				"""
some invalid json

## answer
6
""".strip()
			),
			True,							# should_parse_successfully
			id="invalid_json_fallback",
		),
		pytest.param(
			(								# completion
				"""
{"invalid": "json"}

## answer
7
""".strip()
			),
			True,							# should_parse_successfully
			id="wrong_json_fallback",
		),
		pytest.param(
			(								# completion
				"""
## wrong_field
some value
""".strip()
			),
			False,							# should_parse_successfully
			id="wrong_field_failure",
		),
		pytest.param(
			(								# completion
				"""
No field headers at all
""".strip()
			),
			False,							# should_parse_successfully
			id="no_headers_failure",
		),
	],
)
def test_parse_fallback_behavior(
	adapter: LocalVLLMAdapter,
	completion: str,
	should_parse_successfully: bool,
) -> None:
	"""
	Test parsing fallback behavior for invalid inputs.

	NOTE: If a completion is malformed or does not conform to the expected format,
	we expect the adapter to either parse it successfully (if it can be salvaged)
	or raise an `AdapterParseError` if it cannot be parsed.

	Args:
		adapter: The LocalVLLMAdapter instance.
		completion: The completion string to parse, which may be invalid or malformed.
		should_parse_successfully: Whether the completion is expected to parse successfully.
	"""
	if should_parse_successfully:
		result = adapter.parse(
			SolveMathProblemWithReasoning,
			completion,
			sampling_params=SamplingParams(),
		)
		assert "answer" in result
	else:
		with pytest.raises(AdapterParseError):  # Should raise AdapterParseError
			adapter.parse(
				SolveMathProblemWithReasoning,
				completion,
				sampling_params=SamplingParams(),
			)


@pytest.mark.parametrize(
	"pattern, should_match",
	[
		pytest.param(
			"## answer",							# pattern
			True,									# should_match
			id="valid_header_answer",
		),
		pytest.param(
			"## summary",							# pattern
			True,									# should_match
			id="valid_header_summary",
		),
		pytest.param(
			"## field_name",						# pattern
			True,									# should_match
			id="valid_header_generic",
		),
		pytest.param(
			"##keywords",							# pattern
			False,									# should_match
			id="invalid_no_space",
		),
		pytest.param(
			"# answer",								# pattern
			False,									# should_match
			id="invalid_one_hash",
		),
		pytest.param(
			"answer",								# pattern
			False,									# should_match
			id="invalid_no_hash",
		),
		pytest.param(
			"### answer",							# pattern
			False,									# should_match
			id="invalid_three_hashes",
		),
	],
)
def test_field_header_pattern(pattern: str, should_match: bool) -> None:
	"""Test the field header pattern regex."""
	match_result = FIELD_HEADER_PATTERN.match(pattern)
	if should_match:
		assert match_result is not None
	else:
		assert match_result is None


@pytest.mark.parametrize(
	# Parmeter names
	[
		"demos",
		"inputs",
		"expected_num_messages",
		"expected_last_content",
	],
	# Parameter values
	[
		pytest.param(
			[
				{								# demo
					"input": {"math_problem": "2+2"},
					"output": {"answer": "4"},
				}
			],
			{"math_problem": "What is 7 + 8?"},	# inputs
			4,  								# expected_num_messages
			(  									# expected_last_content
			"""
## math_problem
What is 7 + 8?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip()
			),
			id="single_demo",
		),
		pytest.param(
			[],									# demos
			{"math_problem": "What is 7 + 8?"},	# inputs
			2,  								# expected_num_messages
			(  									# expected_last_content
				"""
## math_problem
What is 7 + 8?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip()
			),
			id="no_demos",
		),
		pytest.param(
			[									# demos
				{
					"input": {"math_problem": "2+2"},
					"output": {"answer": "4"},
				},
				{
					"input": {"math_problem": "3+5"},
					"output": {"answer": "8"},
				},
				{
					"input": {"math_problem": "7*8"},
					"output": {"answer": "56"},
				},
			],
			{"math_problem": "What is 7 + 8?"},	# inputs
			8,  								# expected_num_messages
			(  									# expected_last_content
				"""
## math_problem
What is 7 + 8?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip()
			),
			id="multiple_demos",
		),
	],
)
def test_format_message_structure_single_example(
	adapter: LocalVLLMAdapter,
	demos: list[dict[str, Any]],
	inputs: dict[str, Any],
	expected_num_messages: int,
	expected_last_content: str,
) -> None:
	"""Test that format returns the correct message structure."""
	result = adapter.format(
		signature=SolveMathProblemWithReasoning, demos=demos, inputs=inputs
	)

	assert isinstance(result, list)
	assert len(result) == 1  # Batch size of 1
	# Get the messages from the first output of the batch (corresponding to our single example)
	messages = result[0]
	assert len(messages) == expected_num_messages
	assert messages[0]["role"] == "system"
	assert messages[-1]["role"] == "user"
	assert messages[-1]["content"] == expected_last_content


def test_format_with_response_length_word_constraint(
	adapter: LocalVLLMAdapter,
) -> None:
	"""
	Test format method with word response length constraints.

	Ensure that the system message includes the response length constraints.

	Args:
		adapter: The LocalVLLMAdapter instance.
	"""
	inputs = {"math_problem": "What is 7 + 8?"}
	response_length = ResponseLength(granularity="word", bounds=(5, 10))
	result = adapter.format(
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs=inputs,
		response_length=response_length,
	)
	expected_system_content = """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Your final answer should be between 5 and 10 words.

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip()
	messages = result[0]  # Get the messages from the first batch
	assert messages[0]["content"] == expected_system_content


def test_format_with_response_length_sentence_constraint(
	adapter: LocalVLLMAdapter,
) -> None:
	"""
	Test format method with sentence response length constraints.

	Ensure that the system message includes the response length constraints.

	Args:
		adapter: The LocalVLLMAdapter instance.
	"""
	inputs = {"math_problem": "What is 7 + 8?"}
	response_length = ResponseLength(granularity="sentence", bounds=(1, 3))
	result = adapter.format(
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs=inputs,
		response_length=response_length,
	)
	expected_system_content = """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Your final answer should be between 1 and 3 sentences.

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip()
	messages = result[0]  # Get the messages from the first example
	assert messages[0]["content"] == expected_system_content


def test_format_without_response_length(adapter: LocalVLLMAdapter) -> None:
	"""
	Test format method without response length constraints.

	Ensure that the system message does not include any response length constraints.

	Args:
		adapter: The LocalVLLMAdapter instance.
	"""
	inputs = {"math_problem": "What is 7 + 8?"}
	result = adapter.format(
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs=inputs,
		response_length=None,
	)
	expected_system_content = """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip()
	messages = result[0]  # Get the messages from the first example
	assert messages[0]["content"] == expected_system_content


@pytest.mark.parametrize(
	"inputs, demos, expected_structure",
	[
		# Single input, no demos
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			[],
			[
				[
					{
						"role": "system",
						"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 5 + 3?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
					},
				]
			],
			id="single_input_no_demos",
		),
		# Batch inputs, no demos
		pytest.param(
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			[],
			[
				[
					{
						"role": "system",
						"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 5 + 3?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
					},
				],
				[
					{
						"role": "system",
						"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 10 - 4?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
					},
				],
			],
			id="batch_inputs_no_demos",
		),
		# Single input with demos
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			[
				{
					"input": {"math_problem": "What is 1 + 1?"},
					"output": {"answer": "2"},
				}
			],
			[
				[
					{
						"role": "system",
						"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 1 + 1?
""".strip(),
					},
					{
						"role": "assistant",
						"content": """
## answer

2
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 5 + 3?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
					},
				]
			],
			id="single_input_with_demos",
		),
	],
)
def test_format_expected_structure(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any] | list[dict[str, Any]],
	demos: list[dict[str, Any]],
	expected_structure: list[list[dict[str, str]]],
) -> None:
	"""Test format method returns expected message structures."""
	result = adapter.format(
		signature=SolveMathProblemWithReasoning, demos=demos, inputs=inputs
	)
	assert result == expected_structure


@pytest.mark.parametrize(
	# Parameter names
	[
		"inputs",
		"demos",
		"error_type",
		"error_message",
	],
	[
		pytest.param(
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			[
				[
					{
						"input": {"math_problem": "What is 1 + 1?"},
						"output": {"answer": "2"},
					}
				]
				# Missing second demo list
			],
			ValueError,
			"must have the same length as inputs",
			id="batch_demos_wrong_length",
		),
		pytest.param(
			"invalid_input",
			[],
			TypeError,
			"Expected inputs to be a list or a dict",
			id="invalid_input_type",
		),
	],
)
def test_format_validation_errors(
	adapter: LocalVLLMAdapter,
	inputs: list[dict[str, Any]] | dict[str, Any],
	demos: list[list[dict[str, Any]]],
	error_type: type[Exception],
	error_message: str,
) -> None:
	"""Test format method validation errors."""
	with pytest.raises(error_type, match=error_message):
		adapter.format(
			signature=SolveMathProblemWithReasoning, demos=demos, inputs=inputs
		)


@pytest.mark.parametrize(
	"demos, inputs, expected_messages",
	[
		pytest.param(
			[
				[
					{
						"input": {"math_problem": "What is 1 + 1?"},
						"output": {"answer": "2"},
					}
				],
				[
					{
						"input": {"math_problem": "What is 3 + 3?"},
						"output": {"answer": "6"},
					}
				],
			],
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			[
				[
					{
						"role": "system",
						"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 1 + 1?
""".strip(),
					},
					{
						"role": "assistant",
						"content": """
## answer

2
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 5 + 3?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
					},
				],
				[
					{
						"role": "system",
						"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 3 + 3?
""".strip(),
					},
					{
						"role": "assistant",
						"content": """
## answer

6
""".strip(),
					},
					{
						"role": "user",
						"content": """
## math_problem
What is 10 - 4?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
					},
				],
			],
			id="batch_demos_different_demos",
		),
	],
)
def test_format_with_batch_demos(
	adapter: LocalVLLMAdapter,
	demos: list[list[dict[str, Any]]],
	inputs: list[dict[str, Any]],
	expected_messages: list[list[dict[str, str]]],
) -> None:
	"""Test format method with batch demos."""
	result = adapter.format(
		signature=SolveMathProblemWithReasoning, demos=demos, inputs=inputs
	)
	assert result == expected_messages


@pytest.mark.parametrize(
	"inputs, demos, expected_output",
	[
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			[
				{
					"input": {"math_problem": "What is 1 + 1?"},
					"output": {"answer": "2"},
				}
			],
			[
				{
					"role": "system",
					"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
				},
				{
					"role": "user",
					"content": """
## math_problem
What is 1 + 1?
""".strip(),
				},
				{
					"role": "assistant",
					"content": """
## answer

2
""".strip(),
				},
				{
					"role": "user",
					"content": """
## math_problem
What is 5 + 3?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
				},
			],
			id="single_input_with_demo",
		),
		pytest.param(
			{"math_problem": "What is 7 * 8?"},
			[],
			[
				{
					"role": "system",
					"content": """
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
				},
				{
					"role": "user",
					"content": """
## math_problem
What is 7 * 8?

Respond with the corresponding output fields, starting with the field `## answer`
""".strip(),
				},
			],
			id="single_input_without_demo",
		),
	],
)
def test_format_single_expected_output(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any],
	demos: list[dict[str, Any]],
	expected_output: list[dict[str, str]],
) -> None:
	"""Test format_single method returns expected message structure."""
	result = adapter.format_single(
		signature=SolveMathProblemWithReasoning,
		demos=demos,
		inputs=inputs,
		response_length=None,
	)
	assert result == expected_output


@pytest.fixture
def mock_lm() -> MockGenerativeLocalVLLM:
	"""Creates a mock LocalVLLM instance for testing."""
	return MockGenerativeLocalVLLM()


@pytest.mark.parametrize(
	"inputs, lm_kwargs, demos, mock_responses, expected_call_structure",
	[
		# Single input, single lm_kwargs
		# Format: [num_layers, num_input_messages, num_choices_per_input_message]
		# 1 layer, 1 request, 1 completion
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			{"temperature": 0.7, "max_tokens": 100},
			[],
			[[["## answer\n8"]]],
			{
				"result_length": 1,
				"result_completions": 1,
			},
			id="single_input_single_lm_kwargs",
		),
		# Batch inputs, single lm_kwargs
		# 1 layer, 2 requests, 1 completion each
		pytest.param(
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			{"temperature": 0.5},
			[],
			[[["## answer\n8"], ["## answer\n6"]]],
			{
				"result_length": 2,
				"result_completions": 1,
			},
			id="batch_inputs_single_lm_kwargs",
		),
		# Single input with demo
		# 1 layer, 1 request, 1 completion
		pytest.param(
			{"math_problem": "What is 12 / 3?"},
			{"temperature": 0.0},
			[
				{
					"input": {"math_problem": "What is 6 / 2?"},
					"output": {"answer": "3"},
				}
			],
			[[["## answer\n4"]]],
			{
				"result_length": 1,
				"result_completions": 1,
			},
			id="single_input_with_demo",
		),
	],
)
def test_call_expected_behavior(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any] | list[dict[str, Any]],
	lm_kwargs: dict[str, Any],
	demos: list[dict[str, Any]],
	mock_responses: list[list[list[str]]],
	expected_call_structure: dict[str, Any],
) -> None:
	"""Test __call__ method with expected behavior patterns."""
	# Create MockGenerativeLocalVLLM with responses from parameterized test case
	mock_lm = MockGenerativeLocalVLLM(responses=mock_responses)

	result = adapter(
		lm=mock_lm,
		lm_kwargs=lm_kwargs,
		signature=SolveMathProblemWithReasoning,
		demos=demos,
		inputs=inputs,
	)

	# Verify result structure
	assert len(result) == expected_call_structure["result_length"]
	for batch_result in result:
		assert len(batch_result) == expected_call_structure["result_completions"]


@pytest.mark.parametrize(
	"inputs, lm_kwargs, mock_responses, expected_parsed_results",
	[
		# Single input with simple answer
		# Format: [num_layers, num_input_messages, num_choices_per_input_message]
		# 1 layer, 1 request, 1 completion
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			{"temperature": 0.7},
			[[["## answer\n8"]]],
			[[{"answer": "8"}]],
			id="single_input_simple_answer",
		),
		# Batch inputs with different answers
		# 1 layer, 2 requests, 1 completion each
		pytest.param(
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			{"temperature": 0.5},
			[[["## answer\n8"], ["## answer\n6"]]],
			[[{"answer": "8"}], [{"answer": "6"}]],
			id="batch_inputs_different_answers",
		),
		# Multiple completions (n > 1)
		# 1 layer, 1 request, 2 completions
		pytest.param(
			{"math_problem": "What is 2 * 3?"},
			{"temperature": 0.8, "n": 2},
			[[["## answer\n6", "## answer\nSix"]]],
			[[{"answer": "6"}, {"answer": "Six"}]],
			id="multiple_completions_n_gt_1",
		),
	],
)
def test_call_parsing_results(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any] | list[dict[str, Any]],
	lm_kwargs: dict[str, Any],
	mock_responses: list[list[list[str]]],
	expected_parsed_results: list[list[dict[str, Any]]],
) -> None:
	"""Test __call__ method parsing and result structure."""
	mock_lm = MockGenerativeLocalVLLM(responses=mock_responses)

	result = adapter(
		lm=mock_lm,
		lm_kwargs=lm_kwargs,
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs=inputs,
	)

	def strip_error_keys(d: dict[str, Any]) -> dict[str, Any]:
		return {k: v for k, v in d.items() if k != "error"}

	stripped = [[strip_error_keys(x) for x in inner] for inner in result]
	assert stripped == expected_parsed_results


def test_call_tolerates_parse_errors(adapter: LocalVLLMAdapter) -> None:
	"""
	The adapter should tolerate per-completion parse errors and return error metadata instead
	of raising, mirroring VLLMGeneratorAdapter behavior.
	"""
	mock_lm = MockGenerativeLocalVLLM(responses=[[["this has no headers at all"]]])
	result = adapter(
		lm=mock_lm,
		lm_kwargs={"temperature": 0.7},
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs={"math_problem": "What is 5 + 3?"},
	)
	assert len(result) == 1
	assert len(result[0]) == 1
	error: ExecutionError = result[0][0].get("error")
	assert error.is_parsing_error()
	assert "failed to parse" in error.error_message.lower()
	assert "this has no headers at all" in error.raw_output


@pytest.mark.parametrize(
	"inputs, lm_kwargs, mock_responses, expected_result_length",
	[
		# Single input, single lm_kwargs
		# 1 layer, 1 request, 1 completion
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			{"temperature": 0.7, "max_tokens": 100},
			[[["## answer\n8"]]],
			1,  # Single batch result
			id="single_input_single_lm_kwargs",
		),
		# Batch inputs, single lm_kwargs
		# 1 layer, 2 requests, 1 completion each
		pytest.param(
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			{"temperature": 0.7, "max_tokens": 100},
			[[["## answer\n8"], ["## answer\n6"]]],
			2,  # Two batch results
			id="batch_inputs_single_lm_kwargs",
		),
	],
)
def test_call_single_and_batch_inputs(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any] | list[dict[str, Any]],
	lm_kwargs: dict[str, Any],
	mock_responses: list[list[list[str]]],
	expected_result_length: int,
) -> None:
	"""Test __call__ method with single and batch inputs."""
	mock_lm = MockGenerativeLocalVLLM(responses=mock_responses)

	result = adapter(
		lm=mock_lm,
		lm_kwargs=lm_kwargs,
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs=inputs,
	)

	# Verify result structure
	assert len(result) == expected_result_length
	for batch_result in result:
		assert len(batch_result) == 1  # Single completion per input


# Test cases for __call__ method with batch lm_kwargs


@pytest.mark.parametrize(
	# Parameter names
	[
		"inputs",
		"lm_kwargs",
		"expected_error",
		"error_message",
	],
	# Parameter values
	[
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			[{"temperature": 0.1}, {"temperature": 0.2}],
			ValueError,
			"When lm_kwargs is a list, inputs must also be a list",
			id="lm_kwargs_list_inputs_single_dict",
		),
		pytest.param(
			[
				{"math_problem": "What is 5 + 3?"},
				{"math_problem": "What is 10 - 4?"},
			],
			[{"temperature": 0.1}],
			ValueError,
			"When lm_kwargs is a list, it must have the same length as inputs",
			id="lm_kwargs_list_length_mismatch",
		),
		pytest.param(
			[{"math_problem": "What is 5 + 3?"}],
			[{"temperature": 0.1}, {"temperature": 0.2}],
			ValueError,
			"When lm_kwargs is a list, it must have the same length as inputs",
			id="lm_kwargs_list_length_exceeds",
		),
	],
)
def test_call_batch_lm_kwargs_validation_errors(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any] | list[dict[str, Any]],
	lm_kwargs: dict[str, Any] | list[dict[str, Any]],
	expected_error: type[Exception],
	error_message: str,
) -> None:
	"""Test __call__ method validation errors with batch lm_kwargs."""
	mock_lm = MockGenerativeLocalVLLM()
	with pytest.raises(expected_error, match=re.escape(error_message)):
		adapter(
			lm=mock_lm,
			lm_kwargs=lm_kwargs,
			signature=SolveMathProblemWithReasoning,
			demos=[],
			inputs=inputs,
		)


@pytest.mark.parametrize(
	"signature, inputs, demos, expected_messages",
	[
		pytest.param(
			AnalyzeTextWithReasoning,
			{"input_text": "I love this product!"},
			[],
			[
				[
					{
						"role": "system",
						"content": """
Analyze the inputted text and perform the following tasks:
- Summarize the text
- Determine the sentiment of the text
- Extract key words from the text

Your input is:
`input_text` (str): Input text to process

Your goal is to produce the following outputs:
1. `summary` (str): A summary of the input
2. `sentiment` (str): The sentiment of the input
3. `keywords` (list[str]): Key words from the input

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## input_text
I love this product!

Respond with the corresponding output fields, starting with the field `## summary`, then `## sentiment`, then `## keywords` (must be formatted as a valid Python list[str])
""".strip(),
					},
				]
			],
			id="multi_output_no_demos",
		),
		pytest.param(
			AnalyzeTextWithReasoning,
			{"input_text": "This is terrible!"},
			[
				{
					"input": {"input_text": "I love this product!"},
					"output": {
						"summary": "Customer expresses satisfaction",
						"sentiment": "positive",
						"keywords": ["love", "product"],
					},
				}
			],
			[
				[
					{
						"role": "system",
						"content": """
Analyze the inputted text and perform the following tasks:
- Summarize the text
- Determine the sentiment of the text
- Extract key words from the text

Your input is:
`input_text` (str): Input text to process

Your goal is to produce the following outputs:
1. `summary` (str): A summary of the input
2. `sentiment` (str): The sentiment of the input
3. `keywords` (list[str]): Key words from the input

Please provide your response with each output field under its own header using the format:
## field_name
Your response for that field here
""".strip(),
					},
					{
						"role": "user",
						"content": """
## input_text
I love this product!
""".strip(),
					},
					{
						"role": "assistant",
						"content": """
## summary

Customer expresses satisfaction

## sentiment

positive

## keywords

["love", "product"]
""".strip(),
					},
					{
						"role": "user",
						"content": """
## input_text
This is terrible!

Respond with the corresponding output fields, starting with the field `## summary`, then `## sentiment`, then `## keywords` (must be formatted as a valid Python list[str])
""".strip(),
					},
				]
			],
			id="multi_output_with_demo",
		),
	],
)
def test_format_multi_output_signatures(
	adapter: LocalVLLMAdapter,
	signature: type[dspy.Signature],
	inputs: dict[str, Any],
	demos: list[dict[str, Any]],
	expected_messages: list[list[dict[str, str]]],
) -> None:
	"""Test format method with multi-output signatures."""
	result = adapter.format(signature=signature, demos=demos, inputs=inputs)
	assert result == expected_messages


@pytest.mark.parametrize(
	"completion, signature, sampling_params, expected_result",
	[
		pytest.param(
			# completion
			"""## summary
Customer is unhappy

## sentiment
negative

## keywords
["terrible", "awful"]""",
			AnalyzeTextWithReasoning,		# signature
			SamplingParams(),				# sampling_params
			{								# expected_result
				"summary": "Customer is unhappy",
				"sentiment": "negative",
				"keywords": ["terrible", "awful"],
			},
			id="multi_output_parsing",
		),
		pytest.param(
			# completion
			'{{"summary": "Brief text", "sentiment": "neutral", "keywords": ["text"]}}',
			AnalyzeTextWithReasoning,		# signature
			SamplingParams(				# sampling_params
				structured_outputs=StructuredOutputsParams(json_object=True),
			),
			{								# expected_result
				"summary": "Brief text",
				"sentiment": "neutral",
				"keywords": ["text"],
			},
			id="json_parsing",
		),
		pytest.param(
			# completion
			"## answer\nThe result is 42",
			SolveMathProblemWithReasoning,		# signature
			SamplingParams(),					# sampling_params
			{									# expected_result
				"answer": "The result is 42",
			},
			id="single_field_parsing",
		),
		pytest.param(
			# completion
			"## answer\nThe result is 42\n## monster type: The Glimmermoss",
			SolveMathProblemWithReasoning,		# signature
			SamplingParams(),					# sampling_params
			{									# expected_result
				"answer": "The result is 42\n## monster type: The Glimmermoss",
			},
			id="unknown_header_treated_as_content",
		),
		pytest.param(
			# completion
			"##\u00A0soundness\n8.5\n\n## promise\n9.0",
			_SoundnessPromiseFloat,				# signature
			SamplingParams(),					# sampling_params
			{									# expected_result
				"soundness": 8.5,
				"promise": 9.0,
			},
			id="soundness_promise_nbsp_header_parsing",
		),
	],
)
def test_parse_various_formats(
	adapter: LocalVLLMAdapter,
	completion: str,
	signature: type[dspy.Signature],
	sampling_params: SamplingParams,
	expected_result: dict[str, Any],
) -> None:
	"""Test parse method with various completion formats."""
	result = adapter.parse(signature, completion, sampling_params=sampling_params)
	assert result == expected_result


@pytest.mark.parametrize(
	"inputs, response_length, expected_system_content_contains",
	[
		pytest.param(
			{"math_problem": "What is 5 + 3?"},
			ResponseLength(granularity="word", bounds=(5, 10)),
			"Your final answer should be between 5 and 10 words.",
			id="word_constraint",
		),
		pytest.param(
			{"math_problem": "What is 10 / 2?"},
			ResponseLength(granularity="sentence", bounds=(1, 2)),
			"Your final answer should be between 1 and 2 sentences.",
			id="sentence_constraint",
		),
		pytest.param(
			{"math_problem": "What is 7 * 6?"},
			None,
			"Please provide your response with each output field under its own header",
			id="no_constraint",
		),
	],
)
def test_format_with_response_length_constraints(
	adapter: LocalVLLMAdapter,
	inputs: dict[str, Any],
	response_length: ResponseLength | None,
	expected_system_content_contains: str,
) -> None:
	"""Test format method with various response length constraints."""
	result = adapter.format(
		signature=SolveMathProblemWithReasoning,
		demos=[],
		inputs=inputs,
		response_length=response_length,
	)

	system_message = result[0][0]["content"]
	assert expected_system_content_contains in system_message


if __name__ == "__main__":
	# Run the tests if this file is executed directly
	pytest.main([__file__, "-vv"])
