# Standard library imports
from typing import Any

# Third-party imports
import pytest
from dspy.utils.exceptions import AdapterParseError

# Local imports
from adapter.constraints import ResponseLength
from adapter.vllm_generator_adapter import VLLMGeneratorAdapter
from signatures import (
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from utilities_for_tests import MockGenerativeLocalVLLM


@pytest.fixture
def vllm_generator_adapter():
	return VLLMGeneratorAdapter()


# Test cases for create_system_prompt
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"thought_length",
		"response_length",
		"has_internal_reasoning",
		"expected_system_prompt",
	],
	# Parameter values
	[
		pytest.param(
			SolveMathProblemWithReasoning,  # signature
			ResponseLength(					# thought_length
				granularity="word",
				bounds=(20, 100),
			),
			ResponseLength(					# response_length
				granularity="word",
				bounds=(None, 20),
			),
			True,  							# has_internal_reasoning
			(  								# expected_system_prompt
				"""
# Instructions

Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and contain a `math_operation` that advances the solution towards producing `answer`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `math_operation` (of type `str`) entails a math operation towards solving the math problem.
- Before writing a new `math_operation`, start with some internal reasoning which discusses and guides what to do with the next `math_operation`.
- Each `math_operation` should be between 20 and 100 words.
- Your final answer should be at most 20 words.

## Response Format

Once a user provides `math_problem`, your response must follow this exact template:

<thinking>
<step>
## internal_reasoning
Your internal reasoning about the first `math_operation`
## math_operation
The first reasoning step towards producing `answer`
</step>
<step>
## internal_reasoning
Your internal reasoning about the second `math_operation`
## math_operation
The second reasoning step towards producing `answer`
</step>
...
<step>
## internal_reasoning
Your internal reasoning about the final `math_operation`
## math_operation
The final reasoning step towards producing `answer`
</step>
</thinking>
<answer>
## answer
Your response for `answer` here
</answer>
""".strip()
			),
			id="solve_math_problem_word_bounds_with_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,	# signature
			ResponseLength(					# thought_length
				granularity="sentence",
				bounds=(2, 5),
			),
			ResponseLength(					# response_length
				granularity="word",
				bounds=(None, 15),
			),
			False,  						# has_internal_reasoning
			(  								# expected_system_prompt
				"""
# Instructions
Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and contain a `math_operation` that advances the solution towards producing `answer`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `math_operation` (of type `str`) entails a math operation towards solving the math problem.
- Each `math_operation` should be between 2 and 5 sentences.
- Your final answer should be at most 15 words.

## Response Format
Once a user provides `math_problem`, your response must follow this exact template:

<thinking>
<step>
## math_operation
The first reasoning step towards producing `answer`
</step>
<step>
## math_operation
The second reasoning step towards producing `answer`
</step>
...
<step>
## math_operation
The final reasoning step towards producing `answer`
</step>
</thinking>
<answer>
## answer
Your response for `answer` here
</answer>
""".strip()
			),
			id="solve_math_problem_sentence_bounds_no_internal_reasoning",
		),
		pytest.param(
			GenerateArgumentWithReasoning,	# signature
			ResponseLength(					# thought_length
				granularity="word",
				bounds=(15, 80),
			),
			ResponseLength(					# response_length
				granularity="sentence",
				bounds=(1, 3),
			),
			True,  							# has_internal_reasoning
			(  								# expected_system_prompt
				"""
# Instructions

Generate an argument which takes the provided stance towards the provided topic.

Your inputs will be:
1. `topic` (str): The topic to generate an argument about
2. `stance` (Literal['PRO', 'ANTI']): The stance to take on the topic

Your goal is to produce the following output:
`argument` (str): The generated argument

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and contain a `claim` that advances the solution towards producing `argument`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `claim` (of type `str`) entails a component of the argument that advocates for the given stance towards the topic.
- Before writing a new `claim`, start with some internal reasoning which discusses and guides what to do with the next `claim`.
- Each `claim` should be between 15 and 80 words.
- Your final answer should be between 1 and 3 sentences.

## Response Format

Once a user provides `topic` and `stance`, your response must follow this exact template:

<thinking>
<step>
## internal_reasoning
Your internal reasoning about the first `claim`
## claim
The first reasoning step towards producing `argument`
</step>
<step>
## internal_reasoning
Your internal reasoning about the second `claim`
## claim
The second reasoning step towards producing `argument`
</step>
...
<step>
## internal_reasoning
Your internal reasoning about the final `claim`
## claim
The final reasoning step towards producing `argument`
</step>
</thinking>
<answer>
## argument
Your response for `argument` here
</answer>
""".strip()
			),
			id="generate_argument_mixed_granularity_with_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			ResponseLength(granularity="paragraph", bounds=(1, 2)), # thought_length
			ResponseLength(granularity="word", bounds=(25, None)), 	# response_length
			True,  						# has_internal_reasoning
			(  							# expected_system_prompt
				"""
# Instructions

Solve the provided math problem and return its answer.

Your input is:
`math_problem` (str): The math problem to solve

Your goal is to produce the following output:
`answer` (str): The answer to the math problem

When solving this problem, you must break down your solution into a series of reasoning steps, followed by a final answer.
Each step towards the answer should be encased within <step>...</step> tags, and contain a `math_operation` that advances the solution towards producing `answer`.
Your final answer must remain highly faithful to the reasoning steps and their underlying ideas.

- Preserve the full set of reasoning steps and their original order.
- You may lightly rephrase for clarity and readability, but the meaning must remain unchanged.
- Structure and sequence should closely follow the original reasoning steps.
- Do NOT introduce new ideas, arguments, facts, or examples.
- Do NOT remove or significantly alter any existing reasoning.

Your goal is to produce a clear synthesis that respects both the content and structure of the original reasoning, while allowing minimal refinement.

Your reasoning process should follow the rules below:
- Each `math_operation` (of type `str`) entails a math operation towards solving the math problem.
- Before writing a new `math_operation`, start with some internal reasoning which discusses and guides what to do with the next `math_operation`.
- Each `math_operation` should be between 1 and 2 paragraphs.
- Your final answer should be at least 25 words.

## Response Format

Once a user provides `math_problem`, your response must follow this exact template:

<thinking>
<step>
## internal_reasoning
Your internal reasoning about the first `math_operation`
## math_operation
The first reasoning step towards producing `answer`
</step>
<step>
## internal_reasoning
Your internal reasoning about the second `math_operation`
## math_operation
The second reasoning step towards producing `answer`
</step>
...
<step>
## internal_reasoning
Your internal reasoning about the final `math_operation`
## math_operation
The final reasoning step towards producing `answer`
</step>
</thinking>
<answer>
## answer
Your response for `answer` here
</answer>
""".strip()
			),
			id="solve_math_problem_paragraph_bounds_with_internal_reasoning",
		),
	],
)
def test_create_system_prompt(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	thought_length: ResponseLength,
	response_length: ResponseLength,
	has_internal_reasoning: bool,
	expected_system_prompt: str,
) -> None:
	"""
	Test the system prompt generation for VLLMGeneratorAdapter with various parameter combinations.

	This test checks that the system prompt is correctly generated based on different
	signatures, length constraints, and chain-of-thought configurations.

	Args:
		vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
		signature: The reasoning signature to use for formatting.
		thought_length: Response length constraints for reasoning steps.
		response_length: Response length constraints for final outputs.
		has_internal_reasoning: Whether internal reasoning guidance is provided.
		expected_system_prompt: The expected system prompt string.
	"""
	system_prompt = vllm_generator_adapter.create_system_prompt(
		signature=signature,
		thought_length=thought_length,
		response_length=response_length,
		has_internal_reasoning=has_internal_reasoning,
	)
	# Compare the generated prompt with the expected prompt
	assert system_prompt.strip() == expected_system_prompt.strip()


# Test cases for format_continued_assistant_message
@pytest.mark.parametrize(
	# Parameter names
	[
		"previous_content",
		"internal_reasoning",
		"prefix",
		"continue_reasoning",
		"signature",
		"expected",
	],
	# Parameter values
	[
		pytest.param(
			"",  					# previous_content
			(  						# internal_reasoning
				"I need to start by identifying the key variables in this problem."
			),
			"", 					# prefix
			True, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
			(  						# expected message
				"""
<thinking>
<step>
## internal_reasoning
I need to start by identifying the key variables in this problem.
## math_operation
""".strip() + "\n"
			),
			id="start_new_reasoning_step",
		),
		pytest.param(
			(  						# previous_content
				"""
<thinking>
<step>
## internal_reasoning
First step reasoning
## math_operation
First operation
</step>
""".strip()
			),
			(  						# internal_reasoning for reasoning step
				"Now I'll solve for x."
			),
			"", 					# prefix
			True, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
			(  						# expected message
				"""
<thinking>
<step>
## internal_reasoning
First step reasoning
## math_operation
First operation
</step>
<step>
## internal_reasoning
Now I'll solve for x.
## math_operation
""".strip() + "\n"
			),
			id="continue_reasoning_step",
		),
		pytest.param(
			(  						# previous_content
				"""
<thinking>
<step>
## math_operation
I need to subtract 5 from 12
</step>
<step>
## math_operation
12 - 5 = 12 - 2 - 3 = 10 - 3
</step>
<step>
## math_operation
10 - 3 = 7
</step>
""".strip()
			),
			"",  					# no internal_reasoning for final output
			"", 					# no prefix for final output
			False, 					# continue_reasoning
			SolveMathProblemWithReasoning,		# signature
		(  							# expected message
			"""
<thinking>
<step>
## math_operation
I need to subtract 5 from 12
</step>
<step>
## math_operation
12 - 5 = 12 - 2 - 3 = 10 - 3
</step>
<step>
## math_operation
10 - 3 = 7
</step>
</thinking>
<answer>
## answer
""".strip() + "\n"
		),
			id="transition_to_answer_without_closing_tag",
		),
		pytest.param(
		(  						# previous_content
			"""
<thinking>
<step>
## math_operation
First step
</step>
</thinking>
""".strip()
		),
		(  						# internal_reasoning_for_output
			"Now let's continue with the next step."
		),
		"", 					# prefix
		True, 					# continue_reasoning
		SolveMathProblemWithReasoning,		# signature
		(  						# expected message
			"""
<thinking>
<step>
## math_operation
First step
</step>
<step>
## internal_reasoning
Now let's continue with the next step.
## math_operation
""".strip() + "\n"
		),
		id="continue_reasoning_with_closed_thinking",
	),
	],
)
def test_format_continued_assistant_message(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	previous_content: str,
	internal_reasoning: str,
	prefix: str,
	continue_reasoning: bool,
	signature: type[ReasoningSignature],
	expected: str,
):
	"""
	Test the formatting of continued assistant messages.

	This test checks that the assistant message is formatted correctly based on the
	previous content, internal reasoning, and other parameters.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    previous_content: The content of the previous assistant message.
	    internal_reasoning: The internal reasoning for the next step.
	    prefix: The prefix for the output section.
	    continue_reasoning: Whether to continue reasoning or switch to answer section.
	    signature: The signature class to use for formatting.
	    expected: The expected formatted message string.
	"""
	result = vllm_generator_adapter.format_continued_assistant_message(
		previous_content=previous_content,
		internal_reasoning_for_output=internal_reasoning,
		prefix_for_output=prefix,
		continue_reasoning=continue_reasoning,
		signature=signature,
	)
	assert result == expected


# Test cases for user_message_output_requirements
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"has_internal_reasoning",
		"expected_user_message_output_requirements",
	],
	# Parameter values
	[
		pytest.param(
		SolveMathProblemWithReasoning,  # signature
		True,  						# has_internal_reasoning
		(  							# expected_user_message_output_requirements
			f"""
Structure your response as follows:

1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your math_operations.
	Each `<step>` section should include a `## {"internal_reasoning"}` section (guidance provided to help your thinking), followed by a `## {"math_operation"}` section.
2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		id="solve_math_problem_with_internal_reasoning",
	),
		pytest.param(
		GenerateArgumentWithReasoning,  # signature
		True,  						# has_internal_reasoning
		(  							# expected_user_message_output_requirements
			f"""
Structure your response as follows:

1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your claims.
	Each `<step>` section should include a `## {"internal_reasoning"}` section (guidance provided to help your thinking), followed by a `## claim` section.
2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.
	The `<answer>` section should include sections for `argument`.
""".strip()
		),
		id="generate_argument_with_internal_reasoning",
	),
		pytest.param(
		SolveMathProblemWithReasoning,  # signature
		False,  					# has_internal_reasoning
		(  							# expected_user_message_output_requirements
			"""
Structure your response as follows:

1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your math_operations.
	Each `<step>` section should contain a `## math_operation` section.
2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		id="solve_math_problem_no_internal_reasoning",
	),
	],
)
def test_user_message_output_requirements(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	has_internal_reasoning: bool,
	expected_user_message_output_requirements: str,
):
	"""
	Test the generation of output format requirements for the user message.

	This test checks that the method correctly generates user message output requirements
	based on the signature and chain-of-thought configuration.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    signature: The reasoning signature to use for formatting.
	    has_internal_reasoning: Whether internal reasoning guidance is provided.
	    expected_user_message_output_requirements: The expected output requirements string.
	"""
	result = vllm_generator_adapter.user_message_output_requirements(
		signature=signature,
		has_internal_reasoning=has_internal_reasoning,
	)
	assert result == expected_user_message_output_requirements


# Test cases for format_demo_assistant_message
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"demo",
		"has_internal_reasoning",
		"expected",
	],
	# Parameter values
	[
		pytest.param(
		SolveMathProblemWithReasoning,  # signature
		(  							# demo
			{
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			}
		),
		True,  						# has_internal_reasoning
		(  							# expected
			"""
<thinking>
<step>
## internal_reasoning
I need to add 1 and 1.
## math_operation
1+1=2
</step>
</thinking>
<answer>
## answer
2
</answer>
""".strip()
		),
		id="solve_math_problem_with_internal_reasoning",
	),
		pytest.param(
		SolveMathProblemWithReasoning,  # signature
		(  							# demo
			{
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			}
		),
		False,  					# has_internal_reasoning
		(  							# expected
			"""
<thinking>
<step>
## math_operation
1+1=2
</step>
</thinking>
<answer>
## answer
2
</answer>
""".strip()
		),
		id="solve_math_problem_no_internal_reasoning",
	),
	],
)
def test_format_demo_assistant_message(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	demo: dict[str, Any],
	has_internal_reasoning: bool,
	expected: str,
) -> None:
	"""
	Test formatting of demo assistant messages with reasoning steps and outputs.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    signature: The signature class to use for formatting.
	    demo: The demo to format.
	    has_internal_reasoning: Whether internal reasoning guidance is provided.
	    expected: The expected formatted message.
	"""
	result = vllm_generator_adapter.format_demo_assistant_message(
		signature=signature,
		demo=demo,
		has_internal_reasoning=has_internal_reasoning,
	)
	assert result == expected


# Test cases for format_user_message_content
@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"inputs",
		"main_request",
		"expected_message",
		"expected_error",
	],
	# Parameter values
	[
		pytest.param(
		SolveMathProblemWithReasoning,  # signature
		{							# inputs
			"math_problem": "What is 1+1?"
		},
		False,  					# main_request
		(  							# expected_message
			"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?
""".strip()
		),
		None,  						# expected_error
		id="solve_math_problem_basic_input_formatting",
	),
		pytest.param(
			SolveMathProblemWithReasoning,  # signature
			{						# inputs
				"math_problem": "What is 1+1?"
			},
			True,  					# main_request
			(  						# expected_message
			"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?

To produce `answer`, reason step-by-step by writing a sequence of math_operations.

Structure your response as follows:

1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your math_operations.
	Each `<step>` section should contain a `## math_operation` section.
2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
			None,  					# expected_error
			id="solve_math_problem_main_request_complete",
		),
		pytest.param(
			SolveMathProblemWithReasoning,  # signature
			{							# inputs
				"math_problem": "What is 1+1?"
			},
			True,  						# main_request
		(  								# expected_message
			"""
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?

To produce `answer`, reason step-by-step by writing a sequence of math_operations.

Structure your response as follows:

1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your math_operations.
	Each `<step>` section should contain a `## math_operation` section.
2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		None,  						# expected_error
		id="solve_math_problem_main_request_no_internal_reasoning",
	),
		pytest.param(
		GenerateArgumentWithReasoning,  # signature
		(  							# inputs
			{"topic": "renewable energy", "stance": "PRO"}
		),
		False,  					# main_request
		(  							# expected_message
			"""
Generate an argument which takes the provided stance towards the provided topic.

## topic
renewable energy

## stance
PRO
""".strip()
		),
		None,  						# expected_error
		id="multiple_input_fields_formatting",
	),
		pytest.param(
		SolveMathProblemWithReasoning,  # signature
		{},  						# inputs
		False,  					# main_request
		(  							# expected_message
			"""
## math_problem
What is 1+1?

To produce `answer`, reason step-by-step by writing a sequence of math_operations.

Structure your response as follows:

1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your math_operations.
	Each `<step>` section should contain a `## math_operation` section.
2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.
	The `<answer>` section should include sections for `answer`.
""".strip()
		),
		AssertionError,  			# expected_error
		id="empty_inputs_only_instructions",
	),
	],
)
def test_format_user_message_content(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: type[ReasoningSignature],
	inputs: dict[str, Any],
	main_request: bool,
	expected_message: str | None,
	expected_error: type[Exception] | None,
) -> None:
	"""
	Test formatting of user message content with different inputs and main request flags.

	This test checks that the formatted content exactly matches the expected output
	based on the signature and inputs, or raises the expected error. It verifies that
	the content includes the input fields and, if applicable, the main request guidance.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    signature: The signature class to use for formatting.
	    inputs: The inputs to format in the user message.
	    main_request: True if the user request is the final one (containing the input),
	        and False if it is part of an in-context example (pair of user-assistant messages).
	    expected_message: The exact expected output string (None if error expected).
	    expected_error: The expected error type (None if success expected).
	"""
	if expected_error is not None:
		with pytest.raises(expected_error):
			vllm_generator_adapter.format_user_message_content(
				signature=signature,
				inputs=inputs,
				main_request=main_request,
			)
	else:
		result = vllm_generator_adapter.format_user_message_content(
			signature=signature,
			inputs=inputs,
			main_request=main_request,
		)
		assert result == expected_message


@pytest.mark.parametrize(
	# Parameter names
	[
		"demos",
		"has_internal_reasoning",
		"expected_messages",
		"expected_error",
	],
	# Parameter values
	[
		pytest.param(
			[{  			# demos
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			}],
			True,  			# has_internal_reasoning
			[  				# expected_messages
				{  			# User demo message
					"role": "user",
					"content": """
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?
""".strip(),
				},
				{  			# Assistant demo message
					"role": "assistant",
					"content": """
<thinking>
<step>
## internal_reasoning
I need to add 1 and 1.
## math_operation
1+1=2
</step>
</thinking>
<answer>
## answer
2
</answer>
""".strip(),
				},
			],
			None,  			# expected_error
			id="math_demo_1_with_internal_reasoning",
		),
		pytest.param(
			[{  			# demos
				"input": {"math_problem": "What is 3*3+4?"},
				"reasoning": [
					{
						"internal_reasoning": "Multiply 3 by itself.",
						"math_operation": "3*3=9",
					},
					{
						"internal_reasoning": "Add 9 and 4.",
						"math_operation": "9+4=13",
					}
				],
				"output": {"answer": "13"},
			}],
			True,  			# has_internal_reasoning
			[  				# expected_messages
				{  			# User demo message
					"role": "user",
					"content": """
Solve the provided math problem and return its answer.

## math_problem
What is 3*3+4?
""".strip(),
				},
				{  			# Assistant demo message
					"role": "assistant",
					"content": """
<thinking>
<step>
## internal_reasoning
Multiply 3 by itself.
## math_operation
3*3=9
</step>
<step>
## internal_reasoning
Add 9 and 4.
## math_operation
9+4=13
</step>
</thinking>
<answer>
## answer
13
</answer>
""".strip(),
				},
			],
			None,  				# expected_error
			id="math_demo_2_with_internal_reasoning",
		),
		pytest.param(
			[{  				# demos
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			},
			{  					# demos
				"input": {"math_problem": "What is 3*3+4?"},
				"reasoning": [
					{
						"internal_reasoning": "Multiply 3 by itself.",
						"math_operation": "3*3=9",
					},
					{
						"internal_reasoning": "Add 9 and 4.",
						"math_operation": "9+4=13",
					}
				],
				"output": {"answer": "13"},
			}],
			True,  				# has_internal_reasoning
			[  					# expected_messages
				{  				# User demo message
					"role": "user",
					"content": """
Solve the provided math problem and return its answer.

## math_problem
What is 1+1?
""".strip(),
				},
				{  				# Assistant demo message
					"role": "assistant",
					"content": """
<thinking>
<step>
## internal_reasoning
I need to add 1 and 1.
## math_operation
1+1=2
</step>
</thinking>
<answer>
## answer
2
</answer>
""".strip(),
				},
				{  				# User demo message
					"role": "user",
					"content": """
Solve the provided math problem and return its answer.

## math_problem
What is 3*3+4?
""".strip(),
				},
				{  				# Assistant demo message
					"role": "assistant",
					"content": """
<thinking>
<step>
## internal_reasoning
Multiply 3 by itself.
## math_operation
3*3=9
</step>
<step>
## internal_reasoning
Add 9 and 4.
## math_operation
9+4=13
</step>
</thinking>
<answer>
## answer
13
</answer>
""".strip(),
				},
			],					# expected_messages
			None,  				# expected_error
			id="math_demos_1_2_with_internal_reasoning",
		),
		pytest.param(
			[{					# demos (wrong output field)
				"input": {"math_problem": "2+2"},
				"reasoning": [{
					"internal_reasoning": "Add the numbers",
					"math_operation": "2 + 2 = 4",
				}],
				"output": {"wrong_field": "4"},
			}],
			True,  				# has_internal_reasoning
			None,				# expected_messages
			AssertionError,		# expected_error
			id="error_demo_with_missing_output_field",
		),
		pytest.param(
			[{					# demos (non-list reasoning)
				"input": {"math_problem": "2+2"},
				"reasoning": "not a list",
				"output": {"answer": "4"},
			}],
			True,  				# has_internal_reasoning
			None,				# expected_messages
			AssertionError,		# expected_error
			id="error_demo_with_non_list_reasoning",
		),
		pytest.param(
			[{					# demos (reasoning step missing required field)
				"input": {"math_problem": "2+2"},
				"reasoning": [{
					"internal_reasoning": "Some reasoning",
					"wrong_field": "value",  # Should be "math_operation"
				}],
				"output": {"answer": "4"},
			}],
			True,  				# has_internal_reasoning
			None,				# expected_messages
			AssertionError,		# expected_error
			id="error_demo_with_reasoning_step_missing_required_field",
		),
	],
)
def test_format_demos(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	demos: list[dict[str, Any]],
	has_internal_reasoning: bool,
	expected_messages: list[dict[str, str]] | None,
	expected_error: type[Exception] | None,
) -> None:
	"""
	Test formatting of in-context examples into messages with parameterized demos.

	This test checks that demos are correctly formatted into user-assistant message pairs,
	or raises the expected error. It verifies message roles and content formatting.

	Args:
	    vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
	    demos: List of demos to format.
	    has_internal_reasoning: Whether internal reasoning guidance is provided.
	    expected_messages: List of expected messages (None if error expected).
	    expected_error: The expected error type (None if success expected).
	"""
	if expected_error is not None:
		with pytest.raises(expected_error):
			vllm_generator_adapter.format_demos(
				signature=SolveMathProblemWithReasoning,
				demos=demos,
				has_internal_reasoning=has_internal_reasoning,
			)
	else:
		result = vllm_generator_adapter.format_demos(
			signature=SolveMathProblemWithReasoning,
			demos=demos,
			has_internal_reasoning=has_internal_reasoning,
		)
		assert expected_messages is not None
		assert len(result) == len(expected_messages)
		for msg, expected in zip(result, expected_messages, strict=True):
			assert msg["role"] == expected["role"], (
				f"Expected {expected["role"]} "
				f"but got {msg["role"]}.\n"
				f"Message: {msg["content"]}"
			)
			assert msg["content"].strip() == expected["content"], (
				f"Expected {expected["content"]} "
				f"but got {msg["content"]}.\n"
				f"Message: {msg["content"]}"
			)


@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"inputs",
		"demos",
		"response_length",
		"has_internal_reasoning",
		"previous_content",
		"internal_reasoning_for_output",
		"prefix_for_output",
		"continue_reasoning",
		"expected_message_count",
		"expected_roles",
		"expected_final_message_content",
		"expected_error",
	],
	# Parameter values
	[
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			None,						# previous_content (empty)
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				"system", "user", "assistant"
			],
			[							# xpected_final_message_content
				"<thinking>" + "\n" + "<step>" + "\n## math_operation"
			],
			None,						# expected_error
			id="no_demos_no_previous_content_no_interventions",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			[{  						# demos
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content (empty)
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			5,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant",
				"user",
				"assistant"
			],
			[							# expected_final_message_content
				"<thinking>" + "\n" + "<step>" + "\n## math_operation"
			],
			None,						# expected_error
			id="with_demos_no_previous_content_no_interventions",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			[{  						# demos
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			5,							# expected_message_count
			[							# expected_rolesxs
				"system", 					# system prompt
				"user", 					# in-context example #1 input
				"assistant", 				# in-context example #1 output
				"user", 					# main request
				"assistant"					# assistant response to main request
			],
			[							# expected_final_message_content
				"<thinking>" + "\n" + "<step>" + "\n## math_operation"
			],
			None,						# expected_error
			id="with_demos_no_previous_content_with_continue_reasoning_true",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{										# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			[{  									# demos
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			}],
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[False],					# continue_reasoning
			5,							# expected_message_count
			[							# expected_roles
				"system",					# system prompt
				"user",						# in-context example #1 input
				"assistant",				# in-context example #1 output
				"user",						# main request
				"assistant"					# assistant response to main request
			],
			[								# expected_final_message_content
				"<answer>\n## answer"
			],
			None,						# expected_error
			id="with_demos_no_previous_content_with_continue_reasoning_false",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip(),							# previous_content
			[
				"I should divide 6 by 3.",
			],							# internal_reasoning for reasoning step
			None,						# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			[							# expected_final_message_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
I should divide 6 by 3.
## math_operation
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_no_interventions",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				"Now I need to divide 6 by 3"
			],
			None,						# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			[							# expected_final_message_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
Now I need to divide 6 by 3
## math_operation
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_internal_reasoning_intervention",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				"Now I need to divide 6 by 3"
			],
			["6 / 3 ="],				# prefix_for_output
			[True],						# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			[							# expected_final_message_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
Now I need to divide 6 by 3
## math_operation
6 / 3 =
""".strip()
			],
			None,						# expected_error
			id="no_demos_with_previous_content_full_intervention",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip(),							# previous_content
			[							# internal_reasoning_for_output
				""
			],
			["The answer is"],			# prefix_for_output
			[False],					# continue_reasoning
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			[							# expected_final_message_content (with final answer)
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
</thinking>
<answer>
## answer
The answer is
""".strip()
			],
			AssertionError,				# expected_error
			id="raise_assertion_error_when_continue_reasoning_false_and_prefix_for_output_is_not_empty",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			[
				{  						# demos
					"input": {"math_problem": "What is 1+1?"},
					"reasoning": [
						{
							"internal_reasoning": "I need to add 1 and 1.",
							"math_operation": "1+1=2",
						}
					],
					"output": {"answer": "2"},
				},
				{
					"input": {"math_problem": "What is 3*3+4?"},
					"reasoning": [
						{
							"internal_reasoning": "Multiply 3 by itself.",
							"math_operation": "3*3=9",
						},
						{
							"internal_reasoning": "Add 9 and 4.",
							"math_operation": "9+4=13",
						}
					],
					"output": {"answer": "13"},
				},
				{
					"input": {"math_problem": "What is (3*3+4)*2?"},
					"reasoning": [
						{
							"internal_reasoning": "Multiply 3 by itself.",
							"math_operation": "3*3=9",
						},
						{
							"internal_reasoning": "Add 9 and 4.",
							"math_operation": "9+4=13",
						}
					],
					"output": {"answer": "26"},
				},
				{
					"input": {"math_problem": "What is (3*3+4)*2?"},
					"reasoning": [
						{
							"internal_reasoning": "Multiply 3 by itself.",
							"math_operation": "3*3=9",
						},
						{
							"internal_reasoning": "Add 9 and 4.",
							"math_operation": "9+4=13",
						},
						{
							"internal_reasoning": "Multiply 13 by 2.",
							"math_operation": "13*2=26",
						}
					],
					"output": {"answer": "26"},
				}
			],
			None,						# response_length
			True,  						# has_internal_reasoning
			"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip(),							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			11,							# expected_message_count
			[							# expected_roles
				"system",					# system prompt
				"user",						# in-context example 1 input
				"assistant",				# in-context example 1 output
				"user",						# in-context example 2 input
				"assistant",				# in-context example 2 output
				"user",						# in-context example 3 input
				"assistant",				# in-context example 3 output
				"user",						# in-context example 4 input
				"assistant",				# in-context example 4 output
				"user",						# main request
				"assistant"					# assistant response to main request
			],
			[							# expected_final_message_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
<step>
## math_operation
""".strip()
			],
			AssertionError,				# expected_error
			id=(
				"multiple_demos_with_previous_content_missing_internal_reasoning_raises"
			),
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			(  							# previous_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip()
			),
			[							# internal_reasoning_for_output
				"I should divide 6 by 3",
				"I should handle the division outside of the parentheses",
			],
			[							# prefix_for_output
				"6 / 3 =", "6 divided by 3 is"
			],
			[True, True],				# continue_reasoning (both true)
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			(  							# expected_final_message_content
				[
					"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip() + "\n" + """
<step>
## internal_reasoning
I should divide 6 by 3
## math_operation
6 / 3 =
""".strip(),
					"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
I should handle the division outside of the parentheses
## math_operation
6 divided by 3 is
""".strip(),
				]
			),
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_true",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip(),							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[False, False],				# continue_reasoning (both false)
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			[							# expected_final_message_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
</thinking>
<answer>
## answer
""".strip(),
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
</thinking>
<answer>
## answer
""".strip(),
			],
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_false",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			(  							# previous_content
				"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
""".strip()
			),
			[							# internal reasonings
				"I should divide 6 by 3", "",
			],
			[							# prefixes
				"6 / 3 =", "",
			],
			[True, False],				# continue_reasoning (mixed)
			3,							# expected_message_count
			[							# expected_roles
				"system",
				"user",
				"assistant"
			],
			(  							# expected_final_message_content
				[
					"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
I should divide 6 by 3
## math_operation
6 / 3 =
""".strip(),
					"""
<thinking>
<step>
## internal_reasoning
I should start by simplifying the term in the parentheses.
## math_operation
10 / 2 = 5
</step>
</thinking>
<answer>
## answer
""".strip()
				]
			),
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_mixed",
		),
		pytest.param(
			SolveMathProblemWithReasoning,	# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			(  							# previous_content
				"""
<thinking>
<step>
## math_operation
10 / 2 = 5
</step>
""".strip()
			),
			[							# internal_reasoning_for_output
				"I should divide 6 by 3",
				"Now I will add the term after the parentheses",
			],
			[							# prefix_for_output
				"6 / 3 =", "Six divided by three is"
			],
			[True, True],				# continue_reasoning (both true)
			3,							# expected_message_count
			[							# expected_roles
				"system", "user", "assistant"
			],
			(  							# expected_final_message_content
				[						# Two reasoning steps with internal reasoning provided
					"""
<thinking>
<step>
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
I should divide 6 by 3
## math_operation
6 / 3 =
""".strip(),
					"""
<thinking>
<step>
## math_operation
10 / 2 = 5
</step>
<step>
## internal_reasoning
Now I will add the term after the parentheses
## math_operation
Six divided by three is
""".strip()
				]
			),
			None,						# expected_error
			id="no_demos_with_previous_content_multiple_interventions_continue_reasoning_with_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content (empty)
			["", ""],					# internal_reasoning_for_output (both empty)
			[							# prefix_for_output
				"6 / 3 =", "Now I will add"
			],
			[True, True],				# continue_reasoning (both true)
			3,							# expected_message_count
			[							# expected_roles
				"system", "user", "assistant"
			],
			[							# expected_final_message_content
				"""
<thinking>
<step>
## math_operation
6 / 3 =
""".strip(),
				"""
<thinking>
<step>
## math_operation
Now I will add
""".strip(),
			],
			None,						# expected_error
			id="no_demos_no_previous_content_multiple_interventions_continue_reasoning_both_true_no_internal_reasoning",
		),
		pytest.param(
			SolveMathProblemWithReasoning,			# signature
			{							# inputs
				"math_problem": "What is (10 / 2) + 6 / 3?"
			},
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content (empty)
			[							# internal_reasoning_for_output (both empty)
				"",
				"",
			],
			[							# prefix_for_output
				"6 / 3 =", ""
			],
			[True, False],				# continue_reasoning (mixed: True, False)
			3,							# expected_message_count
			[							# expected_roles
				"system", "user", "assistant"
			],
			[							# expected_final_message_content
				# First generation is a reasoning step without previous content or internal reasoning
				"""
<thinking>
<step>
## math_operation
6 / 3 =
""".strip(),
				# Second generation is a final answer without previous content or internal reasoning
				"""
<answer>
## answer
""".strip(),
			],
			None,						# expected_error
			id="no_demos_no_previous_content_multiple_interventions_continue_reasoning_mixed_no_internal_reasoning",
		),
		pytest.param(
		SolveMathProblemWithReasoning,	# signature
		{								# inputs
			"equation": "What is 2+2?"
		},
		None,							# demos
		None,							# response_length
		True,  							# has_internal_reasoning
		"",								# previous_content
		None,							# internal_reasoning_for_output
		None,							# prefix_for_output
		[[True]],						# continue_reasoning
		2,								# expected_message_count
		[								# expected_roles
			"system", "user",
		],								# expected_roles
		"",								# expected_final_message_content
		AssertionError,					# expected_error
		id="error_wrong_input_field_name",
	),
		pytest.param(
			SolveMathProblemWithReasoning,	# signature
			{							# inputs (too many fields)
				"math_problem": "What is 2+2?",
				"subject": "arithmetic",
				"difficulty": 2,
			},							# inputs
			None,						# demos
			None,						# response_length
			True,  						# has_internal_reasoning
			"",							# previous_content
			None,						# internal_reasoning_for_output
			None,						# prefix_for_output
			[True],						# continue_reasoning
			2,							# expected_message_count
			[							# expected_roles
				"system", "user",
			],							# expected_roles
			"",							# expected_final_message_content
			AssertionError,				# expected_error
			id="error_too_many_input_fields",
		),
	],
)
def test_format_single_trajectory_with_interventions(
	vllm_generator_adapter: VLLMGeneratorAdapter,
	signature: Any,
	inputs: dict[str, Any],
	demos: list[dict[str, Any]] | None,
	response_length: ResponseLength | None,
	has_internal_reasoning: bool,
	previous_content: str,
	internal_reasoning_for_output: str | list[str],
	prefix_for_output: str | list[str],
	continue_reasoning: bool | list[bool],
	expected_message_count: int,
	expected_roles: list[str],
	expected_final_message_content: list[str],
	expected_error: type[Exception] | None,
) -> None:
	"""
	Verify that the `format_single_trajectory_with_interventions` method produces the correct output structure.

	Args:
		vllm_generator_adapter: The VLLMGeneratorAdapter instance to test.
		signature: The signature class to use for formatting.
		inputs: The inputs to format in the user message.
		demos: Optional list of demo examples.
		response_length: Optional response length constraints.
		has_internal_reasoning: Whether internal reasoning guidance is provided.
		previous_content: Previous content to continue reasoning from.
		internal_reasoning_for_output: Internal reasoning for the output section.
		prefix_for_output: Prefix for the output section.
		continue_reasoning: Whether to continue reasoning or switch to answer section.
		expected_message_count: Expected number of messages in the output.
		expected_roles: Expected roles in the output.
		expected_final_message_content: The expected content of the final (assistant) message for
			each resulting trajectory.
		expected_error: Expected error message.
	"""
	if expected_error is not None:
		# TODO[P3]: Check against an expected error message rather than simply the type of exception.
		with pytest.raises(expected_error):
			vllm_generator_adapter.format_single_trajectory_with_interventions(
				signature=signature,
				inputs=inputs,
				demos=demos,
				response_length=response_length,
				has_internal_reasoning=has_internal_reasoning,
				previous_content=previous_content,
				internal_reasoning_for_output=internal_reasoning_for_output,
				prefix_for_output=prefix_for_output,
				continue_reasoning=continue_reasoning,
			)
	else:
		result: list[list[dict[str, Any]]] = (
			vllm_generator_adapter.format_single_trajectory_with_interventions(
				signature=signature,
				inputs=inputs,
				demos=demos,
				response_length=response_length,
				has_internal_reasoning=has_internal_reasoning,
				previous_content=previous_content,
				internal_reasoning_for_output=internal_reasoning_for_output,
				prefix_for_output=prefix_for_output,
				continue_reasoning=continue_reasoning,
			)
		)
		num_interventions = len(expected_final_message_content)
		assert len(result) == num_interventions, \
			f"Expected {num_interventions} trajectories, but got {len(result)}"
		for trajectory_index in range(num_interventions):
			trajectory = result[trajectory_index]
			expected_content = expected_final_message_content[trajectory_index]
			assert len(trajectory) == expected_message_count, \
				f"Trajectory #{trajectory_index} should have {expected_message_count} messages, but got {len(trajectory)}."
			assert [msg["role"] for msg in trajectory] == expected_roles
			final_message = trajectory[-1]
			# We are only interested in verifying the final (assistant) message.
			if expected_roles[-1] == "assistant":
				assert final_message["content"].strip() == expected_content.strip()

			# Check user message content based on whether demos are provided
			# Find the user message (should be the last USER message before the assistant message)
			user_message = None
			if (final_message["role"] == "assistant" and len(trajectory) > 1):
				user_message = trajectory[-2]
			elif final_message["role"] == "user":
				user_message = final_message

			if user_message is not None and user_message["role"] == "user":
				user_message_content = user_message["content"]
				substrings_in_main_user_message = ["To produce", "Structure your response"]
				# When demos are None or empty list, main_request=False, so should NOT include task guidance
				if demos is None or (isinstance(demos, list) and len(demos) == 0):
					for substring in substrings_in_main_user_message:
						assert substring not in user_message_content, (
							f"User message should not include {substring} when no demos are provided"
						)
				else:
					# When demos are provided, main_request=True, so should include task guidance
					for substring in substrings_in_main_user_message:
						assert substring in user_message_content, (
							f"User message should include {substring} when demos are provided"
						)


class TestFormatMainMethod:
	"""
	Test the main format method.

	This test checks that the main format method correctly formats the batch with
	the specified parameters.
	"""

	@pytest.fixture
	def adapter(self):
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize(
		# Parameters:
		[
			"signature",
			"inputs",
			"demos",
			"continue_reasoning",
			"internal_reasoning_for_output",
			"prefix_for_output",
			"previous_content",
			"expected_roles_per_trajectory",
			"expected_tag_counts",
		],
		[
			# Basic cases without interventions (no internal_reasoning_for_output or prefix_for_output)
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "What is 2+2?"
				},
				[],						# demos (empty)
				[[True]],				# continue_reasoning
				None,					# internal_reasoning_for_output (empty)
				None,					# prefix_for_output (empty)
				None,					# previous_content
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant",
				],
				[{						# expected_tag_counts
				# There is no intervention, so the assistant message is empty.
					"<thinking>": 1,			# started reasoning section
					"</thinking>": 0,			# did not end reasoning section
					"<step>": 1,				# started the first reasoning step
					"</step>": 0,				# did not complete the first reasoning step
					"<answer>": 0,				# did not start the answer section
					"</answer>": 0,				# did not complete the answer section
					"internal_reasoning": 0,	# no internal reasoning provided
				}],
				id="solve_math_empty_demos_bool_true_no_interventions_continue_reasoning_true",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "What is 5*3?"
				},
			[{							# demos (single demo)
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					},
				],
				"output": {"answer": "2"},
			}],
				[[False]],  			# continue_reasoning
				None,					# internal_reasoning_for_output (empty)
				None,					# prefix_for_output (empty)
				None,					# previous_content
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant",
					"user",
					"assistant"
				],
				[{						# expected_tag_counts
				# When continue_reasoning is False, but there are no interventions,
				# we prepend the <answer> tag to the assistant so that it immediately
				# starts the answer section.
					"<thinking>": 0,			# started thinking section
					"</thinking>": 0,			# ended thinking section
					"<step>": 0,				# no reasoning
					"</step>": 0,				# no reasoning
					"<answer>": 1,				# answer section started
					"</answer>": 0,				# answer section not completed
					"internal_reasoning": 0,	# no internal reasoning provided
				}],
				id="solve_math_single_demo_bool_false_no_interventions_continue_reasoning_false",
			),
			# Basic cases with no interventions, but with previous content
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "What is 7 * (12 / 4 + 3)?"
				},
				[],						# demos (empty)
				[[True]],  				# continue_reasoning
				[
					[
						"I should multiply the sum by 7.",
					]
				],						# internal_reasoning for reasoning step
				None,					# prefix_for_output (empty)
				[						# previous_content
					"""
<thinking>
<step>
## internal_reasoning
I need to divide 12 by 4
## math_operation
12 / 4 = 3
</step>
<step>
## internal_reasoning
Now I should add 3 to the result
## math_operation
3 + 3 = 6
</step>
""".strip(),
				],
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant"
				],
				[{								# expected_tag_counts
					"<thinking>": 1,			# reasoning section started
					"</thinking>": 0,			# reasoning section not ended
					"<step>": 3,				# two existing steps, one new step
					"</step>": 2,				# two existing steps completed, one new step not completed
					"<answer>": 0,				# answer section not started
					"</answer>": 0,				# answer section not completed
					"internal_reasoning": 3,	# existing internal reasoning + intervention provided
				}],
				id="solve_math_single_intervention_continue_reasoning",
			),
			# Multiple trajectories with no intervention, but with previous content.
			# Mix of continue_reasoning and no continue_reasoning.
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "What is 7 * (12 / 4 + 3)?"
				},
				[{  					# demos (multiple demos)
				"input": {"math_problem": "What is 1+1?"},
				"reasoning": [
					{
						"internal_reasoning": "I need to add 1 and 1.",
						"math_operation": "1+1=2",
					}
				],
				"output": {"answer": "2"},
			},
			{
				"input": {"math_problem": "What is 3*3+4?"},
				"reasoning": [
					{
						"internal_reasoning": "Multiply 3 by itself.",
						"math_operation": "3*3=9",
					},
					{
						"internal_reasoning": "Add 9 and 4.",
						"math_operation": "9+4=13",
					}
				],
				"output": {"answer": "13"},
			}],
				[[True], [False]],  	# continue_reasoning (list)
				[						# internal_reasoning_for_output (list)
					["Next, I need to add 3 to the result"],
					[""]
				],
				[						# prefix_for_output (list)
					["3 + 3 ="],
					[""],
				],
				[
					"""
<thinking>
<step>
## internal_reasoning
The first step is to divide 12 by 4
## math_operation
12 / 4 = 3
</step>
""".strip(),							# previous_content for trajectory 1
					"""
<thinking>
<step>
## internal_reasoning
First, I need to divide 12 by 4.
## math_operation
12 / 4 = 3
</step>
""".strip(),							# previous_content for trajectory 2
				],
				[						# expected_roles_per_trajectory
					"system",					# system prompt
					"user",						# demo #1 input
					"assistant",				# demo #1 output
					"user",						# demo #2 input
					"assistant",				# demo #2 output
					"user",						# user request (example input)
					"assistant", 				# assistant response (example output)
				],
				[
					{							# expected_tag_counts for trajectory 1
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, one new step not completed
						"<answer>": 0,			# answer section not started
						"</answer>": 0,			# answer section not completed
						"internal_reasoning": 2,# existing internal reasoning + intervention provided
					},
					{							# expected_tag_counts for trajectory 2
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 1,		# reasoning section ended
						"<step>": 1,			# one new step started
						"</step>": 1,			# one new step completed
						"<answer>": 1,			# answer section started
						"</answer>": 0,			# answer section not completed
						"internal_reasoning": 1,# existing internal reasoning (intervention for answer section not included)
					}
				],
				id="solve_math_multiple_trajectories_mixed_continue_reasoning",
			),
			# No past context, one intervention
			pytest.param(
				SolveMathProblemWithReasoning,	# signature
				{						# inputs
					"math_problem": "What is 12/4?"
				},
				[],						# demos (empty)
				[[True]],  				# continue_reasoning
				[						# internal_reasoning_for_output
					["I need to perform division"]
				],
				[["12 / 4 ="]],			# prefix_for_output
				None,					# previous_content (empty)
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant"
				],
				[{								# expected_tag_counts
					"<thinking>": 1,			# reasoning section started
					"</thinking>": 0,			# reasoning section not ended
					"<step>": 1,				# one new step being started
					"</step>": 0,				# step not completed
					"<answer>": 0,				# no answer section
					"</answer>": 0,				# no answer section
					"internal_reasoning": 1,	# internal reasoning provided
				}],
				id="solve_math_no_past_context_single_intervention_continue",
			),
			# One trajectory (without past context), one intervention
			pytest.param(
				GenerateArgumentWithReasoning,		# signature
				{						# inputs
					"topic": "renewable energy", "stance": "PRO"
				},
				[{  					# demos (single demo)
					"input": {"topic": "renewable energy", "stance": "PRO"},
					"reasoning": [
						{
							"internal_reasoning": "I need to highlight the benefits of renewable energy.",
							"claim": "Renewable energy reduces carbon emissions",
						}
					],
					"output": {
						"argument": "Renewable energy is crucial for combating climate change by significantly reducing carbon emissions."
					},
				}],
				[[False]],  			# continue_reasoning
				None,					# internal_reasoning_for_output
				None,					# prefix_for_output
				None,					# previous_content (empty)
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant",
					"user",
					"assistant"
				],
				[{								# expected_tag_counts
					"<thinking>": 0,			# started reasoning section
					"</thinking>": 0,			# ended reasoning section
					"<step>": 0,				# no new steps (going to answer)
					"</step>": 0,				# no step completion
					"<answer>": 1,				# answer section started
					"</answer>": 0,				# answer not completed
					"internal_reasoning": 0,	# internal reasoning for answer section not included
				}],
				id="generate_argument_no_past_context_single_intervention_final_answer",
			),

			# One trajectory (with past context), one intervention
			pytest.param(
				SolveMathProblemWithReasoning,	# signature
				{						# inputs
					"math_problem": "What is 15 + 8?"
				},
				[{  					# demos (two demos)
					"input": {"math_problem": "What is 1+1?"},
					"reasoning": [
						{
							"internal_reasoning": "I need to add 1 and 1.",
							"math_operation": "1+1=2",
						}
					],
					"output": {"answer": "2"},
				},
				{
					"input": {"math_problem": "What is 3*3+4?"},
					"reasoning": [
						{
							"internal_reasoning": "Multiply 3 by itself.",
							"math_operation": "3*3=9",
						},
						{
							"internal_reasoning": "Add 9 and 4.",
							"math_operation": "9+4=13",
						}
					],
					"output": {"answer": "13"},
				}],
				[[True]],  				# continue_reasoning
				[						# internal_reasoning_for_output
					["Now I'll add these numbers"]
				],
				[["15 + 8 ="]],			# prefix_for_output
				[						# previous_content
					"""
<thinking>
<step>
## internal_reasoning
I need to add 15 and 8
## math_operation
Let me identify the numbers first
</step>
""".strip(),
				],
				[							# expected_roles_per_trajectory
					"system", 				# system prompt
					"user", 				# demo #1 input
					"assistant", 			# demo #1 output
					"user", 				# demo #2 input
					"assistant", 			# demo #2 output
					"user", 				# user request (example input)
					"assistant", 			# assistant response (example output)
				],
				[{							# expected_tag_counts
					"<thinking>": 1,		# reasoning section started
					"</thinking>": 0,		# reasoning section not ended
					"<step>": 2,			# one existing step, one new step
					"</step>": 1,			# one existing step completed, new step not completed
					"<answer>": 0,			# no answer section
					"</answer>": 0,			# no answer section
					"internal_reasoning": 2,# existing internal reasoning + intervention provided
				}],
				id="solve_math_single_past_context_single_intervention_continue",
			),
			# One trajectory (with past context), one intervention (final answer)
			pytest.param(
				GenerateArgumentWithReasoning, # signature
				{						# inputs
					"topic": "space exploration", "stance": "ANTI"
				},
				[],						# demos (empty)
				[[False]],  			# continue_reasoning
				None,					# internal_reasoning_for_output
				None,					# prefix_for_output
				[						# previous_content
					"""
<thinking>
<step>
## claim
Space exploration is too expensive and risky
</step>
<step>
## claim
The cost of a single mission is outrageous
</step>
""".strip(),
				],
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant"
				],
				[{						# expected_tag_counts
					"<thinking>": 1,			# reasoning section not started (going to answer)
					"</thinking>": 1,			# reasoning section not ended
					"<step>": 2,				# two existing steps from previous content
					"</step>": 2,				# two existing steps completed from previous content
					"<answer>": 1,			# answer section started
					"</answer>": 0,				# answer not completed
					"internal_reasoning": 0,			# intervention for answer section not included
				}],
				id="generate_argument_single_past_context_single_intervention_final_answer",
			),
			# One trajectory (with past context), multiple interventions
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "What is 15+8-3?"
				},
				[{  					# demos (single demo)
					"input": {"math_problem": "What is 1+1?"},
					"reasoning": [
						{
							"internal_reasoning": "I need to add 1 and 1.",
							"math_operation": "1+1=2",
						}
					],
					"output": {"answer": "2"},
				}],
				[[True, True, False]],  # continue_reasoning (list[list]) -> single trajectory with multiple interventions
				[						# internal_reasoning_for_output (list[list]) -> single trajectory with multiple interventions
					[
						"Next, I will subtract 3 from the result",
						"I have to subtract 3 from the result",
						""
					]
				],
				[						# prefix_for_output (list[list]) -> single trajectory with multiple interventions
					[
						"23 - 3 =",
						"Subtracting 3 from 23 yields",
						"",
					]
				],
				[						# previous_content
					"""
<thinking>
<step>
## internal_reasoning
I will start with the first addition
## math_operation
15 + 8 = 23
</step>
""".strip()
				],
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant",
					"user",
					"assistant"
				],
				[
					{							# expected_tag_counts for intervention 1
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section has not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, new step not completed
						"<answer>": 0,			# no answer section
						"</answer>": 0,			# no answer section
						"internal_reasoning": 2,# existing internal reasoning + intervention provided
					},
					{							# expected_tag_counts for intervention 2
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section has not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, new step not completed
						"<answer>": 0,			# no answer section
						"</answer>": 0,			# no answer section
						"internal_reasoning": 2,# existing internal reasoning + intervention provided
					},
					{								# expected_tag_counts for intervention 3
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 1,		# reasoning section ended
						"<step>": 1,			# one existing step (thinking ends)
						"</step>": 1,			# one existing step completed
						"<answer>": 1,			# answer section started
						"</answer>": 0,			# answer not completed
						"internal_reasoning": 1,# existing internal reasoning (intervention for answer section not included)
					}
				],
				id="solve_math_single_past_context_multiple_interventions",
			),
			# Multiple trajectories (with past context), single intervention per trajectory
			pytest.param(
				GenerateArgumentWithReasoning,		# signature
				{									# inputs
					"topic": "artificial intelligence", "stance": "PRO"
				},
				[{  								# demos (multiple demos)
					"input": {"topic": "renewable energy", "stance": "PRO"},
					"reasoning": [
						{
							"internal_reasoning": "I need to highlight the benefits of renewable energy.",
							"claim": "Renewable energy reduces carbon emissions",
						}
					],
					"output": {
						"argument": "Renewable energy is crucial for combating climate change by significantly reducing carbon emissions."
					},
				},
				{
					"input": {"topic": "nuclear power", "stance": "ANTI"},
					"reasoning": [
						{
							"internal_reasoning": "I need to argue against nuclear power.",
							"claim": "Nuclear waste poses long-term risks",
						}
					],
					"output": {
						"argument": "Nuclear power creates radioactive waste that remains dangerous for thousands of years."
					},
				}],
				[[True], [False]],  	# continue_reasoning
				None,					# internal_reasoning_for_output (empty because `use_internal_reasoning_for_thought_generation` is False)
				[						# prefix_for_output (list)
					["For example, "],
					[""]
				],
				[						# previous_content (list)
					"""
<thinking>
<step>
## claim
AI advances medical research significantly
</step>
""".strip(),
					"""
<thinking>
<step>
## claim
AI improves efficiency across industries
</step>
""".strip(),
				],
				[						# expected_roles_per_trajectory
					"system",
					"user",
					"assistant",
					"user",
					"assistant",
					"user",
					"assistant"
				],
				[						# expected tag counts
					{							# expected_tag_counts for trajectory 1
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section has not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, new step not completed
						"<answer>": 0,			# no answer section
						"</answer>": 0,			# no answer section
						"internal_reasoning": 0,# no internal reasoning provided
					},
					{							# expected_tag_counts for trajectory 2
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 1,		# reasoning section has ended
						"<step>": 1,			# one existing step (thinking ends)
						"</step>": 1,			# one existing step completed
						"<answer>": 1,			# answer section started
						"</answer>": 0,			# answer not completed
						"internal_reasoning": 0,# no internal reasoning provided
					}
				],
				id="generate_argument_multiple_past_contexts_single_intervention_each",
			),
			# Multiple trajectories (with past context), multiple interventions per trajectory
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "What is (4*5)+(3*2)?"
				},
				[],						# demos (empty)
				[						# continue_reasoning (list of lists - 2 interventions for traj1, 3 for traj2)
					[True, False], [True, True, False]
				],
				[						# internal_reasoning_for_output (list of lists)
					[
						"Now let me calculate 3 * 2",
						""
					],
					[
						"Now let me calculate 4 * 5",
						"Now let me calculate the first parenthesis",
						""
					]
				],
				[						# prefix_for_output (list of lists)
					["3 * 2 =", ""],
					["4 * 5 =", "(4 * 5) =", ""]
				],
				[						# previous_content (list)
					"""
<thinking>
<step>
## internal_reasoning
Let me start with the first parenthesis
## math_operation
4*5 = 20
</step>
""".strip(),
					"""
<thinking>
<step>
## internal_reasoning
Let me start with the second parenthesis
## math_operation
3*2 = 6
</step>
""".strip(),
				],
				[						# expected_roles_per_trajectory
					"system", "user", "assistant"
				],
				[						# expected tag counts
					{							# expected_tag_counts for trajectory 1, intervention 1
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section has not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, new step not completed
						"<answer>": 0,			# no answer section
						"</answer>": 0,			# no answer section
						"internal_reasoning": 2,# existing internal reasoning + intervention provided
					},
					{							# expected_tag_counts for trajectory 1, intervention 2
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 1,		# reasoning section has ended
						"<step>": 1,			# one existing step (thinking ends)
						"</step>": 1,			# one existing step completed
						"<answer>": 1,			# answer section started
						"</answer>": 0,			# answer not completed
						"internal_reasoning": 1,# existing internal reasoning (intervention for answer section not included)
					},
					{							# expected_tag_counts for trajectory 2, intervention 1
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section has not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, new step not completed
						"<answer>": 0,			# no answer section
						"</answer>": 0,			# no answer section
						"internal_reasoning": 2,# existing internal reasoning + intervention provided
					},
					{							# expected_tag_counts for trajectory 2, intervention 2
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 0,		# reasoning section has not ended
						"<step>": 2,			# one existing step, one new step
						"</step>": 1,			# one existing step completed, new step not completed
						"<answer>": 0,			# no answer section
						"</answer>": 0,			# no answer section
						"internal_reasoning": 2,# existing internal reasoning + intervention provided
					},
					{							# expected_tag_counts for trajectory 2, intervention 3
						"<thinking>": 1,		# reasoning section started
						"</thinking>": 1,		# reasoning section has ended
						"<step>": 1,			# one existing step (thinking ends)
						"</step>": 1,			# one existing step completed
						"<answer>": 1,			# answer section started
						"</answer>": 0,			# answer not completed
						"internal_reasoning": 1,# existing internal reasoning (intervention for answer section not included)
					}
				],
				id="solve_math_multiple_past_contexts_multiple_interventions_each",
			),

		],
	)
	def test_format(
		self,
		adapter: VLLMGeneratorAdapter,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		demos: list[dict[str, Any]],
		continue_reasoning: list[list[bool]],
		internal_reasoning_for_output: list[list[str]],
		prefix_for_output: list[list[str]],
		previous_content: list[str],
		expected_roles_per_trajectory: list[str],
		expected_tag_counts: list[dict[str, int]],
	) -> None:
		"""
		Test formatting with single input and no interventions - parameterized.

		This test covers multiple combinations of:
		- Two different signatures (SolveMathProblemWithReasoning, GenerateArgumentWithReasoning)
		- Three demo configurations (empty, single demo, multiple demos)
		- Two different inputs per signature
		- Various has_internal_reasoning and continue_reasoning parameter combinations

		Args:
		    adapter: The VLLMGeneratorAdapter instance to test.
		    signature: The signature to use for formatting.
		    inputs: Input dictionary for the signature.
		    demos: Demo list to use for in-context examples.
		    continue_reasoning: Whether to continue reasoning (boolean or list).
		    previous_content: Previous content to continue from.
		    expected_roles_per_trajectory: Expected message roles (same for all trajectories).
		    expected_tag_counts: Expected counts of tags in final assistant message.
		"""
		result = adapter.format(
			signature=signature,
			inputs=inputs,
			demos=demos,
			previous_content=previous_content,
			continue_reasoning=continue_reasoning,
			internal_reasoning_for_output=internal_reasoning_for_output,
			prefix_for_output=prefix_for_output,
		)
		assert len(result) == len(expected_tag_counts)
		for trajectory_index, expected_tag_counts_for_trajectory in enumerate(expected_tag_counts):
			formatted_trajectory: list[dict[str, Any]] = result[trajectory_index]
			actual_roles: list[str] = [msg["role"] for msg in formatted_trajectory]
			assert actual_roles == expected_roles_per_trajectory
			if expected_roles_per_trajectory[-1] == "assistant": 	# We expect the last message to be the assistant message
				for tag, expected_count in expected_tag_counts_for_trajectory.items():
					assert expected_count == formatted_trajectory[-1]["content"].count(tag)

	@pytest.mark.parametrize(
		"error_scenario,expected_error",
		[
			pytest.param(
				{
					"inputs": {"math_problem": "test"},
					"demos": [[]],
					"continue_reasoning": [[True]],
				},
				AttributeError,
				id="list_of_lists_demos_unsupported",
			),
			pytest.param(
				{
					"inputs": {"math_problem": "test"},
					"previous_content": [["a", "b"]],
					"continue_reasoning": [[True]],
				},
				NotImplementedError,
				id="previous_content_list_of_lists_unsupported",
			),
			# New: inputs as list is not supported
			pytest.param(
				{
					"inputs": [{"math_problem": "test"}],
					"continue_reasoning": [[True]],
				},
				NotImplementedError,
				id="inputs_list_unsupported",
			),
			# New: continue_reasoning empty list invalid when used in parsing context
			pytest.param(
				{
					"inputs": {"math_problem": "test"},
					"continue_reasoning": [],
				},
				ValueError,
				id="empty_continue_reasoning_list_invalid",
			),
			# New: mismatched keys between signature.input_fields and inputs should assert
			pytest.param(
				{
					"inputs": {},
					"continue_reasoning": [[True]],
				},
				AssertionError,
				id="inputs_missing_required_field",
			),
		],
	)
	# TODO[P3]: Make sure to also test against the expected error message.
	def test_format_error_scenarios(
		self,
		adapter: VLLMGeneratorAdapter,
		error_scenario: dict,
		expected_error: type[Exception],
	) -> None:
		"""Test that format method raises appropriate errors for invalid inputs.

		Args:
			adapter: VLLMGeneratorAdapter under test.
			error_scenario: Keyword arguments passed to adapter.format that trigger errors.
			expected_error: Exception type expected to be raised.
		"""
		with pytest.raises(expected_error):
			adapter.format(signature=SolveMathProblemWithReasoning, **error_scenario)


class TestBatchFormatterErrorCases:
	"""Test error cases and edge conditions."""

	@pytest.fixture
	def adapter(self):
		return VLLMGeneratorAdapter()

	@pytest.fixture
	def signature(self):
		return QuestionAnsweringWithReasoning

class TestCallMethod:
	"""Tests functionality of VLLMGeneratorAdapter.__call__ using a mock LocalVLLM."""

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize(
		[
			"signature",
			"inputs",
			"previous_content",
			"continue_reasoning",
			"internal_reasoning_for_output",
			"prefix_for_output",
			"mock_responses",
			"expected_outputs",
			"expected_error",
			"lm_kwargs",
		],
		[
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[[True], [True]],		# continue_reasoning (two trajectories, each continues)
				None,					# internal_reasoning_for_output (two trajectories, one intervention each)
				None,					# prefix_for_output (two trajectories, one intervention each)
				[						# mock_responses: 1 layer, 2 requests (batched trajectories), n=2
				[
					[
						"## internal_reasoning\ninternal reasoning 1\n## math_operation\nmath operation 1",
						"## internal_reasoning\ninternal reasoning 2\n## math_operation\nmath operation 2",
					],
					[
						"## internal_reasoning\ninternal reasoning 3\n## math_operation\nmath operation 3",
						"## internal_reasoning\ninternal reasoning 4\n## math_operation\nmath operation 4",
					],
				],
			],
				[						# expected_outputs
				[
					{"math_operation": "math operation 1"},
					{"math_operation": "math operation 2"},
				],
				[
					{"math_operation": "math operation 3"},
					{"math_operation": "math operation 4"},
				],
			],
			None,					# expected_error
			[						# lm_kwargs as list of dicts (high-temp sampling, n=2)
				{"temperature": 0.1, "n": 2},
				{"temperature": 0.8, "n": 2}
			],
			id="two_traj_no_prev_both_continue_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				[						# previous_content
					"<thinking>\n<step>\n## math_operation\nprevious step 1\n</step>",
					"<thinking>\n<step>\n## math_operation\nprevious step 2\n</step>",
				],
				[[True], [True]],		# continue_reasoning (two trajectories, each continues)
				None,					# internal_reasoning_for_output (two trajectories, one intervention each)
				None,					# prefix_for_output (two trajectories, one intervention each)
				[						# mock_responses: 1 layer, 2 requests (batched trajectories), n=2
				[
					[
						"## internal_reasoning\ninternal reasoning 5\n## math_operation\nmath operation 5",
						"## internal_reasoning\ninternal reasoning 6\n## math_operation\nmath operation 6",
					],
					[
						"## internal_reasoning\ninternal reasoning 7\n## math_operation\nmath operation 7",
						"## internal_reasoning\ninternal reasoning 8\n## math_operation\nmath operation 8",
					],
				],
			],
				[						# expected_outputs
				[
					{"math_operation": "math operation 5"},
					{"math_operation": "math operation 6"},
				],
				[
					{"math_operation": "math operation 7"},
					{"math_operation": "math operation 8"},
				],
		],
		None,						# expected_error
		[							# lm_kwargs as list of dicts (medium-temp sampling, n=2)
			{"temperature": 0.8, "n": 2},
			{"temperature": 0.8, "n": 2},
		],
		id="two_traj_with_prev_both_continue_with_internal_reasoning",
		),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[[True], [False]],		# continue_reasoning: first continues, second finishes
				[						# internal_reasoning_for_output
					["reasoning 1"], [""]
				],
				[						# prefix_for_output
					["prefix 1"], [""]
				],
				[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[
						"## internal_reasoning\nreasoning 1\n## math_operation\nprefix 1 continuation",
						"## internal_reasoning\nreasoning 1\n## math_operation\nprefix 1 continuation",
					],
					[
						"## internal_reasoning\nreasoning 2\n## answer\nprefix 2 continuation",
						"## internal_reasoning\nreasoning 2\n## answer\nprefix 2 continuation",
						"## internal_reasoning\nreasoning 2\n## answer\nprefix 2 continuation",
					],
				],
			],
				[						# expected_outputs
				[
					{"math_operation": "prefix 1 continuation"},
					{"math_operation": "prefix 1 continuation"},
				],
				[
					{"answer": "prefix 2 continuation"},
					{"answer": "prefix 2 continuation"},
					{"answer": "prefix 2 continuation"},
				],
			],
			None,					# expected_error
			[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
				[{"temperature": 0.8, "n": 2}],
				[{"temperature": 0.8, "n": 3}],
			],
			id="two_traj_no_prev_mixed_continue_false_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[[True], [False]],		# continue_reasoning: first continues, second finishes
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
				[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[" 9"," 10"],
					["\nanswer 1", "\nanswer 2", "\nanswer 3"],
				]
			],
				[						# expected_outputs
					[
						{"math_operation": "9"},
						{"math_operation": "10"},
					],
					[
						{"answer": "answer 1"},
						{"answer": "answer 2"},
						{"answer": "answer 3"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[{"temperature": 0.8, "n": 2}],
					[{"temperature": 0.8, "n": 3}],
				],
				id="two_traj_no_prev_mixed_continue_false_no_interventions",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				[						# previous_content
					"<thinking>\n<step>\n## math_operation\nprevious step 3\n</step>",
					"<thinking>\n<step>\n## math_operation\nprevious step 4\n</step>",
				],
				[[False], [False]],		# continue_reasoning: both finish
				None,					# internal reasonings
				None,					# prefixes
				[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					[
						"## internal_reasoning\nreasoning 3\n## answer\nprefix 3 continuation",
						"## internal_reasoning\nreasoning 3\n## answer\nprefix 3 continuation",
					],
					[
						"## internal_reasoning\nreasoning 4\n## answer\nprefix 4 continuation",
						"## internal_reasoning\nreasoning 4\n## answer\nprefix 4 continuation",
						"## internal_reasoning\nreasoning 4\n## answer\nprefix 4 continuation",
					],
				],
			],
				[						# expected_outputs
					[
						{"answer": "prefix 3 continuation"},
						{"answer": "prefix 3 continuation"},
					],
					[
						{"answer": "prefix 4 continuation"},
						{"answer": "prefix 4 continuation"},
						{"answer": "prefix 4 continuation"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[{"temperature": 0.8, "n": 2}],
					[{"temperature": 0.8, "n": 3}],
				],
				id="two_traj_with_prev_both_finish_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				[						# previous_content
					"<thinking>\n<step>\n## math_operation\nprevious step 5\n</step>",
					"<thinking>\n<step>\n## math_operation\nprevious step 6\n</step>",
				],
				[[False], [False]],		# continue_reasoning: both finish
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
				[						# mock_responses: 1 layer, 2 requests (batched trajectories)
				[
					["\nanswer 4","\nanswer 5"],
					["\nanswer 6", "\nanswer 7", "\nanswer 8"],
				],
			],
				[						# expected_outputs
					[{"answer": "answer 4"}, {"answer": "answer 5"}],
					[{"answer": "answer 6"}, {"answer": "answer 7"}, {"answer": "answer 8"}],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[{"temperature": 0.8, "n": 2}],
					[{"temperature": 0.8, "n": 3}],
				],
				id="two_traj_with_prev_both_finish_no_interventions",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,					# previous_content: single trajectory starting fresh
				[[True, False, True]],	# continue_reasoning: 3 interventions (continue, finish, continue)
				[						# internal_reasoning_for_output
					["reasoning 5", "", "reasoning 7"]
				],
				[						# prefix_for_output
					["prefix 5", "", "prefix 7"]
				],
				[						# mock_responses: 1 layer, 3 requests (all interventions batched), n=2
				[
					[
						"## internal_reasoning\nreasoning 5\n## math_operation\nprefix 5 continuation",
						"## internal_reasoning\nreasoning 5\n## math_operation\nprefix 5 continuation",
					],
					[
						"## internal_reasoning\nreasoning 6\n## answer\nprefix 6 continuation",
						"## internal_reasoning\nreasoning 6\n## answer\nprefix 6 continuation",
					],
					[
						"## internal_reasoning\nreasoning 7\n## math_operation\nprefix 7 continuation",
						"## internal_reasoning\nreasoning 7\n## math_operation\nprefix 7 continuation",
					],
				],
			],
				[						# expected_outputs
					[
						{"math_operation": "prefix 5 continuation"},
						{"math_operation": "prefix 5 continuation"},
					],
					[
						{"answer": "prefix 6 continuation"},
						{"answer": "prefix 6 continuation"},
					],
					[
						{"math_operation": "prefix 7 continuation"},
						{"math_operation": "prefix 7 continuation"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of dicts (medium-temp sampling, n=2)
					{"temperature": 0.8, "n": 2},
				],
				id="single_traj_three_interventions_mixed_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,					# previous_content: single trajectory starting fresh
				[[True, False, True]],	# continue_reasoning: 3 interventions
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
				[						# mock_responses: 1 layer, 3 requests (all interventions batched), n=2
				[
					["\nmath operation 11", "\nmath operation 12"],
					["\nanswer 9", "\nanswer 10"],
					["\nmath operation 13", "\nmath operation 14"],
				],
			],
				[						# expected_outputs
					[{"math_operation": "math operation 11"},{"math_operation": "math operation 12"}],
					[{"answer": "answer 9"},{"answer": "answer 10"}],
					[{"math_operation": "math operation 13"},{"math_operation": "math operation 14"}],
				],
				None,					# expected_error
				[						# lm_kwargs as list of dicts (medium-temp sampling, n=2)
					{"temperature": 0.8, "n": 2},
				],
				id="single_traj_three_interventions_mixed_no_interventions",
			),
			pytest.param(
				SolveMathProblemWithReasoning,			# signature
				{							# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,						# previous_content: two trajectories starting fresh
				[						# continue_reasoning: different patterns per trajectory
					[True, False], [False, True]
				],
				[						# internal_reasoning_for_output
					["reasoning 8", ""],
					["", "reasoning 11"]
				],
				[						# prefix_for_output
					["prefix 8", ""],
					["", "prefix 11"]
				],
				[						# mock_responses: 1 layer, 4 requests (2 traj x 2 interventions batched)
					[
						[
							"## internal_reasoning\nreasoning 8\n## math_operation\nprefix 8 continuation",
							"## internal_reasoning\nreasoning 8\n## math_operation\nprefix 8 continuation",
						],
						[
							"## internal_reasoning\nreasoning 9\n## answer\nprefix 9 continuation",
							"## internal_reasoning\nreasoning 9\n## answer\nprefix 9 continuation",
						],
						[
							"## internal_reasoning\nreasoning 10\n## answer\nprefix 10 continuation",
							"## internal_reasoning\nreasoning 10\n## answer\nprefix 10 continuation",
							"## internal_reasoning\nreasoning 10\n## answer\nprefix 10 continuation",
						],
						[
							"## internal_reasoning\nreasoning 11\n## math_operation\nprefix 11 continuation",
							"## internal_reasoning\nreasoning 11\n## math_operation\nprefix 11 continuation",
							"## internal_reasoning\nreasoning 11\n## math_operation\nprefix 11 continuation",
						],
					],
				],
				[						# expected_outputs
					[
						{"math_operation": "prefix 8 continuation"},
						{"math_operation": "prefix 8 continuation"},
					],
					[
						{"answer": "prefix 9 continuation"},
						{"answer": "prefix 9 continuation"},
					],
					[
						{"answer": "prefix 10 continuation"},
						{"answer": "prefix 10 continuation"},
						{"answer": "prefix 10 continuation"},
					],
					[
						{"math_operation": "prefix 11 continuation"},
						{"math_operation": "prefix 11 continuation"},
						{"math_operation": "prefix 11 continuation"},
					],
				],
				None,						# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 1
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 2
					],
					[
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 1
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 2
					],
				],
				id="two_traj_two_interventions_each_mixed_with_internal_reasoning",
			),
			pytest.param(
				SolveMathProblemWithReasoning,		# signature
				{						# inputs
					"math_problem": "Compute (8/2) + 3"
				},
				None,					# previous_content: two trajectories starting fresh
				[						# continue_reasoning: different patterns per trajectory
					[True, False], [False, True]
				],
				None,					# internal_reasoning_for_output: no interventions
				None,					# prefix_for_output: no interventions
				[						# mock_responses: 1 layer, 4 requests (2 per trajectory)
					[
						["\nmath operation 15", "\nmath operation 16"],
						["\nanswer 11", "\nanswer 12"],
						["\nanswer 13", "\nanswer 14", "\nanswer 15"],
						["\nmath operation 17", "\nmath operation 18", "\nmath operation 19"],
					]
				],
				[
					[
						{"math_operation": "math operation 15"},
						{"math_operation": "math operation 16"},
					],
					[
						{"answer": "answer 11"},
						{"answer": "answer 12"},
					],
					[
						{"answer": "answer 13"},
						{"answer": "answer 14"},
						{"answer": "answer 15"},
					],
					[
						{"math_operation": "math operation 17"},
						{"math_operation": "math operation 18"},
						{"math_operation": "math operation 19"},
					],
				],
				None,					# expected_error
				[						# lm_kwargs as list of list of dicts (medium-temp sampling, n=2 for first trajectory, n=3 for second trajectory)
					[
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 1
						{"temperature": 0.8, "n": 2},		# lm_kwargs for traj 1, intervention 2
					],
					[
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 1
						{"temperature": 0.8, "n": 3},		# lm_kwargs for traj 2, intervention 2
					],
				],
				id="two_traj_two_interventions_each_mixed_no_interventions",
			),
		],
	)
	def test_call(
		self,
		adapter: VLLMGeneratorAdapter,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		previous_content: str | list[str],
		continue_reasoning: bool | list[bool] | list[list[bool]],
		internal_reasoning_for_output: str | list[str] | list[list[str]],
		prefix_for_output: str | list[str] | list[list[str]],
		mock_responses: list[str] | list[list[str]],
		expected_outputs: list[list[dict[str, Any]]] | None,
		expected_error: type[Exception] | None,
		lm_kwargs: dict[str, Any],
	) -> None:
		"""Comprehensive parameterized test for VLLMGeneratorAdapter.__call__.

		This test asserts exact parsed outputs (per conversation and completion) when using
		a mock LLM, and verifies expected exceptions when inputs or outputs are invalid.
		"""
		lm = MockGenerativeLocalVLLM(mock_responses)
		if expected_error is not None:
			with pytest.raises(expected_error):
				adapter(
					signature=signature,
					lm=lm,
					inputs=inputs,
					lm_kwargs=lm_kwargs,
					previous_content=previous_content,
					continue_reasoning=continue_reasoning,
					internal_reasoning_for_output=internal_reasoning_for_output,
					prefix_for_output=prefix_for_output,
				)
		else:
			result = adapter(
				signature=signature,
				lm=lm,
				inputs=inputs,
				lm_kwargs=lm_kwargs,
				previous_content=previous_content,
				continue_reasoning=continue_reasoning,
				internal_reasoning_for_output=internal_reasoning_for_output,
				prefix_for_output=prefix_for_output,
			)
			assert expected_outputs is not None
			assert len(result) == len(expected_outputs)
			for i, expected_candidates in enumerate(expected_outputs):
				assert len(result[i]) == len(expected_candidates)
				for j, expected_parsed in enumerate(expected_candidates):
					assert result[i][j] == expected_parsed

			# Stop-token/order invariant: the adapter's per-message continue_reasoning mapping
			# should match the flattened continue_reasoning list, and stop tokens should follow it.
			expected_continue_by_message = [b for traj in continue_reasoning for b in traj]
			assert len(adapter._trajectory_continue_reasoning) == len(expected_continue_by_message)
			for message_idx, continue_val in enumerate(expected_continue_by_message):
				assert adapter._trajectory_continue_reasoning[message_idx] == continue_val
				expected_stop = "</step>" if continue_val else "</answer>"
				actual_stop = adapter._determine_stop_tokens(continue_val)[0]
				assert actual_stop == expected_stop

class TestVLLMGeneratorAdapterParsing:
	"""Integration tests for parse() method of VLLMGeneratorAdapter."""

	@pytest.fixture
	def adapter(self) -> VLLMGeneratorAdapter:
		return VLLMGeneratorAdapter()

	@pytest.mark.parametrize([
		"signature", "mock_response", "continue_reasoning",
		"expected_fields", "expected_error"
	], [
		# JSON reasoning response scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			'{"internal_reasoning": "I need to solve this step by step", "math_operation": "First, I will compute 2 + 2 = 4"}',
			True,
			{
				"internal_reasoning": "I need to solve this step by step",
				"math_operation": "First, I will compute 2 + 2 = 4",
			},
			None,
			id="parse_json_reasoning"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			'{"answer": "The answer is 4"}',
			False,
			{"answer": "The answer is 4"},
			None,
			id="parse_json_answer"
		),
		# Field header response scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			"""## internal_reasoning
I need to think about this problem carefully
## math_operation
Let me compute 5 * 6 = 30""",
			True,
			{"math_operation": "Let me compute 5 * 6 = 30"},
			None,
			id="parse_headers_reasoning"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			"""## answer
The final solution is 30""",
			False,
			{"answer": "The final solution is 30"},
			None,
			id="parse_headers_final_answer"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			"## answerThe answer is Y.",
			False,
			None,
			AdapterParseError,
			id="parse_glued_header_output_field"
		),
		# Mixed format scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			"""## internal_reasoning
This is my reasoning process
## math_operation
{"result": "Computing the result: 8 / 2 = 4", "value": 4}""",
			True,
			{"math_operation": '{"result": "Computing the result: 8 / 2 = 4", "value": 4}'},
			None,
			id="parse_mixed_header_with_json_content"
		),
		# Error scenarios
		pytest.param(
			SolveMathProblemWithReasoning,
			'{"INTERNAL_REASONING": "Missing closing brace", "MATH_OPERATION": "2+2=4"}',
			True,
			None,
			AdapterParseError,
			id="parse_malformed_json_error"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			'{"wrong_field": "This field is not required", "another_wrong": "Also not needed"}',
			True,
			None,
			AdapterParseError,
			id="parse_missing_required_fields"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			"",
			True,
			None,
			AdapterParseError,
			id="parse_empty_response_error"
		),
		pytest.param(
			SolveMathProblemWithReasoning,
			"""## math_operation
Only one field provided""",
			False,  # Expecting answer fields but only got math_operation
			None,
			AdapterParseError,
			id="parse_incomplete_headers_error"
		),
	])
	def test_parse_method_scenarios(
		self,
		adapter: VLLMGeneratorAdapter,
		signature: type[ReasoningSignature],
		mock_response: str,
		continue_reasoning: bool | list[bool] | list[list[bool]],
		expected_fields: dict[str, Any],
		expected_error: type[Exception] | None,
	) -> None:
		"""Test parse() method with various response formats and reasoning contexts."""
		# Set up adapter context to match the scenario
		adapter._current_continue_reasoning = continue_reasoning

		if expected_error is not None:
			with pytest.raises(expected_error):
				adapter.parse(signature, mock_response, parse_reasoning=continue_reasoning)
		else:
			result = adapter.parse(signature, mock_response, parse_reasoning=continue_reasoning)
			assert result == expected_fields

if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
