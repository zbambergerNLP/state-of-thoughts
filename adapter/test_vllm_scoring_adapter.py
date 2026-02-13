# Standard library imports
from typing import Literal
from unittest.mock import MagicMock

# Third-party imports
import pytest

# Local imports
from adapter.vllm_scoring_adapter import LocalVLLMScoringAdapter
from signatures import (
	GenerateArgumentWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from tree import Input, Output, Reasoning, State
from utilities_for_tests import MockScoringLocalVLLM


def _serialize_reasoning_candidates(
	reasoning_candidates: list[Reasoning] | None,
	output_candidates: list[Output] | None = None,
) -> list[str] | None:
	"""
	Serialize reasoning candidates using `State.model_output_so_far()`.

	This intentionally ignores task inputs; we only need the reasoning/output trajectory
	string exactly as it is rendered at runtime.
	"""
	if reasoning_candidates is None:
		return None
	if not reasoning_candidates:
		return [""]
	formatted: list[str] = []
	for i, candidate in enumerate(reasoning_candidates):
		output: Output = {}
		if output_candidates is not None:
			output = output_candidates[i]
		state = State(input={}, reasoning=candidate, output=output)
		full = state.model_output_so_far()
		# If an output is present, drop the <answer> block and keep </thinking>.
		# This matches how the reranker evaluator derives "thinking-only" context.
		answer_idx = full.find("<answer>")
		formatted.append(full[:answer_idx].rstrip() if answer_idx >= 0 else full)
	return formatted


@pytest.fixture
def scoring_adapter() -> LocalVLLMScoringAdapter:
	"""Fixture for LocalVLLMScoringAdapter."""
	return LocalVLLMScoringAdapter()


@pytest.fixture
def mock_lm() -> MockScoringLocalVLLM:
	"""Fixture for a mock LocalVLLM instance configured for scoring."""
	return MockScoringLocalVLLM()


class TestLocalVLLMScoringAdapterInit:
	"""Test cases for LocalVLLMScoringAdapter initialization."""

	@pytest.mark.parametrize(
		# Parameter names
		"callbacks,expected_callback_count",

		# Parameter values
		[
			pytest.param(
				None,  									# callbacks
				0,  									# expected_callback_count
				id="init_default_no_callbacks",
			),
			pytest.param(
				[MagicMock()],  						# callbacks
				1,  									# expected_callback_count
				id="init_with_single_callback",
			),
			pytest.param(
				[MagicMock(), MagicMock()],  			# callbacks
				2,  									# expected_callback_count
				id="init_with_multiple_callbacks",
			),
		],
	)
	def test_init(
		self,
		callbacks: list[MagicMock] | None,
		expected_callback_count: int,
	) -> None:
		"""
		Test LocalVLLMScoringAdapter initialization with various callback configurations.

		Args:
			callbacks: Optional list of mock callbacks to initialize the adapter with.
			expected_callback_count: Expected number of callbacks in the initialized adapter.
		"""
		adapter = LocalVLLMScoringAdapter(callbacks=callbacks)
		assert len(adapter.callbacks) == expected_callback_count
		if callbacks:
			for i, callback in enumerate(callbacks):
				assert adapter.callbacks[i] == callback

class TestFormatQueries:
	"""Test cases for format_queries method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"input_data",
			"scoring_target",
			"reasoning_candidates",
			"instructions",
			"num_queries",
			"expected_queries",
			"expected_error",
		],
		# Parameter values
		[
			# REASONING target tests with SolveMathProblemWithReasoning
			pytest.param(
				{											# input_data
					"math_problem": "What is 2+2?",
				},
				"reasoning",  								# scoring_target
				[  											# reasoning_candidates
					[{"math_operation": 'First, add 2 and 2'}]
				],
				(											# instructions
					"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
				),
				1,  										# num_queries (reasoning target with 1 candidate)
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
math_problem: What is 2+2?
"""
				],
					None,  									# expected_error
					id="reasoning_single_candidate_math_problem",
				),
			pytest.param(
				{											# input_data
					"math_problem": "What is (5*6+3)/2?",
				},
				"reasoning",  								# scoring_target
				[  											# reasoning_candidates
					[										# candidate #1
						{"math_operation": 'Multiplying 5 by 6 yields 30'}
					],
					[{"math_operation": '5 * 6 = 33'}],		# candidate #2
				],
				(											# instructions
					"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
				),
				2,  										# num_queries (reasoning target with 2 candidates)
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?
""",
					# second candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?
""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_first_step_math_problem",
			),
			pytest.param(
				{											# input_data
					"math_problem": "What is (5*6+3)/2?",
				},
				"reasoning",  								# scoring_target
				[  											# reasoning_candidates
					[
						{"math_operation": "Multiplying 5 by 6 yields 30"},
						{"math_operation": "Adding 3 to 30 yields 33"},
					],
					[
						{"math_operation": "5 * 6 = 33"},
						{"math_operation": "33 + 3 = 36"},
					],
				],
				(											# instructions
					"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
				),
				2,  										# num_queries (reasoning target with 2 candidates)
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?
""",
					# second candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
math_problem: What is (5*6+3)/2?
""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_second_step_math_problem",
			),
			# REASONING target tests with GenerateArgumentWithReasoning
			pytest.param(
				{											# input_data
					"topic": "AI does more harm than good",
					"stance": "PRO",
				},
				"reasoning",  								# scoring_target
				[  											# reasoning_candidates
					[{"claim": 'AI improves efficiency'}]
				],
				(											# instructions
					"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
				),
				1,  										# num_queries (reasoning target with 1 candidate)
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
topic: AI does more harm than good
stance: PRO
""",
				],
				None,  										# expected_error
				id="reasoning_single_candidate_generate_argument",
			),
			pytest.param(
				{											# input_data
					"topic": "Combatting climate change should be a top priority",
					"stance": "ANTI"
				},
				"reasoning",  								# scoring_target
				[  											# reasoning_candidates
					[{"claim": 'Climate policies are too expensive'}],
					[{"claim": 'Economic impact is too severe'}],
					[{"claim": 'Alternative solutions exist'}],
				],
				(											# instructions
					"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
				),
				3,  										# num_queries (reasoning target with 3 candidates)
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
topic: Combatting climate change should be a top priority
stance: ANTI
""",
					# second candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
topic: Combatting climate change should be a top priority
stance: ANTI
""",
					# third candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
topic: Combatting climate change should be a top priority
stance: ANTI
""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_generate_argument",
			),
			# OUTPUT target tests with SolveMathProblemWithReasoning
			pytest.param(
				{											# input_data
					"math_problem": "What is 10/2?",
				},
				"output",  									# scoring_target
				[  											# reasoning_candidates
					[
						{"math_operation": 'Divide 10 by 2'},
						{"math_operation": '10/2 = 5'},
					]
				],
				(											# instructions
					"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
""".strip()
				),
				1,  										# num_queries (output target with 1 reasoning candidate)
				[  											# expected_queries
					# first candidate
					"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
<Query>: # Inputs
math_problem: What is 10/2?
""",
				],
			None,  										# expected_error
			id="output_single_candidate_with_reasoning_math_problem",
		),
		pytest.param(
			{											# input_data
				"math_problem": "What is 3*4?",
			},
			"output",  									# scoring_target
			[  											# reasoning_candidates
				[{"math_operation": '3 times 4'}],
				[{"math_operation": 'Multiply 3 and 4'}],
			],
			(											# instructions
				"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
""".strip()
			),
			2,  										# num_queries (output target with 2 reasoning candidates)
			[  											# expected_queries
				# First candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
<Query>: # Inputs
math_problem: What is 3*4?
""",
				# Second candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
<Query>: # Inputs
math_problem: What is 3*4?
""",
			],
			None,  										# expected_error
			id="output_multiple_candidates_with_reasoning_math_problem",
		),
		# OUTPUT target tests with GenerateArgumentWithReasoning
		pytest.param(
			{											# input_data
				"topic": "Renewable Energy",
				"stance": "PRO"
			},
			"output",  									# scoring_target
			[  											# reasoning_candidates
				[
					{"claim": 'Renewable energy reduces carbon emissions'},
					{"claim": 'It creates jobs'},
				]
			],
			(											# instructions
				"""
Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
""".strip()
			),
			1,  										# num_queries (output target with 1 reasoning candidate)
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge an output for a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.
You will find the output under consideration under the "# Output" heading in the Document.

Judge whether the provided output is a strong final output for the task given the inputs and reasoning.
<Query>: # Inputs
topic: Renewable Energy
stance: PRO
""",
				],
			None,  										# expected_error
			id="output_single_candidate_with_reasoning_generate_argument",
		),
		# ACTION target tests with SolveMathProblemWithReasoning
		pytest.param(
			{											# input_data
				"math_problem": "What is 7+8?",
			},
			"action",  									# scoring_target
			[  											# reasoning_candidates
				[{"math_operation": 'Start by adding 7 and 8'}]
			],
			(											# instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
""".strip()
			),
			1,  										# num_queries (action target with 1 reasoning candidate)
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
<Query>: # Inputs
math_problem: What is 7+8?

# Reasoning
<thinking>
<step>
## math_operation
Start by adding 7 and 8
</step>
""",
				],
			None,  										# expected_error
			id="action_single_candidate_with_reasoning_math_problem",
		),
		pytest.param(
			{											# input_data
				"topic": "Space Exploration is pointless",
				"stance": "ANTI"
			},
			"action",  									# scoring_target
			[  											# reasoning_candidates
				[{"claim": 'Space exploration is expensive'}],
				[{"claim": 'Resources better spent on Earth'}],
			],
			(											# instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
""".strip()
			),
			2,  										# num_queries (action target with 2 reasoning candidates)
			[  											# expected_queries
				# first candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
<Query>: # Inputs
topic: Space Exploration is pointless
stance: ANTI

# Reasoning
<thinking>
<step>
## claim
Space exploration is expensive
</step>
""",
				# second candidate
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Generate an argument which takes the provided stance towards the provided topic."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
<Query>: # Inputs
topic: Space Exploration is pointless
stance: ANTI

# Reasoning
<thinking>
<step>
## claim
Resources better spent on Earth
</step>
""",
			],
			None,  										# expected_error
			id="action_multiple_candidates_with_reasoning_generate_argument",
		),
		# Edge case: No reasoning candidates for ACTION target (root node)
		pytest.param(
			{											# input_data
				"math_problem": "What is 5+3?",
			},
			"action",  									# scoring_target
			None,  										# reasoning_candidates
			(											# instructions
				"""
Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
""".strip()
			),
			1,  										# num_queries (action target with None candidates defaults to 1 for root node)
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to decide what action to take for the next reasoning step for a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

This task requires taking a sequence of reasoning steps to reach a solution.
You must determine what action to take next.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the action under consideration under the "# Action" heading in the Document.

Judge whether the provided action is likely to be a good next step for addressing the user's task.
NOTE: You have 3 actions remaining before you must return a final answer.
<Query>: # Inputs
math_problem: What is 5+3?
""",
			],
			None,  										# expected_error
			id="action_no_candidates_defaults_to_one_query",
		),
		# Edge case: No reasoning candidates for REASONING target (should still work)
		pytest.param(
			{											# input_data
				"math_problem": "What is 1+1?",
			},
			"reasoning",  								# scoring_target
			None,  										# reasoning_candidates
			(											# instructions
				"""
Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
""".strip()
			),
			1,  										# num_queries (reasoning target with None candidates defaults to 1 for root node)
			[  											# expected_queries
				"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Your objective is to judge a reasoning trajectory towards solving a user-assigned task.

The user provided the following task:
"Solve the provided math problem and return its answer."

Since this is a reasoning task, we are interested not only in the final output, but also in the reasoning process that leads to it.
You will find the inputs for this task under the "# Inputs" header in the Query.
You will find the intermediate reasoning trajectory towards solving this problem under the "# Reasoning" header in the Document.

Judge whether the provided reasoning trajectory is a strong partial solution for the task given the inputs.
<Query>: # Inputs
math_problem: What is 1+1?
""",
			],
			None,  										# expected_error
		id="reasoning_no_candidates_defaults_to_one_query",
			),
		],
	)
	def test_format_queries_generates_valid_prompts(
		self,
		scoring_adapter: LocalVLLMScoringAdapter,
		input_data: Input,
		scoring_target: Literal["reasoning", "output", "action"],
		reasoning_candidates: list[Reasoning] | None,
		instructions: str | list[str],
		num_queries: int,
		expected_queries: list[str] | None,
		expected_error: type[Exception] | None,
	) -> None:
		"""
		Test that format_queries generates exact expected output strings.

		Verifies that the formatted queries exactly match the expected output
		for various combinations of signatures, scoring targets, and candidates.

		Args:
			scoring_adapter: The scoring adapter instance for testing.
			input_data: Input dictionary for the task.
			scoring_target: The scoring target (reasoning/output/action).
			reasoning_candidates: Reasoning candidates.
			instructions: Instruction string(s) for the queries.
			num_queries: Number of queries to format.
			expected_queries: List of expected query strings.
			expected_error: Expected exception type if error should occur.
		"""
		formatted_reasoning_candidates = _serialize_reasoning_candidates(reasoning_candidates)
		if expected_error is not None:
			with pytest.raises(expected_error):
				scoring_adapter.format_queries(
					instructions=instructions,
					inputs=input_data,
					scoring_target=scoring_target,
					num_queries=num_queries,
					reasoning_candidates=formatted_reasoning_candidates,
				)
		else:
			formatted_queries = scoring_adapter.format_queries(
				instructions=instructions,
				scoring_target=scoring_target,
				inputs=input_data,
				num_queries=num_queries,
				reasoning_candidates=formatted_reasoning_candidates,
			)

			# Should generate expected number of queries
			assert expected_queries is not None
			assert len(formatted_queries) == len(expected_queries), (
				f"Expected {len(expected_queries)} queries, got {len(formatted_queries)}"
			)

			# Each query should exactly match the expected formatted query string.
			for query_idx, (formatted_query, expected_query) in enumerate(
				zip(formatted_queries, expected_queries, strict=True)
			):
				assert formatted_query == expected_query, (
					f"Query {query_idx} does not match expected output.\n"
					f"Expected:\n{expected_query}\n\n"
					f"Got:\n{formatted_query}"
				)

class TestFormatDocuments:
	"""Test cases for format_documents method."""

	@pytest.mark.parametrize(
		# Parameter names
		[
			"signature",
			"scoring_target",
			"input_data",
			"reasoning_candidates",
			"output_candidates",
			"action_candidates",
			"include_reasoning_in_document_for_output",
			"expected_documents",
			"expected_error",
		],
		# Parameter values
		[
			# REASONING target tests with SolveMathProblemWithReasoning
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"reasoning",  								# scoring_target
				{											# input_data
					"math_problem": "Compute 2+2",
				},
				[  											# reasoning_candidates
					[{"math_operation": 'First, add 2 and 2'}]
				],
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Reasoning
<thinking>
<step>
## math_operation
First, add 2 and 2
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_single_candidate_math_problem",
			),
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"reasoning",  								# scoring_target
				{											# input_data
					"math_problem": "Compute 5*6"
				},
				[  											# reasoning_candidates
					[{"math_operation": 'Multiplying 5 by 6 yields 30'}],
					[{"math_operation": '5 * 6 = 33'}],
				],
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					# first candidate
					"""<Document>: # Reasoning
<thinking>
<step>
## math_operation
Multiplying 5 by 6 yields 30
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					# second candidate
					"""<Document>: # Reasoning
<thinking>
<step>
## math_operation
5 * 6 = 33
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_math_problem",
			),
			# REASONING target tests with GenerateArgumentWithReasoning
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				"reasoning",  								# scoring_target
				{  											# input_data
					"topic": "AI safety research should recieve significant funding",
					"stance": "PRO",
				},
				[  											# reasoning_candidates
					[{"claim": 'AI can automate dangerous tasks'}]
				],
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Reasoning
<thinking>
<step>
## claim
AI can automate dangerous tasks
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_single_candidate_generate_argument",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				"reasoning",  								# scoring_target
				{  											# input_data
					"topic": "Climate change requires immediate action",
					"stance": "PRO",
				},
				[  											# reasoning_candidates
					[{"claim": 'Climate change requires immediate action'}],
					[{"claim": 'Renewable energy is cost-effective'}],
				],
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Reasoning
<thinking>
<step>
## claim
Climate change requires immediate action
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					"""<Document>: # Reasoning
<thinking>
<step>
## claim
Renewable energy is cost-effective
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_multiple_candidates_generate_argument",
			),
			# REASONING target with multiple reasoning steps
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"reasoning",  								# scoring_target
				{											# input_data
					"math_problem": "Compute 1+1"
				},
				[  											# reasoning_candidates
					[
						{"math_operation": 'First step'},
						{"math_operation": 'Second step'},
						{"math_operation": 'Third step'},
					]
				],
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Reasoning
<thinking>
<step>
## math_operation
First step
</step>
<step>
## math_operation
Second step
</step>
<step>
## math_operation
Third step
</step><|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="reasoning_multiple_steps_single_candidate",
			),
			# OUTPUT target tests with SolveMathProblemWithReasoning
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"output",  									# scoring_target
				{											# input_data
					"math_problem": "Compute 3*4",
				},
				[  											# reasoning_candidates
					[{"math_operation": '3 times 4'}]
				],
				[  											# output_candidates
					{"answer": "12"}
				],
				None,  										# action_candidates
				True,  										# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Reasoning
<thinking>
<step>
## math_operation
3 times 4
</step>
</thinking>

# Output
answer: 12<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_single_candidate_with_reasoning_math_problem",
			),
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"output",  									# scoring_target
				{											# input_data
					"math_problem": "Compute 3*4"
				},
				[  											# reasoning_candidates
					[{"math_operation": '3 times 4'}],
					[{"math_operation": 'Multiply 3 and 4'}],
				],
				[  											# output_candidates
					{"answer": "12"},
					{"answer": "12"},
				],
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Output
answer: 12<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					"""<Document>: # Output
answer: 12<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_multiple_candidates_with_reasoning_math_problem",
			),
			# OUTPUT target tests with GenerateArgumentWithReasoning
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				"output",  									# scoring_target
				{  											# input_data
					"topic": "Renewable energy is a top priority for society",
					"stance": "PRO",
				},
				[  											# reasoning_candidates
					[{"claim": 'Renewable energy reduces carbon emissions'}]
				],
				[  											# output_candidates
					{"argument": "Renewable energy is essential for combating climate change."}
				],
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					"""<Document>: # Output
argument: Renewable energy is essential for combating climate change.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_single_candidate_with_reasoning_generate_argument",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				"output",  									# scoring_target
				{  											# input_data
					"topic": "Space Exploration is pointless",
					"stance": "ANTI",
				},
				[  											# reasoning_candidates
					[{"claim": 'Space exploration is expensive'}],
					[{"claim": 'Resources better spent on Earth'}],
				],
				[  											# output_candidates
					{"argument": "Space exploration diverts funds from pressing Earth issues."},
					{"argument": "We should prioritize solving problems on Earth first."},
				],
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					# first candidate
					"""<Document>: # Output
argument: Space exploration diverts funds from pressing Earth issues.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					# second candidate
					"""<Document>: # Output
argument: We should prioritize solving problems on Earth first.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_multiple_candidates_with_reasoning_generate_argument",
			),
			# OUTPUT target with multiple output fields
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				"output",  									# scoring_target
				{  											# input_data
					"topic": "Renewable energy is essential for combating climate change",
					"stance": "PRO",
				},
				[  											# reasoning_candidates
					[
						{"claim": 'Renewable energy reduces carbon emissions'},
						{"claim": 'It creates jobs'}
					]
				],
				[  											# output_candidates
					{
						"argument": "Renewable energy is essential.",
						"conclusion": "We must transition now."
					}
				],
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					# first candidate
					"""<Document>: # Output
argument: Renewable energy is essential.
conclusion: We must transition now.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="output_multiple_fields_single_candidate",
			),
			# ACTION target tests
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"action",  									# scoring_target
				{"math_problem": "Compute 7+8"},			# input_data
				[  											# reasoning_candidates
					[{"math_operation": 'Start by adding 7 and 8'}]
				],
				None,  										# output_candidates
				["continue_reasoning"],						# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					# first candidate
					"""<Document>: # Action
continue_reasoning<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="action_single_candidate_with_reasoning_math_problem",
			),
			pytest.param(
				GenerateArgumentWithReasoning,  			# signature
				"action",  									# scoring_target
				{  											# input_data
					"topic": "Space Exploration is pointless",
					"stance": "ANTI",
				},
				[  											# reasoning_candidates
					[{"claim": 'Space exploration is expensive'}],
					[{"claim": 'Resources better spent on Earth'}],
				],
				None,  										# output_candidates
				[  											# action_candidates
					(
						"Action Name: causal_reasoning\n\n"
						"Description: Perform the following actions in the upcoming reasoning step:\n"
						"* States causes, effects, consequences, or logical implications."
					),
					"finish",
				],
				False,  									# include_reasoning_in_document_for_output
				[  											# expected_documents
					# first candidate
					"""<Document>: # Action
Action Name: causal_reasoning

Description: Perform the following actions in the upcoming reasoning step:
* States causes, effects, consequences, or logical implications.<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
					# second candidate
					"""<Document>: # Action
finish<|im_end|>
<|im_start|>assistant
<think>

</think>

""",
				],
				None,  										# expected_error
				id="action_multiple_candidates_with_reasoning_generate_argument",
			),
			# Edge case: REASONING with no candidates (should return empty list)
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"reasoning",  								# scoring_target
				{											# input_data
					"math_problem": "Compute 2+2",
				},
				None,  										# reasoning_candidates
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[],  										# expected_documents
				None,  										# expected_error
				id="reasoning_no_candidates_returns_empty",
			),
			# Edge case: OUTPUT with no candidates (should return empty list)
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"output",  									# scoring_target
				{											# input_data
					"math_problem": "Compute 2+2",
				},
				None,  										# reasoning_candidates
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[],  										# expected_documents
				None,  										# expected_error
				id="output_no_candidates_returns_empty",
			),
			# Edge case: ACTION with no candidates (should return empty list)
			pytest.param(
				SolveMathProblemWithReasoning,  			# signature
				"action",  									# scoring_target
				{											# input_data
					"math_problem": "Compute 2+2",
				},
				None,  										# reasoning_candidates
				None,  										# output_candidates
				None,  										# action_candidates
				False,  									# include_reasoning_in_document_for_output
				[],  										# expected_documents
				None,  										# expected_error
				id="action_no_candidates_returns_empty",
			),
		],
	)
	def test_format_documents_generates_valid_prompts(
		self,
		scoring_adapter: LocalVLLMScoringAdapter,
		signature: type[ReasoningSignature],
		scoring_target: Literal["reasoning", "output", "action"],
		input_data: Input,
		reasoning_candidates: list[Reasoning] | None,
		output_candidates: list[Output] | None,
		action_candidates: list[str] | None,
		include_reasoning_in_document_for_output: bool,
		expected_documents: list[str],
		expected_error: type[Exception] | None,
	) -> None:
		"""
		Test that format_documents generates exact expected output strings.

		Verifies that the formatted documents exactly match the expected output
		for various combinations of signatures, scoring targets, and candidates.

		Args:
			scoring_adapter: The scoring adapter instance for testing.
			signature: The scoring signature to test with.
			scoring_target: The scoring target (reasoning/output/action).
			reasoning_candidates: List of reasoning candidates.
			output_candidates: List of output candidates.
			action_candidates: List of action candidate strings.
			expected_documents: List of expected document strings.
			expected_error: Expected exception type if error should occur.
		"""
		if expected_error is not None:
			with pytest.raises(expected_error):
				scoring_adapter.format_documents(
					scoring_target=scoring_target,
					reasoning_candidates=_serialize_reasoning_candidates(
						reasoning_candidates,
						output_candidates=output_candidates,
					),
					output_candidates=output_candidates,
					action_candidates=action_candidates,
					include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
				)
		else:
			formatted_documents = scoring_adapter.format_documents(
				scoring_target=scoring_target,
				reasoning_candidates=_serialize_reasoning_candidates(
					reasoning_candidates,
					output_candidates=output_candidates,
				),
				output_candidates=output_candidates,
				action_candidates=action_candidates,
				include_reasoning_in_document_for_output=include_reasoning_in_document_for_output,
			)

			# Should generate expected number of documents
			assert len(formatted_documents) == len(expected_documents), (
				f"Expected {len(expected_documents)} documents, got {len(formatted_documents)}"
			)

			# Each document should exactly match the expected string
			for i, (formatted_doc, expected_doc) in enumerate(
				zip(formatted_documents, expected_documents, strict=True)
			):
				assert formatted_doc == expected_doc, (
					f"Document {i} mismatch:\n"
					f"Expected:\n{expected_doc}\n\n"
					f"Got:\n{formatted_doc}"
				)

if __name__ == "__main__":
	pytest.main([__file__, "-vv"])
