"""
Tests for the controller module.

Expected usage:
```bash
pytest predict/test_controller.py -vv
```
"""

# Standard library imports
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Literal

# Third-party imports
import dspy
import pytest
import torch
from dspy.adapters.utils import get_annotation_name

# Local imports
from constants import OpenSourceModel
from lm.generative_local_lm import GenerativeLocalVLLM
from misc_utils import ExecutionError
from predict.controller.controller import TreeOfThoughtsController
from predict.controller.controller_constants import (
	DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	ControllerOutput,
)
from predict.controller.controller_demos import (
	ARGUMENT_CONTINUE_FINISH_DEMOS,
	STRUCTURE_CONTROLLER_DEMOS,
	STYLE_CONTROLLER_DEMOS,
	STYLE_STRUCTURE_CONTROLLER_DEMOS,
)
from predict.controller.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	PRUNE_TOOL,
	ControllerPrediction,
	ReasoningIntervention,
	return_action_if_single_option,
)
from signatures import (
	GenerateArgumentWithReasoning,
	QuestionAnsweringWithReasoning,
	ReasoningSignature,
	SolveMathProblemWithReasoning,
)
from tree import State
from utilities_for_tests import (
	MockGenerativeLocalVLLM,
	MockPredict,
)

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Test Fixtures for Action Space JSONs
# =============================================================================


@pytest.fixture
def temp_action_space_styles(tmp_path):
	"""Create a temporary JSON file for causal styles with 2 options."""
	styles_json = {
		"name": "style",
		"definition": (
			"Forces the next reasoning step to adopt a specific rhetorical style or "
			"expressive technique, controlling how arguments are articulated and presented. "
			"Interventions along this dimension ensure the next step uses a particular mode "
			"of expression (e.g., figurative language, statistical evidence, narrative "
			"storytelling, formal tone, or direct audience engagement)."
		),
		"choices": {
			"Figurative Language": {
				"definition": "Use metaphor, simile, analogy, or symbolism to make ideas concrete.",
				"internal_reasoning": (
					"I should employ non-literal comparison to make abstract concepts vivid. "
				),
			},
			"Statistical & Data-Driven": {
				"definition": "Present numerical data, statistics, or quantified evidence.",
				"internal_reasoning": (
					"I should use numbers and data to provide concrete, measurable support. "
				),
			},
		},
	}
	styles_path = tmp_path / "style.json"
	with open(styles_path, "w") as f:
		json.dump(styles_json, f, indent="\t")
	return str(styles_path)


@pytest.fixture
def temp_action_space_structures(tmp_path):
	"""Create a temporary JSON file for causal structures with 2 options."""
	structures_json = {
		"name": "structure",
		"definition": (
			"Forces the next reasoning step to adhere to a specific discourse structure, "
			"controlling how ideas connect and relate to each other. Interventions along "
			"this dimension ensure the next step follows a particular organizational pattern "
			"(e.g., presenting a counterpoint, providing evidence, drawing a causal inference, "
			"or offering an example)."
		),
		"choices": {
			"Causal Reasoning": {
				"definition": "State causes, effects, consequences, or logical implications.",
				"internal_reasoning": "I should use causal reasoning.",
				"prefix": "Therefore",
			},
			"Evidence & Support": {
				"definition": "Cite facts, studies, expert testimony, or documented sources.",
				"prefix": "According to",
			},
			"Contrast": {
				"definition": "Present contrasting viewpoints, counterarguments, or exceptions.",
				"prefix": "However",
			},
			"Chronological Sequence": {
				"definition": "Order events or points in time.",
				"prefix": "First",
			},
		},
	}
	structures_path = tmp_path / "structure.json"
	with open(structures_path, "w") as f:
		json.dump(structures_json, f, indent="\t")
	return str(structures_path)


@pytest.fixture
def temp_action_space_subtopics(tmp_path):
	"""Create a temporary JSON file for causal subtopics with 2 options."""
	subtopics_json = {
		"name": "stock_issue",
		"definition": (
			"Forces the next reasoning step to address a specific thematic angle or "
			"argumentative dimension. Interventions along this dimension ensure the next "
			"step focuses on a particular aspect of the debate (e.g., economic impact, "
			"social justice, health and safety, environmental concerns, personal freedom, "
			"practical feasibility, moral principles, or legal issues)."
		),
		"choices": {
			"General Introduction": {
				"definition": "Provide a broad overview, context, or introductory statements without specific thematic focus.",
				"internal_reasoning": (
					"I should provide a general introduction to the topic. "
				),
			},
			"Economic Impact": {
				"definition": "Discuss financial costs, benefits, market effects, or resource allocation.",
				"internal_reasoning": (
					"I should focus on the economic and financial implications of this issue. "
				),
			},
			"Social Justice & Equity": {
				"definition": "Address fairness, equality, discrimination, or marginalized groups.",
				"internal_reasoning": (
					"I should consider the fairness and equity dimensions of this issue. "
				),
			},
		},
	}
	subtopics_path = tmp_path / "stock_issue.json"
	with open(subtopics_path, "w") as f:
		json.dump(subtopics_json, f, indent="\t")
	return str(subtopics_path)


# =============================================================================
# Helper Functions for Creating Expected Tools
# =============================================================================


def create_expected_tool_for_styles() -> dspy.Tool:
	"""Create expected tool for causal styles dimension only."""

	def tool_func(
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning from the fixture
		internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=internal_reasoning_map[style],
			prefix="",
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"style": """
Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
Options:
- "Figurative Language": Use metaphor, simile, analogy, or symbolism to make ideas concrete.
- "Statistical & Data-Driven": Present numerical data, statistics, or quantified evidence.
""".strip()
		},
		arg_types={"style": Literal["Figurative Language", "Statistical & Data-Driven"]},
	)


def create_expected_tool_for_structures() -> dspy.Tool:
	"""Create expected tool for causal structures dimension only."""

	def tool_func(
		structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
	) -> ReasoningIntervention:
		# Use the actual prefix from the fixture
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning="",
			prefix=prefix_map[structure],
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"structure": """
Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
Options:
- "Causal Reasoning": State causes, effects, consequences, or logical implications.
- "Evidence & Support": Cite facts, studies, expert testimony, or documented sources.
- "Contrast": Present contrasting viewpoints, counterarguments, or exceptions.
- "Chronological Sequence": Order events or points in time.
""".strip()
		},
		arg_types={"structure": Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"]},
	)


def create_expected_tool_for_styles_and_structures() -> dspy.Tool:
	"""Create expected tool for causal styles and structures dimensions combined."""

	def tool_func(
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
		structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning and prefix from the fixtures
		internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
		}
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=internal_reasoning_map[style],
			prefix=prefix_map[structure],
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"style": """
Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
Options:
- "Figurative Language": Use metaphor, simile, analogy, or symbolism to make ideas concrete.
- "Statistical & Data-Driven": Present numerical data, statistics, or quantified evidence.
""".strip(),
			"structure": """
Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
Options:
- "Causal Reasoning": State causes, effects, consequences, or logical implications.
- "Evidence & Support": Cite facts, studies, expert testimony, or documented sources.
- "Contrast": Present contrasting viewpoints, counterarguments, or exceptions.
- "Chronological Sequence": Order events or points in time.
""".strip(),
		},
		arg_types={
			"style": Literal["Figurative Language", "Statistical & Data-Driven"],
			"structure": Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
		},
	)


def create_expected_tool_for_subtopics_styles_and_structures() -> dspy.Tool:
	"""Create expected tool for all three dimensions: subtopics, styles, and structures."""

	def tool_func(
		stock_issue: Literal["General Introduction", "Economic Impact", "Social Justice & Equity"],
		style: Literal["Figurative Language", "Statistical & Data-Driven"],
		structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
	) -> ReasoningIntervention:
		# Use the actual internal_reasoning and prefix from the fixtures
		# Combine internal_reasoning from both stock_issue and style
		stock_issue_internal_reasoning_map = {
			"General Introduction": "I should provide a general introduction to the topic. ",
			"Economic Impact": "I should focus on the economic and financial implications of this issue. ",
			"Social Justice & Equity": "I should consider the fairness and equity dimensions of this issue. ",
		}
		style_internal_reasoning_map = {
			"Figurative Language": "I should employ non-literal comparison to make abstract concepts vivid. ",
			"Statistical & Data-Driven": "I should use numbers and data to provide concrete, measurable support. ",
		}
		prefix_map = {
			"Causal Reasoning": "Therefore",
			"Evidence & Support": "According to",
			"Contrast": "However",
			"Chronological Sequence": "First",
		}
		combined_internal_reasoning = (
			stock_issue_internal_reasoning_map[stock_issue] + style_internal_reasoning_map[style]
		)
		return ReasoningIntervention(
			continue_reasoning=True,
			internal_reasoning=combined_internal_reasoning,
			prefix=prefix_map[structure],
		)

	return dspy.Tool(
		name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		func=tool_func,
		desc="""
Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.
You **must** select **one** choice for **each** of the provided dimensions.
Be mindful of the impact that each choice has on the next reasoning step.
When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
""".strip(),
		args={
			"stock_issue": """
Forces the next reasoning step to address a specific thematic angle or argumentative dimension. Interventions along this dimension ensure the next step focuses on a particular aspect of the debate (e.g., economic impact, social justice, health and safety, environmental concerns, personal freedom, practical feasibility, moral principles, or legal issues).
Options:
- "General Introduction": Provide a broad overview, context, or introductory statements without specific thematic focus.
- "Economic Impact": Discuss financial costs, benefits, market effects, or resource allocation.
- "Social Justice & Equity": Address fairness, equality, discrimination, or marginalized groups.
""".strip(),
			"style": """
Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
Options:
- "Figurative Language": Use metaphor, simile, analogy, or symbolism to make ideas concrete.
- "Statistical & Data-Driven": Present numerical data, statistics, or quantified evidence.
""".strip(),
			"structure": """
Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
Options:
- "Causal Reasoning": State causes, effects, consequences, or logical implications.
- "Evidence & Support": Cite facts, studies, expert testimony, or documented sources.
- "Contrast": Present contrasting viewpoints, counterarguments, or exceptions.
- "Chronological Sequence": Order events or points in time.
""".strip(),
		},
		arg_types={
			"stock_issue": Literal["General Introduction", "Economic Impact", "Social Justice & Equity"],
			"style": Literal["Figurative Language", "Statistical & Data-Driven"],
			"structure": Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"],
		},
	)


# =============================================================================
# GPU Skip Markers
# =============================================================================

pytestmark_gpu = pytest.mark.skipif(
	not torch.cuda.is_available(),
	reason="GPU tests require GPU access",
)


# =============================================================================
# Test State Data
# =============================================================================



BATCH_STATES_INPUTS = [
	{"question": "What is 2+2?"},
	{"question": "What is 3*5?"},
	{"question": "What is the capital of France?"},
]

BATCH_STATES_REASONING = [
	[{"reasoning_step": "Addition problem"}],
	[{"reasoning_step": "Multiplication problem"}],
	[{"reasoning_step": "Geography question"}],
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def configure_mock_lm():
	"""Automatically configure MockGenerativeLocalVLLM for all non-GPU tests."""
	mock_lm = MockGenerativeLocalVLLM()
	dspy.settings.configure(lm=mock_lm)
	yield


@pytest.fixture
def simple_state():
	"""Create a simple test state for controller testing."""
	return State(
		input={"question": "What is 2+2?"},
		reasoning=[{"reasoning_step": "First, I need to think about addition."}],
		controller_outputs=[],
		feedback=[],
		output={},
	)


@pytest.fixture
def batch_states():
	"""Create multiple test states for batch testing."""
	states = []
	for i in range(len(BATCH_STATES_INPUTS)):
		state = State(
			input=BATCH_STATES_INPUTS[i],
			reasoning=BATCH_STATES_REASONING[i],
			controller_outputs=[],
			feedback=[],
			output={},
		)
		states.append(state)
	return states


# =============================================================================
# Test: Controller Initialization (__init__)
# =============================================================================

@pytest.mark.parametrize(
	# Parameter names
	[
		"signature",
		"max_reasoning_steps",
		"tools",
		"action_space_paths",
		"early_stopping_enabled",
		"use_native_tool_calls",
		"expected_tools",
		"expected_instructions",
		"expected_input_fields",
		"expected_output_fields",
		"expected_exception",
	],
	[
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"continue_reasoning": DEFAULT_TOOL,
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Answer the provided question with step-by-step reasoning.

You are given `question` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `reasoning_step` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `continue_reasoning`:
        * Description:
                Signals that additional reasoning is needed to solve the task. If the reasoning so far is insufficient, the answer seems uncertain, or you believe there is a mistake in the reasoning so far, select this option.
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"question": {"annotation": "str", "desc": "The question to answer"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `reasoning_step` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
			},
			None,								# expected_exception
			id="qa_signature_default_tools_early_stopping",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			True,								# use_native_tool_calls
			{									# expected_tools
				"continue_reasoning": DEFAULT_TOOL,
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Answer the provided question with step-by-step reasoning.

You are given `question` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `reasoning_step` header above the content of the step itself.
""".strip()),
			{									# expected_input_fields
				"question": {"annotation": "str", "desc": "The question to answer"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `reasoning_step` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
				"tools": {
					"annotation": get_annotation_name(list[dspy.Tool]),
					"desc": "Available tools to influence the next reasoning step.",
				},
			},
			{									# expected_output_fields
				"tool_calls": {
					"annotation": get_annotation_name(dspy.ToolCalls),
					"desc": "The tool call to influence the next reasoning step.",
				}
			},
			None,
			id="qa_signature_default_tools_early_stopping_native_tool_calls",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			False,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{"continue_reasoning": DEFAULT_TOOL}, # expected_tools
			(									# expected_instructions
"""Answer the provided question with step-by-step reasoning.

You are given `question` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `reasoning_step` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `continue_reasoning`:
        * Description:
                Signals that additional reasoning is needed to solve the task. If the reasoning so far is insufficient, the answer seems uncertain, or you believe there is a mistake in the reasoning so far, select this option.
""".strip()),
			{									# expected_input_fields
				"question": {"annotation": "str", "desc": "The question to answer"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `reasoning_step` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
			},
			None,
			id="qa_signature_default_tools_no_early_stopping",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			False,								# early_stopping_enabled
			True,								# use_native_tool_calls
			{"continue_reasoning": DEFAULT_TOOL}, # expected_tools
			(									# expected_instructions
"""Answer the provided question with step-by-step reasoning.

You are given `question` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `reasoning_step` header above the content of the step itself.
""".strip()),
			{									# expected_input_fields
				"question": {"annotation": "str", "desc": "The question to answer"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `reasoning_step` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
				"tools": {
					"annotation": get_annotation_name(list[dspy.Tool]),
					"desc": "Available tools to influence the next reasoning step.",
				},
			},
			{									# expected_output_fields
				"tool_calls": {
					"annotation": get_annotation_name(dspy.ToolCalls),
					"desc": "The tool call to influence the next reasoning step.",
				}
			},
			None,
			id="qa_signature_default_tools_no_early_stopping_native_tool_calls",
		),
		pytest.param(
			SolveMathProblemWithReasoning,		# signature
			10,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"continue_reasoning": DEFAULT_TOOL,
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Solve the provided math problem and return its answer.

You are given `math_problem` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `math_operation` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `continue_reasoning`:
        * Description:
                Signals that additional reasoning is needed to solve the task. If the reasoning so far is insufficient, the answer seems uncertain, or you believe there is a mistake in the reasoning so far, select this option.
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"math_problem": {"annotation": "str", "desc": "The math problem to solve"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `math_operation` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str", "desc": "The thought process behind choosing the next action."
				},
				"action": {
					"annotation": "str", "desc": "The selected action (tool) to guide the next reasoning step."
				},
			},
			None,
			id="math_signature_default_tools",
		),
		pytest.param(
			SolveMathProblemWithReasoning,		# signature
			10,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			True,								# use_native_tool_calls
			{									# expected_tools
				"continue_reasoning": DEFAULT_TOOL,
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Solve the provided math problem and return its answer.

You are given `math_problem` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `math_operation` header above the content of the step itself.
""".strip()),
			{									# expected_input_fields
				"math_problem": {"annotation": "str", "desc": "The math problem to solve"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `math_operation` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
				"tools": {
					"annotation": get_annotation_name(list[dspy.Tool]),
					"desc": "Available tools to influence the next reasoning step.",
				},
			},
			{									# expected_output_fields
				"tool_calls": {
					"annotation": get_annotation_name(dspy.ToolCalls),
					"desc": "The tool call to influence the next reasoning step.",
				}
			},
			None,
			id="math_signature_default_tools_native_tool_calls",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"styles",							# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_styles(),
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `argument` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `argument`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `claim` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`:
        * Description:
                Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step. You **must** select **one** choice for **each** of the provided dimensions. Be mindful of the impact that each choice has on the next reasoning step. When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
        * Arguments:
                - style: Literal["Figurative Language", "Statistical & Data-Driven"]
                        Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"topic": {"annotation": "str", "desc": "The topic to generate an argument about"},
				"stance": {"annotation": "Literal['PRO', 'ANTI']", "desc": "The stance to take on the topic"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `argument`. "
						"Each step's content is under the `claim` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `argument`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
				"action_arguments": {
					"annotation": "dict[str, Any]",
					"desc": "The input arguments for the selected action (tool).",
				},
			},
			None,
			id="generate_argument_style_intervention",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"styles",							# action_space_paths
			True,								# early_stopping_enabled
			True,								# use_native_tool_calls
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_styles(),
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `argument` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `argument`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `claim` header above the content of the step itself.
""".strip()),
			{									# expected_input_fields
				"topic": {"annotation": "str", "desc": "The topic to generate an argument about"},
				"stance": {"annotation": "Literal['PRO', 'ANTI']", "desc": "The stance to take on the topic"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `argument`. "
						"Each step's content is under the `claim` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `argument`."
					),
				},
				"tools": {
					"annotation": get_annotation_name(list[dspy.Tool]),
					"desc": "Available tools to influence the next reasoning step.",
				},
			},
			{									# expected_output_fields
				"tool_calls": {
					"annotation": get_annotation_name(dspy.ToolCalls),
					"desc": "The tool call to influence the next reasoning step.",
				}
			},
			None,
			id="generate_argument_style_intervention_native_tool_calls",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"structures",						# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_structures(),
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `argument` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `argument`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `claim` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`:
        * Description:
                Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step. You **must** select **one** choice for **each** of the provided dimensions. Be mindful of the impact that each choice has on the next reasoning step. When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
        * Arguments:
                - structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"]
                        Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"topic": {"annotation": "str", "desc": "The topic to generate an argument about"},
				"stance": {"annotation": "Literal['PRO', 'ANTI']", "desc": "The stance to take on the topic"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `argument`. "
						"Each step's content is under the `claim` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `argument`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
				"action_arguments": {
					"annotation": "dict[str, Any]",
					"desc": "The input arguments for the selected action (tool).",
				},
			},
			None,
			id="generate_argument_structure_intervention",
		),
		pytest.param(
			GenerateArgumentWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"styles_structures",				# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_styles_and_structures(),
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Generate an argument which takes the provided stance towards the provided topic.

You are given `topic` and `stance` and your goal is to finish with `argument`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `argument` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `argument`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `claim` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`:
        * Description:
                Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step. You **must** select **one** choice for **each** of the provided dimensions. Be mindful of the impact that each choice has on the next reasoning step. When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
        * Arguments:
                - structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"]
                        Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
                - style: Literal["Figurative Language", "Statistical & Data-Driven"]
                        Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"topic": {"annotation": "str", "desc": "The topic to generate an argument about"},
				"stance": {"annotation": "Literal['PRO', 'ANTI']", "desc": "The stance to take on the topic"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `argument`. "
						"Each step's content is under the `claim` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `argument`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
				"action_arguments": {
					"annotation": "dict[str, Any]",
					"desc": "The input arguments for the selected action (tool).",
				},
			},
			None,
			id="generate_argument_style_structure_intervention",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"subtopics_styles_structures",		# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"intervene_on_next_reasoning_step": create_expected_tool_for_subtopics_styles_and_structures(),
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Answer the provided question with step-by-step reasoning.

You are given `question` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `reasoning_step` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `intervene_on_next_reasoning_step`:
        * Description:
                Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step. You **must** select **one** choice for **each** of the provided dimensions. Be mindful of the impact that each choice has on the next reasoning step. When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly.
        * Arguments:
                - stock_issue: Literal["General Introduction", "Economic Impact", "Social Justice & Equity"]
                        Forces the next reasoning step to address a specific thematic angle or argumentative dimension. Interventions along this dimension ensure the next step focuses on a particular aspect of the debate (e.g., economic impact, social justice, health and safety, environmental concerns, personal freedom, practical feasibility, moral principles, or legal issues).
                - structure: Literal["Causal Reasoning", "Evidence & Support", "Contrast", "Chronological Sequence"]
                        Forces the next reasoning step to adhere to a specific discourse structure, controlling how ideas connect and relate to each other. Interventions along this dimension ensure the next step follows a particular organizational pattern (e.g., presenting a counterpoint, providing evidence, drawing a causal inference, or offering an example).
                - style: Literal["Figurative Language", "Statistical & Data-Driven"]
                        Forces the next reasoning step to adopt a specific rhetorical style or expressive technique, controlling how arguments are articulated and presented. Interventions along this dimension ensure the next step uses a particular mode of expression (e.g., figurative language, statistical evidence, narrative storytelling, formal tone, or direct audience engagement).
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"question": {"annotation": "str", "desc": "The question to answer"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `reasoning_step` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
				"action_arguments": {
					"annotation": "dict[str, Any]",
					"desc": "The input arguments for the selected action (tool).",
				},
			},
			None,
			id="qa_signature_subtopic_style_structure_intervention",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			3,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			None,								# action_space_paths
			True,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{									# expected_tools
				"continue_reasoning": DEFAULT_TOOL,
				"finish": FINISH_TOOL,
			},
			(									# expected_instructions
"""Answer the provided question with step-by-step reasoning.

You are given `question` and your goal is to finish with `answer`.
To accomplish this goal, you will need to reason about the problem step by step rather than generating `answer` directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating `answer`.
Refer to the existing reasoning steps under the `reasoning` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `reasoning_step` header above the content of the step itself.
Choose a tool to use from the following options:
(1) `continue_reasoning`:
        * Description:
                Signals that additional reasoning is needed to solve the task. If the reasoning so far is insufficient, the answer seems uncertain, or you believe there is a mistake in the reasoning so far, select this option.
(2) `finish`:
        * Description:
                Signals that the reasoning so far is sufficient for producing a high-quality response for the task. If selected, the next step will involve generating the final output rather than reasoning further.
""".strip()),
			{									# expected_input_fields
				"question": {"annotation": "str", "desc": "The question to answer"},
				"reasoning": {
					"annotation": "str",
					"desc": (
						"The existing reasoning steps towards producing `answer`. "
						"Each step's content is under the `reasoning_step` header."
					),
				},
				"number_of_additional_reasoning_steps": {
					"annotation": "int",
					"desc": (
						"The maximum number of additional reasoning steps you can take before you "
						"must produce `answer`."
					),
				},
			},
			{									# expected_output_fields
				"considerations": {
					"annotation": "str",
					"desc": "The thought process behind choosing the next action.",
				},
				"action": {
					"annotation": "str",
					"desc": "The selected action (tool) to guide the next reasoning step.",
				},
			},
			None,
			id="no_tools_no_action_space_defaults_to_default_tool",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# provided tools (None = default tools)
			"missing_action_space",				# action_space_paths
			False,								# early_stopping_enabled
			False,								# use_native_tool_calls
			{},									# expected_tools
			"",									# expected_instructions
			{},									# expected_input_fields
			{},									# expected_output_fields
			FileNotFoundError,					# expected_exception
			id="missing_action_space_path_raises",
		),
	],
)
def test_controller_initialization(
	signature: type[ReasoningSignature],
	max_reasoning_steps: int,
	tools: list[dspy.Tool] | None,
	action_space_paths: str | None,
	early_stopping_enabled: bool,
	use_native_tool_calls: bool,
	expected_tools: dict[str, dspy.Tool] | None,
	expected_instructions: str,
	expected_input_fields: dict[str, dict[str, str | None]],
	expected_output_fields: dict[str, dict[str, str | None]],
	expected_exception: type[Exception] | None,
	temp_action_space_styles,
	temp_action_space_structures,
	temp_action_space_subtopics,
) -> None:
	"""
	Test TreeOfThoughtsController initialization with various configurations.

	This test verifies that:
	1. The controller properly initializes with different reasoning signatures
	2. The exact expected tools are created, including descriptions, args, and arg_types
	3. The input and output fields of decide_next_step_single are validated

	Parameters:
	    signature: The reasoning signature class to test
	    max_reasoning_steps: Maximum number of reasoning steps allowed
	    tools: Tools configuration (None uses defaults)
		action_space_paths: Fixture key for action space paths or None
	    expected_reasoning_field: Expected name of the primary reasoning field
		expected_tools: Dictionary mapping tool names to expected Tool objects
		expected_decide_input_fields: Expected input fields for decide_next_step_single
		expected_decide_output_fields: Expected output fields for decide_next_step_single
	    early_stopping_enabled: Whether to enable early stopping
	    expected_exception: Expected exception type or None if no exception
	"""
	# Convert fixture key to actual paths
	if action_space_paths == "styles":
		actual_paths = [temp_action_space_styles]
	elif action_space_paths == "structures":
		actual_paths = [temp_action_space_structures]
	elif action_space_paths == "styles_structures":
		actual_paths = [temp_action_space_styles, temp_action_space_structures]
	elif action_space_paths == "subtopics_styles_structures":
		actual_paths = [
			temp_action_space_subtopics,
			temp_action_space_styles,
			temp_action_space_structures,
		]
	elif action_space_paths == "missing_action_space":
		# Used for testing the exception raised when an action space path is missing.
		missing_path = os.path.join(os.path.dirname(temp_action_space_styles), "missing.json")
		actual_paths = [missing_path]
	else:
		actual_paths = None

	if expected_exception is not None:
		with pytest.raises(expected_exception):
			TreeOfThoughtsController(
				signature=signature,
				max_reasoning_steps=max_reasoning_steps,
				tools=tools,
				action_space_paths=actual_paths,
				early_stopping_enabled=early_stopping_enabled,
				use_native_tool_calls=use_native_tool_calls,
			)
		return

	# We expect initialization to succeed
	controller = TreeOfThoughtsController(
		signature=signature,
		max_reasoning_steps=max_reasoning_steps,
		tools=tools,
		action_space_paths=actual_paths,
		early_stopping_enabled=early_stopping_enabled,
		use_native_tool_calls=use_native_tool_calls,
	)
	assert controller.max_reasoning_steps == max_reasoning_steps

	# Verify decide_next_step_single module signature (instructions + I/O fields)
	decide_signature = controller.decide_next_step_single.signature
	assert decide_signature.instructions == expected_instructions

	actual_input_fields = {
		name: {
			"annotation": get_annotation_name(field.annotation),
			"desc": (
				field.json_schema_extra.get("desc")
				if isinstance(field.json_schema_extra, dict)
				else None
			),
		}
		for name, field in decide_signature.input_fields.items()
	}
	actual_output_fields = {
		name: {
			"annotation": get_annotation_name(field.annotation),
			"desc": (
				field.json_schema_extra.get("desc")
				if isinstance(field.json_schema_extra, dict)
				else None
			),
		}
		for name, field in decide_signature.output_fields.items()
	}
	assert actual_input_fields == expected_input_fields
	assert actual_output_fields == expected_output_fields

	# Verify exact tool names match expected
	assert expected_tools is not None
	assert set(controller.tools.keys()) == set(expected_tools.keys())

	# Verify each tool matches expected specifications
	for tool_name, expected_tool in expected_tools.items():
		actual_tool = controller.tools[tool_name]
		assert isinstance(actual_tool, dspy.Tool)
		assert actual_tool.name == expected_tool.name, (
			f"\n\nTool name mismatch.\nExpected: {expected_tool.name}\n"
			f"Actual: {actual_tool.name}"
		)
		assert actual_tool.desc == expected_tool.desc, (
			f"\n\nTool description mismatch.\nExpected: {expected_tool.desc}\n"
			f"Actual: {actual_tool.desc}"
		)
		assert actual_tool.args == expected_tool.args, (
			f"\n\nTool arguments mismatch.\nExpected: {expected_tool.args}\n"
			f"Actual: {actual_tool.args}"
		)
		assert actual_tool.arg_types == expected_tool.arg_types, (
			f"\n\nTool argument types mismatch.\nExpected: {expected_tool.arg_types}\n"
			f"Actual: {actual_tool.arg_types}"
		)


def test_finish_tool_default_description() -> None:
	"""Verify that finish tool uses default description when not customized."""
	controller = TreeOfThoughtsController(
		signature=QuestionAnsweringWithReasoning,
		max_reasoning_steps=5,
		tools=None,
		action_space_paths=None,
		early_stopping_enabled=True,
		finish_tool_description=None,
	)

	finish_tool = controller.tools["finish"]
	expected_desc = (
		"Signals that the reasoning so far is sufficient for producing a "
		"high-quality response for the task.\n"
		"If selected, the next step will involve generating the final output "
		"rather than reasoning further."
	)
	assert finish_tool.desc == expected_desc


def test_finish_tool_custom_description() -> None:
	"""Verify that finish tool uses custom description when provided."""
	custom_desc = (
		"Only finish if you have verified all constraints. "
		"Requires at least 3 reasoning steps."
	)
	controller = TreeOfThoughtsController(
		signature=QuestionAnsweringWithReasoning,
		max_reasoning_steps=5,
		tools=None,
		action_space_paths=None,
		early_stopping_enabled=True,
		finish_tool_description=custom_desc,
	)

	finish_tool = controller.tools["finish"]
	assert finish_tool.desc == custom_desc
	# Verify it still returns the correct intervention
	intervention = finish_tool.func()
	assert isinstance(intervention, ReasoningIntervention)
	assert intervention.continue_reasoning is False


# =============================================================================
# Helper Functions for Creating Mock Responses and Expected Predictions
# =============================================================================

def create_dummy_controller_output(
	action: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	action_arguments: dict[str, Any] | None = None,
	tool_descriptions: str = "Dummy tool description",
	considerations: str = "Dummy considerations",
	continue_reasoning: bool = True,
	internal_reasoning: str = "Dummy internal reasoning",
	prefix: str = "Dummy prefix",
) -> ControllerOutput:
	"""Create a dummy ControllerOutput for testing."""
	return ControllerOutput(
		action=action,
		action_arguments=action_arguments or {},
		tool_descriptions=tool_descriptions,
		considerations=considerations,
		continue_reasoning=continue_reasoning,
		internal_reasoning=internal_reasoning,
		prefix=prefix,
	)


def create_mock_response(
	action: str,
	considerations: str = "Mock considerations",
	arguments: dict[str, Any] | None = None,
) -> str:
	"""Create a mock LLM response string in the expected format.

	The format follows the controller signature order: considerations, action, arguments.

	Parameters:
		action: The action name to include in the response
		considerations: The considerations text
		arguments: Optional arguments dict (will be JSON serialized)

	Returns:
		Formatted mock response string
	"""
	if arguments is not None:
		args_str = json.dumps(arguments)
		return (
			f"## considerations\n"
			f"{considerations}\n"
			f"## action\n"
			f"{action}\n"
			f"## action_arguments\n"
			f"{args_str}"
		)
	return (
		f"## considerations\n"
		f"{considerations}\n"
		f"## action\n"
		f"{action}"
	)


def create_mock_native_tool_call_response(
	tool_name: str,
	arguments: dict[str, Any] | None = None,
) -> str:
	"""Create a mock native tool-calling response in OpenAI tool-call JSON format."""
	function_obj: dict[str, Any] = {"name": tool_name}
	if arguments is not None:
		function_obj["arguments"] = json.dumps(arguments)
	return json.dumps({"tool_calls": [{"function": function_obj}]})


def create_expected_prediction(
	tool_name: str,
	chosen_values: dict[str, Any],
	intervention_kwargs: dict[str, Any],
	tool_description: str = "Mock tool description",
	considerations: str = "Mock considerations",
	error: str = "",
	num_occurrences: int = 1,
) -> ControllerPrediction:
	"""Create an expected ControllerPrediction object for testing.

	Since the actual tool instance isn't available during parameterization,
	we create a dummy tool with the correct name.

	Parameters:
		tool_name: Name of the expected tool
		chosen_values: Expected arguments
		intervention_kwargs: Dict to construct ReasoningIntervention (continue_reasoning, internal_reasoning, prefix)
		tool_description: Description of the tool
		considerations: Expected considerations text
		num_occurrences: Expected occurrence count

	Returns:
		ControllerPrediction instance with expected values
	"""
	if tool_name == DEFAULT_REASONING_INTERVENTION_TOOL_NAME:
		# Infer tool structure from chosen_values keys
		tool = dspy.Tool(
			name=tool_name,
			desc=tool_description,
			func=lambda **kwargs: ReasoningIntervention(**intervention_kwargs),
		)
	elif tool_name == "finish":
		tool = FINISH_TOOL
	elif tool_name == "continue_reasoning":
		tool = DEFAULT_TOOL
	else:
		tool = dspy.Tool(name=tool_name, func=lambda: None, desc=tool_description)

	if intervention_kwargs:
		intervention = ReasoningIntervention(**intervention_kwargs)
	else:
		intervention = tool.func(**chosen_values)

	return ControllerPrediction(
		tool=tool,
		chosen_values=chosen_values,
		intervention=intervention,
		considerations=considerations,
		error=error,
		num_occurrences=num_occurrences,
	)

# =============================================================================
# Test: Controller Forward Method
# =============================================================================

@pytest.mark.parametrize(
	[
		"state",
		"signature",
		"action_space_keys",
		"early_stopping_enabled",
		"use_native_tool_calls",
		"n_samples_generation",
		"temperature",
		"forced_choice_function",
		"mock_responses_func",
		"expected_predictions",
		"expected_exception",
	],
	[
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			lambda: [[[create_mock_response("continue_reasoning")]]], # mock_responses_func
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True, "internal_reasoning": "", "prefix": ""
					},
				)
			]],
			None,									# expected_exception
			id="default_tools_single_sample_no_reasoning",
		),
		pytest.param(
			State(									# state
				input={"question": "Q"},
				reasoning=[],
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			# mock_responses_func: parsing error affects only ONE completion (unknown tool).
			lambda: [[[
				create_mock_response("continue_reasoning", considerations="ok"),
				create_mock_response("unknown_tool", considerations="bad"),
				create_mock_response("finish", considerations="ok2"),
			]]],
			[[
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="ok",
					error="",
					num_occurrences=1,
				),
				create_expected_prediction(
					tool_name="finish",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="ok2",
					error="",
					num_occurrences=1,
				),
				create_expected_prediction(
					tool_name="prune",
					chosen_values={
						"error_type": "parsing",
						"error_message": "Unknown action: 'unknown_tool'",
						"raw_output": None,
					},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Controller failure: unknown action.",
					error="Unknown action: 'unknown_tool'",
					num_occurrences=1,
				),
			]],
			None,
			id="forward_parse_error_affects_single_completion",
		),
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[],
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			True,									# use_native_tool_calls
			2,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_native_tool_call_response("continue_reasoning"),
					create_mock_native_tool_call_response("finish"),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="N/A (native tool call)",
					num_occurrences=1,
				),
				create_expected_prediction(
					tool_name="finish",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="N/A (native tool call)",
					num_occurrences=1,
				),
			]],
			None,									# expected_exception
			id="default_tools_native_tool_calls_two_samples_mixed",
		),
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Good start, but lacking in content."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			lambda: [[[
				create_mock_response("continue_reasoning")
			]]],
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": ""
					},
				)
			]],
			None,									# expected_exception
			id="default_tools_single_sample_some_reasoning",
		),
		# Test 3: Default tools, multiple samples - all duplicates (controller deduplicates)
		# When all samples produce the same action+args, controller returns 1 unique prediction
		# with num_occurrences tracking the count
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Not enough content."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						action="continue_reasoning",
						considerations="First",
					),
					create_mock_response(
						action="continue_reasoning",
						considerations="Second",
					),
					create_mock_response(
						action="continue_reasoning",
						considerations="Third",
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="First",  # Controller behavior: takes the first encountered considerations for the unique action set
					num_occurrences=3,
				),
			]],
			None,									# expected_exception
			id="default_tools_multiple_samples_all_duplicates",
		),
		# Default tools, multiple samples - mixed (continue and finish)
		# 2 continue + 1 finish = 2 unique predictions
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Not enough content."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						action="continue_reasoning",
						considerations="Continue 1",
					),
					create_mock_response(
						action="finish",
						considerations="Finish 1",
					),
					create_mock_response(
						action="continue_reasoning",
						considerations="Continue 2",
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Continue 1",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name="finish",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Finish 1",
					num_occurrences=1,
				),
			]],
			None,									# expected_exception
			id="default_tools_multiple_samples_mixed_actions",
		),
		# Test 5: Action space (styles), single sample
		pytest.param(
			State(									# state
				input={
					"topic": "Climate change action",
					"stance": "PRO",
				},
				reasoning=[{"claim": "Our planet is like a feverish patient needing urgent care."}],
				controller_output_trajectory=[
					create_dummy_controller_output(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						action_arguments={"style": "Figurative Language"},
						tool_descriptions="Use metaphor, simile, analogy, or symbolism to make ideas concrete.",
						considerations="I should employ non-literal comparison to make abstract concepts vivid. ",
					)
				],
				controller_outputs=[],
				feedback=["Vivid metaphor."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles"],								# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			# mock_responses_func (a different style than the previous step):
			(
				lambda: [[[
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="I want to use statistics and data.",
						arguments={"style": "Statistical & Data-Driven"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "",
					},
					considerations="I want to use statistics and data.",
				)
			]],
			None,									# expected_exception
			id="styles_action_space_single_sample",
		),
		pytest.param(
			State(									# state
				input={
					"topic": "Climate change action",
					"stance": "PRO",
				},
				reasoning=[{"claim": "Our planet is like a feverish patient."}],
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["subtopics", "styles", "structures"],	# action_space_keys
			True,									# early_stopping_enabled
			True,									# use_native_tool_calls
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(
				lambda: [[[
					create_mock_native_tool_call_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Economic Impact",
							"style": "Statistical & Data-Driven",
							"structure": "Contrast",
						},
					),
					create_mock_native_tool_call_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Economic Impact",
							"style": "Statistical & Data-Driven",
							"structure": "Contrast",
						},
					),
					create_mock_native_tool_call_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Social Justice & Equity",
							"style": "Figurative Language",
							"structure": "Causal Reasoning",
						},
					),
				]]]
			),
			[[
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={
						"stock_issue": "Economic Impact",
						"style": "Statistical & Data-Driven",
						"structure": "Contrast",
					},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": (
							"I should focus on the economic and financial implications of this issue.  "
							"I should use numbers and data to provide concrete, measurable support. "
						),
						"prefix": "However",
					},
					considerations="N/A (native tool call)",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={
						"stock_issue": "Social Justice & Equity",
						"style": "Figurative Language",
						"structure": "Causal Reasoning",
					},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": (
							"I should consider the fairness and equity dimensions of this issue.  "
							"I should employ non-literal comparison to make abstract concepts vivid.  "
							"I should use causal reasoning."
						),
						"prefix": "Therefore",
					},
					considerations="N/A (native tool call)",
					num_occurrences=1,
				),
			]],
			None,									# expected_exception
			id="native_tool_calls_complex_action_space_dedup_and_prefixes",
		),
		# Test 6: Action space (structures), single sample
		pytest.param(
			State(									# state
				input={
					"topic": "Govermnets should do more to prevent climate change",
					"stance": "PRO",
				},
				reasoning=[{"claim": "According to recent studies, climate change is accelerating."}],
				controller_output_trajectory=[
					create_dummy_controller_output(action_arguments={"structure": "Evidence & Support"})
				],
				controller_outputs=[],
				feedback=["Good usage of evidence to support the claim."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["structures"],							# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Using causal reasoning",
						{"structure": "Causal Reasoning"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"structure": "Causal Reasoning"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use causal reasoning.",
						"prefix": "Therefore",
					},
					considerations="Using causal reasoning",
				)
			]],
			None,									# expected_exception
			id="structures_action_space_single_sample",
		),
		# Test 7: Action space (styles + structures), single sample
		pytest.param(
			State(									# state
				input={"topic": "We should invest in fighting climate change.", "stance": "PRO"},
				reasoning=[
					{"claim": "According to 97% of climate scientists, the warming trend is undeniable."}
				],
				controller_output_trajectory=[
					create_dummy_controller_output(
						action_arguments={"style": "Statistical & Data-Driven", "structure": "Evidence & Support"}
					)
				],
				controller_outputs=[],
				feedback=["Effective use of statistics and evidence."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles", "structures"],				# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"I want to provide statistical and data-driven evidence",
						{"style": "Statistical & Data-Driven", "structure": "Contrast"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven", "structure": "Contrast"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "However",
					},
					considerations="I want to provide statistical and data-driven evidence",
				)
			]],
			None,									# expected_exception
			id="styles_structures_action_space_single_sample",
		),
		# Test 8: Action space, multiple samples with different choices (no duplicates)
		pytest.param(
			State(									# state
				input={"topic": "We should invest in fighting climate change.","stance": "PRO"},
				reasoning=[{"claim": "Our planet is like a feverish patient."}],
				controller_output_trajectory=[
					create_dummy_controller_output(action_arguments={"style": "Figurative Language"})
				],
				controller_outputs=[],
				feedback=["Vivid metaphor."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles"],								# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			2,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Figurative choice",
						{"style": "Figurative Language"},
					),
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Statistical choice",
						{"style": "Statistical & Data-Driven"},
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid. ",
						"prefix": "",
					},
					considerations="Figurative choice",
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "",
					},
					considerations="Statistical choice",
				),
			]],
			None,									# expected_exception
			id="styles_action_space_multiple_different_choices",
		),
		# Test 9: Action space, multiple samples with duplicate choices
		# 2 figurative + 1 statistical = 2 unique predictions
		pytest.param(
			State(									# state
				input={"topic": "We should invest in fighting climate change.","stance": "PRO"},
				reasoning=[{"claim": "Our planet is like a feverish patient."}],
				controller_output_trajectory=[
					create_dummy_controller_output(action_arguments={"style": "Figurative Language"})
				],
				controller_outputs=[],
				feedback=["Good metaphor."],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles"],								# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			3,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"First figurative",
						{"style": "Figurative Language"},
					),
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Statistical",
						{"style": "Statistical & Data-Driven"},
					),
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Second figurative",
						{"style": "Figurative Language"},
					),
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid. ",
						"prefix": "",
					},
					considerations="First figurative",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "",
					},
					considerations="Statistical",
					num_occurrences=1,
				),
			]],
			None,									# expected_exception
			id="styles_action_space_with_duplicate_choices",
		),
		# Test 10: Forced choice function - single forced action
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			(										# forced_choice_function
				lambda tools, state: [("finish", {}, "Forced finish")]
			),
			None,									# mock_responses_func (no mock needed)
			[[										# expected_predictions
				create_expected_prediction(
					"finish",
					{},
					{"continue_reasoning": False, "internal_reasoning": "", "prefix": ""},
					considerations="Forced finish",
				)
			]],
			None,									# expected_exception
			id="forced_choice_single_action",
		),
		# Test 11: Forced choice function - multiple forced actions
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			2,										# n_samples_generation
			0.0,									# temperature
			(										# forced_choice_function
				lambda tools, state: [
					("continue_reasoning", {}, "Forced continue"),
					("finish", {}, "Forced finish"),
				]
			),
			None,									# mock_responses_func (no mock needed)
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced continue",
				),
				create_expected_prediction(
					tool_name="finish",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced finish",
				),
			]],
			None,									# expected_exception
			id="forced_choice_multiple_actions",
		),
		# Test 12: No early stopping
		# When early_stopping_enabled=False, only DEFAULT_TOOL exists
		# Since tools have no arguments, ARGUMENTS field is not included
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=["Good step."],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			False,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[create_mock_response(
					"continue_reasoning",
					"Only option",
				)]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="continue_reasoning",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Forced choice: 'continue_reasoning' was the only available action. No other tools were provided to the controller.",
				)
			]],
			None,									# expected_exception
			id="no_early_stopping",
		),
		# Test 13: Ready for final response (many reasoning steps)
		pytest.param(
			State(									# state
				input={"math_problem": "2+2"},
				reasoning=[
						{"math_operation": "This is a simple addition problem."},
						{"math_operation": "Adding 2 and 2 together."},
						{"math_operation": "The answer is 4."},
					],
				controller_output_trajectory=[create_dummy_controller_output()] * 3,
				controller_outputs=[],
				feedback=["I agree.", "Reasonable next step.", "Correct."],
				output={},
			),
			SolveMathProblemWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[create_mock_response("finish", "Solution complete")]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name="finish",
					chosen_values={},
					intervention_kwargs={
						"continue_reasoning": False,
						"internal_reasoning": "",
						"prefix": "",
					},
					considerations="Solution complete",
				)
			]],
			None,									# expected_exception
			id="ready_for_final_response",
		),
		# Test 14: Combined styles+structures, multiple samples with mixed duplicates
		# Choices 1 and 3 are duplicates, so 3 unique predictions
		pytest.param(
			State(									# state
				input={"topic": "We should invest in fighting climate change.","stance": "PRO"},
				reasoning=[{"claim": "We need to consider the evidence."}],
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["styles", "structures"],				# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			4,										# n_samples_generation
			0.8,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 1",
						arguments={"style": "Figurative Language", "structure": "Causal Reasoning"},
					),
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 2",
						arguments={"style": "Statistical & Data-Driven", "structure": "Evidence & Support"},
					),
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 3 (dup of 1)",
						arguments={"style": "Figurative Language", "structure": "Causal Reasoning"},
					),
					create_mock_response(
						action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						considerations="Choice 4",
						arguments={"style": "Figurative Language", "structure": "Contrast"},
					),
				]]]
			),
			# expected_predictions
			[[
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language", "structure": "Causal Reasoning"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid.  I should use causal reasoning.",
						"prefix": "Therefore",
					},
					considerations="Choice 1",
					num_occurrences=2,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Statistical & Data-Driven", "structure": "Evidence & Support"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use numbers and data to provide concrete, measurable support. ",
						"prefix": "According to",
					},
					considerations="Choice 2",
					num_occurrences=1,
				),
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"style": "Figurative Language", "structure": "Contrast"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should employ non-literal comparison to make abstract concepts vivid. ",
						"prefix": "However",
					},
					considerations="Choice 4",
					num_occurrences=1,
				),
			]],
			None,									# expected_exception
			id="styles_structures_multiple_with_duplicates",
		),
		# Test 15: Single state with reasoning, aligned controller output (prefix)
		pytest.param(
			State(									# state
				input={"topic": "We should invest in fighting climate change.", "stance": "PRO"},
				reasoning=[{"claim": "We need to consider the evidence."}],
				controller_output_trajectory=[create_dummy_controller_output()],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			GenerateArgumentWithReasoning,			# signature
			["structures"],							# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [[[
					create_mock_response(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						"Using causal reasoning",
						{"structure": "Causal Reasoning"},
					)
				]]]
			),
			[[										# expected_predictions
				create_expected_prediction(
					tool_name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					chosen_values={"structure": "Causal Reasoning"},
					intervention_kwargs={
						"continue_reasoning": True,
						"internal_reasoning": "I should use causal reasoning.",
						"prefix": "Therefore", # Verify prefix alignment
					},
					considerations="Using causal reasoning",
				)
			]],
			None,									# expected_exception
			id="single_state_aligned_output",
		),
		# Test 16: Layer of 2 states (same parent), each generating 3 thoughts/predictions
		# State 1: 3 continue
		# State 2: 2 continue, 1 finish
		pytest.param(
			[										# states (list of 2 states)
				State(
					input={"question": "What is 2+2?"},
					reasoning=[
						{"reasoning_step": "We should use an arithmetic operation to solve the problem."}
					],
					controller_output_trajectory=[create_dummy_controller_output()],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={"question": "What is 2+2?"},
					reasoning=[
						{"reasoning_step": "We should use an arithmetic operation to solve the problem."}
					],
					controller_output_trajectory=[create_dummy_controller_output()],
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			2,										# n_samples_generation
			0.7,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					# Layer 0 (Batched)
					[
						# State 1 predictions
						[
							create_mock_response(
								action="continue_reasoning", considerations="S1 C1", arguments=None
							),
							create_mock_response(
								action="continue_reasoning", considerations="S1 C2", arguments=None
							),
						],
						# State 2 predictions
						[
							create_mock_response(
								action="continue_reasoning", considerations="S2 C1", arguments=None,
							),
							create_mock_response(
								action="finish", considerations="S2 F1", arguments=None
							),
						]
					]
				]
			),
			# expected_predictions
			[
				# State 1 Expectations
				[
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True, "internal_reasoning": "", "prefix": "",
						},
						considerations="S1 C1",
						num_occurrences=2,
					),
				],
				# State 2 Expectations
				[
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S2 C1",
					),
					create_expected_prediction(
						tool_name="finish",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S2 F1",
					),
				],
			],
			None,									# expected_exception
			id="layer_method_two_states_same_parent",
		),
		# Test 17: Layer of 2 states with different reasoning trajectories
		# One early state, one later state.
		# Note: In real usage, states in a layer usually have same depth, but controller handles independent states
		# so depth difference is fine for testing robustness.
		pytest.param(
			[										# states
				State(		# Early state
					input={"question": "What is 2+2?"},
					reasoning=[
						{"reasoning_step": "Start reasoning"},
						{"reasoning_step": "Step 2"},
						{"reasoning_step": "Step 3"},
					],
					controller_output_trajectory=[create_dummy_controller_output()] * 3,
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(		# Late state
					input={"question": "What is 2+2?"},
					reasoning=[
						{"reasoning_step": "Start"},
						{"reasoning_step": "Middle"},
						{"reasoning_step": "Almost done"},
					],
					controller_output_trajectory=[create_dummy_controller_output()] * 3,
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_key
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						[
							create_mock_response(
								action="continue_reasoning",
								considerations="Keep going",
								arguments=None,
							)
						],
						[
							create_mock_response(
								action="finish",
								considerations="Values done",
								arguments=None,
							)
						]
					]
				]
			),
			[										# expected_predictions
				[
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="Keep going",
					)
				],
				[
					create_expected_prediction(
						tool_name="finish",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="Values done",
					)
				],
			],
			None,									# expected_exception
			id="layer_method_different_trajectories",
		),
		# Test 18: Batch processing - Two states, single sample each
		pytest.param(
			[										# states (2 simple states)
				State(
					input={"question": "Q1"},
					reasoning=[{"reasoning_step": "R1"}],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={"question": "Q2"},
					reasoning=[{"reasoning_step": "R2"}],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				)
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						[create_mock_response("continue_reasoning", "S0")],
						[create_mock_response("finish", "S1")],
					]
				]
			),
			[										# expected_predictions
				[
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S0",
					)
				],
				[
					create_expected_prediction(
						tool_name="finish",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S1",
					)
				],
			],
			None,									# expected_exception
			id="batch_two_states_single_sample",
		),
		# Test 19: Batch processing - Three states, multiple samples from each
		pytest.param(
			[										# states (3 simple states)
				State(
					input={"question": "Q0"},
					reasoning=[],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={"question": "Q1"},
					reasoning=[],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={"question": "Q2"},
					reasoning=[],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			2,										# n_samples_generation
			0.1,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						# State 0: 2 continue
						[
							create_mock_response("continue_reasoning", "S0-1"),
							create_mock_response("continue_reasoning", "S0-2"),
						],
						# State 1: 1 finish, 1 continue
						[
							create_mock_response("finish", "S1-1"),
							create_mock_response("continue_reasoning", "S1-2"),
						],
						# State 2: 1 continue, 1 finish
						[
							create_mock_response("continue_reasoning", "S2-1"),
							create_mock_response("finish", "S2-2"),
						],
					]
				]
			),
			[										# expected_predictions
				[	# State 0: Deduped to 1
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S0-1",
						num_occurrences=2,
					)
				],
				[	# State 1: 2 unique
					create_expected_prediction(
						tool_name="finish",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S1-1",
					),
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S1-2",
					)
				],
				[	# State 2: 2 unique
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S2-1",
					),
					create_expected_prediction(
						tool_name="finish",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="S2-2",
					)
				],
			],
			None,						# expected_exception
			id="batch_three_states_multiple_samples",
		),
		# Test 20: Batch processing - Two states, 3 samples each ('Structure' action space)
		pytest.param(
			[							# states (2 states with 2 reasoning layers)
				State(					# State 1: first trajectory
					input={
						"topic": "We should regulate AI development.",
						"stance": "PRO",
					},
					reasoning=[
						{"claim": "First, AI capabilities are advancing faster than safety research."},
						{"claim": "Therefore, we cannot predict or control emergent behaviors in advanced systems."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				State(
					input={
						"topic": "We should regulate AI development.",
						"stance": "ANTI",
					},
					reasoning=[
						{"claim": "First, heavy compliance burdens will hurt startups more than incumbents."},
						{"claim": "According to a 2023 study by Stanford HAI, compliance costs could reduce open-source innovation by 40%."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
			],
			GenerateArgumentWithReasoning,			# signature
			["structures"],							# action_space_keys (triggers STRUCTURE_CONTROLLER_DEMOS)
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			3,										# n_samples_generation
			0.1,									# temperature
			return_action_if_single_option,			# forced_choice_function
			(										# mock_responses_func
				lambda: [
					[
						# State 0 (PRO): 2 "Causal Reasoning", 1 "Finish"
						# Controller suggests "Causal Reasoning" -> prefix "Therefore"
						[
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic: Need to explain the consequence of uncontrollability.",
								{"structure": "Causal Reasoning"},
							),
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic (duplicate): Need to conclude why this justifies regulation.",
								{"structure": "Causal Reasoning"},
							),
							create_mock_response(
								"finish",
								"Argument is complete and conclusion follows logically.",
								arguments={},
							),
						],
						# State 1 (CON): 1 "Causal Reasoning", 2 "Contrast"
						# Controller suggests "Contrast" -> prefix "However"
						[
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic: Need to show the effect of reduced innovation on national security.",
								{"structure": "Evidence & Support"},
							),
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic: Need to offset compliance benefits with innovation risks.",
								{"structure": "Contrast"},
							),
							create_mock_response(
								"intervene_on_next_reasoning_step",
								"Reasoning logic (duplicate): Need to present the counterpoint on cost.",
								{"structure": "Contrast"},
							),
						],
					]
				]
			),
			[										# expected_predictions
				[	# State 0 Results
					create_expected_prediction(
						tool_name="intervene_on_next_reasoning_step",
						chosen_values={"structure": "Causal Reasoning"},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "I should use causal reasoning.",
							"prefix": "Therefore", # Prefilled by create_expected_tool_for_structures?
						},
						considerations="Reasoning logic: Need to explain the consequence of uncontrollability.",
						num_occurrences=2,
					),
					create_expected_prediction(
						tool_name="finish",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": False,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations="Argument is complete and conclusion follows logically.",
						num_occurrences=1,
					),
				],
				[	# State 1 Results
					create_expected_prediction(
						tool_name="intervene_on_next_reasoning_step",
						chosen_values={"structure": "Evidence & Support"},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "According to",
						},
						considerations="Reasoning logic: Need to show the effect of reduced innovation on national security.",
						num_occurrences=1,
					),
					create_expected_prediction(
						tool_name="intervene_on_next_reasoning_step",
						chosen_values={"structure": "Contrast"},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "However", # "Contrast" implies "However" prefix
						},
						considerations="Reasoning logic: Need to offset compliance benefits with innovation risks.",
						num_occurrences=2,
					),
				],
			],
			None,									# expected_exception
			id="batch_two_states_shared_duplicate_actions",
		),
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			False,									# early_stopping_enabled (no finish tool)
			False,									# use_native_tool_calls
			10,										# n_samples_generation
			0.0,									# temperature
			return_action_if_single_option,			# forced_choice_function
			None,									# no mock responses (forced choice triggers)
			[										# expected_predictions
				[
					create_expected_prediction(
						tool_name="continue_reasoning",
						chosen_values={},
						intervention_kwargs={
							"continue_reasoning": True,
							"internal_reasoning": "",
							"prefix": "",
						},
						considerations=(
							"Forced choice: 'continue_reasoning' was the only available action. "
							"No other tools were provided to the controller."
						),
						num_occurrences=10,
					)
				]
			],
			None,									# expected_exception
			id="forced_choice_repeats_to_match_n_samples",
		),
		pytest.param(
			State(									# state
				input={"question": "What is 2+2?"},
				reasoning=[{"reasoning_step": "First, I need to think about addition."}],
				controller_output_trajectory=[],
				controller_outputs=[],
				feedback=[],
				output={},
			),
			QuestionAnsweringWithReasoning,			# signature
			None,									# action_space_keys
			True,									# early_stopping_enabled
			False,									# use_native_tool_calls
			1,										# n_samples_generation
			0.0,									# temperature
			(										# mock_responses_func
				lambda tools, state: [("nonexistent_tool", {}, "This tool doesn't exist")]
			),
			None,									# mock_responses_func
			[[]],									# expected_predictions (unused; exception expected)
			AssertionError,							# expected_exception
			id="forced_choice_nonexistent_tool_raises",
		),
	],
)
def test_controller_forward(
	state: State | list[State],
	signature: type[ReasoningSignature],
	action_space_keys: list[str] | None,
	early_stopping_enabled: bool,
	use_native_tool_calls: bool,
	n_samples_generation: int,
	temperature: float,
	forced_choice_function,
	mock_responses_func,
	expected_predictions: list[list[ControllerPrediction]],
	expected_exception: type[Exception] | None,
	temp_action_space_styles: str,
	temp_action_space_structures: str,
	temp_action_space_subtopics: str,
) -> None:
	"""
	Test controller forward method with various configurations.

	This test uses MockPredict and MockGenerativeLocalVLLM from utilities_for_tests.py
	to simulate LLM responses. It verifies that the controller properly handles:
	- Different action space configurations
	- Different state types (no reasoning, some reasoning, ready for final)
	- Different sampling parameters
	- Forced choice functions
	- Early stopping on/off
	- Duplicate and unique action responses

	Parameters:
		state: The state to test with
		signature: The reasoning signature class to test
		action_space_keys: List of keys for action space fixtures or None.
			Valid keys: "styles", "structures", "subtopics"
		early_stopping_enabled: Whether to enable early stopping
		n_samples_generation: Number of samples to generate
		temperature: Temperature for generation
		forced_choice_function: Optional forced choice function
		mock_responses_func: Function that returns mock responses for MockPredict
		expected_predictions: Expected list of list of ControllerPrediction objects.
	"""
	# Convert fixture keys to actual paths
	if action_space_keys is None:
		action_space_paths = None
	else:
		action_space_paths = []
		for key in action_space_keys:
			if key == "styles":
				action_space_paths.append(temp_action_space_styles)
			elif key == "structures":
				action_space_paths.append(temp_action_space_structures)
			elif key == "subtopics":
				action_space_paths.append(temp_action_space_subtopics)

	# Create controller
	controller = TreeOfThoughtsController(
		signature=signature,
		max_reasoning_steps=10,
		action_space_paths=action_space_paths,
		early_stopping_enabled=early_stopping_enabled,
		forced_choice_function=forced_choice_function,
		use_native_tool_calls=use_native_tool_calls,
	)

	# State is provided as a parameter

	# Set up mock predictor if we have mock responses to use
	if mock_responses_func is not None:
		payload = mock_responses_func()
		if isinstance(payload, dict):
			mock_predictor = MockPredict(
				responses=payload["responses"],
				signature=controller.create_controller_signature(),
				chat_exception=payload.get("chat_exception"),
			)
		else:
			mock_predictor = MockPredict(
				responses=payload,
				signature=controller.create_controller_signature(),
			)
		controller.decide_next_step_single = mock_predictor

	# Call forward (or assert expected exception)
	if expected_exception is not None:
		with pytest.raises(expected_exception):
			controller(
				states=state,
				n_samples_generation=n_samples_generation,
				temperature=temperature,
			)
		return

	result = controller(
		states=state,
		n_samples_generation=n_samples_generation,
		temperature=temperature,
	)

	# Verify result structure
	assert isinstance(result, list), "Result should be a list"
	assert len(result) == len(expected_predictions), \
		"Result length should match expected_predictions length"

	for state_idx, (actual_list, expected_list) in enumerate(
		zip(result, expected_predictions, strict=True)
	):
		assert isinstance(actual_list, list), (
			f"State {state_idx}: Each state entry should be a list of predictions"
		)

		# Sort both lists to ensure consistent comparison order
		# Key: tool name, sorted args, considerations
		# TODO[P3]: Remove helpers like this. Consider moving this to the top of the file, or using
		# in-line logic.
		def sort_key(p):
			return (p.tool.name, tuple(sorted(p.chosen_values.items())), p.considerations)

		actual_list_sorted = sorted(actual_list, key=sort_key)
		expected_list_sorted = sorted(expected_list, key=sort_key)

		assert len(actual_list_sorted) == len(expected_list_sorted), (
			f"State {state_idx}: Expected {len(expected_list_sorted)} unique predictions, "
			f"got {len(actual_list_sorted)}"
		)

		for i, (actual, expected) in enumerate(
			zip(actual_list_sorted, expected_list_sorted, strict=True)
		):
			assert isinstance(actual, ControllerPrediction), (
				f"State {state_idx}, Prediction {i} should be ControllerPrediction, "
				f"got {type(actual)}"
			)

			# Verify tool name
			assert actual.tool.name == expected.tool.name, (
				f"State {state_idx}, Prediction {i}: Expected tool '{expected.tool.name}', "
				f"got '{actual.tool.name}'"
			)

			# Verify chosen values
			assert actual.chosen_values == expected.chosen_values, (
				f"State {state_idx}, Prediction {i}: Expected arguments {expected.chosen_values}, "
				f"got {actual.chosen_values}"
			)

			# Verify intervention fields
			assert actual.intervention.continue_reasoning == expected.intervention.continue_reasoning, (
				f"State {state_idx}, Prediction {i}: Expected continue_reasoning {expected.intervention.continue_reasoning}, "
				f"got {actual.intervention.continue_reasoning}"
			)

			assert actual.intervention.internal_reasoning == expected.intervention.internal_reasoning, (
				f"\n\nState {state_idx}, Prediction {i}: Internal reasoning mismatch.\n"
				f"Expected: {repr(expected.intervention.internal_reasoning)}\n"
				f"Actual:   {repr(actual.intervention.internal_reasoning)}"
			)

			assert actual.intervention.prefix == expected.intervention.prefix, (
				f"\n\nState {state_idx}, Prediction {i}: Prefix mismatch.\n"
				f"Expected: '{expected.intervention.prefix}'\n"
				f"Actual:   '{actual.intervention.prefix}'"
			)

			# Verify considerations - check if expected consideration matches actual
			# We might relax this if exact match isn't passed in parametrization,
			# but assuming we use create_expected_prediction which defaults to "Mock considerations"
			assert actual.considerations == expected.considerations, (
				f"\n\nState {state_idx}, Prediction {i}: Considerations mismatch"
			)

			# Verify error surfaced on the prediction
			assert actual.error == expected.error, (
				f"\n\nState {state_idx}, Prediction {i}: Error mismatch.\n"
				f"Expected: {expected.error!r}\n"
				f"Actual:   {actual.error!r}"
			)

			# Verify occurrences
			assert actual.num_occurrences == expected.num_occurrences, (
				f"\n\nState {state_idx}, Prediction {i}: Expected {expected.num_occurrences} occurrences, got {actual.num_occurrences}"
			)


def _validate_controller_predictions_equality(
	actual: list[ControllerPrediction],
	expected: list[ControllerPrediction],
) -> None:
	"""
	Helper function to validate that actual controller predictions match expected predictions.

	Args:
		actual: List of actual ControllerPrediction objects
		expected: List of expected ControllerPrediction objects

	Raises:
		AssertionError: If predictions don't match
	"""
	assert len(actual) == len(expected), (
		f"Expected {len(expected)} predictions, got {len(actual)}"
	)

	for i, (actual_pred, expected_pred) in enumerate(zip(actual, expected, strict=True)):
		assert actual_pred.tool.name == expected_pred.tool.name, (
			f"Prediction {i}: Expected tool '{expected_pred.tool.name}', got '{actual_pred.tool.name}'"
		)
		assert actual_pred.chosen_values == expected_pred.chosen_values, (
			f"Prediction {i}: Chosen values mismatch"
		)
		assert actual_pred.considerations == expected_pred.considerations, (
			f"Prediction {i}: Considerations mismatch"
		)
		assert actual_pred.num_occurrences == expected_pred.num_occurrences, (
			f"Prediction {i}: Expected {expected_pred.num_occurrences} occurrences, got {actual_pred.num_occurrences}"
		)
		assert actual_pred.error == expected_pred.error, (
			f"Prediction {i}: Error flag mismatch"
		)


@pytest.mark.parametrize(
	[
		# Controller __init__ parameters
		"signature",
		"max_reasoning_steps",
		"tools",
		"action_space_paths_spec",
		"early_stopping_enabled",
		"forced_choice_function",
		"use_native_tool_calls",
		"verbosity",

		# Method input parameters
		"mock_predictions_data",
		"expected_n",

		# Expected output parameters
		"expected_predictions",
		"expected_exception",
	],
	[
		# =============================================================================
		# Error Handling Tests (2 cases)
		# =============================================================================

		pytest.param(
			QuestionAnsweringWithReasoning,   # signature
			10,                               # max_reasoning_steps
			None,                             # tools
			None,                             # action_space_paths_spec
			True,                             # early_stopping_enabled
			return_action_if_single_option,   # forced_choice_function
			False,                            # use_native_tool_calls (legacy)
			None,                             # verbosity
			# mock_predictions_data: legacy format with parsing error in middle
			{
				"considerations": ["ok", "bad", "ok2"],
				"action": ["continue_reasoning", "unknown_tool", "finish"],
				"action_arguments": [{}, {}, {}],
				"error": [
					ExecutionError(),
					ExecutionError(
						error_type="parsing",
						error_message="Unknown action: 'unknown_tool'",
						raw_output=None,
					),
					ExecutionError(),
				],
				"error_type": [None, "parsing", None],
				"error_message": [None, "Unknown action: 'unknown_tool'", None],
				"raw_output": [None, None, None],
			},
			3,                                # expected_n
			# Expected: continue, PRUNE (for unknown_tool), finish (preserves original order)
			[
				ControllerPrediction(
					tool=DEFAULT_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="ok",
					num_occurrences=1,
				),
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Unknown action: 'unknown_tool'",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Unknown action: 'unknown_tool'",
				),
				ControllerPrediction(
					tool=FINISH_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=False),
					considerations="ok2",
					num_occurrences=1,
					error="",
				),
			],
			None,                             # expected_exception
			id="legacy_failed_completions_with_parsing_error",
		),

		pytest.param(
			QuestionAnsweringWithReasoning,  # signature
			10,                               # max_reasoning_steps
			None,                             # tools
			None,                             # action_space_paths_spec
			True,                             # early_stopping_enabled
			return_action_if_single_option,  # forced_choice_function
			True,                             # use_native_tool_calls (native)
			None,                             # verbosity
			# mock_predictions_data: native format with generation errors
			{
				"tool_calls": [
					SimpleNamespace(tool_calls=[SimpleNamespace(name="continue_reasoning", args={})]),
					SimpleNamespace(tool_calls=[SimpleNamespace(name="finish", args={})]),
				],
				"error": [
					ExecutionError(error_type="generation", error_message="Generation failed"),
					ExecutionError(error_type="generation", error_message="Generation failed"),
				],
				"error_type": ["generation", "generation"],
				"error_message": ["Generation failed", "Generation failed"],
				"raw_output": [None, None],
			},
			2,                                # expected_n
			# Expected: All generation errors become PRUNE
			[
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "generation",
						"error_message": "Generation failed",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'generation'",
					num_occurrences=2,
					error="Generation failed",
				),
			],
			None,                             # expected_exception
			id="native_failed_completions_with_generation_errors",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# tools
			None,								# action_space_paths_spec
			True,								# early_stopping_enabled
			return_action_if_single_option,		# forced_choice_function
			False,								# use_native_tool_calls
			None,								# verbosity
			{									# mock_predictions_data
				"considerations": ["Test"],
				"action": ["invalid_tool_name"],
				"action_arguments": [{}],
				"error": [
					ExecutionError(
						error_type="parsing",
						error_message="Unknown action: 'invalid_tool_name'",
						raw_output=None,
					)
				],
				"error_type": ["parsing"],
				"error_message": ["Unknown action: 'invalid_tool_name'"],
				"raw_output": [None],
			},
			1,									# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Unknown action: 'invalid_tool_name'",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Unknown action: 'invalid_tool_name'",
				)
			],
			None,								# expected_exception
			id="legacy_action_unknown_tool_maps_to_prune",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# tools
			None,								# action_space_paths_spec
			True,								# early_stopping_enabled
			return_action_if_single_option,		# forced_choice_function
			False,								# use_native_tool_calls
			None,								# verbosity
			{									# mock_predictions_data
				"considerations": ["Test"],
				"action": ["continue_reasoning"],
				"action_arguments": [{}],
				"error": [ExecutionError(error_type="parsing", error_message="Adapter parsing failed")],
				"error_type": ["parsing"],
				"error_message": ["Adapter parsing failed"],
				"raw_output": [None],
			},
			1,									# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Adapter parsing failed",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Adapter parsing failed",
				)
			],
			None,								# expected_exception
			id="legacy_adapter_parsing_error_forces_prune",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# tools
			None,								# action_space_paths_spec
			True,								# early_stopping_enabled
			return_action_if_single_option,		# forced_choice_function
			False,								# use_native_tool_calls
			None,								# verbosity
			{									# mock_predictions_data
				"considerations": ["Test"],
				"action": ["`continue_reasoning`"],
				"action_arguments": [{}],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			1,									# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=DEFAULT_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Test",
					num_occurrences=1,
					error="",
				)
			],
			None,								# expected_exception
			id="legacy_action_quotes_and_backticks_strip",
		),
		pytest.param(
			QuestionAnsweringWithReasoning,		# signature
			5,									# max_reasoning_steps
			None,								# tools
			None,								# action_space_paths_spec
			True,								# early_stopping_enabled
			return_action_if_single_option,		# forced_choice_function
			False,								# use_native_tool_calls
			None,								# verbosity
			{									# mock_predictions_data
				"considerations": ["Test"],
				"action": ["continue_reasoning"],
				"action_arguments": [{}],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			2,									# expected_n
			None,								# expected_exception
			AssertionError,						# expected_exception
			id="legacy_expected_n_mismatch_raises",
		),
		pytest.param(
			GenerateArgumentWithReasoning,    	# signature
			5,                                	# max_reasoning_steps
			[DEFAULT_TOOL],                   	# tools
			"all",								# action_space_paths_spec
			None,                             	# early_stopping_enabled
			None,                             	# forced_choice_function
			True,                             	# use_native_tool_calls
			None,                             	# verbosity
			{									# mock_predictions_data
				"tool_calls": [SimpleNamespace(tool_calls=[SimpleNamespace(
					name="continue_reasoning",
					args={}
				)])],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			1,                                	# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=DEFAULT_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
			],
			None,                             	# expected_exception
			id="native_continue_reasoning_tool",
		),
		pytest.param(
			GenerateArgumentWithReasoning,    	# signature
			5,                                	# max_reasoning_steps
			[DEFAULT_TOOL],                   	# tools
			"all",                            	# action_space_paths_spec
			True,                             	# early_stopping_enabled
			None,                             	# forced_choice_function
			True,                             	# use_native_tool_calls
			None,                             	# verbosity
			{									# mock_predictions_data
				"tool_calls": [SimpleNamespace(tool_calls=[SimpleNamespace(
					name="finish",
					args={}
				)])],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			1,                                	# expected_n
			[
				ControllerPrediction(
					tool=FINISH_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=False),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
			],
			None,                             	# expected_exception
			id="native_finish_tool",
		),
		pytest.param(
			GenerateArgumentWithReasoning,   	# signature
			5,                                	# max_reasoning_steps
			[DEFAULT_TOOL],                   	# tools
			"all",                            	# action_space_paths_spec
			None,                             	# early_stopping_enabled
			None,                             	# forced_choice_function
			True,                             	# use_native_tool_calls
			None,                             	# verbosity
			{									# mock_predictions_data
				"tool_calls": [SimpleNamespace(tool_calls=[SimpleNamespace(
					name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
					args={"dimension": "style", "choice": "Figurative Language"}
				)])],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			1,                                	# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=SimpleNamespace(name=DEFAULT_REASONING_INTERVENTION_TOOL_NAME),
					chosen_values={"dimension": "style", "choice": "Figurative Language"},
					intervention=ReasoningIntervention(
						continue_reasoning=True,
						internal_reasoning="Figurative Language",
						prefix="style",
					),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="Tool 'intervene_on_next_reasoning_step' failed with arguments {'dimension': 'style', 'choice': 'Figurative Language'}. Error: Missing required choice for dimension 'style'",
				),
			],
			None,                             	# expected_exception
			id="native_action_space_intervention_styles",
		),

		pytest.param(
			GenerateArgumentWithReasoning,    	# signature
			5,                                	# max_reasoning_steps
			[DEFAULT_TOOL],                  	# tools
			"all",                           	# action_space_paths_spec
			None,                             	# early_stopping_enabled
			None,                             	# forced_choice_function
			True,                             	# use_native_tool_calls
			None,                             	# verbosity
			{									# mock_predictions_data
				"tool_calls": [SimpleNamespace(tool_calls=[SimpleNamespace(
					name="UnknownTool",
					args={}
				)])],
				"error": [
					ExecutionError(
						error_type="parsing",
						error_message="Unknown action: 'UnknownTool'",
						raw_output=None,
					)
				],
				"error_type": ["parsing"],
				"error_message": ["Unknown action: 'UnknownTool'"],
				"raw_output": [None],
			},
			1,                                	# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Unknown action: 'UnknownTool'",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Unknown action: 'UnknownTool'",
				),
			],
			None,                             	# expected_exception
			id="native_unknown_or_empty_tool_drops_to_prune",
		),
		pytest.param(
			GenerateArgumentWithReasoning,    	# signature
			5,                                	# max_reasoning_steps
			[DEFAULT_TOOL],                   	# tools
			"all",                            	# action_space_paths_spec
			True,                             	# early_stopping_enabled (needed for FINISH_TOOL)
			None,                             	# forced_choice_function
			True,                             	# use_native_tool_calls
			None,                             	# verbosity
			{									# mock_predictions_data
				"tool_calls": [SimpleNamespace(tool_calls=[
					SimpleNamespace(name="continue_reasoning", args={}),
					SimpleNamespace(name="finish", args={}),
				])],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			1,                                	# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=DEFAULT_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
			],
			None,                             	# expected_exception
			id="single_completion_multiple_calls_selects_first",
		),
		pytest.param(
			GenerateArgumentWithReasoning,    	# signature
			5,                                	# max_reasoning_steps
			[DEFAULT_TOOL],                   	# tools
			"all",                            	# action_space_paths_spec
			None,                             	# early_stopping_enabled
			None,                             	# forced_choice_function
			True,                             	# use_native_tool_calls
			None,                             	# verbosity
			{									# mock_predictions_data
				"tool_calls": [SimpleNamespace(tool_calls=[
					SimpleNamespace(name="UnknownTool", args={}),
					SimpleNamespace(name="continue_reasoning", args={}),
				])],
				"error": [
					ExecutionError(
						error_type="parsing",
						error_message="Unknown action: 'UnknownTool'",
						raw_output=None,
					)
				],
				"error_type": ["parsing"],
				"error_message": ["Unknown action: 'UnknownTool'"],
				"raw_output": [None],
			},
			1,                                	# expected_n
			[									# expected_predictions
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Unknown action: 'UnknownTool'",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Unknown action: 'UnknownTool'",
				),
			],
			None,                             	# expected_exception
			id="single_unknown_first_call_drops_completion",
		),

		pytest.param(
			GenerateArgumentWithReasoning,    # signature
			5,                                # max_reasoning_steps
			[DEFAULT_TOOL],                   # tools
			"all",                            # action_space_paths_spec
			None,                             # early_stopping_enabled
			None,                             # forced_choice_function
			True,                             # use_native_tool_calls
			None,                             # verbosity
			# mock_predictions_data: empty tool_calls list (drops completion)
			{
				"tool_calls": [SimpleNamespace(tool_calls=[])],
				"error": [ExecutionError()],
				"error_type": [None],
				"error_message": [None],
				"raw_output": [None],
			},
			1,                                # expected_n
			[
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Empty tool_calls",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Empty tool_calls",
				),
			],
			None,                             # expected_exception
			id="single_empty_tool_calls_drops_completion",
		),

		# =============================================================================
		# Multiple Completions (3 cases)
		# =============================================================================

		pytest.param(
			GenerateArgumentWithReasoning,    # signature
			5,                                # max_reasoning_steps
			[DEFAULT_TOOL],                   # tools
			"all",                            # action_space_paths_spec
			True,                             # early_stopping_enabled (needed for FINISH_TOOL)
			None,                             # forced_choice_function
			True,                             # use_native_tool_calls
			None,                             # verbosity
			# mock_predictions_data: multiple completions with deduplication
			# Pattern: continue, finish, continue (dup), unknown -> dedups to continue(2), finish(1), prune(1)
			{
				"tool_calls": [
					SimpleNamespace(tool_calls=[SimpleNamespace(
						name="continue_reasoning",
						args={}
					)]),
					SimpleNamespace(tool_calls=[SimpleNamespace(
						name="finish",
						args={}
					)]),
					SimpleNamespace(tool_calls=[SimpleNamespace(
						name="continue_reasoning",
						args={}
					)]),
					SimpleNamespace(tool_calls=[SimpleNamespace(
						name="UnknownTool",
						args={}
					)]),
				],
				"error": [
					ExecutionError(),
					ExecutionError(),
					ExecutionError(),
					ExecutionError(
						error_type="parsing",
						error_message="Unknown action: 'UnknownTool'",
						raw_output=None,
					),
				],
				"error_type": [None, None, None, "parsing"],
				"error_message": [None, None, None, "Unknown action: 'UnknownTool'"],
				"raw_output": [None, None, None, None],
			},
			4,                                # expected_n
			[
				ControllerPrediction(
					tool=DEFAULT_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="N/A (native tool call)",
					num_occurrences=2,  # Deduplicated
					error="",
				),
				ControllerPrediction(
					tool=FINISH_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=False),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Unknown action: 'UnknownTool'",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Unknown action: 'UnknownTool'",
				),
			],
			None,                             # expected_exception
			id="multi_preserves_order_and_deduplicates",
		),

		pytest.param(
			GenerateArgumentWithReasoning,    # signature
			5,                                # max_reasoning_steps
			[DEFAULT_TOOL],                   # tools
			"all",                            # action_space_paths_spec
			True,                             # early_stopping_enabled (needed for FINISH_TOOL)
			None,                             # forced_choice_function
			True,                             # use_native_tool_calls
			None,                             # verbosity
			# mock_predictions_data: skips failed first completion, processes valid ones
			{
				"tool_calls": [
					SimpleNamespace(tool_calls=[SimpleNamespace(name="continue_reasoning", args={})]),
					SimpleNamespace(tool_calls=[SimpleNamespace(name="finish", args={})]),
				],
				"error": [
					ExecutionError(error_type="parsing", error_message="Parsing failed"),
					ExecutionError(),
				],
				"error_type": ["parsing", None],
				"error_message": ["Parsing failed", ""],
				"raw_output": [None, None],
			},
			2,                                # expected_n
			[
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Parsing failed",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Parsing failed",
				),
				ControllerPrediction(
					tool=FINISH_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=False),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
			],
			None,                             # expected_exception
			id="multi_processes_error_and_valid_completions",
		),

		pytest.param(
			GenerateArgumentWithReasoning,    # signature
			5,                                # max_reasoning_steps
			[DEFAULT_TOOL],                   # tools
			"all",                            # action_space_paths_spec
			True,                             # early_stopping_enabled (needed for FINISH_TOOL)
			None,                             # forced_choice_function
			True,                             # use_native_tool_calls
			None,                             # verbosity
			# mock_predictions_data: ignores empty tool_calls entries in middle
			{
				"tool_calls": [
					SimpleNamespace(tool_calls=[SimpleNamespace(name="continue_reasoning",args={})]),
					SimpleNamespace(tool_calls=[]),  # Empty - should drop to PRUNE
					SimpleNamespace(tool_calls=[SimpleNamespace(name="finish", args={})]),
				],
				"error": [
					ExecutionError(),
					ExecutionError(),
					ExecutionError(),
				],
				"error_type": [None, None, None],
				"error_message": [None, None, None],
				"raw_output": [None, None, None],
			},
			3,                                # expected_n
			[
				ControllerPrediction(
					tool=DEFAULT_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
				ControllerPrediction(
					tool=PRUNE_TOOL,
					chosen_values={
						"error_type": "parsing",
						"error_message": "Empty tool_calls",
						"raw_output": None,
					},
					intervention=ReasoningIntervention(continue_reasoning=True),
					considerations="Controller failure: 'parsing'",
					num_occurrences=1,
					error="Empty tool_calls",
				),
				ControllerPrediction(
					tool=FINISH_TOOL,
					chosen_values={},
					intervention=ReasoningIntervention(continue_reasoning=False),
					considerations="N/A (native tool call)",
					num_occurrences=1,
					error="",
				),
			],
			None,                             # expected_exception
			id="multi_handles_empty_tool_calls_entries",
		),
	],
)
def test_create_controller_predictions(
	signature: type[ReasoningSignature],
	max_reasoning_steps: int,
	tools: list[dspy.Tool] | None,
	action_space_paths_spec: str | None,
	early_stopping_enabled: bool | None,
	forced_choice_function: Any,
	use_native_tool_calls: bool,
	verbosity: Literal["debug", "info", "warning", "error"] | None,
	mock_predictions_data: dict[str, Any],
	expected_n: int,
	expected_predictions: list[ControllerPrediction] | None,
	expected_exception: type[BaseException] | None,
	temp_action_space_styles: str,
	temp_action_space_structures: str,
	temp_action_space_subtopics: str,
) -> None:
	"""
	Comprehensive test for create_controller_predictions method.

	Tests error handling, native tool calls, single/multiple completion edge cases,
	and deduplication logic across various controller configurations.
	"""
	# Resolve action_space_paths from spec
	action_space_paths = None
	if action_space_paths_spec == "all":
		action_space_paths = [
			temp_action_space_styles,
			temp_action_space_structures,
			temp_action_space_subtopics,
		]

	# Create controller with provided __init__ parameters
	if verbosity is not None:
		controller = TreeOfThoughtsController(
			signature=signature,
			max_reasoning_steps=max_reasoning_steps,
			tools=tools,
			action_space_paths=action_space_paths,
			early_stopping_enabled=early_stopping_enabled,
			forced_choice_function=forced_choice_function,
			use_native_tool_calls=use_native_tool_calls,
			verbosity=verbosity,
		)
	else:
		controller = TreeOfThoughtsController(
			signature=signature,
			max_reasoning_steps=max_reasoning_steps,
			tools=tools,
			action_space_paths=action_space_paths,
			early_stopping_enabled=early_stopping_enabled,
			forced_choice_function=forced_choice_function,
			use_native_tool_calls=use_native_tool_calls,
		)

	# Create mock prediction from data
	if use_native_tool_calls:
		# Native format: use SimpleNamespace
		mock_prediction = SimpleNamespace(completions=mock_predictions_data)
	else:
		# Legacy format: use dspy.Prediction.from_completions
		mock_prediction = dspy.Prediction.from_completions(mock_predictions_data)

	# Execute method under test
	if expected_exception is not None:
		with pytest.raises(expected_exception):
			controller.create_controller_predictions(
				mock_prediction,
				expected_n=expected_n,
			)
	else:
		predictions = controller.create_controller_predictions(
			mock_prediction,
			expected_n=expected_n,
		)

		# Validate predictions match expected
		_validate_controller_predictions_equality(predictions, expected_predictions)


# =============================================================================
# Integration Tests (GPU Required)
# =============================================================================


@pytest.fixture(scope="module")
def shared_gpu_model():
	"""Shared GenerativeLocalVLLM fixture for all GPU integration tests."""
	if not torch.cuda.is_available():
		pytest.skip("GPU not available")

	base_path = "/projects/BSTEWART/model_storage"
	model_name = OpenSourceModel.QWEN_3_30B_A3B_INSTRUCT_2507.value
	model_path = os.path.join(base_path, model_name)

	lm = None
	try:
		logger.info(f"Initializing shared GPU model from: {model_path}")
		lm = GenerativeLocalVLLM(
			model=model_path,
			tensor_parallel_size=1,
			dtype="auto",
			gpu_memory_utilization=0.9,
			max_model_len=16_384,
			enforce_eager=True,
			verbosity="debug",
		)
		logger.info("Shared GPU model initialized successfully")
		dspy.settings.configure(lm=lm)
		yield lm
	finally:
		if lm is not None:
			logger.info("Cleaning up shared GPU model...")
			lm.kill()


@pytestmark_gpu
class TestControllerIntegration:
	"""Integration tests for the controller using real models (requires GPU)."""

	@pytest.fixture(scope="class")
	def local_lm(self, shared_gpu_model):
		"""Use the shared GPU model fixture."""
		return shared_gpu_model

	@pytest.fixture
	def controller(self, local_lm):
		"""Create a controller instance with real LM."""
		dspy.settings.configure(lm=local_lm)
		return TreeOfThoughtsController(
			signature=SolveMathProblemWithReasoning,
			max_reasoning_steps=5,
			tools=[DEFAULT_TOOL],
			early_stopping_enabled=True,
			verbosity="info",
		)

	@pytest.fixture
	def limited_causal_structures_action_space_path(self, tmp_path) -> str:
		"""Create a 2-choice structures action space derived from causal_structures.json."""
		with open(
			"/Users/zachary/dspy_reasoning/experiments/argument_generation/action_space/causal_structures.json",
			encoding="utf-8",
		) as f:
			source = json.load(f)

		choices = source["choices"]
		limited = {
			"name": "structure",
			"definition": source["definition"],
			"choices": {
				"concession_and_contrast": choices["concession_and_contrast"],
				"evidence_and_authority": choices["evidence_and_authority"],
			},
		}
		path = tmp_path / "limited_structures.json"
		with open(path, "w", encoding="utf-8") as f:
			json.dump(limited, f, indent="\t")
		return str(path)

	@pytest.fixture
	def limited_causal_subtopics_action_space_path(self, tmp_path) -> str:
		"""Create a 2-choice subtopics action space derived from causal_subtopics.json."""
		with open(
			"/Users/zachary/dspy_reasoning/experiments/argument_generation/action_space/causal_subtopics.json",
			encoding="utf-8",
		) as f:
			source = json.load(f)

		choices = source["choices"]
		limited = {
			"name": "stock_issue",
			"definition": source["definition"],
			"choices": {
				"risk_and_unintended_consequences": choices["risk_and_unintended_consequences"],
				"cost_benefit_and_impact_analysis": choices["cost_benefit_and_impact_analysis"],
			},
		}
		path = tmp_path / "limited_subtopics.json"
		with open(path, "w", encoding="utf-8") as f:
			json.dump(limited, f, indent="\t")
		return str(path)

	@pytest.mark.parametrize(
		# Parameter names
		[
			"state",
			"action_space_keys",
			"expected_choices_to_make",
			"expected_choices_to_avoid",
			"decision_rationale",
		],
		# Parameter values
		[
			# Structural Transition Tests
			# Clear cut Rebuttal (However)
			pytest.param(
				State(
					input={"topic": "The earth is flat.", "stance": "ANTI"},
					reasoning=[{"claim": "Note that from a local perspective on the ground, the horizon appears to be a flat line."}],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[												# Expected choices to make
					(
						# The earth is not flat, so we expect a "However".
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Contrast"},
					),
					(
						# Evidence that the earth is round or evidence refuting the flat view is also valid
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Evidence & Support"},
					)
				],
				[												# Expected choices to avoid
					(
						# The model is unlikely to explain the implication of the eath being flat.
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Causal Reasoning"},
					)
				],
				"Misleading 'Note that' premise requires 'However' transition to correct it",
				id="structure_preference_contrast_for_rebuttal",
			),
			pytest.param(
				State(
					input={
						"topic": "Early education provides benefits that justify increased funding.",
						"stance": "PRO"
					},
					reasoning=[{"claim": "Studies show that early education has long-term benefits."}],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[												# Expected choices to make
					(DEFAULT_REASONING_INTERVENTION_TOOL_NAME, {"structure": "Evidence & Support"})
				],
				[												# Expected choices to avoid
					(DEFAULT_REASONING_INTERVENTION_TOOL_NAME, {"structure": "Contrast"})
				],
				"Claim needs 'Evidence & Support' to back it up",
				id="structure_preference_evidence_for_claim",
			),
			# Clear cut Deduction (Therefore)
			pytest.param(
				State(
					input={"topic": "Socrates is mortal.", "stance": "PRO"},
					reasoning=[
						{"claim": "Note that all human beings are mortal, and that Socrates is a human being."}
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["structures"],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Causal Reasoning"},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Evidence & Support"},
					)
				],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"structure": "Contrast"},
					)
				],
				"Syllogism premises 'Note that...' require 'Therefore' conclusion",
				id="structure_preference_deductive_syllogism",
			),
			# Style Transition Tests
			pytest.param(
				State(
					input={"topic": "Renewable energy investment is a good idea", "stance": "PRO"},
					reasoning=[
						{"claim": "Renewable energy is like planting seeds for future generations - "},
						{"claim": "we invest today to harvest clean power tomorrow."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["styles"],  									# Only styles action space
				[												# Expected choices to make
					(DEFAULT_REASONING_INTERVENTION_TOOL_NAME, {"style": "Statistical & Data-Driven"})
				],
				[												# Expected choices to avoid
					(DEFAULT_REASONING_INTERVENTION_TOOL_NAME, {"style": "Figurative Language"})
				],
				"After figurative language, should transition to data/statistical style for balance",
				id="style_figurative_to_statistical",
			),
			pytest.param(
				State(
					input={"topic": "Education reform", "stance": "PRO"},
					reasoning=[
						{"claim": "Studies show that smaller class sizes improve student outcomes by 15-20%."},
						{"claim": "Data from OECD countries reveals a strong correlation between teacher-student ratios and academic performance."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["styles"],
				[												# Expected choices to make
					(DEFAULT_REASONING_INTERVENTION_TOOL_NAME, {"style": "Figurative Language"})
				],
				[												# Expected choices to avoid
					(DEFAULT_REASONING_INTERVENTION_TOOL_NAME, {"style": "Statistical & Data-Driven"})
				],
				"After multiple statistical statements, should vary style (e.g., narrative or figurative)",
				id="style_avoid_statistical_repetition",
			),
			# Early Stopping Tests (2 choices only - should pick the clear winner)
			pytest.param(
				State(
					input={
						"topic": "Space exploration deserves more funding.",
						"stance": "PRO",
					},
					reasoning=[
						{"claim": "Space exploration drives technological innovation that benefits society."},
						{"claim": "Historical evidence: NASA research led to GPS, memory foam, and water purification."},
						{"claim": "The economic multiplier effect: every dollar invested returns $7-14 to the economy."},
						{"claim": "Therefore, increasing space exploration funding is a wise investment in our future."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				None,  # No action space - only continue_reasoning vs finish
				[("finish", None)],
				[("continue_reasoning", None)],
				"Complete argument with claim, evidence, and conclusion - should finish",
				id="early_stopping_complete_argument_finish",
			),
			pytest.param(
				State(
					input={
						"topic": "Digital privacy rights",
						"stance": "PRO",
					},
					reasoning=[
						{"claim": "Digital privacy is a fundamental human right in the modern age."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				None,
				[("continue_reasoning", None)],
				[("finish", None)],
				"Only a claim without evidence or conclusion - should continue reasoning",
				id="early_stopping_incomplete_argument_continue",
			),
			# Combined Structural + Style Test
			pytest.param(
				State(
					input={
						"topic": "AI regulation",
						"stance": "PRO",
					},
					reasoning=[
						{"claim": "AI systems must be regulated to prevent societal harm."},
						{"claim": "Consider AI like a powerful river - without proper channels and dams, it can flood and destroy everything in its path."},
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),

				["styles", "structures"],  # Multi-dimensional: Style + Structure
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"style": "Statistical & Data-Driven"},
					)
				],
				None,
				"Claim with figurative language needs statistical data (style transition)",
				id="combined_claim_with_figurative_needs_evidence",
			),
			# Subtopic Selection Test
			# Context makes one subtopic clearly relevant (Economic) vs clearly irrelevant (Social Justice)
			pytest.param(
				State(
					input={
						"topic": "Impact of inflation on consumer purchasing power",
						"stance": "PRO",
					},
					reasoning=[
						{"claim": "First, rising prices reduce aggregate demand and slow GDP growth."}
					],
					controller_output_trajectory=[
						create_dummy_controller_output(
							action=DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
							action_arguments={"stock_issue": "General Introduction"},
							considerations="I want to introduce the concept of inflation.",
						)
					],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["subtopics"],  # Just subtopics
				[
					(
						# Likely to talk about economic impact because it's a clear economic issue
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"stock_issue": "Economic Impact"},
					),
					(
						# Likely to talk about social justice because the economic issue has social implications
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{"stock_issue": "Social Justice & Equity"},
					)
				],
				None,
				"Topic of inflation/purchasing power requires Economic Impact subtopic",
				id="subtopic_relevance_economic",
			),
			# Multi-dimensional: Subtopic + Style
			pytest.param(
				State(
					input={
						"topic": "Poetry is good for the soul",
						"stance": "PRO",
					},
					reasoning=[
						{"claim": "First, here is a beautiful poem by Emily Dickinson:"}
					],
					controller_output_trajectory=[],
					controller_outputs=[],
					feedback=[],
					output={},
				),
				["subtopics", "styles"],
				[
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "General Introduction",
							"style": "Figurative Language"
						},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Economic Impact",
							"style": "Figurative Language"
						},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
						{
							"stock_issue": "Social Justice & Equity",
							"style": "Figurative Language"
						},
					),
					(
						DEFAULT_REASONING_INTERVENTION_TOOL_NAME,

						{
							"stock_issue": "Health and Safety",
							"style": "Figurative Language"
						},
					),
				],
				None,
				"Poetry analysis requires Figurative Language style",
				id="subtopic_and_style_relevance",
			),
		],
	)
	def test_semantic_validation(
		self,
		controller,
		temp_action_space_styles,
		temp_action_space_structures,
		temp_action_space_subtopics,
		state: State,
		action_space_keys: list[str] | None,
		expected_choices_to_make: list[tuple[str, dict[str, Any] | None]] | None,
		expected_choices_to_avoid: list[tuple[str, dict[str, Any] | None]] | None,
		decision_rationale: str,
	):
		"""Test semantic correctness of controller decisions with real LM.

		This test validates that the controller makes semantically appropriate decisions
		for structural transitions, style transitions, and early stopping scenarios.
		"""
		# Convert fixture keys to actual paths
		action_space_paths = None
		if action_space_keys is not None:
			action_space_paths = []
			for key in action_space_keys:
				if key == "styles":
					action_space_paths.append(temp_action_space_styles)
				elif key == "structures":
					action_space_paths.append(temp_action_space_structures)
				elif key == "subtopics":
					action_space_paths.append(temp_action_space_subtopics)

		# Create controller with appropriate action space
		test_controller = TreeOfThoughtsController(
			signature=GenerateArgumentWithReasoning,
			max_reasoning_steps=5,
			action_space_paths=action_space_paths,
			early_stopping_enabled=True,
		)

		# Select demos based on action space configuration
		# Uses argument-based demos since test uses GenerateArgumentWithReasoning
		# - No action space (continue/finish only): use argument continue/finish demos
		# - Styles only: use style demos
		# - Structures only: use structure demos
		# - Both: use combined style+structure demos
		if action_space_keys is None:
			demos_to_use = ARGUMENT_CONTINUE_FINISH_DEMOS  # argument fields + continue/finish
		elif action_space_keys == ["styles"]:
			demos_to_use = STYLE_CONTROLLER_DEMOS
		elif action_space_keys == ["structures"]:
			demos_to_use = STRUCTURE_CONTROLLER_DEMOS
		else:  # Both styles and structures
			demos_to_use = STYLE_STRUCTURE_CONTROLLER_DEMOS

		try:
			controller_result = test_controller(
				states=state,
				n_samples_generation=1,
				temperature=0.1,
				demos=demos_to_use,
			)

			# Extract actual decision
			# Extract actual decision and args
			actual_tool_name = None
			actual_args = None
			if (
				controller_result
				and isinstance(controller_result, list)
				and len(controller_result) > 0
			):
				first_state_result = controller_result[0]
				if isinstance(first_state_result, list) and len(first_state_result) > 0:
					first_response = first_state_result[0]
					if hasattr(first_response, "tool"):
						actual_tool_name = first_response.tool.name
						actual_args = first_response.chosen_values

			# Check expected choices to make (ANY of these)
			if expected_choices_to_make is not None:
				found_match = False
				for expected_tool_name, expected_args in expected_choices_to_make:
					if actual_tool_name == expected_tool_name:
						if expected_args is None:
							found_match = True  # Tool match is sufficient if no args expected
							break
						# Check if expected args are subset of actual args
						if actual_args and all(
							actual_args.get(k) == v for k, v in expected_args.items()
						):
							found_match = True
							break

				assert found_match, (
					f"Semantic validation failed: {decision_rationale}\n"
					f"Expected one of: {expected_choices_to_make}\n"
					f"Got: {actual_tool_name} with args {actual_args}\n"
					f"State input: {state.input}\n"
					f"State reasoning: {state.reasoning}"
				)

			# Check expected choices to avoid (NONE of these)
			if expected_choices_to_avoid is not None:
				for avoid_tool_name, avoid_args in expected_choices_to_avoid:
					is_match = False
					if actual_tool_name == avoid_tool_name:
						if avoid_args is None:
							is_match = True
						elif actual_args and all(
							actual_args.get(k) == v for k, v in avoid_args.items()
						):
							is_match = True

					assert not is_match, (
						f"Semantic validation failed: {decision_rationale}\n"
						f"Should have avoided: {avoid_tool_name} with args {avoid_args}\n"
						f"But got exactly that.\n"
						f"State input: {state.input}\n"
						f"State reasoning: {state.reasoning}"
					)


			logger.info(f"✓ Semantic validation passed: {decision_rationale}")

		except Exception as e:
			pytest.fail(f"Semantic validation failed: {e}")

if __name__ == "__main__":
	gpu_available = torch.cuda.is_available()
	if not gpu_available:
		pytest.main([__file__, "-vv"])
	else:
		pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
