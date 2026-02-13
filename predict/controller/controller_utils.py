"""
Utility functions shared between TreeOfThoughtsController and TreeOfThoughtsControllerReranker.
"""

# Standard library imports
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Third-party imports
import dspy
from dspy import Tool

# Local imports
from predict.controller.controller_constants import (
	ReasoningIntervention,
)
from tree import State

logger = logging.getLogger(__name__)


@dataclass
class ActionSpaceConfig:
	"""
	Configuration loaded from an action space JSON file.

	Attributes:
		name: The name of the dimension (e.g., "Causal Structures").
		definition: Description of what interventions along this dimension do.
		choices: Dictionary mapping choice names to their specifications.
			Each choice spec may contain 'definition', 'internal_reasoning', and/or 'prefix'.
	"""

	name: str
	definition: str
	choices: dict[str, dict[str, str]]


class ControllerPrediction(dspy.Prediction):
	"""
	A prediction from the controller representing a tool choice and its intervention.

	This specialized Prediction contains the controller's decision (action + arguments)
	and the resulting intervention from executing the chosen tool. It provides named
	parameters with reasonable defaults for all controller output fields.

	Attributes:
		tool: The chosen dspy.Tool object.
		chosen_values: Arguments passed to the tool (default: empty dict).
		intervention: The ReasoningIntervention from executing the tool.
		considerations: Justification for why this tool and arguments were chosen.
		error: Error message if the controller failed at any stage (default: "").
		num_occurrences: Number of times this unique action+arguments was chosen (default: 1).
		score: Optional relevance score (e.g., from reranker) for this prediction.
	"""

	def __init__(
		self,
		tool: Tool,
		chosen_values: dict[str, Any],
		intervention: ReasoningIntervention,
		considerations: str = "",
		error: str = "",
		num_occurrences: int = 1,
		score: float | None = None,
	) -> None:
		"""
		Initialize a ControllerPrediction.

		Parameters:
			tool: The tool that was chosen. This is the tool that will be executed.
			chosen_values: The values that were chosen for the tool's arguments.
			intervention: The intervention to apply to the next generation.
			considerations: Justification for why this tool and arguments were chosen.
			error: Error message if the controller failed at any stage. Defaults to "".
			num_occurrences: Number of times this unique action+arguments combination was
				chosen by the controller. Defaults to 1.
			score: Relevance score assigned by the reranker (if applicable).
		"""
		super().__init__(
			tool=tool,
			chosen_values=chosen_values,
			intervention=intervention,
			considerations=considerations,
			error=error,
			num_occurrences=num_occurrences,
		)
		self.score = score


# Type alias for forced choice functions.
# These functions receive available tools (tool_name -> Tool) and state,
# and return a list of (action_name, action_arguments, considerations) tuples or None.
# Each tuple represents a forced prediction, allowing multiple branches from a single node.
# The considerations string explains why this action was chosen given the state and tools.
ForcedChoiceFunction = Callable[
	[dict[str, dspy.Tool], State], list[tuple[str, dict[str, Any], str]] | None
]

# Define basic Tools
DEFAULT_TOOL = dspy.Tool(
	name="continue_reasoning",
	func=lambda: ReasoningIntervention(continue_reasoning=True),
	desc="""
Signals that additional reasoning is needed to solve the task. If the reasoning so far is
insufficient, the answer seems uncertain, or you believe there is a mistake in the reasoning so
far, select this option.
""".strip(),
	args={},
)


FINISH_TOOL = dspy.Tool(
	name="finish",
	func=lambda: ReasoningIntervention(continue_reasoning=False),
	desc="""
Signals that the reasoning so far is sufficient for producing a high-quality response for the task.
If selected, the next step will involve generating the final output rather than reasoning further.
""".strip(),
	args={},
)


def create_finish_tool(description: str | None = None) -> dspy.Tool:
	"""
	Create a finish tool with an optional custom description.

	Parameters:
		description: Optional custom description for the finish tool. If None, uses the default
			FINISH_TOOL description.

	Returns:
		A dspy.Tool instance for finishing reasoning.
	"""
	if description is None:
		return FINISH_TOOL

	return dspy.Tool(
		name="finish",
		func=lambda: ReasoningIntervention(continue_reasoning=False),
		desc=description.strip(),
		args={},
	)

# Internal-only tool used to represent controller failures (not exposed to the model).
PRUNE_TOOL = dspy.Tool(
	name="prune",
	func=lambda: ReasoningIntervention(continue_reasoning=True),
	desc="Internal marker for controller failure; not callable by the model.",
	args={},
)


def create_reasoning_intervention_from_choices(
	configs: list[ActionSpaceConfig],
	choices: dict[str, str],
) -> ReasoningIntervention:
	"""
	Create a ReasoningIntervention by selecting one choice per dimension.

	This function translates from what the model perceives to be the right course of action
	to a concrete intervention over the next reasoning step.

	Parameters:
		configs: List of ActionSpaceConfig objects, one per dimension.
		choices: Dictionary mapping parameter names to chosen values. Each key is a
			parameter/dimension name (derived from config.name via sanitize_param_name),
			and each value is the choice for that dimension.

	Returns:
		A ReasoningIntervention specifying how to influence the next reasoning step.

	Raises:
		ValueError: If a required parameter is missing or an unknown choice is specified.
	"""
	internal_reasoning_parts: list[str] = []
	prefix_parts: list[str] = []

	for config in configs:
		param_name = sanitize_param_name(config.name)
		choice = choices.get(param_name)
		if choice is None:
			raise ValueError(f"Missing required choice for dimension '{config.name}'")

		if choice not in config.choices:
			raise ValueError(
				f"Unknown choice '{choice}' for dimension '{config.name}'. "
				f"Available choices: {list(config.choices.keys())}"
			)

		choice_spec = config.choices[choice]

		# Collect internal_reasoning if present
		if "internal_reasoning" in choice_spec:
			internal_reasoning_parts.append(choice_spec["internal_reasoning"])

		# Collect prefix if present
		if "prefix" in choice_spec:
			prefix_parts.append(choice_spec["prefix"])

	# Combine parts
	internal_reasoning = " ".join(internal_reasoning_parts) if internal_reasoning_parts else ""
	prefix = " ".join(prefix_parts) if prefix_parts else ""

	return ReasoningIntervention(
		continue_reasoning=True,
		internal_reasoning=internal_reasoning,
		prefix=prefix,
	)


def load_action_space_json(json_path: str | Path) -> ActionSpaceConfig:
	"""
	Load an action space configuration from a JSON file.

	Parameters:
		json_path: Path to the JSON file containing the action space configuration.

	Returns:
		ActionSpaceConfig containing the loaded configuration.

	Raises:
		FileNotFoundError: If the JSON file does not exist.
		ValueError: If the JSON structure is invalid.
	"""
	path = Path(json_path)
	if not path.exists():
		raise FileNotFoundError(f"Action space JSON not found: {path}")

	with open(path, encoding="utf-8") as f:
		data = json.load(f)

	# Validate required fields
	if "name" not in data:
		raise ValueError(f"Action space JSON missing 'name' field: {path}")
	if "definition" not in data:
		raise ValueError(f"Action space JSON missing 'definition' field: {path}")
	if "choices" not in data:
		raise ValueError(f"Action space JSON missing 'choices' field: {path}")

	return ActionSpaceConfig(
		name=sanitize_param_name(data["name"]),
		definition=data["definition"],
		choices=data["choices"],
	)


def sanitize_param_name(name: str) -> str:
	"""
	Sanitize a dimension name to be used as a parameter name.

	Converts spaces and special characters to underscores, removes consecutive
	underscores, and converts to lowercase.

	Parameters:
		name: The dimension name to sanitize.

	Returns:
		A valid Python identifier suitable for use as a parameter name.
	"""
	param_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
	param_name = "".join(c if c.isalnum() or c == "_" else "_" for c in param_name)
	while "__" in param_name:
		param_name = param_name.replace("__", "_")
	return param_name.strip("_")


def create_literal_from_dict(options_dict: dict[str, Any]) -> type[Literal]:
	"""
	Create a Literal type from a dictionary's keys.

	This function dynamically creates a Literal type annotation from the keys
	of a dictionary, which is useful for creating type overrides for tools
	with parameter options.

	Parameters:
	    options_dict (dict[str, Any]): Dictionary whose keys will become Literal values.

	Returns:
	    type[Literal]: A Literal type containing all keys from the dictionary.

	Example:
	    >>> options = {"Option1": {...}, "Option2": {...}}
	    >>> literal_type = create_literal_from_dict(options)
	    >>> # literal_type is Literal["Option1", "Option2"]
	"""
	keys: tuple[str, ...] = tuple(options_dict.keys())
	return Literal[*keys]




def execute_tool_safely(
	tool: Tool, action_arguments: dict[str, Any] | None
) -> tuple[ReasoningIntervention, str]:
	"""
	Execute a tool safely, handling missing or incorrect arguments.

	Parameters:
	    tool (Tool): The tool to execute.
	    action_arguments (dict[str, Any] | None): The arguments to pass to the tool.

	Returns:
	    tuple[ReasoningIntervention, str]: A tuple of (intervention, error_message).
	        If execution succeeds, error_message is empty string.
	        If execution fails, intervention has continue_reasoning=False and error_message
	        contains details about the tool, arguments, and error.
	"""
	args = action_arguments or {}
	try:
		result: dict[str, Any] = tool.func(**args)

		# Extract intervention from tool result
		if isinstance(result, ReasoningIntervention):
			intervention = result
		else:
			intervention = ReasoningIntervention(
				continue_reasoning=result["continue_reasoning"],
				internal_reasoning=result["internal_reasoning"],
				prefix=result["prefix"],
			)
		return intervention, ""

	except Exception as e:
		# Create error message with tool name, arguments, and error details.
		error_message = (f"Tool '{tool.name}' failed with arguments {args}. Error: {str(e)}")
		# Return intervention with continue_reasoning=False (i.e., generate final output).
		failed_intervention = ReasoningIntervention(
			continue_reasoning=False,
			internal_reasoning="",
			prefix="",
		)
		return failed_intervention, error_message


def remove_duplicate_actions_with_counts(
	output_dictionaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	"""
	Remove duplicates from a list of dictionaries based on "action" and "action_arguments",
	and annotate each unique dict with "unique_action_response_count".

	Parameters:
	    output_dictionaries (list[dict[str, Any]]): List of action dictionaries.

	Returns:
	    list[dict[str, Any]]: Deduplicated list with count annotations.
	"""
	seen = {}

	for d in output_dictionaries:
		# Create a key by converting "action_arguments" to a sorted string representation
		action_key = d["action"]
		args_key = (
			str(sorted(d["action_arguments"].items()))
			if d["action_arguments"]
			else ""
		)
		key = (action_key, args_key)

		if key in seen:
			seen[key]["unique_action_response_count"] += 1
		else:
			seen[key] = {**d, "unique_action_response_count": 1}

	return list(seen.values())

def return_action_if_single_option(
	tools: dict[str, dspy.Tool],
	state: State,
) -> list[tuple[str, dict[str, Any], str]] | None:
	"""
	Check if only a single option is available (continue_reasoning or finish).

	If exactly one of these options is available, returns a list with one forced choice.
	Otherwise returns None to indicate the controller should proceed normally.

	Parameters:
		tools: Dictionary mapping tool names to Tool instances.
		state: The current state (unused, kept for signature consistency).

	Returns:
		List of (action_name, action_arguments, considerations) tuples if forced,
		or None otherwise. The list allows for multiple branches from a single node.
	"""
	if len(tools) == 1:
		action_name = list(tools.keys())[0]
		considerations = (
			f"Forced choice: '{action_name}' was the only available action. "
			f"No other tools were provided to the controller."
		)
		return [(action_name, {}, considerations)]
	return None
