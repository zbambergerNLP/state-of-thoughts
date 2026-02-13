"""
Controller: A DSPy module that determines the next action to take when solving a reasoning problem.
The controller's action choice maps to a textual prefix that steers a subsequent LLM generation.
"""

# Standard library imports
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_type_hints

# Third-party imports
import dspy
from dspy import Tool

# Local imports
from constants import VERBOSITY_TO_LOGGING_LEVEL
from misc_utils import (
	ExecutionError,
	parse_base_signature,
	parse_literal,
	safe_parse_dict,
)
from predict.controller.controller_constants import (
	DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
)
from predict.controller.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	PRUNE_TOOL,
	ActionSpaceConfig,
	ControllerPrediction,
	ForcedChoiceFunction,
	ReasoningIntervention,
	create_finish_tool,
	create_literal_from_dict,
	create_reasoning_intervention_from_choices,
	execute_tool_safely,
	load_action_space_json,
	return_action_if_single_option,
	sanitize_param_name,
)
from predict.local_predict import LocalPredict
from signatures import (
	ReasoningSignature,
	ensure_reasoning_signature,
)
from tree import State
from tree.tree_constants import ReasoningState

logger = logging.getLogger(__name__)


def _collapse_whitespace(text: str) -> str:
	"""
	Collapse all whitespace (including newlines) into single spaces.
	"""
	return " ".join(text.split())


def _parse_action_space_choices(arg_spec: str) -> list[str]:
	"""
	Parse choice names from an action-space arg spec.

	The controller's action-space tools store arg specs as:
		<dimension definition>\\nOptions:\\n- \"<choice>\": <definition>\\n...
	"""
	parts = arg_spec.split("\nOptions:\n", 1)
	if len(parts) < 2:
		return []

	choices: list[str] = []
	for line in parts[1].splitlines():
		line = line.strip()
		if not line.startswith('- "'):
			continue
		# Format is: - "<choice_name>": <definition>
		rest = line[len('- "'):]
		end_quote = rest.find('"')
		if end_quote == -1:
			continue
		choice_name = rest[:end_quote]
		choices.append(choice_name)
	return choices


def _format_literal_type(choices: list[str]) -> str:
	"""
	Format a Literal[...] type string for display.
	"""
	if not choices:
		return "str"
	escaped = [c.replace("\\", "\\\\").replace('"', '\\"') for c in choices]
	quoted = [f"\"{c}\"" for c in escaped]
	return f"Literal[{', '.join(quoted)}]"


def _format_tool_instruction_block(tool: Tool, tool_num: int) -> str:
	"""
	Render a single tool as a readable multi-line instruction block.
	"""
	description = _collapse_whitespace(tool.desc or "")
	lines: list[str] = [
		f"({tool_num}) `{tool.name}`:",
		"\t* Description:",
		f"\t\t{description}",
	]
	args: dict[str, Any] = getattr(tool, "args", {}) or {}
	if not args:
		return "\n".join(lines)

	lines.append("\t* Arguments:")
	arg_types: dict[str, Any] = getattr(tool, "arg_types", {}) or {}

	for arg_name in sorted(args.keys()):
		arg_spec = str(args[arg_name]).strip()
		arg_definition = arg_spec.split("\nOptions:\n", 1)[0].strip()

		arg_type = arg_types.get(arg_name)
		if arg_type is not None and hasattr(arg_type, "__args__"):
			type_choices = [x for x in getattr(arg_type, "__args__", []) if isinstance(x, str)]
			type_str = _format_literal_type(type_choices)
		else:
			type_str = _format_literal_type(_parse_action_space_choices(arg_spec))

		lines.append(f"\t\t- {arg_name}: {type_str}")
		lines.append(f"\t\t\t{arg_definition}")

	return "\n".join(lines)


class TreeOfThoughtsController(dspy.Module):
	"""
	Plans and intervenes on the next reasoning step in a Tree of Thoughts.

	Supports single-candidate generation which produces one action + argument(s) combination.
	We generate multiple candidates through high-temperature sampling (such that no completion is
	aware of the other candidates/completions).
	"""

	@staticmethod
	def _create_combined_action_tool_func(
		configs: list[ActionSpaceConfig]
	) -> Callable[[dict[str, str]], ReasoningIntervention]:
		"""
		Create a function that takes a dictionary of choices and returns a ReasoningIntervention.

		The tool takes one choice per dimension. When executed, it returns a ReasoningIntervention,
		which reflects whether to continue reasoning or generate final output. If continuing reasoning,
		it also includes an internal reasoning and/or a prefix to inject at the start of the next
		generation.

		Example:
		```
		{
			"dimension1": "choice1",
			"dimension2": "choice2",
			"dimension3": "choice3"
		}
		```
		Returns:
		```
		{
			"internal_reasoning": "The internal reasoning for the choice.",
			"prefix": "The prefix to inject at the start of the next generation."
		}
		```

		Parameters:
			configs: List of ActionSpaceConfig objects, one per dimension.

		Returns:
			A tool function for use in a dspy.Tool that returns a ReasoningIntervention.
		"""
		def tool_func(**kwargs) -> ReasoningIntervention:
			"""Creates a ReasoningIntervention from the Controller's choices."""
			intervention = create_reasoning_intervention_from_choices(configs, kwargs)
			return intervention
		return tool_func

	@staticmethod
	def _load_action_spaces_and_create_combined_tool(
		json_paths: list[str | Path],
		tool_name: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
	) -> Tool:
		"""
		Load multiple action space JSONs and create a SINGLE combined tool.

		For the generative controller, we create ONE tool with (potentially) multiple parameters,
		where each parameter corresponds to one dimension (action space JSON).
		The generative model is expected to generate one choice for each parameter given the
		permitted choices for each parameter.

		Parameters:
			json_paths: List of paths to action space JSON files. The generative controller must
				select one choice for each parameter (corresponding to one json file/dimension).
			tool_name: Name for the combined tool (default: DEFAULT_REASONING_INTERVENTION_TOOL_NAME).

		Returns:
			A single Tool with (potentially) multiple parameters.
		"""
		# Load all configs
		configs: list[ActionSpaceConfig] = []
		for path in json_paths:
			configs.append(load_action_space_json(path))

		# Create parameter names from dimension names
		param_names = [sanitize_param_name(config.name) for config in configs]
		assert len(param_names) == len(set(param_names)), "Parameters must have unique names"

		# Create combined tool function
		tool_func: Callable[[dict[str, str]], ReasoningIntervention] = (
			TreeOfThoughtsController._create_combined_action_tool_func(configs)
		)

		# Build combined description
		desc_lines = [
			"Determine the best choice in each dimension (corresponding to an argument of this function) to improve the quality of the next reasoning step.",
			"You **must** select **one** choice for **each** of the provided dimensions.",
			"Be mindful of the impact that each choice has on the next reasoning step.",
			"When providing a choice, avoid enumeration or unnecessary prefixes/suffixes -- just provide your choice directly."
		]
		tool_desc = "\n".join(desc_lines)

		# Build args dict (parameter descriptions)
		args: dict[str, str] = {}
		type_overrides: dict[str, Any] = {}
		for config, param_name in zip(configs, param_names, strict=True):
			# Build detailed argument description
			arg_desc_lines = [config.definition, "Options:"]
			for choice_name, choice_data in config.choices.items():
				definition = choice_data.get("definition", "")
				arg_desc_lines.append(f'- "{choice_name}": {definition}')
			args[param_name] = "\n".join(arg_desc_lines).strip()
			type_overrides[param_name] = create_literal_from_dict(config.choices)

		logger.debug(
			f"Created combined tool '{tool_name}' with {len(configs)} dimensions: "
			f"`{[c.name for c in configs]}`"
		)
		return dspy.Tool(
			name=tool_name,
			func=tool_func,
			desc=tool_desc,
			args=args,
			arg_types=type_overrides,
		)

	def create_basic_tools(
		self,
		provided_tools: list[dspy.Tool] | None,
		action_space_paths: list[str | Path] | None,
		early_stopping_enabled: bool,
		reasoning_intervention_tool_name: str = DEFAULT_REASONING_INTERVENTION_TOOL_NAME,
		finish_tool_description: str | None = None,
	) -> list[dspy.Tool]:
		"""
		Create the basic tools list for the controller.

		Creates tools from action_space_paths and/or provided_tools. If neither is provided,
		uses DEFAULT_TOOL. Optionally adds FINISH_TOOL if early stopping is enabled.

		Parameters:
			provided_tools: List of dspy.Tool instances provided by the user, or None.
			action_space_paths: Paths to action space JSON files, or None.
			early_stopping_enabled: Whether to include the early stopping tool.
			reasoning_intervention_tool_name: Name for the reasoning intervention tool.
			finish_tool_description: Optional custom description for the finish tool. If None,
				uses the default FINISH_TOOL description.

		Returns:
			List of dspy.Tool instances to use in the controller.
		"""
		tools: list[dspy.Tool] = []

		# Create combined tool from action_space_paths if provided
		# Treat empty lists the same as None: no action-space dimensions to inject.
		if action_space_paths:
			# For generative controller: create ONE combined tool with multiple parameters
			# This tool is meant to perform interventions on the next reasoning step of the
			# generator.
			combined_tool: Tool = self._load_action_spaces_and_create_combined_tool(
				action_space_paths,
				tool_name=reasoning_intervention_tool_name,
			)
			tools.append(combined_tool)
			logger.info(
				f"Created combined tool '{combined_tool.name}' with "
				f"{len(combined_tool.args)} dimension parameters from action space JSONs"
			)

		# Add provided tools
		if provided_tools is not None:
			for tool in provided_tools:
				if isinstance(tool, dspy.Tool):
					tools.append(tool)
				else:  # tool is a callable. Create a dspy.Tool from it.
					assert hasattr(tool, "__name__"), "Tool must have a name"
					assert hasattr(tool, "__doc__"), "Tool must have a description"
					assert get_type_hints(tool) is not None, "Tool must have annotations"
					tools.append(dspy.Tool(tool))

		# Fall back to DEFAULT_TOOL if no tools were provided
		if not tools:
			tools = [DEFAULT_TOOL]

		if early_stopping_enabled:
			if finish_tool_description is None:
				tools.append(FINISH_TOOL)
			else:
				tools.append(create_finish_tool(finish_tool_description))

		return tools

	def __init__(
		self,
		signature: type[ReasoningSignature],
		max_reasoning_steps: int,
		tools: list[Callable | Tool] | None = None,
		action_space_paths: list[str | Path] | None = None,
		forced_choice_function: ForcedChoiceFunction = return_action_if_single_option,
		early_stopping_enabled: bool = True,
		finish_tool_description: str | None = None,
		use_native_tool_calls: bool = False,
		verbosity: Literal["debug", "info", "warning", "error"] = "warning",
	) -> None:
		"""
		Initialize the TreeOfThoughtsController.

		Parameters:
			signature (dspy.Signature): The base signature for the reasoning task.
			max_reasoning_steps (int): The maximum number of reasoning steps allowed.
			tools (list[Callable | Tool] | None): A list of functions (callable objects), or
				`dspy.Tool` instances. Can be used alone or combined with action_space_paths.
				If both are None, uses DEFAULT_TOOL.
			action_space_paths (list[str | Path] | None): Paths to action space JSON files. Each JSON
				defines a dimension (e.g., structure, style, or subtopic) with choices that can be
				selected. Creates ONE combined tool with parameters for each dimension. Can be used
				alone or combined with tools.
			forced_choice_function (ForcedChoiceFunction): A function that takes available tools
				(dict[str, Tool]) and state, returning a list of (action_name, action_arguments,
				considerations) tuples or None if no forced choice. The considerations string
				explains why this action was chosen.
			early_stopping_enabled (bool): Whether to include the early stopping tool.
			finish_tool_description (str | None): Optional custom description for the finish tool.
				If None, uses the default FINISH_TOOL description.
			verbosity (Literal["debug", "info", "warning", "error"]): Verbosity level for logging.
		"""
		super().__init__()

		self.base_signature = ensure_reasoning_signature(signature)
		if not self.base_signature.reasoning_fields:
			raise ValueError("Generator signature must have at least one reasoning field")
		self.input_field_names = list(self.base_signature.input_fields.keys())
		self.output_field_names = list(self.base_signature.output_fields.keys())
		self.reasoning_field_name = list(self.base_signature.reasoning_fields.keys())[0]
		self.max_reasoning_steps = max_reasoning_steps
		self.forced_choice_function = forced_choice_function
		self.finish_tool_description = finish_tool_description
		self.use_native_tool_calls = use_native_tool_calls
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL.get(verbosity, logging.WARNING))

		#TODO[P2]: Add support for more complex forced choice functions that can return
		# interventions with internal_reasoning and prefix fields populated, not just
		# continue_reasoning.

		# Create basic tools from provided tools, action space JSONs, or use default
		tools_list: list[dspy.Tool] = self.create_basic_tools(
			provided_tools=tools,
			action_space_paths=action_space_paths,
			early_stopping_enabled=early_stopping_enabled,
			finish_tool_description=finish_tool_description,
		)

		# Create dict of tools by name
		tool_names = [tool.name for tool in tools_list]
		duplicates = {n for n in tool_names if tool_names.count(n) > 1}
		if duplicates:
			raise ValueError(f"Duplicate tool names detected: {duplicates}")
		self.tools = {tool.name: tool for tool in tools_list}

		# Check if any tool has arguments - if not, we don't need action_arguments in output
		self.tools_have_arguments = any(
			tool.args and len(tool.args) > 0 for tool in self.tools.values()
		)

		# Create single-candidate predictor
		self.decide_next_step_single = LocalPredict(
			signature=self.create_controller_signature(),
			verbose=verbosity,
		)

	@property
	def verbosity(self) -> Literal["debug", "info", "warning", "error"]:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Literal["debug", "info", "warning", "error"]) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL.get(verbosity, logging.WARNING))


	def _create_tool_instructions(self) -> list[str]:
		"""
		Create instructions for available tools.
		Following the DSPy ReACT module format:
			https://github.com/stanfordnlp/dspy/blob/103d3d7b336c58c3ab659d002b2a7b57766937c2/dspy/predict/react.py#L41
		"""
		return [
			_format_tool_instruction_block(tool, idx + 1)
			for idx, tool in enumerate(self.tools.values())
		]

	def create_controller_signature(self) -> dspy.Signature:
		"""
		Create signature for the controller to generate a candidate an action and considerations.
		"""
		inputs, outputs = parse_base_signature(
			input_field_names=self.input_field_names,
			output_field_names=self.output_field_names,
		)
		instructions = [self.base_signature.instructions]
		reasoning_state_field_name = ReasoningState.REASONING.value
		reasoning_step_field_name = self.reasoning_field_name
		tool_instructions = "\n".join(self._create_tool_instructions())
		tool_instructions_suffix = (
			"" if self.use_native_tool_calls
			else f"\nChoose a tool to use from the following options:\n{tool_instructions}\n"
		)
		instructions.extend(
			[
				f"""
You are given {inputs} and your goal is to finish with {outputs}.
To accomplish this goal, you will need to reason about the problem step by step rather than generating {outputs} directly.
You have up to `number_of_additional_reasoning_steps` additional steps to reason about the problem before generating {outputs}.
Refer to the existing reasoning steps under the `{reasoning_state_field_name}` header to inform your next step.
Reasoning steps are ordered sequentially, and each one includes a `{reasoning_step_field_name}` header above the content of the step itself.{tool_instructions_suffix}
""",
			]
		)

		# Fields are stored as tuples: (field_type, field_info)
		signature_fields = {}
		for name, field in self.base_signature.input_fields.items():
			field_type = (
				field.annotation if hasattr(field, "annotation") and field.annotation else str
			)
			signature_fields[name] = (field_type, field)

		signature_fields[reasoning_state_field_name] = (
			str,
			dspy.InputField(
				desc=(
					f"The existing reasoning steps towards producing {outputs}. "
					f"Each step's content is under the `{reasoning_step_field_name}` header."
				)
			),
		)
		signature_fields["number_of_additional_reasoning_steps"] = (
			int,
			dspy.InputField(
				desc=(
					f"The maximum number of additional reasoning steps you can take before "
					f"you must produce {outputs}."
				)
			),
		)
		if self.use_native_tool_calls:
			# Tools are passed in as an input so the adapter can propagate them into vLLM.
			# NOTE: If we are not using tool calls, then the tools are presented within the
			# instructions of the signature (as opposed to an additional input field).
			signature_fields["tools"] = (
				list[dspy.Tool],
				dspy.InputField(desc="Available tools to influence the next reasoning step."),
			)
			signature_fields["tool_calls"] = (
				dspy.ToolCalls,	# A list of dspy.ToolCall objects
				dspy.OutputField(desc="The tool call to influence the next reasoning step."),
			)
		else:
			signature_fields["considerations"] = (
				str,
				dspy.OutputField(desc="The thought process behind choosing the next action.")
			)
			signature_fields["action"] = (
				str,
				dspy.OutputField(desc="The selected action (tool) to guide the next reasoning step.")
			)
			if self.tools_have_arguments:
				signature_fields["action_arguments"] = (
					dict[str, Any],
					dspy.OutputField(desc="The input arguments for the selected action (tool).")
				)

		return dspy.Signature(signature_fields, "\n".join(instructions))

	def _state_to_controller_input(self, state: State) -> dict[str, Any]:
		"""
		Convert the state to the input for the decision-making controller.

		Parameters:
		    state (State): The current state of the tree of thoughts.

		Returns:
		    dict[str, Any]: The input for the decision-making controller.
		"""
		input_fields = {**state.model_dump()["input"]}
		input_fields["reasoning"] = state.model_output_so_far()
		number_of_existing_reasoning_steps = len(state.reasoning)
		input_fields["number_of_additional_reasoning_steps"] = (
			self.max_reasoning_steps - number_of_existing_reasoning_steps
		)
		if self.use_native_tool_calls:
			input_fields["tools"] = list(self.tools.values())
		return input_fields

	def create_predictions_from_forced_choices(
		self,
		forced_results_list: list[tuple[str, dict[str, Any], str]],
	) -> list[ControllerPrediction]:
		"""Create predictions from forced choices.

		Args:
		    forced_results_list (list[tuple[str, dict[str, Any], str]]): List of tuples containing
				the action name (strings), arguments (dictionaries mapping argument names for the
				tool to their values), and considerations (strings representing the thought
				process leading to the choice of the action).

		Returns:
		    list[ControllerPrediction]: List of ControllerPrediction objects reflecting the chosen
				interventions to perform on generations from a given state.
		"""
		forced_preds = []
		for action_name, action_arguments, considerations in forced_results_list:
			assert action_name in self.tools, f"Tool '{action_name}' not in {self.tools.keys()}."
			tool = self.tools[action_name]
			intervention, error_message = execute_tool_safely(tool, action_arguments)
			forced_preds.append(
				ControllerPrediction(
					tool=tool,
					chosen_values=action_arguments,
					intervention=intervention,
					considerations=considerations,
					error=error_message,
				)
			)
		return forced_preds

	def create_controller_predictions(
		self,
		prediction: dspy.Prediction,
		expected_n: int,
	) -> list[ControllerPrediction]:
		"""
		Create controller predictions from a DSPy prediction.

		Args:
			prediction (dspy.Prediction): The prediction object produced by the underlying
				generative LLM. The prediction includes objects corresponding to the output
				fields of the controller's signature.
				NOTE: Some completions may have failed (parsing issues) or all may have failed
				(due to parsing/underlying model-generation issues).
			expected_n (int): The expected number of completions (corresponding to the number of
				interventions to produce from a given state). It is okay if some of the completions
				are failures (due to parsing/underlying model-generation issues) -- we have
				built-in handling for these failures.

		Returns:
			list[ControllerPrediction]: List of ControllerPrediction objects (which can also
			represent failures due to parsing/underlying model-generation issues).
		"""
		completions = prediction.completions

		# Extract error information from ExecutionError objects
		errors_raw = completions["error"]
		errors_raw = errors_raw if isinstance(errors_raw, list) else [errors_raw]
		errors: list[ExecutionError] = []
		for err in errors_raw:
			if err is None:							# No error -> empty ExecutionError
				errors.append(ExecutionError())
			elif isinstance(err, ExecutionError):	# Already an ExecutionError -> add to list
				errors.append(err)
			elif isinstance(err, dict):				# Dictionary -> convert to ExecutionError
				errors.append(ExecutionError(**err))
			else:
				raise TypeError(
					f"Expected each completion error to be ExecutionError|dict|None, got {type(err)}"
				)
		assert len(errors) == expected_n, (
			f"Expected {expected_n} controller completions for 'error', "
			f"but got {len(errors)}."
		)
		# We may synthesize parsing errors during controller-side validation (e.g., empty tool_calls).
		effective_errors = list(errors)

		if self.use_native_tool_calls:
			tool_calls_values = completions["tool_calls"]
			tool_calls_values = (
				tool_calls_values if isinstance(tool_calls_values, list) else [tool_calls_values]
			)
			assert len(tool_calls_values) == expected_n, (
				f"Expected {expected_n} controller completions for 'tool_calls', "
				f"but got {len(tool_calls_values)}."
			)
			action_values: list[str | None] = [None] * expected_n
			args_values: list[dict[str, Any] | None] = [None] * expected_n
			considerations_values: list[str] = ["N/A (native tool call)"] * expected_n
			for i in range(expected_n):
				if effective_errors[i].has_error():
					continue
				tc: dspy.ToolCalls = tool_calls_values[i]
				if tc is None or not getattr(tc, "tool_calls", None):
					effective_errors[i] = ExecutionError(
						error_type="parsing", error_message="Empty tool_calls",
					)
					continue
				tool_call = tc.tool_calls[0]
				action_values[i] = tool_call.name
				args_values[i] = tool_call.args
		else:
			# Check if action field exists (it won't if all completions failed)
			if "action" in completions:
				action_values = completions["action"]
				action_values = action_values if isinstance(action_values, list) else [action_values]
				assert len(action_values) == expected_n, (
					f"Expected {expected_n} completions for 'action', but got {len(action_values)}."
				)

				considerations_values = completions["considerations"]
				considerations_values = (
					considerations_values if isinstance(considerations_values, list) else [considerations_values]
				)
				assert len(considerations_values) == expected_n, (
					f"Expected {expected_n} controller completions for 'considerations', "
					f"but got {len(considerations_values)}."
				)
				if self.tools_have_arguments:
					args_values = completions["action_arguments"]
					args_values = args_values if isinstance(args_values, list) else [args_values]
					assert len(args_values) == expected_n, (
						f"Expected {expected_n} controller completions for "
						f"'action_arguments', but got {len(args_values)}."
					)
				else:
					args_values = [{}] * expected_n
			else:
				# All completions failed - use None values
				action_values = [None] * expected_n
				considerations_values = [""] * expected_n
				args_values = [{}] * expected_n

		# Deduplicate by (action, args) while counting occurrences, preserving first-seen order.
		seen: dict[tuple[str, str], dict[str, Any]] = {}
		for i in range(expected_n):
			err = effective_errors[i]
			raw_out = err.raw_output

			if err.has_error():
				action = "prune"
				args: dict[str, Any] = {
					"error_type": err.error_type,
					"error_message": err.error_message,
					"raw_output": err.raw_output,
				}
				considerations = f"Controller failure: {err.error_type!r}"
			else:
				action_raw = action_values[i]
				if not (action_raw and isinstance(action_raw, str) and action_raw.strip()):
					action = "prune"
					args = {
						"error_type": "parsing",
						"error_message": f"Unknown action: {action_raw!r}",
						"raw_output": raw_out,
					}
					considerations = "Controller failure: unknown action."
				else:
					action = parse_literal(action_raw).strip("\"'`")
					if action not in self.tools:
						action = "prune"
						args = {
							"error_type": "parsing",
							"error_message": f"Unknown action: {action_raw!r}",
							"raw_output": raw_out,
						}
						considerations = "Controller failure: unknown action."
					else:
						args_raw = args_values[i]
						args = {} if args_raw is None else safe_parse_dict(args_raw)
						considerations = str(considerations_values[i] or "")

			args_key = ""
			if args:
				# Stable key for dedup; tolerate non-JSONable values by falling back to str.
				try:
					args_key = str(sorted(args.items()))
				except Exception:
					args_key = str(args)
			key = (action, args_key)
			if key in seen:
				seen[key]["unique_action_response_count"] += 1
			else:
				seen[key] = {
					"action": action,
					"action_arguments": args,
					"considerations": considerations,
					"unique_action_response_count": 1,
				}

		controller_predictions: list[ControllerPrediction] = []
		for output in seen.values():
			action = output["action"]
			args = output["action_arguments"]
			rationale = output["considerations"]
			count = output["unique_action_response_count"]

			if action == "prune":
				tool = PRUNE_TOOL
				err_type = args.get("error_type")
				err_msg = args.get("error_message")
				intervention = ReasoningIntervention(continue_reasoning=True)
				error = (
					err_msg
					if isinstance(err_msg, str) and err_msg
					else f"Controller failure: {err_type!r}"
				)
			else:
				tool = self.tools[action]
				intervention, error = execute_tool_safely(tool, args)

			controller_predictions.append(
				ControllerPrediction(
					tool=tool,
					chosen_values=args,
					intervention=intervention,
					considerations=rationale,
					error=error,
					num_occurrences=count,
				)
			)

		return controller_predictions

	def get_controller_predictions_from_forced_choices(
		self,
		state: State,
		n_samples_generation: int,
	) -> list[ControllerPrediction] | None:
		"""Get forced controller predictions for a given state if available.

		Forced predictions are either stored in the (parent) state before the controller is
		called or generated by the forced choice function (passed into the controller at
		initialization).

		Args:
		    state (State): The current state of the tree of thoughts.
		    n_samples_generation (int): The number of samples to generate.

		Returns:
		    list[ControllerPrediction] | None: The forced controller predictions for the given
				state.
		"""
		# TODO[P3]: Simplify this function, and perhaps use a common utility function for deduplication.
		forced_choices = (
			state.forced_controller_outputs or self.forced_choice_function(self.tools, state)
		)
		if not forced_choices:
			return None
		if len(forced_choices) == 1 and n_samples_generation > 1:
			forced_choices = forced_choices * n_samples_generation
		assert len(forced_choices) == n_samples_generation, (
			f"Forced choices must have length {n_samples_generation}, but got {len(forced_choices)}"
		)
		forced_preds = self.create_predictions_from_forced_choices(forced_choices)

		# Deduplicate by (tool.name, args) while counting occurrences, preserving first-seen order.
		seen: dict[tuple[str, str], dict[str, Any]] = {}
		for pred in forced_preds:
			args_key = ""
			if pred.chosen_values:
				try:
					args_key = str(sorted(pred.chosen_values.items()))
				except Exception:
					args_key = str(pred.chosen_values)
			key = (pred.tool.name, args_key)
			if key in seen:
				seen[key]["unique_action_response_count"] += 1
				continue
			seen[key] = {
				"tool": pred.tool,
				"chosen_values": pred.chosen_values,
				"intervention": pred.intervention,
				"considerations": pred.considerations,
				"error": pred.error,
				"unique_action_response_count": 1,
			}

		deduped: list[ControllerPrediction] = []
		for output in seen.values():
			deduped.append(
				ControllerPrediction(
					tool=output["tool"],
					chosen_values=output["chosen_values"],
					intervention=output["intervention"],
					considerations=output["considerations"],
					error=output["error"],
					num_occurrences=output["unique_action_response_count"],
				)
			)

		total = sum(p.num_occurrences for p in deduped)
		assert total == n_samples_generation, (
			f"Forced choices must sum to {n_samples_generation} occurrences, but got {total}"
		)
		return deduped

	def _generate_lm_predictions(
		self,
		lm_states_with_indices: list[tuple[int, State]],
		n_samples_generation: int,
		temperature: float,
		max_tokens: int,
		demos: list[dict[str, Any]] | None,
		**sampling_kwargs: Any,
	) -> dict[int, list[ControllerPrediction]]:
		"""
		Generate predictions using the language model for states that require it.

		Args:
			lm_states_with_indices: List of tuples (original_index, state).
			n_samples_generation: Number of actions to generate per state (duplicates allowed).
			temperature: Sampling temperature.
			max_tokens: Max tokens for generation.
			demos: Examples for the prompt.
			sampling_kwargs: Additional vLLM sampling parameters (vLLM-native names like top_p/top_k).

		Returns:
			dict[int, list[ControllerPrediction]]: Map of original index to predictions.
		"""
		# Extract states and build batch inputs
		lm_states = [state for _, state in lm_states_with_indices]
		batch_inputs = [self._state_to_controller_input(s) for s in lm_states]

		# Auto-batching kwargs
		input_keys = list(batch_inputs[0].keys())
		batched_kwargs = {k: [inp[k] for inp in batch_inputs] for k in input_keys}

		# vLLM sampling config is passed through LocalPredict (vLLM-native names like top_p/top_k).
		# We keep the internal keys for required params (n/temperature/max_tokens).
		config: dict[str, Any] = dict(sampling_kwargs)
		config.update(
			{
				"n": n_samples_generation,
				"temperature": temperature,
				"max_tokens": max_tokens,
			}
		)
		config["use_native_function_calling"] = self.use_native_tool_calls
		chat_template_kwargs = dict(config.get("chat_template_kwargs") or {})
		config["chat_template_kwargs"] = chat_template_kwargs
		predictions: list[dspy.Prediction] = self.decide_next_step_single(
			config=config,
			demos=demos,
			**batched_kwargs,
		)

		# Allow one-to-one or one-to-N matching; flexible for different predictor behaviors
		if len(predictions) != len(lm_states_with_indices):
			raise ValueError(f"Expected {len(lm_states_with_indices)} preds, got {len(predictions)}")

		results = {}
		for (orig_idx, _), pred in zip(lm_states_with_indices, predictions, strict=True):
			lm_preds: list[ControllerPrediction] = self.create_controller_predictions(
				prediction=pred,
				expected_n=n_samples_generation,
			)

			# Ensure exact count. We never fall back to default tools; instead, any controller
			# failures are represented as CONTROLLER_FAILURE predictions, which are converted
			# into pruned child nodes (skipped by generator/evaluator) in Tree-of-Thoughts.
			total = sum(p.num_occurrences for p in lm_preds)
			assert total == n_samples_generation, (
				f"Expected exactly {n_samples_generation} controller occurrences, but got {total}."
			)
			results[orig_idx] = lm_preds

		return results

	def forward(
		self,
		states: State | list[State],
		n_samples_generation: int = 1,
		temperature: float = 0.7,
		max_tokens: int = 2000,
		demos: list[dict[str, Any]] | None = None,
		**kwargs: Any,
	) -> list[list[ControllerPrediction]]:
		"""
		Forward method that automatically uses batch processing where applicable.

		Parameters:
		    states (Union[State, List[State]]): Single state or list of states to process.
		    n_samples_generation (int): Number of generations per state.
		    temperature (float): Temperature for generation.
		    max_tokens (int): Maximum tokens per generation.
		    demos (Optional[List[dict[str, Any]]]): List of demo inputs for the controller.
		    **kwargs: Additional vLLM sampling parameters (vLLM-native names like top_p/top_k/min_p/use_beam_search).

		Returns:
		    list[list[ControllerPrediction]]: Outer list has one entry per input state.
		        Inner list contains candidate actions (each leading to a distinct child node
		        via controlled generation). Each ControllerPrediction contains: tool,
		        chosen_values, intervention, considerations, tool_execution_error, and
		        num_occurrences.
		"""
		# TODO[P2]: Add support for multi-candidate generation.
		states = states if isinstance(states, list) else [states]

		# Separate states into forced choices and those needing LM calls
		forced_results = {}  			# dict[original_index -> result]
		lm_states_with_indices = []  	# list of (original_index, state) tuples

		# 1. Identify states with forced choices -- they need not be processed by an LLM.
		for i, state in enumerate(states):
			if forced := self.get_controller_predictions_from_forced_choices(
				state=state, n_samples_generation=n_samples_generation,
			):
				forced_results[i] = forced
			else:
				lm_states_with_indices.append((i, state))

		# 2. Generate LM predictions
		lm_results = {}
		if lm_states_with_indices:
			lm_results = self._generate_lm_predictions(
				lm_states_with_indices=lm_states_with_indices,
				n_samples_generation=n_samples_generation,
				temperature=temperature,
				max_tokens=max_tokens,
				demos=demos,
				**kwargs,
			)

		# 3. Merge results for LLM generations and forced choices
		return [lm_results.get(i, forced_results.get(i)) for i in range(len(states))]
