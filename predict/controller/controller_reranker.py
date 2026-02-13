"""
Reranker-based Controller: A DSPy module that determines the next action to take when solving a reasoning problem.
Uses a reranker model to score actions based on relevance to the current context, rather than generating actions.
"""

# Standard library imports
import logging
from itertools import product
from pathlib import Path
from typing import Any

# Third-party imports
import dspy

# Local imports
from adapter.vllm_scoring_adapter import LocalVLLMScoringAdapter
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.scoring_local_lm import RerankResponse, ScoringLocalVLLM
from predict.controller.controller_utils import (
	DEFAULT_TOOL,
	FINISH_TOOL,
	ActionSpaceConfig,
	ControllerPrediction,
	ForcedChoiceFunction,
	ReasoningIntervention,
	create_finish_tool,
	create_reasoning_intervention_from_choices,
	execute_tool_safely,
	load_action_space_json,
	return_action_if_single_option,
	sanitize_param_name,
)
from signatures import (
	ReasoningSignature,
	ensure_reasoning_signature,
)
from tree import State

logger = logging.getLogger(__name__)


def build_action_scoring_instructions(
	task_instructions: str,
	is_reasoning_empty: bool,
	num_actions_remaining: int,
) -> str:
	"""
	Build reranker instructions for scoring the next-step action in Tree-of-Thoughts control.

	The reranker is constrained by the scoring adapter's system message to output only "yes"/"no",
	so this function provides only quality-assessment guidance (not formatting instructions).

	Parameters:
		task_instructions: User task instructions (will be quoted in the prompt).
		is_reasoning_empty: Whether a non-empty reasoning trajectory exists in the query.
		num_actions_remaining: Number of actions remaining before a final answer is required.

	Returns:
		The multi-line instruction string used by the reranker controller.
	"""
	if num_actions_remaining <= 0:
		raise ValueError(
			"num_actions_remaining must be >= 1. If no actions remain, the controller should "
			"not be called."
		)
	base_task = task_instructions.strip()
	objective_line = (
		"Your objective is to decide what action to take for the next reasoning step for a "
		"user-assigned task."
	)
	lines: list[str] = [
		objective_line,
		"",
		"The user provided the following task:",
		f"\"{base_task}\"",
		"",
		"This task requires taking a sequence of reasoning steps to reach a solution.",
		"You must determine what action to take next.",
		"You will find the inputs for this task under the \"# Inputs\" header in the Query.",
	]
	if not is_reasoning_empty:
		lines.append(
			"You will find the intermediate reasoning trajectory towards solving this problem "
			"under the \"# Reasoning\" header in the Query."
		)
	lines.extend(
		[
			"You will find the action under consideration under the \"# Action\" heading in the "
			"Document.",
			"",
			"Judge whether the provided action is likely to be a good next step for addressing "
			"the user's task.",
			f"NOTE: You have {num_actions_remaining} "
			f"{'action' if num_actions_remaining == 1 else 'actions'} remaining before you must "
			"return a final answer.",
		]
	)
	return "\n".join(lines).strip()


class TreeOfThoughtsControllerReranker(dspy.Module):
	"""
	A reranker-based controller module for determining the next action in Tree of Thoughts.

	Instead of generating actions, this controller scores all available actions based on their
	relevance to the current context (reasoning trajectory), and selects the top-scoring actions
	using a reranker model.
	"""

	@staticmethod
	def create_tool(configs: list[ActionSpaceConfig], choices: dict[str, str]) -> dspy.Tool:
		"""
		Create a Tool for a specific combination of choices across dimensions.

		The tool takes no arguments and returns a ReasoningIntervention based on the
		pre-specified choices. The tool name is derived from the choice values.

		Parameters:
			configs: List of ActionSpaceConfig objects, one per dimension.
			choices: Dictionary mapping parameter names to their chosen values.

		Returns:
			A dspy.Tool instance representing this specific combination of choices.
		"""
		# Build dynamic docstring describing this specific combination of choices
		param_names = [sanitize_param_name(config.name) for config in configs]
		docstring_parts = ["Perform the following actions in the upcoming reasoning step:\n"]
		for config in configs:
			choice_spec: dict[str, Any] = config.choices[choices[sanitize_param_name(config.name)]]
			choice_def: str = choice_spec["definition"]
			docstring_parts.append(f"* {choice_def}\n")

		def tool_func() -> ReasoningIntervention:
			return create_reasoning_intervention_from_choices(configs, choices)

		return dspy.Tool(
			name="_".join(sanitize_param_name(choices[pn]) for pn in param_names),
			func=tool_func,
			desc="".join(docstring_parts).strip(),
		)

	@staticmethod
	def _load_action_spaces_and_create_tools(json_paths: list[str | Path]) -> list[dspy.Tool]:
		"""
		Load multiple action space JSONs and create one tool per combination of choices.

		For the reranker controller, we create MANY tools (one per unique combination
		of choices across all dimensions). Each tool takes no arguments and returns
		a ReasoningIntervention when executed.

		For example, if we have two action spaces with 10 and 20 choices respectively,
		this creates 200 tools (10 * 20 = 200).

		Parameters:
			json_paths: List of paths to action space JSON files.

		Returns:
			List of Tool instances, one per combination of choices.
		"""
		configs: list[ActionSpaceConfig] = [load_action_space_json(path) for path in json_paths]
		param_names = [sanitize_param_name(config.name) for config in configs]
		choice_lists: list[list[str]] = [list(config.choices.keys()) for config in configs]
		tools: list[dspy.Tool] = [
			TreeOfThoughtsControllerReranker.create_tool(
				configs, dict(zip(param_names, choices, strict=True)),
			)
			for choices in product(*choice_lists)
		]
		logger.debug(f"Created {len(tools)} tools from `{', '.join([c.name for c in configs])}`")
		return tools

	def create_basic_tools(
		self,
		action_space_paths: list[str | Path] | None,
		early_stopping_enabled: bool,
		finish_tool_description: str | None = None,
	) -> list[dspy.Tool]:
		"""
		Create the basic tools list for the reranker controller.

		Creates tools from action_space_paths if provided (one tool per combination of choices),
		otherwise uses DEFAULT_TOOL. Optionally adds FINISH_TOOL if early stopping is enabled.

		Note: The reranker controller only supports tools with no arguments (each tool represents
		a unique combination of choices). Custom tools with Literal parameters are not supported.

		Parameters:
			action_space_paths: Paths to action space JSON files, or None.
			early_stopping_enabled: Whether to include the early stopping tool.
			finish_tool_description: Optional custom description for the finish tool. If None,
				uses the default FINISH_TOOL description.

		Returns:
			List of Tool instances to use in the controller.
		"""
		tools: list[dspy.Tool] = []

		if action_space_paths is not None:
			# Create one tool per combination of choices across all dimensions.
			tools: list[dspy.Tool] = self._load_action_spaces_and_create_tools(action_space_paths)
			logger.info(
				f"Created {len(tools)} tools from `{', '.join(map(str, action_space_paths))}`"
			)
		else:
			tools = [DEFAULT_TOOL]

		if early_stopping_enabled:
			if finish_tool_description is None:
				tools.append(FINISH_TOOL)
			else:
				tools.append(create_finish_tool(finish_tool_description))

		return tools

	def _enumerate_all_action_candidates(self) -> None:
		"""
		Pre-enumerate all action candidates (tool descriptions) for scoring.

		Populates self.action_candidates (formatted descriptions) and self.action_metadata
		(tool names) for use during scoring. This is called once during __init__
		to avoid recomputing on every forward pass.

		Each tool represents a unique combination of choices across all dimensions,
		and takes no arguments.
		"""
		self.action_candidates: list[str] = []
		self.action_metadata: list[tuple[str, dict[str, Any]]] = []

		for tool_name, tool in self.tools.items():
			# Format action document from tool description
			action_doc = f"Action Name: {tool.name}\n\nDescription: {tool.desc}"
			self.action_candidates.append(action_doc)
			# Tools have no arguments, so pass empty dict
			self.action_metadata.append((tool_name, {}))

		logger.debug(
			f"Enumerated {len(self.action_candidates)} action candidates "
			f"from {len(self.tools)} tools"
		)

	def __init__(
		self,
		signature: type[ReasoningSignature],
		max_reasoning_steps: int,
		action_space_paths: list[str | Path] | None = None,
		forced_choice_function: ForcedChoiceFunction = return_action_if_single_option,
		early_stopping_enabled: bool = True,
		finish_tool_description: str | None = None,
		verbosity: Verbosity = Verbosity.WARNING,
	) -> None:
		"""
		Initialize the TreeOfThoughtsControllerReranker.

		Parameters:
			signature (ReasoningSignature): The base signature for the reasoning task.
			max_reasoning_steps (int): The maximum number of reasoning steps allowed.
			action_space_paths (list[str | Path] | None): Paths to action space JSON files. Each JSON
				defines a dimension (e.g., structure, style, subtopic) with choices that can be
				selected. Tools are created dynamically from these JSONs (one tool per combination
				of choices across dimensions). If None, uses DEFAULT_TOOL.
			forced_choice_function (ForcedChoiceFunction): A function that takes available tools
				(dict[str, Tool]) and state, returning a list of (action_name, action_arguments,
				considerations) tuples or None if no forced choice. The considerations string explains
				why this action was chosen given the state and tools.
			early_stopping_enabled (bool): Whether to include the early stopping tool.
			finish_tool_description (str | None): Optional custom description for the finish tool.
				If None, uses the default FINISH_TOOL description.
			verbosity (Verbosity): Verbosity level for logging (Verbosity enum).
		"""
		super().__init__()

		self.base_signature = ensure_reasoning_signature(signature)
		self.input_field_names = list(self.base_signature.input_fields.keys())
		self.output_field_names = list(self.base_signature.output_fields.keys())
		self.max_reasoning_steps = max_reasoning_steps
		self.forced_choice_function = forced_choice_function
		self.early_stopping_enabled = early_stopping_enabled
		self.finish_tool_description = finish_tool_description
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

		# Create basic tools
		tools_list: list[dspy.Tool] = self.create_basic_tools(
			action_space_paths=action_space_paths,
			early_stopping_enabled=early_stopping_enabled,
			finish_tool_description=finish_tool_description,
		)

		# Create dict of tools by name
		self.tools: dict[str, dspy.Tool] = {tool.name: tool for tool in tools_list}
		self.tools_have_arguments = any(tool.args and len(tool.args) > 0 for tool in self.tools.values())

		# Pre-enumerate all action candidates for scoring
		self._enumerate_all_action_candidates()

		# Initialize scoring adapter for controller action scoring
		self.scoring_adapter = LocalVLLMScoringAdapter(verbosity=verbosity)

		# Initialize LM attribute (set via set_lm method)
		self.lm: ScoringLocalVLLM | None = None

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level and propagate to scoring adapter."""
		self._verbosity = verbosity
		self.scoring_adapter.verbosity = verbosity

	def set_lm(self, lm: ScoringLocalVLLM) -> None:
		"""
		Set the language model for the reranker controller.

		Parameters:
		    lm (ScoringLocalVLLM): The language model to use for reranking.
		"""
		assert isinstance(lm, ScoringLocalVLLM), "lm must be a ScoringLocalVLLM instance"
		self.lm = lm

	def get_lm(self) -> ScoringLocalVLLM:
		"""
		Get the language model for the reranker controller.

		Returns:
		    ScoringLocalVLLM: The language model instance.

		Raises:
		    ValueError: If no language model has been set.
		"""
		if self.lm is None:
			raise ValueError("Language model has not been set. Call set_lm() first.")
		return self.lm

	def _score_actions(
		self,
		state: State,
		reranker_lm: ScoringLocalVLLM,
		demos: list[dict[str, Any]] | None = None,
	) -> list[tuple[str, dict[str, Any], float]]:
		"""
		Score all available action-argument combinations using the reranker.

		This method uses pre-computed action candidates (from __init__) and:
		1. Prepares input and reasoning for scoring
		2. Scores all candidates at once with a single scoring adapter call
		3. Returns the scored combinations sorted by relevance

		Parameters:
		    state (State): The current state containing input and reasoning.
		    reranker_lm (ScoringLocalVLLM): The language model to use for reranking.
		    demos (list[dict[str, Any]] | None): Optional demos for few-shot examples.

		Returns:
		    list[tuple[str, dict[str, Any], float]]: List of (action_name, arguments, score) tuples,
		        sorted by score descending.
		"""
		# TODO[P2]: Add demos to the scoring adapter (in the system message).
		# Use pre-computed action candidates and metadata from __init__
		action_candidates = self.action_candidates
		action_metadata = self.action_metadata

		# Prepare input and reasoning for scoring
		input_dict = dict(state.input)
		input_dict["number_of_additional_reasoning_steps"] = (
			self.max_reasoning_steps - len(state.reasoning)
		)
		# Use canonical state serialization for consistency with generator/evaluator formatting.
		reasoning_candidates = [state.model_output_so_far()]
		is_reasoning_empty = not reasoning_candidates[0].strip()
		scoring_instructions = build_action_scoring_instructions(
			task_instructions=self.base_signature.instructions,
			is_reasoning_empty=is_reasoning_empty,
			num_actions_remaining=self.max_reasoning_steps - len(state.reasoning),
		)

		# Step 3: Score ALL candidates in a single batch call
		# The adapter handles formatting and broadcasts the query across all action candidates.
		# This module owns instruction construction (adapter assumes signature.instructions is complete).
		num_candidates = len(action_candidates)

		# Raise error if no action candidates (e.g., empty tools dict)
		if num_candidates == 0:
			error_msg = (
				"No action candidates to score. This typically occurs when "
				"the tools dictionary is empty. At least one tool must be available."
			)
			logger.error(error_msg)
			raise ValueError(error_msg)

		logger.debug(
			f"Scoring {num_candidates} action candidates (from {len(self.tools)} tools)"
		)

		# TODO[P2]: Include demos in the prompt formatting via the scoring adapter.
		# Demos should be formatted and included in the system prompt (query) that
		# gets scored against action candidates. This requires updating:
		# 1. LocalVLLMScoringAdapter.__call__ to accept demos parameter
		# 2. LocalVLLMScoringAdapter.format_queries to include demos in query formatting
		# 3. The query formatting logic to incorporate few-shot examples similar to
		#    how VLLMGeneratorAdapter handles demos in _create_input_messages
		# 4. Adjusting tests to account for the new parameter and formatting changes.

		rerank_responses: list[RerankResponse] = self.scoring_adapter(
			instructions=scoring_instructions,
			lm=reranker_lm,
			inputs=input_dict,
			scoring_target="action",
			reasoning_candidates=reasoning_candidates,
			action_candidates=action_candidates,
		)

		# Step 4: Extract scores and combine with metadata
		# For constants.ACTION scoring, adapter may return:
		# - 1 response with M scores (one per candidate) - if the query represent the root node.
		# - N responses with M score each - if choosing an action from a non-root node.
		# We need to handle both cases
		scores = []
		assert rerank_responses and len(rerank_responses) > 0, "No rerank responses received"
		# Extract all scores from all responses
		for response in rerank_responses:
			for result in response.results:
				scores.append(result.relevance_score)
		assert len(scores) == num_candidates * len(rerank_responses), (
			f"Expected {num_candidates * len(rerank_responses)} scores but got {len(scores)}"
		)

		# Step 5: Validate and combine metadata with scores
		num_candidates = len(action_metadata)
		num_scores = len(scores)
		if num_scores != num_candidates:
			error_msg = (
				f"Mismatch between number of action candidates ({num_candidates}) "
				f"and number of scores returned ({num_scores}). "
				f"vLLM should return the same number of scores as candidates. "
				f"Tools: {list(self.tools.keys())}"
			)
			logger.error(error_msg)
			raise ValueError(error_msg)

		# Combine metadata with scores
		action_scores = [
			(tool_name, arguments, score)
			for (tool_name, arguments), score in zip(action_metadata, scores, strict=True)
		]

		# Step 6: Sort by relevance score (highest first)
		action_scores.sort(key=lambda x: x[2], reverse=True)

		return action_scores

	def _create_output_dicts_from_scores(
		self,
		action_scores: list[tuple[str, dict[str, Any], float]],
		n_samples: int = 1,
	) -> list[dict[str, Any]]:
		"""
		Create output dictionaries from scored action-argument combinations.

		Parameters:
		    action_scores (list[tuple[str, dict[str, Any], float]]): List of
		        (action_name, arguments, score) tuples.
		    n_samples (int): Number of top action-argument combinations to return.

		Returns:
		    list[dict[str, Any]]: List of action dictionaries.
		"""
		output_dictionaries = []

		# Take top n_samples action-argument combinations
		top_actions = action_scores[:n_samples]

		for idx, (action_name, action_arguments, score) in enumerate(top_actions):
			# Create considerations based on score
			rank = idx + 1
			considerations = f"""Selected action '{action_name}' based on relevance score: {score:.4f}.
This action is ranked #{rank} out of {len(action_scores)} available actions."""

			# Get tool description
			chosen_tool = self.tools[action_name]
			tool_descriptions = chosen_tool.desc

			output_dictionaries.append(
				{
					"action": action_name,
					"action_arguments": action_arguments,
					"tool_descriptions": tool_descriptions,
					"considerations": considerations,
					"score": score,
				}
			)

		return output_dictionaries

	def forward(
		self,
		states: State | list[State],
		n_samples_generation: int = 1,
		demos: list[dict[str, Any]] | None = None,
		**kwargs,
	) -> list[list[ControllerPrediction]]:
		"""
		Forward method that scores actions using the reranker and selects top-k actions.

		Parameters:
		    states (State | list[State]): Single state or list of states to process.
		    n_samples_generation (int): Number of top actions to return per state.
		    demos (list[dict[str, Any]] | None): Optional demos for few-shot examples.
		    **kwargs: Additional keyword arguments (e.g., 'lm', 'temperature', 'max_tokens',
			 	'sampling_kwargs'). Note that sampling parameters (temperature, max_tokens, and
				sampling kwargs) are ignored for reranking. If the language model is not provided
				as a keyword argument, it must be set via set_lm() before calling forward().

		Returns:
		    list[list[ControllerPrediction]]: Outer list has one entry per input state.
		        Inner list contains candidate actions (each leading to a distinct child node
		        via controlled generation). Each ControllerPrediction contains: tool,
		        chosen_values, intervention, considerations, tool_execution_error, and
		        num_occurrences.
		"""
		# Get LM from kwargs if provided, otherwise use self.lm
		# If provided via kwargs, store it in self.lm so it's included within the controller instance
		if "lm" in kwargs:
			reranker_lm = kwargs.pop("lm")
			assert isinstance(reranker_lm, ScoringLocalVLLM), "lm must be a ScoringLocalVLLM instance"
			assert reranker_lm.task == "score", "lm must be a ScoringLocalVLLM instance with task='score'"
			self.lm = reranker_lm
		elif self.lm is None:
			raise ValueError(
				"Language model has not been set. Either call set_lm() before calling forward(), "
				"or provide 'lm' as a keyword argument."
			)
		else:
			reranker_lm = self.lm
			assert isinstance(reranker_lm, ScoringLocalVLLM), "lm must be a ScoringLocalVLLM instance"
			assert reranker_lm.task == "score", "lm must be a ScoringLocalVLLM instance with task='score'"

		# Note: generative_lm is no longer needed - reranker scores action-argument combinations
		if "generative_lm" in kwargs:
			kwargs.pop("generative_lm")
			logger.warning(
				"The 'generative_lm' parameter is no longer needed for reranker controller. "
				"Use a 'reranker_lm' parameter instead (e.g., qwen3-reranker)."
			)

		states = states if isinstance(states, list) else [states]

		# Separate states into forced choices and those needing reranking
		forced_results = {}  # dict[original_index -> result]
		reranker_states_with_indices = []  # List of (original_index, state)

		# First pass: identify forced vs reranker states
		for i, state in enumerate(states):
			forced_result_list = (
				state.forced_controller_outputs
				if state.forced_controller_outputs is not None
				else self.forced_choice_function(self.tools, state)
			)
			if forced_result_list is not None:
				if len(forced_result_list) == 1 and n_samples_generation > 1:
					forced_result_list = forced_result_list * n_samples_generation
				else:
					assert len(forced_result_list) == n_samples_generation, (
						f"Expected {n_samples_generation} forced choices but got {len(forced_result_list)}"
					)
				# forced_result is a list of (action_name, action_arguments, considerations)
				forced_preds = [
					{
						"action": action_name,
						"action_arguments": action_arguments,
						"considerations": considerations,
						"unique_action_response_count": 1,
						"tool_descriptions": self.tools[action_name].desc,
						# Assign the highest score to forced choices
						"score": 1.0,
					}
					for action_name, action_arguments, considerations in forced_result_list
				]
				forced_results[i] = forced_preds
			else:
				reranker_states_with_indices.append((i, state))

		# Second pass: process all reranker states
		reranker_results = {}  # dict[original_index -> result]

		if reranker_states_with_indices:
			for original_idx, state in reranker_states_with_indices:
				# Score all action-argument combinations using the adapter
				action_scores = self._score_actions(state, reranker_lm=reranker_lm, demos=demos)

				# Create output dictionaries from top-scoring action-argument combinations
				output_dictionaries = self._create_output_dicts_from_scores(
					action_scores,
					n_samples=n_samples_generation,
				)
				reranker_results[original_idx] = output_dictionaries

		# Merge results in original order
		final_controller_outputs = []
		for i in range(len(states)):
			if i in forced_results:
				final_controller_outputs.append(forced_results[i])
			elif i in reranker_results:
				final_controller_outputs.append(reranker_results[i])
			else:
				# This should never happen if logic is correct
				raise RuntimeError(f"No result found for state at index {i}")

		# Convert output dictionaries to ControllerPredictions
		predictions: list[list[ControllerPrediction]] = []
		for output_dicts in final_controller_outputs:
			state_predictions: list[ControllerPrediction] = []
			for output_dict in output_dicts:
				tool_name = output_dict["action"]
				tool = self.tools[tool_name]
				intervention, tool_execution_error = execute_tool_safely(
					tool, output_dict["action_arguments"]
				)
				num_occurrences = 1
				if "unique_action_response_count" in output_dict:
					num_occurrences = output_dict["unique_action_response_count"]
				score = score = output_dict["score"] if "score" in output_dict else None
				state_predictions.append(
					ControllerPrediction(
						tool=tool,
						chosen_values=output_dict["action_arguments"],
						intervention=intervention,
						considerations=output_dict["considerations"],
						error=tool_execution_error,
						num_occurrences=num_occurrences,
						score=score,
					)
				)
			predictions.append(state_predictions)
		return predictions
