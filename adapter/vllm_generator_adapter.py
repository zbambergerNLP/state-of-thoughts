"""An adapter for generating responses over the course of multiple steps (LM calls) using a local vLLM model with improved parameter extraction."""

# Standard library imports
import copy
import json
import logging
import re
from typing import Any, cast

# Third-party imports
from dspy.adapters.utils import format_field_value, parse_value
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError
from pydantic.fields import FieldInfo
from vllm import SamplingParams

# Local imports
from adapter.adapter_constants import (
	FINAL_OUTPUT_KIND_TYPE,
	GENERATOR_ADAPTER_NAME,
)
from adapter.constraints import (
	ResponseLength,
	format_response_length_instruction,
	format_thought_length_instruction,
)
from adapter.prompts import (
	GENERATOR_SYSTEM_PROMPT_INTERNAL_REASONING,
	GENERATOR_SYSTEM_PROMPT_VANILLA,
)
from adapter.utils import (
	format_field_description,
	generate_output_field_sections,
	get_final_output_description,
	uncapitalize_first_letter,
)
from adapter.vllm_adapter import FIELD_HEADER_PATTERN, LocalVLLMAdapter
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.generative_local_lm import (
	ChatCompletionResponse,
	GenerativeLocalVLLM,
)
from misc_utils import (
	ExecutionError,
	is_list_of_lists,
	parse_base_signature,
	parse_reasoning_signature,
)
from signatures import ReasoningSignature
from tree.tree_constants import ReasoningState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



class VLLMGeneratorAdapter(LocalVLLMAdapter):
	"""
	An improved version of VLLMGeneratorAdapter that properly extracts reasoning field
	information from signatures rather than through explicit parameters.
	"""

	def _flatten_lm_kwargs_for_messages(
		self,
		continue_reasoning: list[list[bool]],
		lm_kwargs: dict[str, Any] | list[dict[str, Any]] | list[list[dict[str, Any]]],
		num_messages: int,
	) -> list[dict[str, Any]]:
		"""Flatten lm_kwargs into one dict per message, aligned with continue_reasoning.

		Args:
			continue_reasoning: List of lists of booleans. Outer list is per trajectory,
				inner list is per intervention. True to continue reasoning, False to generate final
				answer.
			lm_kwargs: The keyword arguments to pass to the language model. For example:
				- A single dict (broadcast to all messages): {"temperature": 0.7, "n": 5}
				- A list of dicts (on per trajectory):
					`[{"temperature": 0.7, "n": 5}, {"temperature": 0.7, "n": 3}, ...]`
				- A list of lists of dicts (outer list per trajectory, inner list per intervention):
					`[
						[{"temperature": 0.7, "n": 5}, {"temperature": 0.7, "n": 3}],
						[{"temperature": 0.8, "n": 2}],
						...
					]`
			num_messages: The number of messages to generate.

		Returns:
			A list of dictionaries, one per message.
		"""
		if is_list_of_lists(lm_kwargs, dict):
			assert len(lm_kwargs) == len(continue_reasoning) and all(
				len(inner) == len(continue_reasoning[i])
				for i, inner in enumerate(lm_kwargs)
			), (
				"If lm_kwargs is list[list[dict]], it must contain exactly one dict per "
				f"intervention per trajectory, got {type(lm_kwargs)}:\n{lm_kwargs}"
			)
			return [kwargs for trajectory_kwargs in lm_kwargs for kwargs in trajectory_kwargs]

		if isinstance(lm_kwargs, list):
			assert len(lm_kwargs) == len(continue_reasoning), (
				"If lm_kwargs is list[dict], it must contain exactly one dict per "
				f"trajectory, got {type(lm_kwargs)}:\n{lm_kwargs}"
			)
			return [
				lm_kwargs[trajectory_idx]
				for trajectory_idx, trajectory in enumerate(continue_reasoning)
				for _ in trajectory
			]

		assert isinstance(lm_kwargs, dict), (
			"If lm_kwargs is neither list[dict] nor list[list[dict]], it must be exactly "
			f"one dict, got {type(lm_kwargs)}:\n{lm_kwargs}"
		)
		return [lm_kwargs] * num_messages

	def _build_sampling_params_and_remaining_kwargs(
		self,
		lm_kwargs_list: list[dict[str, Any]],
	) -> tuple[list[SamplingParams], list[dict[str, Any]]]:
		"""
		Build SamplingParams per message, injecting stop tokens per message.

		Stop tokens are determined from `self._trajectory_continue_reasoning`, which
		is populated during `format()`.

		Args:
			lm_kwargs_list: List of dictionaries, one per message. Each dictionary contains the
			keyword arguments to pass to the language model for that message (e.g., temperature,
			n, max_tokens, etc.).

		Returns:
			A tuple containing:
				- List of SamplingParams, one per message.
				- List of dictionaries, one per message.
		"""
		sampling_fields = set(SamplingParams.__annotations__)
		sampling_params_list: list[SamplingParams] = []
		remaining_kwargs_list: list[dict[str, Any]] = []

		for message_idx, kwargs in enumerate(lm_kwargs_list):
			current_reasoning = self._trajectory_continue_reasoning[message_idx]
			stop_tokens = self._determine_stop_tokens(current_reasoning)

			sampling_kwargs = {k: v for k, v in kwargs.items() if k in sampling_fields}
			remaining_kwargs = {k: v for k, v in kwargs.items() if k not in sampling_fields}

			# Match LocalVLLMAdapter.get_sampling_params default behavior.
			sampling_kwargs.setdefault("include_stop_str_in_output", True)
			# Always override stop tokens per-message based on continue_reasoning.
			sampling_kwargs["stop"] = stop_tokens

			sampling_params_list.append(SamplingParams(**sampling_kwargs))
			remaining_kwargs_list.append(remaining_kwargs)

		return sampling_params_list, remaining_kwargs_list

	def _call_lm_with_fallback(
		self,
		signature: type[ReasoningSignature],
		lm: GenerativeLocalVLLM,
		new_reasoning_trajectories: list[list[dict[str, Any]]],
		sampling_params_list: list[SamplingParams],
		extra_kwargs: dict[str, Any],
		continue_final_message: bool,
	) -> list[list[dict[str, Any]]]:
		"""Call the LM with batch processing and return parsed results.

		Args:
			signature: The DSPy signature for which to generate completions.
			lm: The GenerativeLocalVLLM instance to use for generating responses.
			new_reasoning_trajectories: A list of lists of dictionaries (reflecting different
				trajectories of messages, potentially with different interventions).
			sampling_params_list: A list of SamplingParams, one per trajectory.
			extra_kwargs: A dictionary of extra keyword arguments to pass to the language model.
			continue_final_message: A boolean indicating whether to continue the final (assistant)
				message.

		Returns:
			A list of lists of dictionaries, one per completion (intervention on an inputted
			trajectory). Each dictionary contains the parsed completion payload for that completion.
		"""
		batch_size = len(new_reasoning_trajectories)
		assert len(sampling_params_list) == batch_size, (
			f"sampling_params_list must have the same length as new_reasoning_trajectories, "
			f"got {len(sampling_params_list)} vs {batch_size}"
		)

		# Always preserve positional correspondence with the interventions/messages by index.
		responses: list[ChatCompletionResponse] = lm.batch(
			messages=new_reasoning_trajectories,
			sampling_params=sampling_params_list,
			continue_final_message=continue_final_message,
			**extra_kwargs,
		)
		assert len(responses) == batch_size, (
			"lm.batch must return exactly one response (each with a list of Choice objects) "
			f"per conversation. Got {len(responses)} responses, expected {batch_size}."
		)

		return self._postprocess_responses(
			signature=signature,
			responses=responses,
		)

	def _postprocess_responses(
		self,
		signature: type[ReasoningSignature],
		responses: list[ChatCompletionResponse],
	) -> list[list[dict[str, Any]]]:
		"""Convert responses into the adapter's parsed return structure.

		This method processes ChatCompletionResponse objects and maps their Choice objects
		to parsed dictionaries. Parser failures and precondition violations are mapped to
		ExecutionError objects with error_type="parsing".

		Args:
			signature: The DSPy signature for which to generate completions.
			responses: A list of ChatCompletionResponse objects, one per message.

		Returns:
			A list of lists of dictionaries, one per message. Each inner list contains
			parsed completions (or error dicts) for one message.
		"""
		results: list[list[dict[str, Any]]] = []

		for message_idx, response in enumerate(responses):
			completions: list[dict[str, Any]] = []
			prefill = (
				self._current_assistant_prefills[message_idx]
				if message_idx < len(self._current_assistant_prefills)
				else ""
			)
			parse_reasoning = self._trajectory_continue_reasoning[message_idx]

			for choice in response.choices:
				# Attempt to parse the completion
				# Parser failures and precondition violations will be caught as AdapterParseError
				# Use newline between prefill and choice so headers in choice are not concatenated
				sep = "\n" if prefill and choice.text else ""
				full_text = (prefill + sep + (choice.text or "")) if prefill else (choice.text or "")
				try:
					value: dict[str, Any] = self.parse(
						signature=signature,
						completion=full_text,
						parse_reasoning=parse_reasoning,
					)
				except AdapterParseError as e:
					value = {
						"error": ExecutionError(
							error_type="parsing",
							raw_output=full_text,
							error_message=str(e),
						)
					}

				if choice.logprobs:
					value["logprobs"] = choice.logprobs
				completions.append(value)

			results.append(completions)

		return results

	def __init__(
		self,
		callbacks: list[BaseCallback] | None = None,
		thinking_start_tag: str = "<thinking>",
		thinking_end_tag: str = "</thinking>",
		reasoning_step_start_tag: str = "<step>",
		reasoning_step_end_tag: str = "</step>",
		answer_start_tag: str = "<answer>",
		answer_end_tag: str = "</answer>",
		verbosity: Verbosity = Verbosity.INFO,
	):
		"""
		Initializes an adapter that calls a local language model multiple times to generate a response.

		Args:
			callbacks: Optional list of callbacks to use for this adapter.
			thinking_start_tag: The tag to use for the start of the thinking section.
			thinking_end_tag: The tag to use for the end of the thinking section.
			reasoning_step_start_tag: The tag to use for the start of each reasoning step.
			reasoning_step_end_tag: The tag to use for the end of each reasoning step.
			answer_start_tag: The tag to use for the start of the answer section.
			answer_end_tag: The tag to use for the end of the answer section.
			verbosity: Verbosity level for logging. Defaults to INFO.
		"""
		super().__init__(callbacks=callbacks)
		self.thinking_start_tag = thinking_start_tag
		self.thinking_end_tag = thinking_end_tag
		self.reasoning_step_start_tag = reasoning_step_start_tag
		self.reasoning_step_end_tag = reasoning_step_end_tag
		self.answer_start_tag = answer_start_tag
		self.answer_end_tag = answer_end_tag
		self._verbosity: Verbosity = verbosity

		# Context variables to track parsing mode
		self._trajectory_continue_reasoning: dict[int, bool] = {}
		self._current_assistant_prefills: list[str] = []  # Track prefills for parsing

	@property
	def verbosity(self) -> Verbosity:
		"""Get the current verbosity level."""
		return self._verbosity

	@verbosity.setter
	def verbosity(self, verbosity: Verbosity) -> None:
		"""Set the verbosity level and update logger."""
		self._verbosity = verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

	def __call__(
		self,
		signature: type[ReasoningSignature],
		lm: GenerativeLocalVLLM,
		inputs: dict[str, Any] | list[dict[str, Any]],
		continue_reasoning: list[list[bool]],
		lm_kwargs: dict[str, Any] | list[dict[str, Any]] | list[list[dict[str, Any]]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
		response_length: ResponseLength | None = None,
		thought_length: ResponseLength | None = None,
		previous_content: list[str] | None = None,
		internal_reasoning_for_output: list[list[str]] | None = None,
		prefix_for_output: list[list[str]] | None = None,
		final_output_kind: FINAL_OUTPUT_KIND_TYPE = "synthesis_faithful",
	) -> list[list[dict[str, Any]]]:
		"""
		Generate completions for the provided input using the local language model.

		Args:
			signature: The DSPy signature for which to generate completions.
			lm: The generative local language model to use for generation.
			inputs: The inputs to use for generation and reasoning trajectories (that each include
			 	these input fields along with partial reasoning towards the answer).
			continue_reasoning: List of lists of booleans. Outer list is per trajectory,
				inner list is per intervention. True to continue reasoning, False to generate final
				answer.
			lm_kwargs: The keyword arguments to pass to the language model. For example:
				- A dict (broadcast to all messages): {"temperature": 0.7, "n": 5}
				- List of dicts (on per trajectory):
					[{"temperature": 0.7, "n": 5}, {"temperature": 0.7, "n": 3}, ...]
				- List of lists of dicts (outer list per trajectory, inner list per intervention):
					[
						[
							{"temperature": 0.7, "n": 5},
							{"temperature": 0.7, "n": 3},
						],
						[
							{"temperature": 0.8, "n": 2},
						],
						...
					]
					This structure matches prefix_for_output, continue_reasoning, and
					internal_reasoning_for_output.
			demos: The demos to use for generation.
			response_length: The maximum length of the response.
			thought_length: The maximum length of the thought.
			previous_content: List of previous reasoning content strings, one per state.
				None if no previous content.
			internal_reasoning_for_output: List of lists of internal reasoning strings.
				Outer list is per trajectory (state/input), and inner list is per intervention.
				None if no internal reasoning.
			prefix_for_output: List of lists of prefix strings. Outer list is per trajectory,
				inner list is per intervention. None if no prefixes.
			final_output_kind: Kind of final output instruction (synthesis_strict,
				synthesis_faithful, synthesis_restructured, conclusion).
				Defaults to synthesis_faithful.

		Returns:
			A list of lists of dictionaries, where each inner list contains completions for one message.
		"""
		demos = [] if demos is None else demos
		# Set logger level based on verbosity
		logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[self.verbosity])

		# Format new_reasoning_trajectories using the existing format method
		new_reasoning_trajectories: list[list[dict[str, Any]]] = self.format(
			signature=signature,
			demos=demos,
			inputs=inputs,
			continue_reasoning=continue_reasoning,
			previous_content=previous_content,
			internal_reasoning_for_output=internal_reasoning_for_output,
			prefix_for_output=prefix_for_output,
			thought_length=thought_length,
			response_length=response_length,
			final_output_kind=final_output_kind,
		)
		num_messages = len(new_reasoning_trajectories)

		# Determine whether to use `continue_final_message` in `chat` call to vLLM model.
		final_trajectory_message_role = new_reasoning_trajectories[0][-1]["role"]
		assert all(
			trajectory[-1]["role"] == final_trajectory_message_role
			for trajectory in new_reasoning_trajectories
		), (
			f"All trajectories should have final messages with the same role.\n"
			f"Got the following roles for final messages: {[trajectory[-1]['role'] for trajectory in new_reasoning_trajectories]}"
		)
		continue_final_message = final_trajectory_message_role == "assistant"

		# Flatten list[list[dict]] to list[dict] (one dict per message)
		lm_kwargs_list = self._flatten_lm_kwargs_for_messages(
			continue_reasoning=continue_reasoning,
			lm_kwargs=lm_kwargs,
			num_messages=num_messages,
		)

		# Validate flattened lm_kwargs_list matches number of messages
		assert len(lm_kwargs_list) == num_messages, (
			f"lm_kwargs flattened length ({len(lm_kwargs_list)}) must match number of messages ({num_messages})"
		)

		sampling_params_list, remaining_kwargs_list = self._build_sampling_params_and_remaining_kwargs(
			lm_kwargs_list=lm_kwargs_list
		)

		# Log formatted messages in DEBUG mode
		# TODO[P3]: Add support for debugging messages like this via explicit verbosity configurations.
		# logger.debug(
		# 	f"\n{'='*80}\nFORMATTED MESSAGES ({len(new_reasoning_trajectories)} total):\n{'='*80}"
		# )
		# for i, trajectory in enumerate(new_reasoning_trajectories, 1):
		# 	logger.debug(f"\n--- Message {i}/{len(new_reasoning_trajectories)} ---")
		# 	for msg in trajectory:
		# 		logger.debug(
		# 			f"\n[{msg['role'].upper()}]:\n{ msg['content']}\n"
		# 		)
		# logger.debug(f"{'='*80}\n")

		extra_kwargs = {} if not remaining_kwargs_list else remaining_kwargs_list[0]
		return self._call_lm_with_fallback(
			lm=lm,
			signature=signature,
			new_reasoning_trajectories=new_reasoning_trajectories,
			sampling_params_list=sampling_params_list,
			extra_kwargs=extra_kwargs,
			continue_final_message=continue_final_message,
		)

	def _call_post_process(
		self,
		outputs: list[list[str | dict[str, Any]]],
		signature: type[ReasoningSignature],
	) -> list[list[dict[str, Any]]]:
		"""
		Override parent method to handle reasoning vs output field parsing.

		Parse each completion attempt for one or more outputs, determining whether
		to parse reasoning fields or output fields based on the continue_reasoning context.

		Args:
		    outputs: The raw outputs from the language model. Each inner list contains
		        completions for one input. Items can be strings (no logprobs) or dicts
		        with "text" and "logprobs" keys.
			signature: The DSPy signature for which to parse the completions.

		Returns:
		    A list of lists of dictionaries, where each inner list contains parsed completions for one message.
		"""
		result = []
		for example_idx, example in enumerate(outputs):
			values = []
			# Get the prefill for this trajectory (if any)
			prefill = (
				self._current_assistant_prefills[example_idx]
				if example_idx < len(self._current_assistant_prefills)
				else ""
			)
			for candidate in example:
				# Handle both string and dict formats from _process_lm_response
				if isinstance(candidate, dict):
					candidate_text = candidate["text"]
					candidate_logprobs = candidate.get("logprobs")
				else:
					candidate_text = candidate
					candidate_logprobs = None
				# Prepend the prefill to the generated text for parsing
				# This ensures headers like "## reasoning" are present for the parser.
				# Use newline between prefill and candidate so headers are not concatenated.
				sep = "\n" if prefill and candidate_text else ""
				full_text = (prefill + sep + candidate_text) if prefill else candidate_text

				# Get parse mode from trajectory mapping (default to True for reasoning)
				parse_reasoning = self._trajectory_continue_reasoning[example_idx]
				try:
					value: dict[str, Any] = self.parse(
						signature, full_text, parse_reasoning=parse_reasoning
					)
				except AdapterParseError as e:
					logger.warning(
						f"Failed to parse generation: {e}\n"
						f"Prefill was {prefill}\n"
						f"Generated text was {candidate_text}\n"
					)
					value = {
						"error": ExecutionError(
							error_type="parsing",
							raw_output=full_text,
							error_message=str(e),
						)
					}

				if candidate_logprobs is not None:
					value["logprobs"] = candidate_logprobs
				values.append(value)
			result.append(values)
		return result

	def parse(
		self,
		signature: type[ReasoningSignature],
		completion: str,
		parse_reasoning: bool = True,
	) -> dict[str, Any]:
		"""
		Custom parse method that handles both reasoning steps and final answers.

		Args:
		    signature: The DSPy signature for which to parse the completion.
		    completion: The completion to parse.
		    parse_reasoning: If True, parse reasoning fields; if False, parse output fields.

		Returns:
		    A dictionary containing the parsed fields (either reasoning or output fields).
		"""
		if parse_reasoning:
			# Parse only reasoning fields defined in the signature
			target_fields: dict[str, FieldInfo] = signature.reasoning_fields.copy()
			field_type = ReasoningState.REASONING
		else:
			# Parse only output fields defined in the signature
			target_fields: dict[str, FieldInfo] = signature.output_fields.copy()
			field_type = ReasoningState.OUTPUT

		# Try JSON parsing first (for guided JSON generation)
		completion_stripped = completion.strip()
		if completion_stripped.startswith("{") and completion_stripped.endswith("}"):
			try:
				json_response: dict[str, Any] = json.loads(completion_stripped)
				# Determine required (non-optional) fields we must have
				required_fields = set(
					signature.reasoning_fields.keys()
					if parse_reasoning
					else signature.output_fields.keys()
				)
				if set(json_response.keys()).issuperset(required_fields):
					return json_response
			except (json.JSONDecodeError, KeyError, TypeError):
				# Fall through to original parsing if JSON parsing fails
				pass

		# Check for single-line XML tag format (e.g., <answer>Paris</answer>)
		# This handles cases where the LM generates a single field in XML format
		xml_pattern = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
		xml_match = xml_pattern.search(completion)
		if xml_match and len(target_fields) == 1:
			# Single field in XML format
			field_name = xml_match.group(1)
			field_value = xml_match.group(2).strip()
			if field_name in target_fields:
				annotation = target_fields[field_name].annotation
				return {field_name: parse_value(field_value, annotation)}

		# Parse using field headers
		sections: list[tuple[str | None, list[str]]] = [(None, [])]
		for line in completion.splitlines():
			# Use search() instead of match() to find headers anywhere in the line
			# This handles cases where models put tags and headers on the same line
			match = FIELD_HEADER_PATTERN.search(line.strip())
			if match:
				header = match.group(1)
				remaining_content = line[match.end() :].strip()
				sections.append((header, [remaining_content] if remaining_content else []))
			else:
				sections[-1][1].append(line)

		# Process sections and remove XML tags from content
		processed_sections = []
		for k, v in sections:
			content = "\n".join(v).strip()
			# Remove XML tags (like </step>, </thinking>, etc.) from the content
			# This is needed when parsing continued messages where LM response includes closing tags
			content = re.sub(r"</?(?:step|thinking|answer)>", "", content).strip()
			processed_sections.append((k, content))

		fields: dict[str, Any] = {}
		for k, v in processed_sections:
			# Note: We don't check (k not in fields) because we want to keep the LAST occurrence
			# of each field, not the first. This is important for continued reasoning where
			# the completion contains multiple steps with the same field names.
			if k is not None and (k in target_fields.keys()):
				try:
					annotation = target_fields[k].annotation
					parsed_value = parse_value(v.strip(), annotation)

					# Skip empty string values; a later occurrence may have content
					if isinstance(parsed_value, str) and not parsed_value.strip():
						continue
					fields[k] = parsed_value
				except Exception as e:
					raise AdapterParseError(
						adapter_name=GENERATOR_ADAPTER_NAME,
						signature=signature,
						lm_response=completion,
						message=f"Failed to parse {field_type} field {k} with value {v} from the LM response. Error message: {e}",
					) from e

		# Check if we got the required fields
		required_fields = set(
			signature.reasoning_fields.keys()
			if parse_reasoning
			else signature.output_fields.keys()
		)
		found_fields = set(fields.keys())

		# Check that we have all the required fields
		missing_required = required_fields - found_fields
		if missing_required:
			# Debug: Show what we actually parsed
			debug_info = f"\nDEBUG - Parsed sections: {processed_sections}\nDEBUG - Found fields: {fields}\nDEBUG - Target fields: {list(target_fields.keys())}"
			raise AdapterParseError(
				adapter_name=GENERATOR_ADAPTER_NAME,
				signature=signature,
				lm_response=completion,
				parsed_result=fields,
				message=f"Expected to find {field_type} fields in the LM response: {list(required_fields)}, "
				f"but missing: {list(missing_required)}{debug_info}",
			)
		return fields

	def format(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any] | list[dict[str, Any]],
		continue_reasoning: list[list[bool]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
		response_length: ResponseLength | None = None,
		thought_length: ResponseLength | None = None,
		previous_content: list[str] | None = None,
		internal_reasoning_for_output: list[list[str]] | None = None,
		prefix_for_output: list[list[str]] | None = None,
		final_output_kind: FINAL_OUTPUT_KIND_TYPE = "synthesis_faithful",
	) -> list[list[dict[str, Any]]]:
		"""
		Formats one or more inputs for the local language model.

		- If `inputs` is a single dictionary, and there are no interventions, then it
			formats a single request (input example), using the provided inputs and demos.
		- If `inputs` is a single dictionary, but there is a list of interventions, then
			it formats multiple requests (input examples) for the same initial input, each
		    with a slight modification (intervention) in the form of internal reasoning or
		    prefix for the model's output.
		- If `inputs` is a list of dictionaries, and there are no interventions, then it formats
			multiple requests (input examples) for each input in the list.
		- If `inputs` is a list of dictionaries and there is a list of interventions, then each
		    input is formatted with its corresponding intervention (i.e., input `i` is formatted
		    with intervention `i`).
		- If `inputs` is a list of dictionaries and there is a list of lists of interventions,
		    then each input is formatted with multiple interventions, where the `i`-th input is formatted
		    with each of the `i`-th interventions in the list of lists of interventions.

		Args:
		    signature: The DSPy signature defining the task's inputs, reasoning steps, outputs, and instructions.
		    inputs: A dictionary or list of dictionaries containing the input data for the current task.
		    continue_reasoning: List of lists of booleans. Outer list is per trajectory,
		        inner list is per intervention. True to continue reasoning, False to generate final answer.
			demos: Either a list of in-context examples to include in the request, or a list of lists of in-context
		        examples. In the latter case, each internal list corresponds to in-context examples for a specific input.
		    response_length: Optional ResponseLength object specifying constraints on the length of responses.
		    thought_length: Optional ResponseLength object specifying constraints on the length of thoughts.
		    previous_content: List of previous content strings, one per trajectory. None if no previous content.
		    internal_reasoning_for_output: List of lists of internal reasoning strings. Outer list is per trajectory,
		        inner list is per intervention. None if no internal reasoning.
		    prefix_for_output: List of lists of prefix strings. Outer list is per trajectory,
		        inner list is per intervention. None if no prefixes.
		    final_output_kind: Kind of final output instruction (SYNTHESIS or CONCLUSION). Defaults to SYNTHESIS.

		Returns:
			A list of lists of messages formatted for the vLLM generator adapter. Each inner list corresponds to a
		    specific input example (or a modified version of it).
		"""
		# TODO[P2]: Support multiple inputs (list[dict[str, Any]]) in the future. Note that this is distinct from
		# the case where we use a single input, but accomodate multiple trajectories for that input (i.e., a list
		# of strings for `previous_content`).
		if isinstance(inputs, list):
			raise NotImplementedError(
				"Multiple inputs (list[dict]) are not yet supported. "
				"Currently only single input (dict) is supported."
			)

		# TODO[P2]: Support list[list[str]] for `previous_content` in the future to handle batches of inputs with
		#  multiple trajectories each.
		if is_list_of_lists(previous_content, str):
			raise NotImplementedError(
				"List of lists for previous_content is not yet supported. "
				"Currently only str or list[str] is supported."
			)

		# Normalize demos to a list of lists of dictionaries
		demos = [] if demos is None else cast(list[dict[str, Any]], demos)

		# Validate parameters - expect proper list[list[...]] structure for interventions
		# continue_reasoning drives the iteration structure, validate it first
		assert isinstance(continue_reasoning, list) and all(
			isinstance(inner, list) and all(isinstance(b, bool) for b in inner)
			for inner in continue_reasoning
		), (
			f"continue_reasoning must be a non-empty list[list[bool]], got {type(continue_reasoning)}"
		)

		num_trajectories = len(continue_reasoning)
		if num_trajectories == 0:
			raise ValueError("continue_reasoning must contain at least one trajectory.")
		num_interventions = [len(inner) for inner in continue_reasoning]
		if any(n == 0 for n in num_interventions):
			raise ValueError(
				"Each inner list in continue_reasoning must contain at least one boolean."
			)

		if previous_content is not None:
			assert (
				isinstance(previous_content, list)
				and all(isinstance(s, str) for s in previous_content)
				and len(previous_content) == num_trajectories
			), (
				f"previous_content must be None or a list[str] that contains exactly one string per trajectory, got {type(previous_content)}"
			)

		if internal_reasoning_for_output is not None:
			assert (
				is_list_of_lists(internal_reasoning_for_output, str)
				and len(internal_reasoning_for_output) == num_trajectories
				and all(
					len(inner) == num_interventions[i]
					for i, inner in enumerate(internal_reasoning_for_output)
				)
			), (
				f"internal_reasoning_for_output must be None or list[list[str]] that contains exactly one string per intervention, got {type(internal_reasoning_for_output)}"
			)

		# Validate prefix_for_output (list[list[str]] or None)
		if prefix_for_output is not None:
			assert (
				is_list_of_lists(prefix_for_output, str)
				and len(prefix_for_output) == num_trajectories
				and all(
					len(inner) == num_interventions[i]
					for i, inner in enumerate(prefix_for_output)
				)
			), (
				f"prefix_for_output must be None or list[list[str]] that contains exactly one string per intervention, got {type(prefix_for_output)}"
			)

		# Clear and populate the trajectory mappings for heterogeneous batch support
		# message_index counts across all messages (trajectories × interventions)
		self._trajectory_continue_reasoning.clear()
		self._current_assistant_prefills = []
		message_index = 0

		result: list[list[dict[str, Any]]] = []
		for trajectory_index, trajectory_continue_reasoning_list in enumerate(
			continue_reasoning
		):
			trajectory_previous_content = (
				None if previous_content is None else previous_content[trajectory_index]
			)
			trajectory_internal_reasoning_list = (
				None
				if internal_reasoning_for_output is None
				else internal_reasoning_for_output[trajectory_index]
			)
			trajectory_prefix_list = (
				None
				if prefix_for_output is None
				else prefix_for_output[trajectory_index]
			)

			# Detect if previous content contains any internal reasoning or
			# the current interventions do (only matters if we have internal reasoning fields)
			has_internal_reasoning = (
				trajectory_previous_content and "internal_reasoning" in trajectory_previous_content
			) or trajectory_internal_reasoning_list is not None

			# Format all interventions for this trajectory
			conversation_messages = self.format_single_trajectory_with_interventions(
				signature=signature,
				inputs=inputs,
				continue_reasoning=trajectory_continue_reasoning_list,
				demos=cast(list[dict[str, Any]], demos),
				response_length=response_length,
				thought_length=thought_length,
				has_internal_reasoning=has_internal_reasoning,
				previous_content=trajectory_previous_content,
				internal_reasoning_for_output=trajectory_internal_reasoning_list,
				prefix_for_output=trajectory_prefix_list,
				final_output_kind=final_output_kind,
			)

			# Store continue_reasoning values and extract prefills for each intervention
			for intervention_idx, intervention_messages in enumerate(
				conversation_messages
			):
				# Store the continue_reasoning value for this specific intervention
				self._trajectory_continue_reasoning[message_index] = (
					trajectory_continue_reasoning_list[intervention_idx]
				)
				message_index += 1

				# Extract assistant prefills for parsing (if any)
				if intervention_messages[-1]["role"] == "assistant":
					self._current_assistant_prefills.append(
						intervention_messages[-1]["content"]
					)
				else:
					self._current_assistant_prefills.append("")

			result.extend(conversation_messages)

		return result

	def _create_input_messages(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		demos: list[dict[str, Any]] | None = None,
		response_length: ResponseLength | None = None,
		thought_length: ResponseLength | None = None,
		has_internal_reasoning: bool = False,
		final_output_kind: FINAL_OUTPUT_KIND_TYPE = "synthesis_faithful",
	) -> list[dict[str, Any]]:
		"""
		Create the base messages for the input.

		The structure of the returned list of messages is as follows:

		1. A system message that contains the system prompt.
		2. A user message containing the input of an in-context example.
		3. An assistant message containing the answer to the in-context example.
		4. A final user message which specifies the main request's input.

		NOTE: Steps (2) and (3) are only included if `demos` are provided, and repeat
		for each in-context example.

		Args:
			signature: The DSPy signature defining the task's inputs, reasoning steps, outputs, and instructions.
			inputs: A dictionary containing the input data for the current task.
			demos: A list of in-context examples to include in the request.
			response_length: Optional ResponseLength object specifying constraints on the length of responses.
			thought_length: Optional ResponseLength object specifying constraints on the length of thoughts.
			has_internal_reasoning: Whether internal reasoning is used.
			final_output_kind: Kind of final output instruction (SYNTHESIS or CONCLUSION). Defaults to SYNTHESIS.

		Returns:
			A list of messages formatted for the vLLM generator adapter. Each message is a dictionary that contains
			the role of the message's speaker (i.e., SYSTEM, USER, or ASSISTANT) and the content of the message.
			This list of messages does not include a final assistant message (used for continuing reasoning).
		"""
		# TODO[P3]: Accomodate for input fields that are optional within the assertion below.
		assert signature.input_fields.keys() == inputs.keys(), (
			f"Input fields in signature ({signature.input_fields.keys()}) do not match input fields in inputs ({inputs.keys()})"
		)

		base_messages = []
		base_messages.append(
			{
				"role": "system",
				"content": self.create_system_prompt(
					signature=signature,
					thought_length=thought_length,
					response_length=response_length,
					has_internal_reasoning=has_internal_reasoning,
					final_output_kind=final_output_kind,
				),
			}
		)
		if demos:
			base_messages.extend(
				self.format_demos(
					signature=signature,
					demos=demos or [],
					has_internal_reasoning=has_internal_reasoning,
				)
			)
			base_messages.append(
				{
					"role": "user",
					"content": self.format_user_message_content(
						signature=signature,
						inputs=inputs,
						main_request=True,
					),
				}
			)
		else:
			base_messages.append(
				{
					"role": "user",
					"content": self.format_user_message_content(
						signature=signature,
						inputs=inputs,
						main_request=False,
					),
				}
			)
		return base_messages

	def format_single_trajectory_with_interventions(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		continue_reasoning: list[bool],
		demos: list[dict[str, Any]] | None = None,
		response_length: ResponseLength | None = None,
		thought_length: ResponseLength | None = None,
		has_internal_reasoning: bool = False,
		previous_content: str | None = None,
		internal_reasoning_for_output: list[str] | None = None,
		prefix_for_output: list[str] | None = None,
		final_output_kind: FINAL_OUTPUT_KIND_TYPE = "synthesis_faithful",
	) -> list[list[dict[str, Any]]]:
		"""Format a single trajectory with one or more interventions.

		This method is called once per trajectory from the format() method.
		It handles all interventions for that single trajectory.

		Args:
			signature: The DSPy signature defining the task's inputs, reasoning steps, outputs, and instructions.
			inputs: A dictionary containing the input data for the current task.
			continue_reasoning: List of booleans, one per intervention.
			demos: A list of in-context examples to include in the request.
			response_length: Optional ResponseLength object specifying constraints on the length of responses.
			thought_length: Optional ResponseLength object specifying constraints on the length of thoughts.
			has_internal_reasoning: Whether internal reasoning is used.
			previous_content: Previous content to continue reasoning from (single trajectory).
			internal_reasoning_for_output: List of internal reasoning strings, one per intervention. None if no internal reasoning.
			prefix_for_output: List of prefix strings, one per intervention. None if no prefixes.
			final_output_kind: Kind of final output instruction (SYNTHESIS or CONCLUSION). Defaults to SYNTHESIS.

		Returns:
			A list of message lists, one per intervention: [[messages1], [messages2], ...].
		"""
		# Create the base messages (system, demos, user) that are common to all interventions
		base_messages = self._create_input_messages(
			signature=signature,
			inputs=inputs,
			demos=demos,
			response_length=response_length,
			thought_length=thought_length,
			has_internal_reasoning=has_internal_reasoning,
			final_output_kind=final_output_kind,
		)

		num_interventions = len(continue_reasoning)
		previous_content = "" if previous_content is None else previous_content

		# Process each intervention for this trajectory
		result: list[list[dict[str, Any]]] = []
		for i in range(num_interventions):
			intervention_internal_reasoning = (
				internal_reasoning_for_output[i]
				if internal_reasoning_for_output is not None else ""
			)
			intervention_prefix = prefix_for_output[i] if prefix_for_output is not None else ""
			intervention_continue = continue_reasoning[i]

			if (
				has_internal_reasoning
				and intervention_continue
				and "## internal_reasoning" in previous_content
				and not intervention_internal_reasoning.strip()
			):
				raise AssertionError(
					"internal_reasoning must be provided when previous_content includes "
					"internal_reasoning and continue_reasoning is True."
				)

			if not intervention_continue and intervention_internal_reasoning.strip():
				raise AssertionError(
					"internal_reasoning must be empty when continue_reasoning is False."
				)
			if not intervention_continue and intervention_prefix.strip():
				raise AssertionError(
					"prefix_for_output must be empty when continue_reasoning is False."
				)

			# Format the assistant message for this intervention
			assistant_content = self.format_continued_assistant_message(
				signature=signature,
				continue_reasoning=intervention_continue,
				previous_content=previous_content,
				internal_reasoning_for_output=intervention_internal_reasoning,
				prefix_for_output=intervention_prefix,
			)

			# Create the complete message list
			messages = copy.deepcopy(base_messages)
			if assistant_content:
				messages.append({
					"role": "assistant",
					"content": assistant_content,
				})
			result.append(messages)

		return result

	def create_system_prompt(
		self,
		signature: type[ReasoningSignature],
		thought_length: ResponseLength | None = None,
		response_length: ResponseLength | None = None,
		has_internal_reasoning: bool = False,
		final_output_kind: FINAL_OUTPUT_KIND_TYPE = "synthesis_faithful",
	) -> str:
		"""Create the system prompt for the vLLM generator adapter.

		This method creates the system prompt that will be used by the vLLM generator adapter.
		It includes task instructions, field descriptions, reasoning field instructions, and
		templates for generating steps and answers.

		Args:
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.
		    thought_length: Optional ResponseLength object specifying constraints on the length of thoughts.
		    response_length: Optional ResponseLength object specifying constraints on the length of responses.
		    has_internal_reasoning: Whether internal reasoning is used.
		        Determines which system prompt template to use (internal_reasoning variants vs vanilla).
		    final_output_kind: Kind of final output instruction (synthesis_faithful,
				synthesis_strict, synthesis_restructured, conclusion). Defaults to synthesis_faithful.

		Returns:
		    A string representing the system prompt.
		"""
		# Extract reasoning field information from signature
		reasoning_field_name, reasoning_field, reasoning_field_type = (
			self._extract_reasoning_field_info(signature)
		)
		# Parse the signature to get input fields and output fields
		# TODO[P3]: Add support for parsing multiple reasoning fields (instead of parsing only one
		# 	via `extract_reasoning_field_info`)
		input_fields, _, output_fields = parse_reasoning_signature(
			input_field_names=list(signature.input_fields.keys()),
			reasoning_field_names=list(signature.reasoning_fields.keys()),
			output_field_names=list(signature.output_fields.keys()),
		)

		# Create a section that describes the input and output fields.
		field_descriptions = format_field_description(signature)

		# Generate the length requirements for thoughts and responses.
		thought_length_instruction = format_thought_length_instruction(
			thought_length=thought_length,
			reasoning_field_name=reasoning_field_name,
		)
		response_length_instruction = format_response_length_instruction(
			response_length=response_length,
		)
		thought_length_instruction_formatted = (
			f"\n- {thought_length_instruction}" if thought_length_instruction else ""
		)
		response_length_instruction_formatted = (
			f"\n- {response_length_instruction}" if response_length_instruction else ""
		)
		template = (
			GENERATOR_SYSTEM_PROMPT_INTERNAL_REASONING
			if has_internal_reasoning else
			GENERATOR_SYSTEM_PROMPT_VANILLA
		)
		extra = reasoning_field.json_schema_extra
		if callable(extra) or extra is None:
			extra = {}
		# TODO[P3]: Make default description a constant variable.
		reasoning_field_desc = str(extra.get("desc", "a reasoning step"))
		system_prompt = template.format(
			task_instructions=signature.instructions,
			field_descriptions=field_descriptions,
			output_fields=output_fields,
			reasoning_field_name=reasoning_field_name,
			reasoning_field_type=reasoning_field_type.__name__,
			reasoning_field_description=uncapitalize_first_letter(reasoning_field_desc),
			thought_length_instruction_formatted=thought_length_instruction_formatted,
			response_length_instruction_formatted=response_length_instruction_formatted,
			input_fields=input_fields,
			output_field_sections=generate_output_field_sections(
				list(signature.output_fields.keys())
			),
			final_output_description=get_final_output_description(final_output_kind),
		).strip()
		return system_prompt

	def format_continued_assistant_message(
		self,
		signature: type[ReasoningSignature],
		continue_reasoning: bool,
		previous_content: str = "",
		internal_reasoning_for_output: str = "",
		prefix_for_output: str = "",
	) -> str:
		"""
		Format an assistant message that continues from previous content, introducing a new reasoning step or the answer.

		Args:
		    signature: The DSPy signature defining the fields. Used to extract reasoning field info.
			continue_reasoning: If True, add a reasoning step. If False, add the answer section.
		    previous_content: The previously generated assistant content (e.g., all prior <thinking> and <step> tags).
		    internal_reasoning_for_output: Internal reasoning guidance from controller (complete statement).
		    prefix_for_output: Prefix to add after the reasoning step header (only for reasoning steps).

		Returns:
		    A string representing the continued assistant message.
		"""

		# Start building the response
		stripped_content = previous_content.strip()
		parts: list[str]
		if not stripped_content and continue_reasoning is False:
			parts = []
		else:
			parts = [stripped_content] if stripped_content else ["<thinking>"]
		if continue_reasoning:
			# If previous content ends with </thinking>, remove it to add more steps
			while parts[0].strip().endswith("</thinking>"):
				parts[0] = parts[0].rstrip().removesuffix("</thinking>").rstrip()
			# If previous content contains an incomplete step, close it first
			# (check after removing </thinking> tag)
			if (
				parts[0].strip()
				and "<step>" in parts[0]
				and not parts[0].strip().endswith("</step>")
			):
				parts.append("</step>")
			parts.append("<step>")
			# Add controller's internal reasoning guidance (complete statement)
			if internal_reasoning_for_output:
				parts.append(
					f"## internal_reasoning\n{internal_reasoning_for_output}"
				)
			# Always add the actual reasoning field name with optional prefix
			reasoning_field_name, _, _ = self._extract_reasoning_field_info(signature)
			parts.append(f"## {reasoning_field_name}")
			if prefix_for_output:
				parts.append(prefix_for_output)
			else:
				parts[-1] += "\n"  # Force the LLM continue from a new line.
		else:
			# Transitioning to answer section
			# If we have previous content, handle closing tags
			if parts:
				# If previous content contains an incomplete step, close it first
				if (
					parts[0].strip()
					and "<step>" in parts[0]
					and not parts[0].strip().endswith("</step>")
					and not parts[0].strip().endswith("</thinking>")
				):
					parts.append("</step>")
				# Close thinking tag if needed
				if not parts[0].strip().endswith("</thinking>"):
					parts.append("</thinking>")
			# Add answer section (regardless of whether we had previous content)
			parts.append("<answer>")
			# Always add the header for the first output field
			output_field_names = list(signature.output_fields.keys())
			# We prefill only the first output field name, and the model generates the rest
			assert len(output_field_names) > 0, (
				"output_field_names must be a non-empty list"
			)
			parts.append(f"## {output_field_names[0]}")
			# Add the prefix only if provided
			if prefix_for_output:
				parts.append(prefix_for_output)
			else:
				parts[-1] += "\n"  # Force the LLM to continue from a new line.
		return "\n".join(parts)

	def user_message_output_requirements(
		self,
		signature: type[ReasoningSignature],
		has_internal_reasoning: bool = False,
	) -> str:
		"""Returns output format requirements for the language model.

		This method generates guidance about the expected output structure, tailored to whether
		internal reasoning is needed for the final response.

		Args:
		    signature: The DSPy signature defining the fields.
		    has_internal_reasoning: Whether internal reasoning should be included in the output.
		        If True, internal_reasoning fields are expected (if present in signature).
		        If False, internal_reasoning fields are not expected even if present in signature.
		Returns:
		    A string describing the expected output format.
		"""
		# Extract reasoning field information from signature
		reasoning_field_name, _, _ = self._extract_reasoning_field_info(signature)

		# Get output field names for formatting
		_, _, output_fields = parse_reasoning_signature(
			input_field_names=list(signature.input_fields.keys()),
			reasoning_field_names=list(signature.reasoning_fields.keys()),
			output_field_names=list(signature.output_fields.keys()),
		)

		# Format instructions for reasoning steps
		if has_internal_reasoning:
			step_instruction = f"Each `<step>` section should include a `## internal_reasoning` section (guidance provided to help your thinking), followed by a `## {reasoning_field_name}` section."
		else:
			step_instruction = f"Each `<step>` section should contain a `## {reasoning_field_name}` section."

		# Format instructions for answer section
		answer_instruction = f"The `<answer>` section should include sections for {output_fields}."

		# Create complete formatting guidance
		# TODO[P3]: Replace spaces with tabs in the format_guidance string below, and adjust tests accordingly.
		format_guidance = (
			f"Structure your response as follows:\n\n"
			f"1. Begin with `<thinking>...</thinking>` tags. Inside these tags, include multiple `<step>...</step>` sections for your {reasoning_field_name}s.\n"
			f"\t{step_instruction}\n"
			f"2. After the `</thinking>` tag, include your final answer within `<answer>...</answer>` tags.\n"
			f"\t{answer_instruction}"
		)
		return format_guidance.strip()

	def format_demo_assistant_message(
		self,
		signature: type[ReasoningSignature],
		demo: dict[str, Any],
		has_internal_reasoning: bool = False,
	) -> str:
		"""Format an assistant message for an in-context example.

		Creates a properly formatted assistant message with <thinking> containing <step> tags
		and <answer> with output fields, based on the provided demo example.

		Args:
		    signature: The DSPy signature defining the fields.
		    demo: A dictionary containing constants.INPUT, "reasoning", and constants.OUTPUT keys.
		    has_internal_reasoning: Whether internal reasoning should be included in the demo.
		        If True, internal_reasoning fields are shown (if present in demo data).
		        If False, internal_reasoning fields are hidden even if present in demo data.

		Returns:
		    A formatted string representing the assistant's response with proper tags.
		"""
		# Extract reasoning field information from signature
		reasoning_field_name, _, _ = self._extract_reasoning_field_info(signature)
		# Build message parts
		message_parts = []
		message_parts.append("<thinking>")
		# Add reasoning steps
		reasoning_steps = demo[ReasoningState.REASONING]
		for step_dict in reasoning_steps:
			message_parts.append("<step>")
			# Include internal_reasoning only if enabled and present in demo (controller guidance)
			if has_internal_reasoning:
				assert "internal_reasoning" in step_dict, (
					"Expected internal reasoning field 'internal_reasoning' in demo reasoning step, but it is missing. "
					"Ensure that has_internal_reasoning is set correctly based on the demo data."
				)
				message_parts.append(
					f"## internal_reasoning\n{step_dict['internal_reasoning']}"
				)
			# Add the reasoning step itself
			step_value = step_dict[reasoning_field_name]
			# Format the reasoning step value according to its type
			if not isinstance(step_value, str):
				step_value = str(step_value)
			message_parts.append(f"## {reasoning_field_name}\n{step_value}")
			message_parts.append("</step>")

		message_parts.append("</thinking>")
		message_parts.append("<answer>")

		# Add each output field
		for field_name, field_info in signature.output_fields.items():
			message_parts.append(f"## {field_name}")
			# Format the output value according to its field info
			formatted_value = format_field_value(
				field_info, demo[ReasoningState.OUTPUT][field_name]
			)
			message_parts.append(formatted_value)

		message_parts.append("</answer>")
		return "\n".join(message_parts)

	def format_user_message_content(
		self,
		signature: type[ReasoningSignature],
		inputs: dict[str, Any],
		prefix: str = "",
		suffix: str = "",
		main_request: bool = False,
		internal_reasoning_for_output: str = "",
	) -> str:
		"""Format the content of the user message.

		This method formats the user message content, including input fields and task guidance
		for the main request. In-context examples (main_request=False) only include input fields.

		Args:
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.
		    inputs: A dictionary containing the input data for the current task.
		    prefix: Optional prefix to add before the user message content.
		    suffix: Optional suffix to add after the user message content.
		    main_request: Whether this is the main request (True) or an in-context example (False).
		    internal_reasoning_for_output: Internal reasoning guidance being provided (if any).

		Returns:
		    A formatted string containing the user message content.
		"""
		assert len(inputs.keys()) > 0, "Inputs must be a non-empty dictionary"
		# Extract reasoning field information from signature
		reasoning_field_name, _, _ = self._extract_reasoning_field_info(signature)

		# Get field names
		input_field_names = list(signature.input_fields.keys())
		output_field_names = list(signature.output_fields.keys())
		_, output_fields = parse_base_signature(input_field_names, output_field_names)

		# Start building the message
		messages = [prefix if prefix else signature.instructions]

		# Format each input field
		for k, v in signature.input_fields.items():
			if k in inputs:
				value = inputs.get(k)
				formatted_field_value = format_field_value(field_info=v, value=value)
				messages.append(f"## {k}\n{formatted_field_value}")

		# Add task guidance and output requirements for main requests only
		if main_request:
			# Check if internal reasoning guidance is provided
			has_internal_reasoning = bool(internal_reasoning_for_output)

			task_guidance = self._generate_task_guidance(
				output_fields=output_fields,
				has_internal_reasoning=has_internal_reasoning,
				reasoning_field_name=reasoning_field_name,
			)
			if task_guidance:
				messages.append(task_guidance)

			# Add output format requirements
			output_requirements = self.user_message_output_requirements(
				signature=signature,
				has_internal_reasoning=has_internal_reasoning,
			)
			if output_requirements:
				messages.append(output_requirements)

		# Add suffix if provided
		if suffix:
			messages.append(suffix)

		return "\n\n".join(messages).strip()

	def _generate_task_guidance(
		self,
		output_fields: str,
		has_internal_reasoning: bool,
		reasoning_field_name: str,
	) -> str:
		"""Generate guidance for completing the task with step-by-step reasoning.

		This method creates task-specific guidance, optionally including information about
		internal reasoning guidance if it's being provided.

		Args:
		    output_fields: A formatted string representation of a list of output fields.
		    has_internal_reasoning: Whether internal reasoning guidance is being provided.
		    reasoning_field_name: Name of the reasoning field.

		Returns:
		    A string containing guidance for completing the task.
		"""
		guidance_parts = []

		guidance_parts.append(
			f"To produce {output_fields}, reason step-by-step by writing a sequence of {reasoning_field_name}s."
		)
		if has_internal_reasoning:
			guidance_parts.append(
				f"Use the internal reasoning guidance provided to help you generate each {reasoning_field_name} and finally {output_fields}."
			)

		return "\n".join(guidance_parts)

	def format_demos(
		self,
		signature: type[ReasoningSignature],
		demos: list[dict[str, Any]],
		has_internal_reasoning: bool = False,
	) -> list[dict[str, str]]:
		"""Format the in-context examples into a list of messages.

		Transforms each demo into a pair of user and assistant messages, where the
		user message contains the inputs and the assistant message contains the
		reasoning steps and outputs.

		Args:
		    signature: The DSPy signature for which to format the few-shot examples.
		    demos: A list of examples. Each example is a dictionary containing:
		        - constants.INPUT: A dictionary mapping input field names to their values
		        - "reasoning": A list of reasoning step dictionaries
		        - constants.OUTPUT: A dictionary mapping output field names to their values
		    has_internal_reasoning: Whether internal reasoning should be included in demos.
		        If True, internal_reasoning fields are shown (if present in demo data).
		        If False, internal_reasoning fields are hidden even if present in demo data.

		Returns:
		    A list of messages alternating between user and assistant roles.
		"""
		# Extract reasoning field information from signature
		reasoning_field_name, _, _ = self._extract_reasoning_field_info(signature)

		# Validate that all demos have the required structure and fields
		for i, demo in enumerate(demos):
			# Check that demo has all three required keys
			assert set(demo.keys()).issuperset(
				{ReasoningState.INPUT, ReasoningState.REASONING, ReasoningState.OUTPUT}
			), (
				f"Demo {i} is missing one or more required keys ('input', 'reasoning', 'output')"
			)
			# Check that input dictionary contains all required input fields
			input_fields_set = set(signature.input_fields.keys())
			demo_input_fields_set = set(demo[ReasoningState.INPUT].keys())
			assert demo_input_fields_set.issuperset(input_fields_set), (
				f"Demo {i} input is missing required fields: {input_fields_set - demo_input_fields_set}"
			)
			# Check that output dictionary contains all required output fields
			output_fields_set = set(signature.output_fields.keys())
			demo_output_fields_set = set(demo[ReasoningState.OUTPUT].keys())
			assert demo_output_fields_set.issuperset(output_fields_set), (
				f"Demo {i} output is missing required fields: {output_fields_set - demo_output_fields_set}"
			)
			# Check that reasoning is a list
			assert isinstance(demo[ReasoningState.REASONING], list), (
				f"Demo {i} reasoning must be a list, got {type(demo[ReasoningState.REASONING])}"
			)
			# Check that each reasoning step has the required field
			for j, step in enumerate(demo[ReasoningState.REASONING]):
				assert reasoning_field_name in step, (
					f"Demo {i} reasoning step {j} is missing the required field '{reasoning_field_name}'"
				)

		messages = []

		# Format each demo into user and assistant messages
		for demo in demos:
			# Create user message with input fields
			messages.append(
				{
					"role": "user",
					"content": self.format_user_message_content(
						signature,
						inputs=demo[ReasoningState.INPUT],
						prefix=signature.instructions,
						main_request=False,
					).strip(),
				}
			)
			# Create assistant message with reasoning and outputs
			messages.append(
				{
					"role": "assistant",
					"content": self.format_demo_assistant_message(
						signature=signature,
						demo=demo,
						has_internal_reasoning=has_internal_reasoning,
					).strip(),
				}
			)
		return messages

	def _extract_reasoning_field_info(
		self, signature: type[ReasoningSignature]
	) -> tuple[str, FieldInfo, type[Any]]:
		"""Extract reasoning field information from a ReasoningSignature.

		Args:
		    signature: The signature to extract information from

		Returns:
		    A tuple of (field_name, field_info, field_type)

		Raises:
		    ValueError: If signature is not a ReasoningSignature or has no reasoning fields
		"""
		# TODO[P3]: Move this to reasoning signature implementation/class.
		if (
			not isinstance(signature, type)
			or not hasattr(signature, "__bases__")
			or ReasoningSignature not in signature.__bases__
		):
			raise ValueError(
				f"Signature must be a ReasoningSignature, got {type(signature)}"
			)

		# Get reasoning fields using the property defined in ReasoningSignatureMeta
		reasoning_fields = getattr(signature, "reasoning_fields", {})
		if not reasoning_fields:
			raise ValueError(
				f"ReasoningSignature {signature.__name__} must have at least one reasoning field"
			)

		# Use the first non-internal reasoning field if available, otherwise use the first field
		field_name = next(
			(name for name in reasoning_fields if name != "internal_reasoning"),
			next(iter(reasoning_fields)),
		)
		field_info = reasoning_fields[field_name]
		field_type = field_info.annotation or str
		return field_name, field_info, field_type

	def _determine_stop_tokens(self, continue_reasoning: bool) -> list[str]:
		"""Determine stop tokens for a single input based on continue_reasoning."""
		if continue_reasoning:
			# When continuing reasoning, stop at end of step
			return ["</step>"]
		else:
			# When generating final answer, stop at end of answer
			return ["</answer>"]
