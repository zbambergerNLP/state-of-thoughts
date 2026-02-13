# Standard library imports
import copy
import logging
from typing import Any

# Third-party imports
import dspy
import dspy.utils
import json_repair
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.base_type import Type
from dspy.adapters.utils import (
	format_field_value,
	get_annotation_name,
	parse_value,
)
from dspy.utils.callback import BaseCallback
from vllm import SamplingParams

# Local imports
from adapter.adapter_constants import FIELD_HEADER_PATTERN
from adapter.constraints import (
	ResponseLength,
	format_response_length_instruction,
)
from adapter.prompts import SIMPLE_MAIN_TEMPLATE
from adapter.tool_schema import format_dspy_tool_as_openai_tool
from adapter.utils import (
	format_field_description,
	normalize_numeric_field_value,
)
from lm.generative_local_lm import (
	ChatCompletionResponse,
	Choice,
	GenerativeLocalVLLM,
)
from misc_utils import ExecutionError
from tree.tree_constants import ReasoningState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class LocalVLLMAdapter:

	def __init__(
		self,
		use_native_function_calling: bool = False,
		native_response_types: list[type[Type]] | None = None,
		callbacks: list[BaseCallback] | None = None,
	) -> None:
		"""
		Initialize the LocalVLLMAdapter.

		Args:
			use_native_function_calling: Whether to use native function calling.
			native_response_types: Optional list of DSPy native response types to handle during
				preprocessing (e.g., citations, other structured native response types).
			callbacks: Optional list of callbacks to execute during execution. Callbacks should implement BaseCallback interface
				from DSPy.
		"""
		self.callbacks = callbacks or []
		self.use_native_function_calling = use_native_function_calling
		# TODO[P2]: Add support for native LLM outputs such as reasoning (for qwen3, r1, etc...)
		# and citations.
		self.native_response_types = native_response_types or []

	@staticmethod
	def _validate_call_preconditions(
		inputs: dict[str, Any] | list[dict[str, Any]],
		lm_kwargs: dict[str, Any] | list[dict[str, Any]],
	) -> None:
		"""Validate `__call__` preconditions for this adapter implementation.

		This adapter supports:
		- single input + single kwargs dict
		- batch inputs + either a single kwargs dict (broadcast) or a list of kwargs dicts

		If `lm_kwargs` is a list, this implementation assumes it is aligned 1:1 with the input
		batch, so we enforce that constraint explicitly.
		"""
		if isinstance(lm_kwargs, list) and not isinstance(inputs, list):
			raise ValueError("When lm_kwargs is a list, inputs must also be a list")
		if isinstance(inputs, list) and isinstance(lm_kwargs, list) and len(inputs) != len(lm_kwargs):
			raise ValueError("When lm_kwargs is a list, it must have the same length as inputs")

	def _parse_chat_completion(
		self,
		signature: type[dspy.Signature],
		completion: str,
	) -> dict[str, Any]:
		"""
		Parse a chat-style completion using ##-header sections.

		This is the default parsing mode in DSPy (for non-structured outputs/tools).

		Args:
			signature: The signature defining expected output fields and their types.
			completion: The raw model completion text.

		Returns:
			A dict mapping output field names to parsed values.
		"""
		sections: list[tuple[str | None, list[str]]] = [(None, [])]

		for line in completion.splitlines():
			stripped_line = line.strip()
			match = FIELD_HEADER_PATTERN.match(stripped_line)
			if match:
				header = match.group(1)
				if header in signature.output_fields:
					remaining_content = stripped_line[match.end() :].strip()
					sections.append((header, [remaining_content] if remaining_content else []))
					continue
				else:
					sections[-1][1].append(line)
			else:
				sections[-1][1].append(line)

		processed_sections = [(k, "\n".join(v).strip()) for k, v in sections]

		# Collect LAST non-empty occurrence of each field first, then parse. This prevents
		# early template echoes (e.g. "<value>") from aborting parsing before the real
		# value appears later in the completion.
		raw_fields: dict[str, str] = {}
		for k, v in processed_sections:
			if k is None or (k not in signature.output_fields):
				continue
			if not v.strip():
				continue
			raw_fields[k] = v

		fields: dict[str, Any] = {}
		for k in signature.output_fields.keys():
			if k not in raw_fields:
				continue
			v = raw_fields[k]
			try:
				annotation = signature.output_fields[k].annotation
				annotation_name = get_annotation_name(annotation).lower()
				value_for_parse: Any = v
				# Normalize numeric field values for parsing if the corresponding output field
				# is numeric (float or int)
				if isinstance(v, str) and ("float" in annotation_name or "int" in annotation_name):
					value_for_parse = normalize_numeric_field_value(v)
				fields[k] = parse_value(value_for_parse, annotation)
			except Exception as e:
				raise dspy.utils.exceptions.AdapterParseError(
					adapter_name="ChatAdapter",
					signature=signature,
					lm_response=completion,
					message=(
						f"Failed to parse field {k} with value {v} from the LM response. "
						f"Error message: {e}"
					),
				) from e

		if fields.keys() != signature.output_fields.keys():
			missing_fields = set(signature.output_fields.keys()) - set(fields.keys())
			found_fields = list(fields.keys())
			expected_fields = list(signature.output_fields.keys())
			raise dspy.utils.exceptions.AdapterParseError(
				adapter_name="ChatAdapter",
				signature=signature,
				lm_response=completion,
				parsed_result=fields,
				message=(
					f"Expected to find output fields in the LM response: {expected_fields}. "
					f"Actual output fields parsed from the LM response: {found_fields}. "
					f"Missing fields: {list(missing_fields)}. "
					f"Raw fields found: {list(raw_fields.keys())}."
				),
			)
		return fields

	def _call_preprocess(
		self,
		lm: GenerativeLocalVLLM,
		lm_kwargs: dict[str, Any],
		signature: type[dspy.Signature],
		inputs: dict[str, Any],
	) -> tuple[type[dspy.Signature], dict[str, Any]]:
		"""
		Preprocess signature/lm_kwargs before calling the local LM.

		Supports:
		- vLLM native tool calling for controller signatures that include a `ToolCalls`
		  output field (and a `dspy.Tool` / `list[dspy.Tool]` input field).

		Args:
			lm: The GenerativeLocalVLLM instance to use for generating responses.
			lm_kwargs: Additional keyword arguments to pass to the GenerativeLocalVLLM's forward method.
				Can be a single dict (applied to all inputs) or a list of dicts (one per input).
			signature: The DSPy signature defining the input and output fields of the task.
			inputs: The input arguments for a single instance.

		Returns:
			A tuple of:
			- processed_signature: The signature used for formatting/parsing the model's text.
				If native function calling is not used, the original signature is returned.
				Otherwise, the signature is modified to include the tool call input and
				output fields.
			- processed_lm_kwargs: The kwargs dict to pass to the LM call. This may include a
			  vLLM/OpenAI-compatible `tools=...` entry when tool calling is enabled.
		"""
		processed_lm_kwargs = dict(lm_kwargs)
		if not self.use_native_function_calling:
			return signature, processed_lm_kwargs

		# TODO[P2]: Ensure that the generative controller (and perhaps the evaluator if using tool
		# calls to produce scores) sets `use_native_function_calling` to True.
		tool_call_input_field_name = ChatAdapter()._get_tool_call_input_field_name(signature)
		tool_call_output_field_name = ChatAdapter()._get_tool_call_output_field_name(signature)

		if tool_call_output_field_name and tool_call_input_field_name is None:
			raise ValueError(
				f"You provided an output field {tool_call_output_field_name} to receive the "
				"tool calls information, but did not provide any tools as the input. Please "
				"provide a list of tools as the input by adding an input field with type "
				"`list[dspy.Tool]`."
			)

		if tool_call_output_field_name and tool_call_input_field_name is not None:
			tools = inputs[tool_call_input_field_name]
			tools = tools if isinstance(tools, list) else [tools]

			dspy_tools: list[dspy.Tool] = []
			for tool in tools:
				if not isinstance(tool, dspy.Tool):
					raise TypeError(
						f"Expected tools to be `dspy.Tool` objects, got {type(tool)} instead."
					)
				dspy_tools.append(tool)

			# vLLM consumes OpenAI-compatible tool schema; we emit JSON Schema with enum support.
			processed_lm_kwargs["tools"] = [
				format_dspy_tool_as_openai_tool(tool) for tool in dspy_tools
			]

			signature_for_native_function_calling = signature.delete(tool_call_output_field_name)
			signature_for_native_function_calling = signature_for_native_function_calling.delete(
				name=tool_call_input_field_name,
			)
			return signature_for_native_function_calling, processed_lm_kwargs

		# TODO[P2]: Add support for native LLM outputs such as reasoning (for qwen3, r1, etc...)
		# For reference, see: https://github.com/stanfordnlp/dspy/blob/becb4c9e849a292f71a1f1623acd375aaa32edef/dspy/adapters/base.py#L103-L112
		return signature, processed_lm_kwargs

	def _build_value_from_choice(
		self,
		processed_signature: type[dspy.Signature],
		original_signature: type[dspy.Signature],
		choice: Choice,
		sampling_params: SamplingParams,
		tools_were_provided: bool,
		tool_call_output_field_name: str | None,
	) -> dict[str, Any]:
		"""
		Convert a OpenAI-style Choice (completion) into a parsed dictionary for DSPy.

		Parses a `Choice` object into a dictionary of parsed output fields.
		A `Choice` object is one possible completion for an input produced by the LLM (a
		`GenerativeLocalVLLM` instance). In practice, the LLM produces a sequence of
		`ChatCompletionResponse` objects (one per input), each containing a list of `Choice`
		objects (one per completion for a given input).

		NOTE:
			- Parse failures yield a dict containing only an "error" key with an ExecutionError
			  object with error_type="parsing".
			- Successful parses include an "error" key with error_type=None.

		Args:
			processed_signature: Signature used for parsing (may exclude tool-call fields).
			original_signature: Original signature; ensures the returned dict contains all output
				fields expected by downstream DSPy consumers.
			choice: A completion `Choice`. Parsing primarily uses `choice.text`, and optionally
				`choice.logprobs`.
			sampling_params: The vLLM SamplingParams for the corresponding request. Used to route
				between chat parsing and structured-output JSON parsing.
			tools_were_provided: Whether the LM call included a `tools=...` argument. If true and the
				signature expects ToolCalls, tool calls are parsed from `choice.text` (not from the
				Choice object).
			tool_call_output_field_name: Output field name that should receive tool calls, if any.

		Returns:
			A dict containing parsed output fields, error metadata, and optional tool call info.
		"""
		candidate_text = choice.text or ""

		if tools_were_provided and tool_call_output_field_name and candidate_text.strip():
			try:
				loaded = json_repair.loads(candidate_text.strip())
				if isinstance(loaded, tuple):
					loaded = loaded[0]

				tool_calls: list[dict[str, Any]] = (
					loaded["tool_calls"]
					if isinstance(loaded, dict) and "tool_calls" in loaded
					else loaded
				)
				tool_calls = [
					{
						"name": v["function"]["name"],
						"args": json_repair.loads(v["function"].get("arguments", "{}") or "{}"),
					}
					for v in tool_calls
				]
				# Completions that are error-free include an empty ExecutionError object, which
				# denotes that no errors occured.
				return {
					tool_call_output_field_name: dspy.ToolCalls.from_dict_list(tool_calls),
					"error": ExecutionError(),
				}
			except Exception as e:
				return {"error": ExecutionError(error_type="parsing", raw_output=candidate_text, error_message=str(e))}

		try:
			parsed = self.parse(
				signature=processed_signature,
				completion=candidate_text,
				sampling_params=sampling_params,
			)
			value = {**parsed, "error": ExecutionError()}
			candidate_logprobs = choice.logprobs
			if candidate_logprobs is not None:
				value["logprobs"] = candidate_logprobs
			return value
		except dspy.utils.exceptions.AdapterParseError as e:
			logger.warning(f"Failed to parse completion; returning error: {e}")
			return {
				"error": ExecutionError(
					error_type="parsing",
					raw_output=candidate_text,
					error_message=str(e),
				),
			}


	def _call_postprocess(
		self,
		processed_signature: type[dspy.Signature],
		original_signature: type[dspy.Signature],
		outputs: list[ChatCompletionResponse],
		lm: GenerativeLocalVLLM | None,
		lm_kwargs: list[dict[str, Any]],
	) -> list[list[dict[str, Any]]]:
		"""
		Postprocess raw LM outputs into parsed dictionaries.

		This repo's `GenerativeLocalVLLM` returns `ChatCompletionResponse` objects containing
		a list of `choices`, each of which contains `text`.

		We parse according to `processed_signature` (which may differ from the original,
		e.g., when native function calling deletes tool-call fields). For compatibility, we
		then ensure all fields from `original_signature.output_fields` exist, filling missing
		fields with None.

		Args:
			processed_signature: The processed signature (after modifying the signature for
				native function calling).
			original_signature: The original signature.
			outputs: The list of ChatCompletionResponse objects (one per input, each containing a
				list of Choice objects with possible completions).
			lm: The GenerativeLocalVLLM instance to use for generating responses.
			lm_kwargs: A list of per-input keyword-argument dictionaries used for the LM call.
				This must have the same length as `outputs`, even if some inputs failed generation
				(upstream code should still provide the per-input kwargs).

		Returns:
			A list of lists of dictionaries, where the outer list corresponds to the inputs (i.e.,
			has length equal to the number of inputs), and each inner list corresponds to the
			completion attempts for each the respective input. Each inner list may contain multiple
			completion attempts if `n` is specified in `lm_kwargs` (i.e., if `n > 1` in
			`vllm.SamplingParams(...)`).
		"""
		_ = lm  # Reserved for future postprocessing needs.
		tool_call_output_field_name = ChatAdapter()._get_tool_call_output_field_name(
			signature=original_signature,
		)
		if len(lm_kwargs) != len(outputs):
			raise ValueError(
				"lm_kwargs must be a list with the same length as outputs "
				f"(got {len(lm_kwargs)} vs {len(outputs)} respectively)."
			)

		results: list[list[dict[str, Any]]] = []
		for response_idx, response in enumerate(outputs):
			current_sampling_params = self.get_sampling_params(lm_kwargs[response_idx])
			# NOTE: During pre-processing, we may have added a `tools=...` entry to the lm_kwargs.
			tools_were_provided = bool(lm_kwargs[response_idx].get("tools"))
			values = [
				self._build_value_from_choice(
					processed_signature=processed_signature,
					original_signature=original_signature,
					choice=choice,
					sampling_params=current_sampling_params,
					tools_were_provided=tools_were_provided,
					tool_call_output_field_name=tool_call_output_field_name,
				)
				for choice in response.choices
			]
			results.append(values)
		return results

	def _prepare_messages_for_call(
		self,
		signature: type[dspy.Signature],
		lm: GenerativeLocalVLLM,
		inputs: dict[str, Any] | list[dict[str, Any]],
		lm_kwargs: dict[str, Any] | list[dict[str, Any]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]],
		response_length: ResponseLength | None,
	) -> tuple[type[dspy.Signature], list[list[dict[str, Any]]], list[dict[str, Any]]]:
		"""Preprocess signature/kwargs and format messages for the LM call.

		Important: `_call_preprocess` may modify lm_kwargs (e.g., adding `tools=...`). This
		method returns the processed kwargs as a list aligned 1:1 with the batch.

		Args:
			signature: The DSPy signature defining the input and output fields of the task.
			lm: The GenerativeLocalVLLM instance to use for generating responses.
			inputs: Either a single input dictionary or a list of input dictionaries, where each
				dictionary contains the input fields defined in the signature. If a list of
				dictionaries is provided, the length of this list must match the length of `demos`
				if `demos` is a list of lists (this corresponds with the "batch dimension" of the
				inputs).
			lm_kwargs: Additional keyword arguments to pass to the GenerativeLocalVLLM's forward
				method. Can be a single dict (applied to all inputs) or a list of dicts (one per
				input).
			demos: Either a list of few-shot examples (demos) or a list of lists of demos (i.e., a
				batch of few-shot examples -- one per input). Each demo should be a dictionary
				containing "input" and "output" keys, where "input" is a dictionary of input fields
				(will be represented with user messages) and "output" is a dictionary of output
				fields (will be represented with assistant messages).
			response_length: Optional ResponseLength object specifying constraints on the length of
				the response.

		Returns:
			A tuple containing the processed signature, the formatted messages, and the processed kwargs list.
		"""
		if isinstance(inputs, list):
			base_kwargs_list = lm_kwargs if isinstance(lm_kwargs, list) else [lm_kwargs] * len(inputs)
			processed_signatures: list[type[dspy.Signature]] = []
			processed_kwargs_list: list[dict[str, Any]] = []
			for item_inputs, item_kwargs in zip(inputs, base_kwargs_list, strict=True):
				processed_sig, processed_kwargs = self._call_preprocess(
					lm=lm,
					lm_kwargs=item_kwargs,
					signature=signature,
					inputs=item_inputs,
				)
				processed_signatures.append(processed_sig)
				processed_kwargs_list.append(processed_kwargs)

			processed_signature = processed_signatures[0]
			if any(s.signature != processed_signature.signature for s in processed_signatures[1:]):
				raise ValueError("All preprocessed signatures must match across the batch.")
		else:
			first_kwargs = lm_kwargs[0] if isinstance(lm_kwargs, list) else lm_kwargs
			processed_signature, processed_kwargs = self._call_preprocess(
				lm=lm,
				lm_kwargs=first_kwargs,
				signature=signature,
				inputs=inputs,
			)
			processed_kwargs_list = [processed_kwargs]

		messages = self.format(
			signature=processed_signature,
			demos=demos,
			inputs=inputs,
			response_length=response_length,
		)
		return processed_signature, messages, processed_kwargs_list

	def __call__(
		self,
		signature: type[dspy.Signature],
		lm: GenerativeLocalVLLM,
		inputs: dict[str, Any] | list[dict[str, Any]],
		lm_kwargs: dict[str, Any] | list[dict[str, Any]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
		response_length: ResponseLength | None = None,
		verbose: bool = True,	# TODO[P2]: Make this a `verbosity` flag with the logging levels defined in `constants.py`.
	) -> list[list[dict[str, Any]]]:
		"""
		Produces a list of LLM outputs given a collection of formatted inputs.

		Args:
		    lm: The GenerativeLocalVLLM instance to use for generating responses.
		    lm_kwargs: Additional keyword arguments to pass to the GenerativeLocalVLLM's forward method.
		        Can be a single dict (applied to all inputs) or a list of dicts (one per input).
		    signature: The DSPy signature defining the input and output fields of the task.
			demos: Either a list of few-shot examples (demos) or a list of lists of demos (i.e., a batch of
				few-shot examples -- one per input). Each demo should be a dictionary containing ReasoningState.INPUT and
				ReasoningState.OUTPUT keys, where ReasoningState.INPUT is a dictionary of input fields (will be represented with user
				messages) and ReasoningState.OUTPUT is a dictionary of output fields (will be represented with assistant
				messages).
		    inputs: Either a single input dictionary or a list of input dictionaries, where each dictionary
				contains the input fields defined in the signature. If a list of dictionaries is provided,
				the length of this list must match the length of `demos` if `demos` is a list of lists (this
				corresponds with the "batch dimension" of the inputs).
		    response_length: Optional ResponseLength object specifying constraints on the length of the response.
		    verbose: Boolean indicating whether to log verbose outputs during execution.

		Returns:
			A list of lists of dictionaries, where the outer list corresponds to the inputs (i.e., has length
			equal to the number of inputs), and each inner list corresponds to the completion attempts for
			each the respective input. Each inner list may contain multiple completion attempts if `n` is specified
			in `lm_kwargs` (i.e., if `n > 1` in `vllm.SamplingParams(...)`).
		"""
		self._validate_call_preconditions(inputs=inputs, lm_kwargs=lm_kwargs)
		if demos is None:
			demos = []
		logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
		processed_signature, messages, processed_lm_kwargs_list = self._prepare_messages_for_call(
			signature=signature,
			lm=lm,
			inputs=inputs,
			lm_kwargs=lm_kwargs,
			demos=demos,
			response_length=response_length,
		)

		sampling_params_list = [self.get_sampling_params(kwargs) for kwargs in processed_lm_kwargs_list]
		sampling_fields = set(SamplingParams.__annotations__)

		forward_kwargs_list = [
			{k: v for k, v in kwargs.items() if k not in sampling_fields}
			for kwargs in processed_lm_kwargs_list
		]
		forward_kwargs = forward_kwargs_list[0] if forward_kwargs_list else {}
		assert not any(fw != forward_kwargs for fw in forward_kwargs_list[1:]), (
			"All non-sampling kwargs must match across the batch (e.g., tools/chat_template_kwargs). "
			"Provide a single shared kwargs dict for the batch or run separate adapter calls."
		)

		outputs: list[ChatCompletionResponse] = lm.batch(
			messages=messages,
			sampling_params=sampling_params_list,
			**forward_kwargs,
		)

		return self._call_postprocess(
			processed_signature=processed_signature,
			original_signature=signature,
			outputs=outputs,
			lm=lm,
			lm_kwargs=processed_lm_kwargs_list,
		)

	def format_single(
		self,
		signature: type[dspy.Signature],
		inputs: dict[str, Any],
		demos: list[dict[str, Any]] | None = None,
		response_length: ResponseLength | None = None,
	) -> list[dict[str, Any]]:
		"""Format a single input into messages for the LM call.

		Args:
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.
		    demos: A list of few-shot examples.
		    inputs: The input arguments for a single instance.
		    response_length: Optional response length constraints.

		Returns:
		    A list of messages for a single conversation.
		"""
		if demos is None:
			demos = []
		field_descriptions = format_field_description(signature)
		response_length_instruction = format_response_length_instruction(
			response_length=response_length
		)
		# Format instruction as a separate line only if it has content
		response_length_instruction_formatted = (
			f"\n\n{response_length_instruction}" if response_length_instruction else ""
		)
		system_message = SIMPLE_MAIN_TEMPLATE.format(
			task_instructions=signature.instructions,
			field_descriptions=field_descriptions,
			response_length_instruction=response_length_instruction,
			response_length_instruction_formatted=response_length_instruction_formatted,
		).strip()
		messages: list[dict[str, Any]] = []
		messages.append({"role": "system", "content": system_message})
		messages.extend(self.format_demos(signature=signature, demos=demos))
		content = self.format_user_message_content(
			signature=signature, inputs=inputs, main_request=True
		)
		messages.append({"role": "user", "content": content})
		return messages

	def format(
		self,
		signature: type[dspy.Signature],
		inputs: dict[str, Any] | list[dict[str, Any]],
		demos: list[dict[str, Any]] | list[list[dict[str, Any]]],
		response_length: ResponseLength | None = None,
	) -> list[list[dict[str, Any]]]:
		"""Format the input messages for the (local, vLLM-based) LM call.

		This method converts the DSPy structured input along with few-shot examples into multiturn
		messages as expected by the LM. Supports both individual and batch processing.

		Messages will have the following structure:
		```
		[
		    {"role": "system", "content": system_message},
		    # Begin few-shot examples
		    {"role": "user", "content": few_shot_example_1_input},
		    {"role": "assistant", "content": few_shot_example_1_output},
		    {"role": "user", "content": few_shot_example_2_input},
		    {"role": "assistant", "content": few_shot_example_2_output},
		    ...
		    # End few-shot examples
		    {"role": "user", "content": current_input},
		]
		```

		The system message should contain a description of the task (i.e., description of input and
		output fields), how to derive a solution for the task, rules to follow, and a template
		for how to structure the response.

		Args:
		    signature: The DSPy signature defining the task's inputs, outputs, and instructions.
		    inputs: The input arguments to the DSPy module.
			demos: A list of few-shot examples.
		    response_length: The response length constraints
		Returns:
		    A list of multiturn message lists, one for each input.
		"""
		# Normalize inputs to list format
		if isinstance(inputs, dict):
			# If inputs is a single dictionary, convert it to a list with one element
			inputs = [inputs]
		elif not isinstance(inputs, list):
			raise TypeError(
				f"Expected inputs to be a list or a dict, got {type(inputs)}"
			)

		# Handle demos - check if it's batch demos or single demos for all
		# TODO[P3]: Add support for custom demos for different inputs/trajectories for a single input
		# NOTE: We assume that we get one list of demos for all inputs of the batch
		batch_demos: list[list[dict[str, Any]]]

		if demos and len(demos) > 0 and isinstance(demos[0], list):
			# Batch demos mode - each input gets its own demo list
			batch_demos = demos
			if len(batch_demos) != len(inputs):
				raise ValueError(
					f"When demos is a list of lists, it must have the same length as inputs. "
					f"Got {len(batch_demos)} demo lists but {len(inputs)} inputs."
				)
		else:
			# Single demos mode - same demos applied to all inputs
			single_demos: list[dict[str, Any]] = demos
			batch_demos = [single_demos] * len(inputs)

		messages: list[list[dict[str, Any]]] = []
		inputs_copy = copy.deepcopy(inputs)

		for i, input_dict in enumerate(inputs_copy):
			single_messages = self.format_single(
				signature=signature,
				demos=batch_demos[i],
				inputs=input_dict,
				response_length=response_length,
			)
			messages.append(single_messages)
		return messages

	def format_demos(
		self,
		signature: type[dspy.Signature],
		demos: list[dict[str, Any]],
	) -> list[dict[str, str]]:
		"""Format the in-context examples into a list of messages.

		Transforms each demo into a pair of user and assistant messages, where the
		user message contains the inputs and the assistant message contains the
		outputs.

		Args:
		    signature: The DSPy signature for which to format the few-shot examples.
		    demos: A list of examples. Each example is a dictionary containing:
		        - INPUT: A dictionary mapping input field names to their values
		        - OUTPUT: A dictionary mapping output field names to their values

		Returns:
		    A list of messages alternating between user and assistant roles.
		"""
		# Validate that all demos are complete
		for i, demo in enumerate(demos):
			# Check that demo has the required keys
			assert set(demo.keys()).issuperset({ReasoningState.INPUT, ReasoningState.OUTPUT}), (
				f"Demo {i} is missing one or more required keys ('input', 'output')"
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
						main_request=False,
					),
				}
			)
			# Create assistant message with outputs
			messages.append(
				{
					"role": "assistant",
					"content": self.format_demo_assistant_message(
						signature=signature, demo=demo
					),
				}
			)
		return messages

	def format_demo_assistant_message(
		self,
		signature: type[dspy.Signature],
		demo: dict[str, Any],
	) -> str:
		"""Format an assistant message for an in-context example.

		Creates a properly formatted assistant message with output fields using ## headers.

		Args:
		    signature: The DSPy signature defining the fields.
		    demo: A dictionary containing input and output keys.

		Returns:
		    A formatted string representing the assistant's response with proper headers.
		"""
		message_parts = []

		# Add each output field
		for field_name, field_info in signature.output_fields.items():
			message_parts.append(f"## {field_name}")
			# Format the output value according to its field info
			formatted_value = format_field_value(field_info, demo[ReasoningState.OUTPUT][field_name])
			message_parts.append(formatted_value)

		# Join all parts with double newlines for readability
		return "\n\n".join(message_parts)

	def format_user_message_content(
		self,
		signature: type[dspy.Signature],
		inputs: dict[str, Any],
		prefix: str = "",
		suffix: str = "",
		main_request: bool = False,
	) -> str:
		"""
		Format the content of the user message.

		The user prompt instructs the language model (assistant) to solve a single instance of the task
		defined in the system message.

		Args:
		    signature (Type[Signature]): The DSPy signature defining the expected input/output fields.
		    inputs (Dict[str, Any]): The input arguments to the DSPy module.
		    prefix (str): Optional prefix to prepend to the user message.
		    suffix (str): Optional suffix to append to the user message.
		    main_request (bool): Whether this is the main request for the task. If True, it will include
		        output requirements in the message.

		Returns:
		    str: The formatted user message content.
		"""
		messages: list[str] = []
		# Avoid repeating task instructions in both system and user messages.
		# If a prefix is explicitly provided, include it; otherwise start directly with fields.
		if prefix:
			messages.append(prefix)
		for k, v in signature.input_fields.items():
			if k in inputs:
				value = inputs.get(k)
				formatted_field_value = format_field_value(field_info=v, value=value)
				messages.append(f"## {k}\n{formatted_field_value}")
		if main_request:
			output_requirements = self.user_message_output_requirements(signature)
			if output_requirements is not None:
				messages.append(output_requirements)
		if suffix:
			messages.append(suffix)
		return "\n\n".join(messages).strip()

	def user_message_output_requirements(
		self, signature: type[dspy.Signature]
	) -> str | None:
		"""Returns a simplified format reminder for the language model.

		In chat-based interactions, language models may lose track of the required output format
		as the conversation context grows longer. This method generates a concise reminder of
		the expected output structure that can be included in user messages.

		Args:
		    signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

		Returns:
		    str: A simplified description of the required output format.

		Note:
		    This is a more lightweight version of `format_field_structure` specifically designed
		    for inline reminders within chat messages.
		"""

		if not signature.output_fields:
			return None

		message = "Respond with the corresponding output fields, starting with the field "
		field_parts: list[str] = []
		for field_name, field_info in signature.output_fields.items():
			part = f"`## {field_name}`"
			if field_info.annotation is not str:
				part += (
					" (must be formatted as a valid Python "
					+ get_annotation_name(field_info.annotation)
					+ ")"
				)
			field_parts.append(part)
		message += ", then ".join(field_parts)
		return message

	def get_sampling_params(self, kwargs: dict[str, Any]) -> SamplingParams:
		"""
		Extract sampling parameters from kwargs and return a SamplingParams object.

		Args:
		    kwargs: A dictionary of keyword arguments that may contain sampling parameters.

		Returns:
		    A tuple containing:
		        - SamplingParams object initialized with the provided or default parameters.
		        - Remaining kwargs that are not sampling parameters.
		"""
		sampling_fields = set(SamplingParams.__annotations__)
		sampling_params = {}  # The parameters used to initialize vllm.SamplingParams
		remaining_kwargs = kwargs.copy()
		for parameter_name in list(remaining_kwargs.keys()):
			if parameter_name in sampling_fields:
				sampling_params[parameter_name] = remaining_kwargs.pop(parameter_name)

		# Default include_stop_str_in_output to True unless explicitly set
		if "include_stop_str_in_output" not in sampling_params:
			sampling_params["include_stop_str_in_output"] = True

		return SamplingParams(**sampling_params)

	def _prepare_sampling_params(
		self, lm_kwargs: dict[str, Any] | list[dict[str, Any]]
	) -> SamplingParams | list[SamplingParams]:
		"""
		Prepare sampling parameters for either single or batch processing.

		Args:
		    lm_kwargs: Either a single dict or list of dicts containing LM parameters.

		Returns:
		    Either a single SamplingParams object or a list of SamplingParams objects.
		"""

		if isinstance(lm_kwargs, list):
			# Batch mode - create SamplingParams for each input
			sampling_params_list = []
			for kwargs in lm_kwargs:
				sampling_params_list.append(self.get_sampling_params(kwargs))
			return sampling_params_list
		else:
			# Single mode - create one SamplingParams for all inputs
			return self.get_sampling_params(lm_kwargs)

	def parse(
		self,
		signature: type[dspy.Signature],
		completion: str,
		sampling_params: SamplingParams,
	) -> dict[str, Any]:
		"""
		Parses the provided completion and extracts the relevant outputs.

		Args:
		    signature: The DSPy signature for which to parse the completion.
		    completion: The completion to parse.
		    sampling_params: Optional sampling parameters used for the request. If
		    	`sampling_params.structured_outputs` is set, JSON parsing is used.
		Returns:
		    A dictionary containing the parsed output fields.
		"""
		if sampling_params.structured_outputs is not None:
			return JSONAdapter().parse(signature, completion)
		return self._parse_chat_completion(signature=signature, completion=completion)
