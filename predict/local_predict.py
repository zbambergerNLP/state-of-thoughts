"""
LocalPredict V1

An adaptation of dspy.Predict meant to support local vLLM language models.
Notably, this variation supports batches of inputs, whereas the original dspy.Predict supports asynchronous calls with individual inputs.
This module serves as the base class for two components of Tree-of-Thoughts:
* Controller: Determines what action to take in the next reasoning step
* Evaluator: Determines the quality of reasoning trajectories
"""

# Standard library imports
import logging
import random
from collections import namedtuple
from typing import Any

# Third-party imports
import dspy
from dspy import BaseLM, Prediction, Signature, settings
from dspy.utils import BaseCallback
from vllm import SamplingParams

# Local imports
from adapter.constraints import ResponseLength
from adapter.vllm_adapter import LocalVLLMAdapter
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.generative_local_lm import ModelExecutionError
from misc_utils import ExecutionError, _is_instance_of_type, _is_optional_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ForwardInputs = namedtuple(
	"ForwardInputs",
	[
		"lm",
		"config",
		"signature",
		"demos",
		"processed_inputs",
	],
)


class LocalPredict(dspy.Predict):
	"""
	A simplified prediction class that works with LocalVLLMAdapter.

	This class is designed to be used with the Controller and Evaluator classes.
	It allows for generating predictions based on a given signature and input fields.
	"""

	def __init__(
		self,
		signature: type[Signature],
		callbacks: list[BaseCallback] | None = None,
		response_length: ResponseLength | None = None,
		verbose: Verbosity = Verbosity.WARNING,
		**config,
	):
		"""
		Initializes a DSPy module that generates simple predictions.

		Args:
		    signature: A dspy signature class that defines how to produce output fields given input fields.
		    callbacks: A list of callbacks to be used during the prediction process.
		    response_length: Configuration for response length constraints.
		    verbose: Verbosity level for logging (default is Verbosity.WARNING).
		    **config: Hyperparameters for the language model such as `n`, `temperature`, and
				`max_tokens` might be overridden at runtime.
		"""
		super().__init__(signature, callbacks, **config)
		self.reset()
		self.stage = random.randbytes(8).hex()
		self.response_length = response_length
		self.verbose = verbose
		self.config = config
		self.callbacks = callbacks or []
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbose])

	def dump_state(self) -> dict[str, Any]:
		"""
		Dump the current state of the predictor.

		Returns:
		    Dictionary containing the current state of the predictor.
		"""
		state: dict[str, Any] = super().dump_state()
		if self.response_length:
			state["response_length"] = str(self.response_length)
		state["verbose"] = self.verbose
		return state

	def _detect_and_prepare_batching(
		self, signature: type[Signature], **kwargs
	) -> list[dict[str, Any]]:
		"""
		Detect if any input fields contain lists and prepare for batching.

		Args:
		    signature: The DSPy signature defining input fields
		    **kwargs: Input arguments that may contain lists

		Returns:
		    batch_inputs: List of input dictionaries for batch processing
		"""

		input_field_names = set(signature.input_fields.keys())
		provided_fields = {k: v for k, v in kwargs.items() if k in input_field_names}

		# Track which fields are being batched and their lengths
		batch_field_lengths = {}

		for field_name, value in provided_fields.items():
			field_annotation = signature.input_fields[field_name].annotation

			# Early validation: non-optional fields cannot be None or empty list
			if not _is_optional_type(field_annotation):
				if value is None or (isinstance(value, list) and len(value) == 0):
					empty_type = "None" if value is None else "empty list"
					raise TypeError(
						f"Field '{field_name}' is not optional but received {empty_type}. "
					)

			# Check 1: Does value match expected type? -> Single input
			if _is_instance_of_type(value, field_annotation):
				# Value matches expected type - this is a single input (not batching)
				continue

			# Check 2: Is value a list of expected type? -> Batching
			elif isinstance(value, list) and len(value) > 0:
				# Check if all elements match the expected type
				if all(_is_instance_of_type(elem, field_annotation) for elem in value):
					# This is batching: list of expected type
					batch_field_lengths[field_name] = len(value)
				else:
					# List elements don't match expected type
					raise TypeError(
						f"Field '{field_name}' expects type {field_annotation} or list[{field_annotation}]. "
						f"Got list with elements of type {[type(elem).__name__ for elem in value[:3]]}..."
					)

			else:
				raise TypeError(
					f"Field '{field_name}' expects type {field_annotation} or list[{field_annotation}], "
					f"but got {type(value).__name__}"
				)

		# If no batch fields found, treat as single input
		if not batch_field_lengths:
			return [kwargs]

		# Validate all batch fields have the same length
		unique_lengths = set(batch_field_lengths.values())
		assert len(unique_lengths) == 1, (
			f"All batched fields must have the same length. "
			f"Found lengths: {batch_field_lengths}"
		)
		batch_size = unique_lengths.pop()
		assert batch_size > 0, "Cannot process empty batch"

		# Create batch inputs by combining batched and broadcasted fields
		batch_inputs = []
		for i in range(batch_size):
			batch_input = {}
			for field_name in input_field_names:
				if field_name in kwargs:
					value = kwargs[field_name]
					if field_name in batch_field_lengths:
						# Batched field: take i-th element
						batch_input[field_name] = value[i]
					else:
						# Broadcast single value to each input in the batch
						batch_input[field_name] = value

			# Add non-input fields (config, etc.) to each batch input
			for field_name, value in kwargs.items():
				if field_name not in input_field_names:
					batch_input[field_name] = value
			batch_inputs.append(batch_input)

		return batch_inputs

	def _forward_preprocess(self, **kwargs) -> ForwardInputs:
		"""
		Pre-processes arguments supplied to LocalPredict at the beginning of a completion call.
		Supports auto-detection of batching from list inputs.

		Args:
		    **kwargs: Arguments supplied to LocalPredict as part of its forward call.

		Returns:
		    A named tuple of ForwardInputs.
		"""
		# Extract the privileged keyword arguments
		signature = dspy.ensure_signature(kwargs.pop("signature", self.signature))
		demos = kwargs.pop("demos", self.demos)
		response_length = kwargs.pop("response_length", self.response_length)

		# Detection and batching - always returns list of dicts
		batch_inputs = self._detect_and_prepare_batching(signature, **kwargs)

		# Always extract config/lm from batch inputs
		# They are broadcasted to all inputs, so we must remove them from all to avoid pickling
		# errors in the adapter
		config = dict(**self.config)

		# Extract config and lm from first input to get values
		runtime_config = batch_inputs[0].pop("config", {})
		lm = batch_inputs[0].pop("lm", self.lm)

		# Remove from the rest of the batch
		# TODO[P3]: Add support for localpredict accepting multiple configs (one for each input in
		# the batch)
		for i in range(1, len(batch_inputs)):
			batch_inputs[i].pop("config", None)
			batch_inputs[i].pop("lm", None)

		if runtime_config:
			config.update(runtime_config)
		lm = lm or settings.lm
		assert isinstance(lm, BaseLM), "No LM is loaded."

		# Support all vLLM SamplingParams fields via config (and common alias keys).
		if ("n" not in config) and ("num_generations" in config):
			config["n"] = config["num_generations"]
		if ("max_tokens" not in config) and ("max_new_tokens" in config):
			config["max_tokens"] = config["max_new_tokens"]

		# Drop explicit None values only for vLLM SamplingParams fields so vLLM can apply defaults.
		# Important: we must preserve non-sampling kwargs (e.g., chat_template_kwargs/tools/chat_template)
		# so they propagate through LocalVLLMAdapter to GenerativeLocalVLLM.batch/forward.
		vllm_sampling_param_names = set(SamplingParams.__annotations__)
		for k in list(config.keys()):
			if (k in vllm_sampling_param_names) and (config[k] is None):
				del config[k]

		# Validate temperature if provided.
		temperature = config.get("temperature")
		if temperature is not None:
			temperature_f = float(temperature)
			if not (0.0 <= temperature_f <= 1.0):
				raise ValueError(f"temperature must be within [0, 1]; got {temperature_f}")
			if temperature_f == 0.0:
				self.logger.warning("temperature=0 yields greedy decoding (deterministic, no sampling).")
			config["temperature"] = temperature_f
		else:
			raise ValueError("temperature must be provided in config.")

		# Fix the demos parameter - convert to list if it's a dict
		if isinstance(demos, dict):
			demos = [demos]

		# Add response_length to each batch input if specified
		if response_length:
			for batch_input in batch_inputs:
				batch_input["response_length"] = response_length

		# Validate all batch inputs have required the fields
		for i, batch_input in enumerate(batch_inputs):
			missing = [k for k in signature.input_fields if k not in batch_input]
			assert not missing, (
				f"Batch input {i} is missing required input fields: {missing}. "
				f"Present: {[k for k in signature.input_fields if k in batch_input]}"
			)

		return ForwardInputs(
			lm=lm,
			config=config,
			signature=signature,
			demos=demos,
			processed_inputs=batch_inputs,
		)

	def _forward_postprocess(
		self, examples: list[list[dict[str, Any]]], signature: type[Signature], **kwargs
	) -> list[Prediction]:
		"""
		Produces a list of `Prediction`s, one for each input, each possibly with multiple completions.

		Args:
		    examples: A list of list of completions produced by the language model.
		        The adapter always returns this format regardless of single/batch input.
		    signature: A DSPy signature class that defines how to produce output fields given input fields.
		    batch_size: The size of the batch
		    **kwargs: Additional keyword arguments that were passed into the adapter.

		Returns:
		    A list of `Prediction`s, where each `Prediction` corresponds with multiple completions for a
		    single input. Always returns a list, even for single inputs (for consistency).
		"""

		predictions = []
		for example_completions in examples:
			if example_completions:
				# Pass raw completions directly to DSPy - let Prediction handle the format
				pred = Prediction.from_completions(
					example_completions, signature=signature
				)
				if kwargs.get("_trace", True) and settings.trace is not None:
					trace = settings.trace
					trace.append((self, {**kwargs}, pred))
				predictions.append(pred)
			else:
				# Return empty prediction if no completions for this input
				predictions.append(Prediction.from_completions({}, signature=signature))

		return predictions

	def forward(self, **kwargs) -> list[Prediction]:
		"""
		Generates predictions for the given inputs. Supports auto-batching when input fields contain lists.

		Args:
		    **kwargs: Input arguments matching the signature's input fields,
		             plus optional configuration arguments.

		Returns:
		    A list of `Prediction`s, where each `Prediction` corresponds with multiple completions for a
		    single input. Always returns a list for consistency, even for single inputs.


		Batching examples:
		- Single: forward(question="Q1", context="C1")
		- Batch: forward(question=["Q1", "Q2"], context=["C1", "C2"])
		- Batch with shared context: forward(question=["Q1", "Q2"], context="shared")
		"""
		ctx = self._forward_preprocess(**kwargs)
		use_native_function_calling = bool(ctx.config.get("use_native_function_calling", False))
		if use_native_function_calling:
			adapter = LocalVLLMAdapter(use_native_function_calling=use_native_function_calling)
		else:
			adapter = LocalVLLMAdapter()

		# Use same response length across the batch if specified
		response_length = ctx.processed_inputs[0].get("response_length")

		with settings.context(send_stream=None):
			try:
				completions = adapter(
					lm=ctx.lm,
					lm_kwargs=ctx.config,
					signature=ctx.signature,
					demos=ctx.demos,
					inputs=ctx.processed_inputs,
					response_length=response_length,
				)
			except ModelExecutionError as e:
				# Model-level failure (e.g., prompt too long, model crash)
				# Convert to error completions format: list[list[dict[str, Any]]]
				# One list per input, one dict per completion (n samples)
				n_samples = ctx.config.get("n", 1)
				completions = []
				for _ in ctx.processed_inputs:
					input_completions = []
					for _ in range(n_samples):
						input_completions.append({
							"error": ExecutionError(
								error_type="generation",
								error_message=str(e),
								raw_output=None,
							)
						})
					completions.append(input_completions)
				logger.warning(f"Model execution failed: {e}")

		# Post-process the completions to produce predictions
		predictions = self._forward_postprocess(completions, ctx.signature, **kwargs)

		# We should have one prediction per input (batch_size)
		# Each prediction may contain multiple completions internally (handled by DSPy)
		assert len(predictions) == len(ctx.processed_inputs), (
			f"Expected {len(ctx.processed_inputs)} predictions but got {len(predictions)}. "
			f"Batch size: {len(ctx.processed_inputs)}, n: {ctx.config['n']}, "
			f"Completions: {len(completions)}. "
		)

		return predictions

	def __call__(self, **kwargs) -> list[Prediction]:
		"""
		Convenience method that returns list of predictions.

		Parameters:
		    **kwargs: Input arguments that can include:

		Returns:
		    List[Prediction]: Always returns a list of predictions for consistency.
		        - For single inputs: List with one Prediction object
		        - For batch inputs: List with one Prediction per input
		        Each Prediction may contain multiple completions if n > 1.
		"""
		return self.forward(**kwargs)
