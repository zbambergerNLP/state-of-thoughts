"""Generative language model wrapper for vLLM.

This module provides the `GenerativeLocalVLLM` class, a DSPy-compatible wrapper for
vLLM-based local language models specifically for text generation tasks. It returns
strictly `ChatCompletionResponse` objects and supports both single and batch processing.
"""

# Standard library imports
import datetime
import fnmatch
import gc
import logging
import time
import uuid
from typing import Any, Literal

# Third-party imports
import dspy
import torch
from pydantic import BaseModel, Field
from vllm import LLM, RequestOutput, SamplingParams, distributed
from vllm.transformers_utils.tokenizer import AnyTokenizer

# Local imports
from constants import VERBOSITY_TO_LOGGING_LEVEL, Verbosity
from lm.lm_constants import (
    CHAT_TEMPLATE_FORMATS,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_SEED,
    ENABLE_THINKING,
)
from lm.lm_utils import DictAccessMixin, Message, Usage

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ModelExecutionError(RuntimeError):
    """Raised when model execution fails (e.g., a vLLM runtime error)."""


def _format_model_execution_error_message(err: Exception) -> str:
    """Format an exception message suitable for surfacing as a model execution failure."""
    message = str(err)
    return message if message else repr(err)


def _normalize_finish_reason(finish_reason: str | None) -> str:
    """Normalize vLLM finish reasons into the (OpenAI-style) set we support.

    Args:
        finish_reason: The finish reason string from vLLM.

    Returns:
        The normalized finish reason string (if None is provided, return "stop" as default).
    """
    if finish_reason is None:
        return "stop"
    if finish_reason in {"stop", "length", "abort", "error", "content_filter", "tool_calls"}:
        return finish_reason
    raise ModelExecutionError(f"Unknown finish reason: {finish_reason}")


class Choice(BaseModel, DictAccessMixin):
    """Individual choice in a chat completion response."""

    finish_reason: Literal[
        "stop",              # Reached stop phrase
        "length",            # Exceeded max tokens
        "abort",             # Aborted by client / engine
        "error",             # Request-level internal error
        "content_filter",    # Content filtered (e.g., bad words/guardrail filter)
        "tool_calls"         # LM chose to invoke a tool (e.g., calculator/search)
    ] = Field(
        ...,
        description="""
Reason generation stopped: stop, length, abort, error, content_filter, or tool_calls.
For additional context see:
1. https://platform.openai.com/docs/api-reference/chat/object#chat-object-choices-finish_reason
2. https://docs.vllm.ai/en/stable/api/vllm/v1/engine/#vllm.v1.engine.FinishReason
""".strip(),
    )
    index: int = Field(..., description="Index of the choice")
    message: Message = Field(..., description="Message object containing role and content")
    text: str | None = Field(None, description="Generated text")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool calls emitted by the model (OpenAI-style tool calling schema).",
    )
    logprobs: Any | None = Field(None, description="Optional logprobs payload (model-dependent).")

class ChatCompletionResponse(BaseModel, DictAccessMixin):
    """Chat completion response matching LiteLLM/OpenAI format. Supports dict and object access."""

    choices: list[Choice] = Field(..., description="List of completion choices")
    created: int = Field(..., description="Unix timestamp of when the response was created")
    model: str = Field(..., description="Model name/identifier")
    usage: Usage = Field(..., description="Token usage statistics")
    id: str | None = Field(None, description="Unique identifier for the completion")

    def dict(self, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(**kwargs)


class GenerativeLocalVLLM(dspy.BaseLM):
    """DSPy-compatible wrapper for vLLM-based generative language models.

    This class is specialized for text generation tasks and returns strictly
    `ChatCompletionResponse` objects. It supports both single and batch processing
    of prompts or conversation threads.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str | None = None,
        seed: int = DEFAULT_SEED,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        hf_token: bool = False,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
        verbosity: Verbosity = Verbosity.WARNING,
        **kwargs: Any,
    ) -> None:
        """Create a new generative language model instance.

        Args:
            model: The model to use. This should be a string of the form `"llm_provider/llm_name"`
                offered by HuggingFace. For example, `"Qwen/Qwen3-30B-A3B-Instruct-2507"`.
            tokenizer: The name of the tokenizer to use. If None, inferred from the model.
            tokenizer_mode: The tokenizer mode. "auto" uses fast tokenizer if available.
            skip_tokenizer_init: If true, skip initialization of tokenizer and detokenizer.
            tensor_parallel_size: The number of GPUs to use for tensor parallelism.
            dtype: The data type for model weights and activations.
            quantization: The method used to quantize model weights (e.g., "awq", "gptq", "fp8").
            seed: Random seed for reproducibility.
            gpu_memory_utilization: Ratio of GPU memory to reserve (0-1).
            swap_space: CPU memory (GiB) per GPU for swap space.
            cpu_offload_gb: CPU memory (GiB) for offloading model weights.
            max_model_len: Maximum sequence length covered by CUDA graphs.
            hf_token: HuggingFace token for remote files.
            enable_prefix_caching: Enable automatic prefix caching.
            enforce_eager: Enable eager execution mode in vLLM.
            verbosity: Verbosity level for logging. One of Verbosity.DEBUG, Verbosity.INFO,
                Verbosity.WARNING, or Verbosity.ERROR. Defaults to Verbosity.WARNING.
            **kwargs: Additional arguments passed to the vLLM constructor.
        """
        self._verbosity = verbosity
        logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])
        logger.info(
            f"model: {model}\n"
            f"tokenizer: {tokenizer}\n"
            f"tokenizer_mode: {tokenizer_mode}\n"
            f"skip_tokenizer_init: {skip_tokenizer_init}\n"
            f"tensor_parallel_size: {tensor_parallel_size}\n"
            f"dtype: {dtype}\n"
            f"quantization: {quantization}\n"
            f"seed: {seed}\n"
            f"gpu_memory_utilization: {gpu_memory_utilization}\n"
            f"max_model_len: {max_model_len}\n"
            f"enable_prefix_caching: {enable_prefix_caching}\n"
            f"enforce_eager: {enforce_eager}\n"
            f"kwargs: {kwargs}\n"
        )
        task = kwargs.pop("task", "generate")
        if isinstance(task, str):
            try:
                task = task.lower()
            except ValueError:
                logger.warning(f"Unknown task '{task}'; defaulting to 'generate'")
                task = "generate"
        self.model = LLM(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            max_model_len=max_model_len,
            hf_token=hf_token,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            **kwargs,
        )
        self.tokenizer: AnyTokenizer = self.model.get_tokenizer()
        self.history: list[dict[str, Any]] = []
        self.model_path = model
        self._model_name = model.split("/")[-1] if "/" in model else model
        self.model_type = "chat"  # DSPy BaseLM compatibility
        self.task = task

    @property
    def model_name(self) -> str:
        """Get the model name (last component of model path)."""
        return self._model_name

    @property
    def verbosity(self) -> Verbosity:
        """Get the current verbosity level."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: Verbosity) -> None:
        """Set the verbosity level and update logger.

        Args:
            verbosity: The new verbosity level to set.
        """
        self._verbosity = verbosity
        logger.setLevel(VERBOSITY_TO_LOGGING_LEVEL[verbosity])

    def _should_disable_thinking_mode(self) -> bool:
        """Check if thinking mode should be automatically disabled for the current model.

        Returns:
            bool: True if thinking mode should be disabled, False otherwise.
        """
        if not self.model_name:
            return False
        patterns = ["Qwen3-*", "*reasoning*", "nemotron"]
        return any(fnmatch.fnmatch(str(self.model_name).lower(), p.lower()) for p in patterns)

    def _validate_single_inputs(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Validate inputs for single prompt or message thread.

        Args:
            prompt: A single prompt string (e.g., "What color is the sky?")
            messages: A single conversation thread (list of message dicts, where each dict contains
                a role and content key). For example:
                [
                    {"role": "user", "content": "What color is the sky?"},
                    {"role": "assistant", "content": "The sky is blue."},
                ]

        Raises:
            ValueError: If both prompt and messages are provided, or if neither is provided.
            AssertionError: If messages is not a valid conversation thread.
        """
        if prompt is not None and messages is not None:
            raise ValueError(
                "Cannot provide both prompt and messages. Provide either a single "
                "prompt string or a single conversation thread (list of messages)."
            )
        if prompt is None and messages is None:
            raise ValueError(
                "Must provide either prompt or messages for generation tasks."
            )
        if messages is not None:
            assert isinstance(messages, list), "messages must be a list"
            assert len(messages) > 0, "messages cannot be empty"
            assert isinstance(messages[0], dict), (
                "messages must be a single conversation thread (list of dicts), "
                "not a batch. Use batch() for batch processing."
            )
            assert all("role" in msg and "content" in msg for msg in messages), (
                "Each message dict must contain role and content keys."
            )

    def _validate_batch_inputs(
        self,
        prompts: list[str] | None = None,
        messages: list[list[dict[str, str]]] | None = None,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> None:
        """Validate inputs for batch processing.

        Args:
            prompts: A batch of prompt strings.
            messages: A batch of conversation threads.
            sampling_params: Sampling parameters (single or list).

        Raises:
            ValueError: If both prompts and messages are provided, or if neither is provided.
            AssertionError: If inputs are invalid or sampling_params length doesn't match.
        """
        if prompts is not None and messages is not None:
            raise ValueError(
                "Cannot provide both prompts and messages. Provide either a batch "
                "of prompt strings or a batch of conversation threads."
            )
        if prompts is None and messages is None:
            raise ValueError(
                "Must provide either prompts or messages for batch processing."
            )

        # Determine batch size and validate batch structure
        if prompts is not None:
            assert isinstance(prompts, list), "prompts must be a list"
            batch_size = len(prompts)
            assert batch_size > 0, "prompts cannot be empty"
            # Validate each prompt using _validate_single_inputs
            for prompt in prompts:
                self._validate_single_inputs(prompt=prompt, messages=None)
        else:
            assert isinstance(messages, list), "messages must be a list"
            assert len(messages) > 0, "messages cannot be empty"
            assert isinstance(messages[0], list), (
                "messages must be a batch (list of lists of dicts), not a single thread."
            )
            batch_size = len(messages)
            # Validate each message thread using _validate_single_inputs
            for message_thread in messages:
                self._validate_single_inputs(prompt=None, messages=message_thread)

        # Validate sampling_params length matches batch size
        if sampling_params is not None:
            if isinstance(sampling_params, SamplingParams):
                pass  # Single SamplingParams is fine - will be broadcast
            else: # sampling_params is a list of SamplingParams
                assert len(sampling_params) == batch_size, (
                    f"Expected the number of sampling parameters ({len(sampling_params)}) "
                    f"to match the batch size ({batch_size})."
                )


    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""
        return len(self.tokenizer(text))

    def _build_usage(
        self,
        messages: list[list[dict[str, str]]],
        completions: list[list[str]],
    ) -> list[Usage]:
        """Construct token usage statistics.

        Args:
            messages: The list of message threads.
            completions: The list of completions generated.

        Returns:
            A list of Usage objects, one per input.
        """
        results = []
        for example_index, example in enumerate(messages):
            prompt_tokens = sum(self._count_tokens(m.get("content", "")) for m in example)
            completion_tokens_list = [
                self._count_tokens(completion) for completion in completions[example_index]
            ]
            total_completion_tokens = sum(completion_tokens_list)
            results.append(Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=prompt_tokens + total_completion_tokens,
            ))
        return results

    def _build_choices(self, completions: list[list[str]]) -> list[list[Choice]]:
        """Construct a list of list of Choice objects from completions.

        Args:
            completions: List of lists of strings.

        Returns:
            List of lists of Choice objects.
        """
        return [[
            Choice(
                finish_reason="stop",
                index=i,
                message=Message(role="assistant", content=text),
                text=text,
            )
            for i, text in enumerate(example)
        ] for example in completions]

    def _build_choices_from_vllm_outputs(
        self, outputs: list[RequestOutput]
    ) -> tuple[list[list[Choice]], list[list[str]]]:
        """
        Build Choice objects directly from vLLM RequestOutput objects.

        Returns both:
        - choices: list[list[Choice]] (one list per input)
        - texts: list[list[str]] (for usage accounting)

        Args:
            outputs: List of vLLM RequestOutput objects (one per input, and can contain multiple
            completions per input).

        Returns:
            Tuple of:
            - choices: List of lists of Choice objects (one list of Choice objects per input)
            - texts: List of lists of strings (one list of completion strings per input)
        """
        all_choices: list[list[Choice]] = []
        all_texts: list[list[str]] = []

        for output in outputs:
            choices_for_input: list[Choice] = []
            texts_for_input: list[str] = []
            for completion in output.outputs:
                finish_reason_str = _normalize_finish_reason(completion.finish_reason)
                texts_for_input.append(completion.text)
                choices_for_input.append(
                    Choice(
                        finish_reason=finish_reason_str,
                        index=completion.index,
                        message=Message(role="assistant", content=completion.text),
                        text=completion.text,
                        # NOTE: It is the caller's responsibility to parse json-encoded tool calls.
                        tool_calls=None,
                        logprobs=completion.logprobs,
                    )
                )
            all_choices.append(choices_for_input)
            all_texts.append(texts_for_input)

        return all_choices, all_texts

    def _build_chat_completion_responses(
        self,
        choices: list[list[Choice]],
        usage: list[Usage],
    ) -> list[ChatCompletionResponse]:
        """Build ChatCompletionResponse objects from choices and usage statistics.

        Args:
            choices: List of lists of Choice objects.
            usage: List of Usage objects.

        Returns:
            List of ChatCompletionResponse objects, one per input.
        """
        result = []
        for example_choices, usage_obj in zip(choices, usage, strict=True):
            result.append(ChatCompletionResponse(
                id=f"chat_completion_{uuid.uuid4()}",
                created=int(time.time()),
                model=self.model_path,
                choices=example_choices,
                usage=usage_obj,
            ))
        return result

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = False,
        chat_template: str | None = None,
        chat_template_content_format: CHAT_TEMPLATE_FORMATS = "auto",
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Generate a response for a single prompt or conversation thread.

        Args:
            prompt: A single prompt string. Either prompt or messages must be provided.
            messages: A single conversation thread (list of message dicts).
            sampling_params: Sampling parameters for text generation.
            use_tqdm: Whether to display a progress bar.
            chat_template: Template for structuring chat.
            chat_template_content_format: Format to render message content.
            continue_final_message: If True, continues the final message.
            tools: List of tools available to the model.
            chat_template_kwargs: Additional kwargs for apply_chat_template.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatCompletionResponse: A single chat completion response.

        Raises:
            ValueError: If both prompt and messages are provided, or if neither is provided.
        """
        # Validate inputs
        self._validate_single_inputs(prompt=prompt, messages=messages)

        # Convert prompt to messages format if needed
        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        # Wrap single thread in list for vLLM API (expects batch format)
        messages_batch = [messages]  # type: ignore
        batch_size = 1

        # Automatically configure chat_template_kwargs for models that need thinking disabled
        if chat_template_kwargs is None:
            chat_template_kwargs = {}

        # Auto-disable thinking mode for Qwen3, R1, and other reasoning models
        if self._should_disable_thinking_mode() and ENABLE_THINKING not in chat_template_kwargs:
            chat_template_kwargs[ENABLE_THINKING] = False

        # Generate responses using the vLLM model
        try:
            outputs: list[RequestOutput] = self.model.chat(
                messages=messages_batch,  # type: ignore
                sampling_params=sampling_params,
                use_tqdm=False,
                chat_template=chat_template,
                chat_template_content_format=chat_template_content_format,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
        except Exception as e:
            raise ModelExecutionError(_format_model_execution_error_message(e)) from e
        logger.debug("\n\nResponses:\n")
        for input_index, output_for_input in enumerate(outputs, start=1):
            # vLLM returns one RequestOutput per input. Completions are usually in
            # `RequestOutput.outputs`.
            for completion_index, completion in enumerate(output_for_input.outputs, start=1):
                logger.debug(f"\n\t\tResponse {input_index}, completion {completion_index}:\n{completion}") # noqa: E501

        assert len(outputs) == batch_size, (
            f"Expected the number of outputs ({len(outputs)}) to be equal to "
            f"the number of inputs ({batch_size})."
        )

        choices, texts = self._build_choices_from_vllm_outputs(outputs)
        usage = self._build_usage(messages=messages_batch, completions=texts)  # type: ignore

        # Track usage if enabled
        if dspy.settings.usage_tracker:
            dspy.settings.usage_tracker.add_usage(self.model, usage)

        responses = self._build_chat_completion_responses(choices, usage)

        # Update history
        self._update_history(
            prompt=prompt,
            messages=messages,  # type: ignore
            response=responses[0],
            kwargs=kwargs,
        )

        return responses[0]

    def batch(
        self,
        prompts: list[str] | None = None,
        messages: list[list[dict[str, str]]] | None = None,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = False,
        chat_template: str | None = None,
        chat_template_content_format: CHAT_TEMPLATE_FORMATS = "auto",
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ChatCompletionResponse]:
        """Process a batch of prompts or conversation threads.

        Args:
            prompts: A batch of prompt strings.
            messages: A batch of conversation threads.
            sampling_params: Sampling parameters (single or list).
            use_tqdm: Whether to display a progress bar.
            chat_template: Template for structuring chat.
            chat_template_content_format: Format to render message content.
            continue_final_message: If True, continues the final message.
            tools: List of tools available to the model.
            chat_template_kwargs: Additional kwargs for apply_chat_template.
            **kwargs: Additional keyword arguments.

        Returns:
            list[ChatCompletionResponse]: A list of chat completion responses, one per input.

        Raises:
            ValueError: If both prompts and messages are provided, or if neither is provided.
        """
        # Validate inputs
        self._validate_batch_inputs(
            prompts=prompts,
            messages=messages,
            sampling_params=sampling_params,
        )

        # Disable vLLM tqdm unless explicitly in DEBUG mode.
        if self._verbosity == Verbosity.DEBUG:
            use_tqdm = True

        # Convert prompts to messages format if needed
        if prompts is not None:
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # At this point, messages is guaranteed to be a batch (list of lists)
        assert isinstance(messages, list), "messages must be a list"
        assert len(messages) > 0, "messages cannot be empty"
        assert isinstance(messages[0], list), (
            "messages must be a batch (list of lists of dicts), not a single thread."
        )

        # Handle sampling_params - convert single to list if needed
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(messages)
        assert len(sampling_params) == len(messages), (
            f"Expected {len(messages)} sampling_params, got {len(sampling_params)}"
        )

        # Automatically configure chat_template_kwargs for models that need thinking disabled
        if chat_template_kwargs is None:
            chat_template_kwargs = {}

        # Auto-disable thinking mode for Qwen3-30B-A3B and similar models
        if self._should_disable_thinking_mode() and ENABLE_THINKING not in chat_template_kwargs:
            chat_template_kwargs[ENABLE_THINKING] = False

        # Generate responses using the vLLM model (batch processing)
        try:
            outputs: list[RequestOutput] = self.model.chat(
                messages=messages,  # type: ignore
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
                chat_template=chat_template,
                chat_template_content_format=chat_template_content_format,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
        except Exception as e:
            raise ModelExecutionError(_format_model_execution_error_message(e)) from e
        # Avoid logging full RequestOutput objects (can include token ids / prompt token ids).
        logger.debug("vLLM returned %d request output(s)", len(outputs))
        assert len(outputs) == len(messages), (
            f"Expected the number of outputs ({len(outputs)}) to be equal to "
            f"the number of inputs ({len(messages)})."
        )

        choices, texts = self._build_choices_from_vllm_outputs(outputs)
        usage = self._build_usage(messages=messages, completions=texts)  # type: ignore

        # Track usage if enabled
        if dspy.settings.usage_tracker:
            dspy.settings.usage_tracker.add_usage(self.model, usage)

        responses = self._build_chat_completion_responses(choices, usage)

        # Update history for each response
        for i, response in enumerate(responses):
            self._update_history(
                prompt=prompts[i] if prompts else None,
                messages=messages[i],
                response=response,
                kwargs=kwargs,
            )

        return responses

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | list[list[dict[str, str]]] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse | list[ChatCompletionResponse]:
        """Dispatch to forward or batch based on input structure.

        Args:
            prompt: Optional prompt string (for single calls).
            messages: Either a single conversation thread (list of dicts) or
                a batch of conversation threads (list of lists of dicts).
            **kwargs: Additional keyword arguments passed to forward/batch.

        Returns:
            For single calls: ChatCompletionResponse.
            For batch calls: list[ChatCompletionResponse].
        """
        if messages is None:
            # No messages provided, call forward with prompt only
            return self.forward(prompt=prompt, **kwargs)

        # Check if messages is a batch (list of lists) or single thread (list of dicts)
        if len(messages) > 0 and isinstance(messages[0], list):
            # Batch mode: messages is list[list[dict[str, str]]]
            return self.batch(messages=messages, **kwargs)  # type: ignore
        else:
            # Single thread mode: messages is list[dict[str, str]]
            if prompt is not None:
                return self.forward(prompt=prompt, **kwargs)
            else:
                return self.forward(messages=messages, **kwargs)  # type: ignore

    def _update_history(
        self,
        prompt: str | None,
        messages: list[dict[str, str]],
        response: ChatCompletionResponse,
        kwargs: dict[str, Any],
    ) -> None:
        """Update the history with a new entry.

        Args:
            prompt: The prompt string (if used).
            messages: The conversation thread.
            response: The chat completion response.
            kwargs: Additional keyword arguments.
        """
        if dspy.settings.disable_history:
            return

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": response,
            "outputs": [choice.text for choice in response.choices],
            "usage": dict(response.usage.model_dump()),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": response.id or str(uuid.uuid4()),
            "model": self.model_name,
            "response_model": response.model,
            "model_type": self.model_type,
        }
        self.history.append(entry)

    def sleep(self, sleep_level: int) -> None:
        """Put the engine to sleep.

        Args:
            sleep_level: The sleep level (1 or 2).
        """
        self.model.sleep(level=sleep_level)

    def wake_up(self, tags: list[str] | None) -> None:
        """Wake up the engine from sleep mode.

        Args:
            tags: Optional list of tags to reallocate memory for.
        """
        self.model.wake_up(tags=tags)

    def kill(self) -> None:
        """Free all model resources and memory."""
        try:
            distributed.parallel_state.destroy_model_parallel()
        except Exception as e:
            logger.warning(f"Failed to destroy model parallel:\t{e}")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
